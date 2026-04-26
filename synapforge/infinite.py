"""Infinite-context 5-tier memory hierarchy.

Five memory tiers, inspired by the biological hippocampal-cortical
memory system and recent long-context LLM research:

  L1  — recent window (full hidden fidelity)
        :class:`RotaryPositionEncoding`, :class:`LocalGQAttention`
  L2  — compressed past (summary tokens at strided positions)
        :class:`HierarchicalMemoryConfig`, :class:`HierarchicalMemory`,
        :class:`DeltaCompress`
  L3  — slow-tau sink (drifts on ~1000-step timescale)
        :class:`AdaptiveSlowTau`
  SSM — diagonal parallel scan that complements L1-L3
        :class:`SSMDiagScan`
  L4  — external vector store (in-RAM, FAISS when available)
        :class:`ExternalVectorMemory`
  L5  — disk archive (numpy memmap, never fully loaded)
        :class:`DiskMemmapArchive`

Orchestrators:
  - :class:`InfiniteReaderConfig`, :class:`InfiniteContextReader`
  - :class:`ChunkedStateCarry`        — TBPTT-style chunk streamer
  - :class:`LongContextMonitor`       — per-position NLL monitor
  - :class:`StreamingInfiniteEvaluator` — resume-able 1M+ token evaluator

References: arxiv 2104.09864 (RoPE), 2406.07522 (Samba/SWA), 2405.21060
(Mamba-2), 2404.07143 (Infini-attention), 2402.13753 (LongRoPE),
2407.01178 (Memory3).

Self-contained — depends only on torch / numpy.  Optional FAISS/hnswlib
acceleration; falls back to torch matmul when unavailable.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Module

# ----------------------------------------------------------------------------
# L1: recent window — RoPE + local sliding-window GQA
# ----------------------------------------------------------------------------


class RotaryPositionEncoding(Module):
    """Rotary position encoding (RoPE, Su et al. 2021 arxiv:2104.09864)."""

    def __init__(
        self,
        head_dim: int,
        max_len: int = 131072,
        base: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_len = int(max_len)
        self.base = float(base)
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(self.max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_dim: int = -2,
        offset: int = 0,
    ) -> torch.Tensor:
        if x.size(-1) != self.head_dim:
            raise ValueError(f"x last dim {x.size(-1)} != head_dim {self.head_dim}")
        T = x.size(seq_dim)
        if offset + T > self.max_len:
            raise ValueError(
                f"seq len {offset + T} exceeds RoPE max_len {self.max_len}"
            )
        cos = self.cos[offset : offset + T]
        sin = self.sin[offset : offset + T]
        shape = [1] * x.dim()
        sd = seq_dim if seq_dim >= 0 else x.dim() + seq_dim
        shape[sd] = T
        shape[-1] = self.head_dim // 2
        cos = cos.reshape(shape).to(x.dtype)
        sin = sin.reshape(shape).to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return torch.cat([rx1, rx2], dim=-1)


class LocalGQAttention(Module):
    """Sliding-window grouped-query attention (Samba-style, arxiv:2406.07522)."""

    def __init__(
        self,
        hidden: int,
        q_heads: int = 4,
        kv_heads: int = 1,
        head_dim: int = 64,
        window: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"q_heads ({q_heads}) must be a multiple of kv_heads ({kv_heads})"
            )
        self.hidden = int(hidden)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.window = int(window)
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(hidden, q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(q_heads * head_dim, hidden, bias=False)
        self.rope = RotaryPositionEncoding(head_dim)
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim)
        q = self.rope(q, seq_dim=1, offset=offset)
        k = self.rope(k, seq_dim=1, offset=offset)
        reps = self.q_heads // self.kv_heads
        if reps > 1:
            k = k.repeat_interleave(reps, dim=2)
            v = v.repeat_interleave(reps, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        idx = torch.arange(T, device=x.device)
        causal = idx[:, None] < idx[None, :]
        window = (idx[None, :] + self.window) < idx[:, None]
        mask = causal | window
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, T, self.q_heads * self.head_dim)
        return self.o_proj(attn)


# ----------------------------------------------------------------------------
# L2: compressed past
# ----------------------------------------------------------------------------


@dataclass
class HierarchicalMemoryConfig:
    hidden: int = 256
    l1_capacity: int = 256
    l2_capacity: int = 1024
    compress_every: int = 8
    l2b_size: int = 0
    l2b_compress_every: int = 8


class HierarchicalMemory(Module):
    """Two-level (optionally three-level) ring buffer with mean-pool compression."""

    def __init__(self, cfg: HierarchicalMemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "l1", torch.zeros(cfg.l1_capacity, cfg.hidden), persistent=False
        )
        self.register_buffer(
            "l2", torch.zeros(cfg.l2_capacity, cfg.hidden), persistent=False
        )
        self.register_buffer(
            "l1_mask", torch.zeros(cfg.l1_capacity, dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "l2_mask", torch.zeros(cfg.l2_capacity, dtype=torch.bool), persistent=False
        )
        self._l1_head = 0
        self._l2_head = 0
        self._l1_fill = 0
        self._l2b_size = int(cfg.l2b_size)
        self._l2b_compress_every = max(1, int(cfg.l2b_compress_every))
        if self._l2b_size > 0:
            self.register_buffer(
                "l2b", torch.zeros(self._l2b_size, cfg.hidden), persistent=False
            )
            self.register_buffer(
                "l2b_mask", torch.zeros(self._l2b_size, dtype=torch.bool), persistent=False
            )
            self._l2b_head = 0
            self._l2_since_last_b = 0
        self.score_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=False)

    @torch.no_grad()
    def reset(self) -> None:
        self.l1.zero_()
        self.l2.zero_()
        self.l1_mask.zero_()
        self.l2_mask.zero_()
        self._l1_head = 0
        self._l2_head = 0
        self._l1_fill = 0
        if self._l2b_size > 0:
            self.l2b.zero_()
            self.l2b_mask.zero_()
            self._l2b_head = 0
            self._l2_since_last_b = 0

    @torch.no_grad()
    def _maybe_spill_to_l2b(self) -> None:
        if self._l2b_size <= 0:
            return
        self._l2_since_last_b += 1
        if self._l2_since_last_b < self._l2b_compress_every:
            return
        k = self._l2b_compress_every
        head = self._l2_head
        start = (head - k) % self.cfg.l2_capacity
        if start < head:
            block = self.l2[start:head]
        else:
            block = torch.cat([self.l2[start:], self.l2[:head]], dim=0)
        summary = block.mean(dim=0)
        self.l2b[self._l2b_head] = summary
        self.l2b_mask[self._l2b_head] = True
        self._l2b_head = (self._l2b_head + 1) % self._l2b_size
        self._l2_since_last_b = 0

    @torch.no_grad()
    def write(self, vec: torch.Tensor) -> None:
        if vec.dim() != 1 or vec.size(0) != self.cfg.hidden:
            raise ValueError(
                f"vec must be [hidden={self.cfg.hidden}], got {tuple(vec.shape)}"
            )
        self.l1[self._l1_head] = vec.detach().to(self.l1.dtype)
        self.l1_mask[self._l1_head] = True
        self._l1_head = (self._l1_head + 1) % self.cfg.l1_capacity
        self._l1_fill = min(self._l1_fill + 1, self.cfg.l1_capacity)
        if self._l1_fill >= self.cfg.compress_every and (
            self._l1_head % self.cfg.compress_every == 0
        ):
            start = (self._l1_head - self.cfg.compress_every) % self.cfg.l1_capacity
            if start < self._l1_head:
                block = self.l1[start : self._l1_head]
            else:
                block = torch.cat([self.l1[start:], self.l1[: self._l1_head]], dim=0)
            summary = block.mean(dim=0)
            self.l2[self._l2_head] = summary
            self.l2_mask[self._l2_head] = True
            self._l2_head = (self._l2_head + 1) % self.cfg.l2_capacity
            self._maybe_spill_to_l2b()

    def read(self, query: torch.Tensor, top_k: int = 16) -> torch.Tensor:
        if query.dim() == 1:
            q = query.unsqueeze(0)
        else:
            q = query
        q_proj = self.score_proj(q)
        tiers = [self.l1, self.l2]
        mtiers = [self.l1_mask, self.l2_mask]
        if self._l2b_size > 0:
            tiers.append(self.l2b)
            mtiers.append(self.l2b_mask)
        mem = torch.cat(tiers, dim=0)
        mask = torch.cat(mtiers, dim=0)
        Bsz = q.size(0)
        if not mask.any():
            zero_row = (q_proj.sum(dim=-1, keepdim=True) * 0.0).unsqueeze(1)
            out = zero_row.expand(Bsz, top_k, self.cfg.hidden).contiguous()
            return out.squeeze(0) if query.dim() == 1 else out
        scores = q_proj @ mem.T
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        k = min(top_k, int(mask.sum().item()))
        vals, idx = scores.topk(k, dim=-1)
        soft = torch.softmax(vals, dim=-1)
        gathered = mem[idx].detach()
        gathered = gathered * soft.unsqueeze(-1)
        residual = (q_proj.unsqueeze(1) * 0.0)
        gathered = gathered + residual
        if k < top_k:
            pad = torch.zeros(Bsz, top_k - k, self.cfg.hidden, device=q.device)
            gathered = torch.cat([gathered, pad], dim=1)
        return gathered.squeeze(0) if query.dim() == 1 else gathered


class DeltaCompress(Module):
    """Insert a low-rank summary token every ``period`` positions."""

    def __init__(self, hidden: int, period: int = 1024, rank: int = 32):
        super().__init__()
        self.hidden = int(hidden)
        self.period = int(period)
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if T < self.period:
            summary = self.up(self.down(x.mean(dim=1, keepdim=True)))
            return torch.cat([x, summary], dim=1)
        pieces = []
        for i in range(0, T, self.period):
            j = min(i + self.period, T)
            chunk = x[:, i:j, :]
            pieces.append(chunk)
            summary = self.up(self.down(chunk.mean(dim=1, keepdim=True)))
            pieces.append(summary)
        return torch.cat(pieces, dim=1)


# ----------------------------------------------------------------------------
# L3: slow-tau sink
# ----------------------------------------------------------------------------


class AdaptiveSlowTau(Module):
    """Low-pass state with learnable very-long tau."""

    def __init__(
        self,
        hidden: int,
        tau_init: float = 3000.0,
        mix: float = 0.1,
        tau_min: float = 100.0,
        tau_max: float = 100000.0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.raw_tau = nn.Parameter(torch.tensor(float(math.log(tau_init))))
        self.mix = nn.Parameter(torch.tensor(float(mix)))
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.register_buffer("state", torch.zeros(hidden), persistent=False)

    @torch.no_grad()
    def reset(self) -> None:
        self.state.zero_()

    def tau(self) -> torch.Tensor:
        return self.raw_tau.exp().clamp(self.tau_min, self.tau_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            pooled = x.detach().mean(dim=0)
        else:
            pooled = x.detach()
        with torch.no_grad():
            tau = float(self.tau().item())
            leak = 1.0 - 1.0 / max(tau, 1.0)
            mix_val = float(self.mix.clamp(0.0, 1.0).item())
            self.state.mul_(leak).add_(mix_val * pooled)
        state_d = self.state.detach()
        if x.dim() == 2:
            return x + state_d.unsqueeze(0)
        return x + state_d


# ----------------------------------------------------------------------------
# SSM path — diagonal parallel scan
# ----------------------------------------------------------------------------


class SSMDiagScan(Module):
    """Diagonal SSM (h_t = a * h_{t-1} + x_t) with stable tanh-parameterised a."""

    def __init__(self, hidden: int, a_init: float = 0.9):
        super().__init__()
        if not -1.0 < a_init < 1.0:
            raise ValueError(f"a_init must be in (-1, 1), got {a_init}")
        self.hidden = int(hidden)
        self.raw_a = nn.Parameter(torch.full((hidden,), float(math.atanh(a_init))))
        self.in_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def a(self) -> torch.Tensor:
        return torch.tanh(self.raw_a)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        u = self.in_proj(x)
        a = self.a()
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype) if h0 is None else h0
        traj = []
        for t in range(T):
            h = a * h + u[:, t, :]
            traj.append(h)
        hidden_traj = torch.stack(traj, dim=1)
        out = self.out_proj(hidden_traj)
        return out, h


# ----------------------------------------------------------------------------
# L4: external vector store (RAM, optional FAISS/HNSW)
# ----------------------------------------------------------------------------


class ExternalVectorMemory:
    """In-RAM vector memory with optional FAISS / HNSW acceleration."""

    VALID_TYPES = ("flat", "ivf_pq", "hnsw")

    def __init__(
        self,
        dim: int,
        capacity: int = 1_000_000,
        use_faiss: bool = True,
        index_type: str = "flat",
        nlist: int = 1024,
        m: int = 16,
        pq_bits: int = 8,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ):
        if index_type not in self.VALID_TYPES:
            raise ValueError(
                f"index_type must be one of {self.VALID_TYPES}, got {index_type!r}"
            )
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.index_type = str(index_type)
        self.nlist = int(nlist)
        self.m = int(m)
        self.pq_bits = int(pq_bits)
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_ef_search = int(hnsw_ef_search)
        self._faiss = None
        self._hnswlib = None
        self._buf: torch.Tensor | None = None
        self._metas: list[Any] = []
        self._size = 0
        self._trained = index_type != "ivf_pq"
        if use_faiss:
            try:
                import faiss  # type: ignore
            except Exception:
                faiss = None  # type: ignore
            if faiss is not None:
                if index_type == "flat":
                    self._faiss = faiss.IndexFlatIP(dim)
                elif index_type == "ivf_pq":
                    quantizer = faiss.IndexFlatIP(dim)
                    self._faiss = faiss.IndexIVFPQ(
                        quantizer, dim, self.nlist, self.m, self.pq_bits,
                        faiss.METRIC_INNER_PRODUCT,
                    )
                elif index_type == "hnsw":
                    try:
                        import hnswlib  # type: ignore
                        h = hnswlib.Index(space="cosine", dim=dim)
                        h.init_index(
                            max_elements=self.capacity,
                            ef_construction=self.hnsw_ef_construction,
                            M=self.hnsw_m,
                        )
                        h.set_ef(self.hnsw_ef_search)
                        self._hnswlib = h
                    except Exception:
                        self._faiss = faiss.IndexHNSWFlat(
                            dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT
                        )
                        self._faiss.hnsw.efConstruction = self.hnsw_ef_construction
                        self._faiss.hnsw.efSearch = self.hnsw_ef_search
        if self._faiss is None and self._hnswlib is None:
            self._buf = torch.zeros(self.capacity, dim, dtype=torch.float32)

    @staticmethod
    def _normalize(v: torch.Tensor) -> torch.Tensor:
        return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    def __len__(self) -> int:
        return self._size

    def train(self, sample_vecs) -> None:
        if self._faiss is None or self.index_type != "ivf_pq":
            self._trained = True
            return
        if isinstance(sample_vecs, torch.Tensor):
            arr = sample_vecs.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(sample_vecs, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"sample_vecs must be [N, {self.dim}], got {arr.shape}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        arr = (arr / norms).astype(np.float32, copy=False)
        if arr.shape[0] < self.nlist:
            raise ValueError(
                f"need >= nlist={self.nlist} training vecs, got {arr.shape[0]}"
            )
        self._faiss.train(arr)
        self._trained = True

    def add(self, vec: torch.Tensor, meta: Any | None = None) -> int:
        v = vec.detach().to(torch.float32).reshape(-1)
        if v.numel() != self.dim:
            raise ValueError(f"vec dim {v.numel()} != {self.dim}")
        if self._size >= self.capacity:
            raise RuntimeError("ExternalVectorMemory is full")
        if not self._trained:
            raise RuntimeError(
                "IVF-PQ index not trained; call .train(sample_vecs) first"
            )
        nv = self._normalize(v)
        idx = self._size
        if self._faiss is not None:
            self._faiss.add(nv.cpu().numpy().reshape(1, -1).astype(np.float32))
        elif self._hnswlib is not None:
            self._hnswlib.add_items(
                nv.cpu().numpy().reshape(1, -1).astype(np.float32),
                np.array([idx], dtype=np.int64),
            )
        else:
            assert self._buf is not None
            self._buf[idx] = nv
        self._metas.append(meta)
        self._size += 1
        return idx

    def add_batch(self, vecs, metas: list | None = None) -> int:
        if isinstance(vecs, torch.Tensor):
            arr = vecs.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"vecs must be [N, {self.dim}], got {arr.shape}")
        n = int(arr.shape[0])
        if self._size + n > self.capacity:
            raise RuntimeError(
                f"would overflow: size={self._size} + n={n} > cap={self.capacity}"
            )
        if not self._trained:
            raise RuntimeError(
                "IVF-PQ index not trained; call .train(sample_vecs) first"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        arr = (arr / norms).astype(np.float32, copy=False)
        first_idx = self._size
        if self._faiss is not None:
            self._faiss.add(arr)
        elif self._hnswlib is not None:
            self._hnswlib.add_items(
                arr, np.arange(first_idx, first_idx + n, dtype=np.int64)
            )
        else:
            assert self._buf is not None
            self._buf[first_idx : first_idx + n] = torch.from_numpy(arr)
        if metas is None:
            self._metas.extend([None] * n)
        else:
            if len(metas) != n:
                raise ValueError(f"metas length {len(metas)} != n {n}")
            self._metas.extend(metas)
        self._size += n
        return first_idx

    def topk(self, query: torch.Tensor, k: int = 4) -> tuple:
        if self._size == 0:
            return torch.zeros(k), [-1] * k, [None] * k
        q = self._normalize(query.detach().to(torch.float32).reshape(-1))
        k_eff = min(k, self._size)
        if self._faiss is not None:
            scores_np, idx_np = self._faiss.search(
                q.cpu().numpy().reshape(1, -1).astype(np.float32), k_eff
            )
            scores = torch.from_numpy(scores_np[0].copy())
            idx = [int(i) for i in idx_np[0].tolist()]
        elif self._hnswlib is not None:
            labels, distances = self._hnswlib.knn_query(
                q.cpu().numpy().reshape(1, -1).astype(np.float32), k=k_eff
            )
            scores = torch.from_numpy(
                (1.0 - distances[0]).astype(np.float32).copy()
            )
            idx = [int(i) for i in labels[0].tolist()]
        else:
            assert self._buf is not None
            bank = self._buf[: self._size]
            sims = (bank @ q).cpu()
            top = sims.topk(k_eff)
            scores = top.values
            idx = [int(i) for i in top.indices.tolist()]
        metas = [self._metas[i] if 0 <= i < len(self._metas) else None for i in idx]
        if k_eff < k:
            pad = k - k_eff
            scores = torch.cat([scores, torch.full((pad,), -float("inf"))])
            idx = idx + [-1] * pad
            metas = metas + [None] * pad
        return scores, idx, metas


# ----------------------------------------------------------------------------
# L5: disk archive (numpy memmap)
# ----------------------------------------------------------------------------


class DiskMemmapArchive:
    """Append-only D-dim vector archive (numpy memmap, never fully loaded)."""

    def __init__(self, root: str | os.PathLike, dim: int, dtype: str = "float16"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.dtype = np.dtype(dtype)
        self._itemsize = int(self.dtype.itemsize) * self.dim
        self._vec_path = self.root / "vecs.bin"
        self._meta_path = self.root / "meta.jsonl"
        self._count_path = self.root / "count"
        if not self._vec_path.exists():
            self._vec_path.touch()
        if not self._meta_path.exists():
            self._meta_path.touch()
        if not self._count_path.exists():
            self._count_path.write_text("0")

    def count(self) -> int:
        try:
            return int(self._count_path.read_text().strip() or "0")
        except Exception:
            return 0

    def append(self, vec: torch.Tensor | np.ndarray, meta: Any = None) -> int:
        import json
        if isinstance(vec, torch.Tensor):
            arr = vec.detach().cpu().numpy().astype(self.dtype).reshape(-1)
        else:
            arr = np.asarray(vec, dtype=self.dtype).reshape(-1)
        if arr.size != self.dim:
            raise ValueError(f"vec has {arr.size} elements, expected {self.dim}")
        idx = self.count()
        with open(self._vec_path, "ab") as f:
            f.write(arr.tobytes())
        with open(self._meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta if meta is not None else {}) + "\n")
        self._count_path.write_text(str(idx + 1))
        return idx

    def get(self, idx: int) -> np.ndarray:
        n = self.count()
        if not 0 <= idx < n:
            raise IndexError(f"idx {idx} out of range [0,{n})")
        with open(self._vec_path, "rb") as f:
            f.seek(idx * self._itemsize)
            raw = f.read(self._itemsize)
        if len(raw) != self._itemsize:
            raise RuntimeError(
                f"short read at idx {idx}: got {len(raw)} bytes, need {self._itemsize}"
            )
        return np.frombuffer(raw, dtype=self.dtype).copy()

    def iter_batch(self, batch: int = 1024):
        n = self.count()
        if n == 0:
            return
        mm = np.memmap(
            self._vec_path, dtype=self.dtype, mode="r", shape=(n, self.dim)
        )
        for i in range(0, n, batch):
            yield mm[i : min(i + batch, n)]


# ----------------------------------------------------------------------------
# Orchestrator: InfiniteContextReader
# ----------------------------------------------------------------------------


@dataclass
class InfiniteReaderConfig:
    hidden: int = 256
    q_heads: int = 4
    kv_heads: int = 1
    head_dim: int = 64
    window: int = 256
    l1_capacity: int = 256
    l2_capacity: int = 1024
    compress_every: int = 8
    l2b_size: int = 0
    l2b_compress_every: int = 8
    slow_tau_init: float = 3000.0
    slow_tau_mix: float = 0.1
    ssm_a_init: float = 0.9
    ext_capacity: int = 100_000
    ext_top_k: int = 4
    ext_index_type: str = "flat"
    ext_nlist: int = 1024
    ext_m: int = 16
    ext_pq_bits: int = 8
    ext_hnsw_m: int = 32
    disk_root: str | None = None
    disk_dtype: str = "float16"


class InfiniteContextReader(Module):
    """Integrated 5-tier memory reader."""

    def __init__(self, cfg: InfiniteReaderConfig):
        super().__init__()
        self.cfg = cfg
        self.l1_attn = LocalGQAttention(
            hidden=cfg.hidden, q_heads=cfg.q_heads, kv_heads=cfg.kv_heads,
            head_dim=cfg.head_dim, window=cfg.window,
        )
        self.l2_mem = HierarchicalMemory(HierarchicalMemoryConfig(
            hidden=cfg.hidden, l1_capacity=cfg.l1_capacity,
            l2_capacity=cfg.l2_capacity, compress_every=cfg.compress_every,
            l2b_size=cfg.l2b_size, l2b_compress_every=cfg.l2b_compress_every,
        ))
        self.l3_slow = AdaptiveSlowTau(
            hidden=cfg.hidden, tau_init=cfg.slow_tau_init, mix=cfg.slow_tau_mix,
        )
        self.ssm = SSMDiagScan(hidden=cfg.hidden, a_init=cfg.ssm_a_init)
        self.l4_ext = ExternalVectorMemory(
            dim=cfg.hidden, capacity=cfg.ext_capacity, use_faiss=True,
            index_type=cfg.ext_index_type, nlist=cfg.ext_nlist,
            m=cfg.ext_m, pq_bits=cfg.ext_pq_bits, hnsw_m=cfg.ext_hnsw_m,
        )
        self.l5_disk: DiskMemmapArchive | None = None
        if cfg.disk_root:
            self.l5_disk = DiskMemmapArchive(cfg.disk_root, cfg.hidden, cfg.disk_dtype)
        self.mixer = nn.Linear(cfg.hidden * 3, cfg.hidden)

    @torch.no_grad()
    def write(self, hidden: torch.Tensor) -> None:
        if hidden.dim() == 2:
            hidden = hidden.mean(dim=0)
        if hidden.dim() != 1 or hidden.size(0) != self.cfg.hidden:
            raise ValueError(
                f"expected [hidden={self.cfg.hidden}], got {tuple(hidden.shape)}"
            )
        self.l2_mem.write(hidden)
        self.l3_slow(hidden.unsqueeze(0))
        try:
            self.l4_ext.add(hidden)
        except RuntimeError:
            if self.l5_disk is not None:
                self.l5_disk.append(hidden)
            return
        if self.l5_disk is not None:
            self.l5_disk.append(hidden)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            q = query.mean(dim=0)
        else:
            q = query
        l2_rows = self.l2_mem.read(q, top_k=16)
        l2_vec = l2_rows.mean(dim=0)
        l3_vec = self.l3_slow.state.detach()
        if len(self.l4_ext) > 0:
            scores, idx, _ = self.l4_ext.topk(q.detach(), k=self.cfg.ext_top_k)
            rows = []
            for i in idx:
                if i < 0:
                    continue
                if self.l4_ext._faiss is not None or self.l4_ext._hnswlib is not None:
                    continue
                if self.l4_ext._buf is not None:
                    rows.append(self.l4_ext._buf[i].to(q.device, q.dtype))
            l4_vec = (
                torch.stack(rows, dim=0).mean(dim=0)
                if rows
                else torch.zeros_like(q)
            )
        else:
            l4_vec = torch.zeros_like(q)
        combined = torch.cat([l2_vec, l3_vec, l4_vec], dim=-1)
        return self.mixer(combined)

    def apply_window_attention(
        self, chunk: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        return self.l1_attn(chunk, offset=offset)


# ----------------------------------------------------------------------------
# Streaming utilities
# ----------------------------------------------------------------------------


class ChunkedStateCarry:
    """Stream a very long sequence by carrying state across chunks (TBPTT)."""

    def __init__(
        self,
        step_fn,
        chunk: int = 4096,
        max_memory_tokens: int = 0,
        disk_archive: DiskMemmapArchive | None = None,
    ):
        self.step_fn = step_fn
        self.chunk = int(chunk)
        self.max_memory_tokens = int(max_memory_tokens)
        self.disk_archive = disk_archive
        self.stats = {"chunks": 0, "evicted": 0, "disk_spills": 0}

    def run(
        self,
        sequence: torch.Tensor,
        init_state: Any | None = None,
    ) -> tuple[list[torch.Tensor], Any]:
        B, T, D = sequence.shape
        outs: list[torch.Tensor] = []
        state = init_state
        tokens_in_ram = 0
        for i in range(0, T, self.chunk):
            j = min(i + self.chunk, T)
            piece = sequence[:, i:j, :]
            out, state = self.step_fn(piece, state)
            if isinstance(state, torch.Tensor):
                state = state.detach()
            elif isinstance(state, (tuple, list)):
                state = tuple(
                    s.detach() if isinstance(s, torch.Tensor) else s for s in state
                )
            out = out.detach()
            outs.append(out)
            tokens_in_ram += out.shape[1] * out.shape[0]
            self.stats["chunks"] += 1
            if self.max_memory_tokens > 0:
                while tokens_in_ram > self.max_memory_tokens and len(outs) > 1:
                    oldest = outs.pop(0)
                    tokens_in_ram -= oldest.shape[1] * oldest.shape[0]
                    self.stats["evicted"] += 1
                    if self.disk_archive is not None:
                        pooled = oldest.mean(dim=1)
                        if pooled.dim() == 2:
                            pooled = pooled.mean(dim=0)
                        try:
                            self.disk_archive.append(
                                pooled,
                                meta={"evicted_chunk_tokens": int(oldest.shape[1])},
                            )
                            self.stats["disk_spills"] += 1
                        except Exception:
                            pass
        return outs, state


class LongContextMonitor:
    """Per-position NLL monitor for long-context regression tests."""

    def __init__(self, positions: tuple[int, ...] = (100, 1000, 10000, 100000)):
        self.positions = tuple(sorted(set(int(p) for p in positions)))
        self._sums = {p: 0.0 for p in self.positions}
        self._cnts = {p: 0 for p in self.positions}

    def add(self, nll_per_token: torch.Tensor | np.ndarray) -> None:
        if isinstance(nll_per_token, torch.Tensor):
            arr = nll_per_token.detach().cpu().float().numpy()
        else:
            arr = np.asarray(nll_per_token, dtype=np.float32)
        arr = arr.reshape(-1)
        N = arr.size
        for p in self.positions:
            if p > N:
                break
            lo = max(0, p - 100)
            self._sums[p] += float(arr[lo:p].mean())
            self._cnts[p] += 1

    def report(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for p in self.positions:
            if self._cnts[p] == 0:
                out[p] = float("nan")
            else:
                out[p] = self._sums[p] / self._cnts[p]
        return out

    def ok(self, threshold: float = 0.05) -> bool:
        rep = self.report()
        vals = [v for v in rep.values() if not math.isnan(v)]
        if len(vals) < 2:
            return True
        return (vals[-1] - vals[0]) < threshold


class StreamingInfiniteEvaluator:
    """Resume-able chunked evaluator for sequences > 1M tokens."""

    def __init__(
        self,
        step_fn,
        chunk_size: int = 8192,
        checkpoint_every: int = 8,
        checkpoint_path: str | os.PathLike | None = None,
        accumulator=None,
    ):
        self.step_fn = step_fn
        self.chunk_size = int(chunk_size)
        self.checkpoint_every = int(checkpoint_every)
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.accumulator = accumulator
        self.state: Any = None
        self.chunks_done: int = 0
        self.tokens_done: int = 0

    def _save_ckpt(self) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump({
                    "state": self.state,
                    "chunks_done": self.chunks_done,
                    "tokens_done": self.tokens_done,
                }, f)
            tmp.replace(self.checkpoint_path)
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    def resume(self) -> bool:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return False
        try:
            with open(self.checkpoint_path, "rb") as f:
                blob = pickle.load(f)
            self.state = blob.get("state", None)
            self.chunks_done = int(blob.get("chunks_done", 0))
            self.tokens_done = int(blob.get("tokens_done", 0))
            return True
        except Exception:
            return False

    def run(self, sequence_iter) -> dict:
        t0 = time.time()
        for piece in sequence_iter:
            if piece.dim() != 3:
                raise ValueError(f"expected [B, T, D], got {tuple(piece.shape)}")
            T = piece.size(1)
            for i in range(0, T, self.chunk_size):
                j = min(i + self.chunk_size, T)
                sub = piece[:, i:j, :]
                out, self.state = self.step_fn(sub, self.state)
                if isinstance(self.state, torch.Tensor):
                    self.state = self.state.detach()
                elif isinstance(self.state, (tuple, list)):
                    self.state = tuple(
                        s.detach() if isinstance(s, torch.Tensor) else s
                        for s in self.state
                    )
                self.chunks_done += 1
                self.tokens_done += sub.shape[0] * sub.shape[1]
                if self.accumulator is not None:
                    try:
                        self.accumulator(out, {
                            "chunks": self.chunks_done,
                            "tokens": self.tokens_done,
                        })
                    except Exception:
                        pass
                if (
                    self.checkpoint_every > 0
                    and self.chunks_done % self.checkpoint_every == 0
                ):
                    self._save_ckpt()
        wall = time.time() - t0
        return {
            "chunks_done": self.chunks_done,
            "tokens_done": self.tokens_done,
            "wall_s": wall,
            "tokens_per_s": self.tokens_done / max(wall, 1e-9),
        }


__all__ = [
    "RotaryPositionEncoding",
    "LocalGQAttention",
    "HierarchicalMemoryConfig",
    "HierarchicalMemory",
    "DeltaCompress",
    "AdaptiveSlowTau",
    "SSMDiagScan",
    "ExternalVectorMemory",
    "DiskMemmapArchive",
    "InfiniteReaderConfig",
    "InfiniteContextReader",
    "ChunkedStateCarry",
    "LongContextMonitor",
    "StreamingInfiniteEvaluator",
]
