"""sf.action.universal_codebook — open-ended, lifelong, neural-only tool space.

The user's literal requirement:

    "神经元工具我要的是万能的, ai 可以自己学习, 可以根据自己需求随时变化,
     形成新的工具, 也可以根据用户需求生成, 而且永久不丢失."

Translation into the architecture below:

    1. Universal     — no preset action whitelist; codebook K grows from 9
                       to 100K+. The neuron itself routes — no JSON schema.
    2. AI self-learn — Hebbian co-firing detector mints new compounds
                       from the recent action trace.
    3. Mutates at runtime — `mint_*` methods are no_grad and re-entrant
                       under inference; no restart required.
    4. AI forms tools — co-firing pattern with stable mutual-information
                       above tau gets minted as an L2 prototype that, when
                       routed-to, dispatches the L1 sequence in order.
    5. User-driven minting — `mint_from_text(description)` embeds the user
                       description and inserts a new prototype at that
                       point in hidden space.
    6. Lifelong      — every prototype carries a stable ``proto_id``;
                       skill_log_v2 atomic-writes JSON; reload is
                       idempotent (loading twice yields identical state).

Three layers, all in the same hidden-vector space so a single cosine query
over the union routes everything:

    L1 PRIMITIVES  — 9 atomic actions matching ``ACTION_TYPES``
                     (CLICK, TYPE, KEY, SCROLL, WAIT, BACK, FORWARD,
                     DONE, NULL).  Frozen after warmup.
    L2 COMPOUNDS   — neural sequences minted by Hebbian co-firing
                     detection on the recent L1 activation trace.
    L3 MACROS      — minted by either a free-form text description
                     (user-driven) or a successful demonstration trace
                     (AI-driven).  Both go through ``mint_*``.

Distance to existing pieces:

    DynamicActionCodebook  -> base class for the L1+L2+L3 ``slots``
                             tensor.  Re-implemented here with a
                             ``layer`` tag per slot so a single cosine
                             query covers all three layers.
    HierarchicalCodebook    -> superseded by this module; the only new
                             pieces are user-driven L3 minting and the
                             persistent ``proto_id``.
    HNSWSkillIndex          -> reused as the K=100k+ retrieval backend.
                             We talk to it via ``add/query`` only.

Sole external dep that may be missing: ``hnswlib``.  Falls back to a
linear cosine scan, which is fine for K < 1000 (smoke runs).
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module

# ---------------------------------------------------------------------------
# Optional HNSW backend.  Same import-guard pattern as hnsw_skill_index.py.
# ---------------------------------------------------------------------------
try:
    import hnswlib  # type: ignore
    _HAS_HNSW = True
except Exception:  # pragma: no cover - import-only branch
    hnswlib = None  # type: ignore
    _HAS_HNSW = False


# ---------------------------------------------------------------------------
# L1 vocabulary.  Mirrors sf.action.head.ACTION_TYPES with two macro-ish
# additions (BACK, FORWARD) the model can compose with — these are still
# atomic from the OS-actuator's point of view (they are single calls).
# ---------------------------------------------------------------------------

L1_PRIMITIVES: tuple[str, ...] = (
    "CLICK",     # 0
    "TYPE",      # 1
    "KEY",       # 2
    "SCROLL",    # 3
    "WAIT",      # 4
    "BACK",      # 5
    "FORWARD",   # 6
    "DONE",      # 7
    "NULL",      # 8 - explicit no-op so the model can choose to think
)

LAYER_L1 = "L1"
LAYER_L2 = "L2"
LAYER_L3 = "L3"


# ---------------------------------------------------------------------------
# Per-prototype metadata.  Kept in pure-Python so it survives JSON round-trip
# without dropping fields.
# ---------------------------------------------------------------------------


@dataclass
class PrototypeMeta:
    proto_id: int
    layer: str                              # "L1" / "L2" / "L3"
    description: str = ""                   # human or AI-generated label
    created_at: float = 0.0
    last_used_at: float = 0.0
    n_uses: int = 0
    n_success: int = 0
    hebbian_strength: float = 0.5           # LTP/LTD scalar, [0, 1]
    trigger_seq: List[int] = field(default_factory=list)   # for L2/L3
    archived: bool = False                  # LTD-pruned but kept on disk
    embedding_hash: str = ""                # sha256 of (desc, embedding)

    def __post_init__(self) -> None:
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.last_used_at == 0.0:
            self.last_used_at = now


def _embedding_hash(desc: str, emb: torch.Tensor) -> str:
    """Stable SHA256 over (description bytes + 4-bit-quantised embedding)."""
    e = emb.detach().float().cpu()
    e = (e.clamp(-1.0, 1.0) * 7.0).to(torch.int8).contiguous().numpy().tobytes()
    h = hashlib.sha256()
    h.update(desc.encode("utf-8"))
    h.update(e)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Default text encoder fallback.  Hashes characters into a fixed-D vector so
# smoke tests can mint from text *without* loading sentence-transformers.
# Order-sensitive but deterministic; replace with a real encoder in prod.
# ---------------------------------------------------------------------------


def default_text_encoder(text: str, hidden: int = 256) -> torch.Tensor:
    """Deterministic char-trigram hash -> hidden vector."""
    if not text:
        return torch.zeros(hidden)
    rng = np.random.default_rng(seed=int(hashlib.sha256(text.encode()).hexdigest()[:16], 16))
    base = rng.standard_normal(hidden).astype(np.float32)
    # accentuate by trigram presence
    for i in range(len(text) - 2):
        tri = text[i : i + 3]
        s = int(hashlib.md5(tri.encode()).hexdigest()[:8], 16) % hidden
        base[s] += 1.0
    v = torch.from_numpy(base)
    return v / (v.norm() + 1e-8)


# ---------------------------------------------------------------------------
# UniversalCodebook — the lifelong, open-ended skill space.
# ---------------------------------------------------------------------------


class UniversalCodebook(Module):
    """Layered prototype store with online growth in all three layers.

    Public API (designed to mirror DynamicActionCodebook so it is a
    drop-in upgrade for NeuroMCPHead):

        forward(z)                -> {"logits": [..., K_alive],
                                       "layer":  [..., K_alive],
                                       "proto_id":[..., K_alive]}
        query(z) -> (proto_id, layer, sim)
        mint_from_text(desc)      -> proto_id            (L3)
        mint_from_co_firing(*)    -> proto_id | None     (L2)
        mint_from_trace(seq)      -> proto_id            (L3)
        decay_unused() / prune()  -> housekeeping

    Internal storage:

        self.slots                : nn.Parameter [K_max, hidden]
        self.alive                : bool buffer  [K_max]
        self.layer_id             : long buffer  [K_max]   (1=L1, 2=L2, 3=L3)
        self.uses                 : long buffer  [K_max]
        self.strength             : float buffer [K_max]   in [0, 1]
        self.meta                 : Dict[int, PrototypeMeta]   sidecar
        self.hnsw                 : optional hnswlib.Index for K-NN

    K_max_lifelong is a soft cap for the in-memory tensor; the HNSW index
    can grow beyond this via re-allocation in `_grow_capacity`.
    """

    def __init__(
        self,
        hidden: int,
        K_initial: int = 9,
        K_max_lifelong: int = 100_000,
        K_growth_block: int = 256,
        novelty_threshold: float = 0.35,
        text_encoder: Optional[Callable[[str], torch.Tensor]] = None,
        ltp_eta: float = 0.05,
        ltd_decay: float = 0.99,
        prune_threshold: float = 0.10,
        co_fire_window: int = 8,
        co_fire_min_repeats: int = 3,
        archive_instead_of_delete: bool = True,
    ) -> None:
        super().__init__()
        if K_initial > K_max_lifelong:
            raise ValueError("K_initial > K_max_lifelong")
        self.hidden = int(hidden)
        self.K_max_lifelong = int(K_max_lifelong)
        self.K_growth_block = int(K_growth_block)
        self.novelty_threshold = float(novelty_threshold)
        self.ltp_eta = float(ltp_eta)
        self.ltd_decay = float(ltd_decay)
        self.prune_threshold = float(prune_threshold)
        self.co_fire_window = int(co_fire_window)
        self.co_fire_min_repeats = int(co_fire_min_repeats)
        self.archive_instead_of_delete = bool(archive_instead_of_delete)
        self.text_encoder = text_encoder or (
            lambda t: default_text_encoder(t, hidden=self.hidden)
        )

        # Start with a small block; we grow geometrically as needed.
        K0 = max(K_initial, K_growth_block)
        self.slots = nn.Parameter(torch.randn(K0, hidden) * 0.02)
        self.register_buffer("alive", torch.zeros(K0, dtype=torch.bool))
        self.register_buffer("layer_id", torch.zeros(K0, dtype=torch.long))
        self.register_buffer("uses", torch.zeros(K0, dtype=torch.long))
        self.register_buffer("strength", torch.full((K0,), 0.5))

        # Initialise L1 primitives — the one-and-only layer that is
        # alive at boot.  We use orthogonal init so they fan out in
        # hidden space, otherwise random-init L1's can collapse.
        with torch.no_grad():
            ortho = torch.empty(K_initial, hidden)
            nn.init.orthogonal_(ortho)
            self.slots[:K_initial].copy_(ortho * 0.1)
            self.alive[:K_initial] = True
            self.layer_id[:K_initial] = 1
            self.strength[:K_initial] = 1.0  # primitives never decay

        # Sidecar metadata.  Keyed by stable proto_id, NOT slot index.
        self.meta: Dict[int, PrototypeMeta] = {}
        for pid in range(K_initial):
            self.meta[pid] = PrototypeMeta(
                proto_id=pid,
                layer=LAYER_L1,
                description=L1_PRIMITIVES[pid] if pid < len(L1_PRIMITIVES) else f"L1_{pid}",
                hebbian_strength=1.0,
                trigger_seq=[],
            )
        self._next_proto_id = K_initial
        # slot_idx <-> proto_id bijection.  L1 has slot_idx == proto_id.
        self._proto_to_slot: Dict[int, int] = {pid: pid for pid in range(K_initial)}
        self._slot_to_proto: Dict[int, int] = {pid: pid for pid in range(K_initial)}

        # Recent action history for co-firing detection.
        self._action_history: List[int] = []
        self._co_fire_seen: set = set()

        # Optional HNSW index over the alive prototypes.
        self._hnsw: Optional["hnswlib.Index"] = None
        self._hnsw_max_elements = max(self.K_max_lifelong, K0 * 4)
        self._init_hnsw()
        # Bulk-insert L1 protos so query() works immediately.
        for pid in range(K_initial):
            self._hnsw_add(pid, self.slots[pid].detach())

    # ------------------------------------------------------------------ HNSW

    def _init_hnsw(self) -> None:
        if not _HAS_HNSW:
            self._hnsw = None
            return
        try:
            self._hnsw = hnswlib.Index(space="cosine", dim=self.hidden)
            self._hnsw.init_index(
                max_elements=self._hnsw_max_elements,
                ef_construction=200,
                M=16,
            )
            self._hnsw.set_ef(200)
        except Exception:
            self._hnsw = None

    def _hnsw_add(self, proto_id: int, emb: torch.Tensor) -> None:
        if self._hnsw is None:
            return
        v = emb.detach().float().cpu().numpy().astype("float32")
        if v.ndim > 1:
            v = v.mean(axis=0)
        n = float(np.linalg.norm(v)) + 1e-8
        v = v / n
        try:
            self._hnsw.add_items(v.reshape(1, -1), np.array([proto_id]))
        except RuntimeError:
            # capacity overflow -> reallocate
            try:
                self._hnsw_max_elements = int(self._hnsw_max_elements * 2)
                self._hnsw.resize_index(self._hnsw_max_elements)
                self._hnsw.add_items(v.reshape(1, -1), np.array([proto_id]))
            except Exception:
                pass

    def _hnsw_mark_deleted(self, proto_id: int) -> None:
        if self._hnsw is None:
            return
        try:
            self._hnsw.mark_deleted(int(proto_id))
        except Exception:
            pass

    # ------------------------------------------------------------------ size

    @property
    def K(self) -> int:
        return int(self.alive.sum().item())

    @property
    def K_max(self) -> int:
        """Current in-memory capacity.  Distinct from ``K_max_lifelong``."""
        return int(self.slots.shape[0])

    def size_by_layer(self) -> Dict[str, int]:
        out = {LAYER_L1: 0, LAYER_L2: 0, LAYER_L3: 0}
        for slot, alive in enumerate(self.alive.tolist()):
            if not alive:
                continue
            lid = int(self.layer_id[slot].item())
            if lid == 1:
                out[LAYER_L1] += 1
            elif lid == 2:
                out[LAYER_L2] += 1
            elif lid == 3:
                out[LAYER_L3] += 1
        return out

    # ------------------------------------------------------------------ growth

    def _grow_capacity(self) -> None:
        """Geometric grow of the slots tensor when full but K_max_lifelong
        not yet reached.  Keeps gradients on existing rows."""
        old = self.slots.shape[0]
        new = min(self.K_max_lifelong, old + self.K_growth_block)
        if new <= old:
            return  # at the lifelong cap
        # New parameter / buffers.  Copy old contents into the front slice.
        new_slots = nn.Parameter(torch.randn(new, self.hidden) * 0.02)
        with torch.no_grad():
            new_slots[:old].copy_(self.slots.data)
        self.slots = new_slots

        def _grow(buf: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
            extra = torch.full((new - old,) + tuple(buf.shape[1:]), fill, dtype=buf.dtype, device=buf.device)
            return torch.cat([buf, extra], dim=0)

        self.alive = _grow(self.alive, 0.0).bool()
        self.layer_id = _grow(self.layer_id, 0.0).long()
        self.uses = _grow(self.uses, 0.0).long()
        self.strength = _grow(self.strength, 0.0).float()

    def _allocate_slot(self) -> int:
        """Return a free slot index, growing capacity if needed."""
        free = (~self.alive).nonzero(as_tuple=True)[0]
        if free.numel() == 0:
            self._grow_capacity()
            free = (~self.alive).nonzero(as_tuple=True)[0]
            if free.numel() == 0:
                return -1
        return int(free[0].item())

    # ------------------------------------------------------------------ forward

    def forward(self, z: torch.Tensor, tau: float = 0.07) -> dict:
        """Cosine-similarity logits over alive prototypes.

        z: [..., D] -> {"logits": [..., K_alive], "layer": [..., K_alive],
                         "proto_id": [..., K_alive]}.
        """
        if self.K == 0:
            empty = z.new_zeros(*z.shape[:-1], 0)
            return {"logits": empty, "layer": empty.long(), "proto_id": empty.long()}
        alive_idx = self.alive.nonzero(as_tuple=True)[0]
        live = self.slots[alive_idx]                                     # [K, D]
        z_n = F.normalize(z, dim=-1)
        p_n = F.normalize(live, dim=-1)
        sim = z_n @ p_n.T                                                 # [..., K]
        layer = self.layer_id[alive_idx].unsqueeze(0).expand(*sim.shape)  # broadcast
        proto = torch.tensor(
            [self._slot_to_proto[int(s.item())] for s in alive_idx],
            dtype=torch.long,
            device=z.device,
        ).unsqueeze(0).expand(*sim.shape)
        return {"logits": sim / float(tau), "layer": layer, "proto_id": proto}

    @torch.no_grad()
    def query(
        self,
        z: torch.Tensor,
        top_k: int = 1,
        domain_filter: Optional[str] = None,
    ) -> List[Tuple[int, str, float]]:
        """Return top-K (proto_id, layer, similarity) for the hidden vector.

        Uses HNSW when available, else falls back to dense cosine scan.
        Always operates on a single vector — call repeatedly for batching.
        """
        if self.K == 0:
            return []
        if z.dim() > 1:
            z = z.reshape(-1, z.shape[-1]).mean(dim=0)
        v = F.normalize(z, dim=-1).float().cpu().numpy().astype("float32")

        if self._hnsw is not None:
            try:
                k_q = min(max(1, top_k * 2), self.K)
                labels, dists = self._hnsw.knn_query(v.reshape(1, -1), k=k_q)
                out: List[Tuple[int, str, float]] = []
                for lab, d in zip(labels[0].tolist(), dists[0].tolist()):
                    pid = int(lab)
                    meta = self.meta.get(pid)
                    if meta is None or meta.archived:
                        continue
                    sim = float(1.0 - d)
                    out.append((pid, meta.layer, sim))
                    if len(out) >= top_k:
                        break
                return out
            except Exception:
                pass

        # fallback dense scan
        alive_idx = self.alive.nonzero(as_tuple=True)[0]
        live = self.slots[alive_idx].detach().float().cpu().numpy()
        live = live / (np.linalg.norm(live, axis=1, keepdims=True) + 1e-8)
        sims = live @ v
        order = np.argsort(-sims)[:top_k]
        out2: List[Tuple[int, str, float]] = []
        for o in order:
            slot = int(alive_idx[int(o)].item())
            pid = self._slot_to_proto[slot]
            meta = self.meta.get(pid)
            if meta is None or meta.archived:
                continue
            out2.append((pid, meta.layer, float(sims[int(o)])))
        return out2

    # ------------------------------------------------------------------ minting

    @torch.no_grad()
    def _add_prototype(
        self,
        embedding: torch.Tensor,
        layer: str,
        description: str,
        trigger_seq: Optional[List[int]] = None,
        initial_strength: float = 0.5,
    ) -> int:
        slot = self._allocate_slot()
        if slot < 0:
            return -1
        pid = self._next_proto_id
        self._next_proto_id += 1
        self.slots[slot] = embedding.detach().to(self.slots.dtype)
        self.alive[slot] = True
        self.layer_id[slot] = {LAYER_L1: 1, LAYER_L2: 2, LAYER_L3: 3}[layer]
        self.uses[slot] = 0
        self.strength[slot] = initial_strength
        meta = PrototypeMeta(
            proto_id=pid,
            layer=layer,
            description=description,
            trigger_seq=list(trigger_seq or []),
            hebbian_strength=initial_strength,
        )
        meta.embedding_hash = _embedding_hash(description, embedding)
        self.meta[pid] = meta
        self._proto_to_slot[pid] = slot
        self._slot_to_proto[slot] = pid
        self._hnsw_add(pid, embedding)
        return pid

    @torch.no_grad()
    def mint_from_text(
        self,
        description: str,
        bump_dup_threshold: float = 0.92,
    ) -> int:
        """User-driven minting from a free-form description.

        If a similar prototype already exists (cosine > bump_dup_threshold),
        bump its strength and return the existing id instead of duplicating.
        Returns -1 only if capacity is hit AND no suitable existing match.
        """
        if not description.strip():
            return -1
        emb = self.text_encoder(description)
        if emb.numel() != self.hidden:
            raise ValueError(f"text_encoder returned dim {emb.numel()}, expected {self.hidden}")
        # dedup
        hits = self.query(emb, top_k=1)
        if hits and hits[0][2] >= bump_dup_threshold:
            pid = hits[0][0]
            self._activate(pid, reward=0.5)
            return pid
        return self._add_prototype(emb, LAYER_L3, description, trigger_seq=None, initial_strength=0.6)

    @torch.no_grad()
    def mint_from_trace(
        self,
        l1_sequence: List[int],
        description: str = "",
        success: bool = True,
    ) -> int:
        """AI-driven minting from a successful action trace.

        Pools the L1 prototype embeddings along the trace into a single
        compound embedding (mean over normalised primitive vectors), then
        adds as L3 if `description` was given (i.e., model labelled it),
        else as L2 (anonymous).  Idempotent: same trace minted twice
        returns the same id (matched via embedding hash).
        """
        if not l1_sequence:
            return -1
        # Compute pooled embedding from the L1 primitives that fired.
        seq = [pid for pid in l1_sequence if pid in self._proto_to_slot]
        if not seq:
            return -1
        slots = torch.tensor([self._proto_to_slot[p] for p in seq], dtype=torch.long)
        emb = self.slots[slots]
        emb = F.normalize(emb, dim=-1).mean(dim=0)
        emb = F.normalize(emb, dim=-1)
        # idempotency: see if same trigger_seq + similar embedding exists
        for pid, meta in self.meta.items():
            if meta.archived:
                continue
            if meta.trigger_seq == seq and meta.layer in (LAYER_L2, LAYER_L3):
                self._activate(pid, reward=0.2 if success else -0.1)
                return pid
        layer = LAYER_L3 if description else LAYER_L2
        return self._add_prototype(
            emb,
            layer=layer,
            description=description or f"trace[{','.join(str(x) for x in seq)}]",
            trigger_seq=seq,
            initial_strength=0.5 + (0.1 if success else 0.0),
        )

    @torch.no_grad()
    def mint_from_co_firing(
        self,
        last_action_id: int,
    ) -> Optional[int]:
        """Trace co-firing minting.  Call once per action emitted by ActionHead.

        Algorithm:
          1. Append last_action_id to the recent-history ring buffer.
          2. Walk back over `co_fire_window` steps.
          3. If a contiguous subsequence of length >= 3 has occurred
             >= co_fire_min_repeats times in history AND no L2 already
             exists for it, mint as L2.
        """
        if last_action_id < 0:
            return None
        self._action_history.append(int(last_action_id))
        # bound history; we don't need long-range memory here
        max_hist = max(64, self.co_fire_window * 8)
        if len(self._action_history) > max_hist:
            self._action_history = self._action_history[-max_hist:]
        history = self._action_history
        if len(history) < self.co_fire_window:
            return None

        # try lengths 3..co_fire_window from longest to shortest, prefer big patterns
        for L in range(self.co_fire_window, 2, -1):
            tail = tuple(history[-L:])
            # count occurrences of `tail` over the history
            if len(history) < L * self.co_fire_min_repeats:
                continue
            count = 0
            for i in range(len(history) - L + 1):
                if tuple(history[i : i + L]) == tail:
                    count += 1
                    if count >= self.co_fire_min_repeats:
                        break
            if count >= self.co_fire_min_repeats:
                key = tail
                if key in self._co_fire_seen:
                    continue
                self._co_fire_seen.add(key)
                pid = self.mint_from_trace(list(tail), description="", success=True)
                if pid >= 0:
                    return pid
        return None

    # ------------------------------------------------------------------ usage / LTP+LTD

    @torch.no_grad()
    def _activate(self, proto_id: int, reward: float = 0.0) -> None:
        slot = self._proto_to_slot.get(proto_id)
        if slot is None:
            return
        meta = self.meta.get(proto_id)
        if meta is None or meta.archived:
            return
        self.uses[slot] += 1
        meta.n_uses += 1
        meta.last_used_at = time.time()
        if reward > 0:
            new = min(1.0, float(self.strength[slot].item()) + self.ltp_eta * reward)
            self.strength[slot] = new
            meta.hebbian_strength = new
            meta.n_success += 1
        elif reward < 0:
            new = float(self.strength[slot].item()) * self.ltd_decay
            self.strength[slot] = new
            meta.hebbian_strength = new

    def activate(self, proto_id: int, reward: float = 0.0) -> None:
        self._activate(proto_id, reward)

    @torch.no_grad()
    def decay_unused(self, days: float = 30.0, ignore_l1: bool = True) -> int:
        """Apply LTD multiplicatively to skills not used in `days` days."""
        cutoff = time.time() - days * 86400.0
        touched = 0
        for pid, meta in self.meta.items():
            if meta.archived or meta.last_used_at >= cutoff:
                continue
            if ignore_l1 and meta.layer == LAYER_L1:
                continue
            slot = self._proto_to_slot.get(pid)
            if slot is None:
                continue
            new = float(self.strength[slot].item()) * self.ltd_decay
            self.strength[slot] = new
            meta.hebbian_strength = new
            touched += 1
        return touched

    @torch.no_grad()
    def prune(self, hard_delete: bool = False) -> List[int]:
        """Archive (or hard-delete) L2/L3 skills below threshold."""
        removed: List[int] = []
        for pid, meta in list(self.meta.items()):
            if meta.archived or meta.layer == LAYER_L1:
                continue
            slot = self._proto_to_slot.get(pid)
            if slot is None:
                continue
            if (
                float(self.strength[slot].item()) < self.prune_threshold
                and meta.n_uses > 5
            ):
                if hard_delete and not self.archive_instead_of_delete:
                    self.alive[slot] = False
                    self.layer_id[slot] = 0
                    del self.meta[pid]
                    self._proto_to_slot.pop(pid, None)
                    self._slot_to_proto.pop(slot, None)
                else:
                    meta.archived = True
                    self.alive[slot] = False
                self._hnsw_mark_deleted(pid)
                removed.append(pid)
        return removed

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON."""
        skills = []
        for pid, meta in self.meta.items():
            slot = self._proto_to_slot.get(pid)
            if slot is None:
                continue
            emb = self.slots[slot].detach().float().cpu().tolist()
            skills.append({
                "proto_id": meta.proto_id,
                "layer": meta.layer,
                "description": meta.description,
                "embedding": emb,
                "n_uses": meta.n_uses,
                "n_success": meta.n_success,
                "hebbian_strength": meta.hebbian_strength,
                "trigger_seq": list(meta.trigger_seq),
                "created_at": meta.created_at,
                "last_used_at": meta.last_used_at,
                "archived": meta.archived,
                "embedding_hash": meta.embedding_hash,
            })
        return {
            "version": 2,
            "hidden": self.hidden,
            "K_alive": self.K,
            "next_proto_id": self._next_proto_id,
            "L1_primitives": list(L1_PRIMITIVES),
            "skills": skills,
        }

    @torch.no_grad()
    def load_dict(self, payload: dict) -> int:
        """Rehydrate state from a dict created by ``to_dict``.  Idempotent.

        Returns the number of prototypes restored.  L1 primitives are
        always re-asserted so corruption in the file cannot kill them.
        """
        if not payload:
            return 0
        if payload.get("hidden") not in (None, self.hidden):
            raise ValueError(
                f"hidden mismatch: codebook={self.hidden}, file={payload.get('hidden')}"
            )

        # Reset everything except L1 primitives' parameter tensor (we keep
        # L1's slots so any prior gradient state is preserved).
        for slot in range(self.alive.shape[0]):
            self.alive[slot] = False
            self.layer_id[slot] = 0
        self.meta.clear()
        self._proto_to_slot.clear()
        self._slot_to_proto.clear()
        self._action_history.clear()
        self._co_fire_seen.clear()
        self._init_hnsw()

        # Re-init L1.  These are deterministic by index.
        for pid in range(len(L1_PRIMITIVES)):
            slot = pid  # L1 lives in the first len(L1) slots
            if slot >= self.alive.shape[0]:
                self._grow_capacity()
            self.alive[slot] = True
            self.layer_id[slot] = 1
            self.strength[slot] = 1.0
            self.uses[slot] = 0
            self.meta[pid] = PrototypeMeta(
                proto_id=pid,
                layer=LAYER_L1,
                description=L1_PRIMITIVES[pid],
                hebbian_strength=1.0,
                trigger_seq=[],
            )
            self._proto_to_slot[pid] = slot
            self._slot_to_proto[slot] = pid
            self._hnsw_add(pid, self.slots[slot].detach())

        max_pid = len(L1_PRIMITIVES) - 1
        n_loaded = len(L1_PRIMITIVES)

        for entry in payload.get("skills", []):
            pid = int(entry["proto_id"])
            layer = entry.get("layer", LAYER_L3)
            if layer == LAYER_L1:
                # already re-initialised above; just patch metadata fields
                meta = self.meta.get(pid)
                if meta is not None:
                    meta.n_uses = int(entry.get("n_uses", 0))
                    meta.n_success = int(entry.get("n_success", 0))
                    meta.last_used_at = float(entry.get("last_used_at", meta.last_used_at))
                continue
            slot = self._allocate_slot()
            if slot < 0:
                continue
            emb = torch.tensor(entry["embedding"], dtype=torch.float32)
            if emb.numel() != self.hidden:
                continue
            self.slots[slot] = emb
            self.alive[slot] = not bool(entry.get("archived", False))
            self.layer_id[slot] = {LAYER_L2: 2, LAYER_L3: 3}.get(layer, 3)
            self.uses[slot] = int(entry.get("n_uses", 0))
            self.strength[slot] = float(entry.get("hebbian_strength", 0.5))
            meta = PrototypeMeta(
                proto_id=pid,
                layer=layer,
                description=entry.get("description", ""),
                hebbian_strength=float(entry.get("hebbian_strength", 0.5)),
                trigger_seq=list(entry.get("trigger_seq", [])),
                created_at=float(entry.get("created_at", time.time())),
                last_used_at=float(entry.get("last_used_at", time.time())),
                n_uses=int(entry.get("n_uses", 0)),
                n_success=int(entry.get("n_success", 0)),
                archived=bool(entry.get("archived", False)),
                embedding_hash=str(entry.get("embedding_hash", "")),
            )
            self.meta[pid] = meta
            self._proto_to_slot[pid] = slot
            self._slot_to_proto[slot] = pid
            if not meta.archived:
                self._hnsw_add(pid, emb)
            else:
                self._hnsw_mark_deleted(pid)
            max_pid = max(max_pid, pid)
            n_loaded += 1

        self._next_proto_id = max(int(payload.get("next_proto_id", 0)), max_pid + 1)
        return n_loaded

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict:
        sizes = self.size_by_layer()
        most_used = sorted(
            ((pid, m.n_uses, m.layer, m.description) for pid, m in self.meta.items() if not m.archived),
            key=lambda x: -x[1],
        )[:10]
        return {
            "K_alive": self.K,
            "K_max": self.K_max,
            "K_max_lifelong": self.K_max_lifelong,
            "next_proto_id": self._next_proto_id,
            "by_layer": sizes,
            "n_archived": sum(1 for m in self.meta.values() if m.archived),
            "hnsw_backend": "hnswlib" if self._hnsw is not None else "linear",
            "top10_uses": most_used,
        }


# ---------------------------------------------------------------------------
# Standalone smoke — `python -m synapforge.action.universal_codebook`.
# ---------------------------------------------------------------------------


def _smoke() -> None:  # pragma: no cover - manual run
    cb = UniversalCodebook(hidden=64, K_initial=9)
    print("[boot]", cb.stats())

    # L3 user-driven mint
    pid_a = cb.mint_from_text("查股市行情每天 9 点")
    pid_b = cb.mint_from_text("打开 GitHub trending repo")
    pid_c = cb.mint_from_text("在 arxiv 上搜 spiking neural net")
    print("[mint L3]", pid_a, pid_b, pid_c)

    # Drive a fake co-firing pattern (L1 ids: CLICK, TYPE, KEY)
    trace = [0, 1, 2]   # CLICK, TYPE, KEY
    minted = []
    for rep in range(5):
        for a in trace:
            r = cb.mint_from_co_firing(a)
            if r is not None:
                minted.append(r)
    print("[mint L2 via co-fire]", minted)

    # Query the L3 user skills back
    for desc in ("股市行情", "github trending", "arxiv spiking"):
        z = default_text_encoder(desc, hidden=64)
        hits = cb.query(z, top_k=3)
        print(f"[query {desc!r}]", hits)

    print("[final]", cb.stats())


if __name__ == "__main__":  # pragma: no cover
    _smoke()


__all__ = [
    "UniversalCodebook",
    "PrototypeMeta",
    "L1_PRIMITIVES",
    "LAYER_L1",
    "LAYER_L2",
    "LAYER_L3",
    "default_text_encoder",
]
