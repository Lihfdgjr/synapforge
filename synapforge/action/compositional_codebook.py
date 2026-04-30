"""
Hierarchical L1/L2 compositional codebook — true neural program synthesis.

Why this matters (per agent synthesis 2026-04-30):
  flat NeuroMCP K=64 prototypes can only LOOKUP existing skills.
  L1 primitives (frozen after warmup) + L2 compounds (sequences of L1 IDs
  encoded by causal attention pooler) → effective K explodes combinatorially:
  256 primitives × depth-3 compounds ≈ 10^4 distinct skills.

Architecture:

  L1: standard PerDomainNeuroMCP-like primitives (frozen post-warmup).
      ID space [0, K_L1).

  L2: sequence of L1 IDs encoded as compound prototype.
      shape: (max_compound_len,) int64
      embedding: small causal attention pooler over L1 embeddings @ those IDs
      → single d-dim vector that lives in same space as L1 primitives, so
      L2 compounds can be retrieved by cosine to intent (same HNSW index).

Co-firing detection (Hebbian compound minting):
  When two consecutive L1 activations in the recent action trace fire with
  mutual information > tau, propose a new L2 compound capturing the pair.
  Per Discovery of Options via Meta-Gradients (2102.05492).

Routing:
  intent_emb -> top-k from union(L1, L2) prototypes
  if top-1 is L2: expand to its L1 sequence, dispatch each step
  if top-1 is L1: dispatch directly

Persistence: same skill_log / HNSW index, but with `level: 1|2` flag and
`compound_seq: List[int]` for L2.

Effort budget per agent: 5-7 days, ~1200 LOC, 80 GPU-h. This file is the
~400-line scaffold — full ablations + Hebbian co-firing logic come later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompoundPrototype:
    compound_id: int
    domain: str
    l1_sequence: List[int]
    embedding: Optional[torch.Tensor] = None
    usage_count: int = 0
    co_fire_score: float = 0.0
    hebbian_strength: float = 0.5
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")


class TemporalAttentionPooler(nn.Module):
    """Causal multi-head attention over a sequence of L1 prototype embeddings.

    Pools the (T, d) sequence of primitives into a single d-dim compound
    embedding via a learned [CLS]-like query token attending over the seq.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        max_len: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, l1_embs: torch.Tensor) -> torch.Tensor:
        """l1_embs: (B, T, d) -> compound: (B, d)"""
        B = l1_embs.shape[0]
        cls = self.cls_token.expand(B, 1, self.d_model)
        x = torch.cat([cls, l1_embs], dim=1)
        out, _ = self.attn(x, x, x, need_weights=False)
        compound = self.norm(out[:, 0])
        return self.proj(compound)


class CoFiringDetector:
    """Tracks recent L1 activations; proposes new L2 compounds when
    consecutive pairs co-fire above mutual-information threshold.

    Lightweight online estimator: maintains (K_L1, K_L1) co-occurrence matrix
    and unigram counts; computes pointwise MI and flags pairs > tau.
    """

    def __init__(
        self,
        k_l1: int,
        mi_threshold: float = 0.30,
        history_window: int = 256,
        min_pair_count: int = 5,
    ) -> None:
        self.k_l1 = k_l1
        self.mi_threshold = mi_threshold
        self.history_window = history_window
        self.min_pair_count = min_pair_count

        self.unigram = torch.zeros(k_l1)
        self.bigram = torch.zeros(k_l1, k_l1)
        self.history: List[int] = []
        self.proposed_pairs: set = set()

    def observe(self, l1_id: int) -> Optional[Tuple[int, int]]:
        """Record an activation. Returns (prev, curr) pair if it crosses MI threshold."""
        if l1_id < 0 or l1_id >= self.k_l1:
            return None

        self.unigram[l1_id] += 1
        self.history.append(l1_id)
        if len(self.history) > self.history_window:
            self.history.pop(0)

        if len(self.history) < 2:
            return None

        prev = self.history[-2]
        curr = l1_id
        self.bigram[prev, curr] += 1

        pair = (prev, curr)
        if pair in self.proposed_pairs:
            return None
        if self.bigram[prev, curr] < self.min_pair_count:
            return None

        n_total = max(self.unigram.sum().item(), 1.0)
        p_prev = self.unigram[prev] / n_total
        p_curr = self.unigram[curr] / n_total
        p_pair = self.bigram[prev, curr] / max(n_total - 1, 1.0)
        if p_prev <= 0 or p_curr <= 0 or p_pair <= 0:
            return None

        pmi = float(torch.log(p_pair / (p_prev * p_curr)))
        if pmi >= self.mi_threshold:
            self.proposed_pairs.add(pair)
            return pair
        return None

    def stats(self) -> dict:
        return {
            "history_len": len(self.history),
            "proposed_pairs": len(self.proposed_pairs),
            "unigram_max": int(self.unigram.max().item()),
            "bigram_density": float((self.bigram > 0).float().mean().item()),
        }


class HierarchicalCodebook(nn.Module):
    """L1 primitives + L2 compounds, both routable via shared cosine lookup.

    Public API:
      forward(intent_emb)        -> (action, info)
      mint_compound(l1_seq)      -> compound_id  (called by CoFiringDetector)
      execute_compound(cid)      -> List[(l1_id, action_vec)]   for dispatch
      freeze_l1()                -> stop updating L1 prototypes
      step_plasticity()          -> co-firing detection + compound minting
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int = 64,
        k_l1: int = 64,
        max_compounds: int = 4096,
        max_compound_len: int = 8,
        mi_threshold: float = 0.30,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.k_l1 = k_l1
        self.max_compounds = max_compounds
        self.max_compound_len = max_compound_len

        self.l1_protos = nn.Parameter(torch.randn(k_l1, d_model) * 0.02)
        self.l1_actions = nn.Parameter(torch.randn(k_l1, action_dim) * 0.02)
        self.l1_frozen = False

        self.pooler = TemporalAttentionPooler(d_model=d_model, max_len=max_compound_len)
        self.compounds: Dict[int, CompoundPrototype] = {}
        self._next_compound_id = 0

        self.detector = CoFiringDetector(
            k_l1=k_l1,
            mi_threshold=mi_threshold,
        )

        self._last_l1_pid: Dict[str, int] = {}

    def freeze_l1(self) -> None:
        self.l1_frozen = True
        self.l1_protos.requires_grad_(False)
        self.l1_actions.requires_grad_(False)

    def unfreeze_l1(self) -> None:
        self.l1_frozen = False
        self.l1_protos.requires_grad_(True)
        self.l1_actions.requires_grad_(True)

    def _all_proto_embs(self) -> Tuple[torch.Tensor, List[int], List[int]]:
        """Concatenate L1 protos + L2 compound embeddings.

        Returns (embs, levels, ids) where:
          embs: (K_L1 + K_L2, d) on l1_protos.device
          levels: 1 for L1, 2 for L2
          ids: original id within its level
        """
        device = self.l1_protos.device
        all_embs = [self.l1_protos]
        levels = [1] * self.k_l1
        ids = list(range(self.k_l1))

        if self.compounds:
            for cid, c in self.compounds.items():
                if c.embedding is not None:
                    all_embs.append(c.embedding.to(device).unsqueeze(0))
                    levels.append(2)
                    ids.append(cid)

        return torch.cat(all_embs, dim=0), levels, ids

    def forward(
        self,
        intent_emb: torch.Tensor,
        domain: str = "default",
    ) -> Tuple[torch.Tensor, dict]:
        """intent_emb: (B, d) -> action: (B, action_dim), info dict."""
        all_embs, levels, ids = self._all_proto_embs()

        h_n = F.normalize(intent_emb, dim=-1)
        p_n = F.normalize(all_embs, dim=-1)
        sim = h_n @ p_n.t()

        weights = F.softmax(sim, dim=-1)
        top_idx = weights.argmax(dim=-1)
        top1 = int(top_idx[0].item())

        l1_action_part = (weights[:, :self.k_l1]) @ self.l1_actions
        action = l1_action_part

        top_level = levels[top1]
        top_id = ids[top1]

        if top_level == 1:
            self.detector.observe(top_id)
            self._last_l1_pid[domain] = top_id

        eps = 1e-8
        ent = -(weights * (weights + eps).log()).sum(-1).mean()

        return action, {
            "top_level": top_level,
            "top_id": top_id,
            "entropy": float(ent.item()),
            "k_l1": self.k_l1,
            "k_l2": len(self.compounds),
            "weights": weights.detach(),
        }

    @torch.no_grad()
    def mint_compound(
        self,
        l1_sequence: List[int],
        domain: str = "default",
    ) -> int:
        """Add a new L2 compound from a sequence of L1 prototype IDs."""
        if len(self.compounds) >= self.max_compounds:
            return -1
        if not l1_sequence or len(l1_sequence) > self.max_compound_len:
            return -1

        device = self.l1_protos.device
        seq_embs = self.l1_protos[torch.tensor(l1_sequence, device=device)]
        compound_emb = self.pooler(seq_embs.unsqueeze(0)).squeeze(0)

        cid = self._next_compound_id
        self._next_compound_id += 1
        compound = CompoundPrototype(
            compound_id=cid,
            domain=domain,
            l1_sequence=list(l1_sequence),
            embedding=compound_emb.detach().cpu(),
        )
        self.compounds[cid] = compound
        return cid

    def execute_compound(self, compound_id: int) -> List[Tuple[int, torch.Tensor]]:
        """Expand a compound into a sequence of (l1_id, action_vec) tuples."""
        c = self.compounds.get(compound_id)
        if c is None:
            return []
        out = []
        for l1_id in c.l1_sequence:
            if 0 <= l1_id < self.k_l1:
                out.append((l1_id, self.l1_actions[l1_id].detach()))
        return out

    @torch.no_grad()
    def step_plasticity(self, domain: str = "default") -> dict:
        """Check detector for co-fire pair → mint compound if eligible."""
        prev = self._last_l1_pid.get(domain)
        if prev is None or len(self.detector.history) < 2:
            return self.detector.stats()

        for pair in list(self.detector.proposed_pairs):
            already = any(c.l1_sequence[:2] == [pair[0], pair[1]]
                          for c in self.compounds.values())
            if not already:
                self.mint_compound([pair[0], pair[1]], domain=domain)

        return {
            **self.detector.stats(),
            "k_l2": len(self.compounds),
        }

    def stats(self) -> dict:
        return {
            "k_l1": self.k_l1,
            "k_l2": len(self.compounds),
            "l1_frozen": self.l1_frozen,
            "max_compounds": self.max_compounds,
            **self.detector.stats(),
        }
