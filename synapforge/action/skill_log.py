"""
Persistent skill memory for NeuroMCP.

Each grown skill = one prototype in DynamicCodebook. We persist:
  - prototype_id     : int, stable across sessions
  - domain           : 'math'|'chat'|'code'|'web'
  - embedding        : List[float], the prototype vector (hidden_dim,)
  - usage_count      : int, total activations across all sessions
  - success_count    : int, activations that led to positive reward
  - created_at       : ISO timestamp
  - last_used_at     : ISO timestamp
  - synapse_pattern  : sparse adjacency of SparseSynapticLayer rows that fire with this proto
  - hebbian_strength : LTP-accumulated weight, [0, 1]

The user's contract:
  "用户下次可以直接使用不需要重新学习, 并且会持续优化这些功能"

Implementation:
  - On boot: load skill_log.json, restore embeddings into per-domain codebooks
  - On every activation: increment usage_count, update last_used_at
  - On positive reward: hebbian_strength += eta, capped at 1.0 (LTP)
  - On no use for N days: hebbian_strength *= decay (LTD)
  - skills with hebbian_strength < threshold get pruned during routine cleanup
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SkillEntry:
    prototype_id: int
    domain: str
    embedding: List[float]
    usage_count: int = 0
    success_count: int = 0
    created_at: str = ""
    last_used_at: str = ""
    synapse_pattern: List[int] = field(default_factory=list)
    hebbian_strength: float = 0.5

    def __post_init__(self) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.created_at:
            self.created_at = now
        if not self.last_used_at:
            self.last_used_at = now


class SkillLog:
    """JSON-backed persistent skill memory.

    Thread-safe-ish: single writer (the trainer) flushes on save_every steps.
    Readers can race but worst case = read stale snapshot.
    """

    def __init__(
        self,
        path: str | Path = "/workspace/runs/skill_log.json",
        save_every: int = 100,
        ltp_eta: float = 0.05,
        ltd_decay: float = 0.99,
        prune_threshold: float = 0.1,
    ) -> None:
        self.path = Path(path)
        self.save_every = save_every
        self.ltp_eta = ltp_eta
        self.ltd_decay = ltd_decay
        self.prune_threshold = prune_threshold

        self.skills: Dict[int, SkillEntry] = {}
        self._dirty_count = 0
        self._next_id = 0

        if self.path.exists():
            self.load()

    def load(self) -> None:
        """Restore skills from JSON on disk."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry_dict in data.get("skills", []):
            entry = SkillEntry(**entry_dict)
            self.skills[entry.prototype_id] = entry
            self._next_id = max(self._next_id, entry.prototype_id + 1)

    def save(self, force: bool = False) -> None:
        """Atomic write to JSON."""
        self._dirty_count += 1
        if not force and self._dirty_count < self.save_every:
            return
        self._dirty_count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        payload = {
            "version": "v42",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "skills": [asdict(s) for s in self.skills.values()],
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def register(
        self,
        domain: str,
        embedding: torch.Tensor,
        synapse_pattern: Optional[List[int]] = None,
    ) -> int:
        """Add a newly-grown skill. Returns prototype_id."""
        pid = self._next_id
        self._next_id += 1
        self.skills[pid] = SkillEntry(
            prototype_id=pid,
            domain=domain,
            embedding=embedding.detach().float().cpu().tolist(),
            synapse_pattern=synapse_pattern or [],
        )
        self.save()
        return pid

    def activate(self, prototype_id: int, reward: float = 0.0) -> None:
        """Record a use. reward > 0 = LTP, reward < 0 = mild LTD."""
        if prototype_id not in self.skills:
            return
        s = self.skills[prototype_id]
        s.usage_count += 1
        s.last_used_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if reward > 0:
            s.success_count += 1
            s.hebbian_strength = min(1.0, s.hebbian_strength + self.ltp_eta * reward)
        elif reward < 0:
            s.hebbian_strength *= self.ltd_decay
        self.save()

    def decay_unused(self, days_threshold: int = 7) -> int:
        """Apply LTD to skills not used in N days. Returns count touched."""
        now_s = time.time()
        cutoff_s = now_s - days_threshold * 86400
        touched = 0
        for s in self.skills.values():
            try:
                last = time.mktime(time.strptime(s.last_used_at, "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                continue
            if last < cutoff_s:
                s.hebbian_strength *= self.ltd_decay
                touched += 1
        self.save(force=True)
        return touched

    def prune(self) -> List[int]:
        """Remove skills below prune_threshold. Returns removed prototype_ids."""
        removed = [
            pid for pid, s in self.skills.items()
            if s.hebbian_strength < self.prune_threshold and s.usage_count > 5
        ]
        for pid in removed:
            del self.skills[pid]
        if removed:
            self.save(force=True)
        return removed

    def query_by_domain(self, domain: str, top_k: int = 16) -> List[SkillEntry]:
        """Get top-K most-used skills in a domain."""
        domain_skills = [s for s in self.skills.values() if s.domain == domain]
        domain_skills.sort(key=lambda s: s.hebbian_strength * (1 + s.usage_count) ** 0.5, reverse=True)
        return domain_skills[:top_k]

    def query_similar(
        self,
        intent_emb: torch.Tensor,
        domain: Optional[str] = None,
        top_k: int = 4,
    ) -> List[Tuple[SkillEntry, float]]:
        """Cosine-similarity query. Returns [(skill, score), ...]."""
        intent = intent_emb.detach().float().cpu()
        intent = intent / (intent.norm() + 1e-8)

        candidates = list(self.skills.values())
        if domain is not None:
            candidates = [s for s in candidates if s.domain == domain]
        if not candidates:
            return []

        embs = torch.tensor([s.embedding for s in candidates])
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-8)
        scores = (embs @ intent).tolist()

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1] * x[0].hebbian_strength, reverse=True)
        return scored[:top_k]

    def stats(self) -> dict:
        per_domain = {}
        for s in self.skills.values():
            per_domain.setdefault(s.domain, {"count": 0, "total_uses": 0, "avg_strength": 0.0})
            per_domain[s.domain]["count"] += 1
            per_domain[s.domain]["total_uses"] += s.usage_count
            per_domain[s.domain]["avg_strength"] += s.hebbian_strength
        for d in per_domain.values():
            if d["count"]:
                d["avg_strength"] /= d["count"]
        return {
            "total_skills": len(self.skills),
            "next_id": self._next_id,
            "per_domain": per_domain,
        }
