"""
RetrievalMemory: Track B per-user cache.

Industry pattern (Claude Memory, ChatGPT Memory, character.ai):
  - User chat NEVER updates model weights
  - Each user gets isolated retrieval cache
  - At inference: retrieve top-K relevant memories, prepend to context
  - User can delete their memory at any time (compliance + safety)

Storage:
  /workspace/runs/user_memory/<user_hash>.jsonl
  /workspace/runs/user_memory/<user_hash>.embed   (optional dense index)

Retrieval:
  - sentence-embed query, cosine top-K
  - For 375M model running locally: use small embedder (BGE-small / paraphrase-MiniLM-L3)
  - Filter by recency (last 30d) + user trust score

Why this is the right pattern (per agent synthesis 2026-04-30):
  - 4/5 adversarial attack classes are white-box-bypassable on weights
  - Frozen-weight retrieval sidesteps gradient attacks entirely
  - Compliance-friendly (user can delete)
  - Cheap (no continual training cost per user)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional


class RetrievalMemory:
    def __init__(
        self,
        cache_dir: str | Path = "/workspace/runs/user_memory",
        recency_days: int = 30,
        max_per_user: int = 1000,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.recency_days = recency_days
        self.max_per_user = max_per_user
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _user_path(self, user_hash: str) -> Path:
        return self.cache_dir / f"{user_hash}.jsonl"

    def add(self, user_hash: str, text: str, sample_id: int) -> None:
        path = self._user_path(user_hash)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "text": text,
                "sample_id": sample_id,
            }, ensure_ascii=False) + "\n")

    def query(self, user_hash: str, query_text: str, top_k: int = 4) -> List[dict]:
        """Naive recency-weighted lexical query.

        Real impl should use sentence embedding cosine. For now we do
        lexical overlap + recency.
        """
        path = self._user_path(user_hash)
        if not path.exists():
            return []

        cutoff_s = time.time() - self.recency_days * 86400
        memories = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    ts_s = time.mktime(time.strptime(e["ts"], "%Y-%m-%dT%H:%M:%S"))
                except (ValueError, KeyError):
                    continue
                if ts_s < cutoff_s:
                    continue
                memories.append((e, ts_s))

        if not memories:
            return []

        q_words = set(query_text.lower().split())
        scored = []
        for e, ts_s in memories:
            text = e["text"]
            t_words = set(text.lower().split())
            overlap = len(q_words & t_words) / max(len(q_words), 1)
            recency = max(0.0, 1.0 - (time.time() - ts_s) / (self.recency_days * 86400))
            score = 0.7 * overlap + 0.3 * recency
            scored.append((score, e))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [e for _, e in scored[:top_k]]

    def delete_user(self, user_hash: str) -> bool:
        path = self._user_path(user_hash)
        if path.exists():
            path.unlink()
            return True
        return False

    def stats(self) -> dict:
        out = {"users": 0, "total_memories": 0}
        for f in self.cache_dir.glob("*.jsonl"):
            out["users"] += 1
            out["total_memories"] += sum(1 for _ in open(f, "r", encoding="utf-8"))
        return out
