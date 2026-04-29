"""
ProvenanceTracker: every sample entering the replay buffer carries:
  - source_id     : 'web:bilibili.com', 'web:arxiv.org', 'chat:user_<hash>'
  - timestamp     : ISO 8601
  - content_hash  : sha256(content)
  - sample_id     : monotonic int, unique within tracker
  - url           : original URL if web
  - user_handle   : user identifier if chat (hashed for PII)
  - parent_id     : if this sample was derived from another (e.g., translation)
  - trust_score   : float [0, 1], starts at 0.5, evolves with usage
  - reject_count  : how many later checks flagged content from this source

Persistence: append-only JSONL at /workspace/runs/provenance.jsonl
Bloom-filter blocklist: /workspace/runs/blocked_hashes.bin (sha256 of bad samples)

Why per-source trust:
  - One bilibili video that turned out to be SEO spam shouldn't taint all bilibili
  - But repeated bad samples from one user_handle should escalate that user's threshold
  - Trust is the bridge between "novel" (low cosine to existing) and "trustworthy"
    (good track record), letting WebPoisonGate accept high-novelty content from
    high-trust sources without blocking learning.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional


@dataclass
class ProvenanceEntry:
    sample_id: int
    source_id: str
    timestamp: str
    content_hash: str
    url: Optional[str] = None
    user_handle_hash: Optional[str] = None
    parent_id: Optional[int] = None
    trust_score: float = 0.5
    reject_count: int = 0


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_user(handle: str, salt: str = "synapforge_v42") -> str:
    h = hashlib.sha256(f"{salt}:{handle}".encode("utf-8")).hexdigest()
    return h[:12]


class ProvenanceTracker:
    def __init__(
        self,
        log_path: str | Path = "/workspace/runs/provenance.jsonl",
        blocklist_path: str | Path = "/workspace/runs/blocked_hashes.txt",
    ) -> None:
        self.log_path = Path(log_path)
        self.blocklist_path = Path(blocklist_path)
        self._next_id = 0
        self._source_trust: Dict[str, float] = defaultdict(lambda: 0.5)
        self._source_count: Dict[str, int] = defaultdict(int)
        self._source_rejects: Dict[str, int] = defaultdict(int)
        self._blocked_hashes: set[str] = set()

        self._load()

    def _load(self) -> None:
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid = e.get("source_id", "")
                    self._source_count[sid] += 1
                    if e.get("reject_count", 0) > 0:
                        self._source_rejects[sid] += 1
                    self._next_id = max(self._next_id, e.get("sample_id", 0) + 1)
            self._recompute_trust()

        if self.blocklist_path.exists():
            with open(self.blocklist_path, "r", encoding="utf-8") as f:
                for line in f:
                    h = line.strip()
                    if h:
                        self._blocked_hashes.add(h)

    def _recompute_trust(self) -> None:
        for sid, total in self._source_count.items():
            rej = self._source_rejects[sid]
            base = 0.5 + 0.4 * (1.0 - rej / max(total, 1))
            base += min(0.1, 0.001 * total)
            self._source_trust[sid] = max(0.05, min(0.99, base))

    def is_blocked(self, content: str) -> bool:
        return hash_content(content) in self._blocked_hashes

    def block(self, content: str) -> None:
        h = hash_content(content)
        self._blocked_hashes.add(h)
        self.blocklist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.blocklist_path, "a", encoding="utf-8") as f:
            f.write(h + "\n")

    def trust_of(self, source_id: str) -> float:
        return self._source_trust.get(source_id, 0.5)

    def admit(
        self,
        content: str,
        source_id: str,
        url: Optional[str] = None,
        user_handle: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> ProvenanceEntry:
        entry = ProvenanceEntry(
            sample_id=self._next_id,
            source_id=source_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            content_hash=hash_content(content),
            url=url,
            user_handle_hash=hash_user(user_handle) if user_handle else None,
            parent_id=parent_id,
            trust_score=self.trust_of(source_id),
        )
        self._next_id += 1
        self._source_count[source_id] += 1

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def report_reject(self, sample_id: int, source_id: str, content: Optional[str] = None) -> None:
        self._source_rejects[source_id] += 1
        self._recompute_trust()
        if content is not None:
            self.block(content)

    def stats(self) -> dict:
        return {
            "total_admitted": self._next_id,
            "total_blocked": len(self._blocked_hashes),
            "sources": {
                sid: {
                    "count": self._source_count[sid],
                    "rejects": self._source_rejects[sid],
                    "trust": round(self._source_trust[sid], 3),
                }
                for sid in self._source_count
            },
        }

    def iter_recent(self, n: int = 100) -> Iterator[ProvenanceEntry]:
        if not self.log_path.exists():
            return
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-n:]:
            try:
                yield ProvenanceEntry(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
