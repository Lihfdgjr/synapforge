"""
Web/Chat Poison Gates: glue between PoisonDetector + ProvenanceTracker +
SourceBudget + (optional) TRAKGate.

Two entry points:

  WebPoisonGate.admit(text, source_id, url) -> GateDecision
    Used by autonomous_daemon. Heavy gates (TRAK if enabled).
    On accept -> writes to web_cache.jsonl with full provenance.

  ChatPoisonGate.admit(text, user_handle, history) -> GateDecision
    Used by production chat. NO weight updates by default — accepted
    samples go to retrieval cache (Track B). Set update_weights=True
    only if you accept the risk per feedback_continual_vs_poison_balance.md.

The 7 ingest gates from agent 3 synthesis:
  G1 hash blocklist        -> ProvenanceTracker.is_blocked
  G2 lang/format           -> _check_lang_format
  G3 SemDeDup              -> dedup_signal in PoisonDetector
  G4 tox/PII/inj           -> PromptGuard signal in PoisonDetector
  G5 source budget         -> SourceBudget.allow
  G6 TracIn/TRAK influence -> TRAKGate.score_sample (web only, async)
  G7 Shadow batch          -> trainer-side, every 32 accepted samples
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

from .poison_detector import PoisonDetector, PoisonScore
from .provenance import ProvenanceEntry, ProvenanceTracker


@dataclass
class GateDecision:
    accept: bool
    low_lr: bool = False
    score: Optional[PoisonScore] = None
    provenance: Optional[ProvenanceEntry] = None
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accept": self.accept,
            "low_lr": self.low_lr,
            "score": self.score.to_dict() if self.score else None,
            "provenance": asdict(self.provenance) if self.provenance else None,
            "reasons": self.reasons,
        }


class SourceBudget:
    """Token-bucket per source + 7d rolling cap (Anthropic 2510.07192 anchor).

    Per-source 7d cap = 125 = 50% of poison fixed-count threshold (250 docs).
    No single source can ever reach the poison level.
    """

    def __init__(
        self,
        per_hour: int = 50,
        per_day: int = 1000,
        per_7d_per_source: int = 125,
    ) -> None:
        self.per_hour = per_hour
        self.per_day = per_day
        self.per_7d_per_source = per_7d_per_source
        self._global_hour_count: Deque[float] = deque()
        self._global_day_count: Deque[float] = deque()
        self._source_7d: Dict[str, Deque[float]] = defaultdict(deque)

    def _trim(self, q: Deque[float], window_s: float) -> None:
        cutoff = time.time() - window_s
        while q and q[0] < cutoff:
            q.popleft()

    def allow(self, source_id: str) -> tuple[bool, str]:
        now = time.time()
        self._trim(self._global_hour_count, 3600)
        self._trim(self._global_day_count, 86400)
        self._trim(self._source_7d[source_id], 7 * 86400)

        if len(self._global_hour_count) >= self.per_hour:
            return False, f"hour cap reached ({self.per_hour})"
        if len(self._global_day_count) >= self.per_day:
            return False, f"day cap reached ({self.per_day})"
        if len(self._source_7d[source_id]) >= self.per_7d_per_source:
            return False, (
                f"source 7d cap reached ({self.per_7d_per_source}) — "
                f"50% of Anthropic poison threshold"
            )

        self._global_hour_count.append(now)
        self._global_day_count.append(now)
        self._source_7d[source_id].append(now)
        return True, ""

    def stats(self) -> dict:
        return {
            "global_last_hour": len(self._global_hour_count),
            "global_last_day": len(self._global_day_count),
            "sources_7d": {
                sid: len(q) for sid, q in self._source_7d.items() if len(q)
            },
        }


def _check_lang_format(text: str, min_tok: int = 16, max_tok: int = 8192) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty text"
    n = len(text.split())
    if n < min_tok:
        return False, f"too short ({n} tokens, min {min_tok})"
    if n > max_tok:
        return False, f"too long ({n} tokens, max {max_tok})"
    if all(ord(c) < 128 for c in text) and any(c.isalpha() for c in text):
        return True, ""
    if any('一' <= c <= '鿿' for c in text):
        return True, ""
    return True, ""


class WebPoisonGate:
    """Gate for autonomous_daemon (web ingest)."""

    def __init__(
        self,
        detector: PoisonDetector,
        provenance: ProvenanceTracker,
        budget: SourceBudget,
        out_jsonl: str | Path = "/workspace/data/web_cache.jsonl",
        rejection_log: str | Path = "/workspace/runs/rejection.log",
    ) -> None:
        self.detector = detector
        self.provenance = provenance
        self.budget = budget
        self.out_jsonl = Path(out_jsonl)
        self.rejection_log = Path(rejection_log)

    def admit(
        self,
        text: str,
        source_id: str,
        url: Optional[str] = None,
        topic: Optional[str] = None,
        question: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> GateDecision:
        if self.provenance.is_blocked(text):
            return self._reject("G1 hash blocked", source_id, text)

        ok, reason = _check_lang_format(text)
        if not ok:
            return self._reject(f"G2 {reason}", source_id, text)

        ok, reason = self.budget.allow(source_id)
        if not ok:
            return self._reject(f"G5 {reason}", source_id, text)

        sample = {"text": text, "source_id": source_id, "url": url}
        score = self.detector.score(sample)

        if not score.accept:
            return self._reject_with_score(score, source_id, text)

        prov = self.provenance.admit(text, source_id, url=url)

        record = {
            "q": question or topic or "",
            "a": text,
            "topic": topic or "web",
            "source_id": source_id,
            "url": url,
            "sample_id": prov.sample_id,
            "trust": prov.trust_score,
            "low_lr": score.low_lr,
            "ts": prov.timestamp,
        }
        if extra:
            record.update(extra)

        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return GateDecision(
            accept=True,
            low_lr=score.low_lr,
            score=score,
            provenance=prov,
            reasons=score.reasons,
        )

    def _reject(self, reason: str, source_id: str, text: str) -> GateDecision:
        self._log_reject(reason, source_id, text)
        return GateDecision(accept=False, reasons=[reason])

    def _reject_with_score(self, score: PoisonScore, source_id: str, text: str) -> GateDecision:
        for r in score.reasons:
            self._log_reject(f"detector: {r}", source_id, text)
        self.provenance.report_reject(0, source_id, text if score.combined > 0.7 else None)
        return GateDecision(accept=False, score=score, reasons=score.reasons)

    def _log_reject(self, reason: str, source_id: str, text: str) -> None:
        self.rejection_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rejection_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source_id": source_id,
                "reason": reason,
                "text_preview": text[:120],
            }, ensure_ascii=False) + "\n")


class ChatPoisonGate:
    """Gate for production chat (Track B: retrieval, not weights).

    Default behavior: accepted samples go to per-user retrieval cache.
    Set update_weights=True to also feed replay buffer (NOT recommended;
    industry consensus per Claude Memory / ChatGPT Memory).
    """

    def __init__(
        self,
        detector: PoisonDetector,
        provenance: ProvenanceTracker,
        budget: SourceBudget,
        retrieval_cache_dir: str | Path = "/workspace/runs/user_memory",
        update_weights: bool = False,
    ) -> None:
        self.detector = detector
        self.provenance = provenance
        self.budget = budget
        self.retrieval_cache_dir = Path(retrieval_cache_dir)
        self.update_weights = update_weights

    def admit(
        self,
        text: str,
        user_handle: str,
        history: Optional[List[str]] = None,
    ) -> GateDecision:
        source_id = f"chat:{user_handle}"

        if self.provenance.is_blocked(text):
            return GateDecision(accept=False, reasons=["G1 hash blocked"])

        ok, reason = _check_lang_format(text, min_tok=4)
        if not ok:
            return GateDecision(accept=False, reasons=[f"G2 {reason}"])

        ok, reason = self.budget.allow(source_id)
        if not ok:
            return GateDecision(accept=False, reasons=[f"G5 {reason}"])

        sample = {
            "text": text,
            "user_handle": user_handle,
            "history": history or [],
        }
        score = self.detector.score(sample)

        if not score.accept:
            self.provenance.report_reject(0, source_id, text if score.combined > 0.7 else None)
            return GateDecision(accept=False, score=score, reasons=score.reasons)

        prov = self.provenance.admit(text, source_id, user_handle=user_handle)

        self.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
        user_file = self.retrieval_cache_dir / f"{prov.user_handle_hash}.jsonl"
        with open(user_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": prov.timestamp,
                "text": text,
                "sample_id": prov.sample_id,
                "low_lr": score.low_lr,
            }, ensure_ascii=False) + "\n")

        return GateDecision(
            accept=True,
            low_lr=score.low_lr,
            score=score,
            provenance=prov,
            reasons=score.reasons + ["routed to retrieval (Track B), not weights"],
        )
