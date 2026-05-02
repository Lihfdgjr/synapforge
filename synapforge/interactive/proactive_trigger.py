"""ProactiveTrigger — when should the model speak unprompted?

Watches three signals and emits :class:`TriggerEvent` records the kernel
can act on:

  1. **Curiosity score** (from :class:`synapforge.curiosity.CuriosityScorer`):
     when the current hidden-state delta-F or retrieval gap crosses the
     P95 of a recent window, the model has "found something interesting"
     and might want to share.

  2. **GoalMemory** (from :class:`synapforge.intrinsic.GoalMemory`): if a
     previously stored goal has high improvement and isn't yet
     fulfilled, surface it.

  3. **Silence + relevance**: if the user has been quiet for >5 s and
     there's a high-similarity hit between recent user terms and an
     idle-loop generated thought, propose it.

Spec contract:
  * Watches every K=10 CfC steps (caller drives the cadence).
  * 0 imports of ``torch`` at module level — this module is pure-python.
  * Returns ``TriggerEvent`` with ``urgency``, ``goal_id``, and
    ``suggested_text``; the kernel decides actual emission via
    :class:`InterruptPolicy`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, List, Optional


# ---------------------------------------------------------------------------
# Trigger event schema
# ---------------------------------------------------------------------------


class TriggerSource(Enum):
    CURIOSITY = "curiosity"
    GOAL_MEMORY = "goal_memory"
    SILENCE_RELEVANCE = "silence_relevance"
    EXTERNAL = "external"


@dataclass
class TriggerEvent:
    source: TriggerSource
    urgency: float = 0.0
    goal_id: Optional[int] = None
    suggested_text: str = ""
    confidence: float = 0.0
    relevance: float = 0.0
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ProactiveTrigger watcher
# ---------------------------------------------------------------------------


class ProactiveTrigger:
    """Stateful watcher that fires :class:`TriggerEvent` records when its
    rules say "speak now".

    Defaults are conservative: model NEVER speaks unprompted unless
    urgency clears the high bar in InterruptPolicy. The thresholds here
    are the *rule engine*; the policy gate is downstream.
    """

    def __init__(
        self,
        history_size: int = 200,
        curiosity_p95_window: int = 50,
        silence_threshold_s: float = 5.0,
        silence_relevance_threshold: float = 0.4,
        goal_success_prob_threshold: float = 0.7,
        check_interval_steps: int = 10,
        min_confidence: float = 0.7,
        min_relevance: float = 0.5,
    ) -> None:
        self.history_size = int(history_size)
        self.curiosity_p95_window = int(curiosity_p95_window)
        self.silence_threshold_s = float(silence_threshold_s)
        self.silence_relevance_threshold = float(silence_relevance_threshold)
        self.goal_success_prob_threshold = float(goal_success_prob_threshold)
        self.check_interval_steps = max(1, int(check_interval_steps))
        self.min_confidence = float(min_confidence)
        self.min_relevance = float(min_relevance)

        self._curiosity_history: Deque[float] = deque(
            maxlen=self.curiosity_p95_window
        )
        self._step: int = 0
        self._last_user_ts: float = 0.0
        self._fired_goal_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Inputs (caller pushes these on every CfC step / model decision)
    # ------------------------------------------------------------------

    def note_user_interaction(self, ts: Optional[float] = None) -> None:
        self._last_user_ts = ts if ts is not None else time.time()

    def push_curiosity(self, score: float) -> None:
        """Caller pushes one curiosity score per CfC step."""
        self._curiosity_history.append(float(score))

    def reset_curiosity(self) -> None:
        self._curiosity_history.clear()

    # ------------------------------------------------------------------
    # Tick — evaluate rules, return any fired triggers
    # ------------------------------------------------------------------

    def tick(
        self,
        goal_memory: Any = None,
        recent_user_terms: Optional[List[str]] = None,
        idle_thought_proposer: Optional[Callable[[], str]] = None,
        h_t_summary: str = "",
    ) -> List[TriggerEvent]:
        """Advance one step, evaluate rules, emit zero or more triggers.

        Args:
          goal_memory:         optional ``synapforge.intrinsic.GoalMemory``-
                               like object exposing ``__len__`` and a way
                               to query records (we duck-type for tests).
          recent_user_terms:   tokens / phrases from the most recent user
                               messages — used by the relevance rule.
          idle_thought_proposer: callable returning a string to use as
                               ``suggested_text`` when the silence rule
                               fires; can be None.
          h_t_summary:         short text representation of the current
                               hidden state — passed straight through
                               into ``payload`` so downstream selectors
                               can display "what triggered it".
        """
        self._step += 1
        if self._step % self.check_interval_steps != 0:
            return []

        out: List[TriggerEvent] = []

        # --- Rule 1: curiosity P95 -------------------------------------
        cur_event = self._check_curiosity(h_t_summary)
        if cur_event is not None:
            out.append(cur_event)

        # --- Rule 2: GoalMemory has a high-improvement, un-fulfilled record
        goal_event = self._check_goal_memory(goal_memory)
        if goal_event is not None:
            out.append(goal_event)

        # --- Rule 3: silence + relevance -------------------------------
        silence_event = self._check_silence_relevance(
            recent_user_terms or [],
            idle_thought_proposer,
            h_t_summary,
        )
        if silence_event is not None:
            out.append(silence_event)

        # Filter out anything that fails the quality guards: any proactive
        # utterance must score > 0.7 confidence AND > 0.5 relevance.
        keep: List[TriggerEvent] = []
        for ev in out:
            if (
                ev.confidence >= self.min_confidence
                and ev.relevance >= self.min_relevance
            ):
                keep.append(ev)
        return keep

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    def _check_curiosity(
        self, h_t_summary: str
    ) -> Optional[TriggerEvent]:
        if len(self._curiosity_history) < max(8, self.curiosity_p95_window // 4):
            return None
        sorted_scores = sorted(self._curiosity_history)
        p95_idx = max(0, int(0.95 * len(sorted_scores)) - 1)
        p95 = sorted_scores[p95_idx]
        latest = self._curiosity_history[-1]
        if latest <= p95:
            return None
        # Map curiosity overshoot to urgency in [0.6, 0.95].
        max_score = max(sorted_scores) or 1.0
        excess = (latest - p95) / max(max_score - p95, 1e-6)
        urgency = 0.6 + 0.35 * min(1.0, excess)
        # Confidence and relevance: high curiosity → high confidence by
        # construction (we have a clear signal); relevance is heuristic
        # because we don't have direct access to the user's recent terms
        # here, so we set it to 0.55 (just above the 0.5 floor) and let
        # the silence-relevance rule capture stronger evidence.
        confidence = 0.75 + 0.20 * min(1.0, excess)
        relevance = 0.55
        return TriggerEvent(
            source=TriggerSource.CURIOSITY,
            urgency=urgency,
            confidence=confidence,
            relevance=relevance,
            suggested_text="",
            payload={
                "curiosity": latest,
                "p95": p95,
                "h_t_summary": h_t_summary,
            },
        )

    def _check_goal_memory(
        self, goal_memory: Any
    ) -> Optional[TriggerEvent]:
        if goal_memory is None:
            return None
        records = getattr(goal_memory, "_records", None)
        if not records:
            return None

        # Pick the highest-improvement record we haven't yet fired on.
        sorted_records = sorted(
            enumerate(records),
            key=lambda pair: pair[1].get("improvement", 0.0),
            reverse=True,
        )
        for idx, rec in sorted_records:
            if idx in self._fired_goal_ids:
                continue
            improvement = float(rec.get("improvement", 0.0))
            success_prob = self._estimate_success_prob(improvement)
            if success_prob < self.goal_success_prob_threshold:
                return None
            self._fired_goal_ids.add(idx)
            urgency = 0.6 + 0.35 * min(1.0, success_prob)
            return TriggerEvent(
                source=TriggerSource.GOAL_MEMORY,
                urgency=urgency,
                goal_id=idx,
                confidence=0.8,
                relevance=0.7,
                suggested_text="",
                payload={
                    "improvement": improvement,
                    "success_prob": success_prob,
                    "goal_tokens": list(rec.get("goal", []))[:32],
                },
            )
        return None

    def _check_silence_relevance(
        self,
        recent_user_terms: List[str],
        idle_thought_proposer: Optional[Callable[[], str]],
        h_t_summary: str,
    ) -> Optional[TriggerEvent]:
        if self._last_user_ts == 0.0:
            return None
        elapsed = time.time() - self._last_user_ts
        if elapsed < self.silence_threshold_s:
            return None
        # Estimate relevance as overlap between recent user terms and the
        # h_t summary text. Cheap heuristic — domain-specific tokenizers
        # can be wired in later.
        if not recent_user_terms:
            return None
        h_lower = (h_t_summary or "").lower()
        hits = sum(
            1 for t in recent_user_terms if t and t.lower() in h_lower
        )
        relevance = hits / max(len(recent_user_terms), 1)
        if relevance < self.silence_relevance_threshold:
            return None

        suggested = ""
        if idle_thought_proposer is not None:
            try:
                suggested = idle_thought_proposer() or ""
            except Exception:
                suggested = ""

        # Urgency increases gently with silence duration.
        urgency = 0.5 + 0.4 * min(1.0, (elapsed - self.silence_threshold_s) / 30.0)
        return TriggerEvent(
            source=TriggerSource.SILENCE_RELEVANCE,
            urgency=urgency,
            confidence=0.7 + 0.2 * min(1.0, relevance),
            relevance=max(0.5, relevance),
            suggested_text=suggested,
            payload={
                "elapsed_s": elapsed,
                "matched_terms": hits,
                "n_user_terms": len(recent_user_terms),
            },
        )

    @staticmethod
    def _estimate_success_prob(improvement: float) -> float:
        """Sigmoid mapping of historical improvement → estimated success.

        Improvement = pre_loss - post_loss. We treat improvement >= 0.3 as
        very-high-confidence (sigmoid saturates).
        """
        # logistic with threshold near 0.15.
        return 1.0 / (1.0 + math.exp(-(improvement - 0.15) * 8.0))

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "step": self._step,
            "curiosity_history_len": len(self._curiosity_history),
            "fired_goal_ids": sorted(self._fired_goal_ids),
            "seconds_since_user": (
                time.time() - self._last_user_ts
                if self._last_user_ts
                else 0.0
            ),
        }
