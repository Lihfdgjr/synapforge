"""Frontier sampler for self-drive goal selection.

Picks a goal from ``GoalMemory`` whose past success_rate is in a
"sweet-spot" band [lo, hi] -- not so easy the model has nothing to learn,
not so hard the model has no signal. Pure-Python (no torch import).

When no goal in the band exists (or the buffer is empty), the caller is
expected to invoke ``SelfGoalProposer`` to invent a fresh one. The
sampler also enforces a recent-K lockout to avoid local mode collapse
on whatever goal happens to land in-band first.

Reference (Voyager 2305.16291 Sec 4.2 "Automatic Curriculum"): the
agent is more likely to acquire a new skill if the proposed task is
neither already-mastered nor totally unsolvable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class GoalRecord:
    """One entry in the goal-attempt buffer."""

    goal_id: int
    goal_tokens: tuple[int, ...]
    n_attempts: int = 0
    n_successes: int = 0

    @property
    def success_rate(self) -> float:
        if self.n_attempts <= 0:
            return 0.0
        return self.n_successes / self.n_attempts


@dataclass
class FrontierSampler:
    """Pick goals whose success_rate is in [sweet_lo, sweet_hi].

    Lockout window of size ``recent_k`` rejects goals seen in the last
    K calls to ``sample()``.

    Usage:
        s = FrontierSampler()
        s.observe(GoalRecord(0, (1,2,3), 10, 5))     # rate 0.5  IN
        s.observe(GoalRecord(1, (4,5,6), 10, 9))     # rate 0.9  out
        s.observe(GoalRecord(2, (7,8,9), 10, 1))     # rate 0.1  out
        rec = s.sample()                              # returns goal 0
        rec2 = s.sample()                             # None (locked out)
    """

    sweet_lo: float = 0.3
    sweet_hi: float = 0.7
    recent_k: int = 10
    min_attempts: int = 3
    _records: dict[int, GoalRecord] = field(default_factory=dict)
    _recent: deque[int] = field(default_factory=lambda: deque(maxlen=10))

    def __post_init__(self) -> None:
        # The default_factory above bakes maxlen=10; if the user changed
        # recent_k, rebuild the deque to honour it.
        if self._recent.maxlen != self.recent_k:
            self._recent = deque(self._recent, maxlen=int(self.recent_k))

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def observe(self, rec: GoalRecord) -> None:
        """Insert or update a goal record by goal_id."""
        existing = self._records.get(rec.goal_id)
        if existing is None:
            self._records[rec.goal_id] = rec
        else:
            existing.n_attempts = rec.n_attempts
            existing.n_successes = rec.n_successes

    def observe_attempt(self, goal_id: int, goal_tokens: Iterable[int],
                        success: bool) -> GoalRecord:
        """Convenience: append one attempt outcome."""
        rec = self._records.get(goal_id)
        if rec is None:
            rec = GoalRecord(goal_id=int(goal_id),
                             goal_tokens=tuple(int(t) for t in goal_tokens))
            self._records[goal_id] = rec
        rec.n_attempts += 1
        if success:
            rec.n_successes += 1
        return rec

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def in_band(self, rec: GoalRecord) -> bool:
        if rec.n_attempts < self.min_attempts:
            return False
        return self.sweet_lo <= rec.success_rate <= self.sweet_hi

    def candidates(self) -> list[GoalRecord]:
        return [r for r in self._records.values() if self.in_band(r)]

    def sample(self) -> Optional[GoalRecord]:
        """Return one in-band goal not in recent-K lockout, else None.

        Tie-breaking: pick the record whose success_rate is closest to
        the band midpoint -- this keeps the sampler away from drifting
        toward an easy edge once a few rounds collect there.
        """
        cands = self.candidates()
        if not cands:
            return None
        cands = [r for r in cands if r.goal_id not in self._recent]
        if not cands:
            return None
        mid = 0.5 * (self.sweet_lo + self.sweet_hi)
        cands.sort(key=lambda r: abs(r.success_rate - mid))
        chosen = cands[0]
        self._recent.append(chosen.goal_id)
        return chosen

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        n_total = len(self._records)
        n_in_band = sum(1 for r in self._records.values() if self.in_band(r))
        rates = [r.success_rate for r in self._records.values()
                 if r.n_attempts >= self.min_attempts]
        return {
            "n_goals": n_total,
            "n_in_band": n_in_band,
            "n_recent_locked": len(self._recent),
            "mean_success_rate": (sum(rates) / len(rates)) if rates else 0.0,
            "band_lo": self.sweet_lo,
            "band_hi": self.sweet_hi,
        }
