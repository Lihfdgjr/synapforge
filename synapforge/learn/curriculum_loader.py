"""sf.learn.curriculum_loader — unified loader for 50- or 500-task curricula.

Reads either:
  * the legacy 50-task ``web_curriculum.CATALOGUE`` (Python tuples), or
  * the 500-task ``web_curriculum_500.json`` (produced by
    ``scripts/web_curriculum_extended.py``).

Provides:
  * ``CurriculumLoader.load(path: str | None)`` -- auto-detect format.
  * Stratified sampling by level.
  * Vygotsky zone-of-proximal-development scheduler: pick the task whose
    predicted success probability is closest to the centre of [LO, HI]
    (default 0.3..0.7), where the prediction is a tiny linear regressor on
    a hidden-state vector + task feature vector.

The "difficulty estimator" is intentionally tiny + dependency-free so it can
run inside the trainer hot loop on CPU.

Backward compatibility: on `path=None` we first try the JSON file at
``synapforge/learn/web_curriculum_500.json``. If absent we fall back to
importing ``synapforge.learn.web_curriculum.CATALOGUE`` (50 tasks).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


@dataclass
class CurriculumTask:
    level: int
    instruction_zh: str
    instruction_en: str
    start_url: str
    success_text_regex: Optional[str]
    success_url_substr: Optional[str]
    max_steps: int
    expected_skills: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        host = self.start_url.split("//", 1)[-1].split("/", 1)[0]
        return f"L{self.level}/{host}"


# --------------------------------------------------------------------- loader


def _legacy_to_unified(catalogue) -> List[CurriculumTask]:
    """Convert legacy WebTask tuples (50-task version) to CurriculumTask list."""
    out: List[CurriculumTask] = []
    for t in catalogue:
        # legacy WebTask has: level, url, instruction_zh, instruction_en,
        # success_text_regex, success_url_substr, max_steps
        out.append(CurriculumTask(
            level=int(t.level),
            instruction_zh=t.instruction_zh,
            instruction_en=t.instruction_en,
            start_url=t.url,
            success_text_regex=t.success_text_regex,
            success_url_substr=t.success_url_substr,
            max_steps=int(t.max_steps),
            expected_skills=[],
        ))
    return out


class CurriculumLoader:
    """Auto-detecting loader for both 50-task and 500-task curricula."""

    DEFAULT_JSON_NAME = "web_curriculum_500.json"

    def __init__(self, tasks: Sequence[CurriculumTask]) -> None:
        if not tasks:
            raise ValueError("empty curriculum")
        self._tasks: List[CurriculumTask] = list(tasks)
        self._by_level: dict[int, List[CurriculumTask]] = {}
        for t in self._tasks:
            self._by_level.setdefault(t.level, []).append(t)
        self._levels = sorted(self._by_level)
        self._max_level = max(self._levels)

    # -------------------------------------------------------------- load
    @classmethod
    def load(cls, path: Optional[str] = None) -> "CurriculumLoader":
        """Auto-detect: if path ends with .json, parse JSON. Otherwise import
        legacy module. If path is None, prefer the 500-task JSON sitting next
        to this file; fall back to legacy CATALOGUE.
        """
        if path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(here, cls.DEFAULT_JSON_NAME)
            if os.path.exists(cand):
                path = cand
        if path and path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            tasks = [CurriculumTask(**t) for t in payload["tasks"]]
            return cls(tasks)
        # Fallback: import legacy module.
        try:
            from synapforge.learn.web_curriculum import CATALOGUE as legacy_cat
        except Exception as exc:
            raise FileNotFoundError(
                f"no JSON at {path!r} and legacy import failed: {exc}"
            ) from exc
        return cls(_legacy_to_unified(legacy_cat))

    # ---------------------------------------------------------- properties
    @property
    def n_tasks(self) -> int:
        return len(self._tasks)

    @property
    def levels(self) -> List[int]:
        return list(self._levels)

    @property
    def max_level(self) -> int:
        return self._max_level

    def by_level(self, level: int) -> List[CurriculumTask]:
        return list(self._by_level.get(level, ()))

    def all_tasks(self) -> List[CurriculumTask]:
        return list(self._tasks)

    # ---------------------------------------------------- stratified sample
    def stratified(self, k: int, rng: Optional[random.Random] = None) -> List[CurriculumTask]:
        """Sample ``k`` tasks evenly across levels (k may be < n_levels)."""
        rng = rng or random.Random()
        per = max(1, k // max(1, len(self._levels)))
        out: List[CurriculumTask] = []
        for L in self._levels:
            pool = self._by_level[L]
            n = min(per, len(pool))
            out.extend(rng.sample(pool, n))
        # Fill remaining quota uniformly at random across all tasks.
        remain = max(0, k - len(out))
        if remain:
            out.extend(rng.sample(self._tasks, min(remain, len(self._tasks))))
        rng.shuffle(out)
        return out[:k]

    # ----------------------------------------------- difficulty estimator
    def _task_feat(self, t: CurriculumTask) -> List[float]:
        """Tiny hand-coded feature vector per task. 6 floats."""
        # Stable hash for steady ordering across runs.
        h = int(hashlib.sha256(
            f"{t.level}|{t.start_url}|{t.instruction_en}".encode("utf-8")
        ).hexdigest(), 16)
        rand_bit = ((h >> 13) & 0xFFFF) / 0xFFFF
        return [
            t.level / max(1, self._max_level),         # 0..1 difficulty floor
            min(1.0, t.max_steps / 64.0),              # step budget
            min(1.0, len(t.instruction_en) / 256.0),   # instruction length
            1.0 if t.success_text_regex else 0.0,      # has regex check
            1.0 if t.success_url_substr else 0.0,      # has url check
            rand_bit,                                  # variance term
        ]

    def predict_success_prob(
        self,
        hidden: Optional[Sequence[float]],
        task: CurriculumTask,
    ) -> float:
        """Predict P(success | hidden, task).

        Uses a simple sigmoid of (hidden·w + bias_from_task), where bias is
        derived from task features via a fixed linear projection. This is
        intentionally cheap; in production the trainer keeps a small MLP and
        replaces this method.
        """
        feat = self._task_feat(task)
        # bias: high level + many steps -> harder -> lower prob.
        # Coefficients chosen so that P drops roughly 0.07 per level.
        bias = 0.95
        bias -= 0.07 * feat[0] * self._max_level   # level penalty
        bias -= 0.05 * feat[1]                     # step budget penalty
        bias -= 0.03 * feat[2]                     # instruction-length penalty
        bias += 0.02 * feat[3]                     # has regex (clearer signal)
        bias += 0.02 * feat[4]                     # has url
        if hidden is not None and len(hidden) > 0:
            n = min(len(hidden), 16)
            mean = sum(float(x) for x in hidden[:n]) / n
            # Hidden mean acts as a "model strength" coefficient: positive
            # mean nudges P up. We squash mean to [-0.4, +0.4].
            mean = max(-1.0, min(1.0, mean)) * 0.4
            bias += mean
        # Sigmoid.
        return 1.0 / (1.0 + math.exp(-bias * 4.0 + 2.0))

    # ----------------------------------------------- Vygotsky scheduler
    def schedule(
        self,
        hidden: Optional[Sequence[float]] = None,
        lo: float = 0.3,
        hi: float = 0.7,
        rng: Optional[random.Random] = None,
        n_candidates: int = 32,
    ) -> CurriculumTask:
        """Pick a task with predicted success P in [lo, hi].

        Vygotsky's "zone of proximal development" — too easy = no learning,
        too hard = collapse. We sample a candidate set, score each, and pick
        the one whose P is closest to the centre of [lo, hi]; if none lie in
        the band we still return the closest match (no exception).
        """
        rng = rng or random.Random()
        if not self._tasks:
            raise RuntimeError("empty curriculum")
        cands = rng.sample(self._tasks, min(n_candidates, len(self._tasks)))
        target = 0.5 * (lo + hi)
        best: Optional[CurriculumTask] = None
        best_score: tuple = (2, float("inf"))  # (band_priority, distance)
        for t in cands:
            p = self.predict_success_prob(hidden, t)
            in_band = lo <= p <= hi
            band = 0 if in_band else 1
            # When in band: rank by closeness to centre.
            # When out of band: rank by min distance to band edge.
            if in_band:
                dist = abs(p - target)
            else:
                dist = min(abs(p - lo), abs(p - hi))
            score = (band, dist)
            if score < best_score:
                best = t
                best_score = score
        assert best is not None
        return best


__all__ = ["CurriculumTask", "CurriculumLoader"]
