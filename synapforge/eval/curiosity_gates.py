"""
5 eval gates for "genuinely curious" vs "scaffolding only".

Per agent synthesis 2026-05-01, the model passes the curiosity test only
if it clears ALL 5 gates over a 72h autonomous run:

  G1. Topic Shannon entropy > 4.5 bits over 72h pursuit log
      (vs ~2.0 for current min-coverage daemon, >50 distinct topics)
  G2. Generated question novelty: 1 - max BLEU-4 vs FineWeb-Edu corpus > 0.7
  G3. MMLU 5 weakest subjects gap-close +3pp avg after 72h (KILLER TEST)
      — proves curiosity targets *known* gaps, not random walk
  G4. Noisy-TV resistance: inject /dev/urandom topic; curiosity score
      decays <30% in 100 steps. RND/ICM fail, STDP-delta should pass.
  G5. Sustained engagement: median consecutive autonomous turns without
      external prompt > 50.

Pass all 5 = "genuinely curious." Pass <3 = "scaffolding only."
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class GateResult:
    name: str
    passed: bool
    score: float
    threshold: float
    details: Dict = field(default_factory=dict)


def gate_g1_topic_entropy(
    pursuit_log: List[str],
    threshold: float = 4.5,
) -> GateResult:
    """G1: Shannon entropy of topic pursuit distribution.

    pursuit_log: list of topic strings, one per cycle of autonomous learning.
    """
    if not pursuit_log:
        return GateResult("G1_topic_entropy", False, 0.0, threshold,
                          {"reason": "empty pursuit log"})

    counts = Counter(pursuit_log)
    n = sum(counts.values())
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    return GateResult(
        "G1_topic_entropy",
        passed=entropy >= threshold,
        score=entropy,
        threshold=threshold,
        details={
            "n_pursuits": n,
            "n_distinct_topics": len(counts),
            "top_5": counts.most_common(5),
        },
    )


def _ngrams(text: str, n: int = 4) -> List[str]:
    tokens = text.lower().split()
    return [" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def _bleu4_self(query: str, corpus_lines: List[str]) -> float:
    """Simplified self-BLEU-4: max 4-gram overlap with any corpus line.

    Returns max BLEU-4 score against corpus_lines ∈ [0, 1]. Higher = less novel.
    """
    q_ngrams = set(_ngrams(query, 4))
    if not q_ngrams:
        return 0.0

    max_overlap = 0.0
    for line in corpus_lines:
        line_ngrams = set(_ngrams(line, 4))
        if not line_ngrams:
            continue
        overlap = len(q_ngrams & line_ngrams) / max(len(q_ngrams), 1)
        max_overlap = max(max_overlap, overlap)
    return max_overlap


def gate_g2_question_novelty(
    generated_questions: List[str],
    reference_corpus_lines: List[str],
    threshold: float = 0.7,
) -> GateResult:
    """G2: Generated questions should be novel vs reference corpus.

    Novelty = 1 - max_BLEU4_overlap. Threshold 0.7 = at most 30% 4-gram overlap.
    """
    if not generated_questions:
        return GateResult("G2_question_novelty", False, 0.0, threshold,
                          {"reason": "no questions"})

    novelties = []
    for q in generated_questions:
        max_overlap = _bleu4_self(q, reference_corpus_lines[:1000])
        novelties.append(1.0 - max_overlap)

    avg_novelty = sum(novelties) / len(novelties)
    return GateResult(
        "G2_question_novelty",
        passed=avg_novelty >= threshold,
        score=avg_novelty,
        threshold=threshold,
        details={
            "n_questions": len(generated_questions),
            "min_novelty": min(novelties),
            "max_novelty": max(novelties),
        },
    )


def gate_g3_mmlu_gap_close(
    mmlu_t0: Dict[str, float],
    mmlu_t72h: Dict[str, float],
    threshold_pp: float = 3.0,
    n_weakest: int = 5,
) -> GateResult:
    """G3 KILLER TEST: 5 weakest subjects at t=0 must improve avg ≥ +3pp at 72h.

    Proves curiosity targets known gaps, not random walk.
    """
    if not mmlu_t0 or not mmlu_t72h:
        return GateResult("G3_mmlu_gap_close", False, 0.0, threshold_pp,
                          {"reason": "missing mmlu scores"})

    sorted_t0 = sorted(mmlu_t0.items(), key=lambda x: x[1])
    weakest = [k for k, _ in sorted_t0[:n_weakest]]

    deltas = []
    per_subject = {}
    for subj in weakest:
        if subj not in mmlu_t72h:
            continue
        delta_pp = (mmlu_t72h[subj] - mmlu_t0[subj]) * 100
        deltas.append(delta_pp)
        per_subject[subj] = {"t0": mmlu_t0[subj], "t72h": mmlu_t72h[subj], "delta_pp": delta_pp}

    avg_delta = sum(deltas) / max(len(deltas), 1)
    return GateResult(
        "G3_mmlu_gap_close",
        passed=avg_delta >= threshold_pp,
        score=avg_delta,
        threshold=threshold_pp,
        details={
            "n_weakest": n_weakest,
            "weakest_subjects": weakest,
            "per_subject": per_subject,
        },
    )


def gate_g4_noisy_tv_resistance(
    curiosity_scores_over_steps: List[float],
    threshold_max_decay_pct: float = 30.0,
    n_steps_window: int = 100,
) -> GateResult:
    """G4: When fed /dev/urandom topic, curiosity score must decay <30% within
    100 steps. Tests STDP-delta noise immunity.

    curiosity_scores_over_steps: list of curiosity scores observed while
    feeding random noise. Decay = how much score drops over the window.

    Note: PASS for STDP means score DROPS rapidly (model loses interest in noise).
    FAIL means score stays elevated (got fooled by noise = noisy-TV).
    """
    if len(curiosity_scores_over_steps) < n_steps_window:
        return GateResult("G4_noisy_tv_resistance", False, 0.0, threshold_max_decay_pct,
                          {"reason": "insufficient samples"})

    initial = curiosity_scores_over_steps[0]
    final = curiosity_scores_over_steps[n_steps_window - 1]
    if initial <= 0:
        decay_pct = 0.0
    else:
        decay_pct = (initial - final) / initial * 100.0

    return GateResult(
        "G4_noisy_tv_resistance",
        passed=decay_pct >= 100.0 - threshold_max_decay_pct,
        score=decay_pct,
        threshold=100.0 - threshold_max_decay_pct,
        details={
            "initial_score": initial,
            "final_score": final,
            "n_steps": n_steps_window,
            "interpretation": "PASS = model dropped interest in noise rapidly",
        },
    )


def gate_g5_sustained_engagement(
    autonomous_turn_runs: List[int],
    threshold: int = 50,
) -> GateResult:
    """G5: median consecutive autonomous turns without external prompt > 50.

    autonomous_turn_runs: list of run lengths (e.g., [3, 12, 89, 4, 67, ...])
    """
    if not autonomous_turn_runs:
        return GateResult("G5_sustained_engagement", False, 0.0, threshold,
                          {"reason": "no runs"})

    sorted_runs = sorted(autonomous_turn_runs)
    n = len(sorted_runs)
    if n % 2 == 0:
        median = (sorted_runs[n // 2 - 1] + sorted_runs[n // 2]) / 2.0
    else:
        median = sorted_runs[n // 2]

    return GateResult(
        "G5_sustained_engagement",
        passed=median >= threshold,
        score=median,
        threshold=threshold,
        details={
            "n_runs": n,
            "max_run": max(sorted_runs),
            "min_run": min(sorted_runs),
            "p25": sorted_runs[n // 4] if n >= 4 else None,
            "p75": sorted_runs[3 * n // 4] if n >= 4 else None,
        },
    )


@dataclass
class CuriosityGateReport:
    g1: GateResult
    g2: GateResult
    g3: GateResult
    g4: GateResult
    g5: GateResult

    @property
    def passed_count(self) -> int:
        return sum([self.g1.passed, self.g2.passed, self.g3.passed,
                    self.g4.passed, self.g5.passed])

    @property
    def verdict(self) -> str:
        n = self.passed_count
        if n == 5:
            return "GENUINELY_CURIOUS"
        if n >= 3:
            return "PARTIALLY_CURIOUS"
        return "SCAFFOLDING_ONLY"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "passed_count": self.passed_count,
            "gates": {
                g.name: {
                    "passed": g.passed,
                    "score": g.score,
                    "threshold": g.threshold,
                    "details": g.details,
                }
                for g in [self.g1, self.g2, self.g3, self.g4, self.g5]
            },
        }


def run_all_gates(
    pursuit_log: List[str],
    generated_questions: List[str],
    reference_corpus_lines: List[str],
    mmlu_t0: Dict[str, float],
    mmlu_t72h: Dict[str, float],
    noisy_tv_scores: List[float],
    autonomous_turn_runs: List[int],
    out_path: Optional[str | Path] = None,
) -> CuriosityGateReport:
    report = CuriosityGateReport(
        g1=gate_g1_topic_entropy(pursuit_log),
        g2=gate_g2_question_novelty(generated_questions, reference_corpus_lines),
        g3=gate_g3_mmlu_gap_close(mmlu_t0, mmlu_t72h),
        g4=gate_g4_noisy_tv_resistance(noisy_tv_scores),
        g5=gate_g5_sustained_engagement(autonomous_turn_runs),
    )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    return report
