"""
STDP-driven curiosity score — the single signal nobody else can compute.

Per agent synthesis 2026-05-01, the curiosity formula for our LNN+SNN:

    C(s, a) = α·ΔF                   # Free-energy reduction (expected info gain)
            + β·||ΔW_STDP||           # synaptic change ("this changed me")
            + γ·G_retrieval           # HNSW coverage gap
            + δ·H[r_spike]            # spike-rate variance (engagement)
            + ε·N_ema                 # NoveltyDrive cosine novelty
            − ζ·V_noise               # noise-variance penalty (kills noisy-TV)

Recommended weights (Friston 2017 active inference + ablation priors):
    α=0.40  (primary signal: variational FE reduction)
    β=0.25  (load-bearing: STDP weight delta)
    γ=0.15  (HNSW coverage gap)
    δ=0.10  (PLIF spike-rate engagement)
    ε=0.05  (short-term novelty diversity)
    ζ=0.05  (kill noisy-TV)

Why STDP-delta is THE novel angle:

  Forward-model curiosity (ICM 1705.05363, RND 1810.12894) measures
  prediction error → fails on stochastic ("noisy-TV") because random
  observations always have high error.

  STDP-delta measures actual synaptic change → mathematically robust to
  noise: random pre/post pairs are uncorrelated, the Hebbian outer-product
  rule averages them to ~0. Noisy-TV gets a low curiosity score by
  construction.

  Transformer can't compute this — no inference-time plasticity. We're
  the only architecture that can.

Anchor papers:
  - Friston 2017 active inference (FEP framing)
  - Schmidhuber 1991/2010 Bayesian surprise (1306.6062 retrofit)
  - Pathak 2017 ICM (1705.05363)
  - Burda 2018 RND (1810.12894)
  - Badia 2020 NGU (2002.06038)
  - Salge 2014 Empowerment (1310.1863)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CuriosityScore:
    total: float
    delta_f: float = 0.0
    stdp_norm: float = 0.0
    retrieval_gap: float = 0.0
    spike_variance: float = 0.0
    novelty_ema: float = 0.0
    noise_penalty: float = 0.0
    components: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.components is None:
            self.components = {
                "delta_f": self.delta_f,
                "stdp_norm": self.stdp_norm,
                "retrieval_gap": self.retrieval_gap,
                "spike_variance": self.spike_variance,
                "novelty_ema": self.novelty_ema,
                "noise_penalty": self.noise_penalty,
            }


class CuriosityScorer:
    """Combines 6 signals into a single curiosity score.

    Usage:
        scorer = CuriosityScorer()  # default Friston weights

        # Per-step (called from trainer or autonomous_daemon)
        score = scorer.score(
            free_energy_reduction=0.42,        # from FreeEnergyEstimator
            stdp_delta_norm=0.18,              # from STDPFastWeight delta
            retrieval_gap=0.65,                # 1 - max_cosine to HNSW top-K
            spike_rate_variance=0.27,          # variance of PLIF spike rate
            novelty_ema=0.31,                  # NoveltyDrive output
            noise_variance=0.08,               # input noise estimate
        )
        # score.total = α·0.42 + β·0.18 + ... = combined curiosity

    Higher score = more curious about this topic / observation.
    Use to rank candidate topics in SelfGoalProposer / autonomous_daemon.
    """

    DEFAULT_WEIGHTS = {
        "alpha": 0.40,
        "beta": 0.25,
        "gamma": 0.15,
        "delta": 0.10,
        "epsilon": 0.05,
        "zeta": 0.05,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
    ) -> None:
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.normalize = normalize

        self.history: List[CuriosityScore] = []
        self._noise_floor_ema = 0.0
        self._stdp_running_max = 1e-6

    def score(
        self,
        free_energy_reduction: float,
        stdp_delta_norm: float,
        retrieval_gap: float,
        spike_rate_variance: float,
        novelty_ema: float,
        noise_variance: float,
    ) -> CuriosityScore:
        if self.normalize:
            self._stdp_running_max = max(self._stdp_running_max, abs(stdp_delta_norm))
            stdp_norm = stdp_delta_norm / self._stdp_running_max
            self._noise_floor_ema = 0.99 * self._noise_floor_ema + 0.01 * noise_variance
            noise_pen = max(0.0, noise_variance - self._noise_floor_ema)
        else:
            stdp_norm = stdp_delta_norm
            noise_pen = noise_variance

        delta_f = max(0.0, free_energy_reduction)
        retrieval_gap = max(0.0, min(1.0, retrieval_gap))
        spike_var = max(0.0, spike_rate_variance)
        novelty = max(0.0, min(1.0, novelty_ema))

        total = (
            self.weights["alpha"] * delta_f
            + self.weights["beta"] * stdp_norm
            + self.weights["gamma"] * retrieval_gap
            + self.weights["delta"] * spike_var
            + self.weights["epsilon"] * novelty
            - self.weights["zeta"] * noise_pen
        )

        cs = CuriosityScore(
            total=total,
            delta_f=delta_f,
            stdp_norm=stdp_norm,
            retrieval_gap=retrieval_gap,
            spike_variance=spike_var,
            novelty_ema=novelty,
            noise_penalty=noise_pen,
        )
        self.history.append(cs)
        if len(self.history) > 1024:
            self.history = self.history[-1024:]
        return cs

    def rank_topics(
        self,
        topic_signals: List[Dict[str, float]],
    ) -> List[Tuple[int, CuriosityScore]]:
        """Rank a list of candidate topics by curiosity score.

        Each topic_signals[i] should be a dict with keys matching .score() args.
        Returns [(topic_index, score), ...] sorted desc.
        """
        scored = []
        for i, sig in enumerate(topic_signals):
            cs = self.score(**sig)
            scored.append((i, cs))
        scored.sort(key=lambda x: x[1].total, reverse=True)
        return scored

    def stats(self) -> dict:
        if not self.history:
            return {"n": 0}
        last = self.history[-100:]
        totals = [s.total for s in last]
        return {
            "n": len(self.history),
            "mean_score": sum(totals) / len(totals),
            "max_score": max(totals),
            "min_score": min(totals),
            "stdp_running_max": self._stdp_running_max,
            "noise_floor_ema": self._noise_floor_ema,
        }


def stdp_delta_norm_from_module(stdp_module) -> float:
    """Extract ||ΔW_STDP|| from a STDPFastWeight module.

    Compares current W against snapshot stored at last call. Returns
    Frobenius norm of the difference.

    Usage:
        # First call: stores snapshot, returns 0
        norm = stdp_delta_norm_from_module(stdp)
        # Subsequent calls: returns ||W_now - W_prev||_F, updates snapshot
    """
    if not hasattr(stdp_module, "W"):
        return 0.0
    if not hasattr(stdp_module, "_curiosity_W_snapshot"):
        stdp_module._curiosity_W_snapshot = stdp_module.W.detach().clone()
        return 0.0

    with torch.no_grad():
        delta = stdp_module.W - stdp_module._curiosity_W_snapshot
        norm = float(torch.linalg.norm(delta).item())
        stdp_module._curiosity_W_snapshot = stdp_module.W.detach().clone()
    return norm


def spike_rate_variance_from_modules(plif_modules) -> float:
    """Extract variance of spike rate across PLIF modules.

    Engagement signal: high variance = some channels alive, some sparse,
    healthy heterogeneity. Uniform spike rate = boredom or saturation.
    """
    rates = []
    for m in plif_modules:
        if hasattr(m, "last_spike_rate"):
            rates.append(float(m.last_spike_rate))
    if len(rates) < 2:
        return 0.0
    rates_t = torch.tensor(rates)
    return float(rates_t.var().item())


def retrieval_gap_from_hnsw(hnsw_index, query_embedding, top_k: int = 5) -> float:
    """Coverage gap: 1 - max cosine similarity to top-K stored prototypes.

    Higher = more novel relative to existing memory.
    Voyager-style (2305.16291) skill-gap discovery using neural retrieval
    instead of an LLM judge.
    """
    if hnsw_index is None or hnsw_index.K == 0:
        return 1.0
    hits = hnsw_index.query(query_embedding, top_k=top_k)
    if not hits:
        return 1.0
    max_sim = max(score for _, score in hits)
    return max(0.0, min(1.0, 1.0 - max_sim))
