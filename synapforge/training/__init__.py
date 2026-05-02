"""synapforge.training -- training-side helpers (EMA tracker, etc.).

This subpackage hosts utilities that are *training-loop adjacent* but not
part of the model architecture itself: gradient accumulation helpers,
exponential-moving-average weight trackers, scheduler glue, and the like.

Kept separate from ``synapforge.optim`` (which builds optimizers) so that
the trainer can import ``from synapforge.training.ema import EMATracker``
without pulling in optimiser construction.
"""
from __future__ import annotations

from .ema import EMATracker, ModelEMA, load_ema
from .grpo import (
    GRPOStats,
    VERIFIERS,
    ast_verifier,
    compute_advantages,
    extract_final_number,
    freeze_reference_policy,
    get_verifier,
    grpo_loss,
    kl_divergence_per_token,
    sample_rollouts_mock,
    sympy_verifier,
)
from .neuromcp_mixin import NeuroMCPMixin
from .sft_loop import (
    InstructionParquetStream,
    response_only_ce_loss,
    write_synth_alpaca_parquet,
)

__all__ = [
    "EMATracker",
    "GRPOStats",
    "InstructionParquetStream",
    "ModelEMA",
    "NeuroMCPMixin",
    "VERIFIERS",
    "ast_verifier",
    "compute_advantages",
    "extract_final_number",
    "freeze_reference_policy",
    "get_verifier",
    "grpo_loss",
    "kl_divergence_per_token",
    "load_ema",
    "response_only_ce_loss",
    "sample_rollouts_mock",
    "sympy_verifier",
    "write_synth_alpaca_parquet",
]
