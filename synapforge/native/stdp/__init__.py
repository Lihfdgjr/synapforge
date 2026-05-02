"""synapforge.native.stdp — STDP-only local learning rule.

This package replaces AdamW for plasticity-tagged weights. STDP
(Spike-Timing-Dependent Plasticity) is a local Hebbian rule:

    Delta_w_ij = alpha * pre_trace_j * post_spike_i  (LTP)
               - alpha * post_trace_i * pre_spike_j  (LTD)

No autograd. No optimizer state. Per-step cost is O(active_spikes),
not O(weights). Saves the H2D grad copy, the CPU step, and the D2H
parameter copy for every plasticity-tagged weight.

Public API
----------
* :class:`STDPOnlyOptimizer`        — drop-in optimizer for plasticity weights
* :class:`HybridOptimizerDispatcher` — routes plasticity to STDP, BP to AdamW
* :class:`SpikeRingBuffer`           — last-K timestep ring buffer for pre+post
* :func:`stdp_update_kernel`         — Triton kernel (pure-Python fallback)
* :func:`per_param_alpha`            — match LR to layer's CfC tau
"""
from __future__ import annotations

from .per_param_lr import per_param_alpha
from .spike_buffer import SpikeRingBuffer
from .stdp_optimizer import STDPOnlyOptimizer

# hybrid_optim_dispatch imports torch via a lazy import — keep optional so
# callers using only stdp_optimizer (no AdamW group) don't pay the cost.
try:
    from .hybrid_optim_dispatch import HybridOptimizerDispatcher
except Exception:  # pragma: no cover - optional path
    HybridOptimizerDispatcher = None  # type: ignore[assignment]

__all__ = [
    "STDPOnlyOptimizer",
    "HybridOptimizerDispatcher",
    "SpikeRingBuffer",
    "per_param_alpha",
]
