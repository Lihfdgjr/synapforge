"""Stub: fused PLIF surrogate forward + backward Triton kernel.

Status: DESIGN-ONLY. ``enable_fused_backward()`` raises
``NotImplementedError`` so callers (the trainer behind
``--triton-fused-backward``) gracefully fall back to the existing
PyTorch ``torch.autograd.Function`` surrogate path. The trainer logs
a clear "disabled: <reason>" line and continues.

Why this exists as a stub
-------------------------
The existing surrogate path in ``synapforge/surrogate.py`` runs five
``torch.autograd.Function`` subclasses (ATan / Sigmoid / Triangle /
FastSigmoid / SLAYER). Forward is exact Heaviside; backward computes
``dspike/d(v - thr)`` in fp32 and casts back to the input dtype.

In the production trainer, every PLIF cell forward does
``spike = SurrogateFn.apply(v - threshold, alpha)``, which on bf16
autocast incurs:

  * ~10us Python dispatch + ctx.save_for_backward overhead per call
  * stash of (v - threshold) into the autograd tape (~B*T*D fp32 buf)
  * a separate kernel launch on backward to compute the surrogate grad

At 10 PLIF layers x 256 timesteps x bs=80, the dispatch overhead alone
is ~25 ms/step (= ~3% of a 0.78s baseline step on A800-80GB), and the
saved ``v - threshold`` tape doubles activation memory for the
spike-train.

Design (target kernel, NOT implemented)
---------------------------------------
``synapforge/backends/triton_block_kernel.py`` already fuses

    h_t  = A_t * h_{t-1} + b_t                  # CfC scan
    s_t  = Heaviside(h_t - thr_d)               # PLIF spike
    h_t  = h_t * (1 - s_t)                      # subtract reset

into a single Triton kernel launch per layer per loop_depth iter.
Backward currently falls back to PyTorch autograd over the cached
``h_pre`` and ``m = h_pre - thr`` buffers (see ``m_ptr`` in the kernel
header).

The fused-backward kernel would:

  1. compute ``ds/dm = surrogate_grad(m, alpha)`` in the SAME tile as
     the forward, write into the spike tile's stash slot.
  2. on backward, the autograd Function reads the precomputed
     ``ds/dm`` instead of dispatching ATan/Sigmoid/etc.
  3. saves the ~B*T*D fp32 ``m`` buffer (since we no longer need to
     replay it on backward), reducing activation memory by ~30%.

Open questions before implementation:

  * Per-layer alpha vs per-channel alpha tile broadcast.
  * Surrogate function selection at kernel-compile time (``alpha`` and
    surrogate type become tl.constexpr) — adds 5x kernel variants but
    keeps the inner loop branchless.
  * Interaction with ``_stable_fp32`` autocast guard (the surrogate
    derivative is computed in fp32 even when forward is bf16; the
    kernel needs an internal upcast for the (alpha * x)**2 path).

Target: 5-10% step-time win on the bs=80 baseline by removing the
Python-side surrogate dispatch + halving the spike-train autograd tape.

Wiring
------
The trainer's ``--triton-fused-backward`` flag (default OFF) calls
``enable_fused_backward()`` once at startup. The current stub raises
``NotImplementedError`` with a "kernel not yet implemented" message
which the trainer catches and logs; no other call sites change.

When the kernel lands:

  * Replace ``enable_fused_backward()`` body with a registry hook that
    swaps ``synapforge.surrogate._REGISTRY`` entries for fused variants
    that read the kernel-stashed ``ds/dm`` buffer.
  * Add a ``synapforge/triton_fused_backward_kernel.py`` next to
    ``triton_block_kernel.py`` for the @triton.jit functions.
  * Update ``docs/PERF_KNOBS.md`` row from "theoretical" to measured.
"""
from __future__ import annotations

__all__ = ["enable_fused_backward", "is_available"]


def is_available() -> bool:
    """True only when the fused-backward kernel is implemented and the
    runtime has Triton + CUDA available. Currently always False."""
    return False


def enable_fused_backward() -> None:
    """Activate the fused PLIF surrogate forward+backward Triton kernel.

    NOT YET IMPLEMENTED. Raises ``NotImplementedError`` so the trainer's
    ``--triton-fused-backward`` opt-in path falls back gracefully to the
    existing PyTorch autograd surrogate without crashing the run. The
    trainer catches this and logs a clear disabled message.
    """
    raise NotImplementedError(
        "fused PLIF surrogate backward Triton kernel not yet implemented "
        "(see synapforge/backends/triton_fused_backward.py docstring for "
        "the design). Run without --triton-fused-backward, or contribute "
        "the kernel."
    )
