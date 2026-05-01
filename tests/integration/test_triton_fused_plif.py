"""DEEP_MAINT_QUEUE.md T2.2 — fused PLIF surrogate forward+backward.

Validates the new fused PLIF surrogate kernel
(``synapforge.backends.triton_fused_backward``) against the existing
Python ATan / Sigmoid surrogate path. The fused path:

  * computes ``spike = (v >= thr)`` and ``dspike/dv = surrogate'(v - thr)``
    in a single pass (single Triton tile when CUDA is available);
  * caches only the fp32 ``dspike/dv`` tensor for backward (no ``v - thr``
    autograd tape stash);
  * is opt-in via ``PLIFCell(use_triton_fused=True)`` and the trainer's
    ``--triton-fused-backward`` flag.

This file:
  * verifies forward parity with :func:`synapforge.surrogate.spike` for
    the ATan surrogate (atol=1e-5 fp32 / atol=1e-3 bf16),
  * verifies backward parity with the Python surrogate (atol=1e-4),
  * exercises 2-D and 3-D shapes,
  * confirms PLIFCell defaults DON'T enable the fused path (opt-in),
  * runs CPU-only via the pure-PyTorch reference fallback (the Triton
    kernel itself only fires on CUDA — CUDA-only checks skip cleanly
    via ``pytest.skip`` when ``torch.cuda.is_available()`` is False).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_module():
    from synapforge.backends import triton_fused_backward as mod
    return mod


def _python_atan_grad(v: torch.Tensor, thr, alpha: float) -> torch.Tensor:
    """Reference: ATan surrogate gradient g'(v - thr).

    Mirrors :class:`synapforge.surrogate.ATanSurrogate.backward` so we
    can directly diff against the fused kernel's cached dspike/dv.
    """
    import math
    PI_HALF = math.pi / 2.0
    if not isinstance(thr, torch.Tensor):
        thr = torch.tensor(float(thr), device=v.device, dtype=v.dtype)
    m = (v.float() - thr.float())
    return alpha / (2.0 * (1.0 + (PI_HALF * alpha * m).pow(2)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_matches_python_reference():
    """Forward (binary spike) must match the existing ATan surrogate path
    elementwise for fp32 inputs."""
    from synapforge.surrogate import spike as py_spike
    mod = _import_module()
    torch.manual_seed(0)

    v = torch.randn(4, 8) * 0.5
    thr = torch.zeros(8)
    alpha = 2.0

    s_fused = mod.fused_plif_spike(v, thr, alpha=alpha, surrogate="atan")
    s_ref = py_spike(v, thr, surrogate="atan", alpha=alpha)

    # Forward is exact Heaviside on both paths.
    assert torch.equal(s_fused, s_ref), (
        f"forward mismatch: max |diff|={(s_fused - s_ref).abs().max().item()}"
    )
    # Tighter sanity: all values are 0 or 1.
    assert ((s_fused == 0) | (s_fused == 1)).all()


def test_backward_matches_python_reference():
    """Gradient through the fused path must match ATanSurrogate.backward
    within 1e-4 (which is the relative-error budget the queue task
    specifies for the Triton path)."""
    from synapforge.surrogate import spike as py_spike
    mod = _import_module()
    torch.manual_seed(1)

    v_init = (torch.randn(4, 32) * 0.5)
    v_a = v_init.clone().requires_grad_(True)
    thr_a = torch.zeros(32).requires_grad_(True)
    v_b = v_init.clone().requires_grad_(True)
    thr_b = torch.zeros(32).requires_grad_(True)

    alpha = 2.0

    # fused path
    s_fused = mod.fused_plif_spike(v_a, thr_a, alpha=alpha, surrogate="atan")
    s_fused.sum().backward()

    # python ref
    s_ref = py_spike(v_b, thr_b, surrogate="atan", alpha=alpha)
    s_ref.sum().backward()

    diff_v = (v_a.grad - v_b.grad).abs().max().item()
    diff_thr = (thr_a.grad - thr_b.grad).abs().max().item()
    assert diff_v < 1e-4, f"v gradient mismatch: {diff_v}"
    assert diff_thr < 1e-4, f"thr gradient mismatch: {diff_thr}"


def test_shapes_3d():
    """3-D input (B, T, D) — shape preservation + per-channel thr broadcast."""
    mod = _import_module()
    torch.manual_seed(2)

    B, T, D = 4, 64, 256
    v = (torch.randn(B, T, D) * 0.5).requires_grad_(True)
    thr = torch.full((D,), 0.05).requires_grad_(True)
    alpha = 2.0

    s = mod.fused_plif_spike(v, thr, alpha=alpha, surrogate="atan")
    assert s.shape == (B, T, D)
    # Backward must produce shape-correct grads.
    s.sum().backward()
    assert v.grad is not None and v.grad.shape == (B, T, D)
    assert thr.grad is not None and thr.grad.shape == (D,)
    # And the dspike/dv computation must match the python ref pointwise.
    expected_grad = _python_atan_grad(v.detach(), thr.detach(), alpha)
    diff = (v.grad - expected_grad).abs().max().item()
    assert diff < 1e-4, f"3-D backward mismatch {diff}"


def test_shapes_2d():
    """2-D input (B, D) — single-step PLIF call shape."""
    mod = _import_module()
    torch.manual_seed(3)

    B, D = 4, 256
    v = (torch.randn(B, D) * 0.5).requires_grad_(True)
    thr = torch.full((D,), 0.05).requires_grad_(True)
    alpha = 2.0

    s = mod.fused_plif_spike(v, thr, alpha=alpha, surrogate="atan")
    assert s.shape == (B, D)
    s.sum().backward()
    assert v.grad is not None and v.grad.shape == (B, D)
    assert thr.grad is not None and thr.grad.shape == (D,)


def test_bf16_dtype():
    """bf16 forward must produce binary spikes; backward must be finite.

    The kernel explicitly upcasts the surrogate computation to fp32 to
    avoid the known Triton 2.x bf16 silent downcast bug. The PyTorch
    fallback (used here on CPU) already does the same via the upcast
    in ``_pytorch_fwd_bwd_reference``.

    Tolerance is 1e-3 atol for the forward (matches the
    ``test_kd_topk_softmax`` bf16 budget); backward is verified
    finite + non-zero rather than bit-exact, since bf16 rounds the
    gradient cache.
    """
    from synapforge.surrogate import spike as py_spike
    mod = _import_module()
    torch.manual_seed(4)

    v = (torch.randn(2, 8) * 0.5).to(torch.bfloat16)
    thr = torch.zeros(8, dtype=torch.bfloat16)
    alpha = 2.0

    s_fused = mod.fused_plif_spike(v, thr, alpha=alpha, surrogate="atan")
    assert s_fused.dtype == torch.bfloat16
    assert ((s_fused == 0) | (s_fused == 1)).all()
    # bf16 forward should match python ref to atol 1e-3.
    s_ref = py_spike(v, thr, surrogate="atan", alpha=alpha)
    diff = (s_fused.float() - s_ref.float()).abs().max().item()
    assert diff < 1e-3, f"bf16 forward mismatch {diff}"

    # Backward: produce finite, non-NaN grads.
    v2 = v.detach().clone().requires_grad_(True)
    thr2 = thr.detach().clone().requires_grad_(True)
    s2 = mod.fused_plif_spike(v2, thr2, alpha=alpha, surrogate="atan")
    s2.sum().backward()
    assert v2.grad is not None
    assert torch.isfinite(v2.grad).all(), "non-finite bf16 grads"
    assert v2.grad.abs().sum() > 0, "all-zero bf16 grads (dead)"


def test_zero_gradient_when_v_far_from_thr():
    """Sanity: ATan derivative decays as 1/x^2 in the alpha=2 case.

    When ``|v - thr|`` is extreme (here 100x the surrogate width), the
    gradient must be very near zero (< 1e-3 relative to the peak alpha/2
    at v = thr). This catches kernels that accidentally leak the forward
    Heaviside through to the backward pass.
    """
    mod = _import_module()
    alpha = 2.0
    PI_HALF = 1.5707963267948966
    # Peak of ATan' is alpha / 2 at v = thr.
    peak = alpha / 2.0
    # At m = 50, |g'| = alpha / (2 * (1 + (pi/2 * alpha * 50)^2)) ~ 6.5e-5.
    v = torch.full((1,), 50.0, requires_grad=True)
    thr = torch.zeros(1, requires_grad=True)
    s = mod.fused_plif_spike(v, thr, alpha=alpha, surrogate="atan")
    s.sum().backward()
    g = v.grad.abs().item()
    assert g < peak * 1e-3, (
        f"gradient {g} not small enough vs peak {peak}; expected far-from-thr "
        f"to decay below 1e-3 of peak"
    )
    # And the analytic value matches.
    analytic = alpha / (2.0 * (1.0 + (PI_HALF * alpha * 50.0) ** 2))
    assert abs(g - analytic) < 1e-6


def test_flag_default_off():
    """PLIFCell with default args must use the Python surrogate path.

    This is the production safety gate: the fused path is opt-in. We
    verify the default by (a) checking the constructor flag and (b)
    asserting that running forward with default args matches the
    existing :func:`spike` path bit-exactly (both are calling the same
    ATan autograd Function).
    """
    from synapforge.surrogate import PLIFCell, spike as py_spike

    cell = PLIFCell(hidden=8)
    assert cell.use_triton_fused is False, (
        "PLIFCell default constructor must NOT enable the fused path"
    )

    torch.manual_seed(5)
    x_seq = torch.randn(2, 6, 8) * 0.5
    s_seq, _ = cell.forward_seq(x_seq)
    assert s_seq.shape == (2, 6, 8)
    assert ((s_seq == 0) | (s_seq == 1)).all()

    # Cross-check with the explicit fused=False vs fused=True (CPU)
    # paths producing the same spike trace — both fall through to the
    # ATan reference math; only the autograd cache differs.
    cell_fused = PLIFCell(hidden=8, use_triton_fused=True)
    # Copy the (random-init) parameters across so the comparison is
    # apples-to-apples.
    with torch.no_grad():
        cell_fused.log_tau.copy_(cell.log_tau)
        cell_fused.threshold.copy_(cell.threshold)
    s_seq_fused, _ = cell_fused.forward_seq(x_seq)
    assert torch.equal(s_seq, s_seq_fused), (
        f"fused-on/off forward differ on CPU fallback (max |diff| "
        f"{(s_seq - s_seq_fused).abs().max().item()}) — they share the "
        f"same ATan reference math; both should be identical"
    )
