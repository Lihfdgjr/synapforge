"""Unit tests for sf.surrogate.

Run:
    /opt/conda/bin/python -m pytest /workspace/synapforge/test_surrogate.py -v
or:
    /opt/conda/bin/python /workspace/synapforge/test_surrogate.py
"""

from __future__ import annotations

import math

import pytest
import torch

from synapforge.surrogate import (
    PLIFCell,
    list_surrogates,
    register,
    spike,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SURROGATES = ["atan", "sigmoid", "triangle", "fast_sigmoid", "slayer"]


# ---------------------------------------------------------------------------
# Forward = Heaviside
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SURROGATES)
def test_forward_is_heaviside(name: str) -> None:
    """Spike forward must be exact step indicator regardless of surrogate."""
    v = torch.linspace(-2.0, 2.0, 21, device=DEVICE)
    thr = torch.tensor(0.0, device=DEVICE)
    s = spike(v, thr, surrogate=name, alpha=2.0)
    expected = (v >= 0).to(v.dtype)
    assert torch.equal(s, expected), f"{name}: forward != heaviside"
    assert s.dtype == v.dtype
    assert ((s == 0) | (s == 1)).all()


# ---------------------------------------------------------------------------
# Backward is finite, non-zero, no NaN/Inf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SURROGATES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_backward_finite_and_nonzero(name: str, dtype: torch.dtype) -> None:
    """Backward must produce finite, non-NaN gradients in fp32/bf16/fp16."""
    if dtype == torch.float16 and DEVICE == "cpu":
        pytest.skip("fp16 backward unstable on CPU")
    v = torch.linspace(-1.0, 1.0, 64, device=DEVICE, dtype=dtype, requires_grad=True)
    thr = torch.zeros((), device=DEVICE, dtype=dtype)
    s = spike(v, thr, surrogate=name, alpha=2.0)
    loss = s.sum()
    loss.backward()
    g = v.grad
    assert g is not None
    assert torch.isfinite(g).all(), f"{name}/{dtype}: non-finite grads"
    assert g.abs().sum() > 0, f"{name}/{dtype}: all-zero gradient (dead)"


# ---------------------------------------------------------------------------
# Peak at threshold; near-zero far from threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SURROGATES)
def test_peak_gradient_at_threshold(name: str) -> None:
    """Surrogate gradient is maximal at v == threshold and decays away.

    For triangle: gradient is *exactly* zero outside [-1/alpha, 1/alpha],
    so we test |x| << support bound vs |x| >> support bound separately.
    """
    alpha = 2.0
    # At threshold
    v0 = torch.zeros(1, device=DEVICE, requires_grad=True)
    spike(v0, torch.zeros((), device=DEVICE), surrogate=name, alpha=alpha).sum().backward()
    g0 = v0.grad.abs().item()
    # Far from threshold
    far = 5.0  # outside triangle support (1/alpha = 0.5)
    v_far = torch.full((1,), far, device=DEVICE, requires_grad=True)
    spike(v_far, torch.zeros((), device=DEVICE), surrogate=name, alpha=alpha).sum().backward()
    g_far = v_far.grad.abs().item()
    assert g0 > 0, f"{name}: zero gradient at threshold"
    assert g0 > g_far, f"{name}: g(0)={g0:g} not > g(far)={g_far:g}"
    # All except SLAYER+sigmoid should be near-zero or exactly zero far away
    assert g_far < g0 * 0.5, f"{name}: insufficient decay {g_far / g0:.3f}"


def test_atan_known_peak_value() -> None:
    """ATan analytical peak = alpha/2 at x = 0."""
    for alpha in [1.0, 2.0, 4.0]:
        v = torch.zeros(1, device=DEVICE, requires_grad=True)
        spike(v, torch.zeros((), device=DEVICE), surrogate="atan", alpha=alpha).sum().backward()
        peak = v.grad.abs().item()
        assert math.isclose(peak, alpha / 2.0, rel_tol=1e-4), (
            f"atan alpha={alpha}: peak {peak} != {alpha/2}"
        )


# ---------------------------------------------------------------------------
# PLIFCell — gradient flows through threshold + log_tau
# ---------------------------------------------------------------------------

def test_plifcell_forward_shapes() -> None:
    cell = PLIFCell(hidden=8).to(DEVICE)
    x = torch.randn(4, 8, device=DEVICE)
    s, v = cell(x)
    assert s.shape == x.shape
    assert v.shape == x.shape
    # Spike values are binary
    assert ((s == 0) | (s == 1)).all()


def test_plifcell_forward_seq_shapes() -> None:
    cell = PLIFCell(hidden=16).to(DEVICE)
    x_seq = torch.randn(3, 7, 16, device=DEVICE)
    s_seq, v_final = cell.forward_seq(x_seq)
    assert s_seq.shape == (3, 7, 16)
    assert v_final.shape == (3, 16)


def test_plifcell_gradient_through_threshold_and_tau() -> None:
    """Loss on spike output should produce nonzero grads on both Parameters."""
    cell = PLIFCell(hidden=16, tau_init=5.0, threshold_init=0.5).to(DEVICE)
    # Drive input to straddle threshold so multiple time steps spike.
    x_seq = torch.randn(2, 12, 16, device=DEVICE) * 0.6
    s_seq, _ = cell.forward_seq(x_seq)
    # Target rate 0.3
    loss = (s_seq.mean() - 0.3).pow(2)
    loss.backward()
    assert cell.threshold.grad is not None
    assert cell.log_tau.grad is not None
    assert cell.threshold.grad.abs().sum() > 0, "threshold got no gradient"
    assert cell.log_tau.grad.abs().sum() > 0, "log_tau got no gradient"
    assert torch.isfinite(cell.threshold.grad).all()
    assert torch.isfinite(cell.log_tau.grad).all()


def test_plifcell_reset_modes() -> None:
    """Drive cell over many steps with strong input so it must spike,
    then check the reset semantics on whichever steps fired."""
    for reset in ("subtract", "zero"):
        # tau_init=2.0 keeps decay small (~0.6) so membrane charges fast.
        cell = PLIFCell(hidden=4, tau_init=2.0, threshold_init=0.5,
                        reset=reset).to(DEVICE)
        # 30-step constant strong drive
        x_seq = torch.full((2, 30, 4), 5.0, device=DEVICE)
        s_seq, _ = cell.forward_seq(x_seq)
        assert s_seq.sum() > 0, f"no spikes at all under strong drive ({reset})"
        # Per-step semantic check: emulate one extra step by hand and verify reset.
        v_prev = torch.full((2, 4), 0.9, device=DEVICE)  # already over thr
        s, v = cell(torch.zeros(2, 4, device=DEVICE), v_prev=v_prev)
        assert (s == 1).all(), f"v_prev=0.9 > thr=0.5 must spike ({reset})"
        if reset == "zero":
            # Hard-reset must zero out post-spike voltage exactly.
            assert torch.allclose(v, torch.zeros_like(v), atol=1e-6),                 f"zero reset not exact: {v}"
        else:  # subtract
            # Soft reset = old_v - thr (after the LIF step decayed v).
            assert (v < v_prev).all(), "subtract reset did not lower voltage"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_register_custom_surrogate() -> None:
    @register("box_ste")
    class _BoxSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.save_for_backward(x)
            ctx.alpha = float(alpha)
            return (x >= 0).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return grad_output * (x.abs() < (1.0 / ctx.alpha)).to(x.dtype), None

    v = torch.linspace(-1.0, 1.0, 32, device=DEVICE, requires_grad=True)
    s = spike(v, torch.zeros((), device=DEVICE), surrogate="box_ste", alpha=2.0)
    s.sum().backward()
    assert v.grad.abs().sum() > 0
    assert "box_ste" in list_surrogates()


def test_unknown_surrogate_raises() -> None:
    with pytest.raises(KeyError):
        spike(torch.zeros(2, device=DEVICE), torch.zeros((), device=DEVICE), surrogate="nope")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
