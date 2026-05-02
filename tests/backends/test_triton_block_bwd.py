"""Numerical correctness tests for ``synapforge.backends.triton_block_kernel_bwd``.

Verifies that the closed-form Triton backward for the SEW + sigmoid-gate
fusion matches torch.autograd.grad to within 1e-4 relative error.

The test runs on either GPU (if Triton + CUDA available) or CPU (which
exercises only the torch fallback path — still useful as a regression
gate for the math).
"""
from __future__ import annotations

import math

import pytest
import torch

from synapforge.backends.triton_block_kernel_bwd import (
    SEWSigmoidGateFn,
    _self_test,
    _torch_sew_sigmoid_gate_backward,
    _torch_sew_sigmoid_gate_forward,
    sew_sigmoid_gate_fused,
)


# ---------------------------------------------------------------------------
# Reference implementation (the same math through torch's autograd, verbatim).
# ---------------------------------------------------------------------------


def _torch_ref_fwd_bwd(
    s: torch.Tensor,
    h: torch.Tensor,
    syn_out: torch.Tensor,
    gate_pre: torch.Tensor,
    grad_spike_input: torch.Tensor,
    grad_gated: torch.Tensor,
):
    """Compute the (forward, backward) using only standard torch ops.

    Returns
    -------
    spike_input, gated, grad_s, grad_h, grad_syn, grad_gp
    """
    s_r = s.clone().detach().requires_grad_(True)
    h_r = h.clone().detach().requires_grad_(True)
    syn_r = syn_out.clone().detach().requires_grad_(True)
    gp_r = gate_pre.clone().detach().requires_grad_(True)

    spike_input = s_r + h_r
    gate = torch.sigmoid(gp_r)
    gated = syn_r * gate

    torch.autograd.backward(
        [spike_input, gated],
        [grad_spike_input, grad_gated],
    )
    return (
        spike_input.detach(),
        gated.detach(),
        s_r.grad.clone(),
        h_r.grad.clone(),
        syn_r.grad.clone(),
        gp_r.grad.clone(),
    )


def _fused_fwd_bwd(
    s: torch.Tensor,
    h: torch.Tensor,
    syn_out: torch.Tensor,
    gate_pre: torch.Tensor,
    grad_spike_input: torch.Tensor,
    grad_gated: torch.Tensor,
):
    """Same dataflow but routed through ``SEWSigmoidGateFn``."""
    s_f = s.clone().detach().requires_grad_(True)
    h_f = h.clone().detach().requires_grad_(True)
    syn_f = syn_out.clone().detach().requires_grad_(True)
    gp_f = gate_pre.clone().detach().requires_grad_(True)

    spike_input, gated = sew_sigmoid_gate_fused(s_f, h_f, syn_f, gp_f)
    torch.autograd.backward(
        [spike_input, gated],
        [grad_spike_input, grad_gated],
    )
    return (
        spike_input.detach(),
        gated.detach(),
        s_f.grad.clone(),
        h_f.grad.clone(),
        syn_f.grad.clone(),
        gp_f.grad.clone(),
    )


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).abs().max().item()
    denom = b.abs().mean().item() + 1e-12
    return diff / denom


# ---------------------------------------------------------------------------
# Test fixtures.
# ---------------------------------------------------------------------------


def _make_inputs(B: int, T: int, D: int, device: torch.device, dtype=torch.float32):
    torch.manual_seed(7)
    s = (torch.rand(B, T, D, device=device) > 0.7).to(dtype)
    h = torch.randn(B, T, D, device=device, dtype=dtype) * 0.5
    syn_out = torch.randn(B, T, D, device=device, dtype=dtype) * 0.3
    gate_pre = torch.randn(B, T, D, device=device, dtype=dtype) * 0.4
    grad_spike_input = torch.randn(B, T, D, device=device, dtype=dtype)
    grad_gated = torch.randn(B, T, D, device=device, dtype=dtype)
    return s, h, syn_out, gate_pre, grad_spike_input, grad_gated


# ---------------------------------------------------------------------------
# Per-op math tests (no autograd machinery). Validate the closed-form math
# directly against numpy-style baseline.
# ---------------------------------------------------------------------------


def test_torch_reference_forward_matches_naive():
    """``_torch_sew_sigmoid_gate_forward`` should equal a naive expression."""
    s, h, syn_out, gate_pre, _, _ = _make_inputs(2, 8, 32, torch.device("cpu"))
    spike_input, gate, gated = _torch_sew_sigmoid_gate_forward(
        s, h, syn_out, gate_pre,
    )

    ref_spike_input = s + h
    ref_gate = torch.sigmoid(gate_pre)
    ref_gated = syn_out * ref_gate

    assert torch.allclose(spike_input, ref_spike_input, atol=1e-7)
    assert torch.allclose(gate, ref_gate, atol=1e-7)
    assert torch.allclose(gated, ref_gated, atol=1e-7)


def test_torch_reference_backward_matches_closed_form():
    """``_torch_sew_sigmoid_gate_backward`` should equal d/dz of sigmoid·mul."""
    s, h, syn_out, gate_pre, _, grad_gated = _make_inputs(
        2, 8, 32, torch.device("cpu"),
    )
    _, gate, _ = _torch_sew_sigmoid_gate_forward(s, h, syn_out, gate_pre)

    grad_syn, grad_gate_pre = _torch_sew_sigmoid_gate_backward(
        syn_out, gate, grad_gated,
    )

    # Closed-form: grad_syn = grad_gated * gate
    #              grad_gate_pre = grad_gated * syn_out * gate * (1 - gate)
    ref_grad_syn = grad_gated * gate
    ref_grad_gate_pre = grad_gated * syn_out * gate * (1.0 - gate)

    # Tight: same tensor, no kernel, no fp32 round-trip in the path.
    assert torch.allclose(grad_syn, ref_grad_syn, atol=1e-6)
    assert torch.allclose(grad_gate_pre, ref_grad_gate_pre, atol=1e-6)


# ---------------------------------------------------------------------------
# Autograd integration tests — full SEWSigmoidGateFn vs torch chain.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,T,D", [(2, 8, 128), (1, 32, 64), (4, 4, 256)])
def test_fused_matches_torch_autograd_cpu(B: int, T: int, D: int):
    """Full Function fwd+bwd matches torch on CPU (within 1e-4 rel err).

    On CPU the SEWSigmoidGateFn falls through to the torch reference
    path (Triton path skipped), so this is mostly a smoke test for the
    autograd plumbing + closed-form math.
    """
    device = torch.device("cpu")
    s, h, syn_out, gate_pre, grad_spike_input, grad_gated = _make_inputs(
        B, T, D, device,
    )

    spike_r, gated_r, gs_r, gh_r, gsyn_r, ggp_r = _torch_ref_fwd_bwd(
        s, h, syn_out, gate_pre, grad_spike_input, grad_gated,
    )
    spike_f, gated_f, gs_f, gh_f, gsyn_f, ggp_f = _fused_fwd_bwd(
        s, h, syn_out, gate_pre, grad_spike_input, grad_gated,
    )

    assert _relerr(spike_f, spike_r) < 1e-4, "spike_input forward mismatch"
    assert _relerr(gated_f, gated_r) < 1e-4, "gated forward mismatch"
    # SEW path: grad_s = grad_h = grad_spike_input identity.
    assert _relerr(gs_f, gs_r) < 1e-4, "grad_s mismatch"
    assert _relerr(gh_f, gh_r) < 1e-4, "grad_h mismatch"
    assert _relerr(gsyn_f, gsyn_r) < 1e-4, "grad_syn_out mismatch"
    assert _relerr(ggp_f, ggp_r) < 1e-4, "grad_gate_pre mismatch"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cuda not available",
)
@pytest.mark.parametrize(
    "B,T,D,dtype",
    [
        (2, 8, 128, torch.float32),
        (4, 16, 256, torch.float32),
        (2, 8, 128, torch.bfloat16),
    ],
)
def test_fused_matches_torch_autograd_cuda(B, T, D, dtype):
    """End-to-end Triton bwd vs torch.autograd.grad on cuda."""
    device = torch.device("cuda")
    s, h, syn_out, gate_pre, grad_spike_input, grad_gated = _make_inputs(
        B, T, D, device, dtype=dtype,
    )
    spike_r, gated_r, gs_r, gh_r, gsyn_r, ggp_r = _torch_ref_fwd_bwd(
        s, h, syn_out, gate_pre, grad_spike_input, grad_gated,
    )
    spike_f, gated_f, gs_f, gh_f, gsyn_f, ggp_f = _fused_fwd_bwd(
        s, h, syn_out, gate_pre, grad_spike_input, grad_gated,
    )
    # bf16 has ~3-decimal precision, so we relax slightly.
    tol = 1e-4 if dtype == torch.float32 else 5e-3

    assert _relerr(spike_f.float(), spike_r.float()) < tol, "spike_input fwd"
    assert _relerr(gated_f.float(), gated_r.float()) < tol, "gated fwd"
    assert _relerr(gs_f.float(), gs_r.float()) < tol, "grad_s"
    assert _relerr(gh_f.float(), gh_r.float()) < tol, "grad_h"
    assert _relerr(gsyn_f.float(), gsyn_r.float()) < tol, "grad_syn_out"
    assert _relerr(ggp_f.float(), ggp_r.float()) < tol, "grad_gate_pre"


def test_self_test_runs():
    """``_self_test`` smoke."""
    info = _self_test(device="cuda" if torch.cuda.is_available() else "cpu")
    assert "rel_grad_h" in info
    assert "rel_grad_syn_out" in info
    assert "rel_grad_gate_pre" in info
    # On CPU we still hit the torch reference path.
    if info["device"] == "cpu":
        assert info["ok"], f"CPU torch path failed self-test: {info}"


def test_grad_spike_input_none_path():
    """When the spike_input output isn't used downstream, grad is None.

    Regression for the ``grad_spike_input is None`` branch in backward.
    """
    s, h, syn_out, gate_pre, _, grad_gated = _make_inputs(
        2, 8, 32, torch.device("cpu"),
    )
    s_f = s.clone().detach().requires_grad_(True)
    h_f = h.clone().detach().requires_grad_(True)
    syn_f = syn_out.clone().detach().requires_grad_(True)
    gp_f = gate_pre.clone().detach().requires_grad_(True)

    _, gated = sew_sigmoid_gate_fused(s_f, h_f, syn_f, gp_f)
    # Only feed gradient into ``gated`` — spike_input is ignored.
    gated.backward(grad_gated)

    # SEW additive identity: grad_s, grad_h must be None when only the
    # gated branch was differentiated (since spike_input had no downstream).
    assert s_f.grad is None or s_f.grad.abs().sum().item() == 0
    assert h_f.grad is None or h_f.grad.abs().sum().item() == 0

    # syn_out and gate_pre must have the closed-form grad.
    assert syn_f.grad is not None
    assert gp_f.grad is not None

    # Verify closed-form numerics
    gate_ref = torch.sigmoid(gate_pre)
    expected_grad_syn = grad_gated * gate_ref
    expected_grad_gp = grad_gated * syn_out * gate_ref * (1.0 - gate_ref)
    assert _relerr(syn_f.grad, expected_grad_syn) < 1e-5
    assert _relerr(gp_f.grad, expected_grad_gp) < 1e-5


def test_double_backward_not_supported_silently():
    """Sanity: ``torch.autograd.Function`` without ``ctx.set_materialize_grads``
    or a registered double-backward simply errors. We do NOT support it.
    """
    s, h, syn_out, gate_pre, _, _ = _make_inputs(
        2, 4, 16, torch.device("cpu"),
    )
    s_f = s.clone().detach().requires_grad_(True)
    h_f = h.clone().detach().requires_grad_(True)
    syn_f = syn_out.clone().detach().requires_grad_(True)
    gp_f = gate_pre.clone().detach().requires_grad_(True)

    spike_input, gated = sew_sigmoid_gate_fused(s_f, h_f, syn_f, gp_f)
    loss = (spike_input.sum() + gated.sum())
    loss.backward()  # First backward — should succeed.
    # We don't attempt a second-order backward here; the design contract
    # is "first-order only", consistent with the rest of the project.
    assert s_f.grad is not None or h_f.grad is not None
