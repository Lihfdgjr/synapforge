"""Tests for synapforge.bio.* — bio-inspired components."""
from __future__ import annotations

import torch
import pytest

import synapforge as sf


# ---------------------------------------------------------------------------
# KWTA
# ---------------------------------------------------------------------------


def test_kwta_sparsity():
    kwta = sf.bio.KWTA(fraction=0.10)
    x = torch.randn(8, 256)
    y = kwta(x)
    nonzero = (y != 0).float().mean().item()
    # ~10% should survive (allow for ties).
    assert 0.05 <= nonzero <= 0.15, nonzero


def test_kwta_invalid_fraction():
    with pytest.raises(ValueError):
        sf.bio.KWTA(fraction=0.0)
    with pytest.raises(ValueError):
        sf.bio.KWTA(fraction=1.5)


def test_kwta_grad_flow():
    kwta = sf.bio.KWTA(fraction=0.20, straight_through=True)
    x = torch.randn(4, 32, requires_grad=True)
    y = kwta(x)
    y.sum().backward()
    assert x.grad is not None
    # All positions get gradient (straight-through).
    assert x.grad.abs().sum() > 0


def test_kwta_bf16():
    kwta = sf.bio.KWTA(0.10)
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    y = kwta(x)
    assert y.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# TauSplit / MultiBandTau
# ---------------------------------------------------------------------------


def test_tau_split():
    ts = sf.bio.TauSplit(64, plif_offset=0.8, cfc_offset=0.5)
    plif = ts.tau_plif()
    cfc = ts.tau_cfc()
    assert plif.shape == (64,)
    assert cfc.shape == (64,)
    assert torch.all(plif > cfc)  # plif_offset > cfc_offset
    plif2, cfc2 = ts()
    assert torch.allclose(plif, plif2)


def test_multi_band_tau_4_bands():
    mb = sf.bio.MultiBandTau(64, bands=("theta", "alpha", "beta", "gamma"))
    h = torch.randn(2, 5, 64)
    tau, weights = mb(h)
    assert tau.shape == (2, 5, 64)
    assert weights.shape == (2, 5, 4)
    # Weights sum to 1.
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
    # tau is positive and bounded.
    assert torch.all(tau >= sf.bio.MultiBandTau.TAU_MIN)
    assert torch.all(tau <= sf.bio.MultiBandTau.TAU_MAX)


def test_multi_band_tau_unknown_band():
    with pytest.raises(ValueError):
        sf.bio.MultiBandTau(32, bands=("xyz",))


# ---------------------------------------------------------------------------
# LearnableThreshold
# ---------------------------------------------------------------------------


def test_learnable_threshold_clamp():
    lt = sf.bio.LearnableThreshold(64, init=0.02, min_val=0.001, max_val=3.0)
    out = lt()
    assert out.shape == (64,)
    assert torch.all(out >= 0.001)
    assert torch.all(out <= 3.0)


def test_learnable_threshold_ema_update():
    lt = sf.bio.LearnableThreshold(32, init=0.5, ema_decay=0.9)
    lt.train()
    u = torch.randn(8, 32) * 2.0
    _ = lt(u)
    assert lt.u_abs_ema is not None
    assert lt.u_abs_ema.abs().sum() > 0


def test_learnable_threshold_calibrate():
    lt = sf.bio.LearnableThreshold(32, init=1.0, ema_decay=0.5)
    lt.train()
    # Drive EMA up.
    for _ in range(20):
        lt(torch.full((8, 32), 2.0))
    pre = lt().mean().item()
    lt.calibrate(scale=0.5)
    post = lt().mean().item()
    # Threshold should re-centre toward EMA*0.5.
    assert post != pre


# ---------------------------------------------------------------------------
# STDPFastWeight
# ---------------------------------------------------------------------------


def test_stdp_forward_shapes():
    stdp = sf.bio.STDPFastWeight(64)
    out = stdp(torch.randn(8, 64))
    assert out.shape == (8, 64)


def test_stdp_grad_no_buffer_inplace():
    """Critical: repeated forwards must not break autograd version checks."""
    stdp = sf.bio.STDPFastWeight(32)
    stdp.train()
    x = torch.randn(4, 32, requires_grad=True)
    y1 = stdp(x, spike=(x > 0).float())
    y2 = stdp(x, spike=(x > 0).float())
    (y1 + y2).sum().backward()
    assert x.grad is not None


def test_stdp_reset():
    stdp = sf.bio.STDPFastWeight(16)
    stdp.train()
    for _ in range(10):
        stdp(torch.randn(2, 16))
    assert stdp.W.abs().sum() > 0 or stdp.pre_trace.abs().sum() > 0
    stdp.reset()
    assert stdp.W.abs().sum() == 0
    assert stdp.pre_trace.abs().sum() == 0
    assert stdp.post_trace.abs().sum() == 0


def test_stdp_clip():
    stdp = sf.bio.STDPFastWeight(8, a_plus=10.0, a_minus=10.0, clip=0.5)
    stdp.train()
    for _ in range(50):
        stdp(torch.randn(4, 8))
    assert torch.all(stdp.W.abs() <= 0.5 + 1e-6)


# ---------------------------------------------------------------------------
# PredictiveCoding
# ---------------------------------------------------------------------------


def test_predictive_coding_residual_init():
    pc = sf.bio.PredictiveCoding(32, depth=1, residual=True)
    h = torch.randn(2, 5, 32)
    # With residual + zero-init, predicted next ~= h, so loss small.
    pred = pc.predict(h)
    assert torch.allclose(pred, h, atol=1e-5)


def test_predictive_coding_loss_grad():
    pc = sf.bio.PredictiveCoding(32, depth=2)
    h_cur = torch.randn(2, 5, 32, requires_grad=True)
    h_next = torch.randn(2, 5, 32)
    loss = pc(h_cur, h_next)
    loss.backward()
    # Predictor parameters must have gradient.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in pc.parameters())
    assert has_grad


# ---------------------------------------------------------------------------
# AstrocyteGate
# ---------------------------------------------------------------------------


def test_astrocyte_gate_shape_2d():
    ag = sf.bio.AstrocyteGate(64)
    y = ag(torch.randn(8, 64))
    assert y.shape == (8, 64)


def test_astrocyte_gate_shape_3d():
    ag = sf.bio.AstrocyteGate(32)
    y = ag(torch.randn(2, 5, 32))
    assert y.shape == (2, 5, 32)


def test_astrocyte_state_drift():
    ag = sf.bio.AstrocyteGate(16, tau=10.0)
    ag.reset()
    initial = ag.state.clone()
    # Pump with strongly biased input; state should drift.
    for _ in range(50):
        ag(torch.full((4, 16), 2.0))
    assert not torch.allclose(initial, ag.state)


def test_astrocyte_grad_flow():
    ag = sf.bio.AstrocyteGate(32)
    x = torch.randn(4, 32, requires_grad=True)
    y = ag(x)
    y.sum().backward()
    assert x.grad is not None
    # gate linear should have grad.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in ag.parameters() if p.requires_grad)
    assert has_grad
