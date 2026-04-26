"""Tests for sf.optim — pure BP, pure plasticity, mixed, and convergence."""
from __future__ import annotations

import torch

from synapforge.optim import (
    MultiSourceParam,
    Param,
    PlasticityAwareAdamW,
    build_optimizer,
)


def _device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Test 1: pure BP path equivalent to torch.optim.AdamW
# ----------------------------------------------------------------------------


def test_pure_bp_matches_torch_adamw():
    torch.manual_seed(0)
    dev = _device()
    d_in, d_out, n_steps = 16, 8, 50

    # Two identical models — one with sf.optim, one with torch.optim.AdamW.
    p_sf = torch.nn.Parameter(torch.randn(d_out, d_in, device=dev))
    p_to = torch.nn.Parameter(p_sf.detach().clone())

    msp = MultiSourceParam(p_sf, sources=["bp"])
    opt_sf = PlasticityAwareAdamW([msp], lr=1e-2, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.01)
    opt_to = torch.optim.AdamW([p_to], lr=1e-2, betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=0.01)

    target = torch.randn(d_out, d_in, device=dev)
    for _ in range(n_steps):
        # Same loss form on both — squared deviation from target
        opt_sf.zero_grad()
        opt_to.zero_grad()
        loss_sf = ((p_sf - target) ** 2).sum()
        loss_to = ((p_to - target) ** 2).sum()
        loss_sf.backward()
        loss_to.backward()
        opt_sf.step()
        opt_to.step()

    rel = (p_sf - p_to).norm() / p_to.norm().clamp_min(1e-9)
    assert rel.item() < 1e-5, f"pure-BP mismatch rel_err={rel.item():.3e}"
    print(f"[T1] pure-BP rel_err vs torch.optim.AdamW = {rel.item():.3e}  OK")


# ----------------------------------------------------------------------------
# Test 2: pure plasticity path (no BP) — params drift by attached delta only
# ----------------------------------------------------------------------------


def test_pure_plasticity_no_bp():
    torch.manual_seed(1)
    dev = _device()
    p = torch.nn.Parameter(torch.zeros(4, 4, device=dev))
    msp = MultiSourceParam(p, sources=["stdp"], weight_per_source={"stdp": 0.5})
    opt = PlasticityAwareAdamW([msp], lr=1e-2, weight_decay=0.0)

    delta = torch.full_like(p.data, 1.0)

    for _ in range(10):
        # No backward — pure plasticity
        msp.attach_plast_delta("stdp", delta)
        opt.step()

    # AdamW with constant +ΔW push: param should march POSITIVE consistently.
    # Adam's normalization makes per-step magnitude ≈ lr (constant grad), so
    # after 10 steps p ≈ +0.1 (lr=1e-2 × 10). Sign and order-of-magnitude check.
    mean_val = p.data.mean().item()
    assert mean_val > 0.05, f"plasticity push gave mean={mean_val:.4f}, expected >0.05"
    assert torch.isfinite(p.data).all(), "NaN/Inf in pure-plasticity update"
    print(f"[T2] pure-plast mean={mean_val:.4f}  OK")


# ----------------------------------------------------------------------------
# Test 3: mixed BP+STDP — both contribute, no NaN
# ----------------------------------------------------------------------------


def test_mixed_bp_stdp_no_nan():
    torch.manual_seed(2)
    dev = _device()
    p = torch.nn.Parameter(torch.randn(8, 8, device=dev))
    msp = MultiSourceParam(
        p, sources=["bp", "stdp"],
        weight_per_source={"bp": 1.0, "stdp": 0.1},
    )
    opt = PlasticityAwareAdamW([msp], lr=1e-3, weight_decay=0.01)

    target = torch.zeros_like(p.data)
    for _ in range(20):
        opt.zero_grad()
        loss = ((p - target) ** 2).sum()
        loss.backward()
        # STDP delta — small random push
        delta = 0.01 * torch.randn_like(p.data)
        msp.attach_plast_delta("stdp", delta)
        opt.step()

    assert torch.isfinite(p.data).all(), "NaN/Inf in mixed BP+STDP update"
    # BP dominates (w=1.0 vs 0.1) so loss should still go down overall
    final_loss = ((p - target) ** 2).sum().item()
    init_loss = (target * target).sum().item() + (p.shape[0] * p.shape[1])  # ~64
    assert final_loss < 60.0, f"mixed BP+STDP failed to descend, final={final_loss:.2f}"
    print(f"[T3] mixed BP+STDP final_loss={final_loss:.2f}  OK")


# ----------------------------------------------------------------------------
# Test 4: 1000-step toy regression with mixed sources, monotonic-ish loss drop
# ----------------------------------------------------------------------------


def test_toy_regression_converges():
    torch.manual_seed(3)
    dev = _device()
    d = 16
    # Build a tiny 2-layer model. Layer-1 is BP-only, layer-2 is BP+STDP.
    W1 = torch.nn.Parameter(torch.randn(d, d, device=dev) * 0.1)
    W2 = Param(
        torch.randn(d, d, device=dev) * 0.1,
        grad_source=["bp", "stdp"],
        weight_per_source={"bp": 1.0, "stdp": 0.05},
    )
    model = torch.nn.ParameterDict({"W1": W1, "W2": W2})

    opt = build_optimizer(model, lr=3e-3, weight_decay=0.0)
    msp_W2 = opt.get_ms_param(W2)
    assert msp_W2 is not None
    assert msp_W2.sources == ["bp", "stdp"]

    # Synthetic target: y = X @ A
    X = torch.randn(64, d, device=dev)
    A = torch.randn(d, d, device=dev)
    y = X @ A

    losses = []
    for step in range(1000):
        opt.zero_grad()
        h = X @ model["W1"]
        h = torch.tanh(h)
        out = h @ model["W2"]
        loss = ((out - y) ** 2).mean()
        loss.backward()

        # STDP-style update: attach a small delta toward (h.T @ out_signed)
        with torch.no_grad():
            stdp_delta = 0.001 * (h.t() @ torch.sign(out)) / X.shape[0]
            msp_W2.attach_plast_delta("stdp", stdp_delta)

        opt.step()
        losses.append(loss.item())

    init_loss, final_loss = losses[0], losses[-1]
    assert final_loss < init_loss * 0.5, (
        f"toy regression did NOT converge: init={init_loss:.3f} "
        f"final={final_loss:.3f}"
    )
    # Check long-term trend: average of first 50 vs last 50 steps
    head = sum(losses[:50]) / 50
    tail = sum(losses[-50:]) / 50
    assert tail < head * 0.5, f"loss didn't drop enough: head={head:.3f} tail={tail:.3f}"
    print(f"[T4] toy regression: {init_loss:.3f} → {final_loss:.3f} "
          f"(head_mean={head:.3f} tail_mean={tail:.3f})  OK")


# ----------------------------------------------------------------------------
# Test 5: bonus — accumulate same source twice (sums), reset clears, hook unhook
# ----------------------------------------------------------------------------


def test_accumulation_and_reset():
    dev = _device()
    p = torch.nn.Parameter(torch.zeros(3, 3, device=dev))
    msp = MultiSourceParam(p, sources=["bp", "stdp"])

    d1 = torch.full_like(p.data, 0.1)
    d2 = torch.full_like(p.data, 0.2)
    msp.attach_plast_delta("stdp", d1)
    msp.attach_plast_delta("stdp", d2)
    assert torch.allclose(msp.plast_delta["stdp"], d1 + d2)

    msp.reset()
    assert msp._bp_grad_cached is None
    assert "stdp" not in msp.plast_delta
    print("[T5] accumulation + reset  OK")


# ----------------------------------------------------------------------------
# Test 6: bonus — undeclared source raises
# ----------------------------------------------------------------------------


def test_undeclared_source_raises():
    dev = _device()
    p = torch.nn.Parameter(torch.zeros(2, 2, device=dev))
    msp = MultiSourceParam(p, sources=["bp"])
    try:
        msp.attach_plast_delta("hebb", torch.zeros_like(p.data))
    except KeyError:
        print("[T6] undeclared-source KeyError  OK")
        return
    raise AssertionError("expected KeyError on undeclared source")


# ----------------------------------------------------------------------------
# Test 7: bonus — NaN delta is filtered, doesn't poison Adam state
# ----------------------------------------------------------------------------


def test_nan_filter():
    dev = _device()
    p = torch.nn.Parameter(torch.ones(3, 3, device=dev))
    msp = MultiSourceParam(p, sources=["bp", "stdp"])
    opt = PlasticityAwareAdamW([msp], lr=1e-2, weight_decay=0.0)

    bad = torch.full_like(p.data, float("nan"))
    msp.attach_plast_delta("stdp", bad)
    opt.step()
    assert torch.isfinite(p.data).all(), "NaN poisoned param"
    # Adam state should NOT have been initialized (we skipped step before that)
    assert "m" not in opt.state[p] or torch.isfinite(opt.state[p]["m"]).all()
    print("[T7] NaN filter  OK")


if __name__ == "__main__":
    test_pure_bp_matches_torch_adamw()
    test_pure_plasticity_no_bp()
    test_mixed_bp_stdp_no_nan()
    test_toy_regression_converges()
    test_accumulation_and_reset()
    test_undeclared_source_raises()
    test_nan_filter()
    print("\nALL TESTS PASSED")
