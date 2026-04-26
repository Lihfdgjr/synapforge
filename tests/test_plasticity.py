"""Unit tests for sf.plasticity. Standalone (no pytest required).

Run:
    CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python /workspace/synapforge/test_plasticity.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Make synapforge importable when run as script.

import torch

from synapforge.plasticity import (
    BCM,
    Hebbian,
    HebbianPlasticity,
    PlasticityEngine,
    PlasticityRule,
    STDP,
    SynaptogenesisGrowPrune,
)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Hebbian
# ---------------------------------------------------------------------------


def test_hebbian_co_firing_positive() -> None:
    """Pre and post both positive → positive delta (co-firing)."""
    dev = _device()
    rule = Hebbian(lr=1.0).to(dev)
    pre = torch.ones(8, 4, device=dev)            # all-on pre
    post = torch.ones(8, 4, device=dev) * 0.5     # weakly active post
    rule.observe(pre=pre, post=post, t=0.0)
    delta = rule.compute_delta_W()
    assert delta is not None, "Hebbian.compute_delta_W returned None on observed input"
    assert delta.shape == (4, 4), f"shape mismatch: {tuple(delta.shape)}"
    assert delta.min().item() > 0, "expected all-positive delta for co-firing pre/post"
    print(f"  Hebbian co-firing: delta.min={delta.min().item():.4f} max={delta.max().item():.4f}")


def test_hebbian_drains_pending() -> None:
    dev = _device()
    rule = Hebbian(lr=1e-3).to(dev)
    rule.observe(pre=torch.ones(2, 3, device=dev), post=torch.ones(2, 3, device=dev))
    _ = rule.compute_delta_W()
    # Second call should return None (pending is drained).
    assert rule.compute_delta_W() is None
    print("  Hebbian drain: ok")


# ---------------------------------------------------------------------------
# STDP — biological sign convention
# ---------------------------------------------------------------------------


def test_stdp_pre_before_post_potentiation() -> None:
    """Pre-spike at t=0..4 then post-spike at t=5 → POSITIVE delta (LTP)."""
    dev = _device()
    rule = STDP(tau_pre=20.0, tau_post=20.0, lr=1.0, decay_after_compute=1.0).to(dev)
    D = 4
    pre_spike = torch.zeros(1, D, device=dev)
    post_spike = torch.zeros(1, D, device=dev)
    pre_one = torch.ones(1, D, device=dev)
    post_one = torch.ones(1, D, device=dev)
    # Pre fires first for 5 steps (post_trace stays at 0).
    for _ in range(5):
        rule.observe(pre=pre_one, post=post_spike, t=1.0)
    # Then post fires (pre_trace is high and decaying).
    rule.observe(pre=pre_spike, post=post_one, t=1.0)
    delta = rule.compute_delta_W()
    assert delta is not None, "STDP delta is None"
    # Sign: ltp = a_plus * outer(post_one, pre_trace_high) > 0
    #       ltd = a_minus * outer(pre_zero, post_trace_low) = 0
    # so delta is positive overall.
    assert delta.min().item() >= 0 - 1e-6, f"expected non-negative LTP, got min={delta.min():.4f}"
    assert delta.mean().item() > 0, f"expected positive LTP mean, got {delta.mean():.4f}"
    print(f"  STDP pre-before-post (LTP): delta.mean={delta.mean().item():.4f}")


def test_stdp_post_before_pre_depression() -> None:
    """Post-spike at t=0..4 then pre-spike at t=5 → NEGATIVE delta (LTD)."""
    dev = _device()
    rule = STDP(tau_pre=20.0, tau_post=20.0, lr=1.0, decay_after_compute=1.0).to(dev)
    D = 4
    zero = torch.zeros(1, D, device=dev)
    one = torch.ones(1, D, device=dev)
    for _ in range(5):
        rule.observe(pre=zero, post=one, t=1.0)
    rule.observe(pre=one, post=zero, t=1.0)
    delta = rule.compute_delta_W()
    assert delta is not None
    # ltp = a_plus * outer(post_zero, pre_trace_low) = 0
    # ltd = a_minus * outer(pre_one, post_trace_high) > 0
    # net delta = -(positive) < 0
    assert delta.mean().item() < 0, f"expected negative LTD, got {delta.mean():.4f}"
    print(f"  STDP post-before-pre (LTD): delta.mean={delta.mean().item():.4f}")


def test_stdp_legacy_callable_form() -> None:
    """STDP(dim=D) gives a forward(pre,post) returning (B,D) modulation."""
    dev = _device()
    D = 8
    rule = STDP(dim=D, lr=1e-3).to(dev)
    assert rule.W_fast is not None and rule.W_fast.shape == (D, D)
    pre = torch.randn(4, D, device=dev)
    post = torch.randn(4, D, device=dev)
    out = rule(pre, post)
    assert out.shape == (4, D)
    # W_fast should now be non-zero.
    assert rule.W_fast.norm().item() > 0
    print(f"  STDP legacy form: W_fast.norm={rule.W_fast.norm().item():.4f}, out.shape={tuple(out.shape)}")


# ---------------------------------------------------------------------------
# BCM
# ---------------------------------------------------------------------------


def test_bcm_high_post_potentiates() -> None:
    """post >> theta → LTP (positive delta)."""
    dev = _device()
    rule = BCM(theta_init=0.1, lr=1.0).to(dev)  # low theta
    pre = torch.ones(8, 4, device=dev)
    post = torch.ones(8, 4, device=dev) * 2.0   # post >> theta
    rule.observe(pre=pre, post=post)
    delta = rule.compute_delta_W()
    assert delta is not None
    assert delta.min().item() > 0
    print(f"  BCM high post (LTP): delta.mean={delta.mean().item():.4f}")


def test_bcm_low_post_depresses() -> None:
    """0 < post < theta → LTD (negative delta)."""
    dev = _device()
    rule = BCM(theta_init=2.0, lr=1.0).to(dev)
    pre = torch.ones(8, 4, device=dev)
    post = torch.ones(8, 4, device=dev) * 0.5
    rule.observe(pre=pre, post=post)
    delta = rule.compute_delta_W()
    assert delta is not None
    assert delta.max().item() < 0
    print(f"  BCM low post (LTD): delta.mean={delta.mean().item():.4f}")


# ---------------------------------------------------------------------------
# SynaptogenesisGrowPrune
# ---------------------------------------------------------------------------


def test_synaptogenesis_returns_signed_mask_delta() -> None:
    dev = _device()
    rule = SynaptogenesisGrowPrune(
        target_density=0.5,
        growth_check_every=1,
        prune_check_every=1,
    ).to(dev)
    pre = torch.ones(2, 4, device=dev)
    post = torch.ones(2, 4, device=dev)
    rule.observe(pre=pre, post=post)
    mask = torch.zeros(4, 4, dtype=torch.bool, device=dev)  # all inactive → wants growth
    W = torch.randn(4, 4, device=dev)
    md = rule.compute_mask_delta(mask, W)
    assert md is not None
    assert md.dtype == torch.int8
    assert md.max().item() == 1, "expected +1 grow entries"
    print(f"  SynaptogenesisGrowPrune: grow_count={int((md == 1).sum().item())}, dtype={md.dtype}")


# ---------------------------------------------------------------------------
# PlasticityEngine
# ---------------------------------------------------------------------------


def test_engine_step_three_rules() -> None:
    """Engine collects deltas from STDP, Hebbian, BCM in one step."""
    dev = _device()
    D = 4
    rules = {
        "w_stdp": STDP(lr=1.0, decay_after_compute=1.0).to(dev),
        "w_hebb": Hebbian(lr=1.0).to(dev),
        "w_bcm": BCM(theta_init=0.1, lr=1.0).to(dev),
    }
    engine = PlasticityEngine(rules, schedule="every:1")
    pre_strong = torch.ones(2, D, device=dev)
    post_weak = torch.ones(2, D, device=dev) * 0.5
    # For STDP: pre fires first (build trace) then post — so observe a
    # pre-only step then a post-mostly step. Otherwise ltp = ltd and delta = 0.
    rules["w_hebb"].observe(pre=pre_strong, post=post_weak, t=1.0)
    rules["w_bcm"].observe(pre=pre_strong, post=post_weak, t=1.0)
    rules["w_stdp"].observe(pre=pre_strong, post=torch.zeros(2, D, device=dev), t=1.0)
    rules["w_stdp"].observe(pre=torch.zeros(2, D, device=dev), post=pre_strong, t=1.0)

    weights = {
        "w_stdp": torch.zeros(D, D, device=dev),
        "w_hebb": torch.zeros(D, D, device=dev),
        "w_bcm": torch.zeros(D, D, device=dev),
    }
    deltas = engine.step(t=0, weight_dict=weights)
    assert set(deltas.keys()) == {"w_stdp", "w_hebb", "w_bcm"}
    for name, d in deltas.items():
        assert d.shape == (D, D), f"{name}: {d.shape}"
        assert d.norm().item() > 0, f"{name}: delta is zero"
    print(f"  Engine step: {len(deltas)} deltas (norms: {[(n, round(d.norm().item(), 4)) for n, d in deltas.items()]})")
    engine.apply(deltas, weights)
    for name, w in weights.items():
        assert w.norm().item() > 0, f"{name}: weight unchanged after apply"
    print("  Engine apply: all weights mutated as expected")


def test_engine_schedule_skipping() -> None:
    """schedule='every:3' skips 2 of 3 calls."""
    dev = _device()
    rules = {"a": Hebbian(lr=1.0).to(dev)}
    engine = PlasticityEngine(rules, schedule="every:3")
    p = torch.ones(1, 2, device=dev)
    rules["a"].observe(pre=p, post=p)
    weights = {"a": torch.zeros(2, 2, device=dev)}
    out1 = engine.step(0, weights)  # _step=1, 1%3 != 0 -> skip
    assert out1 == {}, f"expected skip on step 1, got {out1}"
    rules["a"].observe(pre=p, post=p)
    out2 = engine.step(0, weights)  # _step=2, skip
    assert out2 == {}
    rules["a"].observe(pre=p, post=p)
    out3 = engine.step(0, weights)  # _step=3, fire
    assert "a" in out3
    print("  Engine schedule every:3: ok")


def test_engine_rejects_non_rule() -> None:
    try:
        PlasticityEngine({"a": torch.nn.Linear(4, 4)})  # type: ignore[arg-type]
    except TypeError as exc:
        assert "PlasticityRule" in str(exc)
        print("  Engine type guard: ok")
        return
    raise AssertionError("expected TypeError for non-PlasticityRule")


# ---------------------------------------------------------------------------
# Cross-source merge — both BP grad AND plasticity touch same weight
# ---------------------------------------------------------------------------


def test_merge_bp_and_plasticity_no_version_conflict() -> None:
    """The KEY test. Mutate W with BP-grad-derived delta AND plasticity delta
    in the same step. The deferred-delta model means NO autograd version
    error even though we add to W in-place after backward."""
    dev = _device()
    D = 4
    W = torch.randn(D, D, device=dev, requires_grad=True)
    rule = STDP(lr=0.01, decay_after_compute=1.0).to(dev)
    engine = PlasticityEngine({"W": rule})

    pre = torch.randn(8, D, device=dev)
    post_bp = torch.tanh(pre @ W.t())
    loss = post_bp.pow(2).mean()
    loss.backward()
    # BP grad is now in W.grad. Now do plasticity update in same step.
    rule.observe(pre=pre.detach(), post=post_bp.detach(), t=1.0)
    deltas = engine.step(t=0, weight_dict={"W": W})
    # Apply BOTH: BP grad first, then plasticity delta.
    with torch.no_grad():
        W.add_(-0.01 * W.grad)         # vanilla SGD step
        engine.apply(deltas, {"W": W}) # plasticity step
    print(f"  Merge BP+plasticity: W.norm={W.norm().item():.4f}, no version conflict")


# ---------------------------------------------------------------------------
# Legacy alias compatibility
# ---------------------------------------------------------------------------


def test_legacy_hebbian_plasticity_unchanged() -> None:
    rule = HebbianPlasticity(dim=4, eta=0.01)
    pre = torch.ones(2, 4)
    post = torch.ones(2, 4) * 0.5
    out = rule(pre, post)
    assert out.shape == (2, 4)
    assert rule.W_fast.norm().item() > 0
    print("  Legacy HebbianPlasticity: ok")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


TESTS = [
    test_hebbian_co_firing_positive,
    test_hebbian_drains_pending,
    test_stdp_pre_before_post_potentiation,
    test_stdp_post_before_pre_depression,
    test_stdp_legacy_callable_form,
    test_bcm_high_post_potentiates,
    test_bcm_low_post_depresses,
    test_synaptogenesis_returns_signed_mask_delta,
    test_engine_step_three_rules,
    test_engine_schedule_skipping,
    test_engine_rejects_non_rule,
    test_merge_bp_and_plasticity_no_version_conflict,
    test_legacy_hebbian_plasticity_unchanged,
]


def main() -> int:
    print(f"sf.plasticity unit tests on {_device()}")
    failed = 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    total = len(TESTS)
    print(f"\nResult: {total - failed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
