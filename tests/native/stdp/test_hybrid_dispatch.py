"""Tests for HybridOptimizerDispatcher routing logic.

Verifies:
* params tagged with only-plasticity sources go to STDPOnlyOptimizer
* params with BP (or default no-tag) go to the base AdamW
* counts on each side line up with the original tagging
* checkpoint round-trip works
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from synapforge.native.stdp import (
    HybridOptimizerDispatcher,
    STDPOnlyOptimizer,
)


# Skip if torch unavailable — we exercise both halves of the dispatcher
# (STDP-only and BP+plasticity routing). The BP half needs torch.
torch = pytest.importorskip("torch")


def _make_model_with_mixed_tags():
    """Tiny model: 1 plasticity-only weight, 1 BP weight, 1 mixed weight.

    Returns a SimpleNamespace standing in for nn.Module that exposes
    ``named_parameters()``.
    """
    bp_w = torch.nn.Parameter(torch.zeros(4, 8))
    bp_w._sf_grad_source = ["bp"]

    plast_w = torch.nn.Parameter(torch.zeros(4, 8))
    plast_w._sf_grad_source = ["stdp"]

    mixed_w = torch.nn.Parameter(torch.zeros(4, 8))
    mixed_w._sf_grad_source = ["bp", "stdp"]

    params = [
        ("bp_layer.w", bp_w),
        ("plast_layer.w", plast_w),
        ("mixed_layer.w", mixed_w),
    ]

    class _M:
        def named_parameters(self_inner):
            return iter(params)

    return _M(), bp_w, plast_w, mixed_w


def test_dispatcher_routes_correctly():
    model, bp_w, plast_w, mixed_w = _make_model_with_mixed_tags()
    factory = lambda ps: torch.optim.AdamW(ps, lr=1e-3)
    disp = HybridOptimizerDispatcher.from_model(model, factory)
    # Plasticity-only -> STDP
    assert disp.stdp is not None
    assert "plast_layer.w" in disp.stdp.names()
    # BP and mixed -> base AdamW
    assert disp.base is not None
    base_param_ids = {id(p) for g in disp.base.param_groups for p in g["params"]}
    assert id(bp_w) in base_param_ids
    assert id(mixed_w) in base_param_ids
    # And NOT in STDP
    stdp_param_ids = {id(g.param) for g in disp.stdp.groups}
    assert id(bp_w) not in stdp_param_ids
    assert id(mixed_w) not in stdp_param_ids


def test_dispatcher_counts():
    model, *_ = _make_model_with_mixed_tags()
    factory = lambda ps: torch.optim.AdamW(ps, lr=1e-3)
    disp = HybridOptimizerDispatcher.from_model(model, factory)
    # 3 weights, each 4*8=32. STDP gets 32, base gets 64.
    assert disp.stdp_param_count() == 32
    assert disp.base_param_count() == 64


def test_dispatcher_step_does_not_raise():
    """Smoke: build, observe, step, no exception."""
    model, bp_w, plast_w, mixed_w = _make_model_with_mixed_tags()
    factory = lambda ps: torch.optim.AdamW(ps, lr=1e-3)
    # Asymmetric a_plus > a_minus so co-firing causes net LTP and the
    # weight visibly moves on a single step.
    disp = HybridOptimizerDispatcher.from_model(
        model, factory, base_alpha=0.05, a_plus=0.05, a_minus=0.0
    )
    # Fake a BP grad on bp/mixed (else AdamW step is a noop)
    bp_w.grad = torch.randn_like(bp_w)
    mixed_w.grad = torch.randn_like(mixed_w)
    # Observe a spike pair on the STDP layer (sparse so LTP > LTD)
    pre = np.zeros(8, dtype=np.uint8)
    post = np.zeros(4, dtype=np.uint8)
    pre[0] = 1
    post[0] = 1
    disp.observe_spike("plast_layer.w", pre, post)
    out = disp.step()
    assert "stdp_stats" in out
    assert out["stdp_stats"]["total_groups"] == 1
    # Plast weight moved (was zero, observation+step should grow it)
    assert plast_w.detach().abs().sum().item() > 0


def test_dispatcher_state_dict_roundtrip():
    model, *_ = _make_model_with_mixed_tags()
    factory = lambda ps: torch.optim.AdamW(ps, lr=1e-3)
    disp = HybridOptimizerDispatcher.from_model(model, factory)
    # Run a step to seed state
    disp.observe_spike(
        "plast_layer.w",
        np.ones(8, dtype=np.uint8),
        np.ones(4, dtype=np.uint8),
    )
    disp.step()
    sd = disp.state_dict()
    # New dispatcher loads state
    model2, *_ = _make_model_with_mixed_tags()
    disp2 = HybridOptimizerDispatcher.from_model(model2, factory)
    disp2.load_state_dict(sd)
    assert disp2.stdp._step_count == disp.stdp._step_count


def test_only_plasticity_no_base():
    """Model with only plasticity-only weights yields no base optimizer."""
    plast_w = torch.nn.Parameter(torch.zeros(4, 8))
    plast_w._sf_grad_source = ["stdp"]

    class _M:
        def named_parameters(self_inner):
            return iter([("only.w", plast_w)])

    factory = lambda ps: torch.optim.AdamW(ps, lr=1e-3)
    disp = HybridOptimizerDispatcher.from_model(_M(), factory)
    assert disp.base is None
    assert disp.stdp is not None
    # zero_grad is a noop without base
    disp.zero_grad()
    assert disp.base_param_count() == 0


def test_only_bp_no_stdp():
    """Model without plasticity tags yields no STDP optimizer."""
    bp_w = torch.nn.Parameter(torch.zeros(4, 8))
    bp_w._sf_grad_source = ["bp"]

    class _M:
        def named_parameters(self_inner):
            return iter([("only.w", bp_w)])

    factory = lambda ps: torch.optim.AdamW(ps, lr=1e-3)
    disp = HybridOptimizerDispatcher.from_model(_M(), factory)
    assert disp.stdp is None
    assert disp.base is not None
    assert disp.stdp_param_count() == 0
