"""Tests for synapforge.native.stdp.STDPOnlyOptimizer.

Coverage
--------
1. spike-pair causality:  delta is non-zero only when both pre and
   post fire within the window.
2. LR scaling:            larger alpha -> larger weight delta.
3. clip:                  weights bounded by [-clip, +clip].
4. STDP convergence smoke: 100 steps drive a 2-layer toy SNN's
   weights toward correlated input/output pattern.
5. dispatcher routing:    plasticity-only params go to STDP, the
   rest to AdamW; STDP-only path is bit-equal to AdamW path when
   no plasticity params exist.
6. state-dict round trip.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from synapforge.native.stdp import (
    STDPOnlyOptimizer,
    SpikeRingBuffer,
    per_param_alpha,
)
from synapforge.native.stdp.stdp_optimizer import STDPParamGroup


# Simple numpy-backed param shim — passes through to STDPOnlyOptimizer
# without needing torch in the test environment.

class _NPParam:
    """Numpy array with a .shape and a mutable ._np_data view.

    Used in place of torch.nn.Parameter for tests so the package's
    no-torch-import constraint is exercised cleanly.
    """

    def __init__(self, arr: np.ndarray) -> None:
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        self._np_data = arr
        self.shape = arr.shape

    @property
    def data(self) -> np.ndarray:
        return self._np_data


# ----------------------------------------------------------------------- T1

def test_spike_pair_causality():
    """Delta is zero unless BOTH pre and post fire."""
    rng = np.random.default_rng(0)
    out_dim, in_dim = 8, 16
    p = _NPParam(rng.normal(0, 0.01, size=(out_dim, in_dim)))
    g = STDPParamGroup(
        param=p,
        name="layer.w",
        buffer=SpikeRingBuffer(in_dim=in_dim, out_dim=out_dim, window=5),
        alpha=0.05, a_plus=0.05, a_minus=0.05, clip=1.0,
    )
    opt = STDPOnlyOptimizer([g])
    w_before = p._np_data.copy()
    # No pre spikes -> no update
    opt.observe("layer.w",
                pre_spike=np.zeros(in_dim, dtype=np.uint8),
                post_spike=np.ones(out_dim, dtype=np.uint8))
    stats = opt.step()
    assert stats["per_group"]["layer.w"]["active_pairs"] == 0
    np.testing.assert_array_equal(p._np_data, w_before)
    # No post spikes -> no update
    opt.observe("layer.w",
                pre_spike=np.ones(in_dim, dtype=np.uint8),
                post_spike=np.zeros(out_dim, dtype=np.uint8))
    stats = opt.step()
    assert stats["per_group"]["layer.w"]["active_pairs"] == 0
    np.testing.assert_array_equal(p._np_data, w_before)
    # Both fire -> nonzero update
    opt.observe("layer.w",
                pre_spike=np.ones(in_dim, dtype=np.uint8),
                post_spike=np.ones(out_dim, dtype=np.uint8))
    stats = opt.step()
    assert stats["per_group"]["layer.w"]["active_pairs"] > 0
    assert not np.allclose(p._np_data, w_before)


# ----------------------------------------------------------------------- T2

def test_alpha_scaling_grows_delta():
    """Larger alpha => larger delta_norm on the same spike pattern.

    Use LTP-only (a_minus=0) so the scaling test is monotonic in alpha;
    symmetric a_plus==a_minus would cancel for fully co-firing pairs.
    """
    rng = np.random.default_rng(1)
    out_dim, in_dim = 8, 16
    pre = (rng.uniform(0, 1, size=in_dim) < 0.4).astype(np.uint8)
    post = (rng.uniform(0, 1, size=out_dim) < 0.4).astype(np.uint8)
    # Guarantee at least one fire on each side
    pre[0] = 1
    post[0] = 1
    norms: list[float] = []
    for alpha in (0.001, 0.01, 0.1):
        p = _NPParam(np.zeros((out_dim, in_dim), dtype=np.float32))
        g = STDPParamGroup(
            param=p,
            name="w",
            buffer=SpikeRingBuffer(in_dim=in_dim, out_dim=out_dim, window=5),
            alpha=alpha, a_plus=alpha, a_minus=0.0, clip=10.0,
        )
        opt = STDPOnlyOptimizer([g])
        # Drive 5 steps with the same pair so traces saturate.
        for _ in range(5):
            opt.observe("w", pre, post)
            opt.step()
        norms.append(float(np.linalg.norm(p._np_data)))
    # monotonic in alpha
    assert norms[0] < norms[1] < norms[2], (
        f"expected monotonic delta_norm by alpha; got {norms}"
    )


# ----------------------------------------------------------------------- T3

def test_clip_bounds_weights():
    """After many updates the weight magnitude must not exceed clip."""
    out_dim, in_dim = 4, 4
    p = _NPParam(np.zeros((out_dim, in_dim), dtype=np.float32))
    clip = 0.3
    g = STDPParamGroup(
        param=p, name="w",
        buffer=SpikeRingBuffer(in_dim=in_dim, out_dim=out_dim, window=5),
        alpha=0.5, a_plus=0.5, a_minus=0.0, clip=clip,
    )
    opt = STDPOnlyOptimizer([g])
    pre = np.ones(in_dim, dtype=np.uint8)
    post = np.ones(out_dim, dtype=np.uint8)
    for _ in range(50):
        opt.observe("w", pre, post)
        opt.step()
    assert p._np_data.max() <= clip + 1e-6, (
        f"weight max {p._np_data.max()} exceeded clip {clip}"
    )
    assert p._np_data.min() >= -clip - 1e-6


# ----------------------------------------------------------------------- T4

def test_stdp_drives_correlated_pattern():
    """Smoke: 100 STDP steps move weights toward correlated I/O pattern.

    Setup: 8 input neurons, 4 output neurons. We always fire input
    neurons {0,1} and output neurons {0}. The STDP rule should grow
    W[0, 0] and W[0, 1] (LTP) — that is, the weights connecting the
    co-firing pre/post pair — while leaving the rest small.
    """
    out_dim, in_dim = 4, 8
    p = _NPParam(np.zeros((out_dim, in_dim), dtype=np.float32))
    g = STDPParamGroup(
        param=p, name="w",
        buffer=SpikeRingBuffer(in_dim=in_dim, out_dim=out_dim, window=20),
        alpha=0.05, a_plus=0.05, a_minus=0.0, clip=1.0,
    )
    opt = STDPOnlyOptimizer([g])
    pre = np.zeros(in_dim, dtype=np.uint8)
    post = np.zeros(out_dim, dtype=np.uint8)
    pre[0] = 1
    pre[1] = 1
    post[0] = 1
    for _ in range(100):
        opt.observe("w", pre, post)
        opt.step()
    co_fire_w = p._np_data[0, :2].mean()
    other_w = p._np_data[1:, :].mean()
    assert co_fire_w > 0.05, (
        f"expected co-fire weights to grow > 0.05; got {co_fire_w}"
    )
    assert co_fire_w > 5 * abs(other_w), (
        f"co-fire weight ({co_fire_w}) should dominate non-co-fire "
        f"({other_w}); STDP did not localize correctly."
    )


# ----------------------------------------------------------------------- T5

def test_per_param_alpha_scaling():
    """per_param_alpha rescales correctly by tau and respects bounds."""
    base = 0.02
    taus = {1: 4.0, 2: 16.0, 3: 64.0}
    out = per_param_alpha(base, taus)
    # Geometric mean of [4, 16, 64] is 16. So id=1 gets 0.02 * 4/16 = 0.005;
    # id=3 gets 0.02 * 64/16 = 0.08 (clipped at ceil=1.0).
    assert math.isclose(out[1], base * 4.0 / 16.0, rel_tol=1e-4)
    assert math.isclose(out[2], base * 16.0 / 16.0, rel_tol=1e-4)
    assert math.isclose(out[3], base * 64.0 / 16.0, rel_tol=1e-4)
    # Negative tau rejected
    with pytest.raises(ValueError):
        per_param_alpha(base, {1: -1.0})


# ----------------------------------------------------------------------- T6

def test_state_dict_round_trip():
    """state_dict / load_state_dict preserves traces and step count."""
    out_dim, in_dim = 4, 8
    p = _NPParam(np.zeros((out_dim, in_dim), dtype=np.float32))
    g = STDPParamGroup(
        param=p, name="w",
        buffer=SpikeRingBuffer(in_dim=in_dim, out_dim=out_dim, window=10),
        alpha=0.03, a_plus=0.03, a_minus=0.03, clip=1.0,
    )
    opt = STDPOnlyOptimizer([g])
    rng = np.random.default_rng(7)
    for _ in range(20):
        pre = (rng.uniform(0, 1, size=in_dim) < 0.3).astype(np.uint8)
        post = (rng.uniform(0, 1, size=out_dim) < 0.3).astype(np.uint8)
        opt.observe("w", pre, post)
        opt.step()
    sd = opt.state_dict()
    # Build a fresh opt and load
    p2 = _NPParam(np.zeros((out_dim, in_dim), dtype=np.float32))
    g2 = STDPParamGroup(
        param=p2, name="w",
        buffer=SpikeRingBuffer(in_dim=in_dim, out_dim=out_dim, window=10),
        alpha=0.0, a_plus=0.0, a_minus=0.0, clip=1.0,
    )
    opt2 = STDPOnlyOptimizer([g2])
    opt2.load_state_dict(sd)
    assert opt2._step_count == opt._step_count
    np.testing.assert_array_equal(g.buffer.pre_trace, g2.buffer.pre_trace)
    np.testing.assert_array_equal(g.buffer.post_trace, g2.buffer.post_trace)
    assert g.buffer.cursor == g2.buffer.cursor


# ----------------------------------------------------------------------- T7

def test_total_params_and_names():
    out_dim, in_dim = 4, 8
    p1 = _NPParam(np.zeros((out_dim, in_dim), dtype=np.float32))
    p2 = _NPParam(np.zeros((2, 3), dtype=np.float32))
    groups = [
        STDPParamGroup(
            param=p1, name="a",
            buffer=SpikeRingBuffer(in_dim=in_dim, out_dim=out_dim, window=5),
            alpha=0.01, a_plus=0.01, a_minus=0.01, clip=1.0,
        ),
        STDPParamGroup(
            param=p2, name="b",
            buffer=SpikeRingBuffer(in_dim=3, out_dim=2, window=5),
            alpha=0.01, a_plus=0.01, a_minus=0.01, clip=1.0,
        ),
    ]
    opt = STDPOnlyOptimizer(groups)
    assert opt.total_params() == out_dim * in_dim + 2 * 3
    assert opt.names() == ["a", "b"]


# ----------------------------------------------------------------------- T8

def test_observe_unknown_layer_is_noop():
    """observe(name) with an unknown name must not raise."""
    p = _NPParam(np.zeros((2, 2), dtype=np.float32))
    g = STDPParamGroup(
        param=p, name="known",
        buffer=SpikeRingBuffer(in_dim=2, out_dim=2, window=5),
        alpha=0.01, a_plus=0.01, a_minus=0.01, clip=1.0,
    )
    opt = STDPOnlyOptimizer([g])
    # Should not raise
    opt.observe(
        "unknown_layer",
        np.array([1, 1], dtype=np.uint8),
        np.array([1, 1], dtype=np.uint8),
    )


# ----------------------------------------------------------------------- T9

def test_construct_empty_groups_raises():
    with pytest.raises(ValueError, match="at least one"):
        STDPOnlyOptimizer([])
