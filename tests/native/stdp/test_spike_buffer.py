"""Tests for synapforge.native.stdp.SpikeRingBuffer."""
from __future__ import annotations

import numpy as np
import pytest

from synapforge.native.stdp import SpikeRingBuffer


def test_construction_and_zero_state():
    buf = SpikeRingBuffer(in_dim=4, out_dim=3, window=5)
    assert buf.cursor == 0
    pre, post = buf.get_traces()
    np.testing.assert_array_equal(pre, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(post, np.zeros(3, dtype=np.float32))
    pre_w, post_w = buf.latest_spikes()
    np.testing.assert_array_equal(pre_w, np.zeros(4, dtype=np.uint8))
    np.testing.assert_array_equal(post_w, np.zeros(3, dtype=np.uint8))


def test_push_advances_cursor():
    buf = SpikeRingBuffer(in_dim=4, out_dim=3, window=3)
    pre = np.array([1, 0, 1, 0], dtype=np.uint8)
    post = np.array([0, 1, 0], dtype=np.uint8)
    buf.push(pre, post)
    assert buf.cursor == 1
    buf.push(pre, post)
    assert buf.cursor == 2
    buf.push(pre, post)
    assert buf.cursor == 0  # wraps


def test_trace_decay_correct():
    """Trace = decay * trace + spike  => after 1 step trace == spike."""
    buf = SpikeRingBuffer(in_dim=2, out_dim=2, window=5,
                          decay_pre=0.5, decay_post=0.5)
    pre = np.array([1, 0], dtype=np.uint8)
    post = np.array([0, 1], dtype=np.uint8)
    buf.push(pre, post)
    assert buf.pre_trace[0] == 1.0
    assert buf.pre_trace[1] == 0.0
    assert buf.post_trace[0] == 0.0
    assert buf.post_trace[1] == 1.0
    # Push zeros: trace decays by 0.5
    buf.push(np.zeros(2, dtype=np.uint8), np.zeros(2, dtype=np.uint8))
    assert abs(buf.pre_trace[0] - 0.5) < 1e-6
    assert abs(buf.post_trace[1] - 0.5) < 1e-6


def test_pair_outer_zero_when_no_post_spikes():
    """outer is zero when no post fires in the latest step."""
    buf = SpikeRingBuffer(in_dim=4, out_dim=3, window=3)
    pre = np.array([1, 1, 0, 1], dtype=np.uint8)
    post = np.zeros(3, dtype=np.uint8)
    buf.push(pre, post)
    dW = buf.pair_outer(a_plus=0.1, a_minus=0.0)
    np.testing.assert_array_equal(dW, np.zeros((3, 4), dtype=np.float32))


def test_active_spike_count():
    buf = SpikeRingBuffer(in_dim=4, out_dim=3, window=3)
    pre = np.array([1, 1, 0, 1], dtype=np.uint8)
    post = np.array([0, 1, 1], dtype=np.uint8)
    buf.push(pre, post)
    assert buf.active_spike_count() == 5


def test_reset_clears_everything():
    buf = SpikeRingBuffer(in_dim=4, out_dim=3, window=3)
    buf.push(np.ones(4, dtype=np.uint8), np.ones(3, dtype=np.uint8))
    assert buf.cursor != 0 or buf.pre_trace.sum() > 0
    buf.reset()
    assert buf.cursor == 0
    assert buf.pre_trace.sum() == 0
    assert buf.post_trace.sum() == 0
    assert buf.pre_window.sum() == 0
    assert buf.post_window.sum() == 0


def test_push_validates_shape():
    buf = SpikeRingBuffer(in_dim=4, out_dim=3, window=3)
    with pytest.raises(ValueError, match="pre_spike shape"):
        buf.push(np.ones(5, dtype=np.uint8), np.ones(3, dtype=np.uint8))
    with pytest.raises(ValueError, match="post_spike shape"):
        buf.push(np.ones(4, dtype=np.uint8), np.ones(7, dtype=np.uint8))


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        SpikeRingBuffer(in_dim=0, out_dim=3, window=5)
    with pytest.raises(ValueError):
        SpikeRingBuffer(in_dim=4, out_dim=3, window=0)
