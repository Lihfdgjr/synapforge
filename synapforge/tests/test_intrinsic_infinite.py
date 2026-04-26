"""Tests for intrinsic.* and infinite.* modules."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import pytest

import synapforge as sf


# ---------------------------------------------------------------------------
# Intrinsic
# ---------------------------------------------------------------------------


def test_free_energy_surprise_grad():
    fes = sf.FreeEnergySurprise(16)
    h_prev = torch.randn(4, 16, requires_grad=True)
    h_next = torch.randn(4, 16)
    s = fes.surprise(h_prev, h_next)
    s.backward()
    assert h_prev.grad is not None


def test_novelty_drive_ema():
    nd = sf.NoveltyDrive(8, ema=0.9)
    n1 = nd.novelty(torch.randn(4, 8))
    n2 = nd.novelty(torch.randn(4, 8))
    assert n1.dim() == 0
    assert n2.dim() == 0
    # h_bar should now be non-zero.
    assert nd.h_bar.abs().sum() > 0


def test_homeostatic_regulator():
    hr = sf.HomeostaticRegulator(spike_target=(0.05, 0.15), tau_range=(0.5, 20.0))
    # Inside band.
    assert hr.penalty(0.10, 5.0) == 0.0
    # Outside spike band.
    p = hr.penalty(0.30, 5.0)
    assert p < 0
    # Outside tau band.
    p2 = hr.penalty(0.10, 100.0)
    assert p2 < 0


def test_goal_memory_top_k():
    gm = sf.GoalMemory(capacity=5)
    gm.record([1, 2, 3], pre_loss=5.0, post_loss=4.0)
    gm.record([4, 5], pre_loss=3.0, post_loss=2.0)
    gm.record([6, 7, 8], pre_loss=10.0, post_loss=9.5)
    feat = gm.top_k_features(k=2)
    assert feat.dim() == 1


def test_intrinsic_reward_combine():
    ir = sf.IntrinsicReward(16)
    h_prev = torch.randn(4, 16)
    h_next = torch.randn(4, 16)
    r = ir(h_prev, h_next, spike_rate=0.10, tau=5.0)
    assert r.dim() == 0


def test_self_goal_proposer_fallback():
    """Proposer should not crash even when model raises."""
    class BadModel(torch.nn.Module):
        def forward(self, **kw):
            raise RuntimeError("forced")
    p = sf.SelfGoalProposer(BadModel(), vocab_size=10, max_len=4)
    g = p.propose()
    assert isinstance(g, list)
    assert g[-1] == p.eos


# ---------------------------------------------------------------------------
# Infinite — long-context modules
# ---------------------------------------------------------------------------


def test_rope_shape():
    rope = sf.RotaryPositionEncoding(64)
    x = torch.randn(2, 16, 64)
    y = rope(x, seq_dim=1)
    assert y.shape == x.shape


def test_rope_odd_head_dim_rejected():
    with pytest.raises(ValueError):
        sf.RotaryPositionEncoding(63)


def test_local_gqa_attention():
    attn = sf.LocalGQAttention(hidden=64, q_heads=4, kv_heads=2, head_dim=16, window=4)
    y = attn(torch.randn(2, 16, 64))
    assert y.shape == (2, 16, 64)


def test_hierarchical_memory_write_read():
    cfg = sf.HierarchicalMemoryConfig(hidden=16, l1_capacity=4, l2_capacity=8, compress_every=2)
    mem = sf.HierarchicalMemory(cfg)
    for _ in range(20):
        mem.write(torch.randn(16))
    out = mem.read(torch.randn(16), top_k=4)
    assert out.shape == (4, 16)


def test_delta_compress_grows():
    dc = sf.DeltaCompress(16, period=4)
    y = dc(torch.randn(1, 8, 16))
    # Should add 2 summary tokens for T=8, period=4.
    assert y.shape[1] > 8


def test_adaptive_slow_tau():
    slow = sf.AdaptiveSlowTau(16, tau_init=100.0)
    out = slow(torch.randn(2, 16))
    assert out.shape == (2, 16)
    # state advanced.
    assert slow.state.abs().sum() > 0


def test_ssm_diag_scan():
    ssm = sf.SSMDiagScan(8, a_init=0.9)
    x = torch.randn(2, 5, 8)
    y, h = ssm(x)
    assert y.shape == x.shape
    assert h.shape == (2, 8)


def test_external_vector_memory_torch_fallback():
    ext = sf.ExternalVectorMemory(dim=8, capacity=16, use_faiss=False)
    for i in range(5):
        ext.add(torch.randn(8), meta=i)
    sc, idx, m = ext.topk(torch.randn(8), k=3)
    assert sc.shape == (3,)
    assert len(idx) == 3
    assert len(m) == 3


def test_external_vector_memory_overflow():
    ext = sf.ExternalVectorMemory(dim=4, capacity=2, use_faiss=False)
    ext.add(torch.randn(4))
    ext.add(torch.randn(4))
    with pytest.raises(RuntimeError):
        ext.add(torch.randn(4))


def test_disk_memmap_archive_roundtrip(tmp_path):
    arc = sf.DiskMemmapArchive(tmp_path / "arc", dim=8, dtype="float32")
    v = torch.randn(8)
    idx = arc.append(v, meta={"id": 1})
    assert idx == 0
    assert arc.count() == 1
    out = arc.get(0)
    assert out.shape == (8,)


def test_infinite_context_reader_smoke():
    cfg = sf.InfiniteReaderConfig(
        hidden=32, q_heads=4, kv_heads=2, head_dim=8, window=4,
        l1_capacity=8, l2_capacity=16, compress_every=4, ext_capacity=64,
    )
    reader = sf.InfiniteContextReader(cfg)
    chunk = torch.randn(1, 8, 32)
    y = reader.apply_window_attention(chunk)
    assert y.shape == (1, 8, 32)
    for _ in range(20):
        reader.write(torch.randn(32))
    ctx = reader.read(torch.randn(32))
    assert ctx.shape == (32,)


def test_chunked_state_carry():
    def step(piece, state):
        return piece * 0.5, state
    csc = sf.ChunkedStateCarry(step, chunk=4)
    outs, state = csc.run(torch.randn(2, 12, 16))
    assert len(outs) == 3
    for o in outs:
        assert o.shape == (2, 4, 16)


def test_long_context_monitor_ok():
    mon = sf.LongContextMonitor(positions=(2, 4, 6))
    nll = torch.linspace(1.0, 1.05, 10)  # tiny drift
    mon.add(nll)
    rep = mon.report()
    assert 2 in rep and 4 in rep and 6 in rep
    assert mon.ok(threshold=0.1) is True


def test_streaming_evaluator(tmp_path):
    def step(p, s):
        return p, s
    accum_calls = []
    def acc(out, ctx):
        accum_calls.append(ctx["chunks"])
    se = sf.StreamingInfiniteEvaluator(
        step, chunk_size=4, checkpoint_every=2,
        checkpoint_path=tmp_path / "ck.pkl", accumulator=acc,
    )
    pieces = [torch.randn(1, 8, 16) for _ in range(2)]
    rep = se.run(iter(pieces))
    assert rep["chunks_done"] == 4
    # 2 pieces * B=1 * T=8 = 16 tokens.
    assert rep["tokens_done"] == 16
    assert len(accum_calls) == 4
