"""Tests for wave_mixer / world_model / latent_thinking."""
from __future__ import annotations

import torch
import torch.nn as nn

import synapforge as sf

# ---------------------------------------------------------------------------
# WaveFormer1D + Hyena1D + FNet1D
# ---------------------------------------------------------------------------


def test_wave_former_1d_shape():
    wf = sf.WaveFormer1D(64, max_steps=4)
    x = torch.randn(2, 16, 64)
    y = wf(x, step_t=2)
    assert y.shape == (2, 16, 64)


def test_wave_former_step_clamp():
    wf = sf.WaveFormer1D(32, max_steps=3)
    wf.set_step(99)
    assert wf._step == 2  # clamped


def test_wave_former_grad():
    wf = sf.WaveFormer1D(32)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = wf(x, step_t=1)
    y.sum().backward()
    assert x.grad is not None
    assert wf.log_v.grad is not None


def test_wave_former_bf16():
    wf = sf.WaveFormer1D(32).to(torch.bfloat16)
    x = torch.randn(2, 16, 32, dtype=torch.bfloat16)
    y = wf(x)
    assert y.dtype == torch.bfloat16


def test_hyena_1d_shape():
    hy = sf.Hyena1D(32)
    y = hy(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)


def test_fnet_1d_shape():
    fn = sf.FNet1D(32)
    y = fn(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)


def test_attach_wave_mixer():
    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = nn.Linear(d, d)
            self._coe_step_t = 0
        def forward(self, x):
            return (self.lin(x),)
    blk = Block(32)
    sf.attach_wave_mixer_to_block(blk, 32, kind="wave")
    out = blk(torch.randn(2, 8, 32))
    assert isinstance(out, tuple)
    assert out[0].shape == (2, 8, 32)
    # idempotent
    sf.attach_wave_mixer_to_block(blk, 32, kind="wave")
    assert blk._wave_attached is True


# ---------------------------------------------------------------------------
# WorldModelHead + Loss
# ---------------------------------------------------------------------------


def test_world_model_head_residual_init():
    wmh = sf.WorldModelHead(32)
    h = torch.randn(2, 4, 32)
    out = wmh(h)
    # Zero-init -> next_hidden ~= h.
    assert torch.allclose(out.next_hidden, h, atol=1e-5)
    # variance positive.
    assert torch.all(out.aleatoric_var > 0)
    # reward / terminal in [0, 1].
    assert torch.all((out.reward >= 0) & (out.reward <= 1))
    assert torch.all((out.terminal >= 0) & (out.terminal <= 1))


def test_world_model_loss_smoke():
    wmh = sf.WorldModelHead(16)
    out = wmh(torch.randn(2, 3, 16, requires_grad=True))
    loss_mod = sf.WorldModelLoss()
    loss = loss_mod(out, torch.randn(2, 3, 16))
    assert "total" in loss
    assert loss["total"].dim() == 0


def test_world_model_loss_with_signals():
    wmh = sf.WorldModelHead(8)
    h = torch.randn(2, 3, 8)
    out = wmh(h)
    loss = sf.WorldModelLoss()(
        out,
        torch.randn(2, 3, 8),
        reward_signal=torch.rand(2, 3, 1),
        terminal_signal=torch.zeros(2, 3, 1),
    )
    assert loss["reward_bce"].abs() > 0
    assert "done_bce" in loss


def test_hypothesis_generator():
    gen = sf.HypothesisGenerator(16, n_hypotheses=4)
    h = torch.randn(2, 3, 16)
    out = gen(h)
    assert out.hypotheses.shape == (2, 3, 4, 16)
    assert out.scores.shape == (2, 3, 4)
    # scores sum to 1.
    assert torch.allclose(out.scores.sum(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_world_model_critic_score():
    crit = sf.WorldModelCritic(16)
    hyps = torch.randn(2, 3, 4, 16)
    ctx = torch.randn(2, 3, 16)
    sc = crit(hyps, ctx)
    assert sc.shape == (2, 3, 4)


def test_outcome_ce_loss():
    sc = torch.randn(2, 3, 4)
    rw = torch.randn(2, 3, 4)
    loss = sf.WorldModelCritic.outcome_ce_loss(sc, rw)
    # Either scalar 0-dim or 1-elem (when log_sigma has shape [1])
    assert loss.numel() == 1


# ---------------------------------------------------------------------------
# Latent thinking
# ---------------------------------------------------------------------------


def test_thinking_tokens():
    tt = sf.ThinkingTokens(vocab_size=100, hidden=16)
    assert tt.bot_id == 100
    assert tt.eot_id == 101
    e = tt.embed(tt.bot_id)
    assert e.shape == (16,)
    assert tt.is_thinking(tt.bot_id) is True
    assert tt.is_thinking(50) is False


def test_thinking_action_tokens():
    tat = sf.ThinkingActionTokens(vocab_size=100, hidden=16)
    assert tat.boa_id == 102
    assert tat.eoa_id == 103
    assert tat.is_action(tat.boa_id) is True
    assert tat.is_thinking(tat.boa_id) is False
    assert tat.is_action(tat.bot_id) is False


def test_latent_loop_controller_shape():
    block = nn.Linear(16, 16)
    ll = sf.LatentLoopController(16, block, max_think_steps=3)
    h = torch.randn(2, 6, 16)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 1:3] = True
    ext_h, loss_mask = ll(h, mask)
    assert ext_h.shape == (2, 6, 16)
    assert loss_mask.shape == (2, 6)
    # Loss applies at non-thinking positions.
    assert loss_mask.sum() == (2 * 6 - 2 * 2)


def test_latent_loop_zero_steps_passes_through():
    block = nn.Linear(8, 8)
    ll = sf.LatentLoopController(8, block, max_think_steps=0)
    h = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    out, _ = ll(h, mask)
    assert torch.allclose(out, h)


def test_latent_consistency_loss():
    lc = sf.LatentConsistencyLoss(16)
    h_before = torch.randn(2, 5, 16)
    h_after = torch.randn(2, 5, 16)
    loss = lc(h_before, h_after)
    # Either scalar 0-dim or 1-elem (when log_sigma has shape [1])
    assert loss.numel() == 1


def test_latent_search_beam():
    block = nn.Linear(8, 8)
    beam = sf.LatentSearchBeam(block=block, beam_size=3)
    init = beam.init(torch.randn(8))
    assert init.shape == (3, 8)
    nxt = beam.step(init)
    assert nxt.shape == (3, 8)
    final, idx = beam.finalise(nxt)
    assert final.shape == (8,)
    assert 0 <= idx < 3
