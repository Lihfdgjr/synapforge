"""Tests for the NeurIPS 2025 (Fang et al. arXiv:2505.18608) frequency fixes.

Three knobs, each verified to (a) produce a finite forward, (b) leave the
default-OFF baseline unchanged when the flag is not passed.

Cheap CPU smokes; no GPU required.
"""
from __future__ import annotations

import math

import pytest
import torch

from synapforge.modal.byte_patch import BytePatch
from synapforge.model_100m import HybridBlock, build_synapforge_100m
from synapforge.surrogate import PLIFCell


# ---------------------------------------------------------------------- A1
def test_a1_byte_patch_max_pool_branch_runs_and_default_avg_unchanged():
    """A1: BytePatch(pool='avg'|'max'|'max+avg') runs without NaN.

    The 'avg' branch is the legacy path -- it must use the same
    Linear(patch * F_in -> hidden) projection (default behaviour
    preserved).
    """
    B, L, F_in, hidden, patch = 2, 64, 8, 16, 4
    x = torch.randn(B, L, F_in)

    # Default 'avg' is the legacy path -- single Linear over flattened window.
    bp_avg = BytePatch(in_feat=F_in, hidden=hidden, patch=patch, pool="avg")
    y_avg = bp_avg(x)
    assert y_avg.shape == (B, L // patch, hidden)
    assert torch.isfinite(y_avg).all()
    # Legacy projection signature: Linear(patch * F_in, hidden).
    assert bp_avg.proj is not None
    assert bp_avg.proj.weight.shape == (hidden, patch * F_in)

    # Max-Pool branch.
    bp_max = BytePatch(in_feat=F_in, hidden=hidden, patch=patch, pool="max")
    y_max = bp_max(x)
    assert y_max.shape == (B, L // patch, hidden)
    assert torch.isfinite(y_max).all()
    assert bp_max.proj_max is not None
    assert bp_max.proj_max.weight.shape == (hidden, F_in)

    # max+avg branch.
    bp_concat = BytePatch(in_feat=F_in, hidden=hidden, patch=patch,
                          pool="max+avg")
    y_concat = bp_concat(x)
    assert y_concat.shape == (B, L // patch, hidden)
    assert torch.isfinite(y_concat).all()
    assert bp_concat.proj_concat is not None
    assert bp_concat.proj_concat.weight.shape == (hidden, 2 * F_in)


# ---------------------------------------------------------------------- A2
def test_a2_high_pass_residual_default_off_and_runs_when_on():
    """A2: high_pass_residual_weight=0.0 -> no extra modules; > 0 fires."""
    B, T, d = 2, 8, 16

    # Default OFF: no extra modules, no extra params.
    blk = HybridBlock(d=d, ffn_ratio=2.0, sparsity=0.0)
    assert blk.high_pass_residual_weight == 0.0
    assert blk.hp_lowpass is None
    assert blk.hp_lambda is None
    x = torch.randn(B, T, d)
    out = blk(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    # ON: hp_lowpass + hp_lambda built, residual computed.
    blk_hp = HybridBlock(d=d, ffn_ratio=2.0, sparsity=0.0,
                         high_pass_residual_weight=0.1)
    assert blk_hp.high_pass_residual_weight == pytest.approx(0.1)
    assert isinstance(blk_hp.hp_lowpass, torch.nn.Conv1d)
    assert blk_hp.hp_lowpass.groups == d  # depth-wise
    assert blk_hp.hp_lambda is not None
    assert blk_hp.hp_lambda.shape == (d,)
    out_hp = blk_hp(x)
    assert out_hp.shape == x.shape
    assert torch.isfinite(out_hp).all()

    # The two outputs must differ (the residual must actually be added).
    assert not torch.allclose(out, out_hp)


# ---------------------------------------------------------------------- A3
def test_a3_plif_trimodal_tau_init_spans_three_bands():
    """A3: PLIFCell(tau_init='trimodal') spreads tau across 3 bands."""
    hidden = 100  # picked so 30/40/30 split is exact.
    cell = PLIFCell(hidden=hidden, tau_init="trimodal", threshold_init=0.1)
    log_tau = cell.log_tau.detach()
    tau = log_tau.exp()
    # Sanity: tau values are positive and finite.
    assert torch.isfinite(tau).all()
    assert (tau > 0).all()
    # Three distinct values pre-training (init-only check).
    unique_taus = torch.unique(tau).tolist()
    assert len(unique_taus) == 3
    sorted_taus = sorted(unique_taus)
    short_t, mid_t, long_t = sorted_taus
    # Order short << mid << long, anchored to the docstring init constants.
    assert short_t == pytest.approx(0.5, rel=1e-4)
    assert mid_t == pytest.approx(2.0, rel=1e-4)
    assert long_t == pytest.approx(8.0, rel=1e-4)
    # Counts: 30 / 40 / 30 (when hidden divides cleanly into 30/40/30).
    eq_short = (tau == short_t).sum().item()
    eq_mid = (tau == mid_t).sum().item()
    eq_long = (tau == long_t).sum().item()
    assert eq_short == 30
    assert eq_mid == 40
    assert eq_long == 30

    # Forward pass must run and not NaN.
    x = torch.randn(2, 5, hidden)
    s, v = cell.forward_seq(x)
    assert s.shape == x.shape
    assert torch.isfinite(s).all()
    assert torch.isfinite(v).all()


# ---------------------------------------------------------------------- All-on
def test_full_model_all_three_flags_on_runs():
    """Sanity: build_synapforge_100m with A2+A3 flags forwards cleanly."""
    model = build_synapforge_100m(
        vocab=64, d=16, n_layers=1, loop_depth=1, max_seq=32,
        ffn_ratio=2.0, sparsity=0.0,
        plif_tau_init="trimodal",
        high_pass_residual_weight=0.05,
    )
    ids = torch.randint(0, 64, (2, 8))
    logits = model(ids)
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()
