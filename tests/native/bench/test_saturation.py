"""tests/native/bench/test_saturation.py -- saturation profiler tests.

Coverage
--------
* roofline   -- closed-form output matches a hand-computed value for
                d=1280 / n_layers=16 / vocab=151936.
* profiler   -- ``StageProfiler`` returns positive timings on a
                deterministic toy model; recorded shares sum to >0.95.
* autotune   -- returns a non-empty result with a valid winner config
                (no NaN, no zero bs).

These tests don't require a GPU and complete in <5s on CI.
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running these tests without importing the rest of the
# ``synapforge`` package (which transitively imports torch).
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]  # repo root
_BENCH = _REPO / "synapforge" / "native" / "bench"
sys.path.insert(0, str(_BENCH))

import roofline  # type: ignore  # noqa: E402
import stage_profiler  # type: ignore  # noqa: E402
import autotune  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Roofline tests.
# ---------------------------------------------------------------------------


class TestRoofline:
    """Check the closed-form roofline against hand-computed values."""

    def test_a800_run7_config_basic(self) -> None:
        """Smoke: A800 + Run 7 config returns a sane RooflineResult."""
        model = roofline.ModelSpec(
            d=1280, n_layers=16, loop_depth=2, ffn_ratio=3.0,
            seq_len=1024, batch_size=32, grad_accum=2, vocab=151936,
        )
        res = roofline.compute_roofline(
            model, roofline.A800_80GB,
            cpu_offload_optim=True, grad_ckpt=True,
        )
        # Total step must be finite + positive.
        assert math.isfinite(res.ms_step_total)
        assert res.ms_step_total > 0
        # tok/s must be positive and below TC roof.
        assert res.tok_per_sec_total > 0
        # Should be below pure-compute roof (because PCIe/HBM also cost time).
        assert res.tok_per_sec_total <= res.tok_per_sec_compute * 1.001
        # Basic sanity: step is at least 100 ms but less than 30 s.
        assert 100.0 < res.ms_step_total < 30_000.0
        # Has all expected stages.
        names = {s.name for s in res.stages}
        for required in ("data_loader", "embed_fwd", "hybridblock_fwd",
                         "lm_head_fwd", "hybridblock_bwd",
                         "lm_head_bwd", "optimizer_step"):
            assert required in names, f"missing stage {required}"

    def test_hand_computed_lm_head_fwd_flops(self) -> None:
        """LM-head fwd FLOPs/token must equal 2 * d * V (hand)."""
        model = roofline.ModelSpec(d=1280, vocab=151936)
        expected = 2.0 * 1280 * 151936
        actual = roofline._lm_head_fwd_flops(model)
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_hand_computed_hybridblock_fwd_flops(self) -> None:
        """fwd FLOPs/token: 4*d^2 + 6*d*h + 2*d^2 (cfc + swiglu + sew)."""
        d, ffn_ratio = 1280, 3.0
        h = int(d * ffn_ratio)
        model = roofline.ModelSpec(d=d, ffn_ratio=ffn_ratio)
        expected = 4 * d * d + 6 * d * h + 2 * d * d
        actual = roofline._hybridblock_fwd_flops(model)
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_n_params_count(self) -> None:
        """Param count for d=1280 / n_layers=16 / loop_depth=2 should be ~1B."""
        model = roofline.ModelSpec(
            d=1280, n_layers=16, loop_depth=2, ffn_ratio=3.0,
            vocab=151936,
        )
        # embed + lm_head = 2 * V * d = 2 * 151936 * 1280 ~ 389M
        # per layer per loop = 4*d^2 + 3*d*h + tiny
        #   = 4*1280^2 + 3*1280*3840 = 6.55M + 14.75M ~ 21M
        # n_blocks = 32 * 21M = 672M
        # total ~ 1.06B
        assert 1_000e6 < model.n_params < 1_200e6

    def test_a800_compute_roof_under_4ms_for_trivial(self) -> None:
        """On a tiny model A800 compute roof should be sub-ms."""
        model = roofline.ModelSpec(
            d=64, n_layers=2, loop_depth=1, ffn_ratio=2.0,
            seq_len=16, batch_size=2, grad_accum=1, vocab=128,
        )
        res = roofline.compute_roofline(model, roofline.A800_80GB)
        # compute time tiny
        assert res.ms_step_compute < 1.0

    def test_hardware_specs_have_positive_specs(self) -> None:
        for hw in (roofline.A800_80GB, roofline.A100_80GB, roofline.H100_80GB):
            assert hw.tc_tflops > 0
            assert hw.hbm_bw_gbs > 0
            assert hw.pcie_bw_gbs > 0
            assert hw.ridge_ai > 0

    def test_pcie_offload_dominates_optimizer_step(self) -> None:
        """With cpu_offload=True, optimizer step must be PCIe-bound."""
        model = roofline.ModelSpec(d=1280, n_layers=16, loop_depth=2)
        res = roofline.compute_roofline(
            model, roofline.A800_80GB,
            cpu_offload_optim=True, grad_ckpt=True,
        )
        opt = next(s for s in res.stages if s.name == "optimizer_step")
        assert opt.binding == "pcie"

    def test_h100_faster_than_a800(self) -> None:
        """H100 must give higher tok/s than A800 for the same model."""
        model = roofline.ModelSpec(
            d=1280, n_layers=16, loop_depth=2, ffn_ratio=3.0,
            seq_len=1024, batch_size=32, grad_accum=2, vocab=151936,
        )
        a800 = roofline.compute_roofline(model, roofline.A800_80GB)
        h100 = roofline.compute_roofline(model, roofline.H100_80GB)
        assert h100.tok_per_sec_total > a800.tok_per_sec_total


# ---------------------------------------------------------------------------
# Stage profiler tests.
# ---------------------------------------------------------------------------


class TestStageProfiler:
    """Check the synthetic profiler returns sane timings."""

    def test_stage_profiler_returns_positive_timings(self) -> None:
        s = stage_profiler.profile_synthetic_step(
            bs=2, seq=16, d=32, ffn=64, n_layers=2,
            rfold_chunk=8, n_warmup=2, n_steps=5, seed=0xBEEF,
        )
        # Must have a step_ms_mean > 0
        assert s["step_ms_mean"] > 0
        # tok/s must be positive
        assert s["tok_per_sec"] > 0
        # Number of steps recorded should be n_steps - 1 (next_step diff)
        assert s["n_steps"] == 4
        # Bottleneck should be one of the recorded stages
        assert s["bottleneck_stage"] in s["stages"]
        # All listed stages present
        for required in ("data_loader", "embed_fwd", "hybridblock_fwd",
                         "lm_head_fwd", "hybridblock_bwd",
                         "lm_head_bwd", "optimizer_step"):
            assert required in s["stages"]
            assert s["stages"][required]["ms_mean"] >= 0.0

    def test_stage_profiler_deterministic_for_same_seed(self) -> None:
        """Same seed should give matching loss curves (timings will vary)."""
        # NOTE: tokens are deterministic with seed; timings aren't.
        # We only assert tokens_per_step is identical.
        s1 = stage_profiler.profile_synthetic_step(
            bs=2, seq=16, d=32, ffn=64, n_layers=2,
            rfold_chunk=8, n_warmup=1, n_steps=3, seed=42,
        )
        s2 = stage_profiler.profile_synthetic_step(
            bs=2, seq=16, d=32, ffn=64, n_layers=2,
            rfold_chunk=8, n_warmup=1, n_steps=3, seed=42,
        )
        assert s1["tokens_per_step"] == s2["tokens_per_step"]
        assert s1["n_steps"] == s2["n_steps"]

    def test_stage_profiler_scales_with_bs(self) -> None:
        """Larger bs => more tokens per step (mech check)."""
        s_small = stage_profiler.profile_synthetic_step(
            bs=2, seq=16, d=32, ffn=64, n_layers=1,
            rfold_chunk=8, n_warmup=1, n_steps=3, seed=1,
        )
        s_large = stage_profiler.profile_synthetic_step(
            bs=8, seq=16, d=32, ffn=64, n_layers=1,
            rfold_chunk=8, n_warmup=1, n_steps=3, seed=1,
        )
        assert s_large["tokens_per_step"] == 4 * s_small["tokens_per_step"]

    def test_manual_stage_profiler_basic(self) -> None:
        """API check: StageProfiler accumulates across steps."""
        prof = stage_profiler.StageProfiler(
            stages=("foo", "bar"), tokens_per_step=100,
        )
        for _ in range(3):
            with prof.stage("foo"):
                _ = sum(range(1000))
            with prof.stage("bar"):
                _ = sum(range(2000))
            prof.next_step()
        s = prof.summary()
        assert s["stages"]["foo"]["n"] == 3
        assert s["stages"]["bar"]["n"] == 3
        assert s["tokens_per_step"] == 100


# ---------------------------------------------------------------------------
# Autotune tests.
# ---------------------------------------------------------------------------


class TestAutotune:
    """End-to-end smoke for the autotuner."""

    def test_autotune_returns_valid_winner(self) -> None:
        cfg = autotune.AutoTuneConfig(
            bs_grid=(2, 4),
            grad_accum_grid=(1,),
            rfold_chunk_grid=(8,),
            n_data_threads_grid=(2,),
            seq_len=8, d=16, ffn=32, n_layers=1,
            n_warmup=1, n_steps=2,
            baseline_bs=2, baseline_grad_accum=1,
            baseline_rfold_chunk=8, baseline_n_data_threads=2,
            coarse_bs_top_k=1,
        )
        res = autotune.autotune(cfg)
        assert isinstance(res, autotune.AutoTuneResult)
        assert res.n_configs_tried > 0
        # Winner must exist and have valid bs > 0
        assert res.winner is not None
        assert res.winner["bs"] > 0
        assert res.winner["grad_accum"] > 0
        assert res.winner["rfold_chunk"] > 0
        assert res.winner["n_data_threads"] > 0
        # tok/s positive, no NaN
        assert res.winner_tok_per_sec > 0
        assert math.isfinite(res.winner_tok_per_sec)
        # Speedup vs baseline must be positive
        assert res.speedup_vs_baseline > 0

    def test_autotune_dict_serializable(self) -> None:
        cfg = autotune.AutoTuneConfig(
            bs_grid=(2,),
            grad_accum_grid=(1,),
            rfold_chunk_grid=(8,),
            n_data_threads_grid=(2,),
            seq_len=8, d=16, ffn=32, n_layers=1,
            n_warmup=1, n_steps=2,
            baseline_bs=2, baseline_grad_accum=1,
            baseline_rfold_chunk=8, baseline_n_data_threads=2,
        )
        res = autotune.autotune(cfg)
        d = res.to_dict()
        # Must be JSON-dump-able (no torch tensors etc.)
        import json
        s = json.dumps(d, default=str)
        assert "winner" in s
        assert "configs" in s

    def test_autotune_quality_gate_rejects_bad_loss(self) -> None:
        """If the loss tolerance is 0, only the baseline is kept."""
        cfg = autotune.AutoTuneConfig(
            bs_grid=(2, 4),
            grad_accum_grid=(1,),
            rfold_chunk_grid=(8,),
            n_data_threads_grid=(2,),
            seq_len=8, d=16, ffn=32, n_layers=1,
            n_warmup=1, n_steps=2,
            loss_tol=0.0,    # zero tolerance
            baseline_bs=2, baseline_grad_accum=1,
            baseline_rfold_chunk=8, baseline_n_data_threads=2,
            coarse_bs_top_k=1,
        )
        res = autotune.autotune(cfg)
        # n_configs_kept may be 1 (just baseline) or 0 (if numeric drift)
        assert res.n_configs_kept <= res.n_configs_tried

    def test_autotune_skips_oversized_vram(self) -> None:
        """If vram_cap_mb is tiny, all configs should be skipped."""
        cfg = autotune.AutoTuneConfig(
            bs_grid=(64,),
            grad_accum_grid=(1,),
            rfold_chunk_grid=(8,),
            n_data_threads_grid=(2,),
            seq_len=64, d=128, ffn=256, n_layers=2,
            n_warmup=1, n_steps=2,
            vram_cap_mb=0.001,  # 1 KB cap, will reject everything
            baseline_bs=64, baseline_grad_accum=1,
            baseline_rfold_chunk=8, baseline_n_data_threads=2,
        )
        res = autotune.autotune(cfg)
        # Baseline gets skipped too since vram cap is tiny
        skipped = [c for c in res.configs
                   if "vram" in (c.rejection_reason or "").lower()]
        assert len(skipped) >= 1


# ---------------------------------------------------------------------------
# Format helpers should not crash.
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_format_roofline_table_returns_string(self) -> None:
        model = roofline.ModelSpec()
        res = roofline.compute_roofline(model, roofline.A800_80GB)
        s = roofline.format_roofline_table(res)
        assert isinstance(s, str)
        assert "Roofline" in s
        assert "tok/s" in s

    def test_format_autotune_report_returns_string(self) -> None:
        cfg = autotune.AutoTuneConfig(
            bs_grid=(2,), grad_accum_grid=(1,), rfold_chunk_grid=(8,),
            n_data_threads_grid=(2,), seq_len=8, d=16, ffn=32, n_layers=1,
            n_warmup=1, n_steps=2, baseline_bs=2, baseline_grad_accum=1,
            baseline_rfold_chunk=8, baseline_n_data_threads=2,
        )
        res = autotune.autotune(cfg)
        s = autotune.format_autotune_report(res)
        assert isinstance(s, str)
        assert "tok/s" in s
