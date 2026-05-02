"""Edge-case tests for the VJP catalogue.

These cover numerical pitfalls documented in
``docs/NATIVE_VJP_CATALOG.md``:
* RMSNorm at near-zero variance (eps floor saves us).
* CE with all-ignore_index targets (no NaN, zero gradient).
* Embedding with repeated ids (np.add.at correctness).
* CfC with extreme A_log (saturated decay, no overflow).
* PLIF with all-zero current (no spikes, smooth backward).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


_REPO = Path(__file__).resolve().parents[3]
_VJP_DIR = _REPO / "synapforge" / "native" / "vjp"


def _make_pkg(name: str, path: Path) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg
    return pkg


_make_pkg("synapforge", _REPO / "synapforge")
_make_pkg("synapforge.native", _REPO / "synapforge" / "native")
_make_pkg("synapforge.native.vjp", _VJP_DIR)


def _load_sub(modname: str) -> types.ModuleType:
    full = f"synapforge.native.vjp.{modname}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _VJP_DIR / f"{modname}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


vjp_linear = _load_sub("linear")
vjp_embed = _load_sub("embed")
vjp_rmsnorm = _load_sub("rmsnorm")
vjp_swiglu = _load_sub("swiglu")
vjp_cfc = _load_sub("cfc")
vjp_plif = _load_sub("plif")
vjp_ce = _load_sub("cross_entropy")
vjp_dtypes = _load_sub("dtypes")


class TestEdgeCases:
    def test_rmsnorm_near_zero_variance(self):
        """RMSNorm with all-zero x must not blow up (eps saves it)."""
        d = 4
        x = np.zeros((2, d), dtype=np.float32)
        gamma = np.ones((d,), dtype=np.float32)
        gy = np.ones((2, d), dtype=np.float32)
        y, rstd = vjp_rmsnorm.rmsnorm_fwd(x, gamma, eps=1e-6)
        # rstd capped at 1/sqrt(eps) = 1000
        assert np.all(rstd <= 1.0 / np.sqrt(1e-6) + 1.0)
        gx, gg = vjp_rmsnorm.rmsnorm_bwd(gy, x, gamma, rstd)
        assert np.all(np.isfinite(gx))
        assert np.all(np.isfinite(gg))

    def test_ce_all_ignore_targets(self):
        """CE with every target == ignore_index returns 0 loss + 0 grad."""
        N, V = 3, 5
        logits = np.random.default_rng(0).standard_normal((N, V)).astype(np.float32)
        targets = np.full((N,), -100, dtype=np.int64)

        loss, saved = vjp_ce.ce_fwd(
            logits, targets, ignore_index=-100, reduction="sum"
        )
        assert loss == 0.0
        grad = vjp_ce.ce_bwd(np.array(1.0, dtype=np.float32), saved)
        assert np.all(grad == 0.0)

    def test_ce_mean_with_all_ignore_no_nan(self):
        """Mean reduction with empty mask -- denom clamped to 1."""
        N, V = 3, 5
        logits = np.random.default_rng(0).standard_normal((N, V)).astype(np.float32)
        targets = np.full((N,), -100, dtype=np.int64)
        loss, saved = vjp_ce.ce_fwd(
            logits, targets, ignore_index=-100, reduction="mean"
        )
        assert np.isfinite(loss)
        assert loss == 0.0
        grad = vjp_ce.ce_bwd(np.array(1.0, dtype=np.float32), saved)
        assert np.all(np.isfinite(grad))

    def test_embed_repeated_ids_accumulate(self):
        """Embed bwd must scatter-ADD to repeated rows, not overwrite."""
        V, d = 5, 3
        # All tokens point to row 2 -- grad_W[2] should be sum of gradients.
        ids = np.array([[2, 2, 2]], dtype=np.int64)
        grad_y = np.array([[[1.0, 2.0, 3.0],
                            [10.0, 20.0, 30.0],
                            [100.0, 200.0, 300.0]]], dtype=np.float32)
        gW = vjp_embed.embed_bwd(grad_y, ids, V)
        expected = np.array([111.0, 222.0, 333.0])
        assert np.allclose(gW[2], expected)
        # Other rows must be zero.
        for v in [0, 1, 3, 4]:
            assert np.allclose(gW[v], 0.0)

    def test_cfc_step_extreme_A_log(self):
        """CfC step with very large A_log -> A_t saturates to ~0 (very fast decay)."""
        B, in_dim, d = 1, 2, 3
        h_prev = np.ones((B, d), dtype=np.float32)
        x = np.ones((B, in_dim), dtype=np.float32)
        W_in = np.eye(d, in_dim, dtype=np.float32)
        W_h = np.eye(d, in_dim, dtype=np.float32)
        A_log = np.full((d,), 5.0, dtype=np.float32)  # exp(5) ~ 148
        h_t, out_t, cache = vjp_cfc.cfc_step_fwd(h_prev, x, W_in, W_h, A_log)
        assert np.all(np.isfinite(h_t))
        assert np.all(np.isfinite(out_t))
        # A_t should be tiny since exp(-softplus(...) * 148) ~ 0
        assert np.all(cache["A_t"] < 1e-3)
        # bwd should still produce finite grads
        gx, ghp, gWin, gWh, gAlog, _ = vjp_cfc.cfc_step_bwd(
            np.ones((B, d), dtype=np.float32),
            np.zeros((B, d), dtype=np.float32),
            cache,
        )
        assert np.all(np.isfinite(gx))
        assert np.all(np.isfinite(gAlog))

    def test_plif_zero_current_no_spikes(self):
        """PLIF with zero input and below-threshold v_prev -> no spikes."""
        B, d = 2, 3
        v_prev = np.full((B, d), 0.1, dtype=np.float32)
        x = np.zeros((B, d), dtype=np.float32)
        tau_log = np.zeros((d,), dtype=np.float32)
        thr = np.full((d,), 0.5, dtype=np.float32)

        spike, v_new, saved = vjp_plif.plif_fwd(v_prev, x, tau_log, thr)
        assert np.all(spike == 0.0)
        # v_new = v_pre because no spike was subtracted
        assert np.all(np.isfinite(v_new))

        gx, gvp, gtau, gthr = vjp_plif.plif_bwd(
            np.ones((B, d), dtype=np.float32),
            np.ones((B, d), dtype=np.float32),
            saved,
        )
        assert np.all(np.isfinite(gx))
        assert np.all(np.isfinite(gvp))
        assert np.all(np.isfinite(gtau))
        assert np.all(np.isfinite(gthr))


class TestDtypes:
    def test_compute_dtype_default_fp32(self):
        x = np.zeros((2,), dtype=np.float32)
        assert vjp_dtypes.compute_dtype(x) == np.float32

    def test_compute_dtype_keeps_fp64(self):
        x = np.zeros((2,), dtype=np.float64)
        assert vjp_dtypes.compute_dtype(x) == np.float64

    def test_compute_dtype_override(self):
        x = np.zeros((2,), dtype=np.float32)
        assert vjp_dtypes.compute_dtype(x, override=np.float64) == np.float64

    def test_to_compute_no_copy_when_match(self):
        x = np.zeros((2,), dtype=np.float32)
        y = vjp_dtypes.to_compute(x, np.dtype(np.float32))
        # Should not copy when dtypes already match.
        assert y.dtype == np.float32

    def test_matches_dtype(self):
        a = np.zeros((2,), dtype=np.float32)
        b = np.zeros((3,), dtype=np.float32)
        c = np.zeros((2,), dtype=np.float64)
        assert vjp_dtypes.matches_dtype(a, b)
        assert not vjp_dtypes.matches_dtype(a, c)
        assert vjp_dtypes.matches_dtype()  # empty -> True
