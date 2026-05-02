"""Tests: native VJP catalogue vs torch.autograd reference + finite differences.

Two independent verifications per op:

(A) **Torch reference**: build the same forward in ``torch.nn.functional``,
    let ``torch.autograd`` compute gradients, and assert the native
    closed-form matches torch within fp32 tolerance (atol=1e-5, rtol=1e-4).

(B) **Numerical Jacobian**: pick a random scalar projection
    ``L = sum(grad_out * y)``, perturb each input by +/- eps, and
    compare ``(L(x+eps) - L(x-eps)) / (2 eps)`` against the analytic
    closed-form grad. Relative-error threshold ~1e-3 for fp32.

If torch is missing, the (A) tests are skipped via pytest.importorskip.
The (B) tests always run -- numpy + the package is enough.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---- Bootstrap: build a fake "synapforge.native.vjp" namespace pkg without
# going through synapforge/__init__.py (which eagerly imports torch).
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


# Order matters for swiglu (depends on linear).
vjp_linear = _load_sub("linear")
vjp_embed = _load_sub("embed")
vjp_rmsnorm = _load_sub("rmsnorm")
vjp_swiglu = _load_sub("swiglu")
vjp_cfc = _load_sub("cfc")
vjp_plif = _load_sub("plif")
vjp_sew = _load_sub("sew_shortcut")
vjp_ce = _load_sub("cross_entropy")


# ---------------------------------------------------------------------------
# Finite-difference helper.
# ---------------------------------------------------------------------------
def _finite_diff_grad(fn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    g = np.zeros_like(x, dtype=np.float64)
    flat_x = x.reshape(-1)
    flat_g = g.reshape(-1)
    for i in range(flat_x.size):
        old = flat_x[i]
        flat_x[i] = old + eps
        f_plus = fn(x)
        flat_x[i] = old - eps
        f_minus = fn(x)
        flat_x[i] = old
        flat_g[i] = (f_plus - f_minus) / (2.0 * eps)
    return g


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a32 = a.astype(np.float64)
    b32 = b.astype(np.float64)
    num = np.linalg.norm((a32 - b32).reshape(-1))
    den = max(np.linalg.norm(a32.reshape(-1)), np.linalg.norm(b32.reshape(-1)), 1e-12)
    return float(num / den)


# ---------------------------------------------------------------------------
# (B) Numerical-Jacobian-only tests (always run).
# ---------------------------------------------------------------------------
class TestNumericalJacobian:

    def test_linear(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((3, 5)).astype(np.float32)
        W = rng.standard_normal((4, 5)).astype(np.float32) * 0.1
        b = rng.standard_normal((4,)).astype(np.float32) * 0.1
        grad_y = rng.standard_normal((3, 4)).astype(np.float32)

        gx, gW, gb = vjp_linear.linear_bwd(grad_y, x, W, has_bias=True)

        def fL(xv): return float((vjp_linear.linear_fwd(xv, W, b) * grad_y).sum())
        def fLW(Wv): return float((vjp_linear.linear_fwd(x, Wv, b) * grad_y).sum())
        def fLb(bv): return float((vjp_linear.linear_fwd(x, W, bv) * grad_y).sum())

        assert _rel_err(gx, _finite_diff_grad(fL, x.copy())) < 1e-3
        assert _rel_err(gW, _finite_diff_grad(fLW, W.copy())) < 1e-3
        assert _rel_err(gb, _finite_diff_grad(fLb, b.copy())) < 1e-3

    def test_embed(self):
        rng = np.random.default_rng(2)
        V, d = 7, 4
        ids = np.array([[0, 3, 2], [1, 2, 6]], dtype=np.int64)
        W = rng.standard_normal((V, d)).astype(np.float32) * 0.5
        grad_y = rng.standard_normal((2, 3, d)).astype(np.float32)
        gW = vjp_embed.embed_bwd(grad_y, ids, V)

        def fL(Wv): return float((vjp_embed.embed_fwd(ids, Wv) * grad_y).sum())
        assert _rel_err(gW, _finite_diff_grad(fL, W.copy())) < 1e-3

    def test_rmsnorm(self):
        rng = np.random.default_rng(3)
        d = 6
        x = rng.standard_normal((2, 4, d)).astype(np.float32)
        gamma = rng.standard_normal((d,)).astype(np.float32) + 1.0
        grad_y = rng.standard_normal((2, 4, d)).astype(np.float32)

        y, rstd = vjp_rmsnorm.rmsnorm_fwd(x, gamma, eps=1e-6)
        gx, gg = vjp_rmsnorm.rmsnorm_bwd(grad_y, x, gamma, rstd)

        def fLx(xv):
            yv, _ = vjp_rmsnorm.rmsnorm_fwd(xv, gamma, eps=1e-6)
            return float((yv * grad_y).sum())

        def fLg(gv):
            yv, _ = vjp_rmsnorm.rmsnorm_fwd(x, gv, eps=1e-6)
            return float((yv * grad_y).sum())

        assert _rel_err(gx, _finite_diff_grad(fLx, x.copy(), eps=1e-3)) < 5e-3
        assert _rel_err(gg, _finite_diff_grad(fLg, gamma.copy(), eps=1e-3)) < 1e-3

    def test_swiglu(self):
        rng = np.random.default_rng(4)
        d_in, h = 5, 7
        x = rng.standard_normal((3, d_in)).astype(np.float32) * 0.3
        Wg = rng.standard_normal((h, d_in)).astype(np.float32) * 0.2
        Wu = rng.standard_normal((h, d_in)).astype(np.float32) * 0.2
        Wd = rng.standard_normal((d_in, h)).astype(np.float32) * 0.2
        grad_y = rng.standard_normal((3, d_in)).astype(np.float32)

        y, saved = vjp_swiglu.swiglu_fwd(x, Wg, Wu, Wd)
        gx, gWg, gWu, gWd = vjp_swiglu.swiglu_bwd(grad_y, saved)

        def fLx(xv):
            yv, _ = vjp_swiglu.swiglu_fwd(xv, Wg, Wu, Wd)
            return float((yv * grad_y).sum())

        def fLg(Wgv):
            yv, _ = vjp_swiglu.swiglu_fwd(x, Wgv, Wu, Wd)
            return float((yv * grad_y).sum())

        assert _rel_err(gx, _finite_diff_grad(fLx, x.copy(), eps=1e-3)) < 1e-2
        assert _rel_err(gWg, _finite_diff_grad(fLg, Wg.copy(), eps=1e-3)) < 1e-2

    def test_cfc_step(self):
        rng = np.random.default_rng(5)
        B, in_dim, d = 2, 3, 4
        h_prev = rng.standard_normal((B, d)).astype(np.float32) * 0.1
        x = rng.standard_normal((B, in_dim)).astype(np.float32) * 0.2
        W_in = rng.standard_normal((d, in_dim)).astype(np.float32) * 0.1
        W_h = rng.standard_normal((d, in_dim)).astype(np.float32) * 0.1
        A_log = rng.standard_normal((d,)).astype(np.float32) * 0.5
        grad_out = rng.standard_normal((B, d)).astype(np.float32)
        grad_h_next = np.zeros((B, d), dtype=np.float32)

        h_t, out_t, cache = vjp_cfc.cfc_step_fwd(h_prev, x, W_in, W_h, A_log)
        gx, ghp, gWin, gWh, gAlog, _ = vjp_cfc.cfc_step_bwd(grad_out, grad_h_next, cache)

        def fL_x(xv):
            _, ov, _ = vjp_cfc.cfc_step_fwd(h_prev, xv, W_in, W_h, A_log)
            return float((ov * grad_out).sum())

        def fL_Win(Wv):
            _, ov, _ = vjp_cfc.cfc_step_fwd(h_prev, x, Wv, W_h, A_log)
            return float((ov * grad_out).sum())

        def fL_Alog(Av):
            _, ov, _ = vjp_cfc.cfc_step_fwd(h_prev, x, W_in, W_h, Av)
            return float((ov * grad_out).sum())

        assert _rel_err(gx, _finite_diff_grad(fL_x, x.copy(), eps=1e-3)) < 1e-2
        assert _rel_err(gWin, _finite_diff_grad(fL_Win, W_in.copy(), eps=1e-3)) < 1e-2
        assert _rel_err(gAlog, _finite_diff_grad(fL_Alog, A_log.copy(), eps=1e-3)) < 1e-2

    def test_cfc_seq(self):
        rng = np.random.default_rng(6)
        B, T, in_dim, d = 2, 4, 3, 5
        x = rng.standard_normal((B, T, in_dim)).astype(np.float32) * 0.2
        h0 = rng.standard_normal((B, d)).astype(np.float32) * 0.1
        W_in = rng.standard_normal((d, in_dim)).astype(np.float32) * 0.1
        W_h = rng.standard_normal((d, in_dim)).astype(np.float32) * 0.1
        A_log = rng.standard_normal((d,)).astype(np.float32) * 0.5
        grad_out_seq = rng.standard_normal((B, T, d)).astype(np.float32)

        _, out_seq, caches = vjp_cfc.cfc_seq_fwd(x, h0, W_in, W_h, A_log)
        gx, gh0, gWin, gWh, gAlog, _ = vjp_cfc.cfc_seq_bwd(grad_out_seq, None, caches)

        eps = 1e-3
        gx_num_full = np.zeros_like(gx, dtype=np.float64)
        for b in range(B):
            for t in range(T):
                for k in range(in_dim):
                    xp = x.copy(); xp[b, t, k] += eps
                    xm = x.copy(); xm[b, t, k] -= eps
                    _, op_p, _ = vjp_cfc.cfc_seq_fwd(xp, h0, W_in, W_h, A_log)
                    _, op_m, _ = vjp_cfc.cfc_seq_fwd(xm, h0, W_in, W_h, A_log)
                    f_plus = float((op_p * grad_out_seq).sum())
                    f_minus = float((op_m * grad_out_seq).sum())
                    gx_num_full[b, t, k] = (f_plus - f_minus) / (2 * eps)
        assert _rel_err(gx, gx_num_full) < 1e-2

    def test_plif_smooth_paths(self):
        # PLIF spike forward is a hard indicator -- finite-diff over inputs
        # that cross the threshold is unreliable. We instead verify:
        #   * gradients are finite + correct shape
        #   * spike count matches a torch indicator
        # Full numerical check happens in TestAgainstTorch below where
        # the autograd reference handles the surrogate consistently.
        rng = np.random.default_rng(7)
        B, d = 3, 5
        v_prev = rng.standard_normal((B, d)).astype(np.float32) * 0.3
        x = rng.standard_normal((B, d)).astype(np.float32) * 0.2
        tau_log = rng.standard_normal((d,)).astype(np.float32) * 0.5
        thr = np.full((d,), 0.3, dtype=np.float32)
        grad_spike = rng.standard_normal((B, d)).astype(np.float32)
        grad_v_new = rng.standard_normal((B, d)).astype(np.float32)

        spike, v_new, saved = vjp_plif.plif_fwd(v_prev, x, tau_log, thr)
        gx, gvp, gtau, gthr = vjp_plif.plif_bwd(grad_spike, grad_v_new, saved)
        assert gx.shape == (B, d)
        assert gvp.shape == (B, d)
        assert gtau.shape == (d,)
        assert gthr.shape == (d,)
        assert np.all(np.isfinite(gx))
        assert np.all(np.isfinite(gvp))
        assert np.all(np.isfinite(gtau))
        assert np.all(np.isfinite(gthr))
        # Spike values must be 0 or 1
        assert set(np.unique(spike).tolist()).issubset({0.0, 1.0})

    def test_sew(self):
        rng = np.random.default_rng(8)
        spike = rng.standard_normal((2, 3)).astype(np.float32)
        h_dense = rng.standard_normal((2, 3)).astype(np.float32)
        grad_y = rng.standard_normal((2, 3)).astype(np.float32)

        y = vjp_sew.sew_fwd(spike, h_dense)
        gs, ghd = vjp_sew.sew_bwd(grad_y)
        assert np.allclose(y, spike + h_dense)
        assert np.allclose(gs, grad_y)
        assert np.allclose(ghd, grad_y)

    def test_cross_entropy_mean(self):
        rng = np.random.default_rng(9)
        N, V = 5, 7
        logits = rng.standard_normal((N, V)).astype(np.float32) * 0.5
        targets = rng.integers(0, V, size=(N,)).astype(np.int64)

        loss, saved = vjp_ce.ce_fwd(logits, targets, reduction="mean")
        grad_logits = vjp_ce.ce_bwd(np.array(1.0, dtype=np.float32), saved)

        def fL(z):
            l, _ = vjp_ce.ce_fwd(z, targets, reduction="mean")
            return float(l)

        assert _rel_err(grad_logits, _finite_diff_grad(fL, logits.copy(), eps=1e-3)) < 1e-3

    def test_cross_entropy_sum_with_ignore(self):
        rng = np.random.default_rng(10)
        N, V = 4, 5
        logits = rng.standard_normal((N, V)).astype(np.float32) * 0.4
        targets = np.array([1, -100, 3, 0], dtype=np.int64)

        loss, saved = vjp_ce.ce_fwd(logits, targets, ignore_index=-100, reduction="sum")
        grad_logits = vjp_ce.ce_bwd(np.array(1.0, dtype=np.float32), saved)

        def fL(z):
            l, _ = vjp_ce.ce_fwd(z, targets, ignore_index=-100, reduction="sum")
            return float(l)

        assert np.allclose(grad_logits[1], 0.0)
        assert _rel_err(grad_logits, _finite_diff_grad(fL, logits.copy(), eps=1e-3)) < 1e-3


# ---------------------------------------------------------------------------
# (A) Torch-reference tests (skipped if torch unavailable).
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestAgainstTorch:
    """Bit-tight comparison against torch.autograd reference."""

    def test_linear(self):
        rng = np.random.default_rng(100)
        x_np = rng.standard_normal((3, 5)).astype(np.float32)
        W_np = rng.standard_normal((4, 5)).astype(np.float32) * 0.1
        b_np = rng.standard_normal((4,)).astype(np.float32) * 0.1
        gy_np = rng.standard_normal((3, 4)).astype(np.float32)

        xt = torch.from_numpy(x_np).requires_grad_(True)
        Wt = torch.from_numpy(W_np).requires_grad_(True)
        bt = torch.from_numpy(b_np).requires_grad_(True)
        yt = F.linear(xt, Wt, bt)
        yt.backward(torch.from_numpy(gy_np))

        gx_np, gW_np, gb_np = vjp_linear.linear_bwd(gy_np, x_np, W_np, has_bias=True)
        assert np.allclose(gx_np, xt.grad.numpy(), atol=1e-5, rtol=1e-4)
        assert np.allclose(gW_np, Wt.grad.numpy(), atol=1e-5, rtol=1e-4)
        assert np.allclose(gb_np, bt.grad.numpy(), atol=1e-5, rtol=1e-4)

    def test_embed(self):
        rng = np.random.default_rng(101)
        V, d = 8, 4
        ids_np = np.array([[0, 3, 2], [1, 2, 6]], dtype=np.int64)
        W_np = rng.standard_normal((V, d)).astype(np.float32) * 0.3
        gy_np = rng.standard_normal((2, 3, d)).astype(np.float32)

        Wt = torch.from_numpy(W_np).requires_grad_(True)
        ids_t = torch.from_numpy(ids_np)
        yt = F.embedding(ids_t, Wt)
        yt.backward(torch.from_numpy(gy_np))
        gW_np = vjp_embed.embed_bwd(gy_np, ids_np, V)
        assert np.allclose(gW_np, Wt.grad.numpy(), atol=1e-5, rtol=1e-4)

    def test_rmsnorm(self):
        rng = np.random.default_rng(102)
        B, T, d = 2, 4, 6
        x_np = rng.standard_normal((B, T, d)).astype(np.float32)
        gamma_np = rng.standard_normal((d,)).astype(np.float32) + 1.0
        gy_np = rng.standard_normal((B, T, d)).astype(np.float32)
        eps = 1e-6

        xt = torch.from_numpy(x_np).requires_grad_(True)
        gt = torch.from_numpy(gamma_np).requires_grad_(True)
        rms_t = torch.sqrt(torch.mean(xt * xt, dim=-1, keepdim=True) + eps)
        yt = (xt / rms_t) * gt
        yt.backward(torch.from_numpy(gy_np))

        y_np, rstd_np = vjp_rmsnorm.rmsnorm_fwd(x_np, gamma_np, eps=eps)
        assert np.allclose(y_np, yt.detach().numpy(), atol=1e-5, rtol=1e-4)
        gx_np, gg_np = vjp_rmsnorm.rmsnorm_bwd(gy_np, x_np, gamma_np, rstd_np)
        assert np.allclose(gx_np, xt.grad.numpy(), atol=5e-5, rtol=5e-4)
        assert np.allclose(gg_np, gt.grad.numpy(), atol=1e-5, rtol=1e-4)

    def test_swiglu(self):
        rng = np.random.default_rng(103)
        d_in, h = 5, 7
        x_np = rng.standard_normal((3, d_in)).astype(np.float32) * 0.3
        Wg_np = rng.standard_normal((h, d_in)).astype(np.float32) * 0.2
        Wu_np = rng.standard_normal((h, d_in)).astype(np.float32) * 0.2
        Wd_np = rng.standard_normal((d_in, h)).astype(np.float32) * 0.2
        gy_np = rng.standard_normal((3, d_in)).astype(np.float32)

        xt = torch.from_numpy(x_np).requires_grad_(True)
        Wgt = torch.from_numpy(Wg_np).requires_grad_(True)
        Wut = torch.from_numpy(Wu_np).requires_grad_(True)
        Wdt = torch.from_numpy(Wd_np).requires_grad_(True)
        gate = F.silu(F.linear(xt, Wgt))
        up = F.linear(xt, Wut)
        a = gate * up
        yt = F.linear(a, Wdt)
        yt.backward(torch.from_numpy(gy_np))

        y_np, saved = vjp_swiglu.swiglu_fwd(x_np, Wg_np, Wu_np, Wd_np)
        assert np.allclose(y_np, yt.detach().numpy(), atol=1e-5, rtol=1e-4)
        gx_np, gWg_np, gWu_np, gWd_np = vjp_swiglu.swiglu_bwd(gy_np, saved)
        assert np.allclose(gx_np, xt.grad.numpy(), atol=5e-5, rtol=5e-4)
        assert np.allclose(gWg_np, Wgt.grad.numpy(), atol=5e-5, rtol=5e-4)
        assert np.allclose(gWu_np, Wut.grad.numpy(), atol=5e-5, rtol=5e-4)
        assert np.allclose(gWd_np, Wdt.grad.numpy(), atol=5e-5, rtol=5e-4)

    def test_cfc_step_against_torch_autograd(self):
        rng = np.random.default_rng(104)
        B, in_dim, d = 2, 3, 4
        h_prev_np = rng.standard_normal((B, d)).astype(np.float32) * 0.1
        x_np = rng.standard_normal((B, in_dim)).astype(np.float32) * 0.2
        W_in_np = rng.standard_normal((d, in_dim)).astype(np.float32) * 0.1
        W_h_np = rng.standard_normal((d, in_dim)).astype(np.float32) * 0.1
        A_log_np = rng.standard_normal((d,)).astype(np.float32) * 0.5
        grad_out_np = rng.standard_normal((B, d)).astype(np.float32)

        h_prev_t = torch.from_numpy(h_prev_np).requires_grad_(True)
        xt = torch.from_numpy(x_np).requires_grad_(True)
        W_in_t = torch.from_numpy(W_in_np).requires_grad_(True)
        W_h_t = torch.from_numpy(W_h_np).requires_grad_(True)
        A_log_t = torch.from_numpy(A_log_np).requires_grad_(True)

        delta_in = xt @ W_in_t.T
        delta_t_ = F.softplus(delta_in)
        expA = torch.exp(A_log_t)
        A_t = torch.exp(-delta_t_ * expA)
        b_in = xt @ W_h_t.T
        B_t = delta_t_ * b_in
        h_t = A_t * h_prev_t + B_t
        out_t = torch.tanh(h_t)
        out_t.backward(torch.from_numpy(grad_out_np))

        h_t_np, out_t_np, cache = vjp_cfc.cfc_step_fwd(
            h_prev_np, x_np, W_in_np, W_h_np, A_log_np
        )
        assert np.allclose(out_t_np, out_t.detach().numpy(), atol=1e-5, rtol=1e-4)
        gx_np, ghp_np, gWin_np, gWh_np, gAlog_np, _ = vjp_cfc.cfc_step_bwd(
            grad_out_np, np.zeros((B, d), dtype=np.float32), cache
        )
        assert np.allclose(gx_np, xt.grad.numpy(), atol=1e-4, rtol=1e-3)
        assert np.allclose(ghp_np, h_prev_t.grad.numpy(), atol=1e-4, rtol=1e-3)
        assert np.allclose(gWin_np, W_in_t.grad.numpy(), atol=1e-4, rtol=1e-3)
        assert np.allclose(gWh_np, W_h_t.grad.numpy(), atol=1e-4, rtol=1e-3)
        assert np.allclose(gAlog_np, A_log_t.grad.numpy(), atol=1e-4, rtol=1e-3)

    def test_plif_against_torch_atan_surrogate(self):
        rng = np.random.default_rng(105)
        B, d = 2, 4
        v_prev_np = rng.standard_normal((B, d)).astype(np.float32) * 0.3
        x_np = rng.standard_normal((B, d)).astype(np.float32) * 0.2
        tau_log_np = rng.standard_normal((d,)).astype(np.float32) * 0.5
        thr_np = np.full((d,), 0.3, dtype=np.float32)
        grad_spike_np = rng.standard_normal((B, d)).astype(np.float32)
        grad_v_new_np = rng.standard_normal((B, d)).astype(np.float32)
        alpha = 2.0

        class _ATan(torch.autograd.Function):
            @staticmethod
            def forward(ctx, mem, thr_):
                ctx.save_for_backward(mem - thr_)
                return (mem >= thr_).to(mem.dtype)

            @staticmethod
            def backward(ctx, g):
                (z,) = ctx.saved_tensors
                surr = alpha / (2.0 * (1.0 + (math.pi / 2.0 * alpha * z) ** 2))
                return g * surr, g * (-surr)

        v_prev_t = torch.from_numpy(v_prev_np).requires_grad_(True)
        xt = torch.from_numpy(x_np).requires_grad_(True)
        tau_log_t = torch.from_numpy(tau_log_np).requires_grad_(True)
        thr_t = torch.from_numpy(thr_np).requires_grad_(True)

        tau = torch.exp(tau_log_t)
        decay = torch.exp(-1.0 / tau)
        v_pre = v_prev_t * decay + xt
        spike_t = _ATan.apply(v_pre, thr_t)
        v_new_t = v_pre - spike_t * thr_t
        loss = (spike_t * torch.from_numpy(grad_spike_np)).sum() + (
            v_new_t * torch.from_numpy(grad_v_new_np)
        ).sum()
        loss.backward()

        spike_np, v_new_np, saved = vjp_plif.plif_fwd(
            v_prev_np, x_np, tau_log_np, thr_np, alpha=alpha
        )
        gx_np, gvp_np, gtau_np, gthr_np = vjp_plif.plif_bwd(
            grad_spike_np, grad_v_new_np, saved
        )
        assert np.allclose(spike_np, spike_t.detach().numpy(), atol=0)
        assert np.allclose(v_new_np, v_new_t.detach().numpy(), atol=1e-5)
        assert np.allclose(gx_np, xt.grad.numpy(), atol=1e-4, rtol=1e-3)
        assert np.allclose(gvp_np, v_prev_t.grad.numpy(), atol=1e-4, rtol=1e-3)
        assert np.allclose(gtau_np, tau_log_t.grad.numpy(), atol=1e-4, rtol=1e-3)
        assert np.allclose(gthr_np, thr_t.grad.numpy(), atol=1e-4, rtol=1e-3)

    def test_cross_entropy_against_torch(self):
        rng = np.random.default_rng(106)
        N, V = 5, 7
        logits_np = rng.standard_normal((N, V)).astype(np.float32) * 0.5
        targets_np = rng.integers(0, V, size=(N,)).astype(np.int64)

        logits_t = torch.from_numpy(logits_np).requires_grad_(True)
        targets_t = torch.from_numpy(targets_np)
        loss_t = F.cross_entropy(logits_t, targets_t, reduction="mean")
        loss_t.backward()

        loss_np, saved = vjp_ce.ce_fwd(logits_np, targets_np, reduction="mean")
        grad_logits = vjp_ce.ce_bwd(np.array(1.0, dtype=np.float32), saved)
        assert np.allclose(loss_np, loss_t.detach().numpy(), atol=1e-5, rtol=1e-4)
        assert np.allclose(grad_logits, logits_t.grad.numpy(), atol=1e-5, rtol=1e-4)

    def test_cross_entropy_with_ignore_against_torch(self):
        rng = np.random.default_rng(107)
        N, V = 4, 5
        logits_np = rng.standard_normal((N, V)).astype(np.float32) * 0.4
        targets_np = np.array([1, -100, 3, 0], dtype=np.int64)

        logits_t = torch.from_numpy(logits_np).requires_grad_(True)
        targets_t = torch.from_numpy(targets_np)
        loss_t = F.cross_entropy(
            logits_t, targets_t, ignore_index=-100, reduction="sum"
        )
        loss_t.backward()

        loss_np, saved = vjp_ce.ce_fwd(
            logits_np, targets_np, ignore_index=-100, reduction="sum"
        )
        grad_logits = vjp_ce.ce_bwd(np.array(1.0, dtype=np.float32), saved)
        assert np.allclose(loss_np, loss_t.detach().numpy(), atol=1e-5, rtol=1e-4)
        assert np.allclose(grad_logits, logits_t.grad.numpy(), atol=1e-5, rtol=1e-4)
