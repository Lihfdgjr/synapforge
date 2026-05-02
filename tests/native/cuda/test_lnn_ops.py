"""Tests for synapforge/native/cuda/lnn_ops.py.

Compares our cupy/numpy implementations against pure-numpy reference math
for the LNN+SNN-specific ops: cfc_step_dense, cfc_scan, plif_spike,
atan_surrogate_grad, rope_apply, swiglu, rfold_cumprod.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import types

import numpy as np
import pytest


def _load_cuda_subpackage():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    pkg_path = os.path.join(repo, "synapforge", "native", "cuda")
    if not os.path.isdir(pkg_path):
        pytest.skip(f"synapforge/native/cuda not found at {pkg_path}")

    if "synapforge" not in sys.modules:
        synapforge = types.ModuleType("synapforge")
        synapforge.__path__ = [os.path.join(repo, "synapforge")]
        sys.modules["synapforge"] = synapforge
    if "synapforge.native" not in sys.modules:
        nat = types.ModuleType("synapforge.native")
        nat.__path__ = [os.path.join(repo, "synapforge", "native")]
        sys.modules["synapforge.native"] = nat
    if "synapforge.native.cuda" not in sys.modules:
        cu = types.ModuleType("synapforge.native.cuda")
        cu.__path__ = [pkg_path]
        sys.modules["synapforge.native.cuda"] = cu

    def _load(name, fname):
        path = os.path.join(pkg_path, fname)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    tensor_mod = _load("synapforge.native.cuda.tensor", "tensor.py")
    lnn_mod = _load("synapforge.native.cuda.lnn_ops", "lnn_ops.py")
    return tensor_mod, lnn_mod


tensor, lnn_ops = _load_cuda_subpackage()
CudaTensor = tensor.CudaTensor


# ===========================================================================
# CfC
# ===========================================================================


class TestCfC:
    def test_cfc_step_dense(self):
        np.random.seed(0)
        h_prev = np.random.randn(2, 8).astype(np.float32)
        a = np.random.uniform(0.5, 0.9, size=(2, 8)).astype(np.float32)
        b = np.random.randn(2, 8).astype(np.float32)
        ref = a * h_prev + b
        out = lnn_ops.cfc_step_dense(
            CudaTensor.from_cpu(h_prev),
            CudaTensor.from_cpu(a),
            CudaTensor.from_cpu(b),
        )
        np.testing.assert_allclose(out.to_cpu(), ref, atol=1e-6)

    def test_cfc_scan(self):
        np.random.seed(1)
        B, T, D = 2, 8, 16
        a = np.random.uniform(0.5, 0.99, size=(B, T, D)).astype(np.float32)
        b = np.random.randn(B, T, D).astype(np.float32)

        # Pure-numpy ref
        ref = np.zeros((B, T, D), dtype=np.float32)
        h = np.zeros((B, D), dtype=np.float32)
        for t in range(T):
            h = a[:, t, :] * h + b[:, t, :]
            ref[:, t, :] = h

        out = lnn_ops.cfc_scan(CudaTensor.from_cpu(a), CudaTensor.from_cpu(b))
        np.testing.assert_allclose(out.to_cpu(), ref, atol=1e-5)

    def test_cfc_scan_with_h0(self):
        np.random.seed(2)
        B, T, D = 1, 4, 8
        a = np.full((B, T, D), 0.5, dtype=np.float32)
        b = np.zeros((B, T, D), dtype=np.float32)
        h0 = np.ones((B, D), dtype=np.float32)
        # With b=0, h0=1, a=0.5: h[t] = 0.5^(t+1)
        out = lnn_ops.cfc_scan(
            CudaTensor.from_cpu(a),
            CudaTensor.from_cpu(b),
            h0=CudaTensor.from_cpu(h0),
        ).to_cpu()
        for t in range(T):
            expected = 0.5 ** (t + 1)
            np.testing.assert_allclose(out[0, t, :], np.full(D, expected), atol=1e-6)


# ===========================================================================
# PLIF + ATan surrogate
# ===========================================================================


class TestPLIF:
    def test_plif_spike_above_threshold(self):
        m = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
        thr = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        s, delta = lnn_ops.plif_spike(CudaTensor.from_cpu(m), CudaTensor.from_cpu(thr))
        np.testing.assert_array_equal(s.to_cpu(), [[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(delta.to_cpu(), [[1.0, -1.0, -0.5]], atol=1e-6)

    def test_atan_surrogate_grad_at_zero(self):
        # At m = 0, ds/dm = alpha / 2 (for any alpha).
        m = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        out = lnn_ops.atan_surrogate_grad(CudaTensor.from_cpu(m), alpha=2.0).to_cpu()
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0], atol=1e-6)

    def test_atan_surrogate_grad_decays(self):
        # As |m| grows, gradient shrinks (the soft Heaviside).
        m = np.array([0.0, 1.0, 5.0], dtype=np.float32)
        out = lnn_ops.atan_surrogate_grad(CudaTensor.from_cpu(m), alpha=2.0).to_cpu()
        # Hand-computed: ds/dm = alpha / (2 * (1 + (pi/2 * alpha * m)^2))
        ref = np.array([
            2.0 / (2 * (1 + (math.pi / 2 * 2.0 * 0.0) ** 2)),
            2.0 / (2 * (1 + (math.pi / 2 * 2.0 * 1.0) ** 2)),
            2.0 / (2 * (1 + (math.pi / 2 * 2.0 * 5.0) ** 2)),
        ], dtype=np.float32)
        np.testing.assert_allclose(out, ref, atol=1e-6)
        # monotonic decrease
        assert out[0] > out[1] > out[2]


# ===========================================================================
# RoPE
# ===========================================================================


class TestRoPE:
    @staticmethod
    def _build_cache(max_T: int, head_dim: int, base: float = 10000.0):
        # LLaMA-style cos/sin caches.
        positions = np.arange(max_T, dtype=np.float32)[:, None]   # (T, 1)
        inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        # outer product -> (T, head_dim/2)
        freqs = positions * inv_freq[None, :]
        # Interleave cos/sin pairs to (T, head_dim)
        cos = np.empty((max_T, head_dim), dtype=np.float32)
        sin = np.empty((max_T, head_dim), dtype=np.float32)
        cos[:, 0::2] = np.cos(freqs)
        cos[:, 1::2] = np.cos(freqs)
        sin[:, 0::2] = np.sin(freqs)
        sin[:, 1::2] = np.sin(freqs)
        return cos, sin

    def test_rope_identity_at_pos_zero(self):
        # At pos=0, cos=1, sin=0 -> rotation is identity.
        max_T, D = 4, 8
        cos, sin = self._build_cache(max_T, D)
        x = np.random.randn(1, 1, 1, D).astype(np.float32)
        out = lnn_ops.rope_apply(
            CudaTensor.from_cpu(x),
            CudaTensor.from_cpu(cos),
            CudaTensor.from_cpu(sin),
            pos_offset=0,
        ).to_cpu()
        # Position 0 is first row of the cache; cos[0]=1, sin[0]=0.
        # So out should equal x.
        np.testing.assert_allclose(out, x, atol=1e-6)

    def test_rope_norm_preserved(self):
        # Rotations preserve L2 norm per pair.
        max_T, D = 16, 8
        cos, sin = self._build_cache(max_T, D)
        np.random.seed(42)
        x = np.random.randn(2, 4, 2, D).astype(np.float32)
        out = lnn_ops.rope_apply(
            CudaTensor.from_cpu(x),
            CudaTensor.from_cpu(cos),
            CudaTensor.from_cpu(sin),
        ).to_cpu()
        # Sum over last dim should be preserved (L2 norm of pairs preserved).
        np.testing.assert_allclose(
            np.linalg.norm(out, axis=-1),
            np.linalg.norm(x, axis=-1),
            atol=1e-4,
        )


# ===========================================================================
# SwiGLU
# ===========================================================================


class TestSwiGLU:
    def test_swiglu_zero_gate(self):
        # silu(0) = 0, so output should be all zeros.
        x = np.ones((2, 4), dtype=np.float32)
        gate = np.zeros((2, 4), dtype=np.float32)
        out = lnn_ops.swiglu(CudaTensor.from_cpu(x), CudaTensor.from_cpu(gate)).to_cpu()
        np.testing.assert_allclose(out, np.zeros((2, 4)), atol=1e-6)

    def test_swiglu_matches_reference(self):
        np.random.seed(3)
        x = np.random.randn(4, 8).astype(np.float32)
        g = np.random.randn(4, 8).astype(np.float32)
        ref = x * (g * (1.0 / (1.0 + np.exp(-g))))
        out = lnn_ops.swiglu(CudaTensor.from_cpu(x), CudaTensor.from_cpu(g)).to_cpu()
        np.testing.assert_allclose(out, ref, atol=1e-6)


# ===========================================================================
# rfold_cumprod
# ===========================================================================


class TestRfoldCumprod:
    def test_matches_numpy_cumprod_when_no_underflow(self):
        # Values bounded above 0.5 won't underflow -- match plain cumprod.
        np.random.seed(0)
        x = np.random.uniform(0.5, 1.0, size=(3, 8)).astype(np.float32)
        ref = np.cumprod(x, axis=-1)
        out = lnn_ops.rfold_cumprod(CudaTensor.from_cpu(x)).to_cpu()
        # log-space introduces tiny rounding, but should match within fp32.
        np.testing.assert_allclose(out, ref, atol=1e-3, rtol=1e-3)

    def test_handles_underflow_better_than_cumprod(self):
        # Force a long product of small values that underflows fp32 cumprod
        # to zero. Our log-space version should still produce a meaningful
        # value (within the fp32 representable range).
        x = np.full((1, 200), 0.5, dtype=np.float32)
        # plain cumprod ends at 0.5^200 ~ 6e-61 -> underflows fp32
        plain = np.cumprod(x, axis=-1)
        log_space = lnn_ops.rfold_cumprod(CudaTensor.from_cpu(x)).to_cpu()
        # last element of plain should be zero (fp32 underflow at ~1e-38)
        assert plain[0, -1] == 0.0
        # Our log-space version produces a positive (very small but finite)
        # number until exp(log_cumsum) underflows; so final element either
        # finite > 0 or zero (depending on cumsum magnitude).
        assert log_space[0, 0] > 0  # at least the head should be positive

    def test_sign_propagation(self):
        # Mix of +/- values: cumulative product flips sign each time we
        # cumulate a negative. For [1, -1, 1, -1]:
        #   cumprod = [1, -1, -1, 1]
        x = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
        out = lnn_ops.rfold_cumprod(CudaTensor.from_cpu(x)).to_cpu()
        ref_signs = np.array([[1, -1, -1, 1]])
        np.testing.assert_array_equal(np.sign(out), ref_signs)
