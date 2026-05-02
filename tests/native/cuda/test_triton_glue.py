"""Smoke tests for synapforge/native/cuda/triton_glue.py.

Two scenarios:

1. ``test_module_imports`` -- Confirms triton_glue itself can be loaded
   on a host where torch / triton / cupy are absent. Just an import +
   probe-flag check; no kernel launch attempted.

2. ``test_dlpack_bridge_smoke`` -- When torch IS available (with CUDA),
   pushes a CudaTensor through the DLPack bridge to torch and back, and
   asserts the round-trip preserves the bytes. Uses ``torch.allclose``
   with tolerance 1e-4 fp32.

3. ``test_run_fused_lnn_snn_block_skips_without_gpu`` -- Verifies that
   on hosts without GPU (or without triton), invoking the wrapper
   raises a clean RuntimeError instead of crashing weirdly.

This file does NOT exercise the actual Triton kernel on CPU (Triton
requires CUDA) -- that smoke test is left for the GPU CI lane.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest


def _load_cuda_subpackage():
    """Load synapforge.native.cuda without triggering parent init."""
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
    glue_mod = _load("synapforge.native.cuda.triton_glue", "triton_glue.py")
    return tensor_mod, glue_mod


tensor, triton_glue = _load_cuda_subpackage()
CudaTensor = tensor.CudaTensor


try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import cupy as cp  # noqa: F401
    try:
        cp.zeros(1, dtype=cp.float32) + 0
        _HAS_CUPY = True
    except Exception:
        _HAS_CUPY = False
except Exception:
    _HAS_CUPY = False

_HAS_GPU_TORCH = False
if _HAS_TORCH:
    try:
        _HAS_GPU_TORCH = bool(torch.cuda.is_available())
    except Exception:
        _HAS_GPU_TORCH = False


# ===========================================================================
# Module-level smoke
# ===========================================================================


def test_module_imports():
    """triton_glue must be importable even without torch/triton/cupy."""
    assert hasattr(triton_glue, "HAS_TORCH")
    assert hasattr(triton_glue, "HAS_TRITON")
    assert hasattr(triton_glue, "cuda_tensor_to_torch")
    assert hasattr(triton_glue, "torch_to_cuda_tensor")
    assert hasattr(triton_glue, "run_fused_lnn_snn_block")


def test_run_fused_block_raises_without_torch_or_triton():
    """Without torch+triton, the kernel wrapper must raise RuntimeError.

    This is the CPU-CI default state; the call should fail loudly with
    a recognizable error, not silently no-op or import-error.
    """
    if triton_glue.HAS_TORCH and triton_glue.HAS_TRITON:
        pytest.skip("torch + triton both available; this test only fires on bare hosts")
    a = CudaTensor.zeros((1, 4, 8))
    b = CudaTensor.zeros((1, 4, 8))
    thr = CudaTensor.zeros((8,))
    with pytest.raises(RuntimeError, match="(torch|Triton|triton)"):
        triton_glue.run_fused_lnn_snn_block(a, b, thr)


# ===========================================================================
# torch-bridge smoke (zero-copy when cupy + torch + GPU)
# ===========================================================================


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestTorchBridge:
    def test_cuda_tensor_to_torch_cpu_path(self):
        """numpy fallback: CudaTensor -> torch.Tensor through host copy."""
        x = CudaTensor.from_cpu(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        tt = triton_glue.cuda_tensor_to_torch(x)
        assert tt.shape[0] == 3
        # Bytes preserved.
        np.testing.assert_allclose(tt.detach().cpu().numpy(), [1.0, 2.0, 3.0], atol=1e-6)

    def test_torch_to_cuda_tensor_cpu_path(self):
        import torch
        tt = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        ct = triton_glue.torch_to_cuda_tensor(tt)
        np.testing.assert_allclose(ct.to_cpu(), [4.0, 5.0, 6.0], atol=1e-6)

    def test_roundtrip_preserves_data(self):
        np.random.seed(123)
        host = np.random.randn(2, 3, 4).astype(np.float32)
        x = CudaTensor.from_cpu(host)
        tt = triton_glue.cuda_tensor_to_torch(x)
        ct = triton_glue.torch_to_cuda_tensor(tt)
        np.testing.assert_allclose(ct.to_cpu(), host, atol=1e-4)


# ===========================================================================
# Reference comparison: kernel must equal pure-numpy CfC+PLIF math
# ===========================================================================


def _ref_lnn_snn_block(a: np.ndarray, b: np.ndarray, thr: np.ndarray, h0: np.ndarray):
    """Pure-numpy reference for the fused forward kernel.

    Mirrors triton_block_kernel.fused_lnn_snn_block_kernel:
        h_t   = a_t * h_{t-1} + b_t          (CfC scan)
        m_t   = h_pre - thr                   (membrane)
        s_t   = 1 if m_t > 0 else 0           (Heaviside spike)
        h_out = h_pre * (1 - s_t)             (subtract reset)
    """
    B, T, D = a.shape
    h = h0.astype(np.float32).copy()
    h_pre = np.empty((B, T, D), dtype=np.float32)
    s = np.empty((B, T, D), dtype=np.float32)
    m = np.empty((B, T, D), dtype=np.float32)
    h_post = np.empty((B, T, D), dtype=np.float32)
    for t in range(T):
        h = a[:, t, :].astype(np.float32) * h + b[:, t, :].astype(np.float32)
        h_pre[:, t, :] = h
        m_t = h - thr.astype(np.float32)
        m[:, t, :] = m_t
        s_t = (m_t > 0.0).astype(np.float32)
        s[:, t, :] = s_t
        h = h * (1.0 - s_t)
        h_post[:, t, :] = h
    return h_pre, s, m, h_post


@pytest.mark.skipif(
    not (_HAS_TORCH and triton_glue.HAS_TRITON and _HAS_GPU_TORCH and _HAS_CUPY),
    reason="needs cupy + torch + triton + GPU to launch the real kernel",
)
def test_run_fused_lnn_snn_block_matches_numpy_reference():
    """End-to-end smoke: CudaTensor in, CudaTensor out, fp32 tolerance 1e-4.

    Only fires when the host actually has a GPU plus the full toolchain.
    On CI without GPU this test is skipped.
    """
    np.random.seed(7)
    B, T, D = 2, 8, 16
    a = np.random.uniform(0.5, 0.99, size=(B, T, D)).astype(np.float32)
    b = np.random.uniform(-0.5, 0.5, size=(B, T, D)).astype(np.float32)
    thr = np.full((D,), 0.3, dtype=np.float32)
    h0 = np.zeros((B, D), dtype=np.float32)

    ref_h_pre, ref_s, ref_m, ref_h_post = _ref_lnn_snn_block(a, b, thr, h0)

    h_pre, s, m, h_post = triton_glue.run_fused_lnn_snn_block(
        a=CudaTensor.from_cpu(a),
        b=CudaTensor.from_cpu(b),
        threshold=CudaTensor.from_cpu(thr),
        h0=CudaTensor.from_cpu(h0),
    )

    np.testing.assert_allclose(h_pre.to_cpu(), ref_h_pre, atol=1e-4)
    np.testing.assert_allclose(s.to_cpu(), ref_s, atol=1e-6)
    np.testing.assert_allclose(m.to_cpu(), ref_m, atol=1e-4)
    np.testing.assert_allclose(h_post.to_cpu(), ref_h_post, atol=1e-4)
