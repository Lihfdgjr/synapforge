"""Tests for synapforge/native/cuda/tensor.py + ops/allocator/streams.

Skips GPU-only assertions when cupy isn't available (CI without GPU).
Numpy-fallback path is exercised on every host so the wrapper API is
always covered.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Some hosts have a synapforge package init that pulls in torch. The
# native.cuda subpackage is torch-free; load it without going through
# the parent package so this test file runs even on bare CI.
import importlib.util
import types


def _load_cuda_subpackage():
    """Manually load synapforge.native.cuda without triggering parent init."""
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
    ops_mod = _load("synapforge.native.cuda.ops", "ops.py")
    alloc_mod = _load("synapforge.native.cuda.allocator", "allocator.py")
    streams_mod = _load("synapforge.native.cuda.streams", "streams.py")
    return tensor_mod, ops_mod, alloc_mod, streams_mod


tensor, ops, allocator, streams = _load_cuda_subpackage()
CudaTensor = tensor.CudaTensor

try:
    import cupy as cp  # noqa: F401
    _HAS_CUPY = True
    try:
        cp.zeros(1, dtype=cp.float32) + 0
    except Exception:
        _HAS_CUPY = False
except Exception:
    _HAS_CUPY = False


# ===========================================================================
# CudaTensor surface
# ===========================================================================


class TestCudaTensorSurface:
    def test_zeros_shape_dtype(self):
        x = CudaTensor.zeros((3, 4), dtype=np.float32)
        assert x.shape == (3, 4)
        assert x.dtype == np.float32
        assert x.size == 12
        assert x.ndim == 2
        assert x.nbytes == 12 * 4

    def test_ones_factory(self):
        x = CudaTensor.ones((2, 2))
        np.testing.assert_array_equal(x.to_cpu(), np.ones((2, 2), dtype=np.float32))

    def test_full_factory(self):
        x = CudaTensor.full((3,), 7.0)
        np.testing.assert_array_equal(x.to_cpu(), np.full((3,), 7.0, dtype=np.float32))

    def test_arange_factory(self):
        x = CudaTensor.arange(5)
        np.testing.assert_array_equal(x.to_cpu(), np.arange(5, dtype=np.float32))

    def test_randn_seed_repro(self):
        a = CudaTensor.randn((4, 4), seed=42)
        b = CudaTensor.randn((4, 4), seed=42)
        np.testing.assert_array_equal(a.to_cpu(), b.to_cpu())

    def test_from_cpu_roundtrip(self):
        host = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = CudaTensor.from_cpu(host)
        out = t.to_cpu()
        np.testing.assert_array_equal(out, host)
        # to_cpu must return a fresh copy (mutating doesn't poke the device).
        out[0, 0] = 999.0
        assert t.to_cpu()[0, 0] == 1.0

    def test_array_protocol(self):
        x = CudaTensor.from_cpu(np.array([1, 2, 3], dtype=np.float32))
        # np.asarray should work via __array__
        host = np.asarray(x)
        np.testing.assert_array_equal(host, np.array([1, 2, 3], dtype=np.float32))

    def test_reshape_transpose(self):
        x = CudaTensor.arange(12).reshape(3, 4)
        assert x.shape == (3, 4)
        y = x.transpose(1, 0)
        assert y.shape == (4, 3)

    def test_astype(self):
        x = CudaTensor.arange(5)
        y = x.astype(np.int64)
        assert y.dtype == np.int64

    def test_data_ptr_int(self):
        x = CudaTensor.zeros((4,))
        # Always returns int (0 on CPU fallback, real ptr on cupy).
        assert isinstance(x.data_ptr, int)
        if _HAS_CUPY and x.is_cuda:
            assert x.data_ptr != 0

    def test_pinned_alloc_writable(self):
        host = CudaTensor.pinned_alloc((8,), dtype=np.float32)
        host[...] = np.arange(8, dtype=np.float32)
        np.testing.assert_array_equal(host, np.arange(8, dtype=np.float32))


# ===========================================================================
# ops vs numpy reference (always-on; cupy / numpy fallback is identical math)
# ===========================================================================


class TestOps:
    def test_matmul(self):
        np.random.seed(0)
        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(16, 32).astype(np.float32)
        ref = A @ B
        out = ops.matmul(CudaTensor.from_cpu(A), CudaTensor.from_cpu(B))
        np.testing.assert_allclose(out.to_cpu(), ref, atol=1e-4)

    def test_bmm(self):
        np.random.seed(1)
        A = np.random.randn(4, 8, 16).astype(np.float32)
        B = np.random.randn(4, 16, 32).astype(np.float32)
        ref = np.matmul(A, B)
        out = ops.bmm(CudaTensor.from_cpu(A), CudaTensor.from_cpu(B))
        np.testing.assert_allclose(out.to_cpu(), ref, atol=1e-4)

    def test_gemm_transb(self):
        np.random.seed(2)
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(16, 8).astype(np.float32)
        ref = A @ B.T
        out = ops.gemm(CudaTensor.from_cpu(A), CudaTensor.from_cpu(B), transb=True)
        np.testing.assert_allclose(out.to_cpu(), ref, atol=1e-4)

    def test_silu(self):
        x = np.linspace(-3, 3, 32).astype(np.float32)
        ref = x * (1.0 / (1.0 + np.exp(-x)))
        out = ops.silu(CudaTensor.from_cpu(x))
        np.testing.assert_allclose(out.to_cpu(), ref, atol=1e-6)

    def test_gelu_tanh(self):
        x = np.linspace(-3, 3, 16).astype(np.float32)
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x**3)
        ref = 0.5 * x * (1.0 + np.tanh(inner))
        out = ops.gelu(CudaTensor.from_cpu(x), approximate="tanh")
        np.testing.assert_allclose(out.to_cpu(), ref, atol=1e-6)

    def test_sigmoid_tanh(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        np.testing.assert_allclose(
            ops.sigmoid(CudaTensor.from_cpu(x)).to_cpu(),
            1.0 / (1.0 + np.exp(-x)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            ops.tanh(CudaTensor.from_cpu(x)).to_cpu(),
            np.tanh(x),
            atol=1e-6,
        )

    def test_reductions(self):
        np.random.seed(3)
        x = np.random.randn(4, 8).astype(np.float32)
        t = CudaTensor.from_cpu(x)
        np.testing.assert_allclose(ops.sum(t).to_cpu(), x.sum(), atol=1e-4)
        np.testing.assert_allclose(ops.mean(t).to_cpu(), x.mean(), atol=1e-5)
        np.testing.assert_allclose(ops.var(t).to_cpu(), x.var(), atol=1e-5)
        np.testing.assert_allclose(ops.max(t).to_cpu(), x.max(), atol=0)
        # axis arg
        np.testing.assert_allclose(
            ops.sum(t, axis=1).to_cpu(), x.sum(axis=1), atol=1e-4
        )

    def test_argmax(self):
        x = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float32)
        out = ops.argmax(CudaTensor.from_cpu(x), axis=1).to_cpu()
        np.testing.assert_array_equal(out, np.array([2, 2]))

    def test_topk(self):
        np.random.seed(4)
        x = np.random.randn(2, 16).astype(np.float32)
        vals, idx = ops.topk(CudaTensor.from_cpu(x), k=4, axis=1)
        # ours returns descending; verify against sorted-descending.
        ref_idx_desc = np.argsort(-x, axis=1)[:, :4]
        ref_vals_desc = np.take_along_axis(x, ref_idx_desc, axis=1)
        np.testing.assert_allclose(vals.to_cpu(), ref_vals_desc, atol=1e-6)
        # ensure indices are ours subset of correct top-k (set equality per row)
        for r in range(2):
            assert set(idx.to_cpu()[r].tolist()) == set(ref_idx_desc[r].tolist())

    def test_addcmul_addcdiv(self):
        a = CudaTensor.from_cpu(np.ones(4, dtype=np.float32))
        b = CudaTensor.from_cpu(np.full(4, 2.0, dtype=np.float32))
        c = CudaTensor.from_cpu(np.full(4, 3.0, dtype=np.float32))
        out = ops.addcmul(a, b, c, value=0.5)
        # 1 + 0.5 * 2 * 3 = 4
        np.testing.assert_allclose(out.to_cpu(), np.full(4, 4.0), atol=1e-6)
        out2 = ops.addcdiv(a, b, c, value=0.5)
        # 1 + 0.5 * 2 / 3 = 1.333...
        np.testing.assert_allclose(out2.to_cpu(), np.full(4, 1 + 1.0 / 3), atol=1e-5)

    def test_layernorm(self):
        np.random.seed(5)
        x = np.random.randn(4, 8).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        out = ops.layernorm(
            CudaTensor.from_cpu(x),
            gamma=CudaTensor.from_cpu(gamma),
            beta=CudaTensor.from_cpu(beta),
            eps=1e-5,
        ).to_cpu()
        # Each row should be ~zero mean, ~unit variance.
        np.testing.assert_allclose(out.mean(axis=-1), np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(out.std(axis=-1), np.ones(4), atol=1e-2)

    def test_rmsnorm(self):
        x = np.random.randn(4, 8).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32)
        out = ops.rmsnorm(CudaTensor.from_cpu(x), gamma=CudaTensor.from_cpu(gamma)).to_cpu()
        # rms(out_row) ~ 1
        rms = np.sqrt((out * out).mean(axis=-1))
        np.testing.assert_allclose(rms, np.ones(4), atol=1e-2)

    def test_softmax_stable(self):
        # Big values shouldn't overflow.
        x = np.array([[1e3, 1e3 + 1, 1e3 + 2]], dtype=np.float32)
        out = ops.softmax(CudaTensor.from_cpu(x)).to_cpu()
        ref = np.exp(x - x.max()); ref /= ref.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(out, ref, atol=1e-6)
        # rows sum to 1
        assert abs(out.sum() - 1.0) < 1e-6


# ===========================================================================
# Allocator
# ===========================================================================


class TestAllocator:
    def test_alloc_free_cycle(self):
        pool = allocator.CudaMemPool()
        p1 = pool.alloc(1024)
        s = pool.stats()
        assert s.alloc_calls == 1
        assert s.current_bytes == 1024
        assert s.peak_bytes == 1024
        pool.free(p1)
        s = pool.stats()
        assert s.free_calls == 1
        assert s.current_bytes == 0
        assert s.peak_bytes == 1024

    def test_cache_reuse(self):
        pool = allocator.CudaMemPool()
        p1 = pool.alloc(64)
        pool.free(p1)
        p2 = pool.alloc(64)
        # Should be served from cache (same id-recycled buffer).
        s = pool.stats()
        assert s.cached_blocks == 0  # bucket popped
        assert s.alloc_calls == 2  # both calls counted

    def test_peak_bytes_high_water(self):
        pool = allocator.CudaMemPool()
        p1 = pool.alloc(1000)
        p2 = pool.alloc(2000)  # peak=3000
        pool.free(p1)
        # peak persists even after free.
        assert pool.stats().peak_bytes == 3000

    def test_release_all(self):
        pool = allocator.CudaMemPool()
        p1 = pool.alloc(64); pool.free(p1)
        pool.release_all()
        # cache should be empty
        assert pool.stats().cached_blocks == 0

    def test_alloc_array_shape(self):
        pool = allocator.CudaMemPool()
        arr = pool.alloc_array((4, 8), dtype=np.float32)
        # Both numpy and cupy ndarrays expose .shape and .dtype.
        assert tuple(arr.shape) == (4, 8)
        assert arr.dtype == np.float32


# ===========================================================================
# Streams
# ===========================================================================


class TestStreams:
    def test_pool_construct(self):
        sp = streams.CudaStreamPool()
        assert sp.compute is not None
        assert sp.h2d is not None
        assert sp.d2h is not None
        assert sp.misc is not None

    def test_iter(self):
        sp = streams.CudaStreamPool()
        names = [s.name for s in sp]
        assert names == ["compute", "h2d", "d2h", "misc"]

    def test_context_no_op_on_cpu(self):
        # Smoke: context manager must work even on numpy fallback.
        sp = streams.CudaStreamPool()
        with sp.compute.context() as s:
            assert s is sp.compute
        sp.synchronize_all()  # no-op on CPU

    def test_by_name(self):
        sp = streams.CudaStreamPool()
        assert sp.by_name("compute") is sp.compute
        assert sp.by_name("misc") is sp.misc


# ===========================================================================
# GPU-only assertions
# ===========================================================================


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not available -- GPU-only test")
class TestGpuPath:
    def test_cuda_device_string(self):
        x = CudaTensor.zeros((4,))
        assert x.device.startswith("cuda:")

    def test_data_ptr_nonzero_on_gpu(self):
        x = CudaTensor.zeros((128,))
        assert x.data_ptr != 0

    def test_pinned_alloc_pinned(self):
        # Allocating > 0 bytes should succeed; numpy.frombuffer must return
        # a writable buffer.
        host = CudaTensor.pinned_alloc((1024,), dtype=np.float32)
        host[0] = 7.0
        assert host[0] == 7.0
