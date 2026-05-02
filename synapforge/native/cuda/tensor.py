"""CudaTensor -- thin wrapper over cupy.ndarray (numpy fallback when no GPU).

Goals
-----
1. Expose the same surface as ``numpy.ndarray`` and ``cupy.ndarray`` so it
   composes with the rest of synapforge.native.
2. Expose ``data_ptr`` (raw int64 CUDA pointer) for Triton / CUDA-C ABI
   interop. On the numpy fallback path, ``data_ptr`` returns 0 and any
   GPU-only path raises ``CudaUnavailableError``.
3. Round-trip with numpy: ``.to_cpu()`` and ``CudaTensor.from_cpu(arr)``.
4. Smart pinning: ``CudaTensor.pinned_alloc(shape, dtype)`` allocates a
   pinned host buffer for fast async H2D and exposes an ``async_copy_to``
   helper that uses an explicit cupy stream.

NO ``import torch`` anywhere in this file (or in this package).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as _np

try:  # pragma: no cover -- optional GPU path
    import cupy as _cp

    try:
        _cp.zeros(1, dtype=_cp.float32) + 0
        _HAS_CUPY = True
    except Exception:
        _HAS_CUPY = False
except Exception:  # pragma: no cover
    _cp = None
    _HAS_CUPY = False


__all__ = [
    "CudaTensor",
    "CudaUnavailableError",
    "as_cuda",
]


class CudaUnavailableError(RuntimeError):
    """Raised when a GPU-only path is invoked on a host without cupy."""


# Type alias for the underlying array (cupy if present, else numpy).
ArrayLike = Any  # cupy.ndarray | numpy.ndarray


def _xp():
    """Return the active array module: cupy if usable, else numpy."""
    return _cp if _HAS_CUPY else _np


class CudaTensor:
    """A torch-free GPU tensor, backed by ``cupy.ndarray``.

    Falls back to ``numpy.ndarray`` on hosts where cupy is missing or the
    CUDA runtime fails to initialize. The fallback path is intended for
    smoke tests on CI (Windows / Linux without GPU); production training
    must run on a host with cupy.

    Layout policy
    -------------
    * Default dtype = ``float32``.
    * Default device = ``cuda:0`` when cupy works, else ``cpu``.
    * C-contiguous; ``data_ptr`` is the start-of-buffer address suitable
      for Triton ``T_ptr`` arguments.

    Reference
    ---------
    cupy.ndarray API:
        https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html
    """

    __slots__ = ("_arr", "_device")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, arr: ArrayLike, device: Optional[str] = None):
        """Wrap an existing cupy or numpy ndarray.

        Prefer the factory helpers (``CudaTensor.zeros``, ``.from_cpu``,
        etc.) for new buffers; this constructor is for internal wrapping
        of arrays produced by ops / allocators / kernels.
        """
        self._arr = arr
        if device is None:
            self._device = "cuda:0" if _HAS_CUPY and _cp is not None and isinstance(arr, _cp.ndarray) else "cpu"
        else:
            self._device = device

    # ------------------------------------------------------------------
    # NumPy/cupy-style attributes
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(s) for s in self._arr.shape)

    @property
    def dtype(self) -> _np.dtype:
        # Both numpy and cupy expose ``.dtype``.
        return self._arr.dtype

    @property
    def device(self) -> str:
        """Return ``cuda:N`` when on GPU, else ``cpu``."""
        return self._device

    @property
    def size(self) -> int:
        return int(self._arr.size)

    @property
    def ndim(self) -> int:
        return int(self._arr.ndim)

    @property
    def nbytes(self) -> int:
        return int(self._arr.nbytes)

    @property
    def is_cuda(self) -> bool:
        return self._device.startswith("cuda")

    @property
    def data_ptr(self) -> int:
        """Raw int64 device pointer for Triton / CUDA-C interop.

        Returns 0 on the numpy fallback path. Tests that need the raw
        pointer should guard on ``HAS_CUPY``.
        """
        if not self.is_cuda:
            return 0
        # cupy exposes the device pointer on .data.ptr (int).
        return int(self._arr.data.ptr)

    @property
    def data(self) -> ArrayLike:
        """Return the underlying ndarray (cupy or numpy). Internal API."""
        return self._arr

    # ------------------------------------------------------------------
    # Dunder + ops compatibility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover -- cosmetic
        return (
            f"CudaTensor(shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, ptr=0x{self.data_ptr:x})"
        )

    def __array__(self, dtype: Any = None) -> _np.ndarray:
        """numpy-protocol: ``np.asarray(cuda_tensor)`` returns a CPU copy."""
        host = self.to_cpu()
        if dtype is not None:
            host = host.astype(dtype, copy=False)
        return host

    def __len__(self) -> int:
        return self.shape[0] if self.ndim > 0 else 0

    # ------------------------------------------------------------------
    # CPU <-> GPU transfer
    # ------------------------------------------------------------------

    def to_cpu(self) -> _np.ndarray:
        """Synchronous device-to-host copy. Returns a fresh numpy array."""
        if self.is_cuda:
            assert _cp is not None  # mypy
            return _cp.asnumpy(self._arr)
        return _np.asarray(self._arr).copy()

    @staticmethod
    def from_cpu(arr: _np.ndarray, dtype: Any = None) -> "CudaTensor":
        """Host-to-device copy. Returns a CudaTensor on cuda:0 (or CPU fallback)."""
        host = _np.asarray(arr)
        if dtype is not None:
            host = host.astype(dtype, copy=False)
        if _HAS_CUPY and _cp is not None:
            dev = _cp.asarray(host)
            return CudaTensor(dev, device="cuda:0")
        return CudaTensor(host.copy(), device="cpu")

    # ------------------------------------------------------------------
    # Construction helpers (numpy-style factories)
    # ------------------------------------------------------------------

    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype: Any = _np.float32) -> "CudaTensor":
        return CudaTensor(_xp().zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape: Tuple[int, ...], dtype: Any = _np.float32) -> "CudaTensor":
        return CudaTensor(_xp().ones(shape, dtype=dtype))

    @staticmethod
    def empty(shape: Tuple[int, ...], dtype: Any = _np.float32) -> "CudaTensor":
        return CudaTensor(_xp().empty(shape, dtype=dtype))

    @staticmethod
    def full(shape: Tuple[int, ...], fill_value: float, dtype: Any = _np.float32) -> "CudaTensor":
        return CudaTensor(_xp().full(shape, fill_value, dtype=dtype))

    @staticmethod
    def randn(shape: Tuple[int, ...], dtype: Any = _np.float32, seed: Optional[int] = None) -> "CudaTensor":
        if seed is not None:
            _np.random.seed(seed)
        host = _np.random.randn(*shape).astype(dtype)
        return CudaTensor.from_cpu(host)

    @staticmethod
    def arange(stop: int, dtype: Any = _np.float32) -> "CudaTensor":
        return CudaTensor(_xp().arange(stop, dtype=dtype))

    # ------------------------------------------------------------------
    # Pinned host alloc + async copy
    # ------------------------------------------------------------------

    @staticmethod
    def pinned_alloc(shape: Tuple[int, ...], dtype: Any = _np.float32) -> _np.ndarray:
        """Allocate a *pinned* (page-locked) host numpy array.

        Pinned host memory transfers ~2-3x faster H2D than pageable on a
        PCIe Gen4 link. Use this for the ground-truth source buffer that
        gets streamed to the GPU asynchronously.

        On the numpy fallback (no cupy), returns a regular numpy array.
        """
        nelem = 1
        for s in shape:
            nelem *= int(s)
        if _HAS_CUPY and _cp is not None:
            mem = _cp.cuda.alloc_pinned_memory(nelem * _np.dtype(dtype).itemsize)
            arr = _np.frombuffer(mem, dtype=dtype, count=nelem).reshape(shape)
            # Keep the cupy-managed pinned buffer alive on the array via a base ref.
            arr.base_pinned_keepalive = mem  # type: ignore[attr-defined]
            return arr
        return _np.empty(shape, dtype=dtype)

    def async_copy_from_host(self, host: _np.ndarray, stream: Any = None) -> "CudaTensor":
        """Async H2D copy from a (preferably pinned) host numpy array.

        Returns ``self`` so callers can chain. The caller must keep
        ``host`` alive until the stream synchronizes.
        """
        if not self.is_cuda or _cp is None:
            # numpy fallback: synchronous copy.
            self._arr[...] = host
            return self
        cp_stream = stream
        # If user passed a CudaStreamPool entry it already wraps a cupy
        # stream; accept either the wrapper or the raw cupy stream.
        if hasattr(cp_stream, "_cp_stream"):
            cp_stream = cp_stream._cp_stream  # type: ignore[attr-defined]
        if cp_stream is None:
            # default stream
            self._arr.set(host)
            return self
        with cp_stream:
            self._arr.set(host)
        return self

    def async_copy_to_host(self, host: _np.ndarray, stream: Any = None) -> _np.ndarray:
        """Async D2H copy into a host numpy array. Returns ``host``."""
        if not self.is_cuda or _cp is None:
            host[...] = _np.asarray(self._arr)
            return host
        cp_stream = stream
        if hasattr(cp_stream, "_cp_stream"):
            cp_stream = cp_stream._cp_stream  # type: ignore[attr-defined]
        if cp_stream is None:
            self._arr.get(out=host)
            return host
        with cp_stream:
            self._arr.get(out=host)
        return host

    # ------------------------------------------------------------------
    # Shape ops (forward to xp; return CudaTensor)
    # ------------------------------------------------------------------

    def reshape(self, *shape: int) -> "CudaTensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return CudaTensor(self._arr.reshape(*shape), device=self._device)

    def transpose(self, *axes: int) -> "CudaTensor":
        return CudaTensor(self._arr.transpose(*axes), device=self._device)

    def contiguous(self) -> "CudaTensor":
        if _HAS_CUPY and _cp is not None and isinstance(self._arr, _cp.ndarray):
            arr = _cp.ascontiguousarray(self._arr)
        else:
            arr = _np.ascontiguousarray(self._arr)
        return CudaTensor(arr, device=self._device)

    def astype(self, dtype: Any) -> "CudaTensor":
        return CudaTensor(self._arr.astype(dtype, copy=False), device=self._device)

    def copy(self) -> "CudaTensor":
        return CudaTensor(self._arr.copy(), device=self._device)


def as_cuda(x: Union[CudaTensor, _np.ndarray, Any], dtype: Any = None) -> CudaTensor:
    """Coerce ``x`` to a CudaTensor.

    * CudaTensor   -> returned unchanged (or astype'd)
    * numpy.ndarray -> from_cpu()
    * cupy.ndarray  -> wrapped in CudaTensor
    """
    if isinstance(x, CudaTensor):
        return x.astype(dtype) if dtype is not None else x
    if _HAS_CUPY and _cp is not None and isinstance(x, _cp.ndarray):
        t = CudaTensor(x, device="cuda:0")
        return t.astype(dtype) if dtype is not None else t
    return CudaTensor.from_cpu(_np.asarray(x), dtype=dtype)
