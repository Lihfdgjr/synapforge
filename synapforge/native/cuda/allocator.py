"""CudaMemPool -- simple block allocator on top of cupy.cuda.MemoryPool.

Why a wrapper?
--------------
cupy ships a caching allocator (``cupy.cuda.MemoryPool``) that already
avoids OS-level cudaMalloc round-trips for repeated same-size allocs.
But it does **not** expose:

1. A high-water mark (peak bytes ever used in this process).
2. A consistent allocate/free API that survives cupy not being
   importable -- our test suite needs to exercise the metering logic
   without a GPU.

This wrapper adds both, plus a tiny block cache keyed on
``(nbytes, dtype)`` so allocate-then-free-then-allocate of the same
shape stays in-process. It is a *thin* layer; cupy's pool does the
real heavy lifting.

Honest limitations
------------------
* Not thread-safe (synapforge.native is single-threaded today).
* Does not recover memory on Python GC -- cupy's pool already does
  that on its own MemoryPointer __del__.
* No fragmentation telemetry; for that, fall through to
  ``cupy.cuda.MemoryPool.used_bytes()`` + ``total_bytes()``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

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


__all__ = ["CudaMemPool", "PoolStats"]


class PoolStats:
    """Snapshot of pool counters at a point in time."""

    __slots__ = ("alloc_calls", "free_calls", "current_bytes", "peak_bytes", "cached_blocks")

    def __init__(
        self,
        alloc_calls: int,
        free_calls: int,
        current_bytes: int,
        peak_bytes: int,
        cached_blocks: int,
    ):
        self.alloc_calls = alloc_calls
        self.free_calls = free_calls
        self.current_bytes = current_bytes
        self.peak_bytes = peak_bytes
        self.cached_blocks = cached_blocks

    def as_dict(self) -> Dict[str, int]:
        return {
            "alloc_calls": self.alloc_calls,
            "free_calls": self.free_calls,
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
            "cached_blocks": self.cached_blocks,
        }

    def __repr__(self) -> str:  # pragma: no cover -- cosmetic
        return (
            f"PoolStats(alloc={self.alloc_calls}, free={self.free_calls}, "
            f"current={self.current_bytes / 1024**2:.2f} MiB, "
            f"peak={self.peak_bytes / 1024**2:.2f} MiB, "
            f"cached_blocks={self.cached_blocks})"
        )


class CudaMemPool:
    """Block allocator wrapping cupy.cuda.MemoryPool.

    Usage::

        pool = CudaMemPool()
        # cupy.ndarray-shaped allocs:
        ptr = pool.alloc(256 * 256 * 4)        # 256x256 fp32
        pool.free(ptr)
        print(pool.stats().peak_bytes)

    For real workloads, prefer ``CudaTensor.zeros(shape)`` etc. which
    routes through the default cupy memory pool. This class is for
    callers that need an explicit pool boundary (e.g. a per-layer pool
    that you can ``release_all()`` between training steps).
    """

    def __init__(self):
        self._alloc_calls = 0
        self._free_calls = 0
        self._current_bytes = 0
        self._peak_bytes = 0
        # ``_cache[nbytes] = list of pointers`` for very simple slab reuse.
        # cupy.MemoryPool already does this internally; the wrapper-level
        # cache lets us count cache hits / misses for instrumentation.
        self._cache: Dict[int, List[object]] = {}
        self._live: Dict[int, Tuple[object, int]] = {}

        if _HAS_CUPY and _cp is not None:
            # Use cupy's default memory pool. This is shared across cupy
            # tensors so allocations made through plain CudaTensor.zeros
            # also benefit.
            self._cp_pool = _cp.get_default_memory_pool()
        else:
            self._cp_pool = None

    # ------------------------------------------------------------------
    # Allocation API
    # ------------------------------------------------------------------

    def alloc(self, nbytes: int) -> object:
        """Allocate ``nbytes`` of GPU memory.

        Returns a cupy.cuda.MemoryPointer-like object (or, on the numpy
        fallback, a numpy ``bytearray`` of the same length). Use
        ``free(ptr)`` to release.
        """
        nbytes = int(nbytes)
        # Cache hit?
        bucket = self._cache.get(nbytes)
        if bucket:
            ptr = bucket.pop()
            self._track(ptr, nbytes)
            self._alloc_calls += 1
            return ptr

        # Real alloc.
        if _HAS_CUPY and _cp is not None:
            assert self._cp_pool is not None  # narrows for mypy
            ptr = self._cp_pool.malloc(nbytes)
        else:
            ptr = bytearray(nbytes)  # CPU fallback (deterministic for tests)

        self._track(ptr, nbytes)
        self._alloc_calls += 1
        return ptr

    def free(self, ptr: object) -> None:
        """Return a previously-allocated pointer to the cache."""
        rec = self._live.pop(id(ptr), None)
        if rec is None:
            # Foreign pointer -- not ours, ignore.
            return
        _, nbytes = rec
        self._current_bytes -= nbytes
        self._cache.setdefault(nbytes, []).append(ptr)
        self._free_calls += 1

    def release_all(self) -> None:
        """Drop all cached blocks (does NOT touch live allocations)."""
        # Drop our cache. cupy's underlying pool will GC them lazily.
        self._cache.clear()
        if _HAS_CUPY and _cp is not None and self._cp_pool is not None:
            self._cp_pool.free_all_blocks()

    # ------------------------------------------------------------------
    # Convenience helpers (shape-aware)
    # ------------------------------------------------------------------

    def alloc_array(self, shape: Tuple[int, ...], dtype=_np.float32) -> object:
        """Allocate a cupy.ndarray-sized buffer, return the cupy array."""
        nbytes = int(_np.prod(shape)) * _np.dtype(dtype).itemsize
        if _HAS_CUPY and _cp is not None:
            mem_ptr = self._cp_pool.malloc(nbytes) if self._cp_pool is not None else None
            if mem_ptr is None:
                arr = _cp.empty(shape, dtype=dtype)
            else:
                # Build a cupy ndarray over the pre-allocated pointer.
                arr = _cp.ndarray(shape, dtype=dtype, memptr=mem_ptr)
            self._track(arr, nbytes)
            self._alloc_calls += 1
            return arr
        # CPU fallback
        arr = _np.empty(shape, dtype=dtype)
        self._track(arr, nbytes)
        self._alloc_calls += 1
        return arr

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> PoolStats:
        cached_blocks = sum(len(v) for v in self._cache.values())
        return PoolStats(
            alloc_calls=self._alloc_calls,
            free_calls=self._free_calls,
            current_bytes=self._current_bytes,
            peak_bytes=self._peak_bytes,
            cached_blocks=cached_blocks,
        )

    def used_bytes(self) -> int:
        """Bytes currently in flight (live + cached)."""
        if _HAS_CUPY and _cp is not None and self._cp_pool is not None:
            return int(self._cp_pool.used_bytes())
        return self._current_bytes

    def total_bytes(self) -> int:
        """Total bytes ever requested by this process via cupy's pool."""
        if _HAS_CUPY and _cp is not None and self._cp_pool is not None:
            return int(self._cp_pool.total_bytes())
        return self._peak_bytes

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _track(self, ptr: object, nbytes: int) -> None:
        self._live[id(ptr)] = (ptr, nbytes)
        self._current_bytes += nbytes
        if self._current_bytes > self._peak_bytes:
            self._peak_bytes = self._current_bytes
