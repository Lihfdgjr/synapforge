"""CudaStreamPool -- 4-stream pool for compute / H2D / D2H / misc.

A typical step in synapforge training does:

    H2D copy (input batch)  ->  compute (forward + backward)  ->  D2H copy (loss / metrics)

Putting each on its own stream lets the GPU overlap:
    * stream 0 computes the previous batch
    * stream 1 H2Ds the next batch
    * stream 2 D2Hs the previous step's metrics
    * stream 3 holds long-running copies (checkpoints, eval batches)

Usage::

    pool = CudaStreamPool()
    with pool.compute.context():
        out = ops.matmul(a, b)              # runs on compute stream
    with pool.h2d.context():
        x.async_copy_from_host(host, stream=pool.h2d)
    pool.synchronize_all()                  # before the metric is read

On the numpy fallback (no cupy), the context managers are no-ops and
``synchronize_all`` is a no-op.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import List

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


__all__ = ["CudaStream", "CudaStreamPool"]


class CudaStream:
    """A single stream slot. Wraps cupy.cuda.Stream when available."""

    def __init__(self, name: str = "default"):
        self.name = name
        if _HAS_CUPY and _cp is not None:
            self._cp_stream = _cp.cuda.Stream(non_blocking=True)
        else:
            self._cp_stream = None

    @contextmanager
    def context(self):
        """Make this stream the current stream for the enclosed block."""
        if self._cp_stream is None:
            yield self
            return
        with self._cp_stream:
            yield self

    def synchronize(self) -> None:
        if self._cp_stream is not None:
            self._cp_stream.synchronize()

    @property
    def native(self):
        """The underlying cupy.cuda.Stream (or None on the numpy fallback)."""
        return self._cp_stream


class CudaStreamPool:
    """A 4-stream pool keyed by role.

    Roles:
        compute -- forward + backward kernels
        h2d     -- host-to-device data movement
        d2h     -- device-to-host metric/checkpoint movement
        misc    -- everything else (long-running copies, eval, etc.)
    """

    __slots__ = ("compute", "h2d", "d2h", "misc", "_all")

    def __init__(self):
        self.compute = CudaStream("compute")
        self.h2d = CudaStream("h2d")
        self.d2h = CudaStream("d2h")
        self.misc = CudaStream("misc")
        self._all: List[CudaStream] = [self.compute, self.h2d, self.d2h, self.misc]

    def synchronize_all(self) -> None:
        for s in self._all:
            s.synchronize()

    def by_name(self, name: str) -> CudaStream:
        return getattr(self, name)

    def __iter__(self):
        return iter(self._all)
