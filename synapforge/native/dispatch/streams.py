"""streams.py -- thin CUDA stream wrappers for compute/transfer overlap.

Why
---
``cupy.cuda.Stream`` exposes the underlying CUDA stream object that lets
us issue kernels independently of the default stream. To overlap H2D
copy with compute we need *two* streams running in parallel:

* compute stream -- where forward/backward kernels execute
* transfer stream -- where ``copy_async`` H2D / D2H run

We package them as :class:`StreamPair`. ``copy_to_device`` issues the
copy on the transfer stream and records a CUDA event; the compute
stream then ``wait_event``s on it before consuming the tensor. This
lets the next batch's H2D copy run concurrently with the current
batch's GPU kernels.

Pure-CPU fallback
-----------------
On a host without CUDA / cupy (the dev workstation), this module
imports ``numpy`` only and the ``CudaStream`` becomes a no-op context
manager. ``to_device`` becomes a passthrough. This lets the rest of
the dispatch layer (and tests) run unchanged on CPU-only hosts.

Hard constraint
---------------
**No ``import torch``.** The whole point of synapforge.native is to be
torch-free. We use cupy directly when available.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np

try:  # pragma: no cover - exercised only on cupy-available hosts
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:  # ImportError, but also CUDA-init failures
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_NULL_STREAM_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# CudaStream
# ---------------------------------------------------------------------------

class CudaStream:
    """Thin wrapper around ``cupy.cuda.Stream`` with no-op CPU fallback.

    Usage::

        s = CudaStream(non_blocking=True)
        with s:
            y = cp.matmul(W, x)      # runs on stream s
        s.synchronize()              # block host until s done

    On CPU-only hosts (``CUPY_AVAILABLE = False``), ``CudaStream`` is a
    plain context manager that does nothing and ``synchronize()`` is a
    no-op. This keeps the dispatch logic uniform across CPU dev
    machines and GPU production boxes.
    """

    __slots__ = ("_stream", "_non_blocking", "_null")

    def __init__(self, non_blocking: bool = True, null: bool = False) -> None:
        """
        Parameters
        ----------
        non_blocking
            If True (default), the stream does *not* implicitly sync with
            the legacy default stream. Required for compute/transfer
            overlap.
        null
            If True, force the no-op fallback even when cupy is
            available. Useful for synchronous reference runs in tests.
        """
        self._non_blocking = bool(non_blocking)
        self._null = bool(null) or not CUPY_AVAILABLE
        if self._null:
            self._stream = None
        else:  # pragma: no cover - GPU path
            self._stream = cp.cuda.Stream(non_blocking=non_blocking)

    # ----- context manager -------------------------------------------------

    def __enter__(self) -> "CudaStream":
        if self._null:
            with _NULL_STREAM_LOCK:
                # No-op: there is no current-stream stack on CPU.
                pass
            return self
        self._stream.__enter__()  # pragma: no cover - GPU path
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._null:
            return
        self._stream.__exit__(exc_type, exc, tb)  # pragma: no cover

    # ----- ops -------------------------------------------------------------

    def synchronize(self) -> None:
        """Block the calling host thread until all enqueued work finishes."""
        if self._null:
            return
        self._stream.synchronize()  # pragma: no cover - GPU path

    def record_event(self) -> Any:
        """Record an event on this stream; returns an event handle.

        On CPU fallback returns ``None`` -- ``wait_event(None)`` is a no-op.
        """
        if self._null:
            return None
        return self._stream.record()  # pragma: no cover - GPU path

    def wait_event(self, event: Any) -> None:
        """Block this stream until ``event`` (from another stream) fires.

        ``event`` may be ``None`` (CPU fallback), in which case this is a
        no-op.
        """
        if self._null or event is None:
            return
        self._stream.wait_event(event)  # pragma: no cover - GPU path

    # ----- introspection ---------------------------------------------------

    @property
    def is_null(self) -> bool:
        """True if this is the CPU fallback (no real stream)."""
        return self._null

    @property
    def raw(self) -> Any:
        """Underlying ``cupy.cuda.Stream`` or ``None`` on CPU fallback."""
        return self._stream

    def __repr__(self) -> str:
        kind = "null" if self._null else "cuda"
        return f"CudaStream({kind}, non_blocking={self._non_blocking})"


# ---------------------------------------------------------------------------
# StreamPair
# ---------------------------------------------------------------------------

class StreamPair:
    """Compute + transfer stream pair for H2D/D2H overlap.

    Pattern::

        sp = StreamPair()
        # 1. Issue H2D on transfer stream:
        x_gpu = sp.copy_to_device_async(x_np)
        # 2. Tell compute stream to wait for the copy to finish:
        sp.compute_after_transfer()
        # 3. Run kernel on compute stream:
        with sp.compute:
            y_gpu = matmul(W, x_gpu)
        # 4. Sync host before reading back:
        sp.synchronize()

    Two such pairs (one per concurrent in-flight batch) let the next
    batch's H2D run while the current batch's kernels execute. That's
    the classical CUDA double-buffering trick.

    On CPU-only hosts both streams are no-op and ``copy_to_device_async``
    is a passthrough -- the same code runs unchanged.
    """

    __slots__ = ("compute", "transfer", "_last_transfer_event")

    def __init__(self) -> None:
        self.compute = CudaStream(non_blocking=True)
        self.transfer = CudaStream(non_blocking=True)
        self._last_transfer_event: Any = None

    # ----- transfer --------------------------------------------------------

    def copy_to_device_async(self, x: np.ndarray) -> Any:
        """Async H2D copy on the transfer stream. Returns a device array.

        On CPU fallback returns ``x`` unchanged (the "device" *is* the
        host).
        """
        if self.transfer.is_null:
            return x
        with self.transfer:  # pragma: no cover - GPU path
            d = cp.asarray(x)
        self._last_transfer_event = self.transfer.record_event()
        return d  # pragma: no cover - GPU path

    def copy_to_host_async(self, d: Any) -> np.ndarray:
        """Async D2H copy on the transfer stream. Returns a numpy array.

        Caller MUST call ``synchronize()`` (or the transfer stream's
        own synchronize) before reading the returned buffer.
        """
        if self.transfer.is_null:
            # Already on host -- nothing to do.
            return d
        with self.transfer:  # pragma: no cover - GPU path
            h = cp.asnumpy(d)
        self._last_transfer_event = self.transfer.record_event()
        return h  # pragma: no cover - GPU path

    def compute_after_transfer(self) -> None:
        """Make the compute stream wait for the most recent transfer.

        Call this after every ``copy_*_async`` and *before* enqueueing a
        kernel that consumes the transferred tensor.
        """
        self.compute.wait_event(self._last_transfer_event)

    # ----- sync ------------------------------------------------------------

    def synchronize(self) -> None:
        """Block until both streams are idle."""
        self.compute.synchronize()
        self.transfer.synchronize()

    def __repr__(self) -> str:
        return f"StreamPair(compute={self.compute!r}, transfer={self.transfer!r})"


# ---------------------------------------------------------------------------
# Cross-device transfer helpers
# ---------------------------------------------------------------------------

def to_device(x: np.ndarray, device: str = "cpu",
              stream: Optional[CudaStream] = None) -> Any:
    """Move ``x`` to ``device`` ('cpu' | 'cuda').

    Returns ``np.ndarray`` for ``device='cpu'`` and ``cupy.ndarray`` for
    ``device='cuda'``. On CPU-only hosts ``device='cuda'`` falls back to
    numpy and emits a single-shot warning would be surprising; we just
    return ``x`` unchanged so dispatch logic stays uniform.

    Parameters
    ----------
    x
        Either ``np.ndarray`` (host) or ``cupy.ndarray`` (device).
    device
        ``"cpu"`` or ``"cuda"``. ``"cuda:0"`` etc. accepted; index is
        currently ignored (single-GPU only -- multi-GPU TBD).
    stream
        Optional :class:`CudaStream` to issue the copy on. If ``None``,
        the copy runs on the current stream.
    """
    dev = device.split(":")[0].lower()
    if dev not in ("cpu", "cuda"):
        raise ValueError(f"unknown device: {device!r}")

    # CPU-only fallback: everything is numpy.
    if not CUPY_AVAILABLE:
        return np.asarray(x)

    is_device_array = (cp is not None and isinstance(x, cp.ndarray))

    if dev == "cpu":
        if is_device_array:  # pragma: no cover - GPU path
            if stream is None or stream.is_null:
                return cp.asnumpy(x)
            with stream:
                return cp.asnumpy(x)
        return np.asarray(x)

    # dev == "cuda"
    if is_device_array:  # pragma: no cover - GPU path
        return x
    if stream is None or stream.is_null:  # pragma: no cover - GPU path
        return cp.asarray(x)
    with stream:  # pragma: no cover - GPU path
        return cp.asarray(x)


def asnumpy(x: Any) -> np.ndarray:
    """Coerce ``x`` to ``np.ndarray`` regardless of source device.

    Convenience wrapper for code that needs to write to disk / stdout
    irrespective of whether the trainer is GPU-backed.
    """
    if isinstance(x, np.ndarray):
        return x
    if CUPY_AVAILABLE and cp is not None and isinstance(x, cp.ndarray):  # pragma: no cover
        return cp.asnumpy(x)
    return np.asarray(x)
