"""streams.py -- minimal ``cupy.cuda.Stream`` wrapper used by the aux pkg.

Why a local copy
----------------
The dispatch layer (``synapforge.native.dispatch.streams``) ships a more
fully-featured ``CudaStream`` / ``StreamPair`` pair. ``aux`` would prefer
to import from there, but the dispatch package is on a separate feature
branch and the merge order isn't pinned. To keep ``aux`` runnable in
isolation -- and to keep its surface area minimal -- we ship a small
stand-alone wrapper here.

If ``synapforge.native.dispatch`` is later present, the coordinator can be
re-pointed at its ``StreamPair``; the API is intentionally compatible.

Hard constraint
---------------
**No ``import torch``.** Pure ``numpy`` + (optional) ``cupy``.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np

try:  # pragma: no cover - exercised only on cupy-available hosts
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:  # ImportError / CUDA-init failure
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


_NULL_STREAM_LOCK = threading.Lock()


class AuxStream:
    """Thin wrapper around ``cupy.cuda.Stream`` with no-op CPU fallback.

    Usage::

        s = AuxStream(non_blocking=True)
        with s:
            y = cp.matmul(W, x)
        s.synchronize()

    On CPU-only hosts (``CUPY_AVAILABLE = False``) the stream is a no-op
    context manager and ``synchronize`` returns immediately. This keeps
    the coordinator policy uniform on dev workstations and GPU rentals.
    """

    __slots__ = ("_stream", "_non_blocking", "_null", "_label")

    def __init__(
        self,
        non_blocking: bool = True,
        null: bool = False,
        label: str = "aux",
    ) -> None:
        self._non_blocking = bool(non_blocking)
        self._label = str(label)
        self._null = bool(null) or not CUPY_AVAILABLE
        if self._null:
            self._stream = None
        else:  # pragma: no cover - GPU path
            self._stream = cp.cuda.Stream(non_blocking=non_blocking)

    # ----- context manager -------------------------------------------------

    def __enter__(self) -> "AuxStream":
        if self._null:
            with _NULL_STREAM_LOCK:
                # No current-stream stack on CPU; no-op.
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
        if self._null:
            return
        self._stream.synchronize()  # pragma: no cover - GPU path

    def record_event(self) -> Any:
        if self._null:
            return None
        return self._stream.record()  # pragma: no cover - GPU path

    def wait_event(self, event: Any) -> None:
        if self._null or event is None:
            return
        self._stream.wait_event(event)  # pragma: no cover - GPU path

    # ----- introspection ---------------------------------------------------

    @property
    def is_null(self) -> bool:
        return self._null

    @property
    def label(self) -> str:
        return self._label

    @property
    def raw(self) -> Any:
        return self._stream

    def __repr__(self) -> str:
        kind = "null" if self._null else "cuda"
        return f"AuxStream({self._label}, kind={kind})"


class AuxStreamPair:
    """Compute + transfer stream pair (mirror of dispatch.StreamPair).

    Used by aux components that need to copy host->device on a
    transfer stream while running aux compute on a separate stream
    that waits on the copy event.
    """

    __slots__ = ("compute", "transfer", "_last_event")

    def __init__(self, label: str = "aux") -> None:
        self.compute = AuxStream(non_blocking=True, label=f"{label}.compute")
        self.transfer = AuxStream(non_blocking=True, label=f"{label}.transfer")
        self._last_event: Any = None

    def copy_to_device_async(self, x: np.ndarray) -> Any:
        if self.transfer.is_null:
            return x
        with self.transfer:  # pragma: no cover - GPU path
            d = cp.asarray(x)
        self._last_event = self.transfer.record_event()
        return d  # pragma: no cover - GPU path

    def compute_after_transfer(self) -> None:
        self.compute.wait_event(self._last_event)

    def synchronize(self) -> None:
        self.compute.synchronize()
        self.transfer.synchronize()


def asnumpy(x: Any) -> np.ndarray:
    """Coerce ``x`` to ``np.ndarray`` regardless of source device."""
    if isinstance(x, np.ndarray):
        return x
    if CUPY_AVAILABLE and cp is not None and isinstance(x, cp.ndarray):  # pragma: no cover
        return cp.asnumpy(x)
    return np.asarray(x)


def to_device(x: Any, *, device: str = "cpu",
              stream: Optional[AuxStream] = None) -> Any:
    """Move ``x`` to ``device`` ('cpu' | 'cuda'); CPU-fallback safe."""
    dev = device.split(":")[0].lower()
    if dev not in ("cpu", "cuda"):
        raise ValueError(f"unknown device: {device!r}")
    if not CUPY_AVAILABLE:
        return np.asarray(x)
    is_dev = (cp is not None and isinstance(x, cp.ndarray))
    if dev == "cpu":
        if is_dev:  # pragma: no cover - GPU path
            if stream is None or stream.is_null:
                return cp.asnumpy(x)
            with stream:
                return cp.asnumpy(x)
        return np.asarray(x)
    # cuda
    if is_dev:  # pragma: no cover
        return x
    if stream is None or stream.is_null:  # pragma: no cover
        return cp.asarray(x)
    with stream:  # pragma: no cover
        return cp.asarray(x)
