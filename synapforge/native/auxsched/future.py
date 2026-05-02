"""future.py -- minimal Future-like result carrier for aux drivers.

Why not ``concurrent.futures.Future``
-------------------------------------
The stdlib Future works fine, but we want:

* a ``done()`` non-blocking probe (Future has it, but we also want
  ``poll()`` that returns ``(done, result_or_None)`` in one shot);
* a stream-aware variant that wraps a CUDA event so the future can be
  "done" once the GPU stream signals (the host thread never had to do
  anything);
* zero dependency on ``concurrent.futures`` because some embedders
  swap stdlib threading models.

So we ship our own. Same shape, tiny surface area.

Hard constraint
---------------
**No ``import torch``.** numpy + threading + (optional) cupy.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Optional, Tuple


class AuxFuture:
    """Future-like single-shot result carrier.

    Two completion modes:

    * **Direct**: a worker calls ``set_result(...)`` / ``set_exception(...)``.
    * **Stream-bound**: caller passes a ``cuda_event`` from a CUDA stream
      that, when fired, indicates the GPU work is done. ``done()`` then
      checks the event status (no host sync needed).

    The class is thread-safe for one producer + many consumers, which is
    the only pattern the coordinator uses.
    """

    __slots__ = (
        "_event",
        "_result",
        "_exception",
        "_cuda_event",
        "_label",
        "_meta",
        "_on_done",
    )

    def __init__(
        self,
        *,
        cuda_event: Any = None,
        label: str = "future",
        meta: Optional[dict] = None,
    ) -> None:
        self._event = threading.Event()
        self._result: Any = None
        self._exception: Optional[BaseException] = None
        self._cuda_event = cuda_event
        self._label = label
        self._meta = meta or {}
        self._on_done: Optional[Callable[["AuxFuture"], None]] = None

    # ----- producer side --------------------------------------------------

    def set_result(self, result: Any) -> None:
        if self._event.is_set():
            return  # idempotent; later writes ignored
        self._result = result
        self._event.set()
        if self._on_done is not None:
            try:
                self._on_done(self)
            except BaseException:  # noqa: BLE001 - never crash the worker
                pass

    def set_exception(self, exc: BaseException) -> None:
        if self._event.is_set():
            return
        try:
            exc.__synapforge_traceback__ = traceback.format_exc()  # type: ignore[attr-defined]
        except Exception:
            pass
        self._exception = exc
        self._event.set()
        if self._on_done is not None:
            try:
                self._on_done(self)
            except BaseException:  # noqa: BLE001
                pass

    # ----- consumer side --------------------------------------------------

    def done(self) -> bool:
        """Non-blocking check. Returns True if produced or cuda_event ready."""
        if self._event.is_set():
            return True
        ev = self._cuda_event
        if ev is None:
            return False
        # cupy.cuda.Event has ``.done`` (bool property) since 11.x.
        # Fall back: assume not done.
        try:
            return bool(ev.done)  # pragma: no cover - GPU path
        except Exception:
            return False

    def poll(self) -> Tuple[bool, Any]:
        """Return ``(done, result_or_exc_or_None)`` without blocking.

        If ``done`` is True and an exception was set, returns
        ``(True, exc)`` -- caller is expected to ``raise`` it.
        """
        if not self.done():
            return False, None
        if self._exception is not None:
            return True, self._exception
        return True, self._result

    def result(self, timeout: Optional[float] = None) -> Any:
        """Block until done; re-raise stored exception if any."""
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(f"AuxFuture[{self._label}] timed out after {timeout}s")
        if self._exception is not None:
            raise self._exception
        return self._result

    def add_done_callback(self, fn: Callable[["AuxFuture"], None]) -> None:
        """Register a single ``fn(future)`` to fire when the future completes.

        If the future is already done at registration time, ``fn`` runs
        synchronously on the calling thread.
        """
        self._on_done = fn
        if self.done():
            try:
                fn(self)
            except BaseException:  # noqa: BLE001
                pass

    @property
    def label(self) -> str:
        return self._label

    @property
    def meta(self) -> dict:
        return self._meta

    def __repr__(self) -> str:
        if self._exception is not None:
            return f"AuxFuture[{self._label}](exc={type(self._exception).__name__})"
        if self._event.is_set():
            return f"AuxFuture[{self._label}](done)"
        return f"AuxFuture[{self._label}](pending)"
