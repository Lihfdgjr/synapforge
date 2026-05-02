"""ttt_async.py -- chunked async Test-Time Training.

The problem
-----------
Self-learn TTT runs an inner SGD loop on validation rows: typically
``k=8`` inner steps per outer step. On A800 d=1280 bs=48 each inner
step is ~25 ms, so an 8-step inner loop is **~200 ms** -- 2-3x the
main fwd+bwd time. Run 6/7 trace shows TTT being the single largest
auxiliary blocker.

The async refactor
------------------
We split the ``k=8`` inner loop into:

* ``inline_k`` (default 2) inner steps run synchronously on the main
  stream, so the freshest ``k`` gradient is always reflected.
* ``async_k = k - inline_k`` (default 6) remaining inner steps run on
  a dedicated stream C, **overlapping with the next outer step's
  data prefetch + main forward**.

In steady state the async portion finishes before the next outer step
needs TTT-adapted weights, so quality is preserved. We bound the lag
to **at most one outer step** (any earlier-step async TTT that hasn't
completed by the start of step N is dropped, *not* applied late).

Quality guard
-------------
The driver exposes ``run_inline()`` that computes the full ``k``-step
TTT inline, used as a reference. The async path's final TTT-adapted
state must match the inline reference within 1% on val ppl on a toy
LM. The unit test asserts this; if it ever drifts, ``inline_k`` is
bumped.

Hard constraint
---------------
**No ``import torch``.** Inner-step compute is a caller-supplied
callable; we just schedule it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from synapforge.native.auxsched.future import AuxFuture
from synapforge.native.auxsched.streams import AuxStream


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _TTTPayload:
    step_idx: int
    val_inputs: Any
    val_targets: Any
    inner_state: Any  # caller-defined opaque (e.g. fast-weight dict)
    extra: dict = field(default_factory=dict)


@dataclass
class TTTStepStats:
    """Per outer-step TTT telemetry (returned via the future)."""

    step_idx: int
    inline_k: int = 0
    async_k: int = 0
    inline_loss_first: float = float("nan")  # loss at iter 0 (pre-update)
    inline_loss_last: float = float("nan")   # loss after inline_k iters
    async_loss_last: float = float("nan")    # loss after inline_k+async_k iters
    inline_elapsed_ms: float = 0.0
    async_elapsed_ms: float = 0.0
    dropped: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class TTTAsyncDriver:
    """Chunked async Test-Time Training driver.

    Caller supplies two callables:

    ``inner_step_fn(state, val_inputs, val_targets, iter_idx) -> (state, loss)``
        One TTT inner SGD step. Returns the updated opaque ``state`` and
        the scalar loss after the step. The state is whatever the
        trainer chooses (e.g. a fast-weight dict, a small adapter, a
        LoRA-style delta -- but per project rules NO LoRA on the
        backbone, only test-time scratch).

    ``done_fn(state) -> None`` (optional)
        Called once after all ``inline_k + async_k`` steps complete.
        Lets the trainer e.g. write the adapted state into a buffer
        for the next forward to consume.

    Methods
    -------
    * :meth:`run` -- the main entry point. Runs ``inline_k`` steps
      inline, returns immediately, schedules ``async_k`` steps on the
      aux stream, returns a future. Final TTT-adapted state is
      delivered via ``done_fn`` (called on the worker thread when the
      async chunk finishes).
    * :meth:`run_inline` -- run ALL ``k`` steps synchronously. The
      reference for the parity test.
    * :meth:`shutdown`
    """

    __slots__ = (
        "_inner_step_fn",
        "_done_fn",
        "_total_k",
        "_inline_k",
        "_stream",
        "_thread",
        "_pending",
        "_pending_lock",
        "_wakeup",
        "_shutdown",
        "_metrics",
        "_metrics_lock",
        "_last_future",
    )

    def __init__(
        self,
        inner_step_fn: Callable[[Any, Any, Any, int], tuple[Any, float]],
        *,
        total_k: int = 8,
        inline_k: int = 2,
        done_fn: Optional[Callable[[Any], None]] = None,
        stream: Optional[AuxStream] = None,
        thread_name: str = "aux-ttt",
    ) -> None:
        if total_k < 1:
            raise ValueError("total_k must be >= 1")
        if inline_k < 0 or inline_k > total_k:
            raise ValueError(f"inline_k must be in [0, total_k], got {inline_k}")
        self._inner_step_fn = inner_step_fn
        self._done_fn = done_fn
        self._total_k = int(total_k)
        self._inline_k = int(inline_k)
        self._stream = stream or AuxStream(non_blocking=True, label="aux.ttt")
        self._pending: Optional[tuple[_TTTPayload, AuxFuture, Any, TTTStepStats]] = None
        self._pending_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._shutdown = False
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "submitted": 0,
            "completed": 0,
            "dropped_stale": 0,
            "errors": 0,
            "inline_total_s": 0.0,
            "async_total_s": 0.0,
        }
        self._last_future: Optional[AuxFuture] = None
        self._thread = threading.Thread(
            target=self._worker_loop, name=thread_name, daemon=True
        )
        self._thread.start()

    # ----- public API ----------------------------------------------------

    def run(
        self,
        step_idx: int,
        val_inputs: Any,
        val_targets: Any,
        inner_state: Any,
        extra: Optional[dict] = None,
    ) -> tuple[Any, AuxFuture]:
        """Run ``inline_k`` inner steps NOW; schedule the rest async.

        Returns ``(state_after_inline, future)``. The trainer should use
        ``state_after_inline`` for the *current* outer step's main
        forward (it has the most recent ``inline_k`` gradient applied).
        The future fires when the remaining ``async_k`` steps finish,
        and (if ``done_fn`` was set) ``done_fn(final_state)`` is called
        on the worker thread.
        """
        if self._shutdown:
            raise RuntimeError("TTTAsyncDriver is shut down")
        # Inline portion -- runs on the caller thread, on the caller's
        # default stream. We do NOT enter the aux stream here; that's only
        # for the async portion.
        stats = TTTStepStats(
            step_idx=step_idx,
            inline_k=self._inline_k,
            async_k=self._total_k - self._inline_k,
        )
        t_inline_0 = time.time()
        cur_state = inner_state
        for i in range(self._inline_k):
            cur_state, loss = self._inner_step_fn(
                cur_state, val_inputs, val_targets, i
            )
            if i == 0:
                stats.inline_loss_first = float(loss)
            stats.inline_loss_last = float(loss)
        stats.inline_elapsed_ms = (time.time() - t_inline_0) * 1000.0

        # Async portion -- only schedule if there's actually more work.
        async_k = self._total_k - self._inline_k
        if async_k <= 0:
            stats.async_loss_last = stats.inline_loss_last
            fut = AuxFuture(label=f"ttt[{step_idx}](inline-only)")
            fut.set_result(stats)
            self._last_future = fut
            with self._metrics_lock:
                self._metrics["submitted"] += 1
                self._metrics["completed"] += 1
                self._metrics["inline_total_s"] += stats.inline_elapsed_ms / 1000.0
            if self._done_fn is not None:
                try:
                    self._done_fn(cur_state)
                except BaseException:  # noqa: BLE001
                    pass
            return cur_state, fut

        payload = _TTTPayload(
            step_idx=step_idx,
            val_inputs=val_inputs,
            val_targets=val_targets,
            inner_state=cur_state,
            extra=extra or {},
        )
        fut = AuxFuture(label=f"ttt[{step_idx}]")
        with self._pending_lock:
            old = self._pending
            self._pending = (payload, fut, cur_state, stats)
            if old is not None:
                _, old_fut, _, old_stats = old
                old_stats.dropped = True
                old_fut.set_result(old_stats)
                with self._metrics_lock:
                    self._metrics["dropped_stale"] += 1
        with self._metrics_lock:
            self._metrics["submitted"] += 1
            self._metrics["inline_total_s"] += stats.inline_elapsed_ms / 1000.0
        self._last_future = fut
        self._wakeup.set()
        return cur_state, fut

    def run_inline(
        self,
        step_idx: int,
        val_inputs: Any,
        val_targets: Any,
        inner_state: Any,
    ) -> tuple[Any, TTTStepStats]:
        """Reference implementation: ALL ``total_k`` inner steps inline.

        Used by the parity test to compare against the chunked async
        path. Should NOT be called in the hot training loop.
        """
        stats = TTTStepStats(
            step_idx=step_idx,
            inline_k=self._total_k,
            async_k=0,
        )
        t0 = time.time()
        cur_state = inner_state
        for i in range(self._total_k):
            cur_state, loss = self._inner_step_fn(
                cur_state, val_inputs, val_targets, i
            )
            if i == 0:
                stats.inline_loss_first = float(loss)
            stats.inline_loss_last = float(loss)
        stats.async_loss_last = stats.inline_loss_last
        stats.inline_elapsed_ms = (time.time() - t0) * 1000.0
        return cur_state, stats

    def metrics(self) -> dict:
        with self._metrics_lock:
            return dict(self._metrics)

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._wakeup.set()
        if wait:
            self._thread.join(timeout=timeout)

    def __enter__(self) -> "TTTAsyncDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # ----- worker --------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            self._wakeup.wait(timeout=0.5)
            self._wakeup.clear()
            if self._shutdown:
                with self._pending_lock:
                    item = self._pending
                    self._pending = None
                if item is not None:
                    _, fut, _, stats = item
                    stats.dropped = True
                    fut.set_result(stats)
                return
            with self._pending_lock:
                item = self._pending
                self._pending = None
            if item is None:
                continue
            payload, fut, cur_state, stats = item
            t_async_0 = time.time()
            try:
                with self._stream:
                    for i in range(stats.inline_k, self._total_k):
                        cur_state, loss = self._inner_step_fn(
                            cur_state, payload.val_inputs,
                            payload.val_targets, i,
                        )
                        stats.async_loss_last = float(loss)
                stats.async_elapsed_ms = (time.time() - t_async_0) * 1000.0
                fut.set_result(stats)
                with self._metrics_lock:
                    self._metrics["completed"] += 1
                    self._metrics["async_total_s"] += stats.async_elapsed_ms / 1000.0
                if self._done_fn is not None:
                    try:
                        self._done_fn(cur_state)
                    except BaseException:  # noqa: BLE001
                        pass
            except BaseException as exc:  # noqa: BLE001
                fut.set_exception(exc)
                with self._metrics_lock:
                    self._metrics["errors"] += 1
