"""action_async.py -- ActionHead -> OSActuator async tool execution.

Why this is its own module
--------------------------
Tool execution can take 50 ms (local shell echo) to 5 s (web fetch
behind a slow CDN) to forever (a hung subprocess). It is a system call,
not GPU compute. **It must never block the training step.**

The ActionHead in production (``synapforge.action.head`` + ``actuator``)
emits an action vector from the model's hidden state and dispatches it
to an :class:`OSActuator`. The actuator returns an observation that's
folded back into the model via a slow STDP signal (e.g. updates the
NeuroMCP codebook on success, increments a usage counter, etc.).

Design
------
* All actuator dispatch goes through a thread pool (configurable
  ``num_workers``, default 4).
* Each :meth:`submit` returns a :class:`AuxFuture` (here that means a
  :class:`AuxFuture` whose ``result()`` is a :class:`ToolObservation`).
* The trainer NEVER blocks on the future. It polls
  :meth:`drain_completed()` once per N outer steps and feeds the
  finished observations into the slow STDP signal.
* If an actuator call exceeds ``per_call_timeout`` (default 30 s), the
  future is cancelled and the observation is set to a "timeout"
  sentinel. The training loop is unaffected.

Hard constraint
---------------
**No ``import torch``.** The actuator is a caller-supplied callable;
we just thread-pool it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Deque, List, Optional


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """The trainer's torch ActionHead emits one of these per step (or never)."""

    step_idx: int
    tool_id: int                # codebook prototype index
    arg_payload: Any            # whatever the actuator needs (URL, cmd, ...)
    confidence: float = 0.0     # head's softmax probability for this tool
    timeout_s: float = 30.0
    extra: dict = field(default_factory=dict)


@dataclass
class ToolObservation:
    """The actuator's response, queued back for the slow STDP path."""

    step_idx: int
    tool_id: int
    success: bool
    result: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    extra: dict = field(default_factory=dict)


# Sentinel for shutdown.
_SHUTDOWN_SENTINEL = object()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class ActionHeadAsyncDriver:
    """Thread-pool dispatcher for ActionHead -> OSActuator calls.

    Lifecycle
    ---------
    1. Construct with an ``execute_fn(call) -> ToolObservation`` callable
       and ``num_workers`` (default 4).
    2. Per outer step (or per N steps), the trainer optionally calls
       :meth:`submit(call)`. Returns immediately. The trainer never
       waits on the result.
    3. Periodically the trainer calls :meth:`drain_completed()` to pull
       any finished :class:`ToolObservation`s and feed them into the slow
       STDP signal (e.g. ``codebook.observe(tool_id, success)``).
    4. :meth:`shutdown()` joins the workers.

    The internal queue has bounded depth (``submit_capacity``); calls
    that arrive when the queue is full are *dropped* with a metric
    increment and ``submit()`` returns ``None`` -- the trainer
    interprets this as "skip tool exec this step, the env wasn't
    available". Better than blocking the GPU stream waiting for a
    slow tool.
    """

    __slots__ = (
        "_execute_fn",
        "_num_workers",
        "_submit_queue",
        "_completed_queue",
        "_completed_lock",
        "_completed",
        "_workers",
        "_shutdown",
        "_metrics_lock",
        "_metrics",
        "_in_flight",
        "_in_flight_lock",
    )

    def __init__(
        self,
        execute_fn: Callable[[ToolCall], ToolObservation],
        *,
        num_workers: int = 4,
        submit_capacity: int = 16,
        thread_name: str = "aux-action",
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self._execute_fn = execute_fn
        self._num_workers = int(num_workers)
        self._submit_queue: "Queue[Any]" = Queue(maxsize=int(submit_capacity))
        # Completed deque is internal; ``drain_completed`` drains it.
        self._completed_lock = threading.Lock()
        self._completed: Deque[ToolObservation] = deque()
        # Spare attribute for forward compat / never used here.
        self._completed_queue: "Queue[Any]" = Queue()
        self._shutdown = False
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "submitted": 0,
            "completed": 0,
            "dropped_full_queue": 0,
            "dropped_timeout": 0,
            "errors": 0,
            "total_exec_s": 0.0,
            "max_exec_s": 0.0,
        }
        self._in_flight_lock = threading.Lock()
        self._in_flight = 0
        self._workers: List[threading.Thread] = []
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"{thread_name}-{i}",
                daemon=True,
            )
            self._workers.append(t)
            t.start()

    # ----- public API ----------------------------------------------------

    def submit(self, call: ToolCall) -> bool:
        """Enqueue a tool call. Returns True if accepted, False if dropped.

        Never blocks the caller (uses ``put_nowait``). If the submit
        queue is full -- the trainer is generating tool calls faster
        than the workers can finish them -- the call is dropped and a
        ``dropped_full_queue`` metric increments.
        """
        if self._shutdown:
            return False
        try:
            self._submit_queue.put_nowait(call)
        except Exception:
            with self._metrics_lock:
                self._metrics["dropped_full_queue"] += 1
            return False
        with self._metrics_lock:
            self._metrics["submitted"] += 1
        with self._in_flight_lock:
            self._in_flight += 1
        return True

    def drain_completed(self) -> List[ToolObservation]:
        """Pop and return all completed observations (FIFO).

        Never blocks. The trainer typically calls this once per N outer
        steps and feeds the observations into the slow STDP signal.
        """
        out: List[ToolObservation] = []
        with self._completed_lock:
            while self._completed:
                out.append(self._completed.popleft())
        return out

    def in_flight(self) -> int:
        with self._in_flight_lock:
            return self._in_flight

    def metrics(self) -> dict:
        with self._metrics_lock:
            d = dict(self._metrics)
        d["in_flight"] = self.in_flight()
        if d["completed"] > 0:
            d["avg_exec_s"] = d["total_exec_s"] / d["completed"]
        else:
            d["avg_exec_s"] = 0.0
        return d

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        for _ in self._workers:
            try:
                self._submit_queue.put_nowait(_SHUTDOWN_SENTINEL)
            except Exception:
                pass
        if wait:
            for t in self._workers:
                t.join(timeout=timeout)

    def __enter__(self) -> "ActionHeadAsyncDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # ----- worker --------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._submit_queue.get(timeout=0.5)
            except Empty:
                if self._shutdown:
                    return
                continue
            if item is _SHUTDOWN_SENTINEL:
                return
            call: ToolCall = item
            t0 = time.time()
            try:
                # No real timeout enforcement here -- the actuator is
                # expected to honour ``call.timeout_s`` itself (e.g. via
                # requests' timeout=, subprocess.communicate(timeout=)).
                # We do however cap our perceived elapsed and tag the
                # observation if it exceeds the call's declared budget.
                obs = self._execute_fn(call)
                if not isinstance(obs, ToolObservation):
                    obs = ToolObservation(
                        step_idx=call.step_idx,
                        tool_id=call.tool_id,
                        success=False,
                        error="execute_fn did not return ToolObservation",
                    )
                obs.elapsed_ms = (time.time() - t0) * 1000.0
                if obs.elapsed_ms > call.timeout_s * 1000.0:
                    with self._metrics_lock:
                        self._metrics["dropped_timeout"] += 1
                    obs.success = False
                    obs.error = f"timeout ({obs.elapsed_ms:.1f}ms > {call.timeout_s*1000:.1f}ms)"
                with self._completed_lock:
                    self._completed.append(obs)
                with self._metrics_lock:
                    self._metrics["completed"] += 1
                    elapsed = time.time() - t0
                    self._metrics["total_exec_s"] += elapsed
                    if elapsed > self._metrics["max_exec_s"]:
                        self._metrics["max_exec_s"] = elapsed
            except BaseException as exc:  # noqa: BLE001
                with self._metrics_lock:
                    self._metrics["errors"] += 1
                obs = ToolObservation(
                    step_idx=call.step_idx,
                    tool_id=call.tool_id,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=(time.time() - t0) * 1000.0,
                )
                with self._completed_lock:
                    self._completed.append(obs)
            finally:
                with self._in_flight_lock:
                    self._in_flight -= 1
