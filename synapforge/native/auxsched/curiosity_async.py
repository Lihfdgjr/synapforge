"""curiosity_async.py -- stream-routing policy for ICM curiosity loss.

What this module is
-------------------
A **policy** that decides *where* and *when* the curiosity ICM loss /
update runs relative to the main forward+backward pass. The actual ICM
math (forward model + inverse model + STDP-aware weighting) lives in
``synapforge.curiosity`` and uses torch -- that is intentional, because
ICM has its own gradient graph through small heads.

This file is torch-free: it accepts caller-provided callables and
schedules them on a dedicated CUDA stream so they overlap with the main
training step.

Why is this safe to overlap?
----------------------------
Per the ICM formulation (Pathak 1705.05363) the curiosity heads have
their own params and own loss. The only data they need from the main
step are the hidden states ``h_prev`` and ``h_next``. Once those exist
(end of main forward), the curiosity stream can compute its loss and
do its own backward into the curiosity heads independently.

The only coupling back to the main step is an optional
``curiosity_to_main_grad`` -- a small bonus signal added to the main
backbone. If the trainer wants that signal it must call ``wait_grad()``
before the main backward starts. If not (default), the curiosity stream
can run completely async and the result is just a curiosity-head
parameter update -- no synchronization needed.

Quality guard
-------------
Async-vs-inline parity test compares the curiosity *loss values* and
the *param deltas* on a synthetic ICM (1-layer MLP). The async branch
must match the inline branch within 1e-5 absolute on loss and 1e-4
relative on each param tensor's L2 norm.

Hard constraint
---------------
**No ``import torch``** here. The trainer wires its torch-using ICM
into a callable, and we just schedule the call.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from synapforge.native.auxsched.future import AuxFuture
from synapforge.native.auxsched.streams import AuxStream


# ---------------------------------------------------------------------------
# Public dataclasses (caller-supplied payload)
# ---------------------------------------------------------------------------


@dataclass
class _CuriosityPayload:
    """A frozen snapshot of what the curiosity stream needs to run.

    The caller (the trainer's main fwd code) builds this at the end of
    fwd and pushes it via :meth:`CuriosityAsyncDriver.submit`.

    ``h_prev`` / ``h_next`` are typed ``Any`` because they may be
    ``cupy.ndarray`` on the GPU stream A. The compute callable knows
    how to handle them.
    """

    step_idx: int
    h_prev: Any
    h_next: Any
    action_emb: Any = None
    extra: dict = field(default_factory=dict)


@dataclass
class CuriosityResult:
    """Returned from a completed curiosity step (via the future)."""

    step_idx: int
    loss: float
    grad_norm: float = 0.0
    forward_model_loss: float = 0.0
    inverse_model_loss: float = 0.0
    elapsed_ms: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class CuriosityAsyncDriver:
    """Routes curiosity ICM compute to a dedicated CUDA stream + worker thread.

    Lifecycle
    ---------
    * Construct once per training session.
    * On each main fwd, call :meth:`submit(payload)`. Returns an
      :class:`AuxFuture` whose ``.result()`` produces a
      :class:`CuriosityResult`.
    * Optionally call :meth:`wait_latest()` to block on the last in-flight
      future (e.g. before main backward if you want curiosity gradient
      to flow to backbone).
    * Call :meth:`shutdown()` at the end of training.

    Threading model
    ---------------
    Single dedicated worker thread (so cupy stream context is consistent
    on each call). Submissions go onto a 1-deep queue; if the worker is
    still busy when a new submission arrives, the previous future is
    *dropped from the queue* (the worker still finishes it; only the
    queue slot is reclaimed). This implements the "skip stale aux"
    backpressure -- ICM gradients for step N-2 are useless if N is
    being submitted.
    """

    __slots__ = (
        "_compute_fn",
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
        compute_fn: Callable[[_CuriosityPayload], CuriosityResult],
        *,
        stream: Optional[AuxStream] = None,
        thread_name: str = "aux-curiosity",
    ) -> None:
        """
        Parameters
        ----------
        compute_fn
            Callable ``(payload) -> CuriosityResult``. The trainer's torch
            ICM code goes here (uses payload.h_prev, payload.h_next, etc.,
            does forward+inverse loss, calls .backward() on the curiosity
            head's optimizer, returns metrics). The driver does NOT touch
            torch.
        stream
            ``AuxStream`` to attach the cupy stream context (on GPU
            hosts). Defaults to a fresh non-blocking stream labelled
            ``aux.curiosity``. Passed straight through to the worker
            thread; the compute_fn is run inside ``with stream:`` so any
            cupy ops it issues land on the right stream.
        thread_name
            Worker thread name (debugging).
        """
        self._compute_fn = compute_fn
        self._stream = stream or AuxStream(non_blocking=True, label="aux.curiosity")
        self._pending: Optional[tuple[_CuriosityPayload, AuxFuture]] = None
        self._pending_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._shutdown = False
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "submitted": 0,
            "completed": 0,
            "dropped_stale": 0,
            "errors": 0,
            "total_compute_s": 0.0,
            "max_compute_s": 0.0,
        }
        self._last_future: Optional[AuxFuture] = None
        self._thread = threading.Thread(
            target=self._worker_loop, name=thread_name, daemon=True
        )
        self._thread.start()

    # ----- public API ------------------------------------------------------

    def submit(
        self,
        step_idx: int,
        h_prev: Any,
        h_next: Any,
        action_emb: Any = None,
        extra: Optional[dict] = None,
    ) -> AuxFuture:
        """Enqueue a curiosity step. Returns its :class:`AuxFuture`.

        If a previous payload is still waiting on the queue (worker hasn't
        picked it up), it is *dropped* and its future is set to a
        synthetic "stale-skip" result. This implements the "1-step lag is
        OK; 2-step lag is wasteful" rule.
        """
        if self._shutdown:
            raise RuntimeError("CuriosityAsyncDriver is shut down")
        payload = _CuriosityPayload(
            step_idx=step_idx,
            h_prev=h_prev,
            h_next=h_next,
            action_emb=action_emb,
            extra=extra or {},
        )
        fut = AuxFuture(label=f"curiosity[{step_idx}]")
        with self._pending_lock:
            old = self._pending
            self._pending = (payload, fut)
            if old is not None:
                _, old_fut = old
                old_fut.set_result(
                    CuriosityResult(
                        step_idx=old[0].step_idx,
                        loss=float("nan"),
                        elapsed_ms=0.0,
                        extra={"dropped": "stale-skip"},
                    )
                )
                with self._metrics_lock:
                    self._metrics["dropped_stale"] += 1
        with self._metrics_lock:
            self._metrics["submitted"] += 1
        self._last_future = fut
        self._wakeup.set()
        return fut

    def wait_latest(self, timeout: Optional[float] = None) -> Optional[CuriosityResult]:
        """Block on the most recently submitted future. Returns its result
        (or ``None`` if no submission has been made).

        Use this only when the trainer wants curiosity grads to flow into
        the backbone in the SAME step (i.e. before main backward).
        Otherwise prefer ``maybe_collect_done()``.
        """
        f = self._last_future
        if f is None:
            return None
        return f.result(timeout=timeout)

    def maybe_collect_done(self) -> list[CuriosityResult]:
        """Drain any completed futures (non-blocking). Useful for telemetry."""
        # Implementation: we don't keep a list of all futures (avoids
        # leaking). Caller is expected to track futures from submit().
        # For convenience we expose the latest if it's done.
        out: list[CuriosityResult] = []
        f = self._last_future
        if f is not None and f.done():
            try:
                out.append(f.result(timeout=0.0))
            except Exception:
                pass
        return out

    def metrics(self) -> dict:
        with self._metrics_lock:
            d = dict(self._metrics)
        if d["completed"] > 0:
            d["avg_compute_s"] = d["total_compute_s"] / d["completed"]
        else:
            d["avg_compute_s"] = 0.0
        return d

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._wakeup.set()
        if wait:
            self._thread.join(timeout=timeout)

    def __enter__(self) -> "CuriosityAsyncDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # ----- worker --------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            self._wakeup.wait(timeout=0.5)
            self._wakeup.clear()
            if self._shutdown:
                # Drain any final pending payload so its future doesn't
                # hang forever.
                with self._pending_lock:
                    item = self._pending
                    self._pending = None
                if item is not None:
                    _, fut = item
                    fut.set_exception(
                        RuntimeError("CuriosityAsyncDriver shut down before run")
                    )
                return
            with self._pending_lock:
                item = self._pending
                self._pending = None
            if item is None:
                continue
            payload, fut = item
            t0 = time.time()
            try:
                with self._stream:
                    res = self._compute_fn(payload)
                if not isinstance(res, CuriosityResult):
                    res = CuriosityResult(
                        step_idx=payload.step_idx,
                        loss=float(res) if res is not None else 0.0,
                    )
                res.elapsed_ms = (time.time() - t0) * 1000.0
                fut.set_result(res)
                with self._metrics_lock:
                    self._metrics["completed"] += 1
                    elapsed = time.time() - t0
                    self._metrics["total_compute_s"] += elapsed
                    if elapsed > self._metrics["max_compute_s"]:
                        self._metrics["max_compute_s"] = elapsed
            except BaseException as exc:  # noqa: BLE001
                fut.set_exception(exc)
                with self._metrics_lock:
                    self._metrics["errors"] += 1
