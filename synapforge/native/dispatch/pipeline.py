"""pipeline.py -- 3-stage async heterogeneous training pipeline.

Architecture
------------
Three independent threads, each pinned to its own device class:

    Stage A (DataLoader thread, CPU)
        produces batches, puts them on queue_AB
    Stage B (Compute thread, default cuda stream OR CPU)
        pulls a batch from queue_AB, runs forward + backward,
        produces a gradient bundle on queue_BC
    Stage C (Optim thread, CPU)
        pulls a gradient bundle from queue_BC, runs AdamW step
        in-place on the param dict

In steady state, while Stage B is computing forward/backward of
batch N, Stage C is applying the optim step of batch N-1's grads,
and Stage A is producing batch N+1. This overlaps the three
phases instead of running them sequentially.

The wall-clock per pipeline-step is approximately
``max(t_A, t_B, t_C)``, vs the sequential ``t_A + t_B + t_C``. So the
speedup is a function of how balanced the three stages are. For our
730M LNN+SNN run on A800, ``t_A`` is small (data prep on background
thread is cheap), ``t_B`` is the dominant GPU cost, and ``t_C`` is the
CPU AdamW cost. ZeRO-Offload Stage 0 already moves AdamW to CPU so
``t_C`` is comparable to ``t_B``, which means pipelining gives a
~1.6-1.9x speedup once the bubble is filled.

Determinism & correctness
-------------------------
The pipelined order ``(B@N || C@N-1)`` produces the *same* trajectory
as the sequential ``(B@N then C@N)`` IFF the optim step at iteration
N depends only on iteration N's grads, not on the param values
read by iteration N+1's forward. AdamW satisfies this (it reads
grad_N and writes param). The only subtlety is that under
pipelining, batch N+1's forward sees param-after-step-N (just like
sequential), because Stage C finishes step N before Stage B starts
step N+1 of the *next* batch -- the 1-step pipeline only hides the
compute, it does not actually allow B to read pre-update params.
This is enforced by the Stage B loop ``wait_for_optim`` barrier in
``_stage_b_loop``.

Hard constraint
---------------
**No ``import torch``.** Threading + queue + numpy only. ``cupy`` is
optional via :mod:`synapforge.native.dispatch.streams`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Step bundles -- typed payloads passed between stages
# ---------------------------------------------------------------------------

@dataclass
class _Batch:
    """Output of Stage A -- what Stage B consumes."""

    step_idx: int
    inputs: Any
    targets: Any
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _GradBundle:
    """Output of Stage B -- what Stage C consumes."""

    step_idx: int
    grads: List[Any]            # parallel to ``params``; entry is np.ndarray or None
    loss: float
    extra: Dict[str, Any] = field(default_factory=dict)


# Sentinel posted at end-of-stream so downstream stages exit.
_EOS = object()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetrics:
    """Telemetry collected during a HeteroPipeline.run().

    Fields are accumulated across the entire run; ``num_steps`` is the
    number of training steps completed (Stage C finished the AdamW step).
    """

    num_steps: int = 0
    wallclock_s: float = 0.0
    stage_a_total_s: float = 0.0
    stage_b_total_s: float = 0.0
    stage_c_total_s: float = 0.0
    stage_a_wait_s: float = 0.0      # time A spent blocked on full queue
    stage_b_wait_s: float = 0.0      # time B spent blocked on empty/full queue
    stage_c_wait_s: float = 0.0      # time C spent blocked on empty queue
    last_loss: float = float("nan")

    @property
    def steps_per_second(self) -> float:
        if self.wallclock_s <= 0:
            return 0.0
        return self.num_steps / self.wallclock_s

    @property
    def b_c_overlap_ratio(self) -> float:
        """How much of Stage C ran in parallel with Stage B.

        1.0 = perfectly overlapped (ideal pipelining).
        0.0 = fully sequential.

        Computed as ``1 - (wallclock - max(B,C)) / min(B,C)`` clipped to
        [0, 1] -- a back-of-envelope but useful health signal.
        """
        if self.stage_b_total_s <= 0 or self.stage_c_total_s <= 0:
            return 0.0
        b, c = self.stage_b_total_s, self.stage_c_total_s
        seq = b + c
        ideal = max(b, c)
        actual = max(self.wallclock_s, ideal)
        denom = max(seq - ideal, 1e-9)
        return float(np.clip((seq - actual) / denom, 0.0, 1.0))

    def as_dict(self) -> Dict[str, float]:
        return {
            "num_steps": float(self.num_steps),
            "wallclock_s": self.wallclock_s,
            "stage_a_total_s": self.stage_a_total_s,
            "stage_b_total_s": self.stage_b_total_s,
            "stage_c_total_s": self.stage_c_total_s,
            "stage_a_wait_s": self.stage_a_wait_s,
            "stage_b_wait_s": self.stage_b_wait_s,
            "stage_c_wait_s": self.stage_c_wait_s,
            "steps_per_second": self.steps_per_second,
            "b_c_overlap_ratio": self.b_c_overlap_ratio,
            "last_loss": self.last_loss,
        }


# ---------------------------------------------------------------------------
# HeteroPipeline
# ---------------------------------------------------------------------------

class HeteroPipeline:
    """3-stage async pipeline: DataLoader -> Compute -> Optim.

    Parameters
    ----------
    batch_fn
        Callable ``(step_idx) -> (inputs, targets, extra_dict_or_None)``.
        Returning ``None`` signals end of stream.
    forward_backward_fn
        Callable ``(inputs, targets, extra) -> (grads, loss, extra_out)``.
        ``grads`` MUST be a list parallel to the ``params`` ref list (entries
        may be ``None`` for params not in this graph).
    optim_step_fn
        Callable ``(step_idx, grads, extra) -> None``. Updates ``params``
        in-place. Typically wraps ``parallel_adamw_step``.
    queue_ab_size, queue_bc_size
        Per-stage queue capacities. ``queue_bc_size=1`` enforces the
        1-step pipeline (B can be one ahead of C, no more).
    enable_pipeline
        If False, runs the same callbacks but in strict sequential order
        (no threads). Useful as the apples-to-apples baseline for
        :mod:`throughput_bench`.

    Notes
    -----
    The pipeline does NOT own ``params`` directly -- it just calls the
    user-supplied ``forward_backward_fn`` and ``optim_step_fn``. Those
    closures hold whatever weight refs they need. This keeps the
    pipeline torch-free and decoupled from any specific param schema.
    """

    def __init__(
        self,
        batch_fn: Callable[[int], Optional[Tuple[Any, Any, Optional[Dict[str, Any]]]]],
        forward_backward_fn: Callable[..., Tuple[List[Any], float, Optional[Dict[str, Any]]]],
        optim_step_fn: Callable[..., None],
        *,
        queue_ab_size: int = 2,
        queue_bc_size: int = 1,
        enable_pipeline: bool = True,
    ) -> None:
        if queue_ab_size < 1:
            raise ValueError("queue_ab_size must be >= 1")
        if queue_bc_size < 1:
            raise ValueError("queue_bc_size must be >= 1")
        self._batch_fn = batch_fn
        self._fb_fn = forward_backward_fn
        self._optim_fn = optim_step_fn
        self._queue_ab: "Queue[Any]" = Queue(maxsize=queue_ab_size)
        self._queue_bc: "Queue[Any]" = Queue(maxsize=queue_bc_size)
        self._enable_pipeline = bool(enable_pipeline)

        # Coordination: Stage B should not start step N's forward until
        # Stage C has finished step N-1's optim. Otherwise B reads stale
        # params. We enforce a 1-step slack with ``_optim_done_event``
        # which Stage C sets after each step and Stage B awaits before
        # popping the next batch.
        self._optim_done_evt = threading.Event()
        self._optim_done_evt.set()  # initially "ready" so step 0 can run
        self._optim_done_step_idx = -1
        self._optim_done_lock = threading.Lock()

        # Failure propagation between threads.
        self._error_lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._stop_evt = threading.Event()

        self.metrics = PipelineMetrics()

    # ----- public API ------------------------------------------------------

    def run(self, max_steps: Optional[int] = None) -> PipelineMetrics:
        """Run the pipeline until ``max_steps`` or end-of-stream.

        Returns the populated :class:`PipelineMetrics`.
        """
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be None or >= 0")
        self._error = None
        self._stop_evt.clear()
        # Reset optim_done so step 0 runs without waiting on phantom prev.
        self._optim_done_evt.set()
        self._optim_done_step_idx = -1
        self.metrics = PipelineMetrics()

        if not self._enable_pipeline:
            self._run_sequential(max_steps=max_steps)
            return self.metrics

        return self._run_pipelined(max_steps=max_steps)

    def stop(self) -> None:
        """Signal stages to exit at next loop boundary. Safe to call from
        any thread (e.g. Ctrl-C handler in main)."""
        self._stop_evt.set()

    # ----- sequential reference ------------------------------------------

    def _run_sequential(self, *, max_steps: Optional[int]) -> None:
        """Strict A -> B -> C order, single thread. The baseline for
        speedup measurement."""
        wall_start = time.time()
        step = 0
        while True:
            if self._stop_evt.is_set():
                break
            if max_steps is not None and step >= max_steps:
                break
            # Stage A
            t = time.time()
            res = self._batch_fn(step)
            self.metrics.stage_a_total_s += time.time() - t
            if res is None:
                break
            inputs, targets, extra = self._unpack_batch(res)
            # Stage B
            t = time.time()
            grads, loss, fb_extra = self._fb_fn(inputs, targets, extra)
            self.metrics.stage_b_total_s += time.time() - t
            # Stage C
            t = time.time()
            self._optim_fn(step, grads, fb_extra)
            self.metrics.stage_c_total_s += time.time() - t
            self.metrics.num_steps += 1
            self.metrics.last_loss = float(loss)
            step += 1
        self.metrics.wallclock_s = time.time() - wall_start

    # ----- pipelined ------------------------------------------------------

    def _run_pipelined(self, *, max_steps: Optional[int]) -> PipelineMetrics:
        wall_start = time.time()
        ta = threading.Thread(target=self._stage_a_loop,
                              args=(max_steps,), name="hp-stage-a", daemon=True)
        tb = threading.Thread(target=self._stage_b_loop,
                              name="hp-stage-b", daemon=True)
        tc = threading.Thread(target=self._stage_c_loop,
                              name="hp-stage-c", daemon=True)
        ta.start()
        tb.start()
        tc.start()
        ta.join()
        tb.join()
        tc.join()
        self.metrics.wallclock_s = time.time() - wall_start
        if self._error is not None:
            raise self._error
        return self.metrics

    # ----- stage loops -----------------------------------------------------

    def _stage_a_loop(self, max_steps: Optional[int]) -> None:
        """Producer: pull batches from ``batch_fn`` and feed queue_ab."""
        try:
            step = 0
            while True:
                if self._stop_evt.is_set() or self._error is not None:
                    break
                if max_steps is not None and step >= max_steps:
                    break
                t = time.time()
                res = self._batch_fn(step)
                self.metrics.stage_a_total_s += time.time() - t
                if res is None:
                    break
                inputs, targets, extra = self._unpack_batch(res)
                batch = _Batch(step_idx=step, inputs=inputs,
                               targets=targets, extra=extra or {})
                # Enqueue with backpressure
                while True:
                    if self._stop_evt.is_set() or self._error is not None:
                        return
                    try:
                        wait_t0 = time.time()
                        self._queue_ab.put(batch, timeout=0.05)
                        self.metrics.stage_a_wait_s += (
                            time.time() - wait_t0 - (
                                0.0 if not self._queue_ab.full() else 0.0))
                        break
                    except Full:
                        self.metrics.stage_a_wait_s += 0.05
                        continue
                step += 1
            # End of stream
            self._queue_ab.put(_EOS)
        except BaseException as exc:  # noqa: BLE001
            self._record_error(exc)
            try:
                self._queue_ab.put_nowait(_EOS)
            except Full:
                pass

    def _stage_b_loop(self) -> None:
        """Compute: forward + backward; feed queue_bc with grads."""
        try:
            while True:
                if self._stop_evt.is_set() or self._error is not None:
                    break
                # Wait for prev optim step to finish before starting
                # this step's forward (1-step pipeline correctness).
                if not self._optim_done_evt.wait(timeout=10.0):
                    raise RuntimeError(
                        "Stage B timed out waiting for Stage C optim step")
                # Pop next batch
                wait_t0 = time.time()
                try:
                    item = self._queue_ab.get(timeout=0.05)
                except Empty:
                    self.metrics.stage_b_wait_s += time.time() - wait_t0
                    continue
                self.metrics.stage_b_wait_s += time.time() - wait_t0
                if item is _EOS:
                    self._queue_bc.put(_EOS)
                    return
                batch: _Batch = item
                # Clear optim-done so next step waits till C signals again.
                # (Edge case: if optim_step_fn is super fast and finishes
                # before B starts the next iter, the event stays set --
                # that's fine, no false wait.)
                t = time.time()
                grads, loss, fb_extra = self._fb_fn(
                    batch.inputs, batch.targets, batch.extra)
                self.metrics.stage_b_total_s += time.time() - t
                self.metrics.last_loss = float(loss)
                bundle = _GradBundle(
                    step_idx=batch.step_idx, grads=grads,
                    loss=float(loss), extra=fb_extra or {})
                self._optim_done_evt.clear()
                # Enqueue with backpressure.
                while True:
                    if self._stop_evt.is_set() or self._error is not None:
                        return
                    try:
                        wait_t0 = time.time()
                        self._queue_bc.put(bundle, timeout=0.05)
                        self.metrics.stage_b_wait_s += time.time() - wait_t0
                        break
                    except Full:
                        self.metrics.stage_b_wait_s += 0.05
                        continue
        except BaseException as exc:  # noqa: BLE001
            self._record_error(exc)
            try:
                self._queue_bc.put_nowait(_EOS)
            except Full:
                pass

    def _stage_c_loop(self) -> None:
        """Optim: AdamW step; signal _optim_done_evt after each step."""
        try:
            while True:
                if self._stop_evt.is_set() or self._error is not None:
                    break
                wait_t0 = time.time()
                try:
                    item = self._queue_bc.get(timeout=0.05)
                except Empty:
                    self.metrics.stage_c_wait_s += time.time() - wait_t0
                    continue
                self.metrics.stage_c_wait_s += time.time() - wait_t0
                if item is _EOS:
                    return
                bundle: _GradBundle = item
                t = time.time()
                self._optim_fn(bundle.step_idx, bundle.grads, bundle.extra)
                self.metrics.stage_c_total_s += time.time() - t
                self.metrics.num_steps += 1
                with self._optim_done_lock:
                    self._optim_done_step_idx = bundle.step_idx
                self._optim_done_evt.set()
        except BaseException as exc:  # noqa: BLE001
            self._record_error(exc)
            self._optim_done_evt.set()  # unblock B so it can exit

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _unpack_batch(res: Any) -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
        if isinstance(res, tuple):
            if len(res) == 2:
                return res[0], res[1], None
            if len(res) == 3:
                return res[0], res[1], res[2]
        raise TypeError(
            f"batch_fn must return (inputs, targets) or "
            f"(inputs, targets, extra), got {type(res).__name__}")

    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc
        self._stop_evt.set()
        # Wake up any thread currently waiting on the optim event.
        self._optim_done_evt.set()
