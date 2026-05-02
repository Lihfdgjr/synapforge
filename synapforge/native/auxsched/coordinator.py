"""coordinator.py -- AsyncAuxCoordinator: routes 4 aux components to streams.

Architecture
------------
Four background workers, each owning their own stream / thread::

    Stream A (main)         main forward + backward (NOT owned by us)
    Stream B (curiosity)    ICM forward + inverse + grad on aux head
    Stream C (TTT)          chunked inner loop, async portion
    Stream D (H2D)          host->device prefetch (NOT owned by us)
    CPU thread P1           NeuroMCP plasticity tick (Hebbian + codebook)
    CPU thread pool P2      ActionHead -> OSActuator tool exec

The coordinator is a thin shim: it owns the four drivers
(``CuriosityAsyncDriver``, ``TTTAsyncDriver``, ``NeuroMCPCpuDriver``,
``ActionHeadAsyncDriver``) and exposes a unified interface so the
trainer doesn't have to manage four lifetimes. The drivers themselves
do the real work.

Step protocol
-------------
Per outer step, the trainer is expected to:

1. ``submit_curiosity(step_idx, h_prev, h_next, ...)`` after main fwd
2. ``new_state, fut = submit_ttt(step_idx, val_x, val_y, state)``
3. ``submit_spike_stats(SpikeStats(...))`` after main fwd
4. ``submit_tool_call(ToolCall(...))`` if the ActionHead emitted one
5. (optional) ``wait_aux()`` to block on stream B/C completing if the
   trainer wants curiosity grad to flow into main backward
6. Continue to main backward
7. (optional) ``drain_observations()`` once per N steps to feed the
   slow STDP signal back into the codebook.

Backpressure policy
-------------------
Each driver implements its own backpressure strategy:

* curiosity: replace-stale (1-deep queue, oldest dropped on submit)
* TTT: replace-stale (same)
* NeuroMCP: drop-oldest-on-full (4-deep queue)
* ActionHead: drop-on-full (16-deep queue)

This is per-driver because the right policy depends on the
component. Curiosity / TTT are happy with 1-step lag; NeuroMCP can
tolerate skipped ticks (the EMA buffers smooth it); ActionHead must
*never* block the GPU, so dropping a tool call is preferable.

Hard constraint
---------------
**No ``import torch``.** Pure threading + queue + numpy + (optional)
cupy via ``streams.AuxStream``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from synapforge.native.auxsched.action_async import (
    ActionHeadAsyncDriver,
    ToolCall,
    ToolObservation,
)
from synapforge.native.auxsched.curiosity_async import (
    CuriosityAsyncDriver,
    CuriosityResult,
)
from synapforge.native.auxsched.future import AuxFuture
from synapforge.native.auxsched.neuromcp_cpu import (
    NeuroMCPCpuDriver,
    PlasticityResult,
    SpikeStats,
)
from synapforge.native.auxsched.streams import AuxStream
from synapforge.native.auxsched.ttt_async import TTTAsyncDriver, TTTStepStats


# ---------------------------------------------------------------------------
# Backpressure policy (per-driver knobs surfaced for the trainer)
# ---------------------------------------------------------------------------


class _DropMode(Enum):
    REPLACE_STALE = "replace-stale"      # 1-deep queue; new submit replaces old
    DROP_OLDEST = "drop-oldest"          # bounded queue; oldest evicted on full
    DROP_NEWEST = "drop-newest"          # bounded queue; new submit dropped on full


@dataclass
class AuxBackpressurePolicy:
    """Per-driver backpressure knobs.

    The default values match the spec defaults in each driver:

    * curiosity: REPLACE_STALE (1-deep)
    * ttt:       REPLACE_STALE (1-deep)
    * neuromcp:  DROP_OLDEST   (4-deep) -- drops oldest spike batch on full
    * action:    DROP_NEWEST   (16-deep) -- drops new tool call when full
    """

    curiosity_mode: _DropMode = _DropMode.REPLACE_STALE
    ttt_mode: _DropMode = _DropMode.REPLACE_STALE
    neuromcp_capacity: int = 4
    action_capacity: int = 16
    action_workers: int = 4


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class AuxQueueMetrics:
    """Snapshot of all four queues' metrics for telemetry."""

    curiosity: Dict[str, Any] = field(default_factory=dict)
    ttt: Dict[str, Any] = field(default_factory=dict)
    neuromcp: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    wait_aux_total_s: float = 0.0
    wait_aux_calls: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "curiosity": dict(self.curiosity),
            "ttt": dict(self.ttt),
            "neuromcp": dict(self.neuromcp),
            "action": dict(self.action),
            "wait_aux_total_s": self.wait_aux_total_s,
            "wait_aux_calls": self.wait_aux_calls,
        }


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class AsyncAuxCoordinator:
    """Orchestrates the 4 aux drivers behind a single trainer-facing API.

    Construction
    ------------
    The trainer supplies 4 callables (one per component). Each callable
    is the torch-side compute closure; the coordinator is torch-free
    and just schedules.

    The components are individually optional -- pass ``None`` for a
    callable to disable that component. Disabled components have their
    submit methods become no-ops, which keeps the trainer code path
    branchless.

    Lifecycle
    ---------
    Use as a context manager to ensure clean shutdown::

        with AsyncAuxCoordinator(...) as aux:
            for step in train_loop:
                ...
                aux.submit_curiosity(step, h_prev, h_next)
                _, fut = aux.submit_ttt(step, vx, vy, state)
                aux.submit_spike_stats(SpikeStats(...))
                aux.submit_tool_call(ToolCall(...))
                # main backward here
                if step % 10 == 0:
                    obs = aux.drain_observations()
                    feed_into_slow_stdp(obs)
    """

    def __init__(
        self,
        *,
        curiosity_compute_fn: Optional[Callable] = None,
        ttt_inner_step_fn: Optional[Callable] = None,
        ttt_done_fn: Optional[Callable] = None,
        ttt_total_k: int = 8,
        ttt_inline_k: int = 2,
        neuromcp_tick_fn: Optional[Callable] = None,
        neuromcp_initial_mask: Optional[np.ndarray] = None,
        action_execute_fn: Optional[Callable] = None,
        policy: Optional[AuxBackpressurePolicy] = None,
        curiosity_stream: Optional[AuxStream] = None,
        ttt_stream: Optional[AuxStream] = None,
    ) -> None:
        self._policy = policy or AuxBackpressurePolicy()
        self._wait_aux_total_s = 0.0
        self._wait_aux_calls = 0
        self._wait_aux_lock = threading.Lock()

        # ---- curiosity driver
        self._curiosity: Optional[CuriosityAsyncDriver]
        if curiosity_compute_fn is not None:
            self._curiosity = CuriosityAsyncDriver(
                compute_fn=curiosity_compute_fn,
                stream=curiosity_stream,
            )
        else:
            self._curiosity = None

        # ---- ttt driver
        self._ttt: Optional[TTTAsyncDriver]
        if ttt_inner_step_fn is not None:
            self._ttt = TTTAsyncDriver(
                inner_step_fn=ttt_inner_step_fn,
                done_fn=ttt_done_fn,
                total_k=ttt_total_k,
                inline_k=ttt_inline_k,
                stream=ttt_stream,
            )
        else:
            self._ttt = None

        # ---- neuromcp driver
        self._neuromcp: Optional[NeuroMCPCpuDriver]
        if neuromcp_tick_fn is not None:
            self._neuromcp = NeuroMCPCpuDriver(
                tick_fn=neuromcp_tick_fn,
                queue_capacity=self._policy.neuromcp_capacity,
                initial_mask=neuromcp_initial_mask,
            )
        else:
            self._neuromcp = None

        # ---- action driver
        self._action: Optional[ActionHeadAsyncDriver]
        if action_execute_fn is not None:
            self._action = ActionHeadAsyncDriver(
                execute_fn=action_execute_fn,
                num_workers=self._policy.action_workers,
                submit_capacity=self._policy.action_capacity,
            )
        else:
            self._action = None

    # ----- submit API ----------------------------------------------------

    def submit_curiosity(
        self,
        step_idx: int,
        h_prev: Any,
        h_next: Any,
        action_emb: Any = None,
        extra: Optional[dict] = None,
    ) -> Optional[AuxFuture]:
        if self._curiosity is None:
            return None
        return self._curiosity.submit(
            step_idx=step_idx,
            h_prev=h_prev,
            h_next=h_next,
            action_emb=action_emb,
            extra=extra,
        )

    def submit_ttt(
        self,
        step_idx: int,
        val_inputs: Any,
        val_targets: Any,
        inner_state: Any,
        extra: Optional[dict] = None,
    ) -> tuple[Any, Optional[AuxFuture]]:
        """Run inline_k TTT inner steps NOW; schedule the rest async.

        If TTT is disabled, returns ``(inner_state, None)`` (i.e. caller's
        state is unchanged).
        """
        if self._ttt is None:
            return inner_state, None
        return self._ttt.run(
            step_idx=step_idx,
            val_inputs=val_inputs,
            val_targets=val_targets,
            inner_state=inner_state,
            extra=extra,
        )

    def submit_spike_stats(self, stats: SpikeStats) -> None:
        if self._neuromcp is None:
            return
        self._neuromcp.submit_spikes(stats)

    def submit_tool_call(self, call: ToolCall) -> bool:
        if self._action is None:
            return False
        return self._action.submit(call)

    # ----- consumer API --------------------------------------------------

    def latest_neuromcp_mask(self) -> Optional[np.ndarray]:
        if self._neuromcp is None:
            return None
        return self._neuromcp.latest_mask()

    def latest_neuromcp_result(self) -> Optional[PlasticityResult]:
        if self._neuromcp is None:
            return None
        return self._neuromcp.latest_result()

    def drain_observations(self) -> List[ToolObservation]:
        if self._action is None:
            return []
        return self._action.drain_completed()

    def wait_aux(self, timeout: Optional[float] = None) -> None:
        """Block until in-flight curiosity + TTT futures complete.

        The trainer calls this only when it needs the auxiliary results
        synchronously (e.g. curiosity gradient flowing into main
        backward). Most steps should NOT call this.

        NeuroMCP and ActionHead are not waited on -- their results are
        consumed asynchronously via ``latest_neuromcp_mask`` /
        ``drain_observations``.
        """
        t0 = time.time()
        if self._curiosity is not None:
            try:
                self._curiosity.wait_latest(timeout=timeout)
            except TimeoutError:
                pass
        # TTT: peek at the most recent submitted future via metrics.
        # The driver doesn't expose latest directly; we use the
        # private member if present (this is OK -- coordinator and
        # driver are siblings in the same package).
        if self._ttt is not None:
            f = getattr(self._ttt, "_last_future", None)
            if f is not None and not f.done():
                try:
                    f.result(timeout=timeout)
                except TimeoutError:
                    pass
        with self._wait_aux_lock:
            self._wait_aux_total_s += time.time() - t0
            self._wait_aux_calls += 1

    # ----- metrics -------------------------------------------------------

    def metrics(self) -> AuxQueueMetrics:
        m = AuxQueueMetrics()
        if self._curiosity is not None:
            m.curiosity = self._curiosity.metrics()
        if self._ttt is not None:
            m.ttt = self._ttt.metrics()
        if self._neuromcp is not None:
            m.neuromcp = self._neuromcp.metrics()
        if self._action is not None:
            m.action = self._action.metrics()
        with self._wait_aux_lock:
            m.wait_aux_total_s = self._wait_aux_total_s
            m.wait_aux_calls = self._wait_aux_calls
        return m

    # ----- introspection (used by coordinator tests) ---------------------

    @property
    def curiosity_driver(self) -> Optional[CuriosityAsyncDriver]:
        return self._curiosity

    @property
    def ttt_driver(self) -> Optional[TTTAsyncDriver]:
        return self._ttt

    @property
    def neuromcp_driver(self) -> Optional[NeuroMCPCpuDriver]:
        return self._neuromcp

    @property
    def action_driver(self) -> Optional[ActionHeadAsyncDriver]:
        return self._action

    # ----- shutdown ------------------------------------------------------

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        for d in (self._curiosity, self._ttt, self._neuromcp, self._action):
            if d is not None:
                try:
                    d.shutdown(wait=wait, timeout=timeout)
                except Exception:
                    pass

    def __enter__(self) -> "AsyncAuxCoordinator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)
