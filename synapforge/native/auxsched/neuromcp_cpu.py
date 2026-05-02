"""neuromcp_cpu.py -- CPU-thread NeuroMCP plasticity tick.

What this is
------------
``synapforge.action.neuromcp`` (production, torch-using) maintains a
codebook of action prototypes that grows under novelty plus a sparse
synapse mask trained by Hebbian co-activation EMA. Both updates are
**non-autograd** -- they're discrete bookkeeping operations on running
statistics. There is no backward graph through them.

So they don't need to run on the GPU stream A at all. We can:

1. Have stream A push the relevant **spike statistics** (spike rate per
   neuron, codebook-similarity histogram, prototype usage counts) to a
   queue.
2. A dedicated CPU thread pops from the queue, runs the Hebbian +
   codebook-grow tick, and pushes the **new sparse mask + new alive
   prototype set** back to a result slot.
3. Stream A reads the new mask at the start of the *next* forward
   (one-step lag is fine -- codebook growth only happens once every
   ``growth_cooldown`` steps anyway).

This frees up ~1-2 ms / step on the GPU stream and removes the small
GIL-blocked Python loop that previously ran inline.

Hard constraint
---------------
**No ``import torch``.** Plain numpy. The trainer wires its torch-side
spike collector to push numpy arrays via :meth:`submit_spikes`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpikeStats:
    """Per-step spike telemetry the GPU side hands to the CPU plasticity tick.

    Fields are typed ``np.ndarray`` (CPU) -- the trainer is responsible
    for moving them off-device before passing them in.
    """

    step_idx: int
    spike_rate: np.ndarray      # [num_neurons] mean spike rate this step
    proto_sim: np.ndarray       # [K] cosine sim of latest input to each proto
    proto_used: np.ndarray      # [K] uint8/bool: which protos fired this step
    novelty: float = 0.0        # max distance to alive protos (input-side)
    extra: dict = field(default_factory=dict)


@dataclass
class PlasticityResult:
    """What the CPU tick produces for the next forward to read."""

    step_idx: int
    new_mask: Optional[np.ndarray] = None    # updated sparse synapse mask
    grew_prototype: bool = False             # did codebook add a new proto?
    pruned_count: int = 0                    # how many synapses pruned
    grown_count: int = 0                     # how many synapses grown
    elapsed_ms: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class NeuroMCPCpuDriver:
    """CPU-thread driver for NeuroMCP plasticity bookkeeping.

    The trainer's main fwd code calls :meth:`submit_spikes` at the end
    of each forward. The CPU thread pops, runs the user-supplied
    ``tick_fn``, and stores the result. The next forward calls
    :meth:`latest_mask` to retrieve the updated mask.

    Latency tolerance
    -----------------
    NeuroMCP plasticity is naturally low-frequency (codebook growth has
    a ``growth_cooldown=50`` step gap; sparse-mask growth is checked
    every 20 steps). One step of staleness on the mask is operationally
    indistinguishable from inline.

    Threading model
    ---------------
    Single CPU worker thread (so user state inside ``tick_fn`` -- e.g.
    EMA buffers -- has no race). Submissions go onto a bounded queue
    (default capacity 4). When the queue fills (e.g. CPU tick is
    slower than GPU forward), oldest entries are dropped silently --
    the consequence is that a few transient spike counts are not
    reflected in the running EMA, which the EMA's exponential decay
    forgives by design.
    """

    __slots__ = (
        "_tick_fn",
        "_queue",
        "_thread",
        "_shutdown",
        "_latest_lock",
        "_latest_result",
        "_metrics_lock",
        "_metrics",
        "_initial_mask",
    )

    def __init__(
        self,
        tick_fn: Callable[[SpikeStats, Optional[np.ndarray]], PlasticityResult],
        *,
        queue_capacity: int = 4,
        thread_name: str = "aux-neuromcp-cpu",
        initial_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        tick_fn
            ``(stats, prev_mask) -> PlasticityResult``. The trainer's
            Hebbian / codebook-grow code goes here. ``prev_mask`` is the
            current sparse synapse mask (or None on first call); the
            tick may return a new mask in :class:`PlasticityResult`.
        queue_capacity
            Max in-flight spike batches. Default 4 ~= 4 outer steps of
            slack. Old batches are dropped (oldest-first) when full.
        thread_name
            Worker thread name (debugging).
        initial_mask
            Optional initial mask to seed the running state. If None,
            ``tick_fn`` receives ``None`` on the first call.
        """
        self._tick_fn = tick_fn
        self._queue: "Queue[Optional[SpikeStats]]" = Queue(maxsize=int(queue_capacity))
        self._shutdown = False
        self._latest_lock = threading.Lock()
        self._latest_result: Optional[PlasticityResult] = None
        self._initial_mask = initial_mask
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "submitted": 0,
            "ticked": 0,
            "dropped": 0,
            "errors": 0,
            "total_tick_s": 0.0,
        }
        self._thread = threading.Thread(
            target=self._worker_loop, name=thread_name, daemon=True
        )
        self._thread.start()

    # ----- public API ----------------------------------------------------

    def submit_spikes(self, stats: SpikeStats) -> None:
        """Enqueue a spike-stats batch. Non-blocking; oldest dropped on full."""
        if self._shutdown:
            return
        with self._metrics_lock:
            self._metrics["submitted"] += 1
        # Drop oldest if full -- preserves the most-recent stats.
        try:
            self._queue.put_nowait(stats)
        except Exception:
            try:
                self._queue.get_nowait()
                with self._metrics_lock:
                    self._metrics["dropped"] += 1
            except Empty:
                pass
            try:
                self._queue.put_nowait(stats)
            except Exception:
                pass

    def latest_mask(self) -> Optional[np.ndarray]:
        """Most recent updated mask the next forward should consume.

        Returns ``None`` if no plasticity tick has completed yet, in
        which case the caller should keep using its current mask.
        """
        with self._latest_lock:
            r = self._latest_result
        return None if r is None else r.new_mask

    def latest_result(self) -> Optional[PlasticityResult]:
        with self._latest_lock:
            return self._latest_result

    def metrics(self) -> dict:
        with self._metrics_lock:
            d = dict(self._metrics)
        if d["ticked"] > 0:
            d["avg_tick_s"] = d["total_tick_s"] / d["ticked"]
        else:
            d["avg_tick_s"] = 0.0
        return d

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        # Wake the worker by pushing the sentinel.
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if wait:
            self._thread.join(timeout=timeout)

    def __enter__(self) -> "NeuroMCPCpuDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # ----- worker --------------------------------------------------------

    def _worker_loop(self) -> None:
        prev_mask = self._initial_mask
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                if self._shutdown:
                    return
                continue
            if item is None or self._shutdown:
                return
            stats: SpikeStats = item
            t0 = time.time()
            try:
                res = self._tick_fn(stats, prev_mask)
                if not isinstance(res, PlasticityResult):
                    res = PlasticityResult(step_idx=stats.step_idx)
                res.elapsed_ms = (time.time() - t0) * 1000.0
                if res.new_mask is not None:
                    prev_mask = res.new_mask
                with self._latest_lock:
                    self._latest_result = res
                with self._metrics_lock:
                    self._metrics["ticked"] += 1
                    self._metrics["total_tick_s"] += time.time() - t0
            except BaseException:  # noqa: BLE001
                with self._metrics_lock:
                    self._metrics["errors"] += 1
                # Don't crash the thread on user-tick errors -- swallow,
                # log, keep going. The trainer can inspect metrics.
