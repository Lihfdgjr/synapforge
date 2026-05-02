"""cpu_pool.py -- thread pool for parallel CPU tasks (AdamW shards).

Why
---
Numpy ops auto-parallelise across cores via OpenBLAS / MKL but ONLY
inside a single ``np.matmul`` / ``np.add`` / ... call. The Python loop
*over parameters* (``for p, g in zip(params, grads): ...``) runs on a
single core. With ~1000 param tensors in a 730M LNN+SNN model, that
loop is non-trivial; on a 32-core A800 host we want to parcel param
shards across worker threads so the per-tensor numpy work overlaps in
time.

The classical caveat: Python's GIL serialises *Python bytecode*, but
numpy releases the GIL for every C-level ufunc / BLAS call. So as long
as each task's body is "numpy ops on big arrays" the threads run truly
in parallel on different cores.

This pool is intentionally minimal -- ``submit`` returns a Future, plus
a ``map`` helper for the common "run f over a list" case. We do NOT
use ``concurrent.futures.ThreadPoolExecutor`` directly because:

* We want a long-lived pool with deterministic shutdown.
* We want graceful-shutdown on exception in any worker (record + bubble).
* We want the option to set thread name + ``setDaemon`` for debugging.

Hard constraint
---------------
**No ``import torch``.** Plain ``threading`` + ``queue`` + ``numpy``.
"""

from __future__ import annotations

import threading
import time
import traceback
from queue import Queue
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Future
# ---------------------------------------------------------------------------

class _Future:
    """Minimal Future-like result carrier. NOT threading-safe to chain."""

    __slots__ = ("_event", "_result", "_exception")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: Any = None
        self._exception: Optional[BaseException] = None

    def _set_result(self, result: Any) -> None:
        self._result = result
        self._event.set()

    def _set_exception(self, exc: BaseException) -> None:
        self._exception = exc
        self._event.set()

    def done(self) -> bool:
        return self._event.is_set()

    def result(self, timeout: Optional[float] = None) -> Any:
        """Block until the task finishes; re-raise its exception if any."""
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("Future.result timed out")
        if self._exception is not None:
            raise self._exception
        return self._result


# ---------------------------------------------------------------------------
# CpuWorkerPool
# ---------------------------------------------------------------------------

# Sentinel pushed onto the queue at shutdown so workers wake and exit.
_SENTINEL = object()


class CpuWorkerPool:
    """Thread pool for CPU tasks (parallel AdamW shard updates).

    Usage::

        pool = CpuWorkerPool(num_workers=8, name="adamw")
        try:
            futures = [
                pool.submit(adamw_step, p, g, m, v, lr=lr)
                for p, g, m, v in shards
            ]
            for f in futures:
                f.result()  # raises if a shard crashed
        finally:
            pool.shutdown()

    Or as a context manager::

        with CpuWorkerPool(num_workers=8, name="adamw") as pool:
            results = pool.map(fn, items)

    Behaviour
    ---------
    * Workers are daemon threads -- the interpreter can exit with the
      pool still alive (e.g. on Ctrl-C in training).
    * If a task raises, the exception is captured on its Future. Other
      tasks keep running. ``Future.result()`` re-raises.
    * ``shutdown(wait=True)`` is idempotent; ``submit`` after shutdown
      raises ``RuntimeError``.
    """

    def __init__(self, num_workers: int = 4, name: str = "cpu-pool") -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self._num_workers = int(num_workers)
        self._name = str(name)
        self._queue: "Queue[Any]" = Queue()
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"{self._name}-{i}",
                daemon=True,
            )
            self._workers.append(t)
            t.start()

    # ----- public API ------------------------------------------------------

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def name(self) -> str:
        return self._name

    def submit(self, fn: Callable[..., Any], *args: Any,
               **kwargs: Any) -> _Future:
        """Enqueue a task; returns a Future for its result."""
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("CpuWorkerPool is shut down")
            fut = _Future()
            self._queue.put((fn, args, kwargs, fut))
            return fut

    def map(self, fn: Callable[..., Any], items: Iterable[Any]) -> List[Any]:
        """Run ``fn(item)`` for each item in parallel; return results in
        order. Re-raises the first exception encountered.
        """
        futs: List[Tuple[int, _Future]] = []
        for i, item in enumerate(items):
            futs.append((i, self.submit(fn, item)))
        out: List[Any] = [None] * len(futs)  # type: ignore[list-item]
        first_exc: Optional[BaseException] = None
        for i, f in futs:
            try:
                out[i] = f.result()
            except BaseException as e:  # noqa: BLE001 - re-raise after waiting
                if first_exc is None:
                    first_exc = e
        if first_exc is not None:
            raise first_exc
        return out

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Stop accepting new tasks and join workers.

        Parameters
        ----------
        wait
            If True, block until all in-flight tasks finish. If False,
            push the sentinel and return immediately (workers exit
            after finishing their current task).
        timeout
            Per-thread join timeout (only used when ``wait=True``).
        """
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            for _ in self._workers:
                self._queue.put(_SENTINEL)
        if wait:
            deadline = None if timeout is None else time.time() + timeout
            for t in self._workers:
                remaining = None
                if deadline is not None:
                    remaining = max(0.0, deadline - time.time())
                t.join(timeout=remaining)

    # ----- context manager -------------------------------------------------

    def __enter__(self) -> "CpuWorkerPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # ----- internals -------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                return
            fn, args, kwargs, fut = item
            try:
                result = fn(*args, **kwargs)
                fut._set_result(result)
            except BaseException as exc:  # noqa: BLE001
                # Attach traceback for easier debugging when caller
                # eventually inspects the exception.
                exc.__synapforge_traceback__ = traceback.format_exc()  # type: ignore[attr-defined]
                fut._set_exception(exc)
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# AdamW shard helper
# ---------------------------------------------------------------------------

def adamw_step_shard(
    param: "Any", grad: "Any", m: "Any", v: "Any",
    *,
    lr: float, beta1: float, beta2: float, eps: float, weight_decay: float,
    bc1: float, bc2: float,
) -> None:
    """In-place AdamW step on one (param, grad, m, v) tuple.

    Pure numpy; the GIL is released during ``np.multiply`` / ``np.sqrt``
    so this function can run truly in parallel across worker threads.

    The caller pre-computes ``bc1 = 1 - beta1^t``, ``bc2 = 1 - beta2^t``
    so all shards share the same step counter without re-reading it.
    """
    import numpy as np

    if grad is None:
        return
    np.multiply(m, beta1, out=m)
    m += (1.0 - beta1) * grad
    np.multiply(v, beta2, out=v)
    v += (1.0 - beta2) * (grad * grad)
    m_hat = m / bc1
    v_hat = v / bc2
    update = m_hat / (np.sqrt(v_hat) + eps) + weight_decay * param
    param -= lr * update


def parallel_adamw_step(
    pool: CpuWorkerPool,
    params: Sequence["Any"],
    grads: Sequence["Any"],
    moments_m: Sequence["Any"],
    moments_v: Sequence["Any"],
    *,
    step: int,
    lr: float, beta1: float, beta2: float, eps: float, weight_decay: float,
) -> None:
    """Apply AdamW to all (param, grad, m, v) tuples in parallel.

    Blocks until every shard finishes. Re-raises the first shard
    exception.

    This is the function that goes on Stage C of the pipeline.
    """
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    futs = []
    for p, g, m, v in zip(params, grads, moments_m, moments_v):
        f = pool.submit(
            adamw_step_shard, p, g, m, v,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, bc1=bc1, bc2=bc2,
        )
        futs.append(f)
    first_exc: Optional[BaseException] = None
    for f in futs:
        try:
            f.result()
        except BaseException as e:  # noqa: BLE001
            if first_exc is None:
                first_exc = e
    if first_exc is not None:
        raise first_exc
