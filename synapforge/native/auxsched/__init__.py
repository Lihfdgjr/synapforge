"""synapforge.native.auxsched -- async coordinator for auxiliary training components.

The package name is ``auxsched`` (auxiliary scheduler) rather than the
spec-suggested ``aux`` because Windows reserves ``AUX`` as a legacy DOS
device name (CON/PRN/AUX/NUL): native processes such as the git CLI on
Windows refuse to ``open()`` files inside any folder named ``aux`` /
``aux.*``. The ``auxsched`` rename is the load-bearing portability fix.

Why this package exists
-----------------------
2026-05-02 user (22:20):
    "我们训练的有好奇心，而且有自适应工具神经元"

Run 6/7 timing audit on A800 d=1280 bs=48:

    main fwd + bwd                  80-100 ms
    Curiosity ICM (forward+inverse)   5-10 ms   <- CHEAP
    Self-learn TTT 8-step inner       200 ms    <- 2-3x main step!
    NeuroMCP plasticity tick          1-2 ms    <- pure CPU
    ActionHead -> OSActuator        50-5000 ms  <- variable I/O

These components are SEPARABLE from the main loss path:

* Curiosity ICM -- own forward+inverse heads, own loss; doesn't need the
  main grad path for its own update. Computable in parallel as soon as
  ``(h_prev, h_next)`` exist (i.e. after the main forward).
* TTT inner loop -- 8 SGD steps on val rows. With 1-step staleness it
  can hide under the *next* outer step's data prefetch.
* NeuroMCP plasticity -- Hebbian + codebook lookup, no autograd needed.
  Pure CPU.
* ActionHead tool exec -- system calls (web fetch, shell, browser).
  Should NEVER block the training step.

Physical opportunity (sequential vs streamed)::

    sequential:  80 + 200 + 5 + 2 = 287 ms / step
    streamed:    max(80, 200, 5, 2) = 200 ms / step
                 1.4x speedup steady state, larger at higher TTT-k

Public API
----------
* :class:`AsyncAuxCoordinator` -- 4 background queues (one per
  component). ``submit_*`` enqueue the work, ``maybe_collect_*`` poll
  for finished results, ``wait_aux`` is the optional barrier.
* :class:`AuxStream` -- thin abstraction over ``cupy.cuda.Stream`` with
  CPU fallback (mirrors ``synapforge.native.dispatch.streams.CudaStream``
  but stands alone so this package doesn't hard-depend on dispatch).
* :class:`AuxFuture` -- minimal Future-like result carrier.
* :class:`CuriosityAsyncDriver` -- routing policy for ICM
  (see ``curiosity_async.py``).
* :class:`TTTAsyncDriver` -- TTT chunked async (1-2 inline + 6-7 stream).
* :class:`NeuroMCPCpuDriver` -- CPU thread plasticity tick.
* :class:`ActionHeadAsyncDriver` -- thread-pool tool executor.

Hard constraints
----------------
**Zero ``import torch``** in any production module under
``synapforge/native/auxsched/``. ``cupy.cuda.Stream`` + ``threading`` +
``queue`` only.
"""

from __future__ import annotations

from synapforge.native.auxsched.action_async import (
    ActionHeadAsyncDriver,
    ToolCall,
    ToolObservation,
)
from synapforge.native.auxsched.coordinator import (
    AsyncAuxCoordinator,
    AuxBackpressurePolicy,
    AuxQueueMetrics,
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
from synapforge.native.auxsched.streams import (
    CUPY_AVAILABLE,
    AuxStream,
    AuxStreamPair,
)
from synapforge.native.auxsched.ttt_async import (
    TTTAsyncDriver,
    TTTStepStats,
)

__all__ = [
    "ActionHeadAsyncDriver",
    "AsyncAuxCoordinator",
    "AuxBackpressurePolicy",
    "AuxFuture",
    "AuxQueueMetrics",
    "AuxStream",
    "AuxStreamPair",
    "CUPY_AVAILABLE",
    "CuriosityAsyncDriver",
    "CuriosityResult",
    "NeuroMCPCpuDriver",
    "PlasticityResult",
    "SpikeStats",
    "TTTAsyncDriver",
    "TTTStepStats",
    "ToolCall",
    "ToolObservation",
]
