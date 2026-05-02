"""synapforge.native.dispatch -- heterogeneous CPU+GPU async dispatch layer.

Why this package exists
-----------------------
2026-05-02 user feedback (USER 21:30): *"异步就不能某些部件跑在 cpu 上，
一些跑在 gpu 上"*. Our current ZeRO-Offload Stage 0 is **sequential within
each step**: GPU does forward+backward, then CPU does AdamW, then GPU goes
again. The CPU sits idle during forward/backward; the GPU sits idle during
optim. This package fixes that by pipelining the three independent stages:

    Stage A (DataLoader thread)        feed batches into queue_A
    Stage B (GPU thread / compute)     forward + backward on batch N
    Stage C (CPU thread / AdamW)       optim step on grads from batch N-1

Steady state: while GPU runs step N's forward/backward, CPU is running
step N-1's AdamW. Throughput limited by ``max(t_B, t_C)`` rather than
their sum, so the speedup is ~``(t_B + t_C) / max(t_B, t_C)``. For our
730M LNN+SNN model on A800 we expect ``t_B ≈ t_C`` so the speedup
approaches ~1.6-1.9x once warmed up.

In addition, ``per_block_router`` lets each layer (or block) live on a
different device at runtime -- e.g. embed on CPU, blocks 0-7 on GPU,
blocks 8-15 on CPU, lm_head on GPU. Transfers between devices are
overlapped via a ``StreamPair`` (compute stream + transfer stream).

Hard constraints
----------------
* **Zero ``import torch``** in any file under this package. Pure
  ``numpy`` + ``cupy`` (optional, falls back to numpy on CPU-only host)
  + ``threading`` + ``queue``.
* Steady-state pipeline must produce **bit-identical** final params
  to the sequential reference (modulo fp32 reordering noise).
* Queue back-pressure prevents OOM if Stage C falls behind Stage B.

Public API
----------
* :class:`HeteroPipeline` -- 3-stage async pipeline (data -> GPU -> CPU).
* :class:`CudaStream` -- thin wrapper over ``cupy.cuda.Stream`` with
  pure-CPU fallback (no-op stream).
* :class:`StreamPair` -- compute + transfer stream pair for H2D/D2H
  overlap.
* :class:`CpuWorkerPool` -- thread pool for parallel AdamW shard updates.
* :class:`PerBlockRouter` -- per-block device routing with auto
  cross-device tensor transfer.

This module is the long-term replacement for the sequential code path
in ``synapforge.training.trainer.Trainer.train_step`` once Run 7
stabilises.
"""

from __future__ import annotations

from synapforge.native.dispatch.cpu_pool import CpuWorkerPool
from synapforge.native.dispatch.per_block_router import PerBlockRouter
from synapforge.native.dispatch.pipeline import HeteroPipeline, PipelineMetrics
from synapforge.native.dispatch.streams import (
    CUPY_AVAILABLE,
    CudaStream,
    StreamPair,
    asnumpy,
    to_device,
)

__all__ = [
    "CUPY_AVAILABLE",
    "CpuWorkerPool",
    "CudaStream",
    "HeteroPipeline",
    "PerBlockRouter",
    "PipelineMetrics",
    "StreamPair",
    "asnumpy",
    "to_device",
]
