"""CPUEventBackend — STUB. v0.3 will implement event-driven CPU exec via numba.

Right now this just delegates to GPUDenseBackend on CPU (so it's executable
but not actually event-driven). Tests import it to verify the interface
exists without erroring.
"""

from __future__ import annotations

import torch

from .base import Backend
from .gpu_dense import GPUDenseBackend
from ..ir.graph import IRGraph


class CPUEventBackend(Backend):
    name = "cpu_event"

    def __init__(self) -> None:
        super().__init__()
        self._gpu_dense = GPUDenseBackend(device="cpu")

    def run(self, graph: IRGraph, *inputs, **kwargs):
        # v0.1 stub: just runs PyTorch on CPU. v0.3 will replace with
        # numba event-loop scheduler over sparse spike events.
        cpu_inputs = tuple(
            x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs
        )
        return self._gpu_dense.run(graph, *cpu_inputs, **kwargs)


__all__ = ["CPUEventBackend"]
