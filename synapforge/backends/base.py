"""Backend ABC. Subclasses implement run(graph, *inputs) -> outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..ir.graph import IRGraph


class Backend(ABC):
    """Abstract executor for an IRGraph."""

    name: str = "base"

    @abstractmethod
    def run(self, graph: IRGraph, *inputs, **kwargs):
        """Execute the IR graph on the given inputs."""

    def warmup(self, graph: IRGraph, *inputs, **kwargs) -> None:
        """Optional warmup pass (kernel autotune, jit, ...). Default no-op."""
        return None


def get_backend(name: str) -> Backend:
    if name == "gpu_dense":
        from .gpu_dense import GPUDenseBackend
        return GPUDenseBackend()
    if name == "cpu_event":
        from .cpu_event import CPUEventBackend
        return CPUEventBackend()
    if name == "lava_export":
        from .lava_export import LavaExportBackend
        return LavaExportBackend()
    if name == "cpu_avx2":
        from .cpu_avx2 import CpuAvx2Backend
        return CpuAvx2Backend()
    if name == "triton_block":
        from .triton_block import TritonBlockBackend
        return TritonBlockBackend()
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["Backend", "get_backend"]
