"""LavaExportBackend — STUB. v0.4 will export to Intel Lava neuromorphic SDK."""

from __future__ import annotations

from .base import Backend
from ..ir.graph import IRGraph


class LavaExportBackend(Backend):
    name = "lava_export"

    def run(self, graph: IRGraph, *inputs, **kwargs):
        raise NotImplementedError(
            "Lava export is on the v0.4 roadmap. Use gpu_dense or cpu_event."
        )

    def export(self, graph: IRGraph, out_path: str) -> None:
        raise NotImplementedError("Lava export is on the v0.4 roadmap.")


__all__ = ["LavaExportBackend"]
