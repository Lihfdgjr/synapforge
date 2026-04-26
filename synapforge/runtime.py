"""sf.compile(model, backend) -> Runtime — entry point for execution.

    >>> rt = sf.compile(model, backend="gpu_dense")
    >>> y = rt(x)               # forward
    >>> y.sum().backward()      # backward via standard torch.autograd

Runtime owns:
    - the IRGraph (compiled metadata)
    - the backend instance (executor)
    - optional warmup state (e.g., autotune cache) for v0.2 Triton

Calling rt(*args) -> backend.run(graph, *args). Backward goes through
standard torch.autograd because the GPU dense backend just calls forward.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .backends.base import Backend, get_backend
from .ir.compiler import compile_module
from .ir.graph import IRGraph


class Runtime:
    """Bound model + backend, callable as a function."""

    def __init__(self, graph: IRGraph, backend: Backend) -> None:
        self.graph = graph
        self.backend = backend
        self._warmed_up = False

    def __call__(self, *args, **kwargs) -> Any:
        return self.backend.run(self.graph, *args, **kwargs)

    def warmup(self, *args, **kwargs) -> None:
        self.backend.warmup(self.graph, *args, **kwargs)
        self._warmed_up = True

    def parameters(self):
        """Expose parameters for optimizers (delegates to root module)."""
        root = self.graph.modules.get("root")
        if root is None:
            return iter([])
        return root.parameters()

    def train(self, mode: bool = True) -> Runtime:
        root = self.graph.modules.get("root")
        if root is not None:
            root.train(mode)
        return self

    def eval(self) -> Runtime:
        return self.train(False)

    def __repr__(self) -> str:
        return f"Runtime(backend={self.backend.name}, nodes={len(self.graph)})"


def compile(model: nn.Module, backend: str = "gpu_dense", **backend_kwargs) -> Runtime:
    """Compile a sf.Module to an executable Runtime.

    Args:
        model:   instance of sf.Module (or any torch.nn.Module)
        backend: one of "gpu_dense", "cpu_event", "lava_export"
        **backend_kwargs: forwarded to the Backend ctor (currently unused)

    Returns:
        Runtime — call it like a function: `rt(x)`.
    """
    graph = compile_module(model)
    be = get_backend(backend)
    return Runtime(graph, be)


__all__ = ["compile", "Runtime"]
