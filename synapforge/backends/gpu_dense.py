"""GPUDenseBackend — v0.1: just call model.forward via PyTorch.

Why this is the right thing for v0.1
------------------------------------
The point of v0.1 is to prove the API surface compiles and runs end-to-end
with bit-equivalent semantics to mscfc. We achieve that by treating the
torch.nn.Module attached to the IR's root node as a black-box and calling
its forward(). Backward goes through standard torch.autograd.

v0.2 will add a Triton path that, for IR subgraphs of the form
liquid -> plif -> ..., emits a single fused kernel for the whole chunk.
That's a separate task by another agent — out of v0.1 scope.

Why not autograd.Function wrapping?
-----------------------------------
We could intercept forward and inject a custom kernel per node. But every
custom op risks breaking torch.compile / DDP / grad checkpointing. v0.1 is
a thin shim. Speed comes from kernels, not the dispatcher.
"""

from __future__ import annotations

import torch

from .base import Backend
from ..ir.graph import IRGraph


class GPUDenseBackend(Backend):
    name = "gpu_dense"

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.device = device  # if None, use input's device.

    def run(self, graph: IRGraph, *inputs, **kwargs):
        """Dispatch to the root module's forward.

        v0.1 ignores per-node IR scheduling — the root module's forward
        already calls children in order. Returns whatever the module returns.
        """
        root = graph.modules.get("root")
        if root is None:
            raise RuntimeError("graph has no root module")
        if self.device is not None:
            inputs = tuple(
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            )
            kwargs = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in kwargs.items()
            }
        return root(*inputs, **kwargs)

    def warmup(self, graph: IRGraph, *inputs, **kwargs) -> None:
        """One forward pass to JIT/cudnn-pick algorithms."""
        with torch.no_grad():
            _ = self.run(graph, *inputs, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


__all__ = ["GPUDenseBackend"]
