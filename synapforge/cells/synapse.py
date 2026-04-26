"""sf.SparseSynapse — sparse linear with optional synaptogenesis growth.

v0.1: stores a (out, in) weight + boolean connectivity mask. Forward applies
the mask to the weight (dense GEMM but with masked-out weights = zero, so
algorithmically equivalent to sparse). Backward updates only active weights.

Synaptogenesis (`grow()`) is exposed as a manual op for v0.1 — it samples
new connections proportional to upstream activity x downstream gradient.
v0.2 will integrate Lava-style structural plasticity into the IR scheduler.

API
---
    syn = sf.SparseSynapse(in_dim=256, out_dim=512, sparsity=0.10)
    y = syn(x)
    syn.grow(n_new=128, criterion="random")  # or "hebb", "magnitude"
    syn.prune(criterion="magnitude", n_prune=64)
    print(syn.density())  # fraction of active connections
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..module import Module


class SparseSynapse(Module):
    """Linear layer with structurally sparse boolean mask + grow/prune ops."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sparsity: float = 0.10,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        # Boolean mask. True = active connection. Init random by sparsity frac.
        if not 0.0 < sparsity <= 1.0:
            raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")
        mask = (torch.rand(out_dim, in_dim) < float(sparsity))
        self.register_buffer("mask", mask)

    def density(self) -> float:
        return float(self.mask.float().mean().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mask weights for forward; gradient also flows only through active.
        w = self.weight * self.mask.to(self.weight.dtype)
        return torch.nn.functional.linear(x, w, self.bias)

    @torch.no_grad()
    def grow(self, n_new: int, criterion: str = "random") -> int:
        """Add up to n_new connections. Returns number actually added."""
        inactive = ~self.mask
        n_inactive = int(inactive.sum().item())
        if n_inactive == 0 or n_new <= 0:
            return 0
        n_add = min(n_new, n_inactive)
        if criterion == "random":
            inactive_idx = inactive.nonzero(as_tuple=False)
            sel = torch.randperm(len(inactive_idx), device=self.mask.device)[:n_add]
            picks = inactive_idx[sel]
        elif criterion == "magnitude":
            # Pick inactive slots whose weights are largest by abs value.
            scores = self.weight.abs().masked_fill(self.mask, -float("inf"))
            flat = scores.view(-1)
            _, top = flat.topk(n_add)
            picks = torch.stack([top // self.in_dim, top % self.in_dim], dim=1)
        else:
            raise ValueError(f"unknown grow criterion {criterion!r}")
        for r, c in picks:
            self.mask[int(r), int(c)] = True
            # Re-init the new connection small so it doesn't dominate.
            self.weight[int(r), int(c)] = self.weight.new_empty(()).normal_(0, 0.01)
        return n_add

    @torch.no_grad()
    def prune(self, n_prune: int, criterion: str = "magnitude") -> int:
        """Remove up to n_prune weakest active connections."""
        active = self.mask
        n_active = int(active.sum().item())
        if n_active == 0 or n_prune <= 0:
            return 0
        n_rm = min(n_prune, n_active)
        if criterion == "magnitude":
            scores = self.weight.abs().masked_fill(~active, float("inf"))
            flat = scores.view(-1)
            _, bottom = flat.topk(n_rm, largest=False)
            picks = torch.stack([bottom // self.in_dim, bottom % self.in_dim], dim=1)
        elif criterion == "random":
            active_idx = active.nonzero(as_tuple=False)
            sel = torch.randperm(len(active_idx), device=self.mask.device)[:n_rm]
            picks = active_idx[sel]
        else:
            raise ValueError(f"unknown prune criterion {criterion!r}")
        for r, c in picks:
            self.mask[int(r), int(c)] = False
            self.weight[int(r), int(c)] = 0.0
        return n_rm


__all__ = ["SparseSynapse"]
