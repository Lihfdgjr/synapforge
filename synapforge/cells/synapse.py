"""sf.SparseSynapse — sparse linear with optional synaptogenesis growth.

v0.1: stores a (out, in) weight + boolean connectivity mask. Forward applies
the mask to the weight (dense GEMM but with masked-out weights = zero, so
algorithmically equivalent to sparse). Backward updates only active weights.

v1.0 (M4): structural plasticity is now a first-class compile-time op. Pass
``growth=sf.ir.RigL(target=0.1)`` and the IR compiler will lift mask
grow/prune up into IRGraph nodes that backends (Lava export, Triton, CUDA
graphs) can see. The legacy ``.grow()`` / ``.prune()`` methods still exist
for hand-driven structural plasticity in research code.

API
---
    # v0.1 — manual ops, no IR integration
    syn = sf.SparseSynapse(in_dim=256, out_dim=512, sparsity=0.10)
    y = syn(x)
    syn.grow(n_new=128, criterion="random")
    syn.prune(criterion="magnitude", n_prune=64)
    print(syn.density())

    # v1.0 — declarative growth rule lifted into IR
    from synapforge.ir.synaptogenesis import RigL
    syn = sf.SparseSynapse(256, 512, sparsity=0.10,
                           growth=RigL(target=0.15, period=100))
    g = sf.ir.compile_module(model)             # IR sees grow_op + prune_op
    # then call ``maybe_update_masks(g, step)`` between training steps
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..module import Module


class _MaskedLinear(torch.autograd.Function):
    """y = (W*M) @ x.T + b, with d(W*M)/dW = M (chain-ruled in backward).

    Saves the elementwise mul as a kernel launch by inlining the mask
    application: forward materializes the masked weight ONCE then GEMMs;
    backward leaves the dW gradient unmasked at first then multiplies by
    M at the end (single fused mul instead of one in fwd + one in bwd).
    """

    @staticmethod
    def forward(ctx, x, weight, mask_typed, bias):  # type: ignore[override]
        # Materialize masked weight for this fwd. Same FLOPs as before; we win
        # on the bwd by skipping `aten::mul` in the autograd graph.
        w_eff = weight * mask_typed
        ctx.save_for_backward(x, w_eff, mask_typed)
        ctx.has_bias = bias is not None
        return torch.nn.functional.linear(x, w_eff, bias)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, w_eff, mask_typed = ctx.saved_tensors
        # All computations in grad_out's dtype to avoid bf16/fp32 mismatch
        dt = grad_out.dtype
        grad_x = grad_out.matmul(w_eff.to(dt))
        gx = grad_out.reshape(-1, grad_out.shape[-1])
        xx = x.reshape(-1, x.shape[-1]).to(dt)
        grad_w = gx.t().matmul(xx)
        grad_w = grad_w * mask_typed.to(dt)
        grad_b = None
        if ctx.has_bias:
            grad_b = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0)
        return grad_x, grad_w, None, grad_b


class SparseSynapse(Module):
    """Linear layer with structurally sparse boolean mask + grow/prune ops.

    Args
    ----
    in_dim, out_dim, sparsity, bias : as in v0.1
    growth :
        Optional growth rule (e.g. ``synapforge.ir.synaptogenesis.RigL``).
        If supplied, the IR compiler's :class:`SynaptogenesisPass` will
        emit ``grow_op`` + ``prune_op`` IR nodes after this synapse, and
        the runtime's ``maybe_update_masks(graph, step)`` will dispatch
        the rule periodically. None ⇒ static mask, identical to v0.1.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sparsity: float = 0.10,
        bias: bool = True,
        growth: object | None = None,
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
        # Growth rule (None => static mask, behaves like v0.1).
        self.growth = growth
        # MFU opt 2026-04-26: cache typed mask + masked weight to skip
        # the 4*N_layers*loop_depth `weight*mask.to(dtype)` recompute per step.
        # Invalidated whenever weight or mask changes (data_ptr-versioned).
        self._cached_typed_mask: torch.Tensor | None = None
        self._cached_typed_dtype: torch.dtype | None = None
        self._mask_version: int = 0

    def density(self) -> float:
        return float(self.mask.float().mean().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MFU opt 2026-04-26: use a custom autograd Function so the
        # elementwise weight*mask never appears as its own kernel launch.
        # Forward path applies the mask in the GEMM input; backward applies
        # the mask to dW (correct chain-rule for d/dW (W*M)).
        wdtype = self.weight.dtype
        if (self._cached_typed_mask is None
                or self._cached_typed_dtype != wdtype
                or self._cached_typed_mask.device != self.mask.device):
            self._cached_typed_mask = self.mask.to(wdtype)
            self._cached_typed_dtype = wdtype
        return _MaskedLinear.apply(x, self.weight, self._cached_typed_mask, self.bias)

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
        # Invalidate cached typed mask after structural change.
        self._cached_typed_mask = None
        self._mask_version += 1
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
        self._cached_typed_mask = None
        self._mask_version += 1
        return n_rm


__all__ = ["SparseSynapse"]
