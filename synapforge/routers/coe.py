"""sf.routers.ChainOfExperts — per-loop-step independent routers (CoE).

Reference
---------
- Wang et al., arXiv 2506.18945 (2025-06): "Chain-of-Experts" — recurrent
  shared backbone but each loop step uses an INDEPENDENT router. At ``R``
  loop iterations the model selects ``R`` different expert subsets per
  token, yielding "true depth" rather than re-running the same expert
  combination R times.

Why CoE on top of DeepSeek-MoE?
-------------------------------
A vanilla DeepSeek-MoE has ONE router shared across all loop steps. In
an R=4 RDT loop the same router selects the same experts every step, so
test-time compute scaling is limited (the loop just refines hidden state
with the same effective weights).

CoE swaps the single router for ``n_steps`` independent routers while
KEEPING the shared expert pool. Cost: ``n_steps * (hidden * n_routed)``
extra params (e.g. 4 * 256 * 16 = 16K, < 2% block overhead). Benefit:
each loop step is a different "way of thinking" — true diverse depth.

Integration
-----------
- ``ChainOfExperts`` accepts ``step_t`` arg in forward; selects
  ``self.routers[step_t]`` for that step.
- ``attach_coe_to_block`` returns an ``sf.Module`` wrapper that reads
  ``RDTLoop`` 's ``_coe_step_t`` attribute (set before each step) and
  routes correctly.
- Aux load-balance loss is **accumulated across steps** in the wrapper —
  the wrapper resets at step 0 and adds at every later step.

bf16-friendly. Compatible with both gpu_dense and triton_block backends.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..module import Module
from .moe import (
    FineGrainedExpert,
    MoELoadBalanceLoss,
    RouterOutput,
    SharedExpertGroup,
    TopKRouter,
)


class ChainOfExperts(Module):
    """Per-step routers + shared expert pool.

    Forward::
        y = shared(x) + sum_{e in top_k(routers[step_t](x))}( w_e * routed_e(x) )

    Args
    ----
    hidden:        feature dim.
    n_routed:      number of routed (sparse) experts.
    n_shared:      number of always-active experts.
    top_k:         experts selected per token per step.
    routed_ratio:  inner-dim shrink for routed experts (D/r).
    shared_ratio:  inner-dim shrink for shared experts.
    jitter_std:    train-time noise on router logits.
    n_steps:       number of independent routers — one per loop step.
    """

    def __init__(
        self,
        hidden: int,
        n_routed: int = 16,
        n_shared: int = 2,
        top_k: int = 2,
        routed_ratio: int = 4,
        shared_ratio: int = 2,
        jitter_std: float = 1e-2,
        n_steps: int = 4,
    ):
        super().__init__()
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        self.hidden = int(hidden)
        self.n_routed = int(n_routed)
        self.n_shared = int(n_shared)
        self.top_k = int(top_k)
        self.n_steps = int(n_steps)

        # Per-step routers (one per loop iteration; same shape, distinct weights).
        self.routers = nn.ModuleList(
            [
                TopKRouter(
                    hidden,
                    n_experts=n_routed,
                    top_k=top_k,
                    jitter_std=jitter_std,
                )
                for _ in range(n_steps)
            ]
        )

        # Shared expert pool (same for all steps — that's the "shared backbone").
        self.routed_experts = nn.ModuleList(
            [FineGrainedExpert(hidden, ratio=routed_ratio) for _ in range(n_routed)]
        )
        self.shared = SharedExpertGroup(hidden, n=n_shared, ratio=shared_ratio)

        self._last: RouterOutput | None = None
        self._current_step: int = 0
        # Per-step routing histogram (n_steps, n_routed) — fraction of tokens
        # routed to each expert at each step. Updated in forward (no grad).
        self.register_buffer(
            "_step_routing_hist",
            torch.zeros(n_steps, n_routed),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, step_t: int | None = None) -> torch.Tensor:
        if step_t is not None:
            self._current_step = min(max(0, int(step_t)), self.n_steps - 1)
        t = self._current_step

        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])

        out = self.shared(flat)
        r = self.routers[t](flat)
        self._last = r

        for k in range(self.top_k):
            idx_k = r.indices[:, k]
            w_k = r.weights[:, k].unsqueeze(-1)
            contrib = torch.zeros_like(flat)
            for e in range(self.n_routed):
                mask = idx_k == e
                if not mask.any():
                    continue
                xe = flat[mask]
                ye = self.routed_experts[e](xe)
                contrib[mask] = ye
            out = out + w_k * contrib

        # Update histogram (no grad): count tokens per expert at this step.
        with torch.no_grad():
            counts = torch.zeros(self.n_routed, device=x.device)
            for k in range(self.top_k):
                idx_k = r.indices[:, k]
                counts.scatter_add_(0, idx_k, torch.ones_like(idx_k, dtype=counts.dtype))
            counts = counts / max(1, idx_k.numel())
            self._step_routing_hist[t] = counts

        return out.reshape(orig_shape)

    def last_router_output(self) -> RouterOutput | None:
        return self._last

    def step_routing_histogram(self) -> torch.Tensor:
        """(n_steps, n_routed) tensor of expert-fraction per step.

        Useful for "proves diverse routing across steps" inspection.
        """
        return self._step_routing_hist


class _BlockWithCoE(Module):
    """Composition wrapper: residual CoE post-FFN with cross-step aux accum.

    Replaces the legacy mscfc.coe.attach_coe_to_block monkey-patch with a
    proper sf.Module composition. The wrapper:

      1. Forwards through the body.
      2. Reads ``self._coe_step_t`` (default 0). Set externally by RDTLoop.
      3. Adds residual CoE output keyed on that step.
      4. Resets aux at step 0; accumulates at steps > 0.
    """

    def __init__(
        self,
        body: nn.Module,
        coe: ChainOfExperts,
        aux_loss: MoELoadBalanceLoss,
    ):
        super().__init__()
        self.body = body
        self.coe = coe
        self.aux_loss = aux_loss
        # Externally settable hint (RDTLoop sets this before each call).
        self._coe_step_t: int = 0
        # Accumulated aux loss across loop steps (reset at step 0).
        self.last_moe_aux: torch.Tensor | None = None

    def forward(self, *args, **kwargs):
        out = self.body(*args, **kwargs)
        step_t = int(self._coe_step_t)

        def _apply(head: torch.Tensor) -> torch.Tensor:
            y = head + self.coe(head, step_t=step_t)
            r = self.coe.last_router_output()
            new_aux = self.aux_loss(r) if r is not None else head.new_zeros(())
            if step_t == 0 or self.last_moe_aux is None:
                self.last_moe_aux = new_aux
            else:
                self.last_moe_aux = self.last_moe_aux + new_aux
            return y

        if torch.is_tensor(out):
            return _apply(out)
        if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
            return (_apply(out[0]), *out[1:])
        return out


def attach_coe_to_block(
    block: nn.Module,
    hidden: int,
    n_routed: int = 16,
    n_shared: int = 2,
    top_k: int = 2,
    n_steps: int = 4,
    aux_alpha: float = 1e-2,
) -> _BlockWithCoE:
    """Wrap ``block`` in a residual Chain-of-Experts MoE post-FFN.

    Returns a new ``sf.Module``. RDTLoop should set ``wrapper._coe_step_t``
    before each loop step so the right router fires.

    Example::

        wrapped = attach_coe_to_block(my_block, hidden=256, n_steps=4)
        rdt = RDTLoop(wrapped, hidden=256, max_loops_train=4)  # sets _coe_step_t
        y = rdt(x)
        total_loss = ce_loss + wrapped.last_moe_aux
    """
    coe = ChainOfExperts(
        hidden,
        n_routed=n_routed,
        n_shared=n_shared,
        top_k=top_k,
        n_steps=n_steps,
    )
    aux = MoELoadBalanceLoss(alpha=aux_alpha)
    return _BlockWithCoE(block, coe, aux)


__all__ = [
    "ChainOfExperts",
    "attach_coe_to_block",
]
