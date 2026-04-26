"""sf.routers.DeepSeekMoE — fine-grained + shared expert MoE layer.

Reference
---------
- DeepSeek-MoE, arXiv 2401.06066: splits the FFN into many fine-grained
  experts (2-4x smaller each) plus a small set of always-active shared
  experts. Better expert specialisation than vanilla top-K MoE.
- OpenMythos (kyegomez/OpenMythos, 2026-04-19) uses 16 routed + 2 shared
  with top_k=2 on top of its recurrent-depth backbone.

What this module provides
-------------------------
1. ``FineGrainedExpert``         — D -> D/r -> D SwiGLU FFN.
2. ``SharedExpertGroup``         — n_shared FFNs always summed in.
3. ``TopKRouter``                 — linear router + softmax + top-K.
4. ``DeepSeekMoE``                — full layer wired together.
5. ``MoELoadBalanceLoss``         — Shazeer 2017 aux-loss.
6. ``attach_moe_to_block``        — convenience: wrap any sf.Module with
                                    a residual MoE post-FFN. Composes via
                                    ``nn.Sequential``-style wrapping (no
                                    monkey-patch of body.forward).

Param budget (D=256, r=4, n_routed=16, n_shared=2, top_k=2)::

    each routed expert  ~ 2 * D * (D/r) = 32k
    16 routed + 2 shared = 18 experts ~ 576k
    activated per token = top_k + n_shared = 4 -> ~12.5%

API
---
    >>> from synapforge.routers import DeepSeekMoE, MoELoadBalanceLoss
    >>> moe = DeepSeekMoE(hidden=256, n_routed=16, n_shared=2, top_k=2)
    >>> aux_loss_fn = MoELoadBalanceLoss(alpha=1e-2)
    >>> y = moe(x)
    >>> aux = aux_loss_fn(moe.last_router_output())
    >>> total_loss = ce_loss + aux

bf16-friendly: all internal ops are bf16-safe (no fp64, no log/exp on
unbounded inputs). Both gpu_dense and triton_block backends accept this
module as an opaque dense node.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module

# ---------------------------------------------------------------------------
# 1. Fine-grained expert (SwiGLU FFN)
# ---------------------------------------------------------------------------


class FineGrainedExpert(Module):
    """D -> D/r -> D SwiGLU FFN.

    SwiGLU (gated SiLU like LLaMA/Mistral) helps expert-specialisation
    dynamics in MoE settings.
    """

    def __init__(self, hidden: int, ratio: int = 4, bias: bool = False):
        super().__init__()
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}")
        self.hidden = int(hidden)
        mid = max(4, hidden // ratio)
        self.w_gate = nn.Linear(hidden, mid, bias=bias)
        self.w_up = nn.Linear(hidden, mid, bias=bias)
        self.w_down = nn.Linear(mid, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# 2. Shared experts (always active)
# ---------------------------------------------------------------------------


class SharedExpertGroup(Module):
    """Sum of ``n`` always-active experts (DeepSeek-MoE shared pool)."""

    def __init__(self, hidden: int, n: int = 2, ratio: int = 2, bias: bool = False):
        super().__init__()
        if n < 0:
            raise ValueError("n shared experts must be >= 0")
        self.experts = nn.ModuleList(
            [FineGrainedExpert(hidden, ratio=ratio, bias=bias) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.experts:
            return x.new_zeros(x.shape)
        out = self.experts[0](x)
        for e in self.experts[1:]:
            out = out + e(x)
        return out


# ---------------------------------------------------------------------------
# 3. Top-K router
# ---------------------------------------------------------------------------


@dataclass
class RouterOutput:
    """Output of a TopKRouter — packed for downstream consumption."""

    indices: torch.Tensor  # [N, K] expert ids
    weights: torch.Tensor  # [N, K] softmax weights (sum to 1 per token)
    probs: torch.Tensor    # [N, n_experts] full softmax
    n_experts: int


class TopKRouter(Module):
    """Linear -> softmax -> top-K with optional jitter noise.

    Works on either ``[B, D]`` or ``[B, T, D]`` inputs; flattens internally.
    """

    def __init__(
        self,
        hidden: int,
        n_experts: int,
        top_k: int = 2,
        jitter_std: float = 0.0,
    ):
        super().__init__()
        if top_k < 1 or top_k > n_experts:
            raise ValueError(f"top_k must be in [1, n_experts]={n_experts}, got {top_k}")
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.jitter_std = float(jitter_std)
        self.w = nn.Linear(hidden, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        # Cast logits to fp32 for numerical-stable softmax (bf16-friendly).
        logits = self.w(flat).float()
        if self.training and self.jitter_std > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_std
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        return RouterOutput(
            indices=indices,
            weights=weights.to(x.dtype),
            probs=probs,
            n_experts=self.n_experts,
        )


# ---------------------------------------------------------------------------
# 4. DeepSeek-MoE full layer
# ---------------------------------------------------------------------------


class DeepSeekMoE(Module):
    """Fine-grained + shared MoE.

    Forward::
        y = shared(x) + sum_{e in top_k}( w_e * routed_experts[e](x) )
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
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.n_routed = int(n_routed)
        self.n_shared = int(n_shared)
        self.top_k = int(top_k)
        self.router = TopKRouter(
            hidden, n_experts=n_routed, top_k=top_k, jitter_std=jitter_std
        )
        self.routed_experts = nn.ModuleList(
            [FineGrainedExpert(hidden, ratio=routed_ratio) for _ in range(n_routed)]
        )
        self.shared = SharedExpertGroup(hidden, n=n_shared, ratio=shared_ratio)
        # Cache last router output for aux-loss retrieval.
        self._last: RouterOutput | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        out = self.shared(flat)
        r = self.router(flat)
        self._last = r
        # Dense scatter aggregation. For modest n_routed this is the simplest
        # correct path. For very large expert counts use grouped-matmul.
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
        return out.reshape(orig_shape)

    def last_router_output(self) -> RouterOutput | None:
        return self._last


# ---------------------------------------------------------------------------
# 5. Load-balance auxiliary loss
# ---------------------------------------------------------------------------


class MoELoadBalanceLoss(Module):
    """Shazeer-style load-balance aux loss.

    Given the router output of a DeepSeekMoE layer, computes::

        f_e = mean over tokens of 1[e in top_k(x)]
        P_e = mean over tokens of router_prob_e(x)
        L   = alpha * n_experts * sum_e ( f_e * P_e )

    ``f`` is non-differentiable (hard assignment) but ``P`` is, so the
    gradient pushes the router to equalise usage.
    """

    def __init__(self, alpha: float = 1e-2):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, router_out: RouterOutput) -> torch.Tensor:
        E = router_out.n_experts
        with torch.no_grad():
            one_hot = F.one_hot(router_out.indices, num_classes=E).sum(dim=1)  # [N, E]
            f = one_hot.float().mean(dim=0)                                    # [E]
        P = router_out.probs.mean(dim=0)                                       # [E]
        return self.alpha * E * (f * P).sum()


# ---------------------------------------------------------------------------
# 6. attach_moe_to_block — composition without monkey patching
# ---------------------------------------------------------------------------


class _BlockWithMoE(Module):
    """Composition wrapper: ``y = body(x); return y + moe(y)``.

    Replaces the legacy mscfc.moe.attach_moe_to_block monkey-patch with a
    proper sf.Module composition. The body forward signature is preserved
    — if body returns a Tensor we add MoE residually; if body returns a
    tuple ``(x, *aux)`` we add to the first element only.
    """

    def __init__(
        self,
        body: nn.Module,
        moe: DeepSeekMoE,
        aux_loss: MoELoadBalanceLoss,
    ):
        super().__init__()
        self.body = body
        self.moe = moe
        self.aux_loss = aux_loss
        self.last_moe_aux: torch.Tensor | None = None

    def forward(self, *args, **kwargs):
        out = self.body(*args, **kwargs)
        if torch.is_tensor(out):
            y = out + self.moe(out)
            r = self.moe.last_router_output()
            self.last_moe_aux = self.aux_loss(r) if r is not None else out.new_zeros(())
            return y
        if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
            head = out[0]
            y = head + self.moe(head)
            r = self.moe.last_router_output()
            self.last_moe_aux = self.aux_loss(r) if r is not None else head.new_zeros(())
            return (y, *out[1:])
        return out


def attach_moe_to_block(
    block: nn.Module,
    hidden: int,
    n_routed: int = 16,
    n_shared: int = 2,
    top_k: int = 2,
    aux_alpha: float = 1e-2,
) -> _BlockWithMoE:
    """Wrap ``block`` in a residual DeepSeek-MoE post-FFN.

    Returns a new ``sf.Module`` (no in-place patching of ``block``). The
    returned wrapper exposes ``last_moe_aux`` which the trainer should add
    to the total loss.

    Example::

        wrapped = attach_moe_to_block(my_block, hidden=256)
        out = wrapped(x)
        total_loss = ce_loss + wrapped.last_moe_aux
    """
    moe = DeepSeekMoE(hidden, n_routed=n_routed, n_shared=n_shared, top_k=top_k)
    aux = MoELoadBalanceLoss(alpha=aux_alpha)
    return _BlockWithMoE(block, moe, aux)


__all__ = [
    "FineGrainedExpert",
    "SharedExpertGroup",
    "TopKRouter",
    "RouterOutput",
    "DeepSeekMoE",
    "MoELoadBalanceLoss",
    "attach_moe_to_block",
]
