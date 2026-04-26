"""sf.routers.MoRStack — Mixture of Recursions, per-token depth router.

Reference
---------
Google, arXiv 2507.10524 (2025-06): "Mixture of Recursions". Per-token
adaptive recursion depth on a shared body block. Each token decides
independently whether to take 1, 2, ..., max_depth passes through the
same body.

ACT (Adaptive Computation Time) variant
---------------------------------------
The naive soft-gate variant is leaky: per-token weights don't sum to 1.0,
so deeper passes stack arbitrarily. The Graves 2016 ACT remainder trick
(arXiv 1603.08983) fixes this::

    cumulative_p_t  += p_t  (until hitting threshold tau)
    weight_t         = remainder if would exceed else p_t
    cumulative_p     = cumulative_p_t (clipped)
    still_running   *= (1 - would_halt)

Once a token halts, its weight is zero on later passes; the remainder
mass is what's left of the budget. This keeps total weight per token
≤ 1.0 and makes the depth distribution interpretable.

API
---
    >>> body = MyBlock(d=256)              # any sf.Module returning (x, ...) tuple OR x
    >>> mor = MoRStack(body, hidden=256, max_depth=4, target_depth=2.5)
    >>> y, depth_loss = mor(x)             # (B, T, D), scalar
    >>> # depth_loss adds budget pressure to total loss
    >>> total_loss = ce_loss + depth_loss

Body signatures supported
-------------------------
1. ``body(x) -> x``                                       (simple)
2. ``body(x, *aux) -> (x, *aux)``                          (HybridBlock-ish)
3. ``body(x, h, membrane) -> (x, h, membrane, sr, W_fast)`` (mscfc legacy)

The router only routes the first tensor (x); aux tensors propagate
verbatim from the last "active" call (no soft-mix on aux to avoid
shape-coupling). This means MoR does not currently mix membrane/h
across depths — to enable that, use ``mix_aux=True``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module


def _call_body(body: nn.Module, x: torch.Tensor, aux: Sequence[Any]) -> tuple[torch.Tensor, tuple]:
    """Call body and normalise output to (feature_tensor, aux_tuple).

    Supports::
        body(x)                         -> Tensor       => (Tensor, ())
        body(x, *aux) -> Tensor                          => (Tensor, ())
        body(x, *aux) -> (Tensor, *more)                 => (Tensor, more)
    """
    out = body(x, *aux) if aux else body(x)
    if torch.is_tensor(out):
        return out, ()
    if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
        return out[0], tuple(out[1:])
    raise TypeError(
        f"MoRStack: body forward must return a Tensor or a tuple whose first "
        f"element is a Tensor; got {type(out).__name__}"
    )


def _mix(weight: torch.Tensor, new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Per-token soft mix: w * new + (1 - w) * old.

    weight is shaped (B,) or (B, T); new/old are (B, ..., D). Broadcasts
    weight to match feature dims.
    """
    if weight.dim() == 1 and new.dim() >= 2:
        # (B,) -> (B, 1, 1, ...)
        w = weight.view(weight.shape[0], *([1] * (new.dim() - 1)))
    elif weight.dim() == 2 and new.dim() >= 3:
        # (B, T) -> (B, T, 1, ...)
        w = weight.view(*weight.shape, *([1] * (new.dim() - 2)))
    else:
        w = weight
    return w * new + (1.0 - w) * old


class MoRRouter(Module):
    """Per-token gate that decides whether to recurse another step.

    Input:
        x:           (..., hidden)
        spike_rate:  optional scalar (or tensor) summary feature, e.g. PLIF
                     spike rate of the previous body call. Concatenated to x
                     so the router sees activity-level too. If you pass None
                     the router still works on x alone.

    Output:
        p:           (...) sigmoid probability of recurring deeper.
                     If hard=True a 0/1 mask with straight-through grad.
    """

    def __init__(self, hidden: int, dropout: float = 0.0, with_spike_rate: bool = True):
        super().__init__()
        self.with_spike_rate = bool(with_spike_rate)
        in_dim = hidden + 1 if with_spike_rate else hidden
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        spike_rate: torch.Tensor | None = None,
        hard: bool = False,
    ) -> torch.Tensor:
        if self.with_spike_rate:
            if spike_rate is None:
                sr = x.new_zeros(*x.shape[:-1], 1)
            else:
                sr = spike_rate.detach()
                if sr.dim() == 0:
                    sr = sr.expand(*x.shape[:-1]).unsqueeze(-1)
                else:
                    # Broadcast trailing dims to match x; assume scalar-per-token shape.
                    while sr.dim() < x.dim():
                        sr = sr.unsqueeze(-1)
                    sr = sr.expand(*x.shape[:-1], 1)
            feat = torch.cat([x, sr], dim=-1)
        else:
            feat = x
        logit = self.proj(feat).squeeze(-1)
        p = torch.sigmoid(logit)
        if hard:
            hard_p = (p > 0.5).to(p.dtype)
            return hard_p.detach() + p - p.detach()  # straight-through
        return p


class MoRStack(Module):
    """Per-token adaptive-depth wrapper around a body block.

    Compatible with both ``gpu_dense`` and ``triton_block`` backends — the
    routers/gates are plain ``nn.Linear`` so the IR compiler treats them
    as opaque dense ops (no special backend kernel needed).

    Args
    ----
    body:           the inner block to recurse. Called up to ``max_depth``
                    times per forward. Must return either a Tensor or a
                    tuple whose first element is the feature Tensor.
    hidden:         feature dim; used to size router gates.
    max_depth:      hard upper bound on recursion (incl. the mandatory
                    first pass).
    target_depth:   soft target for the average depth across batch+time.
                    Fed into a quadratic budget penalty.
    budget_weight:  multiplier on the budget loss. Returned as second arg.
    use_act_remainder: if True, use Graves ACT remainder trick (recommended).
    mix_aux:        if True, soft-mix any aux tensors (h, membrane) returned
                    by the body. If False (default), aux is taken from the
                    last "active" call. Default False to avoid shape coupling.
    aux_carry:      iterable of initial values for the body's auxiliary
                    inputs (e.g. ``(h0, membrane0)``). If provided the body
                    is called as ``body(x, *aux_carry)`` and aux is updated
                    each step from the body's tuple-return.
    """

    def __init__(
        self,
        body: nn.Module,
        hidden: int,
        max_depth: int = 4,
        target_depth: float = 2.5,
        budget_weight: float = 0.01,
        use_act_remainder: bool = True,
        mix_aux: bool = False,
        with_spike_rate: bool = True,
    ):
        super().__init__()
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.body = body
        self.hidden = int(hidden)
        self.max_depth = int(max_depth)
        self.target_depth = float(target_depth)
        self.budget_weight = float(budget_weight)
        self.use_act_remainder = bool(use_act_remainder)
        self.mix_aux = bool(mix_aux)
        # max_depth - 1 routers (the first pass is mandatory).
        self.routers = nn.ModuleList(
            [MoRRouter(hidden, with_spike_rate=with_spike_rate) for _ in range(max_depth - 1)]
        )
        # Histogram buffer (B,) of per-token depth weights summed over routers.
        # Updated in forward, read by .last_depth_histogram().
        self.register_buffer("_last_depth_sum", torch.zeros(1), persistent=False)

    # ----------------------------------------------------------------- helpers

    def last_depth_histogram(self) -> torch.Tensor:
        """Return last forward's depth distribution as a (max_depth+1,) tensor.

        Bucket k holds the number of tokens whose total weight fell in
        [k, k+1). Useful for "proves adaptive" prints in tests.
        """
        return self._last_depth_sum

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        x: torch.Tensor,
        *aux_carry: Any,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run up to ``max_depth`` recursive passes through ``body``.

        Returns
        -------
        x:           (..., hidden) — final mixed feature.
        depth_loss: scalar — budget penalty (RELU(avg_depth - target)^2 * weight).
        """
        # Token-shape book-keeping: budget is computed per leading dims.
        # We collapse all leading dims into one "(B*T,)" axis for histogram
        # purposes so MoR works for both (B, D) and (B, T, D).
        leading_shape = x.shape[:-1]
        n_tokens = int(torch.tensor(leading_shape).prod().item()) if leading_shape else 1

        # First pass — mandatory.
        x_new, aux = _call_body(self.body, x, aux_carry)
        x = x_new
        # Track running spike-rate hint for the router (if available in aux).
        sr_hint: torch.Tensor | None = None
        for a in aux:
            # Heuristic: any 0-dim float tensor in aux is treated as a spike rate.
            if torch.is_tensor(a) and a.dim() == 0 and a.is_floating_point():
                sr_hint = a
                break

        # Per-token weight running sum (used for budget loss + histogram).
        weight_sum = x.new_ones(leading_shape) if leading_shape else x.new_ones(1)

        if self.use_act_remainder:
            cumulative_p = x.new_zeros(leading_shape) if leading_shape else x.new_zeros(1)
            still_running = x.new_ones(leading_shape) if leading_shape else x.new_ones(1)
            act_tau = 0.99
        else:
            cumulative_p = None
            still_running = None
            act_tau = None

        for i in range(self.max_depth - 1):
            p = self.routers[i](x, sr_hint, hard=hard)

            if self.use_act_remainder:
                remainder = (1.0 - cumulative_p).clamp(min=0.0)
                would_halt = (cumulative_p + p) >= act_tau
                weight = torch.where(would_halt, remainder, p) * still_running
            else:
                weight = p

            # Early exit if hard mode and no token wants to continue.
            if hard and weight.sum() == 0:
                break

            # Recursive body call.
            x_old = x
            aux_old = aux
            x_call_input = x  # next pass uses current state
            x_new, aux_new = _call_body(self.body, x_call_input, aux_old)

            # Soft-mix x.
            x = _mix(weight, x_new, x_old)

            # Aux: either soft-mix (if shapes allow) or take the new branch.
            if self.mix_aux:
                mixed_aux = []
                for a_old, a_new in zip(aux_old, aux_new):
                    if torch.is_tensor(a_old) and torch.is_tensor(a_new) and a_old.shape == a_new.shape:
                        mixed_aux.append(_mix(weight, a_new, a_old))
                    else:
                        mixed_aux.append(a_new)
                aux = tuple(mixed_aux)
            else:
                aux = aux_new

            # Update running spike-rate hint.
            for a in aux:
                if torch.is_tensor(a) and a.dim() == 0 and a.is_floating_point():
                    sr_hint = a
                    break

            weight_sum = weight_sum + weight

            if self.use_act_remainder:
                cumulative_p = cumulative_p + weight
                still_running = still_running * (~would_halt).to(still_running.dtype)

        # Budget loss: penalise mean depth above target. Quadratic above-target
        # gives a smooth ramp, doesn't hurt sub-target tokens.
        avg_depth = weight_sum.mean()
        budget_loss = self.budget_weight * F.relu(avg_depth - self.target_depth).pow(2)

        # Stash histogram (no grad) so callers can inspect.
        with torch.no_grad():
            flat = weight_sum.detach().reshape(-1).clamp(0.0, float(self.max_depth))
            # Round to integer bucket = floor(weight) for histogram.
            buckets = torch.zeros(self.max_depth + 1, device=x.device, dtype=torch.float32)
            idx = flat.floor().clamp(0, self.max_depth).long()
            buckets.scatter_add_(0, idx, torch.ones_like(flat, dtype=torch.float32))
            buckets = buckets / max(1, n_tokens)
            self._last_depth_sum = buckets

        return x, budget_loss


__all__ = ["MoRStack", "MoRRouter"]
