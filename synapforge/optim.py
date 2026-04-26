"""sf.optim — multi-source optimizer with plasticity merge.

A single nn.Parameter can have multiple gradient sources (BP + STDP + Hebb...)
and they are merged ATOMICALLY into the optimizer step. This solves three
PyTorch pain-points when mixing BP and plasticity:

  (a) version conflicts — autograd does NOT see plast_delta as part of W's
      computation history, so naive `W -= bp_grad + stdp_delta` later trips
      `RuntimeError: a leaf Variable that requires grad is being used in an
       in-place operation`. We bypass this by merging into a single combined
      gradient that the optimizer (an Adam variant) consumes via .data.
  (b) interference — BP grad is dL/dW (we subtract); plasticity emits +ΔW
      (we add). Sign reconciliation happens inside compute_combined_grad.
  (c) lr-imbalance — each source has its own scaling factor in
      weight_per_source, applied BEFORE Adam normalization. Adam's m/v
      moments then absorb the merged signal so a noisy plasticity stream
      doesn't blow up the effective step size.

Sign convention
---------------
   BP:        loss.backward() populates p.grad = dL/dW. We want p ← p - lr*g,
              so we treat its sign as NEGATIVE in the merged grad.
   plast:     ΔW. We want p ← p + lr*ΔW, so its sign is POSITIVE.

Combined gradient g_eff fed to Adam is:
   g_eff = sum_src sign[src] * weight[src] * delta[src]
and the AdamW update is the standard p -= lr * m_hat / (sqrt(v_hat) + eps).

Public API
----------
    >>> import synapforge as sf
    >>> from synapforge.optim import build_optimizer, MultiSourceParam
    >>> # Tag a parameter at construction time so build_optimizer auto-detects:
    >>> w = sf.Param(torch.empty(d, d), grad_source=["bp", "stdp"])
    >>> opt = sf.optim.build_optimizer(model, lr=3e-4)
    >>> # ... training loop ...
    >>> loss.backward()
    >>> sf.plasticity.STDP_step(model, ms_param_table=opt.ms_param_table)
    >>> opt.step()  # consumes BOTH sources, resets caches

Limitations
-----------
* plast_delta is a detached tensor — it does NOT contribute to autograd. If
  you want a differentiable plasticity rule, use a chain that runs through
  the autograd engine itself (planned for v1.0).
* Only AdamW is implemented. SGD-with-multi-source is trivial to fork from
  PlasticityAwareAdamW.step() if needed.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ----------------------------------------------------------------------------
# sf.Param: small wrapper that tags an nn.Parameter with grad sources
# ----------------------------------------------------------------------------


def Param(
    data: torch.Tensor,
    requires_grad: bool = True,
    grad_source: Sequence[str] = ("bp",),
    weight_per_source: Optional[Dict[str, float]] = None,
) -> torch.nn.Parameter:
    """Build an nn.Parameter and tag it with `_sf_grad_source` metadata.

    `build_optimizer` reads `_sf_grad_source` and `_sf_weight_per_source`
    to decide which optimizer wrapper to attach.
    """
    p = torch.nn.Parameter(data, requires_grad=requires_grad)
    p._sf_grad_source = list(grad_source)  # type: ignore[attr-defined]
    if weight_per_source is not None:
        p._sf_weight_per_source = dict(weight_per_source)  # type: ignore[attr-defined]
    return p


# ----------------------------------------------------------------------------
# MultiSourceParam — tracks BP grad + plasticity deltas for one nn.Parameter
# ----------------------------------------------------------------------------


class MultiSourceParam:
    """Wrapper for an nn.Parameter that tracks multiple grad sources.

    Reads BP grad lazily from `param.grad` at combine-time (more reliable
    than register_hook, which runs once per .backward and can race with
    leaf accumulation when the same param is reached from multiple paths).
    Plasticity sources push deltas via attach_plast_delta(name, tensor).
    """

    __slots__ = ("param", "sources", "weight_per_source", "plast_delta",
                 "_bp_grad_cached", "_hook_handle")

    def __init__(
        self,
        param: torch.nn.Parameter,
        sources: Sequence[str] = ("bp",),
        weight_per_source: Optional[Dict[str, float]] = None,
    ) -> None:
        if not param.requires_grad and "bp" in sources:
            raise ValueError(
                f"Param has requires_grad=False but 'bp' is in sources={sources}; "
                "remove 'bp' or set requires_grad=True."
            )
        self.param = param
        self.sources = list(sources)
        self.weight_per_source = dict(weight_per_source or {s: 1.0 for s in sources})
        # Make sure every source listed has a weight even if user passed a
        # partial dict (e.g. {'stdp': 0.1} but sources=['bp','stdp']).
        for s in self.sources:
            self.weight_per_source.setdefault(s, 1.0)
        self.plast_delta: Dict[str, torch.Tensor] = {}
        self._bp_grad_cached: Optional[torch.Tensor] = None
        # Optional gradient-capture hook — kept for backward-compat with the
        # original spec, but compute_combined_grad also falls back to .grad
        # so this is belt-and-suspenders.
        self._hook_handle = None
        if "bp" in self.sources:
            self._hook_handle = param.register_hook(self._bp_grad_hook)

    # ------------------------------------------------------------------ hooks
    def _bp_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is not None:
            self._bp_grad_cached = grad.detach()
        return grad  # never modify

    # ------------------------------------------------------------------ deltas
    def attach_plast_delta(self, source: str, delta: torch.Tensor) -> None:
        """Register a plasticity-emitted ΔW for this param.

        Subsequent calls to the same source ACCUMULATE (sum) — so two STDP
        rules can both target the same weight without one overwriting the
        other. Reset on .reset() (called at end of optimizer.step()).
        """
        if source not in self.sources:
            raise KeyError(
                f"source {source!r} not in declared sources {self.sources}; "
                f"either widen the sources list or rename the plasticity rule."
            )
        if delta.shape != self.param.shape:
            raise ValueError(
                f"plast delta shape {tuple(delta.shape)} != param shape "
                f"{tuple(self.param.shape)} for source {source!r}"
            )
        d = delta.detach().to(dtype=self.param.dtype, device=self.param.device)
        if source in self.plast_delta:
            self.plast_delta[source] = self.plast_delta[source] + d
        else:
            self.plast_delta[source] = d

    # ----------------------------------------------------------------- combine
    def compute_combined_grad(self) -> Optional[torch.Tensor]:
        """Merge all sources into a single effective gradient.

        Sign convention: BP grad is dL/dW (subtract); plast delta is +ΔW
        (add). To present a unified "gradient" g_eff to AdamW (which does
        p -= lr * m_hat/(sqrt(v_hat)+eps)), we feed:
            g_eff = +1 * w_bp * bp_grad   (Adam will subtract → minimize loss)
                   -1 * w_pl * plast_delta (Adam will subtract → so minus ΔW
                                            inverts to +ΔW, which is the
                                            plasticity prescription).
        """
        # Gather the BP grad: prefer hook-cached (set during backward) but
        # fall back to live .grad (some autograd codepaths skip the hook).
        bp = self._bp_grad_cached
        if bp is None and "bp" in self.sources and self.param.grad is not None:
            bp = self.param.grad.detach()

        contribs: List[Tuple[str, torch.Tensor, float]] = []
        if bp is not None and "bp" in self.sources:
            contribs.append(("bp", bp, +1.0))  # subtract → descends loss
        for src, d in self.plast_delta.items():
            if src == "bp":
                continue
            contribs.append((src, d, -1.0))  # invert so AdamW's "-=" applies +ΔW

        if not contribs:
            return None

        combined = torch.zeros_like(self.param.data)
        for src, d, sign in contribs:
            w = float(self.weight_per_source.get(src, 1.0))
            combined.add_(d, alpha=sign * w)
        return combined

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Clear cached BP grad + plast deltas; call at end of optimizer.step()."""
        self._bp_grad_cached = None
        self.plast_delta.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (f"MultiSourceParam(shape={tuple(self.param.shape)}, "
                f"sources={self.sources}, "
                f"weights={self.weight_per_source})")


# ----------------------------------------------------------------------------
# PlasticityAwareAdamW — AdamW that consumes MultiSourceParam.combined_grad
# ----------------------------------------------------------------------------


class PlasticityAwareAdamW(torch.optim.Optimizer):
    """AdamW variant that pulls its gradient from MultiSourceParam.combined_grad.

    Numerically identical to torch.optim.AdamW when sources=["bp"] only and
    the BP grad is the only contributor (verified in test_optim.py). Adds
    plasticity-stream merging without breaking Adam's m/v moment dynamics.
    """

    def __init__(
        self,
        ms_params: Iterable[MultiSourceParam],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0,1), got {betas}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        ms_list = list(ms_params)
        if not ms_list:
            raise ValueError("PlasticityAwareAdamW: empty ms_params list")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = [msp.param for msp in ms_list]
        super().__init__(params, defaults)
        # id(param) → MultiSourceParam — id is stable for the param's lifetime
        self._ms_param_map: Dict[int, MultiSourceParam] = {
            id(msp.param): msp for msp in ms_list
        }

    # Public lookup for plasticity engines that want to .attach_plast_delta
    @property
    def ms_param_table(self) -> Dict[int, MultiSourceParam]:
        return self._ms_param_map

    def get_ms_param(self, param: torch.nn.Parameter) -> Optional[MultiSourceParam]:
        return self._ms_param_map.get(id(param))

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])

            for p in group["params"]:
                msp = self._ms_param_map.get(id(p))
                if msp is None:
                    continue
                grad = msp.compute_combined_grad()
                if grad is None:
                    continue
                if not torch.isfinite(grad).all():
                    # Reset and skip this param to avoid poisoning Adam's
                    # moments with NaN/Inf — log via an attribute flag.
                    msp.reset()
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]
                # Exponential moving averages
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # Bias-corrected estimates
                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t
                # Decoupled weight decay (AdamW)
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                # Step
                step_size = lr / bias_c1
                denom = (v.sqrt() / (bias_c2 ** 0.5)).add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

                # Reset for next step
                msp.reset()

        return loss

    # Override base zero_grad to also clear ms wrappers
    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for msp in self._ms_param_map.values():
            msp._bp_grad_cached = None
            msp.plast_delta.clear()


# ----------------------------------------------------------------------------
# build_optimizer factory — auto-detects sf.Param-tagged params
# ----------------------------------------------------------------------------


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    default_sources: Sequence[str] = ("bp",),
) -> PlasticityAwareAdamW:
    """Enumerate model.parameters(), wrap each in a MultiSourceParam.

    Source detection:
      * if a param has `_sf_grad_source` (set by sf.Param), use it
      * else use `default_sources` (default: ["bp"], pure-BP behavior)
    Per-source weights pulled from `_sf_weight_per_source` if present.
    """
    ms_params: List[MultiSourceParam] = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        sources = list(getattr(p, "_sf_grad_source", default_sources))
        wps = getattr(p, "_sf_weight_per_source", None)
        ms_params.append(
            MultiSourceParam(p, sources=sources, weight_per_source=wps)
        )
    if not ms_params:
        raise ValueError(
            "build_optimizer: model has no requires_grad=True parameters."
        )
    return PlasticityAwareAdamW(
        ms_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )


__all__ = [
    "Param",
    "MultiSourceParam",
    "PlasticityAwareAdamW",
    "build_optimizer",
]
