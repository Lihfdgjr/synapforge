"""HybridOptimizerDispatcher — route plasticity to STDP, BP to AdamW.

This is the single bridge between the new native STDP path and the
existing ``train_100m_kd.py`` pipeline. It owns:

* an ``STDPOnlyOptimizer`` for params tagged with ``"stdp"`` or
  ``"hebb"`` in their ``_sf_grad_source`` metadata
* a regular AdamW (or ``PlasticityAwareAdamW``) for everything else

Public API mimics ``torch.optim.Optimizer`` so it drops into the
trainer with one line. When ``--stdp-only-plasticity`` is OFF we
short-circuit and return the original optimizer unmodified — so
quality is bit-identical to current production.

Wiring path
-----------
In ``train_100m_kd.py``::

    if args.stdp_only_plasticity:
        from synapforge.native.stdp import HybridOptimizerDispatcher
        opt = HybridOptimizerDispatcher.from_model(
            model, base_optim_factory=lambda ps: build_optimizer(ps, lr=lr),
            base_alpha=args.stdp_alpha,
        )
    else:
        opt = build_optimizer(model, lr=lr)   # current path, unchanged

The dispatcher itself does NOT import torch at module level (we lazy
import only inside method bodies). The factory closure passed in by
the caller is what actually constructs ``PlasticityAwareAdamW`` —
this keeps the constraint clean.

Constraint compliance
---------------------
* No ``import torch`` at module level — guarded by lazy imports.
* When ``--stdp-only-plasticity`` is False the dispatcher is *not*
  constructed at all; the trainer takes the original code path.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .stdp_optimizer import STDPOnlyOptimizer


def _grad_sources(p: Any) -> list[str]:
    """Read ``_sf_grad_source`` off a param-like object, default ['bp']."""
    return list(getattr(p, "_sf_grad_source", ("bp",)))


def _is_plasticity_only(p: Any) -> bool:
    """True iff the param has only plasticity sources (no BP).

    A param tagged ``["bp", "stdp"]`` keeps BP routing — STDP merely
    contributes a delta to the existing AdamW path. Only params
    tagged purely ``["stdp"]`` (or ``["hebb"]``, etc) bypass AdamW
    entirely. This matches the Markram dual-rule convention: when
    BP is present it dominates and STDP is a corrective; when BP is
    absent the local rule is the only learning signal.
    """
    sources = _grad_sources(p)
    if not sources:
        return False
    return all(s in ("stdp", "hebb", "synaptogenesis") for s in sources)


class HybridOptimizerDispatcher:
    """Owns STDP and AdamW optimizers, routes step/zero_grad to both.

    Iterators like ``param_groups`` and methods like ``zero_grad`` /
    ``step`` are forwarded to both children. When
    ``base_optim_factory`` returns ``None`` (no BP params) we run
    STDP-only.

    Parameters
    ----------
    base_optim : torch.optim.Optimizer | None
        The optimizer for non-plasticity-only params. May be
        ``None`` if every param in the model is plasticity-only
        (degenerate case used by tests).
    stdp_optim : STDPOnlyOptimizer
        Owns the plasticity-only params.
    """

    def __init__(
        self,
        base_optim: Any,  # torch.optim.Optimizer | None — typed Any to keep no-torch import
        stdp_optim: STDPOnlyOptimizer | None,
    ) -> None:
        if base_optim is None and stdp_optim is None:
            raise ValueError(
                "HybridOptimizerDispatcher: at least one of base_optim or "
                "stdp_optim must be provided."
            )
        self.base = base_optim
        self.stdp = stdp_optim

    # ------------------------------------------------------------------ build
    @classmethod
    def from_model(
        cls,
        model: Any,
        base_optim_factory: Callable[[list], Any],
        *,
        base_alpha: float = 0.02,
        window: int = 20,
        a_plus: float = 0.02,
        a_minus: float = 0.02,
        clip: float = 1.0,
    ) -> HybridOptimizerDispatcher:
        """Split params into two groups and build both optimizers.

        ``base_optim_factory`` is a closure that takes a list of params
        and returns a torch optimizer. Typically the caller passes::

            lambda ps: build_optimizer_from_param_list(ps, lr=lr)

        We construct it lazily so the dispatcher itself never imports
        torch at module load.
        """
        plasticity_named: list[tuple[str, Any]] = []
        bp_named: list[tuple[str, Any]] = []
        for name, p in _iter_named_params(model):
            if not getattr(p, "requires_grad", True):
                continue
            if _is_plasticity_only(p):
                plasticity_named.append((name, p))
            else:
                bp_named.append((name, p))
        # Build STDP optimizer if any plasticity-only params exist
        stdp_optim: STDPOnlyOptimizer | None = None
        if plasticity_named:
            stdp_optim = STDPOnlyOptimizer.from_named_params(
                plasticity_named,
                base_alpha=base_alpha,
                window=window,
                a_plus=a_plus,
                a_minus=a_minus,
                clip=clip,
            )
        # Build base AdamW for the rest
        base_optim = None
        if bp_named:
            base_optim = base_optim_factory([p for _, p in bp_named])
        return cls(base_optim=base_optim, stdp_optim=stdp_optim)

    # ----------------------------------------------------------- counts
    def stdp_param_count(self) -> int:
        """Total weights under STDP control."""
        if self.stdp is None:
            return 0
        return self.stdp.total_params()

    def base_param_count(self) -> int:
        """Total weights under AdamW control."""
        if self.base is None:
            return 0
        n = 0
        for group in self.base.param_groups:
            for p in group["params"]:
                # numel via lazy torch
                n += int(p.numel()) if hasattr(p, "numel") else int(
                    p.shape[0] if p.shape else 1
                )
        return n

    # --------------------------------------------------------------- ops
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Forward to base; STDP has no grads to zero."""
        if self.base is not None:
            self.base.zero_grad(set_to_none=set_to_none)

    def step(self) -> dict:
        """Step both optimizers. STDP runs after AdamW so the local
        rule sees the freshly-updated BP weights (matches the ms
        plasticity OBSERVE/DELTA/APPLY contract).

        Returns a dict::

            {"adamw_loss": ..., "stdp_stats": {...}}
        """
        loss = None
        if self.base is not None:
            loss = self.base.step()
        stdp_stats: dict = {}
        if self.stdp is not None:
            stdp_stats = self.stdp.step()
        return {"adamw_loss": loss, "stdp_stats": stdp_stats}

    def observe_spike(self, name: str, pre_spike: Any, post_spike: Any) -> None:
        """Forward spike observation to the STDP optimizer."""
        if self.stdp is None:
            return
        self.stdp.observe(name, pre_spike, post_spike)

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        out: dict = {}
        if self.base is not None:
            out["base"] = self.base.state_dict()
        if self.stdp is not None:
            out["stdp"] = self.stdp.state_dict()
        return out

    def load_state_dict(self, sd: dict) -> None:
        if "base" in sd and self.base is not None:
            self.base.load_state_dict(sd["base"])
        if "stdp" in sd and self.stdp is not None:
            self.stdp.load_state_dict(sd["stdp"])


def _iter_named_params(model: Any) -> Iterable[tuple[str, Any]]:
    """Yield ``(name, param)`` from any model-like that has named_parameters.

    Falls back to a flat enumeration for test doubles. Avoids isinstance
    checks against torch types so we keep the no-import constraint.
    """
    if hasattr(model, "named_parameters"):
        return model.named_parameters()
    if hasattr(model, "params"):
        return ((str(i), p) for i, p in enumerate(model.params))
    return iter([])


__all__ = ["HybridOptimizerDispatcher"]
