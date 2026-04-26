"""sf.Module — base class with first-class plasticity + event-graph hooks.

Inherits from torch.nn.Module so we get parameter/buffer tracking, .to(),
.cuda(), .state_dict() for free. We add:

  - `plasticity_step()` — invoked after each forward pass to apply Hebbian /
    STDP updates registered via `register_plasticity()`. v0.1 keeps these as
    plain buffers (no autograd path through the rule itself).
  - `event_hook()` / `register_event_hook()` — hooks the IR compiler can use
    to emit event nodes (no-op in v0.1 dense backend).
  - `compile_to_ir()` — passes self to synapforge.ir.compiler. Returns an
    IRGraph the runtime can dispatch to backends.

The convention is that every module in synapforge inherits from sf.Module,
NOT from nn.Module. This gives the IR compiler one entry point. Mixing
nn.Module (e.g., nn.Linear inside a sf.Module) is fine — they're treated as
opaque dense ops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn


class Module(nn.Module):
    """Base class for all synapforge cells / blocks / models.

    Adds plasticity buffers and IR compile hook on top of nn.Module.
    """

    def __init__(self) -> None:
        super().__init__()
        # Plasticity rules registered on this module (name -> rule callable).
        # Rules see (module, inputs, outputs) and update buffers in-place.
        self._plasticity_rules: dict[str, Callable[..., None]] = {}
        # Event hooks for the IR compiler. v0.1 ignores; future backends use.
        self._event_hooks: list[Callable[..., None]] = []
        # Last forward inputs/outputs cached for plasticity_step().
        self._last_io: tuple[Any, Any] | None = None

    # ------------------------------------------------------------------ API

    def register_plasticity(self, name: str, rule: Callable[..., None]) -> None:
        """Attach a plasticity rule to this module.

        rule is called as `rule(module, inputs, outputs)` after every forward.
        It should mutate buffers (e.g., a fast-weight tensor) in-place. v0.1
        does NOT route a gradient through the rule.
        """
        if name in self._plasticity_rules:
            raise KeyError(f"plasticity rule {name!r} already registered")
        self._plasticity_rules[name] = rule

    def register_event_hook(self, hook: Callable[..., None]) -> None:
        """Register an event hook for the IR compiler. No-op in v0.1."""
        self._event_hooks.append(hook)

    def plasticity_step(self) -> None:
        """Manually trigger plasticity rules using cached last forward I/O."""
        if self._last_io is None:
            return
        inputs, outputs = self._last_io
        for rule in self._plasticity_rules.values():
            rule(self, inputs, outputs)

    def compile_to_ir(self):
        """Compile this module to an IR graph (lazy import to avoid cycle)."""
        from .ir.compiler import compile_module
        return compile_module(self)

    # -------------------------------------------------- forward instrumentation

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        # Cache I/O for plasticity_step (skip if no rules to save memory).
        if self._plasticity_rules:
            self._last_io = (args, out)
            # auto-apply rules at the end of forward (training mode only by
            # default; plasticity is a learning signal, not inference state)
            if self.training:
                self.plasticity_step()
        return out


__all__ = ["Module"]
