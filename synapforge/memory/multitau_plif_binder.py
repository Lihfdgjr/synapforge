"""
MultiBandTau-PLIF binder — wires existing MultiBandTau into PLIF.tau_log.

PLIF currently uses a single ``tau_log`` parameter (one τ per channel).
This means the membrane time constant is fixed per channel, with a single
time horizon (~4-32K tokens before fade per agent synthesis).

MultiBandTau (bio/tau.py) is already built with 4 bands:
  - theta (τ≈50 ms)  — slow horizon, holds 100K+ token memory
  - alpha (τ≈25 ms)  — medium
  - beta  (τ≈12.5 ms) — fast
  - gamma (τ≈5 ms)   — very fast, current token

Wiring MultiBandTau into PLIF gives the slow theta channel that lets the
membrane state survive 10K+ steps. This is the single L2 (1-2M) drift fix
identified in agent synthesis 2026-04-30.

Usage:
    from synapforge.cells.plif import PLIF
    from synapforge.bio.tau import MultiBandTau
    from synapforge.memory.multitau_plif_binder import bind_multiband_tau

    plif = PLIF(hidden_size=1024)
    multitau = MultiBandTau(hidden_size=1024)
    bind_multiband_tau(plif, multitau, hidden_for_routing_fn=lambda: latest_h)

    # Now plif.tau() returns multitau-routed tau
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


def bind_multiband_tau(
    plif_module,
    multiband_tau,
    hidden_for_routing_fn: Optional[Callable] = None,
) -> None:
    """In-place patch PLIF.tau() to use MultiBandTau routing.

    PLIF.tau() originally returned `self.tau_log.exp().clamp(...)`.
    After binding, returns the multiband-routed tau. The routing depends
    on a hidden state — provide via `hidden_for_routing_fn` callable that
    returns the latest hidden tensor.

    Critical: the binding overwrites tau() on the instance, NOT the class.
    Multiple PLIFs can be bound to the same or different MultiBandTau.
    """
    if not hasattr(plif_module, "tau_log"):
        raise ValueError(f"{plif_module} has no tau_log; not a PLIF?")
    if not hasattr(multiband_tau, "tau_logs"):
        raise ValueError(f"{multiband_tau} has no tau_logs; not a MultiBandTau?")

    plif_module.add_module("_multiband_tau", multiband_tau)
    plif_module._original_tau_log = plif_module.tau_log
    plif_module._hidden_for_routing_fn = hidden_for_routing_fn

    def _new_tau(self) -> torch.Tensor:
        if (self._hidden_for_routing_fn is None
                or not hasattr(self, "_multiband_tau")):
            return self._original_tau_log.exp().clamp(1e-2, 1e3)
        try:
            h = self._hidden_for_routing_fn()
            if h is None:
                raise RuntimeError("hidden_for_routing_fn returned None")
            tau, _routing = self._multiband_tau(h)
            tau = tau.mean(dim=tuple(range(tau.dim() - 1)))
            return tau.clamp(1e-2, 1e3)
        except Exception:
            return self._original_tau_log.exp().clamp(1e-2, 1e3)

    import types
    plif_module.tau = types.MethodType(_new_tau, plif_module)


def unbind_multiband_tau(plif_module) -> None:
    """Restore PLIF.tau() to original single-tau behavior."""
    if hasattr(plif_module, "_multiband_tau"):
        del plif_module._multiband_tau
    if hasattr(plif_module, "_original_tau_log"):
        del plif_module._original_tau_log
    if hasattr(plif_module, "_hidden_for_routing_fn"):
        del plif_module._hidden_for_routing_fn

    def _orig_tau(self) -> torch.Tensor:
        return self.tau_log.exp().clamp(1e-2, 1e3)

    import types
    plif_module.tau = types.MethodType(_orig_tau, plif_module)


def bind_all_plifs_in_model(
    model: nn.Module,
    hidden_size: int,
    bands: tuple = ("theta", "alpha", "beta", "gamma"),
    hidden_for_routing_fn: Optional[Callable] = None,
) -> int:
    """Bulk bind every PLIF in model to a fresh MultiBandTau. Returns count."""
    from synapforge.bio.tau import MultiBandTau

    n = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "PLIF":
            multitau = MultiBandTau(hidden_size=hidden_size, bands=bands)
            bind_multiband_tau(module, multitau, hidden_for_routing_fn)
            n += 1
    return n
