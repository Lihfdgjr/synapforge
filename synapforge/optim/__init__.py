"""synapforge.optim — multi-source optimizer with plasticity merge.

Public API (unchanged from v0.x single-file ``synapforge.optim``)
----------------------------------------------------------------
* ``Param`` — wrap an ``nn.Parameter`` with grad-source metadata
* ``MultiSourceParam`` — tracks BP grad + plasticity ΔW for one parameter
* ``PlasticityAwareAdamW`` — AdamW that consumes ``MultiSourceParam``
* ``build_optimizer`` — factory that auto-wraps every model parameter

New in 2026-Q2 (Phase 1 of the torch-replacement roadmap)
---------------------------------------------------------
* ``AdamW`` — pure-python, plasticity-free AdamW that operates on
  iterables of ``torch.Tensor``. See ``synapforge/optim/adamw.py`` and
  ``docs/TORCH_REPLACEMENT_PLAN.md``. Drop-in for ``torch.optim.AdamW``
  with ``fused=False`` numerics; provides a torch-API exit ramp without
  changing default behaviour.

The single-file ``synapforge.optim`` module was promoted to this package
in 2026-05 with no public API changes — every existing
``from synapforge.optim import X`` import continues to resolve via the
package ``__init__``. See ``docs/TORCH_REPLACEMENT_PHASE0_AUDIT.md`` for
the migration rationale.
"""
from __future__ import annotations

from ._legacy import (
    MultiSourceParam,
    Param,
    PlasticityAwareAdamW,
    build_optimizer,
)
from .adamw import AdamW
from .cpu_offload_adamw import CPUOffloadAdamW

__all__ = [
    "AdamW",
    "CPUOffloadAdamW",
    "MultiSourceParam",
    "Param",
    "PlasticityAwareAdamW",
    "build_optimizer",
]
