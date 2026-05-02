"""Intrinsic-motivation subpackage.

Historical layout: ``synapforge/intrinsic.py`` was a single flat module.
2026-05-02 wire-in (task #188): converted to a package so the new
self-drive coordinator can live next to the building blocks without
mutating the legacy module.

Back-compat surface:

    from synapforge.intrinsic import (
        FreeEnergySurprise, SelfGoalProposer, ImaginationRollout,
        NoveltyDrive, HomeostaticRegulator, IdleLoop, GoalMemory,
        IntrinsicReward,
    )

still works (re-exported from ``_core``). ``synapforge.intrinsic`` itself
also remains importable as a single attribute root, so any third-party
code that did ``import synapforge.intrinsic as I`` keeps doing so.

New self-drive surface (this PR):

    from synapforge.intrinsic.self_drive_coordinator import SelfDriveCoordinator
    from synapforge.intrinsic.frontier import FrontierSampler
    from synapforge.intrinsic.quality_guard import QualityGuard
"""

from __future__ import annotations

from ._core import (
    FreeEnergySurprise,
    GoalMemory,
    HomeostaticRegulator,
    IdleLoop,
    ImaginationRollout,
    IntrinsicReward,
    NoveltyDrive,
    SelfGoalProposer,
)

# New self-drive components (no torch in these files; pure-Python
# coordination on top of the torch-using building blocks above).
from .frontier import FrontierSampler
from .quality_guard import QualityGuard
from .self_drive_coordinator import SelfDriveCoordinator, SelfDriveConfig

__all__ = [
    # ---- legacy (torch-using) ----
    "FreeEnergySurprise",
    "GoalMemory",
    "HomeostaticRegulator",
    "IdleLoop",
    "ImaginationRollout",
    "IntrinsicReward",
    "NoveltyDrive",
    "SelfGoalProposer",
    # ---- new (torch-free coordination) ----
    "FrontierSampler",
    "QualityGuard",
    "SelfDriveCoordinator",
    "SelfDriveConfig",
]
