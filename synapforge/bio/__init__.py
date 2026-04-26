"""synapforge.bio — bio-inspired mechanisms ported from mscfc.

Drop-in components that compose with sf.LiquidCell / sf.PLIF /
sf.HybridBlock to add cortical priors absent from vanilla CfC+PLIF.

Modules
-------
- ``kwta``                — top-k winners-take-all sparsification
- ``tau``                 — TauSplit / MultiBandTau  shared-leak time constants
- ``learnable_threshold`` — per-channel learnable PLIF threshold
- ``stdp_fast``           — STDP fast-weight memory (buffer-based, plasticity-synced)
- ``predictive``          — Rao-Ballard layer-wise prediction-error head
- ``astrocyte``           — calcium-driven slow modulator gate

All modules subclass ``sf.Module`` so they participate in the IR /
plasticity / event-graph hooks of the framework.  Each is independent —
you can mix and match without breaking sf.HybridBlock.

bf16-friendly: every numerical op is dtype-agnostic.  Buffers default
to ``torch.float32`` for stability of running stats but accept any
input dtype on forward.
"""

from __future__ import annotations

from .kwta import KWTA
from .tau import TauSplit, MultiBandTau
from .learnable_threshold import LearnableThreshold
from .stdp_fast import STDPFastWeight
from .predictive import PredictiveCoding
from .astrocyte import AstrocyteGate

__all__ = [
    "KWTA",
    "TauSplit",
    "MultiBandTau",
    "LearnableThreshold",
    "STDPFastWeight",
    "PredictiveCoding",
    "AstrocyteGate",
]
