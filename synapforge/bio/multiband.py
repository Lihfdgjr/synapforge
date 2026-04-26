"""synapforge.bio.multiband — multi-band cortical-frequency time constants.

Standalone export of :class:`MultiBandTau` from :mod:`synapforge.bio.tau`,
the canonical implementation. Mirroring mscfc/bio.py module layout where
``MultiBandTau`` lived in its own logical section.

The class composes one ``log_tau`` parameter per cortical band (theta /
alpha / beta / gamma) with a learnable router: the router projects the
hidden state onto the bands; softmax over bands gives the mixture
weight; the output ``tau`` is the weighted sum of band-specific taus,
clamped into a safe range.

References:
    - Multi-band cortical oscillations: theta 4-8 Hz, alpha 8-12 Hz,
      beta 13-30 Hz, gamma 30-100 Hz (Buzsaki & Draguhn, 2004).
    - Used by SpikingSSMs / SiLIF / NeuTransformer to decouple time
      constants per cortical sub-population.

Why a separate file? Two reasons:

1. Logical separation -- ``tau.py`` co-locates ``TauSplit`` (PLIF/CfC
   shared base) with ``MultiBandTau`` (multi-frequency); having
   ``multiband.py`` lets a downstream caller import only the band
   machinery without dragging the offset-split logic.
2. Mirrors the mscfc namespace layout that paired ``MultiBandTau`` with
   ``AstrocyteGate`` as Phase-2 multi-timescale primitives.

bf16-friendly: the underlying implementation is dtype-agnostic.
"""

from __future__ import annotations

from .tau import MultiBandTau

__all__ = ["MultiBandTau"]
