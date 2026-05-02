"""synapforge.training -- training-side helpers (EMA tracker, etc.).

This subpackage hosts utilities that are *training-loop adjacent* but not
part of the model architecture itself: gradient accumulation helpers,
exponential-moving-average weight trackers, scheduler glue, and the like.

Kept separate from ``synapforge.optim`` (which builds optimizers) so that
the trainer can import ``from synapforge.training.ema import EMATracker``
without pulling in optimiser construction.
"""
from __future__ import annotations

from .ema import EMATracker, ModelEMA, load_ema

__all__ = ["EMATracker", "ModelEMA", "load_ema"]
