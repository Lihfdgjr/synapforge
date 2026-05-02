"""DEEP_MAINT_QUEUE.md T8.4 — EMA weights re-export at the spec import path.

The canonical implementation lives at :mod:`synapforge.training.ema` because
the trainer + production tests already import from there. The T8.4 spec
(see DEEP_MAINT_QUEUE.md) names the import path as
``synapforge.learn.ema::ModelEMA``; this module is a thin re-export so
both call patterns work::

    from synapforge.learn.ema import ModelEMA          # spec import
    from synapforge.training.ema import EMATracker     # legacy import

``ModelEMA`` and ``EMATracker`` are aliases of the same class.
"""
from __future__ import annotations

from synapforge.training.ema import EMATracker, ModelEMA, load_ema

__all__ = ["EMATracker", "ModelEMA", "load_ema"]
