"""Autonomous web learning + Track B retrieval memory + T8.4 EMA re-export."""

from .autonomous_daemon import AutonomousLearnDaemon, SelfGoalProposer
from .ema import EMATracker, ModelEMA, load_ema
from .retrieval_memory import PerUserMemory, RetrievalMemory

__all__ = [
    "AutonomousLearnDaemon",
    "SelfGoalProposer",
    "RetrievalMemory",
    "PerUserMemory",
    "EMATracker",
    "ModelEMA",
    "load_ema",
]
