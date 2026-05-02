"""synapforge.configs - named model size configs.

Exposes::

    from synapforge.configs import SYNAP1_BASE, SYNAP1_PRO, get_config

so that callers can pick a model size by name without hard-coding
``d=1024 n_layers=14 ffn_ratio=2.5 ...`` everywhere.
"""
from __future__ import annotations

from .synap1 import (
    SYNAP1_BASE,
    SYNAP1_PRO,
    Synap1Config,
    get_config,
    list_configs,
)

__all__ = [
    "Synap1Config",
    "SYNAP1_BASE",
    "SYNAP1_PRO",
    "get_config",
    "list_configs",
]
