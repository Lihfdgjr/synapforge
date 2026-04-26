"""synapforge — purpose-built ML framework for liquid + spiking + plastic networks.

Train dense, deploy async. v0.5 ships:

* PyTorch-backed dense backend (numerical-equivalent to mscfc.LiquidS4Cell)
* Triton block-fused parallel scan (~29x forward speedup, optional)
* PLIF cells with pluggable surrogate gradients
* Hebbian / STDP / BCM plasticity with grow-prune synaptogenesis
* PlasticityAwareAdamW optimizer
* Multi-GPU DDP wrapper with PlasticBufferSync
* Ternary (BitNet 1.58) post-training quantization (~20x weight compression)
* Lava export for Loihi-2 (M11 — pending hardware verification)

Public API
----------
    >>> import synapforge as sf
    >>> import torch
    >>>
    >>> class HybridBlock(sf.Module):
    ...     def __init__(self, d):
    ...         super().__init__()
    ...         self.cfc  = sf.LiquidCell(d, d)
    ...         self.plif = sf.PLIF(d, threshold=0.3)
    ...     def forward(self, x):
    ...         h = self.cfc(x)
    ...         spk, mem = self.plif(h)
    ...         return spk, mem
    >>>
    >>> model = HybridBlock(256)
    >>> rt = sf.compile(model, backend="gpu_dense")

Heavy / optional dependencies (pyarrow, transformers, datasets) are lazy-loaded;
the core API only requires torch and numpy.
"""

from __future__ import annotations

__version__ = "0.5.0"

# --- Core API (always available, torch-only) ---
from .module import Module
from .cells.liquid import LiquidCell
from .cells.plif import PLIF
from .cells.synapse import SparseSynapse
from .plasticity import (
    HebbianPlasticity, STDP,
    PlasticityRule, Hebbian, BCM, SynaptogenesisGrowPrune, PlasticityEngine,
)
from .runtime import compile, Runtime
from .optim import build_optimizer, MultiSourceParam, PlasticityAwareAdamW, Param
from .surrogate import spike, PLIFCell, register as register_surrogate
from . import distributed
from .distributed import init_dist, wrap_model, PlasticBufferSync


def __getattr__(name: str):
    """Lazy import of heavy / optional submodules."""
    if name == "train":
        from .train import train as _train
        return _train
    if name == "ParquetTokenStream":
        from .data import ParquetTokenStream as _PTS
        return _PTS
    raise AttributeError(f"module 'synapforge' has no attribute {name!r}")


__all__ = [
    "__version__",
    "Module",
    "LiquidCell",
    "PLIF",
    "SparseSynapse",
    "HebbianPlasticity",
    "STDP",
    "PlasticityRule",
    "Hebbian",
    "BCM",
    "SynaptogenesisGrowPrune",
    "PlasticityEngine",
    "compile",
    "Runtime",
    "build_optimizer",
    "MultiSourceParam",
    "PlasticityAwareAdamW",
    "Param",
    "spike",
    "PLIFCell",
    "register_surrogate",
    "train",
    "ParquetTokenStream",
    "distributed",
    "init_dist",
    "wrap_model",
    "PlasticBufferSync",
]
