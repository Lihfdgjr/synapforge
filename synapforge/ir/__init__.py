"""synapforge.ir — IR graph + compiler passes."""

from __future__ import annotations

from .compiler import compile_module
from .graph import IRGraph, IRNode
from .synaptogenesis import (
    CompilerPass,
    RigL,
    SynaptogenesisPass,
    maybe_update_masks,
)

__all__ = [
    "IRGraph",
    "IRNode",
    "compile_module",
    "CompilerPass",
    "RigL",
    "SynaptogenesisPass",
    "maybe_update_masks",
]
