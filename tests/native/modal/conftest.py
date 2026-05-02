"""Test fixtures for synapforge.native.modal.

Loads the modal modules without triggering ``synapforge.__init__``,
which currently imports torch. This mirrors the pattern in
``tests/native/dispatch/test_pipeline.py``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODAL_DIR = _REPO_ROOT / "synapforge" / "native" / "modal"


def _load_modal_modules():
    """Load the modal package without going through synapforge.__init__."""
    if "synapforge.native.modal.cross_modal" in sys.modules:
        return (
            sys.modules["synapforge.native.modal.packed_batch"],
            sys.modules["synapforge.native.modal.modal_mask"],
            sys.modules["synapforge.native.modal.cross_modal"],
            sys.modules["synapforge.native.modal.dispatch"],
        )

    # Build fake parent packages so absolute imports work.
    for name in (
        "synapforge",
        "synapforge.native",
        "synapforge.native.modal",
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    packed_batch = _load(
        "synapforge.native.modal.packed_batch",
        _MODAL_DIR / "packed_batch.py",
    )
    parent = sys.modules["synapforge.native.modal"]
    # Re-export packed_batch symbols so the other modules can find them.
    for k in dir(packed_batch):
        if not k.startswith("_"):
            setattr(parent, k, getattr(packed_batch, k))

    modal_mask = _load(
        "synapforge.native.modal.modal_mask",
        _MODAL_DIR / "modal_mask.py",
    )
    cross_modal = _load(
        "synapforge.native.modal.cross_modal",
        _MODAL_DIR / "cross_modal.py",
    )
    dispatch = _load(
        "synapforge.native.modal.dispatch",
        _MODAL_DIR / "dispatch.py",
    )
    return packed_batch, modal_mask, cross_modal, dispatch


# Pre-load before any test imports.
_load_modal_modules()
