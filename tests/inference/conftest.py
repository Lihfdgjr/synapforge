"""Pytest collection hooks for inference tests.

Makes the repo root importable so ``import synapforge.inference`` resolves
regardless of where pytest was invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
