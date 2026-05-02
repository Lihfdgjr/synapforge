"""Native synapforge -- pure LNN+SNN training framework, zero torch in production code."""

from __future__ import annotations

__version__ = "0.1.0-integration"

# NOTE: subpackages (vjp, data, bench, cuda, spike, stdp, kernel, dispatch,
# modal, auxsched) are imported lazily by callers to avoid forcing optional
# heavy dependencies (cupy, triton, transformers) at package import time.
__all__: list[str] = ["__version__"]
