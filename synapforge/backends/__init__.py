"""synapforge.backends -- pluggable execution backends.

Registered in `base.get_backend(name)`:

    gpu_dense     v0.1 baseline (PyTorch passthrough)
    cpu_event     event-driven CPU path
    lava_export   neuromorphic export
    triton_block  v0.2 Triton-fused LNN+SNN block (CfC scan + PLIF + reset)
"""

from __future__ import annotations

from .base import Backend, get_backend

__all__ = ["Backend", "get_backend"]


def _has_triton_block() -> bool:
    """True if the Triton-fused backend module is importable (Triton present)."""
    try:
        from .triton_block import TritonBlockBackend  # noqa: F401
        from .triton_block_kernel import _HAS_TRITON
        return bool(_HAS_TRITON)
    except Exception:
        return False
