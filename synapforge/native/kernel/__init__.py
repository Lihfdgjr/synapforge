"""synapforge.native.kernel — single-dispatch fused HybridBlock kernel.

This package fuses the entire ``synapforge.model_100m.HybridBlock`` forward
chain (RMSNorm + CfC step + PLIF + SEW + synapse + gate + RMSNorm + SwiGLU
FFN) into ONE Triton kernel launch instead of 7-9 dispatches, plus a
matching closed-form backward kernel.

Why this matters
----------------
At the production shape (B=48, T=256, d=1280, layers=16, loop_depth=1) the
old 7-dispatch path costs:

* 7 dispatches/block * 16 layers = 112 launches per forward
* Each launch: ~50us Python + scheduler overhead = 5.6ms wasted/step
* But more importantly: each kernel re-reads the (B,T,d) activation
  tensor from HBM. At d=1280 fp16 that's 50MB per tensor; reloading it
  7 times costs ~350MB/block of HBM bandwidth that vanishes when fused.

The bigger win is **register reuse** + **L2 hit**:
    RMSNorm output stays in SRAM, feeds CfC step directly. The PLIF
    membrane stays in SRAM, feeds the gate sigmoid directly. The whole
    block reads x once, writes y once.

Public surface
--------------
* ``fused_hybrid_fwd``  — Triton @triton.jit (lazy import, GPU-only)
* ``fused_hybrid_bwd``  — Triton @triton.jit (lazy import, GPU-only)
* ``FusedHybridBlock``  — torch.autograd.Function, drop-in glue
* ``HAS_TRITON``        — True only when triton is importable AND a CUDA
                          device is present at module load time. False on
                          Windows/CPU smoke test environments.

Usage from the trainer (when ``--fused-kernel`` is set):

    from synapforge.native.kernel import FusedHybridBlock
    block = FusedHybridBlock.from_hybrid_block(existing_hybrid_block)
    y = block(x)

Limitations
-----------
* No kwta-k support: when the model was constructed with ``kwta_k > 0``
  (sparse top-K gate mask), the fused kernel falls back to the
  non-fused path. The kwta path adds a topk + scatter that doesn't fit
  cleanly inside the single kernel.
* No high-pass residual: same reasoning — the optional Conv1d branch
  (NeurIPS 2025 §3.2) is OFF by default. When on, falls back.
* Sparse-spike synapse path is NOT fused (the row-gather kernel is a
  separate optimisation). When ``sparse_spike_synapse=True`` the dense
  fused path runs anyway (bit-exact) — the user can A/B benchmark.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Triton availability probe (lazy, never raises on import).
# ---------------------------------------------------------------------------
HAS_TRITON: bool = False
_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover -- depends on host
    import triton  # noqa: F401
    import triton.language  # noqa: F401
    HAS_TRITON = True
except Exception as _exc:  # pragma: no cover
    _IMPORT_ERROR = _exc

# Glue layer is always importable (PyTorch fallback when Triton missing).
from .fused_hybrid_torch import (  # noqa: E402
    FusedHybridBlock,
    fused_hybrid_block_apply,
    can_fuse_block,
)

__all__ = [
    "HAS_TRITON",
    "FusedHybridBlock",
    "fused_hybrid_block_apply",
    "can_fuse_block",
]
