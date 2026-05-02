"""SNN-architecture-specific compute kernels.

This subpackage holds *low-level* primitives that exploit properties no
generic transformer/LLM stack can take advantage of:

* :mod:`synapforge.kernels.sparse_spike_matmul` -- ``(s + h) @ W`` where
  ``s`` is a binary spike tensor; reformulated as
  ``h @ W + index_select_sum(W, where s==1)`` so the spike contribution
  costs ``O(K * out_dim)`` instead of ``O(in_dim * out_dim)``.  At a
  density of 5% (PLIF Pro target) the spike contribution is **20x
  cheaper** than dense ``(s.float() + h) @ W``; at 15% it is **~7x
  cheaper**.

* :mod:`synapforge.kernels.spike_pack` -- bit-pack 16 spikes per fp16
  slot (or 8 per int8) so that activation tensors at the
  spike-boundary occupy 16x less memory.  Useful for backbones with
  long context where the spike trains dominate activation memory.

Both modules expose pure-PyTorch reference implementations that work
on CPU and CUDA with any torch >= 2.0 (no triton required), plus
optional Triton fast paths that auto-activate when CUDA + triton are
both available.
"""
from __future__ import annotations

from .sparse_spike_matmul import (
    SPARSE_SPIKE_DEFAULT_THRESHOLD,
    sparse_spike_linear,
    sparse_spike_matmul,
)
from .spike_pack import pack_spikes_uint16, unpack_spikes_uint16

__all__ = [
    "SPARSE_SPIKE_DEFAULT_THRESHOLD",
    "sparse_spike_linear",
    "sparse_spike_matmul",
    "pack_spikes_uint16",
    "unpack_spikes_uint16",
]
