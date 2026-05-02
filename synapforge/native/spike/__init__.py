"""synapforge.native.spike — Bit-packed spike storage + matmul kernels.

The PLIF cell emits ``s in {0, 1}`` of shape ``(B, T, d)``.  Stored as
fp16 each spike wastes 15 of 16 bits.  Packing 16 spikes per uint16
slot gives 16x memory and HBM-bandwidth savings at the PLIF -> synapse
boundary.

This subpackage layers as:

* ``pack`` — pure-numpy pack/unpack utilities (CPU-side, used by tests
  and offline tooling).  Bit-exact round-trip.
* ``packed_matmul`` — Triton kernel doing matmul on packed bits without
  unpacking to HBM.  Pure ``triton.language``, ``import torch``-free
  in the kernel itself.  Forward + backward.
* ``torch_glue`` — PyTorch ``autograd.Function`` glue + the
  ``--packed-spikes`` CLI flag wiring.  Auto-falls-back to dense when
  spike density > 30%.

LNN+SNN-specific
----------------
This optimisation only makes sense for binary-spike architectures.
A dense transformer activation cannot be compressed 16x without
quantisation noise; a binary spike *is* one bit by definition, so
packing is loss-less.  PLIF dead state means current density is 0%;
flag stays dormant until PLIF revives at ~5-15% density.
"""
from synapforge.native.spike.pack import pack_spikes, unpack_spikes

__all__ = [
    "pack_spikes",
    "unpack_spikes",
]
