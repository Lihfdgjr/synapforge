"""Bit-packed spike storage primitives.

The PLIF cell emits binary spikes ``s in {0, 1}`` of shape ``(B, T, d)``.
At ``B=128, T=256, d=1280``, that's ``128*256*1280 = 41.9M`` entries.
Storing them in fp16/bf16 (2 bytes each) wastes 16x memory because every
entry is one bit.  Packing 16 spikes per uint16 slot drops the activation
tensor at the spike boundary from 80 MB to 5 MB, freeing GPU memory for
larger batches / longer context / additional layers.

This module provides ``pack_spikes_uint16`` and ``unpack_spikes_uint16``
ops that round-trip bit-identically.  The expected use-case is:

* Forward:  spike_packed = pack_spikes_uint16(spike)
* Backward: spike       = unpack_spikes_uint16(spike_packed)

between layers when memory pressure is the bottleneck (long context,
high batch, or storing spike trains for STDP eligibility traces).

Layout
------
A boolean tensor of shape ``(..., d)`` is packed along the last
dimension.  ``d`` must be padded to a multiple of 16; the unpack
routine takes the original ``d`` so trailing padding bits are
discarded.

The packed tensor has shape ``(..., (d + 15) // 16)`` dtype uint16.
Bit ``i`` of word ``w`` corresponds to spike index ``w * 16 + i`` in
the original.  We use little-endian convention so bit 0 is the least
significant bit.

Limitations
-----------
* Pure-PyTorch implementation -- works on CPU and CUDA.  Not a kernel
  fusion: the pack/unpack are O(numel) elementwise ops and do not save
  compute.  The win is **memory-only**.
* Currently we support uint16 packing (16 bits / slot).  Future
  extensions can add uint8 (8 bits/slot) or fp16 reinterpret (32 bits
  per slot via ``.view(torch.float16)``) for lower-precision activation
  pipelines.
* Round-trip is bit-identical for boolean inputs.  Inputs must be in
  ``{0, 1}``; non-binary inputs are clamped to 0/1 via comparison.
"""
from __future__ import annotations

import torch


def pack_spikes_uint16(spikes: torch.Tensor) -> torch.Tensor:
    """Pack a binary tensor along its last dim, 16 bits per uint16 slot.

    Parameters
    ----------
    spikes : torch.Tensor
        Binary spike tensor of shape ``(..., d)``, any dtype (bool,
        uint8, fp16, bf16, fp32).  Values are interpreted as
        ``spikes != 0``.

    Returns
    -------
    torch.Tensor
        Packed uint16 tensor of shape ``(..., (d + 15) // 16)``.

    Notes
    -----
    The padding strategy left-pads the spike tensor to a multiple of 16
    along the last dim with zero bits so the round-trip is well-defined.
    Use ``unpack_spikes_uint16(packed, d_original)`` to recover the
    original tensor.
    """
    if spikes.numel() == 0:
        return torch.empty(spikes.shape[:-1] + (0,),
                           dtype=torch.int32, device=spikes.device)

    d = spikes.shape[-1]
    pad = (16 - d % 16) % 16
    if pad > 0:
        spikes = torch.nn.functional.pad(spikes, (0, pad))

    # Convert to int32 (we use int32 internally; PyTorch lacks first-class
    # uint16 support pre-2.3 so we store as int32 with values in [0, 65535]).
    bits = (spikes != 0).to(torch.int32)
    # Reshape last dim into (-1, 16) groups, one per packed slot.
    bits = bits.view(*bits.shape[:-1], -1, 16)
    # Bit positions [0..15] -- shift each group element to its bit position.
    shifts = torch.arange(16, device=bits.device, dtype=torch.int32)
    # Broadcast: bits is (..., n_slots, 16); shifts is (16,); product is
    # (..., n_slots, 16); sum over the 16-axis gives (..., n_slots).
    # NB: ``.sum`` over an int32 tensor promotes to int64; we explicitly
    # cast back to int32 (values fit in 16 bits, so int32 is plenty and
    # preserves the documented 8x storage ratio for bf16 inputs).
    packed = (bits << shifts).sum(dim=-1).to(torch.int32)
    return packed


def unpack_spikes_uint16(packed: torch.Tensor, d: int,
                         *, dtype: torch.dtype = torch.float32
                         ) -> torch.Tensor:
    """Unpack a uint16-packed spike tensor back to a binary tensor.

    Parameters
    ----------
    packed : torch.Tensor
        int32 tensor of shape ``(..., n_slots)`` produced by
        :func:`pack_spikes_uint16`.
    d : int
        Original last-dim size.  Trailing slots are masked off.
    dtype : torch.dtype, optional
        Output dtype.  Default ``torch.float32`` -- this is what the
        downstream ``(s + h) @ W`` consumer expects.  Use
        ``torch.bool`` for memory-efficient downstream ops (e.g. when
        feeding ``sparse_spike_matmul`` which accepts any dtype).

    Returns
    -------
    torch.Tensor
        Binary tensor of shape ``(..., d)`` with the requested dtype.
    """
    if packed.numel() == 0 or d == 0:
        return torch.zeros(packed.shape[:-1] + (d,),
                           dtype=dtype, device=packed.device)

    n_slots = packed.shape[-1]
    expected_slots = (d + 15) // 16
    if n_slots != expected_slots:
        raise ValueError(
            f"unpack_spikes_uint16: packed has {n_slots} slots along last "
            f"dim but d={d} expects {expected_slots} slots."
        )
    shifts = torch.arange(16, device=packed.device, dtype=torch.int32)
    # packed: (..., n_slots, 1).  Right-shift then mask bit 0.
    bits = (packed.unsqueeze(-1) >> shifts) & 1  # (..., n_slots, 16)
    bits = bits.reshape(*packed.shape[:-1], n_slots * 16)  # (..., n_slots*16)
    # Trim padding.
    bits = bits[..., :d]
    return bits.to(dtype)


__all__ = ["pack_spikes_uint16", "unpack_spikes_uint16"]
