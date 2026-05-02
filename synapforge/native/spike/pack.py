"""Pure-numpy spike pack / unpack utilities.

Layout
------
A binary spike tensor ``s`` of shape ``(..., d)`` is packed along the
last dimension into ``packed`` of shape ``(..., (d + 15) // 16)`` with
dtype ``uint16``.

Bit ``i`` of word ``w`` (little-endian within the word, so bit 0 is the
least significant bit) stores ``s[..., w * 16 + i]``.  Trailing bits in
the final word (when ``d`` is not a multiple of 16) are zero.  The
unpack routine takes ``n_orig`` and trims those padding bits off.

Why uint16 (not int32)
----------------------
``synapforge.kernels.spike_pack`` already provides an int32-storage
implementation that gives 8x compression.  This module commits to
genuine ``uint16`` storage for the full **16x** compression, matching
the bandwidth claim in the design doc.

Why pure numpy
--------------
Tests and offline tooling don't need PyTorch.  Keeping the pack /
unpack primitives PyTorch-free lets the bit-exact round-trip test
catch bugs even when torch is uninstalled, and matches the
``synapforge.native`` "kernel-grade primitive" theme: numpy in,
numpy out, no autograd machinery.

The torch-aware autograd-friendly variants live in
``synapforge.native.spike.torch_glue``.

Examples
--------
>>> import numpy as np
>>> s = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
...              dtype=np.float32)
>>> packed = pack_spikes(s)
>>> packed.shape
(1, 1)
>>> packed.dtype
dtype('uint16')
>>> int(packed[0, 0])  # bits 0, 2, 8 set
261
>>> unpack_spikes(packed, n_orig=16, dtype=np.float16).tolist()
[[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def pack_spikes(s: np.ndarray) -> np.ndarray:
    """Pack a binary spike array along its last dim, 16 bits per uint16 slot.

    Parameters
    ----------
    s : np.ndarray
        Binary spike array of shape ``(..., d)``.  Any dtype: bool,
        uint8, fp16, bf16, fp32 accepted.  Values are interpreted as
        ``s != 0`` -- non-binary inputs are clamped to 0/1.

    Returns
    -------
    np.ndarray
        Packed array of shape ``(..., (d + 15) // 16)`` dtype ``uint16``.

    Notes
    -----
    Padding strategy: when ``d`` is not a multiple of 16, the last word
    is padded on its high-order bits (bit indices >= ``d % 16``) with
    zero.  The round-trip is well-defined because ``unpack_spikes``
    discards bits beyond ``n_orig``.

    Bit-endianness: little-endian within the word, so bit 0 is the
    least significant bit.  This matches the natural slot layout where
    ``packed[..., w]`` covers spikes ``[w*16 : (w+1)*16]`` in increasing
    order.
    """
    s = np.asarray(s)
    if s.size == 0:
        # Preserve leading dims; empty last dim -> 0 packed slots.
        return np.zeros(s.shape[:-1] + (0,), dtype=np.uint16)

    d = s.shape[-1]
    n_slots = (d + 15) // 16
    pad = n_slots * 16 - d
    if pad > 0:
        # np.pad is overkill; concat zeros along last axis.
        zero_shape = s.shape[:-1] + (pad,)
        s = np.concatenate(
            [s, np.zeros(zero_shape, dtype=s.dtype)], axis=-1
        )

    # Convert to uint16 boolean (0 or 1) and reshape into slot groups.
    bits = (s != 0).astype(np.uint16)
    bits = bits.reshape(s.shape[:-1] + (n_slots, 16))
    # Bit positions 0..15.
    shifts = np.arange(16, dtype=np.uint16)
    # Each slot: sum_{i=0}^{15} bit[i] * 2^i.
    packed = (bits << shifts).sum(axis=-1).astype(np.uint16)
    return packed


def unpack_spikes(
    packed: np.ndarray,
    n_orig: int,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Unpack a uint16-packed spike array back to a binary array.

    Parameters
    ----------
    packed : np.ndarray
        uint16 array of shape ``(..., n_slots)`` produced by
        :func:`pack_spikes`.
    n_orig : int
        Original last-dim size.  Trailing padding bits in the final
        word are masked off.
    dtype : np.dtype, optional
        Output dtype.  Default ``np.float16`` (matching the design doc;
        downstream ``synapse(s + h)`` expects floats and the network is
        bf16/fp16 throughout).  Pass ``np.bool_`` for memory-efficient
        downstream consumers, or ``np.float32`` for tests / reference
        comparisons.

    Returns
    -------
    np.ndarray
        Unpacked array of shape ``(..., n_orig)`` with the requested
        dtype.

    Raises
    ------
    ValueError
        If ``packed`` does not have the slot count expected from
        ``n_orig``.
    """
    if dtype is None:
        dtype = np.float16

    packed = np.asarray(packed, dtype=np.uint16)
    n_slots_actual = packed.shape[-1] if packed.ndim > 0 else 0
    n_slots_expected = (n_orig + 15) // 16

    if n_orig == 0:
        return np.zeros(packed.shape[:-1] + (0,), dtype=dtype)

    if n_slots_actual != n_slots_expected:
        raise ValueError(
            f"unpack_spikes: packed has {n_slots_actual} slots along last "
            f"dim but n_orig={n_orig} expects {n_slots_expected} slots."
        )

    shifts = np.arange(16, dtype=np.uint16)
    # packed: (..., n_slots) -> (..., n_slots, 1); shift then & 1 -> bits.
    bits = (packed[..., None] >> shifts) & np.uint16(1)
    # Flatten the slot axis with the within-slot bit axis.
    bits = bits.reshape(packed.shape[:-1] + (n_slots_actual * 16,))
    # Trim padding to exactly ``n_orig`` bits.
    bits = bits[..., :n_orig]
    return bits.astype(dtype)


__all__ = ["pack_spikes", "unpack_spikes"]
