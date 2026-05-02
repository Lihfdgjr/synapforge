"""Round-trip + edge-case tests for ``synapforge.kernels.spike_pack``.

The contract is: ``unpack_spikes_uint16(pack_spikes_uint16(s), d) == s``
bit-identically for any boolean / 0-1 input.  We verify this for a
range of last-dim sizes spanning multiples-of-16 and not.
"""
from __future__ import annotations

import pytest
import torch

torch = pytest.importorskip("torch")

from synapforge.kernels.spike_pack import (  # noqa: E402
    pack_spikes_uint16,
    unpack_spikes_uint16,
)


@pytest.mark.parametrize("d", [1, 7, 15, 16, 17, 31, 32, 64, 128, 1280])
def test_round_trip_bit_identical(d):
    """pack -> unpack must reproduce the input exactly."""
    torch.manual_seed(d)
    s = (torch.rand(3, 5, d) > 0.5).to(torch.float32)
    packed = pack_spikes_uint16(s)
    expected_slots = (d + 15) // 16
    assert packed.shape[-1] == expected_slots, (
        f"d={d}: packed last-dim is {packed.shape[-1]}, expected "
        f"{expected_slots}."
    )
    unpacked = unpack_spikes_uint16(packed, d, dtype=torch.float32)
    assert torch.equal(unpacked, s), \
        f"d={d}: round-trip not bit-identical."


@pytest.mark.parametrize("d", [16, 1280])
def test_round_trip_bool_dtype(d):
    """Round-trip with bool output dtype (lowest-memory consumer path)."""
    torch.manual_seed(0)
    s = (torch.rand(2, 4, d) > 0.5)
    packed = pack_spikes_uint16(s)
    unpacked = unpack_spikes_uint16(packed, d, dtype=torch.bool)
    assert torch.equal(unpacked, s)


def test_all_zeros():
    """All-zero input packs to all-zero slots."""
    s = torch.zeros(2, 4, 32, dtype=torch.float32)
    packed = pack_spikes_uint16(s)
    assert torch.all(packed == 0)
    unpacked = unpack_spikes_uint16(packed, 32, dtype=torch.float32)
    assert torch.equal(unpacked, s)


def test_all_ones():
    """All-ones in 16-aligned dim packs to all 65535 (0xFFFF)."""
    s = torch.ones(2, 4, 32, dtype=torch.float32)
    packed = pack_spikes_uint16(s)
    # 0xFFFF = 65535 -- 16 bits all on per slot.
    assert torch.all(packed == 65535)
    unpacked = unpack_spikes_uint16(packed, 32, dtype=torch.float32)
    assert torch.equal(unpacked, s)


def test_padded_dim_round_trip():
    """When d is not a multiple of 16, padding bits must be discarded on unpack."""
    d = 17
    s = (torch.rand(2, 4, d) > 0.5).to(torch.float32)
    packed = pack_spikes_uint16(s)
    assert packed.shape[-1] == 2  # ceil(17/16) = 2 slots
    unpacked = unpack_spikes_uint16(packed, d, dtype=torch.float32)
    assert unpacked.shape[-1] == d
    assert torch.equal(unpacked, s)


def test_memory_savings_ratio():
    """Document the headline memory-savings number.

    For a (B, T, d) bf16 spike tensor:
        original_bytes = B * T * d * 2
        packed_bytes   = B * T * ceil(d/16) * 4   (int32 storage)
                       = B * T * d/16 * 4
                       = B * T * d / 4
    Ratio: 2 / (1/4) = 8x in our int32 storage.

    With future torch uint16 support:
        packed_bytes_native = B * T * ceil(d/16) * 2 = B * T * d / 8
        Ratio: 2 / (1/8) = 16x.

    This test asserts the int32-storage ratio (current path).
    """
    B, T, d = 4, 64, 1280
    s = (torch.rand(B, T, d) > 0.5).to(torch.bfloat16)
    packed = pack_spikes_uint16(s)
    original_bytes = s.element_size() * s.numel()  # bf16: 2 * 4*64*1280
    packed_bytes = packed.element_size() * packed.numel()  # int32: 4 * 4*64*80
    ratio = original_bytes / packed_bytes
    # 2*1280 / (4*80) = 2560 / 320 = 8.0  (int32 storage)
    assert ratio == pytest.approx(8.0, abs=0.01), (
        f"Expected 8x int32-storage savings, got {ratio:.2f}x"
    )
