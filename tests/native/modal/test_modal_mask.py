"""Tests for synapforge.native.modal.modal_mask.

Coverage
--------
* reset_flag is True at every sample boundary, False elsewhere.
* token_modal_id matches the per-sample modal_id over each sample's slice.
* token_sample_id is monotonic and matches offsets.
* pairwise_modal_mask sample- and token-granularity behave correctly.
* The mask blocks cross-sample bleed (the only thing the kernel cares about).
"""
from __future__ import annotations

import numpy as np
import pytest

from synapforge.native.modal.modal_mask import (
    ModalMaskBuilder,
    apply_modal_gate,
    build_modal_mask,
)
from synapforge.native.modal.packed_batch import (
    MODAL_REGISTRY,
    ModalBatchPacker,
)


class TestResetFlag:
    def test_reset_at_sample_boundaries(self) -> None:
        packer = ModalBatchPacker()
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5], dtype=np.int32)
        c = np.array([6], dtype=np.int32)
        packed = packer.pack({"text": [a, b, c]})
        bundle = build_modal_mask(packed)
        # Lengths 3, 2, 1 -> reset at indices 0, 3, 5.
        expected = np.array([True, False, False, True, False, True])
        np.testing.assert_array_equal(bundle.reset_flag, expected)

    def test_reset_token_count_equals_sample_count(self) -> None:
        packer = ModalBatchPacker()
        seqs = [np.arange(i + 1, dtype=np.int32) for i in range(5)]
        packed = packer.pack({"text": seqs})
        bundle = build_modal_mask(packed)
        assert bundle.reset_flag.sum() == 5

    def test_empty_batch_yields_empty_bundle(self) -> None:
        packer = ModalBatchPacker()
        packed = packer.pack({})
        bundle = build_modal_mask(packed)
        assert bundle.n_tokens == 0


class TestTokenModalId:
    def test_modal_id_constant_within_sample(self) -> None:
        packer = ModalBatchPacker()
        text = np.array([1, 2, 3], dtype=np.int32)
        image = np.arange(4, dtype=np.int32)
        packed = packer.pack({"text": [text], "image": [image]})
        bundle = build_modal_mask(packed)
        # First 3 tokens are text (modal_id=0), next 4 are image (modal_id=1).
        np.testing.assert_array_equal(bundle.token_modal_id[:3], [0, 0, 0])
        np.testing.assert_array_equal(bundle.token_modal_id[3:], [1, 1, 1, 1])

    def test_modal_id_matches_registry(self) -> None:
        packer = ModalBatchPacker()
        per_modal = {
            name: [np.zeros(2, dtype=np.int32)]
            for name in MODAL_REGISTRY
        }
        packed = packer.pack(per_modal)
        bundle = build_modal_mask(packed)
        # Each sample is length 2 and they appear in registry order.
        offset = 0
        for name, spec in MODAL_REGISTRY.items():
            seg = bundle.token_modal_id[offset:offset + 2]
            np.testing.assert_array_equal(seg, [spec.modal_id, spec.modal_id])
            offset += 2


class TestTokenSampleId:
    def test_token_sample_id_blocks_correctly(self) -> None:
        packer = ModalBatchPacker()
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5], dtype=np.int32)
        c = np.array([6, 7, 8, 9], dtype=np.int32)
        packed = packer.pack({"text": [a, b, c]})
        bundle = build_modal_mask(packed)
        expected = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        np.testing.assert_array_equal(bundle.token_sample_id, expected)


class TestPairwiseModalMask:
    def test_sample_granularity_diagonal(self) -> None:
        packer = ModalBatchPacker()
        builder = ModalMaskBuilder()
        packed = packer.pack({
            "text": [np.array([1, 2], dtype=np.int32),
                     np.array([3], dtype=np.int32)],
            "image": [np.array([10, 11], dtype=np.int32)],
        })
        m = builder.pairwise_modal_mask(packed, granularity="sample")
        assert m.shape == (3, 3)
        # text-text pair (rows 0,1) -> True
        assert m[0, 1]
        # text-image pair -> False
        assert not m[0, 2]
        assert not m[1, 2]
        # diagonal (same modal as itself) -> True
        for i in range(3):
            assert m[i, i]

    def test_token_granularity_blocks_per_sample(self) -> None:
        packer = ModalBatchPacker()
        builder = ModalMaskBuilder()
        packed = packer.pack({
            "text": [np.array([1, 2, 3], dtype=np.int32),
                     np.array([4, 5], dtype=np.int32)],
        })
        m = builder.pairwise_modal_mask(packed, granularity="token")
        # 5 total tokens; sample 0 = [0,1,2], sample 1 = [3,4]
        assert m.shape == (5, 5)
        # Token 0 with token 2 -> same sample -> True
        assert m[0, 2]
        # Token 0 with token 3 -> different samples -> False
        assert not m[0, 3]
        # Token 4 with token 3 -> same sample -> True
        assert m[4, 3]

    def test_unknown_granularity(self) -> None:
        packer = ModalBatchPacker()
        builder = ModalMaskBuilder()
        packed = packer.pack({"text": [np.array([1], dtype=np.int32)]})
        with pytest.raises(ValueError, match="granularity"):
            builder.pairwise_modal_mask(packed, granularity="bogus")


class TestApplyModalGate:
    def test_shape_validation(self) -> None:
        h_in = np.zeros((4, 8), dtype=np.float32)
        h_new = np.zeros((4, 8), dtype=np.float32)
        reset = np.array([True, False, True, False])
        out = apply_modal_gate(h_in, h_new, reset)
        assert out.shape == h_new.shape

    def test_shape_mismatch_rejected(self) -> None:
        h_in = np.zeros((4, 8), dtype=np.float32)
        h_new = np.zeros((4, 16), dtype=np.float32)
        reset = np.array([True, False, True, False])
        with pytest.raises(ValueError):
            apply_modal_gate(h_in, h_new, reset)


class TestModalGateBlocksCrossSampleBleed:
    """Integration: the reset flag matches sample boundaries in the
    *packed* layout, so a CfC kernel that consumes (token_modal_id,
    reset_flag) cannot see state from the previous sample."""

    def test_first_token_of_every_sample_resets(self) -> None:
        packer = ModalBatchPacker()
        # Mix text with image -- different vocabs -- highest-risk bleed.
        text = np.array([1, 2, 3], dtype=np.int32)
        image = np.array([10, 20, 30, 40], dtype=np.int32)
        audio = np.array([5], dtype=np.int32)
        packed = packer.pack({
            "text": [text],
            "image": [image],
            "audio": [audio],
        })
        bundle = build_modal_mask(packed)
        # Reset at every sample boundary AND each is the start of a
        # different modality.
        reset_positions = np.where(bundle.reset_flag)[0]
        np.testing.assert_array_equal(
            reset_positions, packed.offsets[:-1]
        )
        # The modal id changes at each reset (different modalities).
        for pos in reset_positions[1:]:
            assert bundle.token_modal_id[pos] != bundle.token_modal_id[pos - 1]
