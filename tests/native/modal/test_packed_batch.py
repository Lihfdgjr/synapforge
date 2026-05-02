"""Tests for synapforge.native.modal.packed_batch.

Coverage
--------
* round-trip: pack(unpack(x)) returns identical sequences (bit-exact).
* offsets layout is monotonic and ends at total token count.
* unknown modality / oversized sequence / out-of-vocab are rejected.
* group_by_modal returns disjoint coverage of all samples.
* pad_per_modal_sizes / packed_size accounting helpers agree.
"""
from __future__ import annotations

import numpy as np
import pytest

from synapforge.native.modal.packed_batch import (
    MODAL_BY_ID,
    MODAL_REGISTRY,
    ModalBatchPacker,
    PackedBatch,
    pad_per_modal_sizes,
    packed_size,
)


class TestPackUnpackRoundTrip:
    """Pack-then-unpack must return bit-identical sequences."""

    def test_single_text_seq(self) -> None:
        packer = ModalBatchPacker()
        seq = np.array([1, 7, 42, 99], dtype=np.int32)
        packed = packer.pack({"text": [seq]})
        out = packer.unpack(packed)
        assert "text" in out
        assert len(out["text"]) == 1
        np.testing.assert_array_equal(out["text"][0], seq)

    def test_multi_modal_round_trip(self) -> None:
        packer = ModalBatchPacker()
        text_a = np.array([1, 2, 3, 4], dtype=np.int32)
        text_b = np.array([5, 6, 7], dtype=np.int32)
        image_a = np.arange(8, dtype=np.int32)
        audio_a = np.array([10, 20, 30, 40, 50], dtype=np.int32)

        per_modal = {
            "text": [text_a, text_b],
            "image": [image_a],
            "audio": [audio_a],
        }
        packed = packer.pack(per_modal)
        assert packed.n_samples == 4
        assert packed.n_tokens == 4 + 3 + 8 + 5

        out = packer.unpack(packed)
        np.testing.assert_array_equal(out["text"][0], text_a)
        np.testing.assert_array_equal(out["text"][1], text_b)
        np.testing.assert_array_equal(out["image"][0], image_a)
        np.testing.assert_array_equal(out["audio"][0], audio_a)

    def test_round_trip_all_nine_modalities(self) -> None:
        """All 9 modalities round-trip cleanly."""
        packer = ModalBatchPacker()
        rng = np.random.default_rng(42)
        per_modal = {}
        expected = {}
        for name, spec in MODAL_REGISTRY.items():
            length = min(spec.t_max, 32)
            arr = rng.integers(
                low=0, high=spec.vocab_size, size=length, dtype=np.int32
            )
            per_modal[name] = [arr]
            expected[name] = arr
        packed = packer.pack(per_modal)
        out = packer.unpack(packed)
        for name, arr in expected.items():
            assert name in out
            np.testing.assert_array_equal(out[name][0], arr)

    def test_empty_batch(self) -> None:
        packer = ModalBatchPacker()
        packed = packer.pack({})
        assert packed.n_samples == 0
        assert packed.n_tokens == 0
        assert packed.offsets.tolist() == [0]
        out = packer.unpack(packed)
        assert out == {}


class TestOffsetsLayout:
    def test_offsets_monotonic_and_terminal(self) -> None:
        packer = ModalBatchPacker()
        seqs = {
            "text": [np.array([1, 2, 3], dtype=np.int32),
                     np.array([4, 5], dtype=np.int32)],
            "image": [np.array([10, 11, 12, 13], dtype=np.int32)],
        }
        packed = packer.pack(seqs)
        assert packed.offsets[0] == 0
        assert packed.offsets[-1] == packed.n_tokens
        diffs = np.diff(packed.offsets)
        assert np.all(diffs >= 0)
        np.testing.assert_array_equal(diffs, packed.seq_lens)

    def test_modal_ids_correct(self) -> None:
        packer = ModalBatchPacker()
        packed = packer.pack({
            "text":  [np.array([1, 2], dtype=np.int32)],
            "image": [np.array([3, 4], dtype=np.int32)],
        })
        # text -> modal_id=0, image -> modal_id=1.
        assert packed.modal_ids.tolist() == [0, 1]

    def test_slice_returns_correct_segment(self) -> None:
        packer = ModalBatchPacker()
        a = np.array([100, 101, 102], dtype=np.int32)
        b = np.array([200, 201], dtype=np.int32)
        packed = packer.pack({"text": [a, b]})
        np.testing.assert_array_equal(packed.slice(0), a)
        np.testing.assert_array_equal(packed.slice(1), b)

    def test_slice_out_of_range(self) -> None:
        packer = ModalBatchPacker()
        packed = packer.pack({"text": [np.array([1], dtype=np.int32)]})
        with pytest.raises(IndexError):
            packed.slice(99)


class TestValidation:
    def test_unknown_modality_rejected(self) -> None:
        packer = ModalBatchPacker()
        with pytest.raises(KeyError):
            packer.pack({"NOT_A_MODAL": [np.array([1], dtype=np.int32)]})

    def test_oversized_sequence_rejected(self) -> None:
        packer = ModalBatchPacker()
        # text T_max=256
        too_long = np.zeros(257, dtype=np.int32)
        with pytest.raises(ValueError, match="exceeds T_max"):
            packer.pack({"text": [too_long]})

    def test_out_of_vocab_rejected(self) -> None:
        packer = ModalBatchPacker()
        # byte vocab is 256, so 256 itself is invalid.
        bad = np.array([255, 256], dtype=np.int32)
        with pytest.raises(ValueError, match="out of vocab"):
            packer.pack({"image": [bad]})

    def test_negative_token_rejected(self) -> None:
        packer = ModalBatchPacker()
        bad = np.array([-1, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="out of vocab"):
            packer.pack({"text": [bad]})

    def test_2d_sequence_rejected(self) -> None:
        packer = ModalBatchPacker()
        bad = np.zeros((4, 4), dtype=np.int32)
        with pytest.raises(ValueError, match="must be 1-D"):
            packer.pack({"text": [bad]})


class TestGroupByModal:
    def test_groups_cover_all_samples_disjointly(self) -> None:
        packer = ModalBatchPacker()
        packed = packer.pack({
            "text": [np.array([1, 2], dtype=np.int32),
                     np.array([3], dtype=np.int32)],
            "image": [np.array([10, 11, 12], dtype=np.int32)],
            "audio": [np.array([20], dtype=np.int32)],
        })
        groups = packer.group_by_modal(packed)
        # text id=0, image id=1, audio id=2
        assert set(groups.keys()) == {0, 1, 2}
        # All sample indices [0, 1, 2, 3] must appear exactly once.
        all_idx = []
        for idxs, lens in groups.values():
            all_idx.extend(idxs.tolist())
        assert sorted(all_idx) == [0, 1, 2, 3]

    def test_lens_match_seq_lens(self) -> None:
        packer = ModalBatchPacker()
        packed = packer.pack({
            "text": [np.array([1, 2, 3], dtype=np.int32)],
            "image": [np.array([10, 11], dtype=np.int32)],
        })
        groups = packer.group_by_modal(packed)
        # sample 0 (text) is len 3.
        text_idx, text_lens = groups[0]
        assert text_idx.tolist() == [0]
        assert text_lens.tolist() == [3]
        image_idx, image_lens = groups[1]
        assert image_idx.tolist() == [1]
        assert image_lens.tolist() == [2]


class TestAccountingHelpers:
    def test_packed_size(self) -> None:
        assert packed_size([4, 7, 10]) == 21
        assert packed_size([]) == 0

    def test_pad_per_modal_sizes_uses_t_max(self) -> None:
        # text T_max=256, image T_max=1024.
        # Padded baseline: 1 text * 256 + 1 image * 1024 = 1280.
        seq_lens = [10, 50]
        modal_ids = [
            MODAL_REGISTRY["text"].modal_id,
            MODAL_REGISTRY["image"].modal_id,
        ]
        assert pad_per_modal_sizes(seq_lens, modal_ids) == 256 + 1024

    def test_packed_vs_padded_savings_realistic_mix(self) -> None:
        """A representative bs=8-per-modal mix should save 50%+."""
        # 8 samples each of text/image/audio, with seqs at ~30% of T_max.
        rng = np.random.default_rng(0)
        seq_lens: list[int] = []
        modal_ids: list[int] = []
        for name in ("text", "image", "audio"):
            spec = MODAL_REGISTRY[name]
            for _ in range(8):
                ln = int(rng.integers(low=spec.t_max // 4,
                                      high=spec.t_max // 2 + 1))
                seq_lens.append(ln)
                modal_ids.append(spec.modal_id)
        packed_n = packed_size(seq_lens)
        padded_n = pad_per_modal_sizes(seq_lens, modal_ids)
        assert packed_n < padded_n
        # We expect 30-50% of padded for this distribution.
        ratio = packed_n / padded_n
        assert ratio < 0.6


class TestAttnMask:
    def test_attn_mask_built_when_requested(self) -> None:
        packer = ModalBatchPacker()
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4, 5], dtype=np.int32)
        packed = packer.pack({"text": [a, b]}, build_attn_mask=True)
        assert packed.attn_mask_per_modal is not None
        # Shape (B=2, max_L=3)
        assert packed.attn_mask_per_modal.shape == (2, 3)
        # Sample 0 has length 2, sample 1 has length 3.
        np.testing.assert_array_equal(
            packed.attn_mask_per_modal[0], [True, True, False]
        )
        np.testing.assert_array_equal(
            packed.attn_mask_per_modal[1], [True, True, True]
        )

    def test_attn_mask_off_by_default(self) -> None:
        packer = ModalBatchPacker()
        packed = packer.pack({"text": [np.array([1, 2], dtype=np.int32)]})
        assert packed.attn_mask_per_modal is None
