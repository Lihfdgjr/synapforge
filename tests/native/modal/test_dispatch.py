"""Tests for synapforge.native.modal.dispatch."""
from __future__ import annotations

import numpy as np
import pytest

from synapforge.native.modal.dispatch import (
    BYTE_VOCAB,
    ModalDispatchEmbed,
    QWEN_VOCAB,
)
from synapforge.native.modal.packed_batch import (
    MODAL_REGISTRY,
    ModalBatchPacker,
)


class TestModalDispatchEmbedConstruction:
    def test_default_table_sizes(self) -> None:
        embed = ModalDispatchEmbed(hidden=8, seed=0)
        assert embed.qwen_table.shape == (QWEN_VOCAB, 8)
        assert embed.byte_table.shape == (BYTE_VOCAB, 8)

    def test_table_byte_count(self) -> None:
        embed = ModalDispatchEmbed(hidden=4, seed=0)
        # qwen 151643 * 4 * 4 bytes (fp32)
        assert embed.qwen_table_bytes == 151643 * 4 * 4
        # byte 256 * 4 * 4 bytes
        assert embed.byte_table_bytes == 256 * 4 * 4

    def test_hidden_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ModalDispatchEmbed(hidden=0)


class TestLookup:
    def test_byte_modality_uses_byte_table(self) -> None:
        embed = ModalDispatchEmbed(hidden=4, seed=42)
        token_ids = np.array([3, 17, 200], dtype=np.int32)
        modal_ids = np.array([1, 1, 1], dtype=np.int32)  # image -> byte
        h = embed.lookup(token_ids, modal_ids)
        np.testing.assert_array_equal(h, embed.byte_table[[3, 17, 200]])

    def test_text_modality_uses_qwen_table(self) -> None:
        embed = ModalDispatchEmbed(hidden=4, seed=42)
        token_ids = np.array([0, 5, 150000], dtype=np.int32)
        modal_ids = np.array([0, 0, 0], dtype=np.int32)  # text -> qwen
        h = embed.lookup(token_ids, modal_ids)
        np.testing.assert_array_equal(h, embed.qwen_table[[0, 5, 150000]])

    def test_mixed_modal_dispatch(self) -> None:
        embed = ModalDispatchEmbed(hidden=4, seed=42)
        token_ids = np.array([5, 17, 7, 99], dtype=np.int32)
        modal_ids = np.array([0, 1, 0, 1], dtype=np.int32)  # text/image
        h = embed.lookup(token_ids, modal_ids)
        # Row 0 = qwen[5], row 1 = byte[17], row 2 = qwen[7], row 3 = byte[99].
        np.testing.assert_array_equal(h[0], embed.qwen_table[5])
        np.testing.assert_array_equal(h[1], embed.byte_table[17])
        np.testing.assert_array_equal(h[2], embed.qwen_table[7])
        np.testing.assert_array_equal(h[3], embed.byte_table[99])

    def test_byte_oor_rejected(self) -> None:
        embed = ModalDispatchEmbed(hidden=4)
        with pytest.raises(ValueError, match="byte token id out of range"):
            embed.lookup(
                np.array([300], dtype=np.int32),
                np.array([1], dtype=np.int32),  # image -> byte vocab=256
            )

    def test_qwen_oor_rejected(self) -> None:
        embed = ModalDispatchEmbed(hidden=4)
        with pytest.raises(ValueError, match="qwen token id out of range"):
            embed.lookup(
                np.array([200000], dtype=np.int32),
                np.array([0], dtype=np.int32),  # text -> qwen vocab
            )

    def test_unknown_modal_id_rejected(self) -> None:
        embed = ModalDispatchEmbed(hidden=4)
        with pytest.raises(KeyError):
            embed.lookup(
                np.array([1], dtype=np.int32),
                np.array([99], dtype=np.int32),
            )

    def test_shape_mismatch_rejected(self) -> None:
        embed = ModalDispatchEmbed(hidden=4)
        with pytest.raises(ValueError):
            embed.lookup(
                np.array([1, 2], dtype=np.int32),
                np.array([0], dtype=np.int32),
            )

    def test_empty_lookup(self) -> None:
        embed = ModalDispatchEmbed(hidden=4)
        out = embed.lookup(
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        )
        assert out.shape == (0, 4)


class TestLookupPacked:
    def test_packed_lookup_matches_explicit(self) -> None:
        packer = ModalBatchPacker()
        text = np.array([5, 10], dtype=np.int32)
        image = np.array([3, 17], dtype=np.int32)
        packed = packer.pack({"text": [text], "image": [image]})

        embed = ModalDispatchEmbed(hidden=4, seed=0)
        h_packed = embed.lookup_packed(packed)
        # Build the explicit call manually.
        token_modal = np.array([0, 0, 1, 1], dtype=np.int32)
        h_explicit = embed.lookup(packed.concat_tokens, token_modal)
        np.testing.assert_array_equal(h_packed, h_explicit)


class TestLogits:
    def test_logits_shape(self) -> None:
        embed = ModalDispatchEmbed(hidden=4, seed=0)
        hidden = np.ones((3, 4), dtype=np.float32)
        modal_ids = np.array([0, 1, 1], dtype=np.int32)
        logits, vocab_per_tok = embed.logits(hidden, modal_ids)
        assert logits.shape == (3, max(QWEN_VOCAB, BYTE_VOCAB))
        # text -> qwen vocab; image -> byte vocab.
        assert vocab_per_tok.tolist() == [QWEN_VOCAB, BYTE_VOCAB, BYTE_VOCAB]

    def test_logits_match_manual_matmul(self) -> None:
        embed = ModalDispatchEmbed(hidden=4, seed=0)
        rng = np.random.default_rng(0)
        hidden = rng.standard_normal((2, 4)).astype(np.float32)
        modal_ids = np.array([1, 1], dtype=np.int32)  # both image -> byte
        logits, vocab_per_tok = embed.logits(hidden, modal_ids)
        # Manual: h @ byte_table.T
        ref = hidden @ embed.byte_table.T
        np.testing.assert_allclose(
            logits[:, :BYTE_VOCAB], ref, atol=1e-5
        )

    def test_logits_shape_validation(self) -> None:
        embed = ModalDispatchEmbed(hidden=4)
        with pytest.raises(ValueError, match="hidden width"):
            embed.logits(
                np.zeros((2, 5), dtype=np.float32),  # wrong width
                np.array([0, 0], dtype=np.int32),
            )


class TestMemorySavingsTotalBytes:
    def test_dispatch_table_smaller_than_unified_table_for_byte_logits(self) -> None:
        """Per-modal dispatch yields tiny byte logits vs huge unified.
        Verify by comparing logit-tensor *width*."""
        embed = ModalDispatchEmbed(hidden=4)
        # 8K image tokens -> per-modal logit width = 256, not 151899.
        hidden = np.zeros((8192, 4), dtype=np.float32)
        modal_ids = np.full(8192, 1, dtype=np.int32)  # all image
        logits, vocab_per_tok = embed.logits(hidden, modal_ids)
        # The "live" logit slab is `vocab_per_tok` columns wide.
        live_bytes = int((vocab_per_tok[0]) * hidden.shape[0] * 4)
        unified_bytes = int((QWEN_VOCAB + BYTE_VOCAB) * hidden.shape[0] * 4)
        # > 100x smaller for byte logits.
        assert live_bytes * 100 < unified_bytes
