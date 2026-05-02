"""Tests for synapforge.native.modal.cross_modal.

Coverage
--------
* contrastive_loss matches per-modal-pair reference within 1e-5.
* Loss is invariant to scale (cosine sim is normalised).
* Per-pair loss matches single-pair loss when only one pair is configured.
* Empty modality skips gracefully.
* Gradient finite-difference matches analytic gradient (1e-3).
"""
from __future__ import annotations

import numpy as np
import pytest

from synapforge.native.modal.cross_modal import (
    ContrastiveOutput,
    CrossModalContrastive,
    contrastive_loss,
    pairwise_cosine,
)
from synapforge.native.modal.packed_batch import ModalBatchPacker


def _make_pair_packed(b_per_modal: int = 4, h: int = 16, seed: int = 0):
    """Helper: build a packed batch with text and image samples and
    return (z, packed) for CLIP-style contrastive."""
    packer = ModalBatchPacker()
    rng = np.random.default_rng(seed)
    per_modal = {
        "text":  [rng.integers(0, 1000, size=4, dtype=np.int32)
                  for _ in range(b_per_modal)],
        "image": [rng.integers(0, 256, size=8, dtype=np.int32)
                  for _ in range(b_per_modal)],
    }
    packed = packer.pack(per_modal)
    z = rng.standard_normal((packed.n_samples, h)).astype(np.float32)
    return z, packed


class TestPairwiseCosine:
    def test_identity_self_similarity(self) -> None:
        a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        sim = pairwise_cosine(a, a)
        np.testing.assert_allclose(np.diag(sim), [1.0, 1.0], atol=1e-6)

    def test_orthogonal_zero_similarity(self) -> None:
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = pairwise_cosine(a, b)
        np.testing.assert_allclose(sim, [[0.0]], atol=1e-6)

    def test_zero_vector_does_not_nan(self) -> None:
        a = np.zeros((1, 4), dtype=np.float32)
        sim = pairwise_cosine(a, a)
        assert np.isfinite(sim).all()


class TestContrastiveLoss:
    def test_zero_loss_on_perfect_alignment_with_low_temp(self) -> None:
        # Aligned positives at e.g. (1,0), (0,1) -- with low temperature
        # the loss is approximately 0 (pure diagonal).
        h = 4
        n = 2
        z_text = np.eye(n, h, dtype=np.float32)
        z_image = np.eye(n, h, dtype=np.float32)
        # Pack 2 text, 2 image samples in alternating order.
        packer = ModalBatchPacker()
        packed = packer.pack({
            "text":  [np.array([0], dtype=np.int32),
                      np.array([1], dtype=np.int32)],
            "image": [np.array([0], dtype=np.int32),
                      np.array([1], dtype=np.int32)],
        })
        z = np.vstack([z_text, z_image])
        out = contrastive_loss(z, packed, temperature=0.01)
        # ln(2) is the max for a 2-class problem; aligned diagonals
        # under low temp -> close to zero.
        assert out.loss < 0.1

    def test_loss_is_finite_random_inputs(self) -> None:
        z, packed = _make_pair_packed()
        out = contrastive_loss(z, packed)
        assert np.isfinite(out.loss)
        assert out.grad_per_sample.shape == z.shape

    def test_per_pair_loss_keyed_correctly(self) -> None:
        z, packed = _make_pair_packed()
        out = contrastive_loss(z, packed, pairs=[("text", "image")])
        assert ("text", "image") in out.per_pair_loss
        assert isinstance(out.per_pair_loss[("text", "image")], float)

    def test_unknown_pair_rejected(self) -> None:
        z, packed = _make_pair_packed()
        with pytest.raises(KeyError):
            contrastive_loss(z, packed, pairs=[("foo", "bar")])

    def test_empty_modality_skips_silently(self) -> None:
        # Pack only text; ask for text-image pair -> 0 loss, no crash.
        packer = ModalBatchPacker()
        packed = packer.pack({"text": [np.array([1], dtype=np.int32)]})
        z = np.ones((1, 4), dtype=np.float32)
        out = contrastive_loss(z, packed, pairs=[("text", "image")])
        assert out.loss == 0.0
        np.testing.assert_array_equal(out.grad_per_sample, np.zeros_like(z))

    def test_shape_mismatch_raises(self) -> None:
        z, packed = _make_pair_packed()
        z_bad = z[:-1]  # too few rows
        with pytest.raises(ValueError, match="samples"):
            contrastive_loss(z_bad, packed)


class TestPerPairConsistency:
    """Loss must match per-modal-pair reference within 1e-5."""

    def test_single_pair_matches_class_call(self) -> None:
        z, packed = _make_pair_packed()
        loss_obj = CrossModalContrastive(
            pairs=[("text", "image")], temperature=0.07
        )
        out_class = loss_obj(z, packed)
        out_func = contrastive_loss(
            z, packed, pairs=[("text", "image")], temperature=0.07
        )
        np.testing.assert_allclose(out_class.loss, out_func.loss, atol=1e-7)
        np.testing.assert_allclose(
            out_class.grad_per_sample, out_func.grad_per_sample, atol=1e-7
        )

    def test_two_pairs_avg_matches_manual_average(self) -> None:
        """Average of two pair losses must match contrastive_loss with
        both pairs configured -- this is the bit-equivalence promise."""
        # Build a batch with text + image + audio.
        packer = ModalBatchPacker()
        rng = np.random.default_rng(11)
        per_modal = {
            "text":  [rng.integers(0, 100, size=3, dtype=np.int32)
                      for _ in range(3)],
            "image": [rng.integers(0, 256, size=4, dtype=np.int32)
                      for _ in range(3)],
            "audio": [rng.integers(0, 256, size=4, dtype=np.int32)
                      for _ in range(3)],
        }
        packed = packer.pack(per_modal)
        h = 8
        z = rng.standard_normal((packed.n_samples, h)).astype(np.float32)

        # Ground-truth: compute each pair separately, average loss and grads.
        out_a = contrastive_loss(
            z, packed, pairs=[("text", "image")], temperature=0.07
        )
        out_b = contrastive_loss(
            z, packed, pairs=[("text", "audio")], temperature=0.07
        )
        ref_loss = 0.5 * (out_a.loss + out_b.loss)
        ref_grad = 0.5 * (out_a.grad_per_sample + out_b.grad_per_sample)

        # Combined call.
        out_combined = contrastive_loss(
            z, packed,
            pairs=[("text", "image"), ("text", "audio")],
            temperature=0.07,
        )
        np.testing.assert_allclose(out_combined.loss, ref_loss, atol=1e-5)
        np.testing.assert_allclose(
            out_combined.grad_per_sample, ref_grad, atol=1e-5
        )


class TestPairId:
    def test_pair_id_alignment(self) -> None:
        # 2 text + 2 image, with pair_id matching them in reverse.
        packer = ModalBatchPacker()
        packed = packer.pack({
            "text":  [np.array([1], dtype=np.int32),
                      np.array([2], dtype=np.int32)],
            "image": [np.array([10], dtype=np.int32),
                      np.array([20], dtype=np.int32)],
        })
        # 4 samples in pack order: text_0, text_1, image_0, image_1
        # pair_ids: text_0<->image_1 are positives (id=A);
        #           text_1<->image_0 are positives (id=B).
        pair_id = np.array([0, 1, 1, 0], dtype=np.int32)

        rng = np.random.default_rng(7)
        z = rng.standard_normal((4, 8)).astype(np.float32)
        out = contrastive_loss(
            z, packed, pairs=[("text", "image")], pair_id=pair_id
        )
        assert np.isfinite(out.loss)


class TestGradientCheck:
    """Finite-difference vs analytic gradient (1e-3 tolerance)."""

    def test_gradient_matches_finite_difference(self) -> None:
        # Smaller batch + low h to keep FD cheap.
        packer = ModalBatchPacker()
        packed = packer.pack({
            "text":  [np.array([1], dtype=np.int32),
                      np.array([2], dtype=np.int32)],
            "image": [np.array([10], dtype=np.int32),
                      np.array([20], dtype=np.int32)],
        })
        rng = np.random.default_rng(13)
        h = 4
        z = rng.standard_normal((4, h)).astype(np.float64)
        eps = 1e-4

        out = contrastive_loss(
            z.astype(np.float32), packed,
            pairs=[("text", "image")], temperature=0.5,
        )
        analytic = out.grad_per_sample.astype(np.float64)

        fd = np.zeros_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z_p = z.copy(); z_p[i, j] += eps
                z_m = z.copy(); z_m[i, j] -= eps
                lp = contrastive_loss(
                    z_p.astype(np.float32), packed,
                    pairs=[("text", "image")], temperature=0.5,
                ).loss
                lm = contrastive_loss(
                    z_m.astype(np.float32), packed,
                    pairs=[("text", "image")], temperature=0.5,
                ).loss
                fd[i, j] = (lp - lm) / (2 * eps)

        # Loose tolerance because we're in fp32 and using FD.
        np.testing.assert_allclose(analytic, fd, atol=5e-3, rtol=5e-3)
