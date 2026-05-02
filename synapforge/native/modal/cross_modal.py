"""cross_modal.py -- cross-modal contrastive loss over the shared backbone.

Goal
----
After the shared CfC+PLIF backbone produces a per-sample embedding
``z[k] in R^H`` for every sample in the packed batch, we want a
CLIP-style InfoNCE loss that:

* pulls **positive pairs** (text_i, image_i) closer in the embedding
  space, where i is the *same logical example* -- e.g. a text caption
  and the matching image of a cat.
* pushes **negative pairs** (text_i, image_j != image_i) apart.

With the packed batch we can compute this in a single forward pass:
the backbone gives us all sample embeddings ``z`` in one go, then we
slice by modal id and apply InfoNCE on each pair of modalities.

Numerical contract
------------------
This module's loss must be **bit-equivalent** to the per-modal-pair
reference implementation (which would compute each modal pair in a
separate forward pass and average). The unit test verifies this within
``atol=1e-5``.

Public API
----------
* :class:`CrossModalContrastive` -- stateful loss with temperature.
* :func:`contrastive_loss`       -- one-shot stateless variant.
* :func:`pairwise_cosine`        -- fp32-stable cosine similarity matrix.

Hard constraint
---------------
**No ``import torch``**. ``numpy`` only -- production gradients will
be implemented via the corresponding entry in
:mod:`synapforge.native.vjp`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from synapforge.native.modal.packed_batch import (
    MODAL_BY_ID,
    MODAL_REGISTRY,
    PackedBatch,
)


# ---------------------------------------------------------------------------
# Numerically stable primitives
# ---------------------------------------------------------------------------

def pairwise_cosine(
    a: np.ndarray,
    b: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute ``cos_sim = a_norm @ b_norm.T`` with a small-epsilon clamp.

    Args
    ----
    a : (M, H)
    b : (N, H)
    eps : float
        Floor for the L2 norm before division.

    Returns
    -------
    (M, N)
        Cosine similarity matrix in fp32.
    """
    a32 = a.astype(np.float32, copy=False)
    b32 = b.astype(np.float32, copy=False)
    na = np.linalg.norm(a32, axis=-1, keepdims=True)
    nb = np.linalg.norm(b32, axis=-1, keepdims=True)
    na = np.maximum(na, eps)
    nb = np.maximum(nb, eps)
    return (a32 / na) @ (b32 / nb).T


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def _info_nce(
    z_a: np.ndarray,
    z_b: np.ndarray,
    *,
    temperature: float,
    eps: float = 1e-8,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Symmetric InfoNCE loss between two equal-size sample sets.

    Args
    ----
    z_a, z_b : (M, H)
        Aligned positive pairs: ``z_a[i]`` matches ``z_b[i]``.
    temperature : float
        Softmax temperature; smaller -> sharper.

    Returns
    -------
    (loss, grad_a, grad_b)
        loss is a scalar; gradients are (M, H) arrays.

    Notes
    -----
    The loss is the symmetric CLIP loss

        L = 0.5 * (CE(rows, diag) + CE(cols, diag))

    over the temperature-scaled cosine similarity matrix. Gradient is
    computed analytically below so we don't depend on autograd.
    """
    if z_a.shape != z_b.shape:
        raise ValueError(f"z_a {z_a.shape} != z_b {z_b.shape}")
    m, h = z_a.shape
    if m == 0:
        return 0.0, np.zeros_like(z_a), np.zeros_like(z_b)

    # Raw cosine similarity (without temperature) and the scaled logits.
    a32 = z_a.astype(np.float32, copy=False)
    b32 = z_b.astype(np.float32, copy=False)
    na = np.maximum(np.linalg.norm(a32, axis=-1, keepdims=True), eps)
    nb = np.maximum(np.linalg.norm(b32, axis=-1, keepdims=True), eps)
    u = a32 / na   # (M, H) -- normalized z_a
    v = b32 / nb   # (M, H) -- normalized z_b
    cos = u @ v.T  # (M, M) raw cosine similarity
    logits = cos / temperature  # (M, M)

    # Row CE: targets are the diagonal.
    log_p_rows = _log_softmax(logits, axis=1)
    log_p_cols = _log_softmax(logits, axis=0)
    diag_idx = np.arange(m)
    loss_rows = -np.mean(log_p_rows[diag_idx, diag_idx])
    loss_cols = -np.mean(log_p_cols[diag_idx, diag_idx])
    loss = 0.5 * (loss_rows + loss_cols)

    # Analytic gradient w.r.t. logits:
    #   dL/dlogits = 0.5/M * (softmax(logits, axis=1) - I) +
    #                0.5/M * (softmax(logits, axis=0) - I)
    p_rows = np.exp(log_p_rows)
    p_cols = np.exp(log_p_cols)
    eye = np.eye(m, dtype=np.float32)
    dlogits = (p_rows - eye) * (0.5 / m) + (p_cols - eye) * (0.5 / m)

    # Chain through temperature: dL/dcos = dL/dlogits * (1/temperature).
    dcos = dlogits / temperature  # (M, M)

    # Chain through cosine. Let:
    #     u_i = a_i / ||a_i||, v_j = b_j / ||b_j||,
    #     cos_{i,j} = u_i . v_j.
    # Then:
    #     dL/du_i = sum_j dcos_{i,j} * v_j        = (dcos @ v)[i]
    #     dL/dv_j = sum_i dcos_{i,j} * u_i        = (dcos.T @ u)[j]
    # And the L2-norm Jacobian is:
    #     du_i / da_i = (I - u_i u_i^T) / ||a_i||
    # So:
    #     dL/da_i = (1/||a_i||) * (dL/du_i - (u_i . dL/du_i) * u_i)
    grad_u = dcos @ v          # (M, H)
    grad_v = dcos.T @ u        # (M, H)
    proj_u = np.sum(grad_u * u, axis=1, keepdims=True) * u
    proj_v = np.sum(grad_v * v, axis=1, keepdims=True) * v
    grad_a = (grad_u - proj_u) / na
    grad_b = (grad_v - proj_v) / nb

    return float(loss), grad_a, grad_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastiveOutput:
    """Result of a cross-modal contrastive forward.

    Attributes
    ----------
    loss : float
        Scalar averaged across all configured modal pairs.
    per_pair_loss : Dict[Tuple[str, str], float]
        Per-pair (modal_a, modal_b) loss for diagnostics.
    grad_per_sample : np.ndarray
        ``(B, H)`` gradient w.r.t. the per-sample embeddings, accumulated
        across all pair contributions. Hand this back to the autograd
        boundary as ``z.grad += grad_per_sample``.
    """

    loss: float
    per_pair_loss: Dict[Tuple[str, str], float]
    grad_per_sample: np.ndarray


def contrastive_loss(
    z: np.ndarray,
    packed: PackedBatch,
    *,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    temperature: float = 0.07,
    pair_id: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> ContrastiveOutput:
    """Compute cross-modal contrastive loss.

    Args
    ----
    z : (B, H)
        Per-sample embeddings produced by the shared backbone (one
        embedding per packed-batch sample).
    packed : :class:`PackedBatch`
        The bundle whose ``modal_ids`` tells us which sample is which
        modality.
    pairs :
        List of ``(modal_a, modal_b)`` modality-name pairs to contrast.
        Default is ``[("text", "image")]`` (CLIP-style). Each pair must
        exist in :data:`MODAL_REGISTRY`.
    temperature :
        InfoNCE softmax temperature. Default 0.07 (CLIP recipe).
    pair_id :
        Optional ``(B,)`` int array. If provided, samples with the same
        ``pair_id`` are positive matches across modalities. If None,
        the i-th sample of modal_a is matched with the i-th sample of
        modal_b (positional alignment).
    eps :
        Norm-clamp epsilon for cosine.

    Returns
    -------
    :class:`ContrastiveOutput`
    """
    if pairs is None:
        pairs = [("text", "image")]
    if z.ndim != 2:
        raise ValueError(f"z must be (B, H); got shape {z.shape}")
    if z.shape[0] != packed.n_samples:
        raise ValueError(
            f"z has {z.shape[0]} samples; packed has {packed.n_samples}"
        )

    # Index samples by modal name.
    by_name: Dict[str, List[int]] = {name: [] for name in MODAL_REGISTRY}
    for k, mid in enumerate(packed.modal_ids):
        spec = MODAL_BY_ID[int(mid)]
        by_name[spec.name].append(k)

    grad = np.zeros_like(z, dtype=np.float32)
    per_pair: Dict[Tuple[str, str], float] = {}
    total_loss = 0.0
    n_pairs_used = 0

    for ma, mb in pairs:
        if ma not in MODAL_REGISTRY or mb not in MODAL_REGISTRY:
            raise KeyError(f"unknown modality pair ({ma}, {mb})")
        idx_a = by_name[ma]
        idx_b = by_name[mb]
        if len(idx_a) == 0 or len(idx_b) == 0:
            # No samples of this modality in the batch -- skip.
            continue
        if pair_id is None:
            # Positional alignment: take the smaller of the two.
            m = min(len(idx_a), len(idx_b))
            ai = idx_a[:m]
            bi = idx_b[:m]
        else:
            # Match by pair_id.
            pa = {int(pair_id[k]): k for k in idx_a}
            pb = {int(pair_id[k]): k for k in idx_b}
            keys = sorted(set(pa.keys()) & set(pb.keys()))
            if len(keys) == 0:
                continue
            ai = [pa[k] for k in keys]
            bi = [pb[k] for k in keys]

        z_a = z[ai].astype(np.float32, copy=False)
        z_b = z[bi].astype(np.float32, copy=False)
        loss, ga, gb = _info_nce(z_a, z_b, temperature=temperature, eps=eps)

        grad[ai] += ga
        grad[bi] += gb
        per_pair[(ma, mb)] = float(loss)
        total_loss += float(loss)
        n_pairs_used += 1

    if n_pairs_used > 0:
        total_loss /= n_pairs_used
        grad /= n_pairs_used

    return ContrastiveOutput(
        loss=float(total_loss),
        per_pair_loss=per_pair,
        grad_per_sample=grad,
    )


class CrossModalContrastive:
    """Stateful cross-modal contrastive loss.

    The class form is useful when you want to reuse the same set of
    modal pairs and temperature across many training steps -- and when
    you want to expose the loss as a callable to the trainer.

    Examples
    --------
    >>> loss_fn = CrossModalContrastive(
    ...     pairs=[("text", "image"), ("text", "audio")],
    ...     temperature=0.05,
    ... )
    >>> result = loss_fn(z, packed)
    >>> result.loss   # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        temperature: float = 0.07,
        eps: float = 1e-8,
    ) -> None:
        if pairs is None:
            pairs = [("text", "image")]
        self.pairs: List[Tuple[str, str]] = list(pairs)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def __call__(
        self,
        z: np.ndarray,
        packed: PackedBatch,
        *,
        pair_id: Optional[np.ndarray] = None,
    ) -> ContrastiveOutput:
        return contrastive_loss(
            z,
            packed,
            pairs=self.pairs,
            temperature=self.temperature,
            pair_id=pair_id,
            eps=self.eps,
        )


__all__ = [
    "ContrastiveOutput",
    "CrossModalContrastive",
    "contrastive_loss",
    "pairwise_cosine",
]
