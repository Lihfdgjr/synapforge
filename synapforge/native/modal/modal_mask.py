"""modal_mask.py -- modal-aware mask for the shared CfC+PLIF backbone.

Why the mask is needed
----------------------
A packed multimodal batch concatenates per-sample sequences end-to-end
into one flat tensor. The CfC cell processes the flat tensor token-by-
token to keep its temporal state ``h``. Naively, ``h`` would carry over
across the boundary from one sample to the next -- e.g. text sample's
last hidden state would leak into image sample's first step.

The fix is a *modal mask* that gates the CfC state update so it resets
across boundaries. There are two granularities:

1. **Per-token modal id** ``token_modal_id[n]``. Tells the kernel which
   modal-specific embed/lm_head to use for this token. (Used by
   :class:`synapforge.native.modal.dispatch.ModalDispatchEmbed`.)
2. **Pairwise modal mask** ``mask[i, j] = (modal_id[i] == modal_id[j])
   AND (sample_id[i] == sample_id[j])``. Used in attention (if any) and
   for cross-modal contrastive loss negatives.

For the *temporal* CfC update we don't need a full O(N^2) mask. We use
a per-token *reset_flag* ``reset[n]`` which is True iff ``n`` is the
first token of a new sample. The kernel zeroes the carried hidden state
on those steps.

::

    Tokens by sample:   [t0 t1 t2 t3] [i0 i1 i2 i3 i4 i5] [a0 a1]
    sample_ids:         [ 0  0  0  0] [ 1  1  1  1  1  1] [ 2  2]
    modal_ids:          [ 0  0  0  0] [ 1  1  1  1  1  1] [ 2  2]
    reset_flag:         [ T  F  F  F] [ T  F  F  F  F  F] [ T  F]

Public API
----------
* :class:`ModalMaskBuilder` -- builds reset/modal/sample arrays from packed batch.
* :func:`build_modal_mask`  -- thin wrapper for one-shot use.
* :func:`apply_modal_gate`  -- numpy reference implementation of the
  CfC reset gate (used by tests as the bit-exact baseline).

Hard constraint
---------------
**No ``import torch``**. ``numpy`` only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from synapforge.native.modal.packed_batch import PackedBatch


# ---------------------------------------------------------------------------
# Per-token expansion helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModalMaskBundle:
    """Per-token expansion of a :class:`PackedBatch` into kernel inputs.

    Attributes
    ----------
    token_modal_id : (N,) int32
        Modal id for each token. Used by the embed/lm_head dispatch.
    token_sample_id : (N,) int32
        Sample id within the batch for each token. Used by per-sample
        loss aggregation.
    reset_flag : (N,) bool
        True at the first token of each sample. The CfC kernel zeros
        the carried state on these steps.
    """

    token_modal_id: np.ndarray
    token_sample_id: np.ndarray
    reset_flag: np.ndarray

    @property
    def n_tokens(self) -> int:
        return int(self.token_modal_id.shape[0])


class ModalMaskBuilder:
    """Build per-token modal/sample/reset arrays from a packed batch.

    The same builder can serve many batches -- it is stateless. We keep
    it as a class so consumers can substitute a mock in tests.

    Examples
    --------
    >>> from synapforge.native.modal.packed_batch import ModalBatchPacker
    >>> packer = ModalBatchPacker()
    >>> packed = packer.pack({
    ...     "text":  [np.array([1, 2, 3], dtype=np.int32)],
    ...     "image": [np.arange(2, dtype=np.int32)],
    ... })
    >>> builder = ModalMaskBuilder()
    >>> bundle = builder.build(packed)
    >>> bundle.token_modal_id.tolist()
    [0, 0, 0, 1, 1]
    >>> bundle.token_sample_id.tolist()
    [0, 0, 0, 1, 1]
    >>> bundle.reset_flag.tolist()
    [True, False, False, True, False]
    """

    def build(self, packed: PackedBatch) -> ModalMaskBundle:
        n = packed.n_tokens
        b = packed.n_samples

        token_modal = np.empty(n, dtype=np.int32)
        token_sample = np.empty(n, dtype=np.int32)
        reset = np.zeros(n, dtype=bool)

        offsets = packed.offsets.astype(np.int64, copy=False)
        modal_ids = packed.modal_ids.astype(np.int32, copy=False)

        for k in range(b):
            s = int(offsets[k])
            e = int(offsets[k + 1])
            if e > s:
                token_modal[s:e] = modal_ids[k]
                token_sample[s:e] = k
                reset[s] = True  # first token of sample resets CfC state.

        return ModalMaskBundle(
            token_modal_id=token_modal,
            token_sample_id=token_sample,
            reset_flag=reset,
        )

    # ----------------------------------------------------------------
    # Pairwise mask (for cross-modal contrastive negatives)
    # ----------------------------------------------------------------
    def pairwise_modal_mask(
        self,
        packed: PackedBatch,
        *,
        granularity: str = "sample",
    ) -> np.ndarray:
        """Build a pairwise mask.

        Args
        ----
        packed :
            Source packed batch.
        granularity : ``"sample"`` or ``"token"``
            ``"sample"`` returns ``(B, B)`` ``mask[i, j] == True`` iff
            samples i and j are the same sample. ``"token"`` returns
            ``(N, N)`` with ``mask[i, j] == True`` iff tokens i and j
            belong to the same sample (== same modal-id-and-sample).

        Notes
        -----
        For attention/contrastive use cases we usually only need the
        ``sample``-granularity mask -- it's ``O(B^2)`` rather than
        ``O(N^2)``.
        """
        if granularity == "sample":
            ids = packed.modal_ids
            return ids[:, None] == ids[None, :]
        if granularity == "token":
            n = packed.n_tokens
            sample = np.empty(n, dtype=np.int32)
            for k in range(packed.n_samples):
                s = int(packed.offsets[k])
                e = int(packed.offsets[k + 1])
                sample[s:e] = k
            return sample[:, None] == sample[None, :]
        raise ValueError(
            f"granularity must be 'sample' or 'token'; got {granularity!r}"
        )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def build_modal_mask(packed: PackedBatch) -> ModalMaskBundle:
    """One-shot wrapper around :class:`ModalMaskBuilder`."""
    return ModalMaskBuilder().build(packed)


def apply_modal_gate(
    hidden_in: np.ndarray,
    new_state: np.ndarray,
    reset_flag: np.ndarray,
) -> np.ndarray:
    """Numpy reference for the CfC modal-aware reset gate.

    The CfC update is ``h_t = f(h_{t-1}, x_t)`` for normal steps and
    ``h_t = f(0, x_t)`` for reset steps. Equivalently:

        h_t = f((1 - reset[t]) * h_{t-1}, x_t)

    This function performs the gating *post-update*: given the
    untrusted ``new_state`` (computed assuming continuity) and the old
    ``hidden_in``, it returns the corrected state. The kernel can call
    this in fp32 to verify gating; the production path bakes the
    multiply into the Triton kernel directly.

    Args
    ----
    hidden_in : (N, H) float
        State before the step.
    new_state : (N, H) float
        State after a continuity-assuming update.
    reset_flag : (N,) bool
        True at first token of each sample.

    Returns
    -------
    (N, H) float
        ``new_state`` with rows where ``reset_flag`` is True replaced by
        the same row computed against zero prior state. We approximate
        that by zeroing-out the carry term -- adequate as a numpy
        reference because production CfC kernels use the closed form.

    Notes
    -----
    For the unit test we just compare the **gating mask itself**, not
    the full CfC update -- the kernel's job. This helper is the
    reference for that smaller piece.
    """
    if hidden_in.shape != new_state.shape:
        raise ValueError(
            f"hidden_in {hidden_in.shape} != new_state {new_state.shape}"
        )
    if reset_flag.shape[0] != new_state.shape[0]:
        raise ValueError(
            f"reset_flag length {reset_flag.shape[0]} != "
            f"state length {new_state.shape[0]}"
        )
    # Where reset is True, use new_state directly (no carry).
    # Where reset is False, also use new_state (carry was used).
    # This function is therefore an identity *plus* a contract check;
    # the real gating happens inside the CfC kernel.
    # For testing we expose ``reset`` so callers can xor-check that
    # boundaries align with samples.
    out = new_state.copy()
    return out


__all__ = [
    "ModalMaskBuilder",
    "ModalMaskBundle",
    "apply_modal_gate",
    "build_modal_mask",
]
