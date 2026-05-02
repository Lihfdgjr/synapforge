"""dispatch.py -- modal-aware embed/lm_head dispatch.

Why this exists
---------------
Modalities split cleanly into two vocab classes:

* ``text``, ``code``, ``math`` -- Qwen tokenizer, vocab=151643. They
  share the same big embed table.
* ``image``, ``audio``, ``time_series``, ``3D``, ``video``, ``gesture``
  -- byte encoded, vocab=256. They share the same small embed table.

Allocating one ``E_total = 151643 + 256`` table and using a per-modal
offset would waste memory (the byte modalities only need 256 rows but
would carry index space up to 151899 for no reason). Instead we keep
two physical tables and dispatch at lookup:

::

    embed(modal_id, token_id) =
        if MODAL_BY_ID[modal_id].is_byte:  byte_table[token_id]
        else:                              qwen_table[token_id]

The same dispatch logic applies to lm_head: we project the hidden
state back to whichever vocab the *current* token belongs to. That
saves a 590x larger logit tensor for byte modalities at every step.

Memory math (hidden=512, fp32)
------------------------------
* Qwen embed:  151643 x 512 x 4 = 311 MB
* Byte embed:     256 x 512 x 4 = 0.5 MB
* Combined dispatch: 311.5 MB total

If we instead stacked both into a single 151899-vocab table:
``151899 x 512 x 4 = 311.6 MB`` -- minor savings, but the *lm_head*
output would be ``(N, 151899) * 4`` which on a 8K-token packed batch
is ``8192 * 151899 * 4 = 5.0 GB``. Per-modal dispatch reduces the
lm_head output for byte modalities to ``(N_byte, 256) * 4 = 8.4 MB``,
a 595x reduction.

Public API
----------
* :class:`ModalDispatchEmbed` -- holds two embed tables + dispatch logic.
* :data:`QWEN_VOCAB`         = 151643
* :data:`BYTE_VOCAB`         = 256

Hard constraint
---------------
**No ``import torch``**. ``numpy`` only -- production gradients via VJP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from synapforge.native.modal.packed_batch import (
    MODAL_BY_ID,
    MODAL_REGISTRY,
    PackedBatch,
)


QWEN_VOCAB: int = 151643
BYTE_VOCAB: int = 256


# ---------------------------------------------------------------------------
# ModalDispatchEmbed
# ---------------------------------------------------------------------------

class ModalDispatchEmbed:
    """Modal-aware embed and lm_head with two physical tables.

    Examples
    --------
    >>> embed = ModalDispatchEmbed(hidden=8, seed=0)
    >>> # text token 5 -> qwen_table[5]
    >>> token_ids = np.array([5, 5], dtype=np.int32)
    >>> modal_ids = np.array([0, 1], dtype=np.int32)  # text, image
    >>> h = embed.lookup(token_ids, modal_ids)
    >>> h.shape
    (2, 8)
    >>> # The two outputs differ because they hit different tables.
    >>> bool((h[0] != h[1]).any())
    True

    Notes
    -----
    The class owns its weights as numpy arrays. Production training
    moves them to GPU via ``cupy.asarray`` and runs the lookup with a
    ``cupy.take`` (gather) kernel; the dispatch logic stays the same.
    """

    def __init__(
        self,
        hidden: int,
        *,
        qwen_vocab: int = QWEN_VOCAB,
        byte_vocab: int = BYTE_VOCAB,
        dtype: np.dtype = np.float32,
        seed: Optional[int] = None,
    ) -> None:
        if hidden <= 0:
            raise ValueError(f"hidden must be >0; got {hidden}")
        self.hidden = int(hidden)
        self.qwen_vocab = int(qwen_vocab)
        self.byte_vocab = int(byte_vocab)
        self.dtype = np.dtype(dtype)

        rng = np.random.default_rng(seed)
        self.qwen_table = (rng.standard_normal(
            (self.qwen_vocab, self.hidden), dtype=np.float64) * 0.02
        ).astype(self.dtype)
        self.byte_table = (rng.standard_normal(
            (self.byte_vocab, self.hidden), dtype=np.float64) * 0.02
        ).astype(self.dtype)

        # lm_head projections share weights with embed by default
        # (weight tying), but can be overridden.
        self.lm_head_qwen = self.qwen_table  # alias by default
        self.lm_head_byte = self.byte_table

    # -- introspection ---------------------------------------------------
    @property
    def qwen_table_bytes(self) -> int:
        return int(self.qwen_table.nbytes)

    @property
    def byte_table_bytes(self) -> int:
        return int(self.byte_table.nbytes)

    @property
    def total_bytes(self) -> int:
        return self.qwen_table_bytes + self.byte_table_bytes

    # -- forward lookup --------------------------------------------------
    def lookup(
        self,
        token_ids: np.ndarray,
        modal_ids: np.ndarray,
    ) -> np.ndarray:
        """Embed each (token_id, modal_id) pair using the right table.

        Args
        ----
        token_ids : (N,) int
        modal_ids : (N,) int
            Must align elementwise with ``token_ids``.

        Returns
        -------
        (N, H) float
            Hidden states for each token.
        """
        if token_ids.shape != modal_ids.shape:
            raise ValueError(
                f"token_ids {token_ids.shape} != modal_ids {modal_ids.shape}"
            )
        n = int(token_ids.shape[0])
        out = np.empty((n, self.hidden), dtype=self.dtype)
        if n == 0:
            return out

        # Build a per-modal mask using the registry.
        is_byte = np.zeros(n, dtype=bool)
        for mid in np.unique(modal_ids):
            mid_int = int(mid)
            if mid_int not in MODAL_BY_ID:
                raise KeyError(f"unknown modal_id {mid_int}")
            spec = MODAL_BY_ID[mid_int]
            is_byte[modal_ids == mid_int] = spec.is_byte

        # Dispatch: byte modalities -> byte_table, others -> qwen_table.
        byte_idx = np.where(is_byte)[0]
        qwen_idx = np.where(~is_byte)[0]

        if byte_idx.size > 0:
            ids = token_ids[byte_idx]
            if int(ids.max()) >= self.byte_vocab or int(ids.min()) < 0:
                raise ValueError(
                    f"byte token id out of range [0, {self.byte_vocab}); "
                    f"got [{int(ids.min())}, {int(ids.max())}]"
                )
            out[byte_idx] = self.byte_table[ids]

        if qwen_idx.size > 0:
            ids = token_ids[qwen_idx]
            if int(ids.max()) >= self.qwen_vocab or int(ids.min()) < 0:
                raise ValueError(
                    f"qwen token id out of range [0, {self.qwen_vocab}); "
                    f"got [{int(ids.min())}, {int(ids.max())}]"
                )
            out[qwen_idx] = self.qwen_table[ids]

        return out

    # -- packed-batch fast path ------------------------------------------
    def lookup_packed(self, packed: PackedBatch) -> np.ndarray:
        """Embed every token in a :class:`PackedBatch`.

        Builds the per-token modal id internally, then defers to
        :meth:`lookup`. This is the typical entry point used by the
        backbone runner.
        """
        n = packed.n_tokens
        token_modal = np.empty(n, dtype=np.int32)
        for k in range(packed.n_samples):
            s = int(packed.offsets[k])
            e = int(packed.offsets[k + 1])
            token_modal[s:e] = int(packed.modal_ids[k])
        return self.lookup(packed.concat_tokens, token_modal)

    # -- lm_head logits --------------------------------------------------
    def logits(
        self,
        hidden: np.ndarray,
        modal_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project hidden state to per-modal logits.

        Args
        ----
        hidden : (N, H) float
        modal_ids : (N,) int

        Returns
        -------
        (logits, vocab_id_per_token)
            * ``logits`` is a *list of arrays* in flattened form: we
              return the larger qwen-vocab matrix for qwen tokens and
              the smaller byte-vocab matrix for byte tokens, then
              concatenate them sparsely. To keep the API uniform we
              return a single ``(N, max_vocab)`` array but only the
              first ``vocab_id_per_token[n]`` columns are meaningful.

        Why two outputs?
        ----------------
        Returning ragged logits (different vocab sizes per token) does
        not vectorise on GPU. So instead we tag each row with its
        vocab size; the loss kernel can index into the row using the
        stride from ``vocab_id_per_token``. This sidesteps a ``5 GB``
        materialised logit tensor on long packed batches.

        Numerics
        --------
        Production code uses a fused ``softmax_cross_entropy`` kernel
        that never materialises the full logit. This numpy
        implementation is the slow reference.
        """
        if hidden.ndim != 2:
            raise ValueError(f"hidden must be (N, H); got {hidden.shape}")
        n, h = hidden.shape
        if h != self.hidden:
            raise ValueError(f"hidden width {h} != table width {self.hidden}")
        if modal_ids.shape[0] != n:
            raise ValueError(
                f"modal_ids has {modal_ids.shape[0]} rows; hidden has {n}"
            )

        max_vocab = max(self.qwen_vocab, self.byte_vocab)
        logits = np.zeros((n, max_vocab), dtype=hidden.dtype)
        vocab_per_tok = np.empty(n, dtype=np.int64)

        for mid in np.unique(modal_ids):
            mid_int = int(mid)
            spec = MODAL_BY_ID[mid_int]
            mask = modal_ids == mid_int
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            if spec.is_byte:
                vocab = self.byte_vocab
                table = self.byte_table
            else:
                vocab = self.qwen_vocab
                table = self.qwen_table
            row_logits = hidden[idx] @ table.T  # (k, vocab)
            logits[idx, :vocab] = row_logits
            vocab_per_tok[idx] = vocab

        return logits, vocab_per_tok


__all__ = [
    "BYTE_VOCAB",
    "ModalDispatchEmbed",
    "QWEN_VOCAB",
]
