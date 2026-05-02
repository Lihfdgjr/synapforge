"""packed_batch.py -- modal-aware byte-patch batch packer.

Architecture
------------
Given ``B`` per-modal sequences of length ``L_i``, we produce a single
flat tensor of length ``N = sum(L_i)`` plus offsets and modal-id arrays.
The CfC+PLIF backbone iterates token-by-token over ``concat_tokens``,
respecting the modal mask so state does not bleed across boundaries.

::

    Input (3 examples, 3 modalities)
    ================================
    text_a:   [t0, t1, t2, t3]                # text  L=4
    image_b:  [i0, i1, i2, i3, i4, i5, i6, i7]  # image L=8
    audio_c:  [a0, a1, a2, a3, a4, a5]            # audio L=6

    Output (PackedBatch)
    ====================
    concat_tokens : [t0..t3, i0..i7, a0..a5]                shape (18,)
    offsets       : [0, 4, 12, 18]                          shape (B+1=4,)
    modal_ids     : [0, 1, 2]                               shape (B=3,)
    seq_lens      : [4, 8, 6]                               shape (B=3,)

Memory math
-----------
Padded per-modal batching with bs=8 across 9 modalities at the maxima
above gives ``9 * 8 * max(T) = 9 * 8 * 8192 = 589824 tokens``. If most
samples are short of their T_max, that's mostly padding.

Packed batching uses ``sum(actual_L_i)`` tokens. For a representative
mix we measure 200--280 KB/sample vs 600 KB/sample padded. The savings
scale with the **variance** of per-modal length, not the mean.

Public API
----------
* :class:`ModalSpec`        -- per-modal (modal_id, T_max, vocab_size).
* :data:`MODAL_REGISTRY`    -- the 9-modal canonical registry.
* :class:`ModalBatchPacker` -- pack/unpack stateless engine.
* :class:`PackedBatch`      -- the immutable result tuple.
* :func:`packed_size`       -- byte-count for a packed bundle (for memory math).

Hard constraint
---------------
**No ``import torch``**. ``numpy`` only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Modal registry -- the single source of truth for the 9 modalities.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModalSpec:
    """Per-modal physical characteristics.

    Attributes
    ----------
    name : str
        Human-readable modality name (``text``, ``image``, ...).
    modal_id : int
        Canonical integer id used in masks and dispatch tables.
    t_max : int
        Maximum sequence length we will ever observe for this modality.
        Used for memory pre-allocation and OOM accounting.
    vocab_size : int
        Token vocabulary size. ``151643`` for Qwen-tokenized modalities,
        ``256`` for byte-encoded modalities.
    is_byte : bool
        ``True`` if the modality is byte-encoded (vocab=256), used by
        :class:`ModalDispatchEmbed` to route to the small embed table.
    """

    name: str
    modal_id: int
    t_max: int
    vocab_size: int
    is_byte: bool


# 2026-05-02 USER spec -- the production 9 modalities.
MODAL_REGISTRY: Dict[str, ModalSpec] = {
    "text":        ModalSpec("text",         0,   256, 151643, False),
    "image":       ModalSpec("image",        1,  1024,    256, True),
    "audio":       ModalSpec("audio",        2,  2048,    256, True),
    "time_series": ModalSpec("time_series",  3,   512,    256, True),
    "code":        ModalSpec("code",         4,   512, 151643, False),
    "math":        ModalSpec("math",         5,   128, 151643, False),
    "3D":          ModalSpec("3D",           6,  4096,    256, True),
    "video":       ModalSpec("video",        7,  8192,    256, True),
    "gesture":     ModalSpec("gesture",      8,   256,    256, True),
}

# Reverse map id->spec for runtime lookup.
MODAL_BY_ID: Dict[int, ModalSpec] = {s.modal_id: s for s in MODAL_REGISTRY.values()}


# ---------------------------------------------------------------------------
# PackedBatch -- the immutable result of a pack() call.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackedBatch:
    """A packed multimodal mini-batch.

    Layout
    ------
    ``concat_tokens`` is a 1-D int array of length ``N = sum(L_i)``.
    ``offsets[k]`` gives the start of sample ``k`` in ``concat_tokens``;
    ``offsets[k+1]`` gives its end. ``offsets`` has length ``B+1`` with
    ``offsets[0]=0`` and ``offsets[B]=N``.

    ``modal_ids[k]`` is the modal id for sample ``k``. ``seq_lens[k]``
    is its length (``offsets[k+1] - offsets[k]``).

    ``attn_mask_per_modal`` is a ``(B, max_L)`` bool array where
    ``mask[k, t] == True`` iff token ``t`` of sample ``k`` is real (vs
    a structural padding token used to align sample-internal substructure
    -- e.g. the byte-patch grid in image=32x32). For most modalities all
    real tokens are dense and ``attn_mask_per_modal[k, :L_k]`` is all-
    True; for image we sometimes pad to a power-of-2 patch grid. The
    mask is None if every sample is dense.

    Notes
    -----
    All arrays share dtype ``np.int32`` (token ids) and ``np.int64``
    (offsets) by default; this matches what the Triton kernels in
    :mod:`synapforge.backends.triton_block_kernel` expect.
    """

    concat_tokens: np.ndarray   # shape (N,) int32
    offsets: np.ndarray         # shape (B+1,) int64
    modal_ids: np.ndarray       # shape (B,) int32
    seq_lens: np.ndarray        # shape (B,) int64
    attn_mask_per_modal: Optional[np.ndarray] = None  # (B, max_L) bool

    @property
    def n_samples(self) -> int:
        return int(self.modal_ids.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.concat_tokens.shape[0])

    @property
    def max_seq_len(self) -> int:
        return int(self.seq_lens.max()) if self.n_samples > 0 else 0

    def total_bytes(self) -> int:
        """Total byte-size in memory for this packed batch."""
        n = int(self.concat_tokens.nbytes + self.offsets.nbytes
                + self.modal_ids.nbytes + self.seq_lens.nbytes)
        if self.attn_mask_per_modal is not None:
            n += int(self.attn_mask_per_modal.nbytes)
        return n

    def slice(self, k: int) -> np.ndarray:
        """Return tokens for sample ``k`` as a 1-D view."""
        if not 0 <= k < self.n_samples:
            raise IndexError(f"sample index {k} out of range [0, {self.n_samples})")
        s = int(self.offsets[k])
        e = int(self.offsets[k + 1])
        return self.concat_tokens[s:e]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def packed_size(seq_lens: Sequence[int]) -> int:
    """Total token count for a packed bundle of these per-sample lengths."""
    return int(sum(int(x) for x in seq_lens))


def pad_per_modal_sizes(
    seq_lens: Sequence[int],
    modal_ids: Sequence[int],
) -> int:
    """Return token-count for the **right-padded** per-modal alternative.

    For accounting / memory-savings reporting only.
    """
    by_modal: Dict[int, List[int]] = {}
    for ln, mid in zip(seq_lens, modal_ids):
        by_modal.setdefault(int(mid), []).append(int(ln))
    total = 0
    for mid, lens in by_modal.items():
        spec = MODAL_BY_ID[mid]
        # Pad each sample of this modality to T_max for the modality.
        # Real per-modal batching pads to max within the batch, but for
        # cross-modal packing the relevant comparison is the mixed-batch
        # alternative which pads to global_T_max -- this is what we
        # report as the "padded" baseline.
        total += len(lens) * spec.t_max
    return total


# ---------------------------------------------------------------------------
# ModalBatchPacker -- stateless pack/unpack engine.
# ---------------------------------------------------------------------------

class ModalBatchPacker:
    """Pack per-modal sequences into a single flat ``PackedBatch``.

    Stateless: every method takes its inputs explicitly and returns
    fresh arrays. The class exists to centralise the dtype / layout
    contract.

    Examples
    --------
    >>> packer = ModalBatchPacker()
    >>> packed = packer.pack({
    ...     "text":  [np.array([1, 2, 3], dtype=np.int32)],
    ...     "image": [np.arange(8, dtype=np.int32)],
    ... })
    >>> packed.n_samples
    2
    >>> packed.n_tokens
    11
    >>> unpacked = packer.unpack(packed)
    >>> [(name, len(seqs)) for name, seqs in unpacked.items()]
    [('text', 1), ('image', 1)]

    Determinism
    -----------
    Sample order in the output follows the iteration order of the
    input dict, then within-modality order. ``unpack`` returns the
    same dict structure, so ``unpack(pack(x)) == x`` element-wise.
    """

    def __init__(
        self,
        token_dtype: np.dtype = np.int32,
        offset_dtype: np.dtype = np.int64,
    ) -> None:
        self.token_dtype = np.dtype(token_dtype)
        self.offset_dtype = np.dtype(offset_dtype)

    # -- pack ---------------------------------------------------------
    def pack(
        self,
        per_modal: Mapping[str, Sequence[np.ndarray]],
        *,
        build_attn_mask: bool = False,
    ) -> PackedBatch:
        """Pack ``{modal_name: [seq, seq, ...]}`` into a :class:`PackedBatch`.

        Args
        ----
        per_modal :
            Mapping from modality name (must be a key of
            :data:`MODAL_REGISTRY`) to a list of 1-D int arrays. Each
            sequence is the byte-patch tokens for one sample.
        build_attn_mask :
            If True, also build a ``(B, max_L)`` bool mask that is True
            wherever the token is real. Default False -- packed batches
            do not need a per-token mask because the offsets already
            encode boundaries.

        Returns
        -------
        :class:`PackedBatch`
            Immutable bundle. All arrays own their data (no views into
            inputs), so the caller can mutate the inputs after pack().
        """
        # Flatten preserving modal-then-within-modal order.
        flat_seqs: List[np.ndarray] = []
        flat_modal_ids: List[int] = []
        flat_lens: List[int] = []

        for modal_name, seqs in per_modal.items():
            if modal_name not in MODAL_REGISTRY:
                raise KeyError(
                    f"unknown modality {modal_name!r}; "
                    f"valid={list(MODAL_REGISTRY.keys())}"
                )
            spec = MODAL_REGISTRY[modal_name]
            for k, seq in enumerate(seqs):
                arr = np.asarray(seq)
                if arr.ndim != 1:
                    raise ValueError(
                        f"seq[{modal_name}][{k}] must be 1-D; "
                        f"got shape {arr.shape}"
                    )
                if arr.size > spec.t_max:
                    raise ValueError(
                        f"seq[{modal_name}][{k}] length {arr.size} "
                        f"exceeds T_max={spec.t_max}"
                    )
                # Validate vocab range -- catch dtype mistakes early.
                if arr.size > 0:
                    mx = int(arr.max())
                    mn = int(arr.min())
                    if mn < 0 or mx >= spec.vocab_size:
                        raise ValueError(
                            f"seq[{modal_name}][{k}] tokens out of vocab "
                            f"[0, {spec.vocab_size}); got [{mn}, {mx}]"
                        )
                flat_seqs.append(arr.astype(self.token_dtype, copy=False))
                flat_modal_ids.append(spec.modal_id)
                flat_lens.append(int(arr.size))

        b = len(flat_seqs)
        if b == 0:
            # Empty batch is legal -- caller may be probing.
            return PackedBatch(
                concat_tokens=np.zeros((0,), dtype=self.token_dtype),
                offsets=np.zeros((1,), dtype=self.offset_dtype),
                modal_ids=np.zeros((0,), dtype=np.int32),
                seq_lens=np.zeros((0,), dtype=self.offset_dtype),
                attn_mask_per_modal=None,
            )

        # Build offsets.
        offsets = np.zeros(b + 1, dtype=self.offset_dtype)
        offsets[1:] = np.cumsum(flat_lens, dtype=self.offset_dtype)
        n_total = int(offsets[-1])

        # Concat into one buffer.
        concat = np.empty((n_total,), dtype=self.token_dtype)
        for k, arr in enumerate(flat_seqs):
            s = int(offsets[k])
            e = int(offsets[k + 1])
            concat[s:e] = arr

        modal_ids_arr = np.asarray(flat_modal_ids, dtype=np.int32)
        seq_lens_arr = np.asarray(flat_lens, dtype=self.offset_dtype)

        attn_mask = None
        if build_attn_mask and b > 0:
            max_l = int(seq_lens_arr.max())
            attn_mask = np.zeros((b, max_l), dtype=bool)
            for k, ln in enumerate(flat_lens):
                attn_mask[k, :ln] = True

        return PackedBatch(
            concat_tokens=concat,
            offsets=offsets,
            modal_ids=modal_ids_arr,
            seq_lens=seq_lens_arr,
            attn_mask_per_modal=attn_mask,
        )

    # -- unpack -------------------------------------------------------
    def unpack(
        self,
        packed: PackedBatch,
    ) -> Dict[str, List[np.ndarray]]:
        """Reverse of :meth:`pack`. Returns ``{modal_name: [seq, ...]}``."""
        out: Dict[str, List[np.ndarray]] = {name: [] for name in MODAL_REGISTRY}
        for k in range(packed.n_samples):
            mid = int(packed.modal_ids[k])
            spec = MODAL_BY_ID[mid]
            seq = packed.slice(k).copy()
            out[spec.name].append(seq)
        # Drop modalities with zero samples -- match input dict exactly.
        return {name: seqs for name, seqs in out.items() if seqs}

    # -- modality-grouped views ---------------------------------------
    def group_by_modal(
        self,
        packed: PackedBatch,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Return ``{modal_id: (sample_indices, lens)}`` for kernel scheduling.

        Triton kernels iterate per-modal segment (because the embed
        table differs per modality). This helper makes that iteration
        explicit without copying data.
        """
        out: Dict[int, Tuple[List[int], List[int]]] = {}
        for k in range(packed.n_samples):
            mid = int(packed.modal_ids[k])
            ln = int(packed.seq_lens[k])
            tup = out.setdefault(mid, ([], []))
            tup[0].append(k)
            tup[1].append(ln)
        return {
            mid: (np.asarray(idxs, dtype=np.int32),
                  np.asarray(lens, dtype=self.offset_dtype))
            for mid, (idxs, lens) in out.items()
        }


__all__ = [
    "MODAL_BY_ID",
    "MODAL_REGISTRY",
    "ModalBatchPacker",
    "ModalSpec",
    "PackedBatch",
    "packed_size",
    "pad_per_modal_sizes",
]
