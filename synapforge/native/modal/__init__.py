"""synapforge.native.modal -- byte-patch optimized batching for 9 modalities.

Why this package exists
-----------------------
USER 2026-05-02 22:20: *"我们训练的还是全模态"*. Run 7 currently runs
text-only at fixed ``T=256``, but the production model trains on 9
modalities through a shared CfC+PLIF backbone (early-fusion, byte-patch
encoded). Each modality has a different physical sequence length:

==========  ======  ========  =========================
Modal       T_max   Vocab     Notes
==========  ======  ========  =========================
text         256    151643    Qwen tokenizer
image       1024       256    32x32 byte patches
audio       2048       256    16kHz, 128ms windows
time_series  512       256    sensor stream
code         512    151643    Qwen tokenizer
math         128    151643    Qwen tokenizer
3D          4096       256    voxel byte stream
video       8192       256    frames * pixel patches
gesture      256       256    joint angles per frame
==========  ======  ========  =========================

If we batch each modality with right-padding to its own ``T_max``, then
mix them in one mini-batch, the GPU spends 60-70% of its FLOPs on
padding tokens. Packing the modalities into a single flat token tensor
with offsets sidesteps that waste entirely.

Public API
----------
* :class:`ModalBatchPacker` -- packs per-modal sequences into a flat tensor.
* :class:`PackedBatch`      -- the (concat_tokens, offsets, modal_ids) tuple.
* :class:`ModalMaskBuilder` -- builds modal-aware masks for CfC/attention.
* :class:`CrossModalContrastive` -- cross-modal contrastive loss (CLIP-style).
* :class:`ModalDispatchEmbed` -- modal-aware embed/lm_head dispatch table.
* :data:`MODAL_REGISTRY`    -- metadata for the 9 supported modalities.

Hard constraints
----------------
* **Zero ``import torch``** in any file under this package. Pure
  ``numpy`` (CPU), with optional ``cupy`` (GPU).
* All public functions return ``numpy.ndarray`` (or ``cupy.ndarray``);
  the model code (which uses torch) is responsible for the final
  ``torch.from_numpy`` boundary.
"""
from __future__ import annotations

from synapforge.native.modal.cross_modal import (
    CrossModalContrastive,
    contrastive_loss,
    pairwise_cosine,
)
from synapforge.native.modal.dispatch import (
    BYTE_VOCAB,
    QWEN_VOCAB,
    ModalDispatchEmbed,
)
from synapforge.native.modal.modal_mask import (
    ModalMaskBuilder,
    apply_modal_gate,
    build_modal_mask,
)
from synapforge.native.modal.packed_batch import (
    MODAL_REGISTRY,
    ModalBatchPacker,
    ModalSpec,
    PackedBatch,
    pad_per_modal_sizes,
    packed_size,
)

__all__ = [
    "BYTE_VOCAB",
    "CrossModalContrastive",
    "MODAL_REGISTRY",
    "ModalBatchPacker",
    "ModalDispatchEmbed",
    "ModalMaskBuilder",
    "ModalSpec",
    "PackedBatch",
    "QWEN_VOCAB",
    "apply_modal_gate",
    "build_modal_mask",
    "contrastive_loss",
    "packed_size",
    "pad_per_modal_sizes",
    "pairwise_cosine",
]
