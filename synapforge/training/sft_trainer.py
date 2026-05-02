"""synapforge.training.sft_trainer -- SFT (instruction-tune) trainer subclass.

T9.4 (DEEP_MAINT_QUEUE.md H5): switch from LM-only to instruction-tune
SFT to break the val ppl plateau (~1000-2000 on raw LM data per scaling
laws) toward <= 10. Reuses :class:`BaseTrainer` plumbing and overrides
:meth:`compute_loss` to apply response-only CE masking.

Bit-exactness contract
----------------------
:meth:`SFTTrainer.compute_loss` produces the same loss as
``train_100m_sft.py``'s ``response_only_ce_loss`` call given the same
(logits, labels, loss_mask) -- the math is shared via
:func:`synapforge.training.sft_loop.response_only_ce_loss` (the exact
same function, just called from a different entry point).

Data contract
-------------
The train / val streams MUST yield ``(tokens_in, tokens_out, loss_mask)``
triples. :meth:`prepare_batch` accepts that shape. The recommended
producer is :class:`synapforge.training.sft_loop.InstructionParquetStream`,
which is what ``train_100m_sft.py`` already uses.

Programmatic use
----------------
::

    cfg = SFTTrainerConfig(
        out_dir="/tmp/sft_run",
        warmstart_ckpt="best_step_10000.pt",   # phase 1 KD ckpt
        steps=5000,
        batch_size=16,
        seq_len=512,
    )
    train_stream = InstructionParquetStream(...)
    val_stream   = InstructionParquetStream(..., loop=False)
    trainer = SFTTrainer(cfg, model, optim, train_stream, val_stream)
    trainer.run()
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .core_trainer import BaseTrainer, TrainerConfig, autocast_ctx
from .sft_loop import response_only_ce_loss


@dataclass
class SFTTrainerConfig(TrainerConfig):
    """Extension of :class:`TrainerConfig` with SFT-specific knobs."""

    # SFT loss
    response_only_loss: bool = True
    # Defaults bumped for SFT: bs=16, seq=512 is the sweet spot
    # documented in train_100m_sft.py header.
    batch_size: int = 16
    seq_len: int = 512
    # SFT runs longer per token but with smaller bs.
    save_every: int = 250
    eval_every: int = 250

    mode: str = "sft"


class SFTTrainer(BaseTrainer):
    """Phase 2 SFT trainer.

    Reuses BaseTrainer for everything except :meth:`compute_loss` and
    :meth:`prepare_batch`. The training data stream emits triples
    ``(tokens_in, tokens_out, loss_mask)``; the loss applies CE only on
    positions where the mask is non-zero (response-only training).

    Setting ``cfg.response_only_loss = False`` degrades to the
    instruction-LM baseline (full CE on all non-pad positions). The
    base CE math is bit-exact equivalent to
    ``F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))``
    when the mask is all-ones (asserted in sft_loop tests).
    """

    def __init__(self, cfg: SFTTrainerConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.cfg: SFTTrainerConfig = cfg  # type: ignore[assignment]

    def prepare_batch(self, raw):
        """Normalize a raw stream batch to ``(x, y, mask)`` tensors on device."""
        if len(raw) == 3:
            x, y, mask = raw
        elif len(raw) == 2:
            x, y = raw
            mask = torch.ones_like(y, dtype=torch.float32)
        else:
            raise ValueError(
                f"SFT batch must be 2- or 3-tuple; got len={len(raw)}"
            )
        x = x.to(self.device)
        y = y.to(self.device)
        mask = mask.to(self.device, dtype=torch.float32)
        # Override mask with all-ones if response-only is off (ablation).
        if not self.cfg.response_only_loss:
            mask = torch.ones_like(mask)
        return (x, y, mask)

    def compute_loss(self, batch) -> dict:
        """Compute response-only CE.

        Returns ``{"loss", "ce", "n_tokens"}``. ``ce`` and ``loss`` are
        identical (no auxiliary terms in pure SFT); we keep both keys so
        the metric log emits a ``ce=...`` column for compat with the KD
        log format.
        """
        x, y, mask = batch
        with autocast_ctx(self.device, self.dtype, self.device == "cuda"):
            logits = self.model(x)
            ce = response_only_ce_loss(
                logits.float(),
                y,
                mask,
                label_smoothing=self.cfg.label_smoothing,
            )
        return {
            "loss": ce,
            "ce": ce,
            "n_tokens": float(mask.sum().item()),
        }

    def val_step(self, batch) -> dict:
        """Validation: same response-only CE."""
        x, y, mask = batch
        with autocast_ctx(self.device, self.dtype, self.device == "cuda"):
            logits = self.model(x)
            ce = response_only_ce_loss(
                logits.float(),
                y,
                mask,
                label_smoothing=0.0,
            )
        return {"loss": ce}


__all__ = ["SFTTrainer", "SFTTrainerConfig"]
