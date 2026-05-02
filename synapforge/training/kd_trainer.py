"""synapforge.training.kd_trainer -- KD distillation trainer subclass.

Phase 0 of the phased recipe (memory:
``feedback_phased_training_2026q2``): LM + KD distillation. Teacher's
top-K softmax is mixed with student CE at weight ``kd_weight`` and
temperature ``kd_temperature`` (Hinton).

Bit-exactness contract
----------------------
This file's :meth:`KDTrainer.compute_loss` must produce the SAME
loss value as ``train_100m_kd.py``'s inner loop given the same
(student_logits, teacher_logits, x, y) tuple. The contract is:

    base_loss = CE(logits, y, label_smoothing) + z_w * z_loss
    if step % kd_every == 0 AND teacher available AND kd_weight > 0:
        kd     = kd_loss(student_logits, teacher_logits, T, chunk, topk)
        loss   = (1 - kd_weight) * base_loss + kd_weight * kd
    else:
        loss   = base_loss

Test ``tests/training/test_core_trainer.py::test_kd_math_bitexact``
verifies the kd_loss math itself; this docstring documents the loop
contract.

CLI / programmatic use
----------------------
::

    cfg = KDTrainerConfig(
        out_dir="/tmp/run",
        kd_weight=0.4,
        kd_temperature=4.0,
        kd_topk=2048,
    )
    trainer = KDTrainer(cfg, model, optim, train_stream, val_stream,
                        teacher=teacher_model)
    trainer.run()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
import torch.nn.functional as F

from .core_trainer import BaseTrainer, TrainerConfig
from .kd_math import kd_loss


@dataclass
class KDTrainerConfig(TrainerConfig):
    """Extension of :class:`TrainerConfig` with KD-specific knobs."""

    # KD distillation
    kd_weight: float = 0.4
    kd_temperature: float = 4.0
    kd_chunk: int = 0  # 0 = auto-tune
    kd_topk: int = 2048  # 0 disables top-K, falls back to chunked full-vocab
    kd_every: int = 1  # 1 = every step; 4 = every 4th step (skip 3 for speed)

    mode: str = "kd"


class KDTrainer(BaseTrainer):
    """KD distillation trainer.

    Subclass of :class:`BaseTrainer` that overrides :meth:`compute_loss`
    to add the KD KL term. The base class handles everything else:
    optimizer step, LR schedule, ckpt save/load, val loop, EMA, etc.

    The teacher model is passed at construction time. It MUST be in
    eval mode and frozen (``requires_grad_(False)``); the trainer does
    NOT enforce this -- the caller is responsible (matches the
    train_100m_kd.py contract).
    """

    def __init__(
        self,
        cfg: KDTrainerConfig,
        model,
        optim,
        train_stream: Iterator,
        val_stream: Optional[Iterator] = None,
        teacher=None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(cfg, model, optim, train_stream, val_stream, log_fn)
        self.teacher = teacher
        # Type narrow for IDE / readers.
        self.cfg: KDTrainerConfig = cfg  # type: ignore[assignment]

    @staticmethod
    def _teacher_forward(teacher, x: torch.Tensor) -> torch.Tensor:
        """Run teacher forward; handle HF (CausalLMOutput) vs Tensor return."""
        out = teacher(x)
        if hasattr(out, "logits"):
            return out.logits
        return out

    def compute_loss(self, batch) -> dict:
        """Compute KD-mixed loss.

        Mathematically identical to ``train_100m_kd.py`` inner loop:

            base_loss = CE(logits, y, label_smoothing) + z_w * z_loss
            kd_term   = kd_loss(logits, teacher_logits, T, chunk, topk)
                        if KD active else 0
            loss      = (1 - kd_w) * base_loss + kd_w * kd_term  if KD active
                        else  base_loss

        Returns ``{"loss", "ce", "kd", "z"}``. The base trainer logs
        every component automatically.
        """
        x, y = batch[0], batch[1]
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.amp.autocast(
            device_type=self.device,
            dtype=self.dtype,
            enabled=self.device == "cuda",
        ):
            logits = self.model(x)
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_y = y.reshape(-1)
            ce = F.cross_entropy(
                flat_logits, flat_y,
                label_smoothing=self.cfg.label_smoothing,
            )

            # z-loss (default OFF when z_loss_weight=0).
            if self.cfg.z_loss_weight > 0:
                # Match train_100m_kd: full logsumexp when topk=0.
                if self.cfg.z_loss_topk and self.cfg.z_loss_topk > 0:
                    k = min(int(self.cfg.z_loss_topk), flat_logits.size(-1))
                    top_vals, _ = flat_logits.topk(k, dim=-1)
                    log_z = torch.logsumexp(top_vals, dim=-1)
                else:
                    log_z = torch.logsumexp(flat_logits, dim=-1)
                z_loss = (log_z ** 2).mean()
                base_loss = ce + self.cfg.z_loss_weight * z_loss
            else:
                z_loss = torch.zeros((), device=logits.device)
                base_loss = ce

            # KD term.
            kd_active = (
                self.teacher is not None
                and self.cfg.kd_weight > 0
                and self.step % max(1, self.cfg.kd_every) == 0
            )
            if kd_active:
                with torch.no_grad():
                    t_logits = self._teacher_forward(self.teacher, x)
                kd = kd_loss(
                    logits, t_logits,
                    T=self.cfg.kd_temperature,
                    chunk_override=self.cfg.kd_chunk,
                    topk=self.cfg.kd_topk,
                )
                loss = (1.0 - self.cfg.kd_weight) * base_loss \
                    + self.cfg.kd_weight * kd
            else:
                kd = torch.zeros((), device=logits.device)
                loss = base_loss

        return {
            "loss": loss,
            "ce": ce,
            "kd": kd,
            "z": z_loss,
        }


__all__ = ["KDTrainer", "KDTrainerConfig"]
