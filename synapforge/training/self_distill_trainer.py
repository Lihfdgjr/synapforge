"""synapforge.training.self_distill_trainer -- self-distillation subclass (STUB).

Self-distillation: the model trains against its own EMA-shadow
distribution (a soft-target version of next-token CE). Phase 1.5
between the KD pass (Phase 0) and the SFT pass (Phase 2).

Migration path
--------------
The legacy ``train_100m_self_distill.py`` (~530 LOC) implements the
recipe with these moving parts:

* ``compute_self_kl(student_logits, ema_logits, T)`` -- KL of
  ``student_logits`` vs ``ema_logits`` at temperature T (Hinton).
* ``SFTBatcher`` -- a custom batcher that materialises
  ``(prompt_ids, response_ids, mask)`` tuples; same shape as the
  SFT loop's stream output.
* The training loop computes ``ema_logits`` by forwarding the same
  batch through ``ema_model`` (a frozen EMA-decayed shadow), then
  mixes ``alpha * CE + (1-alpha) * self_kl``.

To migrate cleanly, this subclass would need:
1. An ``ema_model`` argument at construction (or built lazily from
   ``self.ema_state`` once a few warmup steps have populated it).
2. ``compute_loss`` performs two forwards: student (with grad) and
   ema (no grad), then mixes. The forward order can use the same
   side-stream pattern as KD's teacher forward.

For now the stub raises NotImplementedError and the legacy script
remains the production entry point.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core_trainer import BaseTrainer, TrainerConfig


@dataclass
class SelfDistillTrainerConfig(TrainerConfig):
    """Stub config for future self-distill migration."""

    # Match train_100m_self_distill.py defaults
    self_kl_weight: float = 0.5
    self_kl_temperature: float = 2.0
    ema_decay: float = 0.999  # used for the shadow model (not just save)

    mode: str = "self_distill"


class SelfDistillTrainer(BaseTrainer):
    """Stub self-distillation trainer.

    See module docstring for the migration path. Callers should use
    ``python train_100m_self_distill.py`` until the BaseTrainer
    ema-forward hook lands.
    """

    def compute_loss(self, batch) -> dict:
        raise NotImplementedError(
            "SelfDistillTrainer is a placeholder; use "
            "train_100m_self_distill.py for production self-distill "
            "until BaseTrainer EMA-shadow forward hook lands."
        )


__all__ = ["SelfDistillTrainer", "SelfDistillTrainerConfig"]
