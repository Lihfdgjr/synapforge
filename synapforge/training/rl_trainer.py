"""synapforge.training.rl_trainer -- RL trainer subclass (STUB).

Phase 4 of the phased training recipe (memory:
``feedback_phased_training_2026q2``): GRPO-style RL on top of an SFT
ckpt. Uses :mod:`synapforge.training.grpo` for the rollout/advantage
math.

Migration path
--------------
The legacy ``train_100m_rl.py`` is the production RL trainer. It
contains custom logic (rollout sampling via ``_sample_rollout_live``,
GSM8K / HumanEval prompt loaders, GRPO step ``_one_grpo_step``) that
does NOT cleanly map onto the BaseTrainer ``compute_loss`` shape
because:

* RL does N rollouts per "step" before computing a loss
* Rollout decoding requires generation (no teacher forcing)
* The "batch" stream is a list of RLPrompt, not (x, y)

To migrate, BaseTrainer would need:
1. A ``rollout(batch) -> trajectories`` hook called before
   ``compute_loss``.
2. ``compute_loss(trajectories) -> loss_dict`` working over the
   trajectories tensor instead of raw (x, y).
3. A reference policy snapshot mechanism (frozen copy of the model
   for KL constraint).

For now, this subclass raises NotImplementedError. Callers should
continue invoking ``python train_100m_rl.py`` directly. The dispatcher
in ``synapforge.training.__main__`` documents the workaround.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core_trainer import BaseTrainer, TrainerConfig


@dataclass
class RLTrainerConfig(TrainerConfig):
    """Stub config for future RL migration. Knobs match the legacy CLI."""

    # GRPO knobs (matched to train_100m_rl.py defaults)
    grpo_group_size: int = 8
    grpo_kl_coef: float = 0.04
    rollout_temperature: float = 0.7
    rollout_max_new_tokens: int = 256
    prompt_set: str = "gsm8k"  # gsm8k | humaneval

    mode: str = "rl"


class RLTrainer(BaseTrainer):
    """Stub RL trainer.

    See module docstring for the migration path. Callers should use
    ``python train_100m_rl.py`` until the BaseTrainer rollout hook
    lands.
    """

    def compute_loss(self, batch) -> dict:
        raise NotImplementedError(
            "RLTrainer is a placeholder; use train_100m_rl.py for "
            "production RL until BaseTrainer rollout hook lands. See "
            "synapforge/training/rl_trainer.py docstring for migration "
            "path."
        )


__all__ = ["RLTrainer", "RLTrainerConfig"]
