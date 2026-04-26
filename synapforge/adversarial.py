"""sf.adversarial — adversarial training primitives for synapforge.

Three pieces (ported from mscfc/adversarial.py + a generic FGSM/PGD pair):

  * FastGradientSignAttack    -- single-step FGSM (Goodfellow et al. 1412.6572)
  * ProjectedGradientDescentAttack -- iterative PGD-Linf (Madry 1706.06083)
  * TokenDiscriminator        -- 2-layer MLP for hidden-state classification
  * adversarial_losses        -- BCE generator/discriminator pair
  * AdversarialTrainer        -- wraps an outer optimiser, mixes adv samples

bf16-friendly: attacks always operate in fp32 internally to avoid sign-flip
nondeterminism in low precision; results cast back.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. FGSM
# ---------------------------------------------------------------------------


class FastGradientSignAttack:
    """Single-step Linf-bounded FGSM.

    ``call(x, y)`` returns ``x_adv`` where each element is moved by
    ``epsilon`` in the direction of the gradient sign.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> None:
        self.model = model
        self.epsilon = float(epsilon)
        self.loss_fn = loss_fn or self._default_loss
        self.clip_min = clip_min
        self.clip_max = clip_max

    @staticmethod
    def _default_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dtype in (torch.long, torch.int64) and pred.dim() >= 2:
            v = pred.shape[-1]
            return F.cross_entropy(pred.reshape(-1, v).float(), y.reshape(-1))
        return F.mse_loss(pred.float(), y.float())

    def call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            # Discrete inputs (token ids) — refuse, return unchanged.
            return x.detach()
        x_adv = x.detach().float().clone().requires_grad_(True)
        try:
            pred = self.model(tokens=x_adv)
        except TypeError:
            pred = self.model(x_adv)
        if isinstance(pred, tuple):
            pred = pred[0]
        loss = self.loss_fn(pred, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + self.epsilon * grad.sign()
        if self.clip_min is not None or self.clip_max is not None:
            x_adv = x_adv.clamp(min=self.clip_min, max=self.clip_max)
        return x_adv.to(x.dtype)


# ---------------------------------------------------------------------------
# 2. PGD
# ---------------------------------------------------------------------------


class ProjectedGradientDescentAttack:
    """Iterative Linf-bounded PGD (Madry et al.)."""

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.007,
        steps: int = 10,
        random_start: bool = True,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> None:
        self.model = model
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.random_start = bool(random_start)
        self.loss_fn = loss_fn or FastGradientSignAttack._default_loss
        self.clip_min = clip_min
        self.clip_max = clip_max

    def call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            return x.detach()
        x0 = x.detach().float()
        if self.random_start:
            delta = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
            x_adv = (x0 + delta).clone()
        else:
            x_adv = x0.clone()
        for _ in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            try:
                pred = self.model(tokens=x_adv)
            except TypeError:
                pred = self.model(x_adv)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.loss_fn(pred, y)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            # Project back into the Linf ball around x0.
            x_adv = torch.max(torch.min(x_adv, x0 + self.epsilon), x0 - self.epsilon)
            if self.clip_min is not None or self.clip_max is not None:
                x_adv = x_adv.clamp(min=self.clip_min, max=self.clip_max)
        return x_adv.to(x.dtype)


# ---------------------------------------------------------------------------
# 3. Discriminator (MSCFC port)
# ---------------------------------------------------------------------------


class TokenDiscriminator(nn.Module):
    """2-layer MLP → P(real | hidden)."""

    def __init__(self, hidden_size: int, mlp_hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


def _bce_with_logits(logit: torch.Tensor, target: float) -> torch.Tensor:
    tgt = torch.full_like(logit, float(target))
    return F.binary_cross_entropy_with_logits(logit, tgt)


def adversarial_losses(
    discriminator: TokenDiscriminator,
    real_hidden: torch.Tensor,
    fake_hidden: torch.Tensor,
    smooth: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (L_adv_generator, L_adv_discriminator)."""
    real_flat = real_hidden.reshape(-1, real_hidden.size(-1)).float()
    fake_flat = fake_hidden.reshape(-1, fake_hidden.size(-1)).float()
    d_real = discriminator(real_flat)
    d_fake_detached = discriminator(fake_flat.detach())
    loss_d = 0.5 * (
        _bce_with_logits(d_real, 1.0 - smooth)
        + _bce_with_logits(d_fake_detached, 0.0 + smooth)
    )
    d_fake = discriminator(fake_flat)
    loss_g = _bce_with_logits(d_fake, 1.0 - smooth)
    return loss_g, loss_d


# ---------------------------------------------------------------------------
# 4. AdversarialTrainer — wraps an optimiser
# ---------------------------------------------------------------------------


@dataclass
class AdversarialTrainerCfg:
    epsilon: float = 0.03
    alpha: float = 0.007
    pgd_steps: int = 5
    adv_weight: float = 0.5
    use_pgd: bool = True


class AdversarialTrainer:
    """Wraps an outer optimiser to inject FGSM/PGD adv samples in training.

    Lifecycle:
        adv = AdversarialTrainer(model, optim, cfg)
        loss = adv.train_step(x, y, loss_fn)  # mixes clean + adv examples
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: AdversarialTrainerCfg | None = None,
        defense: object | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg or AdversarialTrainerCfg()
        self.defense = defense
        if self.cfg.use_pgd:
            self.attack = ProjectedGradientDescentAttack(
                model,
                epsilon=self.cfg.epsilon,
                alpha=self.cfg.alpha,
                steps=self.cfg.pgd_steps,
            )
        else:
            self.attack = FastGradientSignAttack(model, epsilon=self.cfg.epsilon)

    def _forward_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        try:
            pred = self.model(tokens=x)
        except TypeError:
            pred = self.model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.shape != y.shape and y.dtype not in (torch.long, torch.int64):
            vmin = min(pred.shape[-1], y.shape[-1])
            pred = pred[..., :vmin]
            y = y[..., :vmin]
        return loss_fn(pred, y)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> float:
        loss_fn = loss_fn or (lambda p, t: F.mse_loss(p.float(), t.float()))
        self.optimizer.zero_grad(set_to_none=True)
        # Generate adversarial samples (no grad to model params during attack).
        for p in self.model.parameters():
            p.requires_grad_(False)
        try:
            x_adv = self.attack.call(x, y)
        finally:
            for p in self.model.parameters():
                p.requires_grad_(True)
        loss_clean = self._forward_loss(x, y, loss_fn)
        loss_adv = self._forward_loss(x_adv, y, loss_fn)
        loss = (1.0 - self.cfg.adv_weight) * loss_clean + self.cfg.adv_weight * loss_adv
        if self.defense is not None:
            self.defense.pre_step()
        loss.backward()
        if self.defense is not None:
            self.defense.after_grads()
        self.optimizer.step()
        if self.defense is not None:
            if self.defense.regressed_and_rollback():
                pass
            self.defense.tick()
        return float(loss.detach())


__all__ = [
    "FastGradientSignAttack",
    "ProjectedGradientDescentAttack",
    "TokenDiscriminator",
    "adversarial_losses",
    "AdversarialTrainerCfg",
    "AdversarialTrainer",
]
