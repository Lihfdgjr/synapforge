"""
WeightFirewall: bound the damage any single batch can do.

Two complementary mechanisms (combined per agent synthesis):

  A. KL-anchor to baseline (TR-DPO 2404.09656 style)
       L_total = L_task + lambda_kl * KL(p_current || p_baseline_frozen)
     where p_baseline_frozen is updated only on canary-pass.

  B. Per-step gradient norm clipping with adaptive threshold
       g_norm = ||grad||
       if g_norm > k * EMA(g_norm, alpha=0.99): clip to k * EMA
       k = 5.0 (rare benign spikes); k = 3.0 in poison-suspect mode

  C. Synaptic Intelligence (SI) per-parameter importance:
       Omega_i = sum over training of |grad_i * delta_i|
       L_si = lambda_si * sum_i Omega_i * (theta_i - theta_baseline_i)^2
     This penalizes moving "important" parameters far from their baseline,
     preserving learned skills while still allowing adaptation.

The firewall is queried EVERY trainer step:
    if firewall.allow_step(grad_norm, batch_kl):
        opt.step()
    else:
        opt.zero_grad()  # skip; log incident
        firewall.suspect_count += 1
        if firewall.suspect_count > 10:
            firewall.rollback_to(baseline_ckpt)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FirewallState:
    g_norm_ema: float = 1.0
    kl_ema: float = 0.0
    suspect_count: int = 0
    rollback_count: int = 0
    blocks_total: int = 0
    last_canary_loss: float = float("inf")


class WeightFirewall:
    def __init__(
        self,
        baseline_state_dict: Optional[dict] = None,
        kl_clip: float = 0.5,
        grad_clip_k: float = 5.0,
        ema_alpha: float = 0.99,
        suspect_block_threshold: int = 10,
        si_lambda: float = 1.0,
    ) -> None:
        self.kl_clip = kl_clip
        self.grad_clip_k = grad_clip_k
        self.ema_alpha = ema_alpha
        self.suspect_block_threshold = suspect_block_threshold
        self.si_lambda = si_lambda

        self.state = FirewallState()
        self._baseline_state_dict = baseline_state_dict
        self._omega: dict = {}
        self._grad_norm_history: deque = deque(maxlen=100)
        self._suspect_mode = False

    def update_baseline(self, model: nn.Module) -> None:
        self._baseline_state_dict = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        self.state.suspect_count = 0
        self._suspect_mode = False

    def enter_suspect_mode(self) -> None:
        self._suspect_mode = True

    def exit_suspect_mode(self) -> None:
        self._suspect_mode = False
        self.state.suspect_count = 0

    @property
    def grad_clip_threshold(self) -> float:
        k = 3.0 if self._suspect_mode else self.grad_clip_k
        return k * max(self.state.g_norm_ema, 1e-3)

    def allow_step(self, grad_norm: float, batch_kl: float = 0.0) -> bool:
        self._grad_norm_history.append(grad_norm)
        self.state.g_norm_ema = self.ema_alpha * self.state.g_norm_ema + (1 - self.ema_alpha) * grad_norm
        self.state.kl_ema = self.ema_alpha * self.state.kl_ema + (1 - self.ema_alpha) * batch_kl

        block = False
        if grad_norm > self.grad_clip_threshold:
            block = True
        if batch_kl > self.kl_clip * 2:
            block = True

        if block:
            self.state.suspect_count += 1
            self.state.blocks_total += 1
            if self.state.suspect_count >= self.suspect_block_threshold:
                self._suspect_mode = True
            return False

        self.state.suspect_count = max(0, self.state.suspect_count - 1)
        return True

    def update_si_omega(self, model: nn.Module, lr: float) -> None:
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            delta = -lr * p.grad
            contrib = (p.grad * delta).abs()
            if name not in self._omega:
                self._omega[name] = torch.zeros_like(p, device="cpu")
            self._omega[name].add_(contrib.detach().cpu())

    def si_loss(self, model: nn.Module) -> torch.Tensor:
        if not self._omega or self._baseline_state_dict is None:
            return torch.zeros(1, device=next(model.parameters()).device)
        device = next(model.parameters()).device
        loss = torch.zeros(1, device=device)
        for name, p in model.named_parameters():
            if name not in self._omega or name not in self._baseline_state_dict:
                continue
            base = self._baseline_state_dict[name].to(device)
            omega = self._omega[name].to(device)
            loss = loss + (omega * (p - base) ** 2).sum()
        return self.si_lambda * loss

    def kl_anchor_loss(
        self,
        model: nn.Module,
        baseline_model: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            base_logits = baseline_model(input_ids).logits if hasattr(baseline_model(input_ids), "logits") else baseline_model(input_ids)
        cur_logits = model(input_ids)
        if hasattr(cur_logits, "logits"):
            cur_logits = cur_logits.logits
        log_p_cur = torch.log_softmax(cur_logits, dim=-1)
        p_base = torch.softmax(base_logits, dim=-1)
        kl = (p_base * (torch.log(p_base + 1e-8) - log_p_cur)).sum(-1).mean()
        return kl

    def rollback_decision(self, canary_loss: float) -> bool:
        delta = canary_loss - self.state.last_canary_loss
        if self.state.last_canary_loss == float("inf"):
            self.state.last_canary_loss = canary_loss
            return False
        if delta > 0.10 * self.state.last_canary_loss:
            self.state.rollback_count += 1
            return True
        self.state.last_canary_loss = canary_loss
        return False

    def stats(self) -> dict:
        return {
            "g_norm_ema": round(self.state.g_norm_ema, 4),
            "kl_ema": round(self.state.kl_ema, 4),
            "suspect_count": self.state.suspect_count,
            "suspect_mode": self._suspect_mode,
            "blocks_total": self.state.blocks_total,
            "rollback_count": self.state.rollback_count,
            "last_canary_loss": round(self.state.last_canary_loss, 4) if self.state.last_canary_loss != float("inf") else None,
        }
