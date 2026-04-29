"""
MoE FFN drop-in replacement for SwiGLU.

Recipes synthesized from:
  - DeepSeek-MoE 2401.06066 : fine-grained experts + shared expert
  - Mixtral 2401.04088      : top-2 sparse, noisy gating
  - OLMoE 2409.02060        : 64 routed × 8 active
  - Lifelong-MoE 2305.16380 : per-domain expert specialization
  - Jamba 2403.19887        : interleave dense + MoE layers

Our config (375M model):
  d_model = 1024
  ffn_mult = 4 (so dim = 4096 in dense baseline)
  n_routed = 8, n_shared = 1, top_k = 2
  expert_dim = 1024 (router-dim) → expert_internal = ffn_mult * 1024
  routed_capacity = 1024 / n_routed = 128 each
  Shared expert has full ffn_mult * 1024 dim.

Total params per layer:
  routed:  8 * (3 * 1024 * (ffn_mult*1024/n_routed)) ≈ 12.6M
  shared:  3 * 1024 * (ffn_mult*1024) ≈ 12.6M
  total:   ≈ 25M  (vs dense 12.6M)

Top-2 means only 2 routed experts run per token, so FLOPs ≈ 2/8 * routed + shared
  = 0.25 * 12.6 + 12.6 = 15.7M FLOPs/token (vs dense 12.6M)
  → 1.25× compute, 2× params, +capacity
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert: gate + up + down."""

    def __init__(self, d_in: int, d_internal: int, d_out: Optional[int] = None) -> None:
        super().__init__()
        d_out = d_out if d_out is not None else d_in
        self.gate = nn.Linear(d_in, d_internal, bias=False)
        self.up = nn.Linear(d_in, d_internal, bias=False)
        self.down = nn.Linear(d_internal, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class NoisyTopKGate(nn.Module):
    """Mixtral-style gate: top-k softmax + load-balance loss."""

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2, noise_eps: float = 1e-2) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_eps = noise_eps
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        self.w_noise = nn.Linear(d_model, n_experts, bias=False)
        nn.init.zeros_(self.w_noise.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B*T, d) -> (gates, top_idx, lb_loss)."""
        clean = self.w_gate(x)
        if self.training:
            noise_std = F.softplus(self.w_noise(x)) + self.noise_eps
            logits = clean + torch.randn_like(clean) * noise_std
        else:
            logits = clean

        top_logits, top_idx = logits.topk(self.top_k, dim=-1)
        gates = F.softmax(top_logits, dim=-1)

        importance = F.softmax(clean, dim=-1).sum(0)
        ce = (importance / x.shape[0]) * self.n_experts
        lb_loss = (ce * ce).mean()

        return gates, top_idx, lb_loss


class MoEFFN(nn.Module):
    """Fine-grained MoE FFN: shared expert + top-k routed experts."""

    def __init__(
        self,
        d_model: int,
        ffn_mult: int = 4,
        n_routed: int = 8,
        n_shared: int = 1,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_routed = n_routed
        self.n_shared = n_shared
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        d_routed_internal = (ffn_mult * d_model) // n_routed
        d_shared_internal = ffn_mult * d_model

        self.routed = nn.ModuleList([
            SwiGLUExpert(d_model, d_routed_internal) for _ in range(n_routed)
        ])
        self.shared = nn.ModuleList([
            SwiGLUExpert(d_model, d_shared_internal) for _ in range(n_shared)
        ])
        self.gate = NoisyTopKGate(d_model, n_routed, top_k=top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, d) -> (out, lb_loss)."""
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        shared_out = sum(s(x_flat) for s in self.shared) / max(len(self.shared), 1)

        gates, top_idx, lb_loss = self.gate(x_flat)
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.n_routed):
            mask = (top_idx == i).any(dim=-1)
            if not mask.any():
                continue
            sel_input = x_flat[mask]
            sel_gates = (top_idx[mask] == i).float() * gates[mask]
            sel_w = sel_gates.sum(-1, keepdim=True)
            expert_out = self.routed[i](sel_input) * sel_w
            routed_out[mask] += expert_out

        out = (shared_out + routed_out).reshape(B, T, D)
        return out, lb_loss

    def expert_load(self) -> torch.Tensor:
        """Diagnostic: counts of which experts ran in last forward (placeholder)."""
        return torch.zeros(self.n_routed)
