"""sf.PLIF — Parametric LIF spiking neuron with learnable threshold.

Math
----
    tau     = exp(tau_log).clamp(1e-2, 1e3)        per-channel learnable
    decay   = exp(-dt / tau)
    mem_t   = mem_{t-1} * decay + current_t
    spike_t = ATan_surrogate(mem_t - thr)          forward: indicator
    mem_t   = mem_t - spike_t * thr                if reset_by_subtract
            = mem_t * (1 - spike_t)                else hard reset to 0

Surrogate gradient (Fang et al., ICCV 2021):
    d spike / d mem = alpha / (2 * (1 + (pi/2 * alpha * (mem - thr))^2))

API
---
    plif = sf.PLIF(hidden_dim, threshold=0.3, alpha=2.0,
                   reset_by_subtract=True, learnable_threshold=True)
    spk, mem = plif(current, membrane=None, dt=1.0)
    spike_rate = plif.last_spike_rate     # scalar tensor for monitoring

The threshold can be a constant or per-channel learnable nn.Parameter
(default learnable, init=0.3 to match LiquidCell tanh bound (-1,1)).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..module import Module


class _ATanSurrogate(torch.autograd.Function):
    """ATan surrogate gradient (Fang et al. 2021)."""

    @staticmethod
    def forward(ctx, mem: torch.Tensor, thr: torch.Tensor, alpha: float):
        ctx.save_for_backward(mem, thr if torch.is_tensor(thr) else None)
        ctx.alpha = float(alpha)
        ctx.thr_const = None if torch.is_tensor(thr) else float(thr)
        if torch.is_tensor(thr):
            return (mem >= thr).to(mem.dtype)
        return (mem >= float(thr)).to(mem.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        mem, thr_t = ctx.saved_tensors
        thr = thr_t if thr_t is not None else ctx.thr_const
        if torch.is_tensor(thr):
            thr_val = thr.detach()
        else:
            thr_val = thr
        alpha = ctx.alpha
        x = alpha * (mem - thr_val)
        # arctan derivative scaled
        grad_surrogate = alpha / (2.0 * (1.0 + (math.pi / 2.0 * x).pow(2)))
        grad_mem = grad_output * grad_surrogate
        # Threshold also receives -grad if learnable.
        grad_thr = None
        if thr_t is not None:
            grad_thr = -grad_mem.sum(
                dim=tuple(range(grad_mem.dim() - 1))
            ) if grad_mem.dim() > 1 else -grad_mem
        return grad_mem, grad_thr, None


def _spike(mem: torch.Tensor, thr, alpha: float) -> torch.Tensor:
    return _ATanSurrogate.apply(mem, thr, alpha)


class PLIF(Module):
    """Parametric LIF with optional learnable threshold + reset-by-subtract."""

    def __init__(
        self,
        hidden_dim: int,
        threshold: float = 0.3,
        alpha: float = 2.0,
        tau_init: float = 1.0,
        learnable_threshold: bool = True,
        reset_by_subtract: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.reset_by_subtract = bool(reset_by_subtract)

        # Per-channel learnable membrane time constant.
        self.tau_log = nn.Parameter(
            torch.full((hidden_dim,), math.log(tau_init))
        )
        if learnable_threshold:
            self.threshold = nn.Parameter(
                torch.full((hidden_dim,), float(threshold))
            )
        else:
            self.register_buffer(
                "threshold", torch.tensor(float(threshold))
            )
        # Last spike-rate buffer (scalar) for cheap monitoring without graph.
        self.register_buffer("last_spike_rate", torch.tensor(0.0))

    def tau(self) -> torch.Tensor:
        return self.tau_log.exp().clamp(1e-2, 1e3)

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor | None = None,
        dt: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step LIF update.

        Args
            current:  (..., hidden_dim) input drive (e.g., LiquidCell output)
            membrane: (..., hidden_dim) prior membrane potential. None -> zeros.
            dt:       scalar time step (default 1.0)

        Returns
            spike:    (..., hidden_dim) {0,1} indicator (with surrogate grad)
            mem_new:  (..., hidden_dim) post-spike membrane
        """
        if membrane is None:
            membrane = torch.zeros_like(current)
        decay = torch.exp(-dt / self.tau())
        mem_new = membrane * decay + current
        spk = _spike(mem_new, self.threshold, self.alpha)
        if self.reset_by_subtract:
            thr = self.threshold if torch.is_tensor(self.threshold) else float(self.threshold)
            mem_new = mem_new - spk * (thr.detach() if torch.is_tensor(thr) else thr)
        else:
            mem_new = mem_new * (1.0 - spk)

        with torch.no_grad():
            self.last_spike_rate.copy_(spk.mean().detach())
        return spk, mem_new


__all__ = ["PLIF"]
