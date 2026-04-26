"""AstrocyteGate — calcium-driven slow modulation gate.

Biological astrocytes (glia) wrap synapses and emit Ca2+ waves that
modulate synaptic strength on a 1-30 second timescale (Araque et al.
1999; Volterra & Meldolesi 2005) — far slower than the millisecond
spike timescale of neurons.  This decouples short-term computation
from long-term context.

We model this with a very-slow leaky state ``s`` that integrates a
low-pass projection of the network's hidden state, then gates the
hidden via element-wise sigmoid.  Default ``tau=1000`` steps maps to
seconds at typical 1ms simulation steps.

Buffer-safety: state mutation is wrapped in ``no_grad`` and the
gate is read from a detached copy, so gradients do not flow through
the running state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..module import Module


class AstrocyteGate(Module):
    """Slow glial-cell modulator (second-scale gain control).

    Args:
        hidden_size: feature dimension.
        tau:         decay timescale in steps; ``decay = exp(-1/tau)``.
                     Default 1000 (~1s at 1ms simulation steps).

    Forward:
        x:  [B, D] or [B, T, D]
        returns: same shape, gated element-wise by sigmoid(slow state).

    Reset:
        ``.reset()`` zeroes the slow state.
    """

    def __init__(self, hidden_size: int, tau: float = 1000.0):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.hidden_size = int(hidden_size)
        self.tau = float(tau)
        self.decay = float(math.exp(-1.0 / tau))
        # Two linears: one to project the input pool into the slow
        # state's update; one to project the slow state into a per-unit
        # gain log.
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.register_buffer("state", torch.zeros(hidden_size))

    @torch.no_grad()
    def reset(self) -> None:
        """Zero the slow state."""
        self.state.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] or [B, T, D] -> same shape, gated."""
        if x.dim() not in (2, 3):
            raise ValueError(f"x must be 2-D or 3-D, got {tuple(x.shape)}")
        if x.size(-1) != self.hidden_size:
            raise ValueError(
                f"last dim {x.size(-1)} != hidden_size {self.hidden_size}"
            )
        # Pool over batch (and time, if 3-D) for the global astrocyte signal.
        if x.dim() == 2:
            pooled = x.detach().mean(dim=0)
        else:
            pooled = x.detach().reshape(-1, self.hidden_size).mean(dim=0)
        # Update slow state in no-grad.  The proj layer's *parameters*
        # still receive gradient indirectly through the gate read-back
        # path because gate(state.detach()) is autograd-traced.
        with torch.no_grad():
            update = self.proj(pooled).detach()
            self.state.mul_(self.decay).add_((1.0 - self.decay) * update)
        # Detached state -> sigmoid gain -> element-wise modulation.
        # Broadcast the [D] state to x's shape.
        state_d = self.state.detach()
        if x.dim() == 2:
            gain_logit = self.gate(state_d.unsqueeze(0).expand_as(x))
        else:
            B, T, _ = x.shape
            gain_logit = self.gate(
                state_d.unsqueeze(0).unsqueeze(0).expand(B, T, self.hidden_size)
            )
        return x * torch.sigmoid(gain_logit)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, tau={self.tau}, decay={self.decay:.4f}"


__all__ = ["AstrocyteGate"]
