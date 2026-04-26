"""LearnableThreshold — per-channel spike threshold with optional EMA tracking.

Vanilla PLIF (Wu et al. 2021, Fang et al. 2021) hard-codes the spike
threshold at 1.0; this kills neurons whose membrane potential settles
below threshold permanently.  Making threshold a learnable parameter
fixes that, but introduces a calibration problem — the right threshold
depends on the firing rate of incoming spikes.

This module:
  - holds a per-channel ``raw`` parameter (``nn.Parameter``)
  - clamps to ``[min_val, max_val]`` on every forward
  - optionally maintains an EMA of recent membrane-potential
    statistics so a calibration helper can re-centre threshold mid-training

Note: weight-decay should be **0** on this parameter (use a separate
no-decay group in the optimizer).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..module import Module


class LearnableThreshold(Module):
    """Per-channel learnable PLIF spike threshold with optional EMA.

    Args:
        hidden_size: feature dimension.
        init:        initial threshold value (broadcast to all channels).
        min_val:     lower clamp.
        max_val:     upper clamp.
        ema_decay:   EMA decay for the calibration buffer.  If 0 the
                     EMA is disabled (no buffer is updated).

    Forward:
        Without arguments returns the clamped threshold tensor.
        With a membrane tensor passed via ``forward(u)`` it also
        updates the EMA of ``|u|.mean()`` per channel (no-grad).
    """

    def __init__(
        self,
        hidden_size: int,
        init: float = 0.02,
        min_val: float = 1e-3,
        max_val: float = 3.0,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if not min_val < max_val:
            raise ValueError(f"min_val ({min_val}) must be < max_val ({max_val})")
        self.hidden_size = int(hidden_size)
        self.raw = nn.Parameter(torch.full((hidden_size,), float(init)))
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.ema_decay = float(ema_decay)
        if ema_decay > 0.0:
            self.register_buffer("u_abs_ema", torch.zeros(hidden_size))
        else:
            self.u_abs_ema = None  # type: ignore[assignment]

    def forward(self, u: torch.Tensor | None = None) -> torch.Tensor:
        """Return the clamped threshold.

        If ``u`` is provided (membrane potential ``[..., hidden_size]``)
        update the EMA of its per-channel mean absolute value (no-grad).
        """
        if u is not None and self.u_abs_ema is not None and self.training:
            with torch.no_grad():
                # Reduce all but the last axis.
                flat = u.detach().abs().reshape(-1, self.hidden_size).mean(dim=0)
                self.u_abs_ema.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * flat)
        return self.raw.clamp(self.min_val, self.max_val)

    @torch.no_grad()
    def calibrate(self, scale: float = 0.5) -> None:
        """Re-centre threshold to ``scale * EMA(|u|)``.

        Call once per N steps if dead-neuron rate exceeds budget.
        ``scale=0.5`` is a reasonable default (median-trigger).
        Does nothing if EMA is disabled or empty.
        """
        if self.u_abs_ema is None:
            return
        if not torch.any(self.u_abs_ema > 0):
            return
        self.raw.copy_(
            (scale * self.u_abs_ema).clamp(self.min_val, self.max_val)
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"min={self.min_val}, max={self.max_val}, "
            f"ema_decay={self.ema_decay}"
        )


__all__ = ["LearnableThreshold"]
