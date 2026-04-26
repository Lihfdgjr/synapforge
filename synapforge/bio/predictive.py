"""PredictiveCoding — Rao-Ballard layer-wise prediction-error feedback.

Predictive-coding theory (Rao & Ballard 1999; Friston's free-energy
principle) holds that each cortical region predicts the *next* region's
input and propagates the residual upward as the only signal.  The model
that learns to predict best wins.

In a deep network this becomes a per-block prediction-error head:

    pred_h_next = predictor(h_cur)
    pred_loss   = MSE(pred_h_next, h_next.detach())

Adding ``pred_loss`` to the main objective encourages hierarchical
predictive representations — a cortical hallmark of real thinking.

The predictor is a single learnable linear (or 2-layer MLP, optional);
``h_next`` is detached so gradients flow only through the predictor,
not through the next block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module


class PredictiveCoding(Module):
    """Layer-wise prediction-error head (Rao-Ballard 1999).

    Args:
        hidden_size:  feature dimension of both ``h_cur`` and ``h_next``.
        depth:        1 (linear predictor, default) or 2 (MLP with GELU).
        residual:     when True, predict ``h_next - h_cur`` instead of
                      ``h_next`` directly.  Usually trains faster
                      because the null hypothesis is "next ~= cur".
        detach_target: when True (default), detach ``h_next`` so the
                      predictor's loss does not flow back into the
                      next block.

    Forward:
        h_cur:  [B, T, D] or [..., D]
        h_next: [B, T, D] or [..., D]  with the same shape
        returns: scalar MSE loss (mean-reduced).
    """

    def __init__(
        self,
        hidden_size: int,
        depth: int = 1,
        residual: bool = True,
        detach_target: bool = True,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if depth not in (1, 2):
            raise ValueError(f"depth must be 1 or 2, got {depth}")
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.residual = bool(residual)
        self.detach_target = bool(detach_target)
        if depth == 1:
            self.predictor: nn.Module = nn.Linear(hidden_size, hidden_size)
            # Zero-init final weight so prediction starts at zero
            # (matches null residual hypothesis).
            if residual:
                nn.init.zeros_(self.predictor.weight)
                nn.init.zeros_(self.predictor.bias)
        else:
            mid = hidden_size * 2
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size, mid),
                nn.GELU(),
                nn.Linear(mid, hidden_size),
            )
            if residual:
                nn.init.zeros_(self.predictor[-1].weight)
                nn.init.zeros_(self.predictor[-1].bias)

    def predict(self, h_cur: torch.Tensor) -> torch.Tensor:
        """Return the predicted ``h_next``.  Useful for inspection."""
        delta = self.predictor(h_cur)
        if self.residual:
            return h_cur + delta
        return delta

    def forward(self, h_cur: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        """Compute MSE prediction-error loss."""
        if h_cur.shape != h_next.shape:
            raise ValueError(
                f"shape mismatch: h_cur={tuple(h_cur.shape)} vs "
                f"h_next={tuple(h_next.shape)}"
            )
        if h_cur.size(-1) != self.hidden_size:
            raise ValueError(
                f"last dim {h_cur.size(-1)} != hidden_size {self.hidden_size}"
            )
        pred = self.predict(h_cur)
        target = h_next.detach() if self.detach_target else h_next
        return F.mse_loss(pred, target)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, depth={self.depth}, "
            f"residual={self.residual}, detach_target={self.detach_target}"
        )


__all__ = ["PredictiveCoding"]
