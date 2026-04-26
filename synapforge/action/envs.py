"""sf.action.envs — toy environments + helpers for the action stack.

FourButtonEnv     deterministic 4-button click env.  Used to validate
                  NeuroMCPHead end-to-end (matches the original mscfc
                  validation runs).
PatchEncoder      tiny image -> patch-token encoder (linear-proj patches
                  + sinusoidal pos embed).
SpatialXYHead     attention-pooled coordinate prediction (continuous xy).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..module import Module


class FourButtonEnv:
    """Deterministic 4-button click task (the validation reference).

    Image is 64x64 RGB with a single coloured square on one of 4 fixed
    positions.  Reward = 1.0 if predicted xy is within 0.15 of the true
    button center, else 0.0.

    Used by mscfc.action's first PoC (4-button hit_rate -> 100%).
    """

    BUTTONS: tuple[tuple[float, float], ...] = (
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
    )

    def reset(self, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.randint(0, 4, (batch_size,))
        img = torch.zeros(batch_size, 3, 64, 64)
        color = torch.tensor([1.0, 0.5, 0.0])
        for b in range(batch_size):
            cx, cy = self.BUTTONS[target[b]]
            ix, iy = int(cx * 64), int(cy * 64)
            img[b, :, max(0, iy - 8):iy + 8, max(0, ix - 8):ix + 8] = color.view(3, 1, 1)
        return img, target

    def step(
        self, x_pred: torch.Tensor, y_pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        rewards = torch.zeros(target.size(0))
        for b in range(target.size(0)):
            tx, ty = self.BUTTONS[int(target[b].item())]
            if abs(float(x_pred[b].item()) - tx) < 0.15 and abs(float(y_pred[b].item()) - ty) < 0.15:
                rewards[b] = 1.0
        return rewards


class PatchEncoder(Module):
    """Image -> patch-token encoder with sinusoidal positional embedding."""

    def __init__(self, patch: int = 8, hidden: int = 256, img_size: int = 64) -> None:
        super().__init__()
        self.patch = int(patch)
        self.img_size = int(img_size)
        self.grid = self.img_size // self.patch
        self.proj = nn.Linear(3 * self.patch * self.patch, hidden)
        self.register_buffer("pos_emb", self._make_pos_emb(self.grid * self.grid, hidden))

    @staticmethod
    def _make_pos_emb(n: int, d: int) -> torch.Tensor:
        pe = torch.zeros(n, d)
        pos = torch.arange(n).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        p = self.patch
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, -1, c * p * p)
        z = self.proj(x)
        return z + self.pos_emb.unsqueeze(0)


class SpatialXYHead(Module):
    """Attention-pooled (x, y) head.  Outputs xy in [0, 1]^2 over a grid."""

    def __init__(self, hidden: int, grid: int = 8) -> None:
        super().__init__()
        self.score = nn.Linear(hidden, 1)
        ys = (torch.arange(grid).float() + 0.5) / grid
        xs = (torch.arange(grid).float() + 0.5) / grid
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer(
            "coords", torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(z).squeeze(-1)
        weights = scores.softmax(dim=-1)
        xy = (weights.unsqueeze(-1) * self.coords).sum(dim=1)
        return xy, weights


__all__ = ["FourButtonEnv", "PatchEncoder", "SpatialXYHead"]
