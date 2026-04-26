"""ImagePatchEmbed -- Fuyu-style 8x8 RGB patch embedding.

Pipeline:
    (B, 3, H, W) -uint8 or float-
        --> 8x8 non-overlapping patches via einops-style reshape
        --> linear projection 3*8*8 -> hidden
        --> add 2D sinusoidal positional encoding (row, col)
        --> prepend learned <|image|> marker
        --> (B, 1 + T_img, hidden) where T_img = (H/8) * (W/8)

Notes
-----
- We use einops-free pure torch for portability.
- Position encoding is built lazily and cached per (rows, cols) shape.
- bf16-friendly: linear weights are fp32 master, cast at forward.
- No attention here. The HybridBlock downstream models temporal/causal flow.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..module import Module


def sinusoidal_2d(rows: int, cols: int, dim: int, device, dtype) -> torch.Tensor:
    """Sinusoidal 2D position encoding: half dim for row, half for col.

    Returns (rows*cols, dim) flattened in row-major order.
    """
    if dim % 4 != 0:
        # round dim_each to a multiple of 2 so sin/cos pair cleanly; pad zeros
        dim_each = (dim // 2) - ((dim // 2) % 2)
    else:
        dim_each = dim // 2
    if dim_each <= 0:
        return torch.zeros(rows * cols, dim, device=device, dtype=dtype)

    # Frequency bands -- log-spaced (Vaswani-style).
    div_term = torch.exp(
        torch.arange(0, dim_each, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / dim_each)
    )

    pos_r = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    pos_c = torch.arange(cols, device=device, dtype=torch.float32).unsqueeze(1)
    enc_r = torch.zeros(rows, dim_each, device=device, dtype=torch.float32)
    enc_c = torch.zeros(cols, dim_each, device=device, dtype=torch.float32)
    enc_r[:, 0::2] = torch.sin(pos_r * div_term)
    enc_r[:, 1::2] = torch.cos(pos_r * div_term)
    enc_c[:, 0::2] = torch.sin(pos_c * div_term)
    enc_c[:, 1::2] = torch.cos(pos_c * div_term)

    # Broadcast to (rows, cols, dim_each) then concat along last to (rows*cols, dim)
    enc_r = enc_r.unsqueeze(1).expand(rows, cols, dim_each)
    enc_c = enc_c.unsqueeze(0).expand(rows, cols, dim_each)
    enc = torch.cat([enc_r, enc_c], dim=-1).reshape(rows * cols, 2 * dim_each)
    if enc.shape[-1] < dim:
        enc = torch.nn.functional.pad(enc, (0, dim - enc.shape[-1]))
    return enc.to(dtype)


class ImagePatchEmbed(Module):
    """8x8 (or configurable) RGB patch embedding, Fuyu-style.

    Forward
    -------
    image: (B, 3, H, W) float in [0,1] (or uint8 -- auto-normalized).
    Returns (B, T, hidden) with T = (H // patch) * (W // patch).
    A learned <|image|> marker is concatenated as the FIRST token of T.
    """

    def __init__(
        self,
        hidden: int = 512,
        patch: int = 8,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.patch = int(patch)
        self.in_channels = int(in_channels)
        in_feat = in_channels * patch * patch
        self.proj = nn.Linear(in_feat, hidden, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        # <|image|> modality marker (one learnable vector).
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)
        # Cache for sinusoidal table -- key is (rows, cols, dtype).
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    # --- helpers -------------------------------------------------------------

    def _patchify(self, image: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, T, C*patch*patch). H, W must be % patch."""
        B, C, H, W = image.shape
        p = self.patch
        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"image (H={H}, W={W}) must be divisible by patch={p}"
            )
        rows, cols = H // p, W // p
        # (B, C, rows, p, cols, p)
        x = image.reshape(B, C, rows, p, cols, p)
        # -> (B, rows, cols, C, p, p) -> (B, rows*cols, C*p*p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, rows * cols, C * p * p)
        return x

    def _pos(self, rows: int, cols: int, device, dtype) -> torch.Tensor:
        key = (rows, cols, str(dtype))
        if key not in self._pos_cache:
            self._pos_cache[key] = sinusoidal_2d(rows, cols, self.hidden, device, dtype)
        return self._pos_cache[key]

    # --- forward -------------------------------------------------------------

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(
                f"ImagePatchEmbed expects (B,C,H,W); got {tuple(image.shape)}"
            )
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0
        B, C, H, W = image.shape
        rows, cols = H // self.patch, W // self.patch
        # Patchify and project.
        flat = self._patchify(image)                        # (B, T, C*p*p)
        flat = flat.to(self.proj.weight.dtype)
        z = self.proj(flat)                                 # (B, T, hidden)
        # Add 2D positional encoding.
        pos = self._pos(rows, cols, image.device, z.dtype)  # (T, hidden)
        z = z + pos.unsqueeze(0)
        # Prepend modality marker.
        marker = self.marker.to(z.dtype).expand(B, 1, self.hidden)
        z = torch.cat([marker, z], dim=1)                   # (B, 1+T, hidden)
        return z

    @staticmethod
    def expected_token_count(H: int, W: int, patch: int = 8) -> int:
        return 1 + (H // patch) * (W // patch)
