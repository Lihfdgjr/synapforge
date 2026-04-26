"""VideoPatchEmbed -- spatiotemporal 8x8 RGB patch x 4-frame embedding.

Pipeline:
    (B, T_f, 3, H, W)
        --> group every `temporal_patch=4` frames into one spatiotemporal cube
        --> 8x8 spatial patches inside each cube
        --> linear projection (3 * temporal_patch * patch * patch -> hidden)
        --> 3D positional encoding: (t, row, col) -- decomposed into 1D per axis,
            sinusoidal, summed, projected
        --> prepend learned <|video|> marker
        --> (B, 1 + T_unified, hidden) with T_unified = T_f' * (H/8) * (W/8)

Causal mask (temporal axis only) is exposed via .build_temporal_causal_mask()
for downstream consumers that want it. Our HybridBlock backbone is already
causal (LiquidCell + PLIF run a left-to-right scan), so the mask is informational.

Notes
-----
- T_f must be divisible by temporal_patch. We pad with zero frames on the
  right if not (cheap).
- bf16-friendly: weights master-fp32, cast at forward.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module
from .image import sinusoidal_2d
from .audio import sinusoidal_1d


class VideoPatchEmbed(Module):
    """Spatiotemporal patch embedding for video.

    Forward
    -------
    video: (B, T_f, 3, H, W) float in [0, 1] OR uint8 (auto-normalized).
    Returns (B, 1 + T, hidden) with T = (T_f // temporal_patch) * (H/p) * (W/p).
    """

    def __init__(
        self,
        hidden: int = 512,
        spatial_patch: int = 8,
        temporal_patch: int = 4,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.sp = int(spatial_patch)
        self.tp = int(temporal_patch)
        self.in_channels = int(in_channels)
        in_feat = in_channels * self.tp * self.sp * self.sp
        self.proj = nn.Linear(in_feat, hidden, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)
        # 3D pos = sin(t) + sin(r) + sin(c) in shared hidden dim.
        # We split hidden into 3 chunks (with floor); pad zeros.
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    def _patchify(self, video: torch.Tensor) -> torch.Tensor:
        """(B, T_f, C, H, W) -> (B, T_total, C*tp*sp*sp).

        Where T_total = (T_f // tp) * (H / sp) * (W / sp).
        Order: outer = temporal cube index, inner = (row, col) row-major.
        """
        B, T_f, C, H, W = video.shape
        if T_f % self.tp != 0:
            pad = self.tp - (T_f % self.tp)
            video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad))  # pad time dim
            T_f = video.shape[1]
        if H % self.sp != 0 or W % self.sp != 0:
            raise ValueError(
                f"video (H={H}, W={W}) must be divisible by spatial_patch={self.sp}"
            )
        T_cubes = T_f // self.tp
        rows, cols = H // self.sp, W // self.sp
        # (B, T_cubes, tp, C, rows, sp, cols, sp)
        x = video.reshape(B, T_cubes, self.tp, C, rows, self.sp, cols, self.sp)
        # -> (B, T_cubes, rows, cols, C, tp, sp, sp)
        x = x.permute(0, 1, 4, 6, 3, 2, 5, 7).contiguous()
        x = x.reshape(B, T_cubes * rows * cols, C * self.tp * self.sp * self.sp)
        return x, T_cubes, rows, cols

    def _pos_3d(self, T_cubes: int, rows: int, cols: int, device, dtype) -> torch.Tensor:
        key = (T_cubes, rows, cols, str(dtype))
        if key in self._pos_cache:
            return self._pos_cache[key]
        d = self.hidden
        # split into thirds for (t, r, c) with leftover zero-padded.
        d_each = d // 3
        pos_t = sinusoidal_1d(T_cubes, d_each, device, dtype)
        pos_r = sinusoidal_1d(rows, d_each, device, dtype)
        pos_c = sinusoidal_1d(cols, d_each, device, dtype)
        # Broadcast to (T_cubes, rows, cols, d_each)
        pt = pos_t.view(T_cubes, 1, 1, d_each).expand(T_cubes, rows, cols, d_each)
        pr = pos_r.view(1, rows, 1, d_each).expand(T_cubes, rows, cols, d_each)
        pc = pos_c.view(1, 1, cols, d_each).expand(T_cubes, rows, cols, d_each)
        enc = torch.cat([pt, pr, pc], dim=-1)
        if enc.shape[-1] < d:
            enc = F.pad(enc, (0, d - enc.shape[-1]))
        enc = enc.reshape(T_cubes * rows * cols, d)
        self._pos_cache[key] = enc
        return enc

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() != 5:
            raise ValueError(
                f"VideoPatchEmbed expects (B, T_f, C, H, W); got {tuple(video.shape)}"
            )
        if video.dtype == torch.uint8:
            video = video.to(torch.float32) / 255.0
        flat, T_cubes, rows, cols = self._patchify(video)        # (B, T, F)
        flat = flat.to(self.proj.weight.dtype)
        z = self.proj(flat)                                      # (B, T, hidden)
        pos = self._pos_3d(T_cubes, rows, cols, z.device, z.dtype)
        z = z + pos.unsqueeze(0)
        B = video.shape[0]
        marker = self.marker.to(z.dtype).expand(B, 1, self.hidden)
        z = torch.cat([marker, z], dim=1)
        return z

    @staticmethod
    def build_temporal_causal_mask(T_cubes: int, rows: int, cols: int,
                                   device=None) -> torch.Tensor:
        """Block-causal mask along the temporal axis only.

        Shape (T, T) where T = T_cubes * rows * cols. Within the same cube
        (frame group) tokens may attend to each other; across cubes only
        past cubes. Useful if a transformer adapter wraps the backbone.
        """
        T = T_cubes * rows * cols
        cube_id = torch.arange(T, device=device) // (rows * cols)
        m = cube_id.unsqueeze(0) <= cube_id.unsqueeze(1)
        return m  # bool (T, T)
