"""TimeSeriesEmbed -- multi-channel 1D time-series embedding.

For finance / IoT / physiological / sensor data. Variable channel count
(handled via padding to `max_channels` and a learnable per-channel embed).

Input
-----
series: (B, T_raw, C) float -- C channels, arbitrary T_raw.

Pipeline
--------
1. Per-channel learnable embed: project each scalar c -> a small per-channel
   vector. C is variable across calls; we pad to max_channels and use a mask.
2. Concatenate channel embeddings along feature dim -> (B, T_raw, hidden_in).
3. 1D conv with kernel = stride = patch_t -> (B, T_token, hidden).
4. Add 1D positional encoding.
5. Prepend learned <|series|> marker.

Returns
-------
(B, 1 + T_token, hidden) where T_token = ceil(T_raw / patch_t).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module
from .audio import sinusoidal_1d


class TimeSeriesEmbed(Module):
    """Multi-channel time-series embedding.

    Forward
    -------
    series: (B, T_raw, C) where C in [1, max_channels].
    Returns (B, 1+T_token, hidden) with T_token = ceil(T_raw / patch_t).
    """

    def __init__(
        self,
        hidden: int = 512,
        patch_t: int = 8,
        max_channels: int = 64,
        per_channel_dim: int = 8,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.patch_t = int(patch_t)
        self.max_channels = int(max_channels)
        self.per_channel_dim = int(per_channel_dim)
        # Per-channel learnable basis: maps scalar (1) -> per_channel_dim,
        # via a per-channel linear layer (weight + bias). Implemented as
        # one (max_channels, per_channel_dim) weight + (max_channels, per_channel_dim) bias
        # we apply via einsum.
        self.chan_w = nn.Parameter(torch.zeros(max_channels, per_channel_dim))
        self.chan_b = nn.Parameter(torch.zeros(max_channels, per_channel_dim))
        nn.init.normal_(self.chan_w, std=0.02)
        nn.init.normal_(self.chan_b, std=0.02)
        # 1D conv to patchify time; in_channels=max_channels*per_channel_dim,
        # out_channels=hidden.
        in_ch = max_channels * per_channel_dim
        self.tconv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=hidden,
            kernel_size=self.patch_t,
            stride=self.patch_t,
            bias=False,
        )
        nn.init.normal_(self.tconv.weight, std=0.02)
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    def _pos(self, T_token: int, device, dtype) -> torch.Tensor:
        key = (T_token, str(dtype))
        if key not in self._pos_cache:
            self._pos_cache[key] = sinusoidal_1d(T_token, self.hidden, device, dtype)
        return self._pos_cache[key]

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        if series.dim() != 3:
            raise ValueError(
                f"TimeSeriesEmbed expects (B,T,C); got {tuple(series.shape)}"
            )
        B, T_raw, C = series.shape
        if C > self.max_channels:
            raise ValueError(
                f"channels {C} > max_channels {self.max_channels}; "
                "rebuild with larger max_channels"
            )
        # Pad-channels so we can do batched einsum.
        if C < self.max_channels:
            pad = torch.zeros(
                B, T_raw, self.max_channels - C,
                device=series.device, dtype=series.dtype,
            )
            series = torch.cat([series, pad], dim=-1)
        # Per-channel linear: out (B, T, max_channels, per_channel_dim)
        # = series.unsqueeze(-1) * chan_w + chan_b (broadcast over channels).
        s = series.unsqueeze(-1).to(self.chan_w.dtype)             # (B, T, mc, 1)
        w = self.chan_w.unsqueeze(0).unsqueeze(0)                    # (1,1,mc,pc)
        b = self.chan_b.unsqueeze(0).unsqueeze(0)                    # (1,1,mc,pc)
        emb = s * w + b                                              # (B,T,mc,pc)
        # Mask out the unused channels (zeros in series got chan_b leak; rezero).
        if C < self.max_channels:
            mask = torch.zeros(
                self.max_channels, device=series.device, dtype=emb.dtype
            )
            mask[:C] = 1.0
            emb = emb * mask.view(1, 1, -1, 1)
        # Flatten last two dims -> (B, T, mc*pc), then conv1d wants (B, F, T).
        flat = emb.reshape(B, T_raw, self.max_channels * self.per_channel_dim)
        x = flat.transpose(1, 2)                                    # (B, F, T)
        # Pad to multiple of patch_t.
        if T_raw % self.patch_t != 0:
            pad_t = self.patch_t - (T_raw % self.patch_t)
            x = F.pad(x, (0, pad_t))
        z = self.tconv(x)                                           # (B, hidden, T_token)
        z = z.transpose(1, 2).contiguous()                          # (B, T_token, hidden)
        T_token = z.shape[1]
        pos = self._pos(T_token, series.device, z.dtype)           # (T_token, hidden)
        z = z + pos.unsqueeze(0)
        marker = self.marker.to(z.dtype).expand(B, 1, self.hidden)
        z = torch.cat([marker, z], dim=1)
        return z

    @staticmethod
    def expected_token_count(T_raw: int, patch_t: int = 8) -> int:
        return 1 + (T_raw + patch_t - 1) // patch_t
