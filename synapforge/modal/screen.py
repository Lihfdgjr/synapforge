"""ScreenPatchEmbed -- desktop screenshot embedding (high-res, 32x32 patches, ROI).

Different from ImagePatchEmbed:
  - Adaptive patch size 32x32 (vs image's 8x8) so 1080p stays at ~2k tokens.
  - Optional cursor-aware ROI crop: pass `cursor=(cx, cy)` and `roi=(h, w)` to
    crop a focused window before patchifying. Useful for closed-loop computer
    use where the model only needs a foveal patch around the cursor.
  - <|screen|> modality marker (distinct from <|image|>).
  - Compatible with NeuroMCPHead: the resulting (B, T, hidden) feeds the same
    HybridBlock the action head reads from.

Forward
-------
screen: (B, 3, H, W) float in [0,1] OR uint8.
cursor: optional (B, 2) long tensor of (cy, cx); if given with `roi=(h,w)`,
        we crop a window around the cursor (clamped to image bounds).
Returns (B, 1+T, hidden) with T = (H'/p) * (W'/p) where H', W' are post-crop
dims, and a learned <|screen|> marker is the first token.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module
from .image import sinusoidal_2d


class ScreenPatchEmbed(Module):
    """Desktop screenshot patch embedding with optional cursor ROI."""

    def __init__(
        self,
        hidden: int = 512,
        patch: int = 32,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.patch = int(patch)
        self.in_channels = int(in_channels)
        in_feat = in_channels * patch * patch
        self.proj = nn.Linear(in_feat, hidden, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        # <|screen|> marker (distinct learnable vector).
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    # --- helpers ---------------------------------------------------------

    def _crop_around_cursor(
        self,
        image: torch.Tensor,
        cursor: torch.Tensor,
        roi_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Per-sample crop of (h, w) centered at cursor (cy, cx). Padded if OOB."""
        B, C, H, W = image.shape
        rh, rw = roi_hw
        # Snap roi dims to multiple of patch.
        p = self.patch
        rh = max(p, (rh // p) * p)
        rw = max(p, (rw // p) * p)
        out = torch.zeros(B, C, rh, rw, device=image.device, dtype=image.dtype)
        for b in range(B):
            cy, cx = int(cursor[b, 0].item()), int(cursor[b, 1].item())
            y0 = max(0, cy - rh // 2)
            x0 = max(0, cx - rw // 2)
            y1 = min(H, y0 + rh)
            x1 = min(W, x0 + rw)
            y0 = max(0, y1 - rh)  # back-shift if near right edge
            x0 = max(0, x1 - rw)
            crop = image[b:b + 1, :, y0:y1, x0:x1]
            ch, cw = crop.shape[-2], crop.shape[-1]
            out[b:b + 1, :, :ch, :cw] = crop
        return out

    def _patchify(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        p = self.patch
        if H % p != 0 or W % p != 0:
            # Pad to multiple of patch (right/bottom).
            ph = (p - H % p) % p
            pw = (p - W % p) % p
            image = F.pad(image, (0, pw, 0, ph))
            H, W = image.shape[-2], image.shape[-1]
        rows, cols = H // p, W // p
        x = image.reshape(B, C, rows, p, cols, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, rows * cols, C * p * p)
        return x

    def _pos(self, rows: int, cols: int, device, dtype) -> torch.Tensor:
        key = (rows, cols, str(dtype))
        if key not in self._pos_cache:
            self._pos_cache[key] = sinusoidal_2d(rows, cols, self.hidden, device, dtype)
        return self._pos_cache[key]

    # --- forward ---------------------------------------------------------

    def forward(
        self,
        screen: torch.Tensor,
        cursor: torch.Tensor | None = None,
        roi: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if screen.dim() != 4:
            raise ValueError(
                f"ScreenPatchEmbed expects (B,C,H,W); got {tuple(screen.shape)}"
            )
        if screen.dtype == torch.uint8:
            screen = screen.to(torch.float32) / 255.0
        if cursor is not None and roi is not None:
            screen = self._crop_around_cursor(screen, cursor, roi)
        B, C, H, W = screen.shape
        flat = self._patchify(screen)
        flat = flat.to(self.proj.weight.dtype)
        z = self.proj(flat)
        # Recompute rows/cols after potential pad.
        p = self.patch
        H2 = z.shape[1]  # rows*cols
        # Recover rows, cols from updated H, W (after pad in _patchify, image was
        # mutated only inside that function; redo math here.)
        H_pad = ((H + p - 1) // p) * p
        W_pad = ((W + p - 1) // p) * p
        rows, cols = H_pad // p, W_pad // p
        pos = self._pos(rows, cols, screen.device, z.dtype)
        z = z + pos.unsqueeze(0)
        marker = self.marker.to(z.dtype).expand(B, 1, self.hidden)
        z = torch.cat([marker, z], dim=1)
        return z

    @staticmethod
    def expected_token_count(H: int, W: int, patch: int = 32) -> int:
        rows = (H + patch - 1) // patch
        cols = (W + patch - 1) // patch
        return 1 + rows * cols
