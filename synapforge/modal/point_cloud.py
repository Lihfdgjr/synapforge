"""PointCloudEmbed -- 3D point cloud embedding via voxel hashing.

Input
-----
points: (B, N, F) float -- N points per sample, each with F features.
        F=6 default: [x, y, z, r, g, b]. F can be 3 (xyz only) or any >=3.
mask:   optional (B, N) bool -- True for valid points (False = pad).

Pipeline
--------
1. Normalize xyz to [0, 1)^3 per-sample (min-max over valid points).
2. Hash each point into a (V x V x V) voxel grid (V=8 -> 512 voxels).
3. Per-voxel: max-pool features across all points falling into that voxel
   (PointNet trick). Empty voxels get a learned `empty` feature.
4. Linear-project (F -> hidden) the per-voxel features.
5. Add learned 3D positional encoding (V**3, hidden).
6. Prepend learned <|3d|> marker.

Returns
-------
(B, 1 + V**3, hidden). Empty voxels still occupy a slot but get the empty
feature; this keeps the token count fixed and dispatch simple. (V=8 means
512 tokens per cloud which fits the budget.)

Notes
-----
- Vanilla torch only; no torch_geometric / open3d.
- Differentiable: voxel hashing is non-diff but the projection AFTER pooling
  IS diff. Gradients flow through max-pool on the points that "won".
- bf16-friendly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..module import Module


def sinusoidal_3d(V: int, dim: int, device, dtype) -> torch.Tensor:
    """3D sinusoidal positional encoding for (V x V x V) grid -> (V**3, dim).

    Splits dim into 3 chunks (one per axis) padded with zeros if not divisible.
    """
    each = max(2, (dim // 3) - ((dim // 3) % 2))
    if each <= 0:
        return torch.zeros(V * V * V, dim, device=device, dtype=dtype)
    import math
    div = torch.exp(
        torch.arange(0, each, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / each)
    )

    def axis_enc(coord: torch.Tensor) -> torch.Tensor:
        c = coord.float().unsqueeze(1)
        e = torch.zeros(coord.shape[0], each, device=device, dtype=torch.float32)
        e[:, 0::2] = torch.sin(c * div)
        e[:, 1::2] = torch.cos(c * div)
        return e

    coords = torch.arange(V, device=device)
    enc_z = axis_enc(coords).unsqueeze(1).unsqueeze(2).expand(V, V, V, each)
    enc_y = axis_enc(coords).unsqueeze(0).unsqueeze(2).expand(V, V, V, each)
    enc_x = axis_enc(coords).unsqueeze(0).unsqueeze(1).expand(V, V, V, each)
    enc = torch.cat([enc_z, enc_y, enc_x], dim=-1).reshape(V * V * V, 3 * each)
    if enc.shape[-1] < dim:
        enc = torch.nn.functional.pad(enc, (0, dim - enc.shape[-1]))
    return enc.to(dtype)


class PointCloudEmbed(Module):
    """3D point cloud -> token sequence via voxel hashing + max pooling.

    Forward
    -------
    points: (B, N, F) float, F >= 3. xyz are first 3 channels.
    mask:   (B, N) bool optional. False entries excluded from min/max + pool.
    Returns (B, 1 + V**3, hidden).
    """

    def __init__(
        self,
        hidden: int = 512,
        voxel_grid: int = 8,
        feat_dim: int = 6,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.V = int(voxel_grid)
        self.feat_dim = int(feat_dim)
        if self.V < 2:
            raise ValueError("voxel_grid must be >= 2")
        self.proj = nn.Linear(feat_dim, hidden, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        # Learned vector for empty voxels (pre-projection feature).
        self.empty_feat = nn.Parameter(torch.zeros(feat_dim))
        nn.init.normal_(self.empty_feat, std=0.02)
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    def _pos(self, device, dtype) -> torch.Tensor:
        key = (str(dtype),)
        if key not in self._pos_cache:
            self._pos_cache[key] = sinusoidal_3d(self.V, self.hidden, device, dtype)
        return self._pos_cache[key]

    def _voxel_pool(
        self,
        points: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Voxel-hash + max-pool. Returns (B, V**3, F)."""
        B, N, F_dim = points.shape
        V = self.V
        Vt = V * V * V
        device = points.device

        xyz = points[..., :3]
        if mask is None:
            valid = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            valid = mask
        # Replace invalid xyz with 0 for safe min/max (we only use valid below).
        xyz_safe = torch.where(valid.unsqueeze(-1), xyz, torch.zeros_like(xyz))

        # Per-sample min/max across VALID points.
        # If a sample has 0 valid points we use [0,1] to avoid /0.
        big = torch.full_like(xyz, float("inf"))
        small = torch.full_like(xyz, float("-inf"))
        xyz_for_min = torch.where(valid.unsqueeze(-1), xyz, big)
        xyz_for_max = torch.where(valid.unsqueeze(-1), xyz, small)
        mn = xyz_for_min.amin(dim=1, keepdim=True)  # (B,1,3)
        mx = xyz_for_max.amax(dim=1, keepdim=True)
        no_valid = ~valid.any(dim=1)  # (B,)
        mn = torch.where(
            no_valid.view(-1, 1, 1),
            torch.zeros_like(mn),
            mn,
        )
        mx = torch.where(
            no_valid.view(-1, 1, 1),
            torch.ones_like(mx),
            mx,
        )
        rng = (mx - mn).clamp(min=1e-6)
        # Normalize xyz into [0, 1).
        norm = ((xyz_safe - mn) / rng).clamp(0.0, 1.0 - 1e-6)
        # Bucket into voxel idx. Do math in fp32 + clamp to handle bf16
        # precision (where 1.0 - 1e-6 rounds up to 1.0 -> ijk == V).
        ijk = (norm.float() * V).floor().long().clamp_(0, V - 1)
        i, j, k = ijk[..., 0], ijk[..., 1], ijk[..., 2]
        vidx = (i * V + j) * V + k  # (B, N) in [0, V**3)

        # Mask invalid points by sending them to a sentinel slot we ignore later.
        # We'll set their feat to -inf so they never win max-pool.
        feats = points  # (B, N, F)
        out = torch.full(
            (B, Vt, F_dim), float("-inf"), device=device, dtype=feats.dtype
        )
        feats_for_pool = torch.where(
            valid.unsqueeze(-1),
            feats,
            torch.full_like(feats, float("-inf")),
        )
        # Scatter-max via index_reduce_ (in-place on a non-leaf clone).
        # PyTorch >=2.1 supports torch.scatter_reduce with reduce="amax".
        out = out.scatter_reduce(
            dim=1,
            index=vidx.unsqueeze(-1).expand(-1, -1, F_dim),
            src=feats_for_pool,
            reduce="amax",
            include_self=True,
        )
        # Empty voxels remain -inf; replace with learned empty_feat.
        empty_mask = (out == float("-inf")).all(dim=-1, keepdim=True)
        empty_fill = self.empty_feat.to(out.dtype).view(1, 1, -1).expand_as(out)
        out = torch.where(empty_mask, empty_fill, out)
        return out

    def forward(
        self,
        points: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if points.dim() != 3:
            raise ValueError(
                f"PointCloudEmbed expects (B,N,F); got {tuple(points.shape)}"
            )
        F_in = points.shape[-1]
        if F_in < 3:
            raise ValueError(f"feat_dim must be >= 3 (xyz); got {F_in}")
        # If feat_dim mismatch, pad or truncate to expected feat_dim.
        if F_in != self.feat_dim:
            if F_in < self.feat_dim:
                pad = torch.zeros(
                    *points.shape[:-1], self.feat_dim - F_in,
                    device=points.device, dtype=points.dtype,
                )
                points = torch.cat([points, pad], dim=-1)
            else:
                points = points[..., : self.feat_dim]
        pooled = self._voxel_pool(points, mask)             # (B, V**3, F)
        pooled = pooled.to(self.proj.weight.dtype)
        z = self.proj(pooled)                                # (B, V**3, hidden)
        pos = self._pos(points.device, z.dtype)             # (V**3, hidden)
        z = z + pos.unsqueeze(0)
        B = points.shape[0]
        marker = self.marker.to(z.dtype).expand(B, 1, self.hidden)
        z = torch.cat([marker, z], dim=1)                   # (B, 1+V**3, hidden)
        return z

    @staticmethod
    def expected_token_count(voxel_grid: int = 8) -> int:
        return 1 + voxel_grid * voxel_grid * voxel_grid
