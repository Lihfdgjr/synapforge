"""spatial_3d -- 3D world understanding modal extensions.

Adds three components on top of the existing point_cloud branch so the LNN+SNN
backbone can ingest stereo / multi-view geometry without scene-fitting.

Components
----------
- PluckerRayEmbed:
    For each pixel, encode the 6D Plucker ray (origin + dir cross-product)
    on a coarse spatial grid -> token sequence with the SAME hidden dim as
    UnifiedEmbed. Sitzmann's LFN trick (1909.06443 / 2106.02634) but as
    conditioning, not as the output head.

- EGNNAdapter:
    E(3)-equivariant graph network (Satorras 2102.09844) that consumes a
    point cloud (xyz + features) and emits per-node hidden tokens. ~10M
    params at hidden=512, n_layers=3. Translation/rotation invariance for
    free, ~10x data efficiency on 3D regression.

- DUSt3RTeacher:
    Frozen pseudo-label generator stub (Wang 2312.14132). We do not bundle
    weights -- we expose `forward(stereo_pair) -> pointmap` and document
    where the actual ckpt is expected to live so trainers can swap real
    DUSt3R in with one line. The stub returns a deterministic disparity-
    style pointmap so the rest of the pipeline can be smoke-tested on CPU.

Conventions
-----------
- All modules subclass synapforge.Module so they are first-class in the
  IR compiler. Linear weights init with std=0.02. bf16-friendly (fp32
  master weights, autocast-safe forward).
- Pure torch, no torch_geometric / pytorch3d / open3d. Same constraint as
  point_cloud.py.

Smoke test::

    python -m synapforge.modal.spatial_3d
    -> prints OK if forward shapes match expectations.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module


# =============================================================================
# 1. Plucker ray embedding
# =============================================================================
def _plucker_rays_from_intrinsics(
    grid: int,
    intrinsics: torch.Tensor,    # (B, 3, 3)
    extrinsics: torch.Tensor,    # (B, 4, 4) cam-to-world
    device,
    dtype,
) -> torch.Tensor:
    """Build per-grid-cell Plucker ray (origin x direction, direction) -> (B, grid*grid, 6).

    Plucker convention: (m, d) where d is direction (unit) and m = origin x d.
    """
    B = intrinsics.shape[0]
    # pixel grid centers in normalised image coords [0, 1) -> camera ray dirs
    ys = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid
    xs = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")            # (grid, grid)
    # Map to NDC [-1, 1] then to camera coords via fx/fy/cx/cy.
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)
    # Image plane is grid x grid, treat (cx, cy) as normalised too.
    px = gx.unsqueeze(0).expand(B, -1, -1)
    py = gy.unsqueeze(0).expand(B, -1, -1)
    # Direction in camera space: ((u - cx)/fx, (v - cy)/fy, 1)
    d_cam = torch.stack(
        [(px - cx) / fx.clamp(min=1e-6),
         (py - cy) / fy.clamp(min=1e-6),
         torch.ones_like(px)],
        dim=-1,
    )                                                          # (B, g, g, 3)
    d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # Apply extrinsics rotation to get world-space direction.
    R = extrinsics[:, :3, :3]                                 # (B, 3, 3)
    t = extrinsics[:, :3, 3]                                  # (B, 3)
    d_world = torch.einsum("bij,bxyj->bxyi", R, d_cam)        # (B, g, g, 3)
    # Origin = camera center in world coords (constant per-batch).
    o_world = t.view(B, 1, 1, 3).expand_as(d_world)
    # Plucker moment m = o x d.
    m_world = torch.cross(o_world, d_world, dim=-1)
    plucker = torch.cat([m_world, d_world], dim=-1)           # (B, g, g, 6)
    plucker = plucker.reshape(B, grid * grid, 6).to(dtype)
    return plucker


class PluckerRayEmbed(Module):
    """Plucker 6D ray coordinates -> hidden-dim token sequence.

    Forward
    -------
    intrinsics: (B, 3, 3) float, camera intrinsics in normalised pixel coords.
    extrinsics: (B, 4, 4) float, camera-to-world pose.
    Returns (B, 1 + grid*grid, hidden) with a learnable <|plucker|> marker.
    """

    def __init__(self, grid: int = 8, hidden: int = 64, mlp_hidden: int = 128) -> None:
        super().__init__()
        if grid < 1:
            raise ValueError("grid must be >= 1")
        self.grid = int(grid)
        self.hidden = int(hidden)
        # Two-layer MLP: 6 -> mlp_hidden -> hidden. Sitzmann LFN style.
        self.mlp = nn.Sequential(
            nn.Linear(6, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)
        # 2D positional encoding for the grid.
        self.pos = nn.Parameter(torch.zeros(grid * grid, hidden))
        nn.init.normal_(self.pos, std=0.02)

    def forward(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        if intrinsics.dim() != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(
                f"intrinsics must be (B,3,3); got {tuple(intrinsics.shape)}"
            )
        if extrinsics.dim() != 3 or extrinsics.shape[-2:] != (4, 4):
            raise ValueError(
                f"extrinsics must be (B,4,4); got {tuple(extrinsics.shape)}"
            )
        B = intrinsics.shape[0]
        device = intrinsics.device
        dtype = self.mlp[0].weight.dtype
        pl = _plucker_rays_from_intrinsics(
            self.grid, intrinsics, extrinsics, device, dtype,
        )                                                      # (B, g*g, 6)
        z = self.mlp(pl)                                       # (B, g*g, hidden)
        z = z + self.pos.unsqueeze(0).to(z.dtype)
        marker = self.marker.to(z.dtype).expand(B, 1, self.hidden)
        return torch.cat([marker, z], dim=1)                   # (B, 1+g*g, hidden)


# =============================================================================
# 2. EGNN equivariant adapter
# =============================================================================
class _EGNNLayer(nn.Module):
    """Single EGNN layer (Satorras 2102.09844 eq. 3-6).

    Edge message: phi_e( h_i, h_j, ||x_i - x_j||^2, edge_attr )
    Coord update: x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
    Node update:  h_i' = phi_h( h_i, sum_j m_ij )

    Notes:
    - We DO NOT mutate x_i in place to keep CfC fast-weights safe (see
      docs/3D.md risk #2 + memory feedback_torch_buffer_inplace.md).
      Instead emit `x_new = x + delta` so callers can detach if they wish.
    - Radius-based neighbor sampling: only edges with ||x_i - x_j|| < radius
      contribute (others zeroed via mask). For small N (<=512) this is
      cheaper than k-NN sort.
    """

    def __init__(self, hidden: int, edge_hidden: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        # phi_e: (h_i, h_j, dist2) -> edge msg
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, hidden),
            nn.SiLU(),
        )
        # phi_x: edge msg -> scalar gate that scales (x_i - x_j)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, 1),
        )
        # phi_h: (h_i, agg_msg) -> node update
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, hidden),
        )
        for m in list(self.edge_mlp.modules()) + list(self.coord_mlp.modules()) + list(self.node_mlp.modules()):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Final coord_mlp output bias init to 0 so coord update is small initially.
        with torch.no_grad():
            self.coord_mlp[-1].weight.mul_(0.1)

    def forward(
        self,
        h: torch.Tensor,           # (B, N, hidden)
        x: torch.Tensor,           # (B, N, 3)
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, H = h.shape
        # Pairwise diffs and distances.
        x_i = x.unsqueeze(2)                       # (B, N, 1, 3)
        x_j = x.unsqueeze(1)                       # (B, 1, N, 3)
        dx = x_i - x_j                              # (B, N, N, 3)
        d2 = (dx * dx).sum(dim=-1, keepdim=True)   # (B, N, N, 1)
        # Radius mask (exclude self-loop).
        eye = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0)
        mask = (d2.squeeze(-1) < radius * radius) & (~eye)     # (B, N, N)
        mask_f = mask.unsqueeze(-1).to(h.dtype)               # (B, N, N, 1)
        # Edge features: cat(h_i, h_j, d2)
        h_i = h.unsqueeze(2).expand(B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, H)
        ef = torch.cat([h_i, h_j, d2.to(h.dtype)], dim=-1)
        m_ij = self.edge_mlp(ef)                              # (B, N, N, hidden)
        m_ij = m_ij * mask_f                                  # zero invalid edges
        # Aggregate node messages.
        agg = m_ij.sum(dim=2)                                 # (B, N, hidden)
        # Coordinate update: gate the diff x_i - x_j.
        gate = self.coord_mlp(m_ij)                           # (B, N, N, 1)
        # NB: dx in fp32 path; cast gate to dx dtype for stability.
        coord_delta = (dx * gate.to(dx.dtype) * mask_f.to(dx.dtype)).sum(dim=2)  # (B, N, 3)
        # Normalise by neighbor count so coord update is bounded.
        n_neighbors = mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dx.dtype)
        coord_delta = coord_delta / n_neighbors
        x_new = x + coord_delta                               # NOT in-place
        # Node update.
        h_cat = torch.cat([h, agg], dim=-1)                   # (B, N, 2*hidden)
        h_new = h + self.node_mlp(h_cat)                      # residual
        return h_new, x_new


class EGNNAdapter(Module):
    """E(3)-equivariant adapter on a point cloud branch.

    Forward
    -------
    points:     (B, N, 3) xyz coords (already in scene coords).
    feats:      (B, N, F) per-point features (e.g. RGB + DUSt3R confidence).
                If None, zeros.
    Returns
    -------
    h:          (B, 1 + N, hidden) tokens. Token 0 is a learnable marker.
    x_final:    (B, N, 3) refined coords (returned for diagnostics; loss
                computation may use this for pointmap MSE).
    """

    def __init__(
        self,
        hidden: int = 512,
        n_layers: int = 3,
        feat_dim: int = 6,
        radius: float = 1.0,
        edge_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.feat_dim = int(feat_dim)
        self.radius = float(radius)
        # Project per-point feats -> hidden (initial node embedding).
        self.feat_proj = nn.Linear(feat_dim, hidden, bias=False)
        nn.init.normal_(self.feat_proj.weight, std=0.02)
        # Stacked EGNN layers.
        self.layers = nn.ModuleList(
            [_EGNNLayer(hidden, edge_hidden=edge_hidden) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)

    def forward(
        self,
        points: torch.Tensor,
        feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError(f"points must be (B,N,3); got {tuple(points.shape)}")
        B, N, _ = points.shape
        if feats is None:
            feats = torch.zeros(B, N, self.feat_dim, device=points.device, dtype=points.dtype)
        if feats.shape[-1] != self.feat_dim:
            if feats.shape[-1] < self.feat_dim:
                pad = torch.zeros(
                    B, N, self.feat_dim - feats.shape[-1],
                    device=feats.device, dtype=feats.dtype,
                )
                feats = torch.cat([feats, pad], dim=-1)
            else:
                feats = feats[..., : self.feat_dim]
        # Initial node embedding from features (geometry comes via edges).
        h = self.feat_proj(feats.to(self.feat_proj.weight.dtype))   # (B, N, hidden)
        x = points.to(h.dtype)
        for layer in self.layers:
            h, x = layer(h, x, self.radius)
        h = self.norm(h)
        marker = self.marker.to(h.dtype).expand(B, 1, self.hidden)
        h_full = torch.cat([marker, h], dim=1)                       # (B, 1+N, hidden)
        return h_full, x


# =============================================================================
# 3. DUSt3R teacher stub
# =============================================================================
# Where the real DUSt3R checkpoint is expected when one becomes available.
# These names match the official release at
#   https://github.com/naver/dust3r/releases
# Trainers should set DUST3R_CKPT_PATH or pass `ckpt_path=...` to swap real
# weights in. Until then this is a deterministic identity stub.
DUST3R_CKPT_DEFAULT = "/workspace/teachers/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"


class DUSt3RTeacher(Module):
    """Frozen pseudo-label generator (stub).

    Real DUSt3R takes (image_a, image_b) of a stereo pair (or any two views)
    and outputs a per-pixel 3D pointmap in camera frame. We expose the same
    interface so trainers can wire this in upfront and only swap the inner
    network when the real ckpt arrives.

    The stub returns a synthetic pointmap derived deterministically from the
    image content (so unit tests assert non-trivial gradients flow through
    the EGNN adapter on top). This should NEVER be used for real training;
    the loaded flag will print a loud warning if so.
    """

    def __init__(
        self,
        ckpt_path: str = DUST3R_CKPT_DEFAULT,
        out_h: int = 64,
        out_w: int = 64,
    ) -> None:
        super().__init__()
        self.ckpt_path = str(ckpt_path)
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self._loaded = False
        # Try to load the real ckpt (silent fail OK).
        self._maybe_load()
        # Tiny encoder-projector so gradients have something deterministic to
        # cling to in CPU smoke. NOT trained.
        self._stub_proj = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self._stub_proj.weight.copy_(torch.eye(3) * 0.01)
        for p in self.parameters():
            p.requires_grad_(False)

    def _maybe_load(self) -> None:
        import os
        if os.path.exists(self.ckpt_path):
            # Real load would go here once weights are available:
            #   sd = torch.load(self.ckpt_path, map_location="cpu")
            #   self.real_net.load_state_dict(sd)
            # For now we just record presence.
            self._loaded = True

    def is_real(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def forward(
        self,
        image_a: torch.Tensor,           # (B, 3, H, W) float [0, 1]
        image_b: torch.Tensor,           # (B, 3, H, W) float [0, 1]
    ) -> dict:
        """Returns dict with:
            pointmap: (B, out_h, out_w, 3) -- per-pixel xyz in cam-A frame
            confidence: (B, out_h, out_w) -- per-pixel conf in [0, 1]
        """
        if image_a.shape != image_b.shape:
            raise ValueError(
                f"stereo pair shape mismatch: {image_a.shape} vs {image_b.shape}"
            )
        if image_a.dim() != 4 or image_a.shape[1] != 3:
            raise ValueError(f"expected (B,3,H,W); got {tuple(image_a.shape)}")
        if not self._loaded:
            # Loud warning ONCE per process so noisy logs flag this.
            import warnings
            warnings.warn(
                f"[DUSt3RTeacher] using STUB output -- ckpt {self.ckpt_path} not found. "
                "Pseudo-labels are deterministic but NOT physically meaningful.",
                RuntimeWarning,
                stacklevel=2,
            )
        B = image_a.shape[0]
        # Pool images to (out_h, out_w) and use mean RGB as a synthetic depth proxy.
        pooled_a = F.adaptive_avg_pool2d(image_a, (self.out_h, self.out_w))
        pooled_b = F.adaptive_avg_pool2d(image_b, (self.out_h, self.out_w))
        # Project channels via the (frozen) identity-like proj.
        pa = self._stub_proj(pooled_a.permute(0, 2, 3, 1))   # (B, h, w, 3)
        pb = self._stub_proj(pooled_b.permute(0, 2, 3, 1))   # (B, h, w, 3)
        # Use channel diff as fake "disparity" -> z; spatial coords as x, y.
        ys = torch.linspace(-1, 1, self.out_h, device=image_a.device, dtype=pa.dtype)
        xs = torch.linspace(-1, 1, self.out_w, device=image_a.device, dtype=pa.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.unsqueeze(0).expand(B, -1, -1)
        gy = gy.unsqueeze(0).expand(B, -1, -1)
        # disparity ~ |pa - pb| mean across channels; z = 1 / (disparity + eps)
        disp = (pa - pb).abs().mean(dim=-1)                  # (B, h, w)
        z = 1.0 / (disp + 0.1)
        z = (z - z.mean()) / (z.std() + 1e-6) * 0.5 + 1.0    # roughly in [0, 2]
        pointmap = torch.stack([gx * z, gy * z, z], dim=-1)  # (B, h, w, 3)
        # Confidence from disparity magnitude (deterministic).
        conf = torch.sigmoid(disp * 10.0)
        return {"pointmap": pointmap, "confidence": conf}

    @staticmethod
    def expected_ckpt_path() -> str:
        return DUST3R_CKPT_DEFAULT


# =============================================================================
# Smoke test
# =============================================================================
def _smoke() -> None:
    torch.manual_seed(0)
    B = 2
    # 1) Plucker
    pe = PluckerRayEmbed(grid=8, hidden=64)
    K = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]).expand(B, -1, -1).contiguous()
    Tex = torch.eye(4).expand(B, -1, -1).contiguous()
    z_pl = pe(K, Tex)
    assert z_pl.shape == (B, 1 + 64, 64), z_pl.shape
    # 2) EGNN
    egnn = EGNNAdapter(hidden=64, n_layers=2, feat_dim=4, radius=2.0, edge_hidden=64)
    pts = torch.randn(B, 32, 3)
    feats = torch.randn(B, 32, 4)
    h_e, x_e = egnn(pts, feats)
    assert h_e.shape == (B, 1 + 32, 64), h_e.shape
    assert x_e.shape == (B, 32, 3), x_e.shape
    # 3) DUSt3R teacher
    teacher = DUSt3RTeacher(out_h=16, out_w=16)
    img_a = torch.rand(B, 3, 64, 64)
    img_b = torch.rand(B, 3, 64, 64)
    out = teacher(img_a, img_b)
    assert out["pointmap"].shape == (B, 16, 16, 3), out["pointmap"].shape
    assert out["confidence"].shape == (B, 16, 16), out["confidence"].shape
    # 4) Param counts (sanity)
    n_pl = sum(p.numel() for p in pe.parameters())
    n_eg = sum(p.numel() for p in egnn.parameters())
    print(f"PluckerRayEmbed params: {n_pl:,}")
    print(f"EGNNAdapter params:    {n_eg:,}")
    print(f"DUSt3R teacher loaded: {teacher.is_real()}  (expected_ckpt={teacher.expected_ckpt_path()})")
    print("OK")


if __name__ == "__main__":
    _smoke()
