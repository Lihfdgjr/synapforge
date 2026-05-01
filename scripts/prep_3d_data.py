"""prep_3d_data -- synthetic CLEVR-3D-style dataset for code-path validation.

Renders a tiny stereo dataset of cubes/spheres/cylinders on a checkerboard plane
using pure PyTorch raycasting (no OpenGL / Blender / pyrender required). Emits
parquet rows with paired stereo views, ground-truth pointmap, intrinsics, and a
short text caption. ~1000 examples is enough to validate the train_3d.py code
path on CPU.

Real CLEVR-3D download path (when GPU + bandwidth available)::

    # arxiv 2403.13554, hosted by Tsinghua THCHS3D mirror
    wget https://3dsrbench.github.io/data/clevr3d.tar.gz
    # or use HF Hub
    huggingface-cli download --repo-type dataset CLEVR-3D/clevr3d --local-dir ./clevr3d

The synthetic generator below is **NOT** a CLEVR replacement -- it reproduces
the SHAPE of the data so the trainer's collate / loss / EGNN forward exercise
all run end-to-end. Eval gates require the real dataset.

Usage::

    python scripts/prep_3d_data.py --n-examples 1000 --out /workspace/data/clevr3d_synth.parquet
    python scripts/prep_3d_data.py --n-examples 10 --out /tmp/clevr3d_smoke.parquet
"""
from __future__ import annotations

import argparse
import io
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# NOTE: do NOT import torch here -- this script is pure numpy. The trainer
# (train_3d.py) is the only consumer that needs torch.

SHAPES = ("cube", "sphere", "cylinder")
COLORS = {
    "red": (1.0, 0.2, 0.2),
    "green": (0.2, 1.0, 0.2),
    "blue": (0.2, 0.2, 1.0),
    "yellow": (1.0, 1.0, 0.2),
    "purple": (0.7, 0.2, 0.9),
}


def _png_bytes_from_uint8(arr: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as PNG bytes (no Pillow dependency).

    Falls back to a minimal lossless format compatible with downstream
    decoders: a tiny header + raw bytes. We DO use stdlib zlib for compression.
    Format: 4-byte 'SF3D' magic + uint16 H + uint16 W + zlib-compressed RGB.
    Trainer decoder must mirror this; documented in train_3d.py.
    """
    H, W, _ = arr.shape
    import zlib
    payload = zlib.compress(arr.tobytes(), level=1)
    header = b"SF3D" + struct.pack(">HH", H, W)
    return header + payload


def _make_intrinsics(H: int, W: int, fov_deg: float = 60.0) -> np.ndarray:
    """Pinhole intrinsics in NORMALISED pixel coords (matches PluckerRayEmbed).

    fx = fy = 0.5 / tan(fov/2); cx = cy = 0.5.
    """
    f = 0.5 / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array(
        [[f, 0.0, 0.5],
         [0.0, f, 0.5],
         [0.0, 0.0, 1.0]], dtype=np.float32,
    )
    return K


def _camera_extrinsics(angle: float, radius: float = 5.0, height: float = 2.0) -> np.ndarray:
    """Build a cam-to-world 4x4 looking at the origin from (radius, angle, height)."""
    cx = radius * math.cos(angle)
    cy = height
    cz = radius * math.sin(angle)
    cam_pos = np.array([cx, cy, cz], dtype=np.float32)
    fwd = -cam_pos / (np.linalg.norm(cam_pos) + 1e-6)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-6)
    new_up = np.cross(right, fwd)
    R = np.stack([right, new_up, -fwd], axis=1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = cam_pos
    return T


def _rasterize(
    H: int,
    W: int,
    K: np.ndarray,
    Tcw: np.ndarray,
    objects: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """CPU raycast onto an (H, W) image. Returns (rgb uint8, depth float32).

    Each object is a dict {shape, center: (3,), radius/half: float, color: (3,)}.
    Plus a y=-1 ground plane with checkerboard texture.
    """
    f = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    R = Tcw[:3, :3]
    cam_pos = Tcw[:3, 3]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    depth = np.full((H, W), 1e6, dtype=np.float32)
    # Per-pixel: compute world-space ray direction.
    # Build pixel grid in normalised coords [0, 1] then back-project.
    py = (np.arange(H, dtype=np.float32) + 0.5) / H
    px = (np.arange(W, dtype=np.float32) + 0.5) / W
    pyy, pxx = np.meshgrid(py, px, indexing="ij")
    # Camera-frame direction: ((u - cx)/f, (v - cy)/f, 1)
    d_cam = np.stack([(pxx - cx) / f, (pyy - cy) / f, np.ones_like(pxx)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    # World-frame.
    d_world = d_cam @ R.T                          # (H, W, 3)
    # --- ground plane intersect at y=-1 ---
    denom = d_world[..., 1]
    safe = np.abs(denom) > 1e-4
    t_plane = np.where(safe, (-1.0 - cam_pos[1]) / np.where(safe, denom, 1.0), 1e6)
    t_plane = np.where((t_plane > 0) & safe, t_plane, 1e6)
    hit_plane = t_plane[..., None] * d_world + cam_pos
    # Checkerboard via cosine pattern (smooth-ish).
    cb = ((np.floor(hit_plane[..., 0]) + np.floor(hit_plane[..., 2])) % 2.0).astype(np.float32)
    plane_col = np.stack([cb * 0.6 + 0.2, cb * 0.6 + 0.2, cb * 0.6 + 0.2], axis=-1)
    update = t_plane < depth
    rgb = np.where(update[..., None], plane_col, rgb)
    depth = np.where(update, t_plane, depth)
    # --- object intersects ---
    for obj in objects:
        c = np.array(obj["center"], dtype=np.float32)
        col = np.array(obj["color"], dtype=np.float32)
        if obj["shape"] == "sphere":
            r = float(obj["radius"])
            oc = cam_pos - c
            b = (d_world * oc).sum(axis=-1)
            cterm = (oc * oc).sum() - r * r
            disc = b * b - cterm
            mask = disc > 0
            t_sph = np.where(mask, -b - np.sqrt(np.maximum(disc, 0.0)), 1e6)
            t_sph = np.where((t_sph > 0) & mask, t_sph, 1e6)
            hit = t_sph[..., None] * d_world + cam_pos
            normal = (hit - c) / (np.linalg.norm(hit - c, axis=-1, keepdims=True) + 1e-6)
            shade = np.clip(normal[..., 1] * 0.5 + 0.5, 0, 1)
            sphere_col = col * shade[..., None]
            update = t_sph < depth
            rgb = np.where(update[..., None], sphere_col, rgb)
            depth = np.where(update, t_sph, depth)
        elif obj["shape"] == "cube":
            half = float(obj["half"])
            t_min = (c - half - cam_pos) / np.where(np.abs(d_world) > 1e-6, d_world, 1e-6)
            t_max = (c + half - cam_pos) / np.where(np.abs(d_world) > 1e-6, d_world, 1e-6)
            t1 = np.minimum(t_min, t_max).max(axis=-1)
            t2 = np.maximum(t_min, t_max).min(axis=-1)
            mask = (t2 > t1) & (t1 > 0)
            t_cube = np.where(mask, t1, 1e6)
            hit = t_cube[..., None] * d_world + cam_pos
            local = (hit - c) / max(half, 1e-6)
            face_strength = np.abs(local).max(axis=-1, keepdims=True)
            shade = (face_strength * 0.5 + 0.5).squeeze(-1)
            cube_col = col * np.clip(shade, 0, 1)[..., None]
            update = t_cube < depth
            rgb = np.where(update[..., None], cube_col, rgb)
            depth = np.where(update, t_cube, depth)
        elif obj["shape"] == "cylinder":
            r = float(obj["radius"])
            h = float(obj.get("height", 1.0))
            # Solve in xz plane only.
            ox = cam_pos[0] - c[0]
            oz = cam_pos[2] - c[2]
            dx = d_world[..., 0]
            dz = d_world[..., 2]
            A = dx * dx + dz * dz
            B = 2 * (ox * dx + oz * dz)
            C = ox * ox + oz * oz - r * r
            disc = B * B - 4 * A * C
            mask = (disc > 0) & (A > 1e-6)
            t_cyl = np.where(mask, (-B - np.sqrt(np.maximum(disc, 0.0))) / np.maximum(A, 1e-6), 1e6)
            hit = t_cyl[..., None] * d_world + cam_pos
            in_h = (hit[..., 1] > c[1] - h) & (hit[..., 1] < c[1] + h)
            mask = mask & (t_cyl > 0) & in_h
            t_cyl = np.where(mask, t_cyl, 1e6)
            shade = np.clip(0.6 + (hit[..., 1] - c[1]) / (h + 1e-6) * 0.4, 0, 1)
            cyl_col = col * shade[..., None]
            update = t_cyl < depth
            rgb = np.where(update[..., None], cyl_col, rgb)
            depth = np.where(update, t_cyl, depth)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    depth = np.where(depth >= 1e6, 0.0, depth).astype(np.float32)
    return rgb, depth


def _depth_to_pointmap(depth: np.ndarray, K: np.ndarray, Tcw: np.ndarray) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) world-frame pointmap."""
    H, W = depth.shape
    f = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    py = (np.arange(H, dtype=np.float32) + 0.5) / H
    px = (np.arange(W, dtype=np.float32) + 0.5) / W
    pyy, pxx = np.meshgrid(py, px, indexing="ij")
    d_cam = np.stack([(pxx - cx) / f, (pyy - cy) / f, np.ones_like(pxx)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    pts_cam = d_cam * depth[..., None]
    pts_world = pts_cam @ Tcw[:3, :3].T + Tcw[:3, 3]
    return pts_world.astype(np.float32)


def _make_scene(rng: np.random.Generator) -> tuple[list[dict], str]:
    """Generate 1-3 random objects + a short caption."""
    n = int(rng.integers(1, 4))
    out = []
    captions = []
    color_names = list(COLORS.keys())
    for _ in range(n):
        shape = SHAPES[int(rng.integers(0, 3))]
        cname = color_names[int(rng.integers(0, len(color_names)))]
        col = COLORS[cname]
        cx = float(rng.uniform(-1.5, 1.5))
        cz = float(rng.uniform(-1.5, 1.5))
        if shape == "sphere":
            r = float(rng.uniform(0.3, 0.6))
            out.append({"shape": "sphere", "center": (cx, -1.0 + r, cz), "radius": r, "color": col})
            captions.append(f"a {cname} sphere")
        elif shape == "cube":
            half = float(rng.uniform(0.25, 0.5))
            out.append({"shape": "cube", "center": (cx, -1.0 + half, cz), "half": half, "color": col})
            captions.append(f"a {cname} cube")
        else:
            r = float(rng.uniform(0.2, 0.4))
            h = float(rng.uniform(0.5, 0.8))
            out.append({"shape": "cylinder", "center": (cx, -1.0 + h, cz), "radius": r, "height": h, "color": col})
            captions.append(f"a {cname} cylinder")
    return out, "There is " + " and ".join(captions) + "."


def _gen_one(idx: int, H: int, W: int, rng: np.random.Generator) -> dict:
    K = _make_intrinsics(H, W)
    angle_a = float(rng.uniform(0, 2 * math.pi))
    angle_b = angle_a + float(rng.uniform(0.15, 0.4))   # small baseline
    Ta = _camera_extrinsics(angle_a)
    Tb = _camera_extrinsics(angle_b)
    objects, caption = _make_scene(rng)
    rgb_a, dep_a = _rasterize(H, W, K, Ta, objects)
    rgb_b, dep_b = _rasterize(H, W, K, Tb, objects)
    pointmap = _depth_to_pointmap(dep_a, K, Ta)         # in left-cam world frame
    return {
        "idx": idx,
        "image_left": _png_bytes_from_uint8(rgb_a),
        "image_right": _png_bytes_from_uint8(rgb_b),
        "depth_left": dep_a.tobytes(),
        "pointmap_gt": pointmap.tobytes(),
        "intrinsics": K.tobytes(),
        "extrinsics_left": Ta.tobytes(),
        "extrinsics_right": Tb.tobytes(),
        "image_h": H,
        "image_w": W,
        "caption": caption,
        "n_objects": len(objects),
        "object_meta": json.dumps(objects, default=lambda o: list(o) if isinstance(o, tuple) else str(o)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-examples", type=int, default=1000)
    p.add_argument("--image-h", type=int, default=64)
    p.add_argument("--image-w", type=int, default=64)
    p.add_argument("--out", type=str, required=True, help="Parquet output path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny config (32x32 image, 10 examples) for fast code-path validation.",
    )
    args = p.parse_args()
    if args.smoke:
        # Force tiny-config so smoke runs in seconds.
        args.n_examples = min(args.n_examples, 10)
        args.image_h = min(args.image_h, 32)
        args.image_w = min(args.image_w, 32)

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    for i in range(args.n_examples):
        rows.append(_gen_one(i, args.image_h, args.image_w, rng))
        if (i + 1) % max(1, args.n_examples // 10) == 0:
            elapsed = time.time() - t0
            print(f"[prep_3d] {i+1}/{args.n_examples} elapsed={elapsed:.1f}s", flush=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out_path), compression="zstd")
    sz = out_path.stat().st_size / (1024 ** 2)
    print(f"[prep_3d] wrote {len(rows)} rows -> {out_path} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
