"""synth_multimodal_smoke -- pure-numpy synthetic multimodal generators.

When the rental has no internet, or for the investor-demo CPU smoke run, this
script writes 100 deterministic synthetic rows per modality so the downstream
trainer never sees an empty parquet. Rows match the unified schema documented
in ``docs/MULTIMODAL_DATA.md``::

    bytes    : binary
    caption  : string
    modality : string
    source   : string  (always "synth_smoke")
    meta     : string  (JSON)

Generators (all deterministic given --seed):

    image       64x64 RGB uint8 colored shapes + label "a {color} {shape}"
    audio       16 kHz mono float32, 1 s sine + harmonics + noise; phoneme tag
    video       16 frames of 32x32 RGB uint8 with a moving rectangle
    time_series 256-step OHLCV w/ daily seasonality + GBM drift
    graph       Erdos-Renyi 16-node graph w/ 32-dim node features + class label
    biosignal   1024-sample 8-channel sin+1/f noise (ECG-shaped)
    spatial_3d  256 points sampled on cube/sphere primitives w/ RGB

Usage::

    python scripts/synth_multimodal_smoke.py --help
    python scripts/synth_multimodal_smoke.py --smoke      # 100 rows per modal
    python scripts/synth_multimodal_smoke.py --n 50 --seed 7
    python scripts/synth_multimodal_smoke.py --modality image --n 200

Constraints:
    - Only numpy + stdlib. No PIL/torch/imageio dependency.
    - pyarrow is needed to write parquet; falls back to JSONL if missing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("synth_multimodal_smoke")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ALL_MODALITIES = (
    "image", "audio", "video", "time_series",
    "graph", "biosignal", "spatial_3d",
)


# --------------------------------------------------------------------- helpers
def _try_import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except ImportError:
        return None, None


def _row(payload: bytes, caption: str, modality: str, meta: dict) -> dict:
    return {
        "bytes": payload,
        "caption": caption,
        "modality": modality,
        "source": "synth_smoke",
        "meta": json.dumps(meta, separators=(",", ":")),
    }


# ---------------------------------------------------------------------- image
def gen_image(seed: int, size: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    palette = [(220, 60, 60), (60, 220, 60), (60, 80, 220),
               (240, 220, 60), (180, 80, 200), (60, 200, 200)]
    color_names = ["red", "green", "blue", "yellow", "purple", "cyan"]
    shapes = ["circle", "square", "triangle"]
    H = W = size
    img = np.zeros((H, W, 3), dtype=np.uint8) + 30
    ci = int(rng.integers(0, len(palette)))
    si = int(rng.integers(0, len(shapes)))
    cy, cx = int(rng.integers(H // 4, 3 * H // 4)), int(rng.integers(W // 4, 3 * W // 4))
    r = int(rng.integers(4, H // 4))
    col = palette[ci]
    if shapes[si] == "circle":
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = col
    elif shapes[si] == "square":
        y0, y1 = max(0, cy - r), min(H, cy + r)
        x0, x1 = max(0, cx - r), min(W, cx + r)
        img[y0:y1, x0:x1] = col
    else:
        for dy in range(r):
            y = cy - r // 2 + dy
            if 0 <= y < H:
                w = max(1, r - dy)
                x0, x1 = max(0, cx - w // 2), min(W, cx + w // 2 + 1)
                img[y, x0:x1] = col
    caption = f"a {color_names[ci]} {shapes[si]}"
    return _row(img.tobytes(), caption, "image",
                {"H": H, "W": W, "C": 3, "dtype": "uint8"})


# ---------------------------------------------------------------------- audio
def gen_audio(seed: int, sr: int = 16000, dur_s: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    N = int(sr * dur_s)
    f0 = float(rng.uniform(110.0, 880.0))
    t = np.arange(N, dtype=np.float32) / sr
    n_harm = int(rng.integers(1, 5))
    wav = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
              for k in range(1, n_harm + 1)).astype(np.float32)
    wav += rng.normal(0, 0.02, N).astype(np.float32)
    wav = (wav / (np.max(np.abs(wav)) + 1e-6) * 0.6).astype(np.float32)
    vowels = ["ah", "ee", "oh", "uu", "ay"]
    label = vowels[min(len(vowels) - 1, int((f0 - 110) / 200))]
    caption = f"a synthetic tone at {int(f0)} Hz like '{label}'"
    return _row(wav.tobytes(), caption, "audio",
                {"sample_rate": sr, "dur_s": dur_s, "dtype": "float32"})


# ---------------------------------------------------------------------- video
def gen_video(seed: int, T: int = 16, size: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    clip = np.zeros((T, 3, size, size), dtype=np.uint8) + 20
    x0 = float(rng.uniform(2, size - 4))
    y0 = float(rng.uniform(2, size - 4))
    vx = float(rng.uniform(-1.0, 1.0))
    vy = float(rng.uniform(-1.0, 1.0))
    bw = int(rng.integers(2, 6))
    col = np.array([rng.integers(80, 255), rng.integers(80, 255),
                    rng.integers(80, 255)], dtype=np.uint8)
    for f in range(T):
        cx, cy = x0 + vx * f, y0 + vy * f
        x0i, y0i = max(0, int(cx)), max(0, int(cy))
        x1i, y1i = min(size, x0i + bw), min(size, y0i + bw)
        for c in range(3):
            clip[f, c, y0i:y1i, x0i:x1i] = col[c]
    direction = "right" if vx > 0 else "left"
    speed = "fast" if abs(vx) + abs(vy) > 0.6 else "slow"
    caption = f"a {speed} rectangle moving {direction}"
    return _row(clip.tobytes(), caption, "video",
                {"T": T, "C": 3, "H": size, "W": size, "dtype": "uint8"})


# ---------------------------------------------------------------- time_series
def gen_time_series(seed: int, T: int = 256, channels: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    mu = float(rng.uniform(-1e-3, 1e-3))
    sigma = float(rng.uniform(5e-3, 2e-2))
    s = 100.0
    rows = []
    for _ in range(T):
        r = rng.normal(mu, sigma)
        ns = max(0.01, s * (1.0 + r))
        hi = max(s, ns) * (1.0 + abs(rng.normal(0, sigma / 2)))
        lo = min(s, ns) * (1.0 - abs(rng.normal(0, sigma / 2)))
        vol = float(rng.exponential(scale=10.0))
        rows.append([s, hi, lo, ns, vol])
        s = ns
    sig = np.asarray(rows, dtype=np.float32)
    sig = sig[:, :channels]
    direction = "up" if sig[-1, 3] > sig[0, 0] else "down"
    return _row(sig.tobytes(),
                f"OHLCV {T}-step series trending {direction}",
                "time_series",
                {"T": T, "channels": channels, "dtype": "float32"})


# ---------------------------------------------------------------------- graph
def gen_graph(seed: int, n_nodes: int = 16, node_feat: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    nodes = rng.normal(0, 1, (n_nodes, node_feat)).astype(np.float32)
    src = list(range(n_nodes))
    dst = [(i + 1) % n_nodes for i in range(n_nodes)]
    n_extra = int(rng.integers(0, n_nodes))
    for _ in range(n_extra):
        a, b = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
        if a != b:
            src.append(a); dst.append(b)
    edges = np.stack([np.array(src), np.array(dst)], axis=-1).astype(np.int64)
    payload = nodes.tobytes() + edges.tobytes()
    caption = (f"a molecule-shaped graph with {n_nodes} nodes "
               f"and {len(src)} edges")
    return _row(payload, caption, "graph",
                {"n_nodes": n_nodes, "node_feat": node_feat,
                 "n_edges": len(src), "dtype": "float32+int64"})


# ------------------------------------------------------------------ biosignal
def gen_biosignal(seed: int, T: int = 1024, channels: int = 8,
                  sr: int = 256) -> dict:
    rng = np.random.default_rng(seed)
    hr = float(rng.uniform(50, 110))
    period = sr * 60.0 / hr
    sig = np.zeros((T, channels), dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    for c in range(channels):
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = float(rng.uniform(0.5, 1.5))
        sig[:, c] = (amp * np.sin(2 * np.pi * t / period + phase)
                     + rng.normal(0, 0.1, T) * 0.5).astype(np.float32)
    if hr < 60:
        tag = "bradycardia"
    elif hr > 100:
        tag = "tachycardia"
    else:
        tag = "normal"
    return _row(sig.tobytes(), f"ECG at {int(hr)} bpm ({tag})",
                "biosignal",
                {"T": T, "channels": channels, "sample_rate": sr,
                 "dtype": "float32"})


# ----------------------------------------------------------------- spatial_3d
def gen_spatial_3d(seed: int, n_pts: int = 256) -> dict:
    rng = np.random.default_rng(seed)
    kind = int(rng.integers(0, 2))
    if kind == 0:
        phi = rng.uniform(0, 2 * np.pi, n_pts)
        cos_th = rng.uniform(-1, 1, n_pts)
        sin_th = np.sqrt(np.clip(1 - cos_th ** 2, 0, 1))
        r = float(rng.uniform(0.5, 1.5))
        x = r * sin_th * np.cos(phi)
        y = r * sin_th * np.sin(phi)
        z = r * cos_th
        caption = f"a sphere of radius {r:.2f}"
    else:
        half = float(rng.uniform(0.4, 1.2))
        face = rng.integers(0, 6, n_pts)
        u = rng.uniform(-half, half, n_pts)
        v = rng.uniform(-half, half, n_pts)
        x = np.where(face == 0, +half, np.where(face == 1, -half, u))
        y = np.where(face == 2, +half, np.where(face == 3, -half,
              np.where(face <= 1, u, v)))
        z = np.where(face == 4, +half, np.where(face == 5, -half,
              np.where(face <= 3, v, u)))
        caption = f"a cube of side {2 * half:.2f}"
    rgb = rng.uniform(0, 1, (n_pts, 3)).astype(np.float32)
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    full = np.concatenate([pts, rgb], axis=-1)
    return _row(full.tobytes(), caption, "spatial_3d",
                {"n_pts": n_pts, "feat": 6, "dtype": "float32"})


GENERATORS: dict[str, Callable[..., dict]] = {
    "image": gen_image,
    "audio": gen_audio,
    "video": gen_video,
    "time_series": gen_time_series,
    "graph": gen_graph,
    "biosignal": gen_biosignal,
    "spatial_3d": gen_spatial_3d,
}


# ----------------------------------------------------------------- write side
def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pa, pq = _try_import_pyarrow()
    if pa is None:
        # Fallback: jsonl with base64 for `bytes` so we still produce output.
        import base64
        with path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "bytes_b64": base64.b64encode(r["bytes"]).decode("ascii"),
                    "caption": r["caption"], "modality": r["modality"],
                    "source": r["source"], "meta": r["meta"],
                }) + "\n")
        return
    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, str(path), compression="zstd")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/multimodal/synth_smoke",
                   help="output root dir")
    p.add_argument("--n", type=int, default=100,
                   help="rows per modality (default 100)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modality", default="ALL",
                   help="comma-separated subset, default ALL")
    p.add_argument("--smoke", action="store_true",
                   help="alias for --n 100; implied default")
    args = p.parse_args(argv)

    if args.modality.upper() == "ALL":
        modal_list = list(ALL_MODALITIES)
    else:
        modal_list = [m.strip() for m in args.modality.split(",") if m.strip()]
        unknown = [m for m in modal_list if m not in GENERATORS]
        if unknown:
            logger.error(f"unknown modalities: {unknown}; "
                         f"valid={list(GENERATORS)}")
            return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    t0 = time.time()
    for m in modal_list:
        gen = GENERATORS[m]
        sub = out_root / m
        sub.mkdir(parents=True, exist_ok=True)
        rows = []
        t_m = time.time()
        for i in range(args.n):
            rows.append(gen(args.seed * 1000 + i))
        _write(rows, sub / "train.parquet")
        summary[m] = {
            "n": args.n, "elapsed_s": round(time.time() - t_m, 3),
            "out": str(sub / "train.parquet"),
        }
        logger.info(f"[{m}] {args.n} rows -> {sub / 'train.parquet'} "
                    f"({summary[m]['elapsed_s']}s)")

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"summary -> {summary_path}  total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
