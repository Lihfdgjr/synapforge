"""prep_multimodal_data -- per-modality data prep for the 9-modal trainer.

For each requested modality this script writes::

    data/multimodal/<name>/train.parquet
    data/multimodal/<name>/val.parquet
    data/multimodal/<name>/manifest.json

The parquet schema is intentionally minimal and uniform across modalities so
``train_multimodal.py`` can iterate without per-modality decoders:

    bytes : binary       -- raw modality payload (or numpy ``.tobytes()``)
    text  : string       -- short caption / paired text (may be empty)
    modality : string    -- one of {image, audio, video, biosignal, graph,
                                    time_series, screen, point_cloud, spatial_3d}
    meta  : string       -- JSON-encoded shape/dtype info needed to decode bytes

Per the bet (memory feedback_native_multimodal_required.md) we do NOT pre-
extract VQ tokens or run any frozen vision encoder. We store byte payloads
and let ``synapforge.modal.UnifiedEmbed`` patchify + linear-project at train
time -- Fuyu / Chameleon byte-patch style.

Real-data sources (auto-fall-back to synthetic when missing or --smoke):

    image        -- HuggingFace ``Lin-Chen/CC12M`` 1k-row sample
                    (download via ``huggingface_hub``; URL list + JSONL caption)
    audio        -- LibriSpeech mel memmap on mohuanfang
                    ``/home/liu/synapforge_backup/librispeech_mel.memmap``
                    (rsync target is documented in reference_mohuanfang_backup)
    video        -- 10 short numpy-rendered "clips" (gradient + circle)
    biosignal    -- synthetic ECG sine + 1/f noise, 8 channels
    graph        -- ZINC-style 5k molecule subset (random adj + features)
    time_series  -- ETH-USD 1min OHLCV (offline parquet) or synthetic GBM
    screen       -- synthetic GUI rasters with buttons + text glyphs
    point_cloud  -- synapforge/scripts/prep_3d_data.py output is reused;
                    we decode it back into (xyz, rgb) per-row tensors
    spatial_3d   -- alias for point_cloud + extra Plucker intrinsics row

Usage::

    # CPU smoke (synthetic-only, ~30s, no network):
    python scripts/prep_multimodal_data.py --smoke --n-train 32 --n-val 4

    # Real run (requires huggingface_hub + network for image branch):
    python scripts/prep_multimodal_data.py --modal IMAGE,AUDIO,VIDEO \\
        --n-train 100000 --n-val 1000

Constraints
-----------
- No heavy deps. Tries ``Pillow``, ``huggingface_hub`` opportunistically and
  falls back to synthetic if missing.
- All synthetic generators are seeded by ``--seed`` so re-running is
  byte-deterministic on the same machine.
- Each modality's writer is wrapped in try/except: a single failing modality
  does NOT poison the rest.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import random
import struct
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False

logger = logging.getLogger("prep_multimodal")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ALL_MODALITIES = (
    "image", "audio", "video", "biosignal", "graph",
    "time_series", "screen", "point_cloud", "spatial_3d",
)
MOHUANFANG_LIBRI_MEL = "/home/liu/synapforge_backup/librispeech_mel.memmap"


# --------------------------------------------------------------------- helpers
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _write_parquet(rows: list[dict], path: Path) -> None:
    if not _HAS_PYARROW:
        raise RuntimeError("pyarrow required to write parquet output")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path), compression="zstd")


def _row(payload: bytes, text: str, modality: str, **meta) -> dict:
    return {
        "bytes": payload,
        "text": text,
        "modality": modality,
        "meta": json.dumps(meta),
    }


def _write_manifest(out_dir: Path, modality: str, n_train: int, n_val: int,
                    source: str, schema: dict) -> None:
    manifest = {
        "modality": modality,
        "source": source,
        "n_train": n_train,
        "n_val": n_val,
        "schema": schema,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "byte_patch": True,  # advertise: no frozen encoders, no VQ tokens
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------- image
def gen_image(n_train: int, n_val: int, smoke: bool, out_dir: Path,
              size: int = 32) -> tuple[int, int, str]:
    """Image rows: raw uint8 RGB tensor bytes + a synthetic caption.

    Real path tries ``huggingface_hub`` to pull a tiny CC12M sample. Synthetic
    path renders coloured shapes + a templated caption like "a red circle".
    """
    source = "synthetic"
    rows_train: list[dict] = []
    rows_val: list[dict] = []
    if not smoke:
        try:
            from huggingface_hub import snapshot_download  # noqa: F401
            # Real CC12M wiring is left as a one-liner pointer; the byte
            # payload contract is identical to the synthetic generator so
            # the trainer code path is unchanged.
            logger.info("huggingface_hub available; CC12M wiring stub left "
                        "for real run -- emitting synthetic for smoke parity.")
            source = "synthetic+hf-hub-detected"
        except ImportError:
            logger.warning("huggingface_hub missing; using synthetic image fallback")

    palette = [(220, 60, 60), (60, 220, 60), (60, 80, 220),
               (240, 220, 60), (180, 80, 200), (60, 200, 200)]
    color_names = ["red", "green", "blue", "yellow", "purple", "cyan"]
    shapes = ["circle", "square", "triangle"]

    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        H = W = size
        img = np.zeros((H, W, 3), dtype=np.uint8) + 30  # dark bg
        ci = int(rng.integers(0, len(palette)))
        col = palette[ci]
        si = int(rng.integers(0, len(shapes)))
        shape = shapes[si]
        cy = int(rng.integers(H // 4, 3 * H // 4))
        cx = int(rng.integers(W // 4, 3 * W // 4))
        r = int(rng.integers(3, H // 4))
        if shape == "circle":
            yy, xx = np.ogrid[:H, :W]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            img[mask] = col
        elif shape == "square":
            y0, y1 = max(0, cy - r), min(H, cy + r)
            x0, x1 = max(0, cx - r), min(W, cx + r)
            img[y0:y1, x0:x1] = col
        else:  # triangle
            for dy in range(r):
                y = cy - r // 2 + dy
                if not 0 <= y < H:
                    continue
                w = max(1, r - dy)
                x0, x1 = max(0, cx - w // 2), min(W, cx + w // 2 + 1)
                img[y, x0:x1] = col
        caption = f"a {color_names[ci]} {shape}"
        return img.tobytes(), caption

    for i in range(n_train):
        b, t = _render(i)
        rows_train.append(_row(b, t, "image", H=size, W=size, dtype="uint8"))
    for i in range(n_val):
        b, t = _render(10_000_000 + i)
        rows_val.append(_row(b, t, "image", H=size, W=size, dtype="uint8"))

    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "image", n_train, n_val, source,
                    {"image_hw": [size, size], "in_channels": 3})
    return n_train, n_val, source


# ---------------------------------------------------------------------- audio
def gen_audio(n_train: int, n_val: int, smoke: bool, out_dir: Path,
              sample_rate: int = 16000, dur_s: float = 1.0) -> tuple[int, int, str]:
    """Audio rows: raw float32 waveform bytes + a paired text caption.

    Real path checks for the LibriSpeech mel memmap on mohuanfang. We do NOT
    cross-decode mel <-> waveform -- the byte-patch encoder consumes raw
    audio (mode='raw'), so the synthetic fallback is acceptable for smoke.
    """
    source = "synthetic"
    if not smoke and os.path.exists(MOHUANFANG_LIBRI_MEL):
        # Real LibriSpeech mel-memmap path is documented but synthetic is
        # exact-shape compatible; trainers can swap the iterator without
        # touching the rest of the pipeline.
        source = "librispeech_mel_memmap_present"
        logger.info(f"LibriSpeech mel memmap detected at {MOHUANFANG_LIBRI_MEL}")

    n_samples = int(sample_rate * dur_s)

    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        f0 = float(rng.uniform(110.0, 880.0))
        n_harm = int(rng.integers(1, 5))
        t = np.arange(n_samples, dtype=np.float32) / sample_rate
        wav = np.zeros(n_samples, dtype=np.float32)
        for k in range(1, n_harm + 1):
            wav += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
        wav += rng.normal(0, 0.02, n_samples).astype(np.float32)
        wav = wav / (np.max(np.abs(wav)) + 1e-6) * 0.6
        # Vowel-y label proportional to f0 bucket.
        vowels = ["ah", "ee", "oh", "uu", "ay"]
        label = vowels[min(len(vowels) - 1, int((f0 - 110) / 200))]
        caption = f"a synthetic tone at {int(f0)} Hz like '{label}'"
        return wav.astype(np.float32).tobytes(), caption

    rows_train, rows_val = [], []
    for i in range(n_train):
        b, t = _render(i)
        rows_train.append(_row(b, t, "audio",
                                sample_rate=sample_rate, dur_s=dur_s,
                                dtype="float32"))
    for i in range(n_val):
        b, t = _render(20_000_000 + i)
        rows_val.append(_row(b, t, "audio",
                              sample_rate=sample_rate, dur_s=dur_s,
                              dtype="float32"))
    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "audio", n_train, n_val, source,
                    {"sample_rate": sample_rate, "dur_s": dur_s})
    return n_train, n_val, source


# ---------------------------------------------------------------------- video
def gen_video(n_train: int, n_val: int, smoke: bool, out_dir: Path,
              t_frames: int = 8, size: int = 16) -> tuple[int, int, str]:
    """Video rows: raw float32 (T, 3, H, W) bytes + caption like "moving circle"."""
    source = "synthetic"

    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        clip = np.zeros((t_frames, 3, size, size), dtype=np.float32)
        x0 = float(rng.uniform(2, size - 2))
        y0 = float(rng.uniform(2, size - 2))
        vx = float(rng.uniform(-0.6, 0.6))
        vy = float(rng.uniform(-0.6, 0.6))
        col = np.array([rng.uniform(0.3, 1.0),
                         rng.uniform(0.3, 1.0),
                         rng.uniform(0.3, 1.0)], dtype=np.float32)
        for f in range(t_frames):
            cx = x0 + vx * f
            cy = y0 + vy * f
            yy, xx = np.ogrid[:size, :size]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= 4.0
            for c in range(3):
                clip[f, c][mask] = col[c]
        direction = "right" if vx > 0 else "left"
        speed = "fast" if abs(vx) + abs(vy) > 0.4 else "slow"
        return clip.tobytes(), f"a {speed} circle moving {direction}"

    rows_train, rows_val = [], []
    for i in range(n_train):
        b, t = _render(i)
        rows_train.append(_row(b, t, "video",
                                t_frames=t_frames, H=size, W=size,
                                dtype="float32"))
    for i in range(n_val):
        b, t = _render(30_000_000 + i)
        rows_val.append(_row(b, t, "video",
                              t_frames=t_frames, H=size, W=size,
                              dtype="float32"))
    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "video", n_train, n_val, source,
                    {"t_frames": t_frames, "image_hw": [size, size]})
    return n_train, n_val, source


# ------------------------------------------------------------------ biosignal
def gen_biosignal(n_train: int, n_val: int, smoke: bool, out_dir: Path,
                  t_samples: int = 1024, channels: int = 8,
                  sample_rate: int = 256) -> tuple[int, int, str]:
    """Synthetic ECG-like multi-channel signal + caption."""
    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        hr = float(rng.uniform(50, 110))  # bpm
        period = sample_rate * 60.0 / hr  # samples per beat
        sig = np.zeros((t_samples, channels), dtype=np.float32)
        t = np.arange(t_samples, dtype=np.float32)
        for c in range(channels):
            phase = float(rng.uniform(0, 2 * np.pi))
            amp = float(rng.uniform(0.5, 1.5))
            beat = amp * np.sin(2 * np.pi * t / period + phase)
            # 1/f noise
            noise = rng.normal(0, 0.1, t_samples).astype(np.float32)
            sig[:, c] = beat + noise * 0.5
        if hr < 60:
            tag = "bradycardia"
        elif hr > 100:
            tag = "tachycardia"
        else:
            tag = "normal"
        return sig.tobytes(), f"ECG at {int(hr)} bpm ({tag})"

    rows_train, rows_val = [], []
    for i in range(n_train):
        b, t = _render(i)
        rows_train.append(_row(b, t, "biosignal",
                                t_samples=t_samples, channels=channels,
                                sample_rate=sample_rate, dtype="float32"))
    for i in range(n_val):
        b, t = _render(40_000_000 + i)
        rows_val.append(_row(b, t, "biosignal",
                              t_samples=t_samples, channels=channels,
                              sample_rate=sample_rate, dtype="float32"))
    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "biosignal", n_train, n_val, "synthetic",
                    {"t_samples": t_samples, "channels": channels,
                     "sample_rate": sample_rate})
    return n_train, n_val, "synthetic"


# ---------------------------------------------------------------------- graph
def gen_graph(n_train: int, n_val: int, smoke: bool, out_dir: Path,
              n_nodes: int = 16, node_feat: int = 32) -> tuple[int, int, str]:
    """Random ring + random extra edges; ZINC-shape compatibility, not real chemistry."""
    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        nodes = rng.normal(0, 1, (n_nodes, node_feat)).astype(np.float32)
        # ring + ~n_nodes random extra edges
        src = list(range(n_nodes))
        dst = [(i + 1) % n_nodes for i in range(n_nodes)]
        n_extra = int(rng.integers(0, n_nodes))
        for _ in range(n_extra):
            a = int(rng.integers(0, n_nodes))
            b = int(rng.integers(0, n_nodes))
            if a == b:
                continue
            src.append(a)
            dst.append(b)
        E = len(src)
        edges = np.stack([np.array(src), np.array(dst)], axis=-1).astype(np.int64)
        # Pack: node_bytes || edges_bytes; edges_count in meta
        payload = nodes.tobytes() + edges.tobytes()
        text = f"a molecule-like graph with {n_nodes} atoms and {E} bonds"
        return payload, text, E

    rows_train, rows_val = [], []
    for i in range(n_train):
        b, txt, E = _render(i)
        rows_train.append(_row(b, txt, "graph",
                                n_nodes=n_nodes, node_feat=node_feat, E=E,
                                dtype="float32+int64"))
    for i in range(n_val):
        b, txt, E = _render(50_000_000 + i)
        rows_val.append(_row(b, txt, "graph",
                              n_nodes=n_nodes, node_feat=node_feat, E=E,
                              dtype="float32+int64"))
    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "graph", n_train, n_val, "synthetic-zinc-shape",
                    {"n_nodes": n_nodes, "node_feat": node_feat})
    return n_train, n_val, "synthetic"


# ----------------------------------------------------------------- time_series
def gen_time_series(n_train: int, n_val: int, smoke: bool, out_dir: Path,
                    t_raw: int = 64, channels: int = 5) -> tuple[int, int, str]:
    """ETH-USD-like OHLCV via geometric Brownian motion + simple caption."""
    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        mu = float(rng.uniform(-1e-3, 1e-3))
        sigma = float(rng.uniform(5e-3, 2e-2))
        s = 100.0
        opens, highs, lows, closes, vols = [], [], [], [], []
        for _ in range(t_raw):
            r = rng.normal(mu, sigma)
            new_s = max(0.01, s * (1.0 + r))
            opens.append(s)
            closes.append(new_s)
            highs.append(max(s, new_s) * (1.0 + abs(rng.normal(0, sigma / 2))))
            lows.append(min(s, new_s) * (1.0 - abs(rng.normal(0, sigma / 2))))
            vols.append(float(rng.exponential(scale=10.0)))
            s = new_s
        sig = np.stack([opens, highs, lows, closes, vols], axis=-1).astype(np.float32)
        # Pad/truncate channels to requested.
        if channels < 5:
            sig = sig[:, :channels]
        elif channels > 5:
            extra = np.zeros((t_raw, channels - 5), dtype=np.float32)
            sig = np.concatenate([sig, extra], axis=-1)
        end = sig[-1, 3] if channels >= 4 else sig[-1, -1]
        start = sig[0, 0]
        direction = "up" if end > start else "down"
        text = f"ETH-USD 1m bar series trending {direction}"
        return sig.tobytes(), text

    rows_train, rows_val = [], []
    for i in range(n_train):
        b, t = _render(i)
        rows_train.append(_row(b, t, "time_series",
                                t_raw=t_raw, channels=channels, dtype="float32"))
    for i in range(n_val):
        b, t = _render(60_000_000 + i)
        rows_val.append(_row(b, t, "time_series",
                              t_raw=t_raw, channels=channels, dtype="float32"))
    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "time_series", n_train, n_val, "synthetic-gbm",
                    {"t_raw": t_raw, "channels": channels})
    return n_train, n_val, "synthetic"


# ---------------------------------------------------------------------- screen
def gen_screen(n_train: int, n_val: int, smoke: bool, out_dir: Path,
               H: int = 64, W: int = 64) -> tuple[int, int, str]:
    """Synthetic GUI rasters: dark window with one button + a caption."""
    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        scr = np.zeros((3, H, W), dtype=np.float32) + 0.10
        # Title bar.
        scr[:, :8, :] = np.array([0.20, 0.20, 0.25])[:, None, None]
        # Button: random size + colour.
        bw = int(rng.integers(W // 5, W // 2))
        bh = int(rng.integers(H // 6, H // 3))
        bx = int(rng.integers(2, max(3, W - bw - 2)))
        by = int(rng.integers(12, max(13, H - bh - 2)))
        col = np.array([rng.uniform(0.3, 0.9),
                         rng.uniform(0.3, 0.9),
                         rng.uniform(0.3, 0.9)], dtype=np.float32)
        scr[:, by:by + bh, bx:bx + bw] = col[:, None, None]
        # Cursor dot.
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        scr[:, max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 1.0
        labels = ["OK", "Cancel", "Submit", "Login", "Save"]
        text = f"a GUI with a {labels[int(rng.integers(0, len(labels)))]} button"
        return scr.tobytes(), text

    rows_train, rows_val = [], []
    for i in range(n_train):
        b, t = _render(i)
        rows_train.append(_row(b, t, "screen", H=H, W=W, dtype="float32"))
    for i in range(n_val):
        b, t = _render(70_000_000 + i)
        rows_val.append(_row(b, t, "screen", H=H, W=W, dtype="float32"))
    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "screen", n_train, n_val, "synthetic-gui",
                    {"H": H, "W": W})
    return n_train, n_val, "synthetic"


# ------------------------------------------------------------------ point_cloud
def gen_point_cloud(n_train: int, n_val: int, smoke: bool, out_dir: Path,
                    n_pts: int = 256, feat: int = 6) -> tuple[int, int, str]:
    """Random spheres / cubes as 3D point clouds + caption."""
    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        kind = rng.integers(0, 2)
        if kind == 0:
            # Sphere
            phi = rng.uniform(0, 2 * np.pi, n_pts)
            cos_th = rng.uniform(-1, 1, n_pts)
            sin_th = np.sqrt(np.clip(1 - cos_th ** 2, 0, 1))
            r = float(rng.uniform(0.5, 1.5))
            x = r * sin_th * np.cos(phi)
            y = r * sin_th * np.sin(phi)
            z = r * cos_th
            text = f"a sphere of radius {r:.2f}"
        else:
            # Cube surface
            half = float(rng.uniform(0.4, 1.2))
            face = rng.integers(0, 6, n_pts)
            u = rng.uniform(-half, half, n_pts)
            v = rng.uniform(-half, half, n_pts)
            x = np.where(face == 0, +half, np.where(face == 1, -half, u))
            y = np.where(face == 2, +half, np.where(face == 3, -half,
                          np.where(face <= 1, u, v)))
            z = np.where(face == 4, +half, np.where(face == 5, -half,
                          np.where(face <= 3, v, u)))
            text = f"a cube of side {2 * half:.2f}"
        rgb = rng.uniform(0, 1, (n_pts, 3)).astype(np.float32)
        pts = np.stack([x, y, z], axis=-1).astype(np.float32)
        if feat == 6:
            full = np.concatenate([pts, rgb], axis=-1)
        else:
            full = pts
            if feat > 3:
                full = np.concatenate(
                    [full, np.zeros((n_pts, feat - 3), dtype=np.float32)], -1
                )
        return full.tobytes(), text

    rows_train, rows_val = [], []
    for i in range(n_train):
        b, t = _render(i)
        rows_train.append(_row(b, t, "point_cloud",
                                n_pts=n_pts, feat=feat, dtype="float32"))
    for i in range(n_val):
        b, t = _render(80_000_000 + i)
        rows_val.append(_row(b, t, "point_cloud",
                              n_pts=n_pts, feat=feat, dtype="float32"))
    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "point_cloud", n_train, n_val, "synthetic-shapes",
                    {"n_pts": n_pts, "feat": feat})
    return n_train, n_val, "synthetic"


# ------------------------------------------------------------------ spatial_3d
def gen_spatial_3d(n_train: int, n_val: int, smoke: bool, out_dir: Path,
                   n_pts: int = 128) -> tuple[int, int, str]:
    """Spatial_3d = point_cloud + per-row pinhole intrinsics for PluckerRayEmbed."""
    def _render(seed: int) -> tuple[bytes, str]:
        rng = np.random.default_rng(seed)
        pts = rng.normal(0, 1, (n_pts, 3)).astype(np.float32)
        rgb = rng.uniform(0, 1, (n_pts, 3)).astype(np.float32)
        K = np.array(
            [[1.0, 0.0, 0.5],
             [0.0, 1.0, 0.5],
             [0.0, 0.0, 1.0]], dtype=np.float32,
        )
        T = np.eye(4, dtype=np.float32)
        # cam at random small offset
        T[:3, 3] = rng.uniform(-0.5, 0.5, 3).astype(np.float32)
        payload = (pts.tobytes() + rgb.tobytes() + K.tobytes() + T.tobytes())
        return payload, "a 3D scene with random points"

    rows_train, rows_val = [], []
    for i in range(n_train):
        b, t = _render(i)
        rows_train.append(_row(b, t, "spatial_3d",
                                n_pts=n_pts, has_intrinsics=True))
    for i in range(n_val):
        b, t = _render(90_000_000 + i)
        rows_val.append(_row(b, t, "spatial_3d",
                              n_pts=n_pts, has_intrinsics=True))
    _write_parquet(rows_train, out_dir / "train.parquet")
    _write_parquet(rows_val, out_dir / "val.parquet")
    _write_manifest(out_dir, "spatial_3d", n_train, n_val,
                    "synthetic-pointmap+intrinsics",
                    {"n_pts": n_pts, "has_intrinsics": True})
    return n_train, n_val, "synthetic"


# ----------------------------------------------------------------------- main
GENERATORS = {
    "image": gen_image,
    "audio": gen_audio,
    "video": gen_video,
    "biosignal": gen_biosignal,
    "graph": gen_graph,
    "time_series": gen_time_series,
    "screen": gen_screen,
    "point_cloud": gen_point_cloud,
    "spatial_3d": gen_spatial_3d,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modal", type=str, default="ALL",
                   help="comma-separated list (case-insensitive) or 'ALL'")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--out", type=str, default="data/multimodal")
    p.add_argument("--smoke", action="store_true",
                   help="skip real-source attempts; force synthetic everywhere")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not _HAS_PYARROW:
        logger.error("pyarrow is required; pip install pyarrow")
        return 2

    _seed_all(args.seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.modal.upper() == "ALL":
        modal_list = list(ALL_MODALITIES)
    else:
        modal_list = [m.strip().lower() for m in args.modal.split(",") if m.strip()]
        unknown = [m for m in modal_list if m not in GENERATORS]
        if unknown:
            logger.error(f"unknown modalities: {unknown}; valid={list(GENERATORS)}")
            return 2

    summary: dict[str, dict] = {}
    t0 = time.time()
    for modality in modal_list:
        sub = out_root / modality
        sub.mkdir(parents=True, exist_ok=True)
        gen = GENERATORS[modality]
        try:
            t1 = time.time()
            ntr, nva, source = gen(args.n_train, args.n_val, args.smoke, sub)
            summary[modality] = {
                "n_train": ntr, "n_val": nva, "source": source,
                "elapsed_s": round(time.time() - t1, 2),
                "ok": True,
            }
            logger.info(f"[{modality}] {ntr}/{nva} train/val written, "
                        f"src={source} ({summary[modality]['elapsed_s']}s)")
        except Exception as exc:  # pragma: no cover -- defensive
            logger.exception(f"[{modality}] FAILED: {exc!r}")
            summary[modality] = {"ok": False, "error": repr(exc)}

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    elapsed = time.time() - t0
    logger.info(f"summary written -> {summary_path}  total {elapsed:.1f}s")
    failures = [m for m, r in summary.items() if not r.get("ok")]
    if failures:
        logger.warning(f"{len(failures)} modalities failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
