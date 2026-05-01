"""synth_image_pretrain.py — generate synthetic (caption, image_patches) pairs.

Phase-2 multimodal pretrain demo-fallback for the byte-patch image stream
(see ``feedback_native_multimodal_required.md``: images are NOT routed
through a vision encoder; raw bytes are patched into the token stream).

Each row pairs a Qwen-tokenized caption with a 768-byte image-patch
sequence:

    image       32x32 RGB synthetic (gradient + noise + one shape)
    patches     4x4 grid of 4x4 RGB downsampled patches
                = 16 patches * (4*4*3 = 48 bytes) = 768 bytes per image
    caption     "<color> <shape>[ on <bg>]" tokenized with Qwen 2.5 0.5B
    text        list[int] of token ids; image_patches list[int 0-255]

The downsampled patch (one 4x4 averaged tile per image quadrant strip)
is what the byte-patch trainer sees -- the raw 3072-byte image is too
expensive to drop into the token stream every sample. We keep it
deterministic given ``--seed`` so retraining + ablations reproduce.

Determinism: ``np.random.RandomState(seed)`` per row (not module-level)
-- adding rows or reordering generators leaves earlier rows unchanged.

Constraints (matches T3.0 / T3.1 patterns):
    - Only PIL + numpy + pandas + pyarrow. No torch / live HuggingFace.
    - Tokenizer is loaded best-effort from
      ``Qwen/Qwen2.5-0.5B`` (or local cache); on offline / failure we
      fall back to a deterministic stub returning [1,2,3]. Tests pin the
      stub path so CI never needs network.

Usage::

    python scripts/synth_image_pretrain.py \\
        --output /workspace/data/synth_image_50K.parquet \\
        --n 50000 --seed 42

    python scripts/synth_image_pretrain.py --smoke    # 10 rows
    python scripts/synth_image_pretrain.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False

try:
    from PIL import Image, ImageDraw
    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False


# --- image geometry ---------------------------------------------------------
IMG_SIZE = 32           # 32x32 RGB image
PATCH_GRID = 4          # 4x4 patch grid -> 16 patches
PATCH_PX = 4            # each downsampled patch is 4x4 RGB pixels
N_PATCHES = PATCH_GRID * PATCH_GRID                      # 16
PATCH_BYTES = PATCH_PX * PATCH_PX * 3                    # 48
PATCH_SEQ_BYTES = N_PATCHES * PATCH_BYTES                # 768

# --- vocabulary -------------------------------------------------------------
COLORS = {
    "red":     (220, 40, 40),
    "green":   (40, 200, 80),
    "blue":    (50, 90, 230),
    "yellow":  (240, 220, 60),
    "purple":  (170, 70, 200),
    "cyan":    (60, 200, 220),
    "orange":  (240, 140, 50),
    "white":   (245, 245, 245),
}
COLOR_NAMES = list(COLORS.keys())

BG_COLORS = {
    "black":   (15, 15, 20),
    "navy":    (10, 20, 80),
    "olive":   (60, 80, 30),
    "maroon":  (90, 25, 35),
    "teal":    (10, 90, 95),
    "gray":    (90, 90, 100),
}
BG_NAMES = list(BG_COLORS.keys())

SHAPES = ("circle", "rectangle", "line", "triangle")


# --- tokenizer (best-effort, with offline stub) -----------------------------
class _StubTokenizer:
    """Deterministic offline stand-in for Qwen tokenizer.

    Returns ``[1, 2, 3]`` for any text input. Matches the contract the
    task spec mandates so unit tests can pin this behaviour without
    needing transformers / HuggingFace network.
    """

    name_or_path = "stub"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [1, 2, 3]


def _load_tokenizer(name: str = "Qwen/Qwen2.5-0.5B"):  # pragma: no cover -- network
    """Try to load the Qwen tokenizer; fall back to ``_StubTokenizer``.

    Order: local rental cache -> HuggingFace name. On any failure we
    return ``_StubTokenizer`` and warn on stderr -- pretrain rows still
    flow, just with placeholder text ids that the rental run will
    re-tokenize at training time if needed.
    """
    candidates = [
        "/workspace/teachers/qwen2.5-0.5b",
        name,
    ]
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"[synth_img] transformers unavailable ({exc!r}); using stub",
              file=sys.stderr)
        return _StubTokenizer()
    for path in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            print(f"[synth_img] loaded Qwen tokenizer from {path!r}",
                  file=sys.stderr)
            return tok
        except Exception as exc:
            print(f"[synth_img] tokenizer candidate {path!r} failed: {exc!r}",
                  file=sys.stderr)
    print("[synth_img] no Qwen tokenizer reachable; using stub",
          file=sys.stderr)
    return _StubTokenizer()


# --- image generation -------------------------------------------------------
def _gradient_background(rng: np.random.RandomState,
                         bg: Tuple[int, int, int]) -> np.ndarray:
    """Vertical or radial gradient seeded by ``bg``."""
    H = W = IMG_SIZE
    img = np.empty((H, W, 3), dtype=np.uint8)
    direction = rng.randint(0, 3)  # 0=vert, 1=horiz, 2=diag
    base = np.asarray(bg, dtype=np.float32)
    accent = np.clip(base + rng.randint(-30, 31, size=3), 0, 255).astype(np.float32)
    for y in range(H):
        for x in range(W):
            if direction == 0:
                t = y / max(H - 1, 1)
            elif direction == 1:
                t = x / max(W - 1, 1)
            else:
                t = (x + y) / max(2 * (H - 1), 1)
            img[y, x] = (base * (1 - t) + accent * t).astype(np.uint8)
    # Light noise so gradient isn't perfectly linear.
    noise = rng.randint(-8, 9, size=(H, W, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _draw_shape(img: np.ndarray, rng: np.random.RandomState,
                shape: str, color: Tuple[int, int, int]) -> None:
    """Mutate ``img`` in-place, drawing one shape with ``color``."""
    if not _HAVE_PIL:  # pragma: no cover
        # Fallback: stamp a coloured patch in the centre.
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        r = max(3, h // 5)
        img[cy - r: cy + r, cx - r: cx + r] = color
        return
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    H = W = IMG_SIZE
    # Random centre + size; inset by 2 to keep shape visible.
    cx = int(rng.randint(W // 4, 3 * W // 4))
    cy = int(rng.randint(H // 4, 3 * H // 4))
    r = int(rng.randint(4, max(5, H // 4)))
    if shape == "circle":
        bbox = (cx - r, cy - r, cx + r, cy + r)
        draw.ellipse(bbox, fill=color)
    elif shape == "rectangle":
        x0 = max(0, cx - r)
        y0 = max(0, cy - r)
        x1 = min(W - 1, cx + r)
        y1 = min(H - 1, cy + r)
        draw.rectangle((x0, y0, x1, y1), fill=color)
    elif shape == "line":
        x0 = int(rng.randint(0, W - 1))
        y0 = int(rng.randint(0, H - 1))
        x1 = int(rng.randint(0, W - 1))
        y1 = int(rng.randint(0, H - 1))
        draw.line((x0, y0, x1, y1), fill=color, width=max(1, r // 3))
    else:  # triangle
        pts = [
            (cx, max(0, cy - r)),
            (max(0, cx - r), min(H - 1, cy + r)),
            (min(W - 1, cx + r), min(H - 1, cy + r)),
        ]
        draw.polygon(pts, fill=color)
    img[:] = np.asarray(pil, dtype=np.uint8)


def gen_image(seed: int) -> Tuple[np.ndarray, str, dict]:
    """Generate one (image, caption, meta) tuple.

    Image is HxWx3 uint8. Caption mentions both shape and primary color
    plus the background. Meta carries the discrete labels for tests.
    """
    rng = np.random.RandomState(seed)
    color_name = COLOR_NAMES[rng.randint(0, len(COLOR_NAMES))]
    shape = SHAPES[rng.randint(0, len(SHAPES))]
    bg_name = BG_NAMES[rng.randint(0, len(BG_NAMES))]
    color = COLORS[color_name]
    bg = BG_COLORS[bg_name]
    img = _gradient_background(rng, bg)
    _draw_shape(img, rng, shape, color)
    caption = f"a {color_name} {shape} on a {bg_name} background"
    return img, caption, {
        "shape": shape,
        "color": color_name,
        "bg": bg_name,
    }


def image_to_patches(img: np.ndarray) -> bytes:
    """Downsample ``img`` to a 4x4 grid of 4x4 RGB tiles -> 768 bytes.

    Algorithm: split the 32x32 image into a 4x4 grid of 8x8 cells; mean-
    pool each cell to a 4x4 tile by taking 4 sub-cells of 2x2 pixels and
    averaging each. This preserves spatial structure (the trainer can
    reconstruct the layout from the byte order) while staying byte-cheap.
    """
    if img.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"expected ({IMG_SIZE},{IMG_SIZE},3); got {img.shape}")
    cell = IMG_SIZE // PATCH_GRID                   # 8
    sub = cell // PATCH_PX                          # 2
    out = np.empty((N_PATCHES, PATCH_BYTES), dtype=np.uint8)
    p = 0
    for gy in range(PATCH_GRID):
        for gx in range(PATCH_GRID):
            block = img[gy * cell:(gy + 1) * cell,
                        gx * cell:(gx + 1) * cell]            # 8x8x3
            tile = np.empty((PATCH_PX, PATCH_PX, 3), dtype=np.uint8)
            for ty in range(PATCH_PX):
                for tx in range(PATCH_PX):
                    sub_block = block[ty * sub:(ty + 1) * sub,
                                      tx * sub:(tx + 1) * sub]   # 2x2x3
                    tile[ty, tx] = sub_block.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            out[p] = tile.reshape(-1)
            p += 1
    raw = out.reshape(-1).tobytes()
    if len(raw) != PATCH_SEQ_BYTES:  # pragma: no cover -- math invariant
        raise AssertionError(f"patch byte count mismatch: {len(raw)}")
    return raw


# --- row generation ---------------------------------------------------------
def gen_row(seed: int, tokenizer) -> dict:
    """Build one parquet row: text ids + image patches + caption + meta."""
    img, caption, meta = gen_image(seed)
    patch_bytes = image_to_patches(img)
    ids = tokenizer.encode(caption, add_special_tokens=False)
    return {
        "text": [int(x) for x in ids],
        "image_patches": list(patch_bytes),  # list<uint8> 0-255
        "caption": caption,
        "shape": meta["shape"],
        "color": meta["color"],
        "bg": meta["bg"],
    }


# --- write side -------------------------------------------------------------
def _write_parquet(rows: Sequence[dict], out_path: str) -> int:
    if not _HAVE_ARROW:  # pragma: no cover
        raise RuntimeError("pyarrow + pandas required to write parquet")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "text": [r["text"] for r in rows],
        "image_patches": [r["image_patches"] for r in rows],
        "caption": [r["caption"] for r in rows],
        "shape": [r["shape"] for r in rows],
        "color": [r["color"] for r in rows],
        "bg": [r["bg"] for r in rows],
    })
    pq.write_table(pa.Table.from_pandas(df), out_path, compression="zstd")
    return len(df)


def _write_manifest(out_path: str, n: int, seed: int,
                    tokenizer_src: str) -> None:
    mfile = str(out_path) + ".manifest.json"
    Path(mfile).parent.mkdir(parents=True, exist_ok=True)
    with open(mfile, "w", encoding="utf-8") as f:
        json.dump({
            "kind": "synth_image_pretrain",
            "rows": n,
            "seed": seed,
            "image_size": IMG_SIZE,
            "patch_grid": PATCH_GRID,
            "patch_px": PATCH_PX,
            "patch_bytes_per_image": PATCH_SEQ_BYTES,
            "tokenizer": tokenizer_src,
            "shapes": list(SHAPES),
            "colors": COLOR_NAMES,
            "bg_colors": BG_NAMES,
            "warning": "synthetic only; phase-2 demo-fallback fodder",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    print(f"[synth_img] manifest -> {mfile}")


# --- main -------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=50000,
                    help="Number of (caption, image) pairs (default 50000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output",
                    default="/workspace/data/synth_image_50K.parquet",
                    help="Output parquet path")
    ap.add_argument("--smoke", action="store_true",
                    help="Override --n to 10 for fast end-to-end check")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B",
                    help="HF tokenizer id; falls back to stub on failure")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        args.n = 10
    if not _HAVE_ARROW:
        print("[synth_img] FATAL: pyarrow + pandas required", file=sys.stderr)
        return 2

    tokenizer = _load_tokenizer(args.tokenizer)
    tok_src = getattr(tokenizer, "name_or_path", "stub") or "stub"

    t0 = time.time()
    rows: List[dict] = []
    for i in range(args.n):
        # Per-row seed = (args.seed * 1_000_003 + i): determinism + low
        # collision so adding rows leaves earlier ones byte-stable.
        rows.append(gen_row(args.seed * 1_000_003 + i, tokenizer))
        if (i + 1) % 5000 == 0:
            print(f"[synth_img] {i + 1:,}/{args.n:,} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    n = _write_parquet(rows, args.output)
    print(f"[synth_img] wrote {n:,} rows -> {args.output} "
          f"({PATCH_SEQ_BYTES} bytes/img, tokenizer={tok_src})", flush=True)
    _write_manifest(args.output, n, args.seed, tok_src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
