"""synth_audio_pretrain.py -- generate synthetic audio mel-spectrogram patches.

Companion to ``synth_chinese_pretrain.py`` and ``synth_image_pretrain.py``
for phase 2 multimodal training. Builds 50K (caption, mel_bytes) pairs from
three synthetic 1-second audio classes:

    1. Sine wave  -- random fundamental in 220-880 Hz (A3 .. A5 musical range)
    2. Noise      -- white Gaussian noise at random amplitude
    3. Chirp      -- linear frequency sweep (low -> high or high -> low)

The byte-patch convention follows the rest of SynapForge multimodal data:
the mel-spectrogram is computed in pure numpy (FFT + mel filter bank) and
quantized to uint8 (256 levels). Default geometry::

    sample_rate = 16000 Hz
    duration    = 1.0 s         -> 16000 samples
    n_fft       = 400           (25 ms window)
    hop_length  = 160           (10 ms hop) -> 100 mel frames
    n_mels      = 80            -> 80 * 100 = 8000 bytes per row

The model treats each row's ``mel_bytes`` as an 8000-byte stream patched
into the token sequence -- byte-patch native, NOT a frozen audio encoder
+ projection (per ``feedback_native_multimodal_required``).

Determinism: the rows depend ONLY on ``--seed``. Re-running with the same
seed produces byte-identical output.

Usage::

    python scripts/synth_audio_pretrain.py --smoke         # n=10 sanity
    python scripts/synth_audio_pretrain.py \\
        --out  /workspace/data/pretrain/synth_audio/train.parquet \\
        --n    50000 \\
        --seed 42

Constraints:
    - numpy + stdlib only. ``librosa`` is NOT a hard dep.
    - ``pyarrow`` needed for parquet output; falls back to JSONL if missing.
    - CPU only, no GPU.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


# --- audio + mel constants -------------------------------------------------
SAMPLE_RATE = 16000
DURATION_S = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)  # 16000
N_FFT = 400          # 25 ms window @ 16 kHz
HOP_LENGTH = 160     # 10 ms hop -> 100 frames per second
N_MELS = 80
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 100
MEL_BYTES_PER_ROW = N_MELS * N_FRAMES  # 80 * 100 = 8000

# Frequency sweep range for chirp + sine fundamental.
F_LOW = 220.0   # A3
F_HIGH = 880.0  # A5

# 3 synthetic audio classes.
CLASS_SINE = "sine"
CLASS_NOISE = "noise"
CLASS_CHIRP = "chirp"
CLASS_LIST = (CLASS_SINE, CLASS_NOISE, CLASS_CHIRP)


# --- mel filter bank (pure numpy) -----------------------------------------
def hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney mel scale: ``mel = 2595 * log10(1 + f / 700)``."""
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filterbank(
    sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
    fmin: float = 0.0, fmax: Optional[float] = None,
) -> np.ndarray:
    """Triangular mel filter bank, shape (n_mels, n_fft//2 + 1).

    Identical math to ``librosa.filters.mel`` with ``htk=False, norm=None``.
    Cached at module level via the ``_FILTERBANK`` lazy global below; this
    function exists for testability + smoke runs.
    """
    if fmax is None:
        fmax = sr / 2.0
    n_freq_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freq_bins)
    mel_min = hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    mel_max = hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freq_bins), dtype=np.float32)
    for m in range(n_mels):
        lo, ctr, hi = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        # Up-slope from lo -> ctr
        up = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        # Down-slope from ctr -> hi
        down = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        triangle = np.maximum(0.0, np.minimum(up, down))
        fb[m] = triangle.astype(np.float32)
    return fb


_FILTERBANK: Optional[np.ndarray] = None


def _get_filterbank() -> np.ndarray:
    global _FILTERBANK
    if _FILTERBANK is None:
        _FILTERBANK = build_mel_filterbank()
    return _FILTERBANK


def _hann_window(n: int) -> np.ndarray:
    """Periodic Hann window (matches librosa default)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n).astype(np.float64)


def stft_magnitude(
    waveform: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Center-padded STFT magnitude. Shape (n_fft//2+1, n_frames).

    Uses ``np.fft.rfft`` per-frame -- vectorised enough for 16K samples
    (the per-row cost is ~0.4 ms on a 2020 laptop CPU).
    """
    pad = n_fft // 2
    padded = np.pad(waveform, (pad, pad), mode="reflect")
    win = _hann_window(n_fft).astype(np.float32)
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    n_freq = n_fft // 2 + 1
    out = np.empty((n_freq, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start:start + n_fft] * win
        spec = np.fft.rfft(frame, n=n_fft)
        out[:, i] = np.abs(spec).astype(np.float32)
    return out


def mel_spectrogram(
    waveform: np.ndarray, n_mels: int = N_MELS,
    n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute log-mel-spectrogram. Shape ``(n_mels, N_FRAMES)``.

    Pipeline: |STFT|^2 -> mel filterbank -> log10(eps + .) -> caller quant.

    Truncation: center-padded STFT yields ``1 + (n_samples + n_fft) //
    hop_length`` frames, i.e. one extra boundary frame. We truncate to
    ``N_FRAMES`` (= ``N_SAMPLES // hop_length`` = 100) so the byte-patch
    consumer downstream gets a fixed 8000-byte contract per row.
    """
    spec_mag = stft_magnitude(waveform, n_fft=n_fft, hop_length=hop_length)
    power = spec_mag ** 2
    fb = _get_filterbank()
    mel_power = fb @ power  # (n_mels, n_frames)
    # Slight floor for log; matches librosa's amin=1e-10.
    log_mel = np.log10(np.maximum(mel_power, 1e-10))
    # Truncate the trailing center-pad boundary frame: librosa-equivalent
    # STFT emits 101 frames for a 16 K-sample input; we want exactly 100.
    log_mel = log_mel[:, :N_FRAMES]
    return log_mel.astype(np.float32)


def quantize_mel_to_uint8(log_mel: np.ndarray) -> np.ndarray:
    """Quantize a log-mel spectrogram to uint8 256-level bytes.

    Per-sample min/max normalisation: each row maps its own ``log_mel``
    range to [0, 255]. The trainer's byte-patch decoder is range-agnostic
    (it learns the embedding), so this is fine and avoids needing a
    global statistics calibration step.
    """
    lo = float(log_mel.min())
    hi = float(log_mel.max())
    if hi - lo < 1e-9:
        return np.zeros_like(log_mel, dtype=np.uint8)
    norm = (log_mel - lo) / (hi - lo)
    return np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)


# --- audio synthesisers ---------------------------------------------------
def synth_sine(rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Pure sine at random freq in [F_LOW, F_HIGH]. Tiny noise floor."""
    f0 = float(rng.uniform(F_LOW, F_HIGH))
    t = np.arange(N_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    wav = np.sin(2.0 * np.pi * f0 * t).astype(np.float32)
    wav += rng.normal(0.0, 0.005, N_SAMPLES).astype(np.float32)
    wav *= 0.6  # leave headroom
    meta = {"klass": CLASS_SINE, "freq_hz": f0,
            "f_start_hz": f0, "f_end_hz": f0,
            "noise_amp": 0.0}
    return wav.astype(np.float32), meta


def synth_noise(rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """White Gaussian noise at random amplitude."""
    amp = float(rng.uniform(0.1, 0.7))
    wav = (rng.normal(0.0, amp, N_SAMPLES)).astype(np.float32)
    meta = {"klass": CLASS_NOISE, "freq_hz": 0.0,
            "f_start_hz": 0.0, "f_end_hz": 0.0,
            "noise_amp": amp}
    return wav, meta


def synth_chirp(rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Linear chirp from f0 -> f1, with light noise.

    Random direction (low->high vs high->low) + random endpoints in the
    musical range [F_LOW, F_HIGH].
    """
    a = float(rng.uniform(F_LOW, F_HIGH))
    b = float(rng.uniform(F_LOW, F_HIGH))
    if abs(a - b) < 50.0:
        b = a + 200.0 if a < (F_LOW + F_HIGH) / 2 else a - 200.0
    f0, f1 = a, b
    t = np.arange(N_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    # Linear chirp: phase = 2*pi*(f0 * t + (f1-f0)/(2*T) * t^2)
    T = float(N_SAMPLES) / SAMPLE_RATE
    inst_phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t * t)
    wav = np.sin(inst_phase).astype(np.float32)
    wav += rng.normal(0.0, 0.01, N_SAMPLES).astype(np.float32)
    wav *= 0.6
    meta = {"klass": CLASS_CHIRP, "freq_hz": (f0 + f1) / 2.0,
            "f_start_hz": f0, "f_end_hz": f1,
            "noise_amp": 0.0}
    return wav.astype(np.float32), meta


CLASS_SYNTH = {
    CLASS_SINE: synth_sine,
    CLASS_NOISE: synth_noise,
    CLASS_CHIRP: synth_chirp,
}


# --- caption builder -------------------------------------------------------
def build_caption(meta: dict) -> str:
    """Plain-English caption mentioning the audio property.

    Property keywords (always one of: ``Hz``, ``noise``, ``chirp``) drive
    ``test_caption_mentions_audio_property``.
    """
    klass = meta["klass"]
    if klass == CLASS_SINE:
        return f"a sine wave at {int(round(meta['freq_hz']))} Hz"
    if klass == CLASS_NOISE:
        return f"white noise at amplitude {meta['noise_amp']:.2f}"
    if klass == CLASS_CHIRP:
        f0 = int(round(meta["f_start_hz"]))
        f1 = int(round(meta["f_end_hz"]))
        direction = "low to high" if f1 > f0 else "high to low"
        return f"a chirp from {direction} ({f0} to {f1} Hz)"
    raise ValueError(f"unknown audio class {klass!r}")


# --- one-row generator ----------------------------------------------------
def generate_row(seed_int: int) -> dict:
    """Build one (caption, mel_bytes) row. Pure-deterministic on seed_int.

    Returns dict with keys::

        text       str   -- caption (also the supervision text)
        caption    str   -- duplicate of ``text`` for clarity downstream
        mel_bytes  bytes -- exactly ``MEL_BYTES_PER_ROW`` bytes (8000)
        freq_hz    float -- provenance (0.0 for noise rows)
        klass      str   -- one of CLASS_LIST
    """
    rng = np.random.default_rng(seed_int)
    klass = CLASS_LIST[int(rng.integers(0, len(CLASS_LIST)))]
    wav, meta = CLASS_SYNTH[klass](rng)
    log_mel = mel_spectrogram(wav)
    quant = quantize_mel_to_uint8(log_mel)
    mel_bytes = quant.tobytes()
    if len(mel_bytes) != MEL_BYTES_PER_ROW:  # invariant check
        raise RuntimeError(
            f"mel_bytes len {len(mel_bytes)} != expected {MEL_BYTES_PER_ROW}"
        )
    caption = build_caption(meta)
    return {
        "text": caption,
        "caption": caption,
        "mel_bytes": mel_bytes,
        "freq_hz": float(meta["freq_hz"]),
        "klass": klass,
    }


# --- writer ---------------------------------------------------------------
def write_audio_parquet(rows: Sequence[dict], out_path: str) -> int:
    """Write the rows to parquet. JSONL fallback when pyarrow missing."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not _HAVE_ARROW:
        jp = Path(out_path).with_suffix(".jsonl")
        with open(jp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "text": r["text"],
                    "caption": r["caption"],
                    "mel_bytes_b64": base64.b64encode(r["mel_bytes"]).decode("ascii"),
                    "freq_hz": r["freq_hz"],
                    "klass": r["klass"],
                }, ensure_ascii=False) + "\n")
        print(f"[synth_audio] pyarrow missing; JSONL fallback -> {jp}",
              file=sys.stderr)
        return len(rows)

    table = pa.table({
        "text": [r["text"] for r in rows],
        "caption": [r["caption"] for r in rows],
        "mel_bytes": [r["mel_bytes"] for r in rows],
        "freq_hz": [r["freq_hz"] for r in rows],
        "klass": [r["klass"] for r in rows],
    })
    pq.write_table(table, out_path, compression="zstd")
    return len(rows)


def write_manifest(out_path: str, n: int, seed: int) -> None:
    mfile = str(out_path) + ".manifest.json"
    Path(mfile).parent.mkdir(parents=True, exist_ok=True)
    with open(mfile, "w", encoding="utf-8") as f:
        json.dump({
            "kind": "synth_audio_mel",
            "rows": n,
            "seed": seed,
            "sample_rate": SAMPLE_RATE,
            "duration_s": DURATION_S,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "n_frames": N_FRAMES,
            "mel_bytes_per_row": MEL_BYTES_PER_ROW,
            "classes": list(CLASS_LIST),
            "freq_range_hz": [F_LOW, F_HIGH],
            "warning": "synthetic mel patches; not a substitute for real audio",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    print(f"[synth_audio] manifest -> {mfile}")


# --- CLI -----------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/multimodal/synth_audio/train.parquet")
    ap.add_argument("--n", type=int, default=50000,
                    help="Rows to generate (default 50K).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="Override --n to 10 for fast end-to-end check.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    n = 10 if args.smoke else args.n

    t0 = time.time()
    rows: list = []
    for i in range(n):
        # Per-row seed mixes the global seed with the row index so the row
        # set is order-stable AND each row independently reproducible.
        row_seed = (args.seed * 1_000_003 + i) & 0xFFFFFFFF
        rows.append(generate_row(int(row_seed)))
        if (i + 1) % 5000 == 0:
            print(f"[synth_audio] generated {i + 1:,}/{n:,} "
                  f"({time.time() - t0:.1f}s)")

    n_written = write_audio_parquet(rows, args.out)
    write_manifest(args.out, n_written, args.seed)
    print(f"[synth_audio] wrote {n_written:,} rows -> {args.out} "
          f"({time.time() - t0:.1f}s, {MEL_BYTES_PER_ROW} bytes/row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
