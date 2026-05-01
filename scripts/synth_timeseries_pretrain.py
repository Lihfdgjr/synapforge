"""synth_timeseries_pretrain.py — generate synthetic time-series sequences (T3.4).

Demo / pretrain admixture for the time-series modality. Generates plausible
1-D signals across three domains, quantized to 256 integer levels (uint8
range mapping) so they can be tokenized as plain bytes by the
``modal/byte_patch`` ingestion path of the v1.5+ trainer.

Domains
-------
* **stock**  -- AR(1) random walk with drift + occasional jump events.
                Approximates equity log-returns at minute-bar granularity.
* **sensor** -- ARMA(2,1) process simulating temperature / humidity sensor
                streams (slow trend + autocorrelated noise).
* **bio**    -- Superposition of multiple sinusoids at 1Hz, 5Hz, 12Hz with
                mild amplitude modulation -- ECG / EEG-shaped envelopes.

Each sequence is exactly **256 timesteps**. After per-sequence min/max
normalisation we quantise to **256 integer levels** packed as int8.

Output schema (parquet, one row = one sequence)
-----------------------------------------------
    timestamps    list<int8>  256 quantized levels
    domain        str         "stock" | "sensor" | "bio"
    caption       str         e.g. "stock prices over 256 ticks"
    text          str         caption (kept for ParquetTokenStream column compat)

Companion ``.manifest.json`` records seed, row count, domain mix.

Determinism
-----------
Seeded by ``--seed`` (default 42). The RNG advances domain → AR / ARMA
draws → quantisation in a fixed order; same seed yields byte-identical
output (asserted by ``test_deterministic``).

Usage
-----
    python scripts/synth_timeseries_pretrain.py \\
        --out /workspace/data/pretrain/synth_ts/train.parquet \\
        --n 100000 --seed 42

    python scripts/synth_timeseries_pretrain.py --smoke  # 10 rows, fast
    python scripts/synth_timeseries_pretrain.py --help
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


SEQ_LEN = 256
N_LEVELS = 256

DOMAINS = ("stock", "sensor", "bio")

CAPTIONS = {
    "stock": "stock prices over 256 ticks",
    "sensor": "sensor reading over 256 timesteps",
    "bio": "biological signal over 256 samples",
}


# --- generators (pure-python deterministic; numpy used only for vector ops) -


def _gen_stock(rng: "random.Random") -> List[float]:
    """AR(1) random walk + drift + ~3% chance of jump per step.

    Models a price series y_t = phi*y_{t-1} + drift + eps_t, with phi ~
    0.985 (highly autocorrelated as real prices are), occasional Gaussian
    jumps to inject heavy-tailed events.
    """
    phi = 0.985
    drift = rng.gauss(0.0, 0.005)
    sigma = 0.6
    jump_p = 0.03
    jump_sigma = 4.0

    y = 0.0
    out = []
    for _ in range(SEQ_LEN):
        eps = rng.gauss(0.0, sigma)
        if rng.random() < jump_p:
            eps += rng.gauss(0.0, jump_sigma)
        y = phi * y + drift + eps
        out.append(y)
    return out


def _gen_sensor(rng: "random.Random") -> List[float]:
    """ARMA(2,1): y_t = a1*y_{t-1} + a2*y_{t-2} + eps_t + b1*eps_{t-1}.

    Coefficients chosen for a stable, slowly-varying process (think
    indoor temperature). Stationary roots: |a1| + |a2| < 1.
    """
    a1, a2, b1 = 0.55, 0.30, 0.40
    sigma = 0.4

    y_prev1, y_prev2 = 0.0, 0.0
    eps_prev = 0.0
    out = []
    for _ in range(SEQ_LEN):
        eps = rng.gauss(0.0, sigma)
        y = a1 * y_prev1 + a2 * y_prev2 + eps + b1 * eps_prev
        out.append(y)
        y_prev2 = y_prev1
        y_prev1 = y
        eps_prev = eps
    return out


def _gen_bio(rng: "random.Random") -> List[float]:
    """Sum of three sinusoids (1Hz, 5Hz, 12Hz) with random phase + AM.

    Envelope models a biological rhythm where the high-frequency carrier
    waxes and wanes (e.g. heartbeat amplitude varying with respiration).
    Sampled at fs=128 Hz so SEQ_LEN=256 covers ~2 seconds.
    """
    fs = 128.0
    freqs = (1.0, 5.0, 12.0)
    amps = tuple(rng.uniform(0.4, 1.0) for _ in freqs)
    phases = tuple(rng.uniform(0.0, 2 * math.pi) for _ in freqs)
    am_freq = rng.uniform(0.3, 0.8)
    am_phase = rng.uniform(0.0, 2 * math.pi)
    noise_sigma = 0.05

    out = []
    for t in range(SEQ_LEN):
        s = 0.0
        for f, a, p in zip(freqs, amps, phases):
            s += a * math.sin(2 * math.pi * f * (t / fs) + p)
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * am_freq * (t / fs) + am_phase)
        s = s * envelope + rng.gauss(0.0, noise_sigma)
        out.append(s)
    return out


GENERATORS = {
    "stock": _gen_stock,
    "sensor": _gen_sensor,
    "bio": _gen_bio,
}


def _quantize_to_int8(values: Sequence[float]) -> List[int]:
    """Per-sequence min/max normalise then quantise to 256 levels stored as int8.

    Output range: -128..127 (signed). Constant sequences map to all zeros.
    """
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0] * len(values)
    out = []
    for v in values:
        # Map to [0, 255].
        u = (v - lo) / span
        q = int(round(u * (N_LEVELS - 1)))
        if q < 0:
            q = 0
        elif q > N_LEVELS - 1:
            q = N_LEVELS - 1
        # Re-center to int8 range [-128, 127].
        out.append(q - 128)
    return out


def gen_sequence(domain: str, rng: "random.Random") -> Tuple[List[int], str]:
    """Public-ish helper: produce (quantized_values_int8, caption) for ``domain``.

    Used directly by the smoke / unit tests so they can exercise the
    generator without going through ``main()`` argv parsing.
    """
    if domain not in GENERATORS:
        raise ValueError(f"unknown domain {domain!r}; want one of {DOMAINS}")
    raw = GENERATORS[domain](rng)
    return _quantize_to_int8(raw), CAPTIONS[domain]


def _generate_rows(n: int, seed: int) -> List[dict]:
    """Generate ``n`` rows with deterministic domain rotation."""
    rng = random.Random(seed)
    rows: List[dict] = []
    for i in range(n):
        # Round-robin through DOMAINS so n=30 always covers all three.
        domain = DOMAINS[i % len(DOMAINS)]
        ts, caption = gen_sequence(domain, rng)
        rows.append(
            dict(
                timestamps=ts,
                domain=domain,
                caption=caption,
                text=caption,  # alias for ParquetTokenStream's "text" column
            )
        )
    return rows


def _write_manifest(out: str, n: int, seed: int) -> None:
    mfile = str(out) + ".manifest.json"
    Path(mfile).parent.mkdir(parents=True, exist_ok=True)
    counts = {d: 0 for d in DOMAINS}
    # The generator is round-robin so counts are a function of n only.
    for i in range(n):
        counts[DOMAINS[i % len(DOMAINS)]] += 1
    with open(mfile, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "synth_timeseries",
                "rows": n,
                "seed": seed,
                "seq_len": SEQ_LEN,
                "n_levels": N_LEVELS,
                "domains": list(DOMAINS),
                "domain_counts": counts,
                "warning": "synthetic only; admix at <=2% with real time-series",
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
        )
    print(f"[synth_ts] manifest -> {mfile}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out",
                    default="/workspace/data/pretrain/synth_ts/train.parquet")
    ap.add_argument("--n", type=int, default=100_000,
                    help="Total sequences to emit (default 100K)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="Generate just 10 sequences and exit")
    args = ap.parse_args(argv)

    n = 10 if args.smoke else args.n
    t0 = time.time()
    rows = _generate_rows(n, args.seed)

    if not _HAVE_ARROW:
        # JSONL fallback so callers still get *something*.
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        jp = out_p.with_suffix(".jsonl")
        print(f"[synth_ts] pyarrow missing; writing JSONL fallback -> {jp}",
              file=sys.stderr)
        with open(jp, "w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _write_manifest(args.out, len(rows), args.seed)
        return 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "timestamps": [r["timestamps"] for r in rows],
            "domain": [r["domain"] for r in rows],
            "caption": [r["caption"] for r in rows],
            "text": [r["text"] for r in rows],
        }
    )
    pq.write_table(pa.Table.from_pandas(df), args.out, compression="zstd")
    print(
        f"[synth_ts] wrote {len(df):,} rows -> {args.out} "
        f"({time.time() - t0:.1f}s)"
    )
    _write_manifest(args.out, len(df), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
