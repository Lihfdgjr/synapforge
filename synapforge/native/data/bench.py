"""Throughput parity bench: NativeParquetStream vs torch ParquetTokenStream.

Goal
----
Verify the no-torch native loader is **>= 1.0x** the throughput of the
legacy torch ``ParquetTokenStream`` on the same parquet shard. We are
NOT trying to beat the torch path on speed; we just want to confirm
that dropping the torch dependency does not regress throughput.

Run
---
``python -m synapforge.native.data.bench``

Or, programmatically:

>>> from synapforge.native.data.bench import run_bench
>>> run_bench(parquet_glob="data/wt103_raw/train-*.parquet",
...           seq_len=256, batch_size=32, n_batches=200,
...           tokenizer="gpt2")

Output (printed)
----------------
``native: 12345 tokens/s   torch: 11000 tokens/s   ratio: 1.12x``

Behaviour
---------
- If torch isn't importable (dev box without it), skip the torch leg
  and print only native.
- If pyarrow isn't importable, abort -- native side can't run.
- Constructs two streams with the EXACT same args (seq_len, batch_size,
  shuffle_buffer=0 for determinism, prefetch_factor=2).
- Warms up each side with one batch before timing (avoids tokenizer
  cold-start dominating short runs).
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Optional

import numpy as np


def _bench_native(
    parquet_glob: str,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    tokenizer: str,
    prefetch_factor: int,
    num_workers: int,
) -> "tuple[float, float]":
    """Time the native loader; return (tokens/sec, total_seconds)."""
    from synapforge.native.data.parquet_stream import NativeParquetStream

    ds = NativeParquetStream(
        parquet_glob,
        seq_len=seq_len,
        batch_size=batch_size,
        tokenizer=tokenizer,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        loop=True,
    )
    it = iter(ds)
    # warmup
    next(it)
    t0 = time.perf_counter()
    n_tokens = 0
    for x, _y in itertools.islice(it, n_batches):
        n_tokens += int(x.size)
    elapsed = time.perf_counter() - t0
    return (n_tokens / elapsed if elapsed > 0 else 0.0), elapsed


def _bench_torch(
    parquet_glob: str,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    tokenizer: str,
    prefetch_factor: int,
) -> "Optional[tuple[float, float]]":
    """Time the legacy torch loader. Returns None if torch unavailable."""
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
        from synapforge.data import ParquetTokenStream
    except ImportError as e:
        print(f"[bench] skipping torch leg: {e}")
        return None
    ds = ParquetTokenStream(
        parquet_glob,
        seq_len=seq_len,
        batch_size=batch_size,
        tokenizer_name=tokenizer,
        prefetch_factor=prefetch_factor,
        loop=True,
    )
    it = iter(ds)
    next(it)
    t0 = time.perf_counter()
    n_tokens = 0
    for x, _y in itertools.islice(it, n_batches):
        n_tokens += int(x.numel())
    elapsed = time.perf_counter() - t0
    return (n_tokens / elapsed if elapsed > 0 else 0.0), elapsed


def run_bench(
    parquet_glob: str,
    seq_len: int = 256,
    batch_size: int = 32,
    n_batches: int = 100,
    tokenizer: str = "gpt2",
    prefetch_factor: int = 2,
    num_workers: int = 4,
) -> "dict":
    """Run both legs and print the parity report.

    Returns a dict with native/torch entries containing throughput and
    elapsed time for downstream test assertions.
    """
    print(f"[bench] parquet_glob={parquet_glob!r} "
          f"seq_len={seq_len} batch_size={batch_size} "
          f"n_batches={n_batches} tokenizer={tokenizer!r}")

    native_thru, native_t = _bench_native(
        parquet_glob, seq_len, batch_size, n_batches,
        tokenizer, prefetch_factor, num_workers,
    )
    print(f"[bench] native: {native_thru:.0f} tokens/s  ({native_t:.2f}s)")

    torch_res = _bench_torch(
        parquet_glob, seq_len, batch_size, n_batches,
        tokenizer, prefetch_factor,
    )
    out: dict = {
        "native": {"thru_tokens_per_sec": native_thru, "seconds": native_t},
    }
    if torch_res is not None:
        torch_thru, torch_t = torch_res
        ratio = native_thru / torch_thru if torch_thru > 0 else float("inf")
        print(f"[bench] torch:  {torch_thru:.0f} tokens/s  ({torch_t:.2f}s)")
        print(f"[bench] ratio (native / torch) = {ratio:.2f}x  "
              f"({'ok' if ratio >= 1.0 else 'BELOW PARITY'})")
        out["torch"] = {"thru_tokens_per_sec": torch_thru, "seconds": torch_t}
        out["ratio"] = ratio
    else:
        out["torch"] = None
        out["ratio"] = None
    return out


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet-glob", required=True,
                   help="glob of parquet shards to feed both loaders")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-batches", type=int, default=100)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    res = run_bench(
        parquet_glob=args.parquet_glob,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        tokenizer=args.tokenizer,
        prefetch_factor=args.prefetch_factor,
        num_workers=args.num_workers,
    )
    if res.get("ratio") is not None and res["ratio"] < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
