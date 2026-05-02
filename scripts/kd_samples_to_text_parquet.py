"""kd_samples_to_text_parquet.py — convert KD distill ``.samples.jsonl``
into a plain ``text`` parquet so the existing ``ParquetTokenStream``
trainer dataloader can consume teacher-generated continuations as
ordinary CE-loss training data.

Why
---
The pre-generated KD distill parquet (output of ``scripts/gen_kd_data.py``)
carries top-K teacher log-probs in a custom schema. Wiring those into
the trainer's KD loss path needs a separate dataloader merge (staged
in M5.1 follow-up PR). But TODAY we can still pull the immediate quality
win — feeding the CE-loss with *clean teacher trajectories* — by:

  1. Reading the ``.samples.jsonl`` sidecar (one JSON per row: prompt,
     completion).
  2. Concatenating ``prompt + completion`` into a single ``text`` field.
  3. Writing a parquet with column ``text`` (string) — exactly the
     schema ``ParquetTokenStream`` expects.

Then the operator passes the resulting parquet's path to
``--data-glob`` (or as one of multiple globs) and the student trains
its CE-loss on the teacher's clean continuations directly. The KD
loss path keeps its existing live-teacher behaviour on web rows.

Usage
-----
    python scripts/kd_samples_to_text_parquet.py \\
        --samples /workspace/data/kd_distill_v1.parquet.samples.jsonl \\
        --output  /workspace/data/kd_distill_v1_text.parquet \\
        --min-tok 32 --max-tok 1024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", required=True,
                    help="Path to *.samples.jsonl from gen_kd_data.py")
    ap.add_argument("--output", required=True,
                    help="Output parquet path (column: text)")
    ap.add_argument("--min-tok", type=int, default=32,
                    help="Drop rows whose total_tok_len < this (degenerate)")
    ap.add_argument("--max-tok", type=int, default=1024,
                    help="Drop rows whose total_tok_len > this (over-long)")
    ap.add_argument("--drop-empty-completion", action="store_true", default=True,
                    help="Drop rows where the completion field is empty (default ON)")
    args = ap.parse_args(argv)

    if not _HAVE_ARROW:
        print("[kd_samples_to_text_parquet] FATAL: pyarrow required", file=sys.stderr)
        return 2

    src = Path(args.samples)
    if not src.exists():
        print(f"[kd_samples_to_text_parquet] {src} not found", file=sys.stderr)
        return 1

    rows: list[str] = []
    n_total = 0
    n_drop_empty = 0
    n_drop_short = 0
    n_drop_long = 0
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            comp = (d.get("completion") or "").strip()
            prompt = d.get("prompt") or ""
            tok_len = int(d.get("total_tok_len") or 0)
            if args.drop_empty_completion and not comp:
                n_drop_empty += 1
                continue
            if tok_len > 0 and tok_len < args.min_tok:
                n_drop_short += 1
                continue
            if tok_len > 0 and tok_len > args.max_tok:
                n_drop_long += 1
                continue
            # Concatenate prompt + completion -- the trainer's ParquetTokenStream
            # treats this as a single continuous text row, which is the natural
            # mode for next-token CE.
            text = prompt + comp
            rows.append(text)

    if not rows:
        print("[kd_samples_to_text_parquet] no valid rows after filtering", file=sys.stderr)
        return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": rows})
    pq.write_table(table, args.output, compression="zstd")

    n_kept = len(rows)
    avg_chars = sum(len(r) for r in rows) / max(n_kept, 1)
    print(
        f"[kd_samples_to_text_parquet] wrote {n_kept:,} rows "
        f"(of {n_total:,}) -> {args.output} "
        f"avg_chars={avg_chars:.0f} "
        f"dropped: empty={n_drop_empty} "
        f"short={n_drop_short} long={n_drop_long}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
