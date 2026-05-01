#!/usr/bin/env python3
"""Run the full paper-grade bench harness against a SynapForge checkpoint.

Loads model + tokenizer once, dispatches each requested benchmark, aggregates
the per-bench summaries into a single JSON, and prints a delta table against
public small-model baselines.

Usage:
    python scripts/run_all_bench.py \\
        --ckpt runs/sf_100m/best.pt \\
        --tokenizer Qwen/Qwen2.5-0.5B \\
        --benches humaneval,mmlu,gsm8k \\
        --out bench.json [--n 50]

`--n` is forwarded as a per-bench cap (smoke runs). Drop it for a full pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Public reference numbers (paper / lm-eval-harness / model cards). All
# figures are 0-shot accuracy or pass@1 unless noted.
BASELINES: Dict[str, Dict[str, float]] = {
    "tinyllama-1.1b": {
        "humaneval": 0.049, "mbpp": 0.085, "mmlu": 0.253,
        "gsm8k": 0.024,    "hellaswag": 0.595, "lambada": 0.580,
    },
    "smollm2-135m": {
        "humaneval": 0.012, "mbpp": 0.020, "mmlu": 0.293,
        "gsm8k": 0.013,    "hellaswag": 0.418, "lambada": 0.338,
    },
    "qwen2.5-0.5b": {
        "humaneval": 0.305, "mbpp": 0.395, "mmlu": 0.475,
        "gsm8k": 0.418,    "hellaswag": 0.523, "lambada": 0.498,
    },
    "mythos-target": {  # what we'd like to hit eventually
        "humaneval": 0.40, "mbpp": 0.50, "mmlu": 0.55,
        "gsm8k": 0.60,    "hellaswag": 0.65, "lambada": 0.62,
    },
}

# Per-bench primary metric key (what we extract from each summary).
METRIC_KEY: Dict[str, str] = {
    "humaneval": "pass@1",
    "mbpp":      "pass@1",
    "mmlu":      "acc",
    "gsm8k":     "acc",
    "hellaswag": "acc",
    "lambada":   "acc",
}


def _print_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print: bench | ours | tinyllama | smollm2 | qwen0.5b | mythos."""
    cols = ["bench", "ours", "tinyllama-1.1b", "smollm2-135m", "qwen2.5-0.5b", "mythos-target"]
    rows = []
    for bench, summary in results.items():
        if "error" in summary:
            rows.append([bench, "ERR", "—", "—", "—", "—"])
            continue
        ours = summary.get(METRIC_KEY.get(bench, "acc"))
        row = [bench, f"{ours:.3f}" if ours is not None else "—"]
        for base_name in cols[2:]:
            v = BASELINES.get(base_name, {}).get(bench)
            row.append(f"{v:.3f}" if v is not None else "—")
        rows.append(row)
    widths = [max(len(c), max(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    sep = "  ".join("-" * w for w in widths)
    print()
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print(sep)
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print(sep)
    # Print delta-vs-tinyllama row (the most directly comparable).
    deltas = []
    for bench, summary in results.items():
        if "error" in summary:
            deltas.append((bench, None))
            continue
        ours = summary.get(METRIC_KEY.get(bench, "acc"))
        ref = BASELINES["tinyllama-1.1b"].get(bench)
        if ours is None or ref is None:
            deltas.append((bench, None))
        else:
            deltas.append((bench, ours - ref))
    print("delta vs tinyllama-1.1b:")
    for bench, d in deltas:
        msg = f"  {bench:<10}  " + (f"{d:+.3f}" if d is not None else "—")
        print(msg)


def _slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--benches", default="humaneval,mbpp,mmlu,gsm8k,hellaswag,lambada",
                    help="Comma-separated subset of {humaneval,mbpp,mmlu,gsm8k,hellaswag,lambada}")
    ap.add_argument("--n", type=int, default=None,
                    help="Cap examples per bench (smoke). Drop for full eval.")
    ap.add_argument("--out", default=None,
                    help="JSON output path. Defaults to runs/bench/<ckpt-stem>.<ts>.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synapforge.bench import BENCH_REGISTRY, run_bench
    from synapforge.eval.generate import load_synapforge, load_tokenizer

    bench_names = [b.strip() for b in args.benches.split(",") if b.strip()]
    unknown = [b for b in bench_names if b not in BENCH_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown benches: {unknown}; valid: {sorted(BENCH_REGISTRY)}")

    print(f"[bench] loading model: {args.ckpt}", flush=True)
    model, src = load_synapforge(args.ckpt)
    print(f"[bench]   src: {src}", flush=True)
    tok = load_tokenizer(args.tokenizer)

    results: Dict[str, Dict[str, Any]] = {}
    t_start = time.time()
    for name in bench_names:
        print(f"\n[bench] >>> {name}", flush=True)
        t0 = time.time()
        try:
            kw = {}
            if args.n is not None:
                # MMLU uses n_per_subject, others use n.
                kw["n_per_subject" if name == "mmlu" else "n"] = args.n
            results[name] = run_bench(name, model=model, tok=tok, **kw)
        except Exception as e:
            print(f"[bench]   ERROR: {type(e).__name__}: {e}", flush=True)
            results[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[bench]   wall: {time.time() - t0:.1f}s", flush=True)

    elapsed = time.time() - t_start

    out_path = args.out or f"runs/bench/{Path(args.ckpt).stem}.{_slug()}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "ckpt":      args.ckpt,
        "tokenizer": args.tokenizer,
        "n_cap":     args.n,
        "wall_s":    elapsed,
        "results":   results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"\n[bench] wrote {out_path}  (total {elapsed:.1f}s)")
    _print_table(results)


if __name__ == "__main__":
    main()
