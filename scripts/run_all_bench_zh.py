#!/usr/bin/env python3
"""Bilingual bench orchestrator (EN + ZH).

Runs both the English (HumanEval/MBPP/MMLU/GSM8K/HellaSwag/LAMBADA) and
Chinese (CMMLU/C-Eval/AGIEval-zh/GSM8K-zh) bench suites against a single
checkpoint, prints a side-by-side parity table, and writes a combined JSON.

Usage:
    python scripts/run_all_bench_zh.py \\
        --ckpt runs/sf_100m/best.pt \\
        --tokenizer Qwen/Qwen2.5-0.5B \\
        --langs en,zh \\
        --out bench_bilingual.json [--n 20]

`--n` caps examples per bench (smoke). Drop for full eval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Per-bench primary metric key.
METRIC_KEY: Dict[str, str] = {
    "humaneval":  "pass@1",
    "mbpp":       "pass@1",
    "mmlu":       "acc",
    "gsm8k":      "acc",
    "hellaswag":  "acc",
    "lambada":    "acc",
    "cmmlu":      "acc",
    "ceval":      "acc",
    "agieval_zh": "acc",
    "gsm8k_zh":   "acc",
}

# Public reference numbers for ZH suite. Values from Qwen2.5 / DeepSeek /
# CMMLU+C-Eval leaderboards. Approximate paper-grade baselines.
BASELINES_ZH: Dict[str, Dict[str, float]] = {
    "qwen2.5-0.5b":   {"cmmlu": 0.520, "ceval": 0.547, "agieval_zh": 0.295, "gsm8k_zh": 0.310},
    "qwen2.5-1.5b":   {"cmmlu": 0.642, "ceval": 0.679, "agieval_zh": 0.398, "gsm8k_zh": 0.515},
    "tinyllama-1.1b": {"cmmlu": 0.247, "ceval": 0.252, "agieval_zh": 0.215, "gsm8k_zh": 0.014},
    "mythos-target":  {"cmmlu": 0.55,  "ceval": 0.58,  "agieval_zh": 0.40,  "gsm8k_zh": 0.55},
}

BASELINES_EN: Dict[str, Dict[str, float]] = {
    "tinyllama-1.1b": {"humaneval": 0.049, "mbpp": 0.085, "mmlu": 0.253,
                       "gsm8k": 0.024, "hellaswag": 0.595, "lambada": 0.580},
    "qwen2.5-0.5b":   {"humaneval": 0.305, "mbpp": 0.395, "mmlu": 0.475,
                       "gsm8k": 0.418, "hellaswag": 0.523, "lambada": 0.498},
    "mythos-target":  {"humaneval": 0.40, "mbpp": 0.50, "mmlu": 0.55,
                       "gsm8k": 0.60, "hellaswag": 0.65, "lambada": 0.62},
}


def _print_block(title: str, results: Dict[str, Dict[str, Any]],
                 baselines: Dict[str, Dict[str, float]]) -> None:
    if not results:
        return
    cols = ["bench", "ours"] + list(baselines.keys())
    rows: List[List[str]] = []
    for bench, summary in results.items():
        if "error" in summary:
            rows.append([bench, "ERR"] + ["—"] * len(baselines))
            continue
        ours = summary.get(METRIC_KEY.get(bench, "acc"))
        row = [bench, f"{ours:.3f}" if ours is not None else "—"]
        for base_name in baselines:
            v = baselines[base_name].get(bench)
            row.append(f"{v:.3f}" if v is not None else "—")
        rows.append(row)
    widths = [max(len(c), max((len(r[i]) for r in rows), default=0))
              for i, c in enumerate(cols)]
    sep = "  ".join("-" * w for w in widths)
    print(f"\n=== {title} ===")
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print(sep)
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print(sep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--langs", default="en,zh",
                    help="Comma-separated subset of {en,zh}")
    ap.add_argument("--n", type=int, default=None,
                    help="Cap examples per bench (smoke).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synapforge.bench import (BENCH_REGISTRY, EN_BENCHES, ZH_BENCHES,
                                  run_bench)
    from synapforge.eval.generate import load_synapforge, load_tokenizer

    langs = {l.strip().lower() for l in args.langs.split(",") if l.strip()}
    bench_names: List[str] = []
    if "en" in langs:
        bench_names.extend(EN_BENCHES)
    if "zh" in langs:
        bench_names.extend(ZH_BENCHES)
    if not bench_names:
        raise SystemExit(f"--langs must contain en and/or zh; got {args.langs!r}")

    print(f"[bench-bi] loading model: {args.ckpt}", flush=True)
    model, src = load_synapforge(args.ckpt)
    print(f"[bench-bi]   src: {src}", flush=True)
    tok = load_tokenizer(args.tokenizer)

    en_results: Dict[str, Dict[str, Any]] = {}
    zh_results: Dict[str, Dict[str, Any]] = {}
    t_start = time.time()
    for name in bench_names:
        print(f"\n[bench-bi] >>> {name}", flush=True)
        t0 = time.time()
        try:
            kw: Dict[str, Any] = {}
            if args.n is not None:
                if name in ("mmlu", "cmmlu", "ceval"):
                    kw["n_per_subject"] = args.n
                elif name == "agieval_zh":
                    kw["n_per_task"] = args.n
                else:
                    kw["n"] = args.n
            res = run_bench(name, model=model, tok=tok, **kw)
        except Exception as e:
            print(f"[bench-bi]   ERROR: {type(e).__name__}: {e}", flush=True)
            res = {"error": f"{type(e).__name__}: {e}"}
        if name in EN_BENCHES:
            en_results[name] = res
        else:
            zh_results[name] = res
        print(f"[bench-bi]   wall: {time.time() - t0:.1f}s", flush=True)

    elapsed = time.time() - t_start
    out_path = args.out or f"runs/bench/{Path(args.ckpt).stem}.bilingual.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "ckpt":      args.ckpt,
        "tokenizer": args.tokenizer,
        "n_cap":     args.n,
        "wall_s":    elapsed,
        "en":        en_results,
        "zh":        zh_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print(f"\n[bench-bi] wrote {out_path}  (total {elapsed:.1f}s)")
    _print_block("English",  en_results, BASELINES_EN)
    _print_block("Chinese",  zh_results, BASELINES_ZH)


if __name__ == "__main__":
    main()
