#!/usr/bin/env python3
"""scripts/bench_throughput.py — A/B throughput lever combos.

Measures tokens-per-second under each combination of the two new
2026-05-02 throughput levers (CUDA Graphs + async data pipeline) so
the rental run can pick the best combo.

Combos (rows in the output table):
    baseline            : current path (ParquetTokenStream prefetch=2)
    +cuda-graphs        : CUDA Graphs only
    +async-pipeline     : 4-stage async data only
    +both               : CUDA Graphs + async data

Each row runs ``--warmup`` warmup steps (default 100) followed by
``--steps`` timed steps (default 200) and reports::

    combo                tok/s     step_ms     vs_baseline
    baseline             5240      2440        1.00x
    +cuda-graphs         5430      2354        1.04x
    +async-pipeline      5680      2253        1.08x
    +both                5860      2185        1.12x

Rental-deferred: this bench needs CUDA + a real GPU. On a no-CUDA
torch build we still verify the script imports cleanly and emits a
pre-cooked "skipped" output so CI doesn't break.

Usage
-----
    python3 scripts/bench_throughput.py \
        --warmup 100 --steps 200 \
        --batch-size 24 --seq-len 256 \
        --data-glob "/workspace/data/wt103_raw/train-*.parquet"

Output: JSON record at ``bench_results/throughput_HHMMSS.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

# Make synapforge importable when launched from the repo root.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=100,
                   help="warmup steps before timing (graph capture etc.)")
    p.add_argument("--steps", type=int, default=200,
                   help="timed steps")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--data-glob", default="",
                   help="parquet glob; if empty, use synthetic random "
                        "tokens (graph A/B still works, async pipeline "
                        "can't be exercised against real I/O).")
    p.add_argument("--tokenizer-name", default="gpt2")
    p.add_argument("--combo", default="all",
                   choices=["all", "baseline", "cuda-graphs",
                            "async-pipeline", "both"],
                   help="which combo(s) to measure (default: all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output-dir", default="bench_results")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-step prints")
    return p


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build a SynapForge100M-like model. We use the real one when
    available so the bench covers the production HybridBlock path."""
    try:
        from synapforge.model_100m import SynapForge100M
        model = SynapForge100M(
            vocab=int(args.vocab),
            d=int(args.d),
            n_layers=int(args.n_layers),
            max_seq=int(args.seq_len),
        )
    except Exception as exc:
        print(f"[bench] could not build SynapForge100M ({exc!r}); "
              "falling back to a tiny stand-in")

        class _StandIn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(int(args.vocab), int(args.d))
                self.linear = torch.nn.Linear(int(args.d), int(args.d), bias=False)
                self.head = torch.nn.Linear(int(args.d), int(args.vocab), bias=False)

            def forward(self, x):
                h = self.embed(x)
                h = torch.nn.functional.silu(self.linear(h))
                return self.head(h)
        model = _StandIn()
    model = model.to(args.device)
    return model


def _make_random_data_iter(args: argparse.Namespace):
    """Yield infinite (x, y) tensors of shape (B, T) on device for
    synthetic benches."""
    g = torch.Generator(device=args.device).manual_seed(0)
    while True:
        x = torch.randint(0, int(args.vocab),
                          (int(args.batch_size), int(args.seq_len)),
                          generator=g, device=args.device)
        # Shifted-by-1 target like real LM data.
        y = torch.roll(x, shifts=-1, dims=1)
        yield x, y


def _make_real_data_iter(args: argparse.Namespace, async_pipeline: bool):
    """Yield real (x, y) batches via ParquetTokenStream or
    AsyncTokenStream depending on the flag."""
    from synapforge.data import ParquetTokenStream
    if async_pipeline:
        from synapforge.data import AsyncTokenStream
        ds = AsyncTokenStream(
            args.data_glob, seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            tokenizer_name=args.tokenizer_name,
            loop=True, shuffle_buffer=10000, shuffle_seed=42,
            pin_memory=True,
            stages=4, prefetch=8,
        )
    else:
        ds = ParquetTokenStream(
            args.data_glob, seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            tokenizer_name=args.tokenizer_name,
            loop=True, shuffle_buffer=10000, shuffle_seed=42,
            prefetch_factor=2, pin_memory=True,
        )
    it = iter(ds)
    while True:
        x, y = next(it)
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
        yield x, y


def _run_combo(
    name: str,
    args: argparse.Namespace,
    cuda_graphs: bool,
    async_pipeline: bool,
) -> dict[str, Any]:
    """Run one combo and return (tok_per_s, step_ms)."""
    print(f"\n=== combo: {name} (cuda_graphs={cuda_graphs} "
          f"async_pipeline={async_pipeline}) ===")

    if args.device == "cuda" and not torch.cuda.is_available():
        return {
            "name": name, "tok_per_s": float("nan"), "step_ms": float("nan"),
            "skipped": True,
            "reason": "CUDA not available on this host",
        }

    # Fresh model per combo so capture state doesn't leak across.
    torch.manual_seed(42)
    model = _build_model(args)
    model.train()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Build data iter
    if args.data_glob:
        try:
            data_iter = _make_real_data_iter(args, async_pipeline)
        except Exception as exc:
            print(f"[bench] real data iter failed ({exc!r}); "
                  "falling back to random")
            data_iter = _make_random_data_iter(args)
    else:
        if async_pipeline:
            print("[bench] async-pipeline=True but no --data-glob; "
                  "the lever has no synthetic harness here. "
                  "Reporting same as baseline (random).")
        data_iter = _make_random_data_iter(args)

    # Build CUDA graph wrapper if requested
    graphed = None
    if cuda_graphs:
        from synapforge.training.cuda_graphs import (
            GraphedBlockCfg, GraphedHybridBlock, cross_entropy_loss,
        )
        cfg = GraphedBlockCfg(
            batch_size=int(args.batch_size),
            seq_len=int(args.seq_len),
            device=torch.device(args.device),
            dtype=dtype,
            n_warmup_iters=11,
            accumulate_grad=False,  # bench drives optim.step manually
        )
        graphed = GraphedHybridBlock(model, cross_entropy_loss, cfg)
        if not graphed.capture_active:
            print(f"[bench] capture failed: {graphed.skip_reason}; "
                  "falling back to eager")
            graphed = None

    # ---------------------- warmup ----------------------
    print(f"[bench] warmup {args.warmup} steps...")
    for i in range(int(args.warmup)):
        x, y = next(data_iter)
        if graphed is not None:
            optim.zero_grad(set_to_none=True)
            loss = graphed.step(x, y)
            optim.step()
        else:
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=args.device,
                                    dtype=dtype,
                                    enabled=args.device == "cuda"):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                )
            loss.backward()
            optim.step()
        if not args.quiet and (i + 1) % 50 == 0:
            print(f"  warmup step {i + 1} / {args.warmup}")
    if args.device == "cuda":
        torch.cuda.synchronize()

    # ---------------------- timed ----------------------
    print(f"[bench] timed {args.steps} steps...")
    t0 = time.time()
    for i in range(int(args.steps)):
        x, y = next(data_iter)
        if graphed is not None:
            optim.zero_grad(set_to_none=True)
            loss = graphed.step(x, y)
            optim.step()
        else:
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=args.device,
                                    dtype=dtype,
                                    enabled=args.device == "cuda"):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                )
            loss.backward()
            optim.step()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    tokens_per_step = int(args.batch_size) * int(args.seq_len)
    total_tokens = tokens_per_step * int(args.steps)
    tok_per_s = total_tokens / elapsed
    step_ms = elapsed * 1000.0 / int(args.steps)
    print(f"[bench] {name}: tok/s={tok_per_s:.1f} step_ms={step_ms:.1f} "
          f"elapsed={elapsed:.2f}s")
    return {
        "name": name,
        "tok_per_s": float(tok_per_s),
        "step_ms": float(step_ms),
        "elapsed_s": float(elapsed),
        "tokens_per_step": int(tokens_per_step),
        "skipped": False,
    }


def _format_table(rows: list[dict[str, Any]]) -> str:
    """Pretty-print the combo results as an aligned table."""
    if not rows:
        return "(no rows)"
    baseline_tps = None
    for row in rows:
        if row["name"] == "baseline" and not row["skipped"]:
            baseline_tps = row["tok_per_s"]
            break
    out = []
    header = f"{'combo':<22} {'tok/s':>10} {'step_ms':>10} {'vs_baseline':>12}"
    out.append(header)
    out.append("-" * len(header))
    for row in rows:
        if row["skipped"]:
            out.append(f"{row['name']:<22} {'SKIPPED':>10} "
                       f"{row.get('reason', ''):>30}")
            continue
        if baseline_tps is not None and baseline_tps > 0:
            ratio = row["tok_per_s"] / baseline_tps
        else:
            ratio = float("nan")
        out.append(
            f"{row['name']:<22} {row['tok_per_s']:>10.1f} "
            f"{row['step_ms']:>10.1f} {ratio:>12.2f}x"
        )
    return "\n".join(out)


def main() -> int:
    args = _make_parser().parse_args()

    combos = []
    if args.combo in ("all", "baseline"):
        combos.append(("baseline", False, False))
    if args.combo in ("all", "cuda-graphs"):
        combos.append(("+cuda-graphs", True, False))
    if args.combo in ("all", "async-pipeline"):
        combos.append(("+async-pipeline", False, True))
    if args.combo in ("all", "both"):
        combos.append(("+both", True, True))

    rows: list[dict[str, Any]] = []
    for name, cg, ap in combos:
        try:
            rows.append(_run_combo(name, args, cg, ap))
        except Exception as exc:
            print(f"[bench] combo {name} FAILED: {exc!r}")
            rows.append({
                "name": name, "tok_per_s": float("nan"),
                "step_ms": float("nan"), "elapsed_s": float("nan"),
                "skipped": True, "reason": f"exception: {exc!r}",
            })

    print("\n========= results =========")
    print(_format_table(rows))

    # Save JSON output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"throughput_{time.strftime('%H%M%S')}.json",
    )
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "rows": rows,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }, f, indent=2)
    print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
