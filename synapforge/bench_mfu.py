"""bench_mfu.py — Measure synapforge_100m Model FLOPs Utilization on A100.

Honest measurement:
  - achieved_TFLOPS = total_flops_per_step / wall_time_per_step
  - MFU = achieved_TFLOPS / 312 (A100 SXM bf16 dense peak)

Counts both forward + backward FLOPs via torch.utils.flop_counter
(weight-shared loop_depth=4 inner repeats are counted N=4 times — the
RDT trick saves PARAMS, not compute, so MFU should reflect that).

Usage:
    GPU=1 python bench_mfu.py --backend gpu_dense --steps 50
    GPU=1 python bench_mfu.py --backend triton_block --steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

# Force GPU 1 BEFORE importing torch.
_GPU = os.environ.get("CUDA_VISIBLE_DEVICES")
if _GPU is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, "/workspace")
from synapforge.model_100m import build_synapforge_100m  # noqa: E402

A100_BF16_PEAK_TFLOPS = 312.0
A100_HBM_PEAK_GBs = 2039.0  # SXM4 80GB HBM2e


def _device_info() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    p = torch.cuda.get_device_properties(0)
    return {
        "device": p.name,
        "sm_count": p.multi_processor_count,
        "memory_GB": p.total_memory / 1e9,
        "cc": f"{p.major}.{p.minor}",
    }


def _real_data_iterator(batch: int, seq: int, vocab: int):
    """Try to read wt103 parquet; fall back to torch.randint if absent."""
    try:
        import pyarrow.parquet as pq
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        path = "/workspace/data/wt103_raw/train-00000.parquet"
        rdr = pq.ParquetFile(path)
        # Stream first row group, encode rows, slide window.
        col = next(c for c in rdr.schema.names if "text" in c.lower() or c == rdr.schema.names[0])
        all_ids: list[int] = []
        for batch_arr in rdr.iter_batches(batch_size=128, columns=[col]):
            for s in batch_arr.column(0).to_pylist():
                if not s:
                    continue
                all_ids.extend(tok.encode(s))
                all_ids.append(50256)
                if len(all_ids) > batch * (seq + 1) * 80:
                    break
            if len(all_ids) > batch * (seq + 1) * 80:
                break
        ids = torch.tensor(all_ids, dtype=torch.long)
        n_windows = ids.numel() // (seq + 1)
        ids = ids[: n_windows * (seq + 1)].view(n_windows, seq + 1)
        print(f"[data] real wt103: {n_windows} windows, vocab effective={int(ids.max().item())+1}")

        ptr = 0

        def _next() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal ptr
            if ptr + batch > n_windows:
                ptr = 0
            chunk = ids[ptr : ptr + batch]
            ptr += batch
            return chunk[:, :-1], chunk[:, 1:]

        return _next
    except Exception as e:
        print(f"[data] wt103 unavailable ({e}); falling back to randint.")

        def _rand() -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.randint(0, vocab, (batch, seq + 1), dtype=torch.long)
            return x[:, :-1], x[:, 1:]

        return _rand


def _count_flops_one_step(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    """Count fwd+bwd FLOPs for ONE realistic step using FlopCounterMode."""
    fc = FlopCounterMode(model, depth=None, display=False)
    with fc:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
    total = fc.get_total_flops()
    # Reset gradients so the next iter is clean.
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    return int(total)


def _wrap_model_with_backend(model: torch.nn.Module, backend: str) -> torch.nn.Module:
    """Apply optional Triton fusion. gpu_dense leaves as-is."""
    if backend == "gpu_dense":
        return model
    if backend == "triton_block":
        try:
            from synapforge.backends.triton_block_kernel import _HAS_TRITON
            if not _HAS_TRITON:
                print("[backend] Triton kernel unavailable; using gpu_dense.")
                return model
            from synapforge.backends.triton_block import TritonBlockBackend  # noqa
            from synapforge.runtime import compile as sf_compile
            rt = sf_compile(model, backend="triton_block")
            return rt
        except Exception as e:
            print(f"[backend] triton_block failed ({type(e).__name__}: {e}); falling back.")
            return model
    raise ValueError(backend)


def run(backend: str, steps: int, warmup: int, batch: int, seq: int) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = _device_info()
    print(f"[device] {info}")

    # Build model
    model = build_synapforge_100m()
    nparam = model.num_parameters()
    print(f"[model] params={nparam:,} ({nparam/1e6:.2f}M)")

    # bf16 cast for A100 (peak is bf16/fp16 312 TFLOPS).
    model = model.to(dev).to(torch.bfloat16)
    runner = _wrap_model_with_backend(model, backend)

    # Data
    next_batch = _real_data_iterator(batch, seq, vocab=50257)

    # Static FLOPs from one fwd+bwd
    x0, y0 = next_batch()
    x0 = x0.to(dev)
    y0 = y0.to(dev)
    print("[flops] counting per-step fwd+bwd FLOPs ...")
    # FlopCounter needs an nn.Module — Runtime wrapper isn't iterable; count
    # using the underlying root module (same compute graph).
    fc_target = runner.graph.modules.get('root') if hasattr(runner, 'graph') else runner
    flops_per_step = _count_flops_one_step(fc_target, x0, y0)
    print(f"[flops] {flops_per_step:,} = {flops_per_step/1e9:.1f} GFLOPs / step")

    # Warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"[warmup] {warmup} steps ...")
    for _ in range(warmup):
        x, y = next_batch()
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = runner(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Timed loop
    print(f"[time] {steps} measured steps ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(steps):
        x, y = next_batch()
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = runner(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    sec_per_step = elapsed / steps
    achieved_TFLOPS = (flops_per_step / sec_per_step) / 1e12
    mfu = achieved_TFLOPS / A100_BF16_PEAK_TFLOPS

    out = {
        "backend": backend,
        "params_M": round(nparam / 1e6, 3),
        "shape": {"batch": batch, "seq": seq, "hidden": 512, "layers": 10, "loop_depth": 4},
        "flops_per_step_G": round(flops_per_step / 1e9, 3),
        "sec_per_step": round(sec_per_step, 4),
        "tokens_per_sec": round(batch * seq / sec_per_step, 1),
        "achieved_TFLOPS": round(achieved_TFLOPS, 2),
        "peak_TFLOPS": A100_BF16_PEAK_TFLOPS,
        "MFU": round(mfu, 4),
        "MFU_pct": round(mfu * 100, 1),
        "peak_mem_GB": round(peak_mem, 2),
        "device": info,
    }
    print("[result]", json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gpu_dense", "triton_block"], default="gpu_dense")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", type=str, default="/workspace/runs/synapforge_mfu/result.json")
    args = ap.parse_args()
    out = run(args.backend, args.steps, args.warmup, args.batch, args.seq)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Append to a "log" of runs.
    log_path = args.out.replace(".json", "_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(out) + "\n")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
