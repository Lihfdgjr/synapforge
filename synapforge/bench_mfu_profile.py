"""bench_mfu_profile.py — torch.profiler trace of synapforge_100m steps.

Saves Chrome trace + a top-K op summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "1"))

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule

sys.path.insert(0, "/workspace")
from synapforge.model_100m import build_synapforge_100m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gpu_dense", "triton_block"], default="triton_block")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", type=str, default="/workspace/runs/synapforge_mfu/profile.json")
    ap.add_argument("--summary", type=str, default="/workspace/runs/synapforge_mfu/profile_summary.txt")
    args = ap.parse_args()

    dev = torch.device("cuda")
    model = build_synapforge_100m().to(dev).to(torch.bfloat16)

    runner = model
    if args.backend == "triton_block":
        from synapforge.runtime import compile as sf_compile
        runner = sf_compile(model, backend="triton_block")

    # Real-ish data
    x = torch.randint(0, 50257, (32, 256), device=dev)
    y = torch.randint(0, 50257, (32, 256), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        L = F.cross_entropy(runner(x).reshape(-1, 50257), y.reshape(-1))
        L.backward()
        opt.step()
    torch.cuda.synchronize()

    # Profile
    sch = schedule(wait=1, warmup=2, active=args.steps, repeat=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sch,
        record_shapes=False,
        with_stack=False,
        with_flops=False,
    ) as prof:
        for _ in range(args.steps + 3):
            opt.zero_grad(set_to_none=True)
            L = F.cross_entropy(runner(x).reshape(-1, 50257), y.reshape(-1))
            L.backward()
            opt.step()
            prof.step()

    prof.export_chrome_trace(args.out)

    table_cuda = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    table_cpu = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    with open(args.summary, "w") as f:
        f.write(f"# synapforge_100m profile, backend={args.backend}, steps={args.steps}\n\n")
        f.write("## TOP 10 OPS BY self_cuda_time:\n")
        f.write(table_cuda + "\n\n")
        f.write("## TOP 10 OPS BY self_cpu_time:\n")
        f.write(table_cpu + "\n\n")
    print(table_cuda)
    print(f"[saved] trace -> {args.out}")
    print(f"[saved] summary -> {args.summary}")


if __name__ == "__main__":
    main()
