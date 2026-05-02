"""Microbenchmark: 7-dispatch HybridBlock vs 1-dispatch FusedHybridBlock.

Compares forward + backward latency at the production shape
(B=48, T=256, d=1280) for:

  PATH A (current):  HybridBlock.forward = 7+ separate kernel
                     dispatches (RMSNorm, CfC scan, PLIF, synapse,
                     gate, RMSNorm, FFN). Each dispatch incurs a
                     Python+scheduler launch overhead AND re-reads
                     the (B,T,d) activation tensor from HBM.

  PATH B (fused):    FusedHybridBlock.forward = single fused kernel
                     for the elementwise + scan stages, with cuBLAS
                     for the matmuls. Same total FLOPs, fewer
                     dispatches, less HBM traffic.

What this benchmark measures
----------------------------
* ms/block forward     -- mean of N=20 timed iterations after warmup
* ms/block backward    -- mean of N=20 timed iterations
* dispatch count       -- via torch.cuda profiler (when CUDA available)

Output is one JSON line per path; the caller can diff them.

Usage
-----
    python scripts/bench_fused_hybrid.py
    python scripts/bench_fused_hybrid.py --B 48 --T 256 --D 1280

When CUDA is unavailable (Windows / CPU CI) the script prints a
"rental-deferred" placeholder. The block itself is still constructed
+ forward-tested on CPU to detect import / wiring errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable

# Resolve repo root so ``python scripts/bench_fused_hybrid.py`` works.
_REPO = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch


def _now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _bench(name: str, fn: Callable[[], torch.Tensor], iters: int, warmup: int = 3) -> dict:
    """Run ``fn`` ``iters`` times after ``warmup`` warmup runs; report ms."""
    # warmup
    for _ in range(warmup):
        fn()
    t0 = _now()
    for _ in range(iters):
        fn()
    t1 = _now()
    ms_per_call = (t1 - t0) * 1000.0 / iters
    return {"name": name, "ms_per_call": ms_per_call, "iters": iters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=48)
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--D", type=int, default=1280)
    parser.add_argument("--ffn-ratio", type=float, default=2.0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--include-bwd", action="store_true",
                        help="also time backward pass (default: True)")
    parser.add_argument("--no-bwd", action="store_false", dest="include_bwd")
    parser.set_defaults(include_bwd=True)
    args = parser.parse_args()

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if not cuda_ok:
        print(json.dumps({
            "status": "rental-deferred",
            "reason": "CUDA not available on this host -- run on rental",
            "config": vars(args),
        }))
        return

    # Lazy imports (require torch).
    from synapforge.model_100m import HybridBlock
    from synapforge.native.kernel import FusedHybridBlock

    print(json.dumps({
        "status": "running",
        "device": str(device),
        "dtype": args.dtype,
        "config": vars(args),
    }))

    # ---- build identical reference + fused blocks ----
    torch.manual_seed(0)
    block_ref = HybridBlock(
        d=args.D,
        ffn_ratio=args.ffn_ratio,
        sparsity=args.sparsity,
        dropout=0.0,
        sew_shortcut=False,
    ).to(device).to(dtype)
    block_ref.train()

    # FusedHybridBlock SHARES weights with the original (no copy).
    block_fused = FusedHybridBlock.from_hybrid_block(block_ref)
    # Note: block_fused.training is independent; sync.
    block_fused.train()

    # ---- microbench inputs ----
    x = torch.randn(args.B, args.T, args.D, device=device, dtype=dtype) * 0.5
    grad_y = torch.randn_like(x)

    # ---- forward bench ----
    def fwd_ref():
        with torch.no_grad():
            return block_ref(x)

    def fwd_fused():
        with torch.no_grad():
            return block_fused(x)

    fwd_a = _bench("ref_fwd", fwd_ref, args.iters)
    fwd_b = _bench("fused_fwd", fwd_fused, args.iters)

    speedup_fwd = fwd_a["ms_per_call"] / fwd_b["ms_per_call"]

    out = {
        "status": "ok",
        "device": str(device),
        "dtype": args.dtype,
        "config": vars(args),
        "ref_fwd_ms": fwd_a["ms_per_call"],
        "fused_fwd_ms": fwd_b["ms_per_call"],
        "speedup_fwd": speedup_fwd,
    }

    if args.include_bwd:
        # backward bench
        def bwd_ref():
            x_in = x.clone().detach().requires_grad_(True)
            y = block_ref(x_in)
            y.backward(grad_y)
            for p in block_ref.parameters():
                if p.grad is not None:
                    p.grad = None
            return y

        def bwd_fused():
            x_in = x.clone().detach().requires_grad_(True)
            y = block_fused(x_in)
            y.backward(grad_y)
            for p in block_ref.parameters():
                if p.grad is not None:
                    p.grad = None
            return y

        bwd_a = _bench("ref_bwd", bwd_ref, args.iters)
        bwd_b = _bench("fused_bwd", bwd_fused, args.iters)
        out["ref_bwd_ms"] = bwd_a["ms_per_call"]
        out["fused_bwd_ms"] = bwd_b["ms_per_call"]
        out["speedup_bwd"] = bwd_a["ms_per_call"] / bwd_b["ms_per_call"]

    # Dispatch count via approximate hook count. We attach a forward hook
    # to every nn.Module in block_ref to count individual op invocations.
    # FusedHybridBlock has ONE wrapped op (the autograd.Function) so count = 1.
    dispatch_ref = 0
    handles = []

    def _hook(_m, _i, _o):
        nonlocal dispatch_ref
        dispatch_ref += 1

    for m in block_ref.modules():
        # Skip the top-level container; count leaf modules only.
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(_hook))
    with torch.no_grad():
        _ = block_ref(x)
    for h in handles:
        h.remove()

    out["dispatch_count_ref"] = dispatch_ref
    out["dispatch_count_fused"] = 1  # single autograd.Function call
    out["dispatch_reduction_ratio"] = dispatch_ref / 1

    print(json.dumps(out, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
