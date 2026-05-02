"""Throughput benchmark for the SEW + sigmoid-gate fused backward.

Compares two forward+backward paths for the elementwise tail of one
HybridBlock pass at the production shape (B=24, T=256, d=1280):

  1. ``torch_path``  — naive ``s + h``, ``sigmoid(gp)``, ``syn * gate``
                       through torch.autograd.
  2. ``fused_path``  — ``sew_sigmoid_gate_fused(...)`` (Triton fwd+bwd
                       when CUDA is available, torch fallback otherwise).

We isolate just the SEW + sigmoid + mul fusion (the new kernel), NOT
the full HybridBlock, because:

* The CfC + PLIF + reset path is ALREADY Triton-fused (via
  ``triton_block_kernel.py``) and benchmarked separately.
* The Linear / SparseSynapse projections are cuBLAS — not on the
  Python-overhead bottleneck this PR addresses.

This benchmark answers "did the new kernel reduce the elementwise tail
overhead?" — the metric the user asked for ("tok/s improvement").

Usage
-----
    python scripts/bench_hybrid_block.py
    python scripts/bench_hybrid_block.py --B 24 --T 256 --D 1280

Output is a one-line JSON record per path. On a no-GPU host the script
prints a "rental-deferred" placeholder (we cannot run Triton kernels
without CUDA).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow `python scripts/bench_hybrid_block.py` from the repo root without
# `pip install -e .`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


def _torch_path_step(s, h, syn_out, gate_pre, grad_spike_input, grad_gated):
    """Naive torch chain — same as the SEWSigmoidGateFn forward but
    using only stock ops (each one a separate kernel launch)."""
    s_r = s.clone().detach().requires_grad_(True)
    h_r = h.clone().detach().requires_grad_(True)
    syn_r = syn_out.clone().detach().requires_grad_(True)
    gp_r = gate_pre.clone().detach().requires_grad_(True)

    spike_input = s_r + h_r
    gate = torch.sigmoid(gp_r)
    gated = syn_r * gate

    torch.autograd.backward(
        [spike_input, gated],
        [grad_spike_input, grad_gated],
    )
    return spike_input, gated


def _fused_path_step(s, h, syn_out, gate_pre, grad_spike_input, grad_gated, fused_fn):
    s_f = s.clone().detach().requires_grad_(True)
    h_f = h.clone().detach().requires_grad_(True)
    syn_f = syn_out.clone().detach().requires_grad_(True)
    gp_f = gate_pre.clone().detach().requires_grad_(True)

    spike_input, gated = fused_fn(s_f, h_f, syn_f, gp_f)
    torch.autograd.backward(
        [spike_input, gated],
        [grad_spike_input, grad_gated],
    )
    return spike_input, gated


def _bench_path(name, fn, n_warmup, n_iter, on_cuda):
    if on_cuda:
        torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    if on_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if on_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "path": name,
        "iters": n_iter,
        "elapsed_s": elapsed,
        "ms_per_iter": (elapsed / n_iter) * 1000.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--D", type=int, default=1280)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument(
        "--cuda-only",
        action="store_true",
        help="If set, abort on no-GPU host instead of running CPU-only.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        msg = (
            "rental-deferred: no CUDA device available on this host. "
            "Run on the rental (117.74.66.77:41614) with the same args. "
            "Local Windows has torch=cpu only — Triton kernels need CUDA."
        )
        print(json.dumps({"status": "rental-deferred", "reason": msg}, indent=2))
        if args.cuda_only:
            sys.exit(2)
        # Still produce a CPU baseline to validate the autograd path correctness;
        # tok/s numbers from CPU are NOT comparable to the rental.
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    # Lazy import after we've decided the device — keeps the script
    # importable on no-GPU CI.
    try:
        from synapforge.backends.triton_block_kernel_bwd import sew_sigmoid_gate_fused
    except Exception as exc:
        print(json.dumps({"status": "import-error", "error": str(exc)}))
        sys.exit(1)

    print(f"# device={device} dtype={dtype} B={args.B} T={args.T} D={args.D}")

    torch.manual_seed(0)
    s = (torch.rand(args.B, args.T, args.D, device=device) > 0.7).to(dtype)
    h = torch.randn(args.B, args.T, args.D, device=device, dtype=dtype) * 0.5
    syn_out = torch.randn(args.B, args.T, args.D, device=device, dtype=dtype) * 0.3
    gate_pre = torch.randn(args.B, args.T, args.D, device=device, dtype=dtype) * 0.4
    grad_spike_input = torch.randn(args.B, args.T, args.D, device=device, dtype=dtype)
    grad_gated = torch.randn(args.B, args.T, args.D, device=device, dtype=dtype)

    on_cuda = device.type == "cuda"

    torch_result = _bench_path(
        "torch_path",
        lambda: _torch_path_step(s, h, syn_out, gate_pre, grad_spike_input, grad_gated),
        args.warmup,
        args.iters,
        on_cuda,
    )
    fused_result = _bench_path(
        "fused_path",
        lambda: _fused_path_step(
            s, h, syn_out, gate_pre, grad_spike_input, grad_gated,
            sew_sigmoid_gate_fused,
        ),
        args.warmup,
        args.iters,
        on_cuda,
    )

    # Tokens per iteration = B * T (each loop processes that many tokens
    # of context for ALL D channels).
    tokens_per_iter = args.B * args.T
    torch_result["tok_per_s"] = (
        tokens_per_iter / (torch_result["ms_per_iter"] / 1000.0)
    )
    fused_result["tok_per_s"] = (
        tokens_per_iter / (fused_result["ms_per_iter"] / 1000.0)
    )
    speedup = torch_result["ms_per_iter"] / max(fused_result["ms_per_iter"], 1e-9)

    summary = {
        "device": str(device),
        "dtype": str(dtype),
        "shape": {"B": args.B, "T": args.T, "D": args.D},
        "torch_path": torch_result,
        "fused_path": fused_result,
        "speedup_torch_over_fused": speedup,
        "elementwise_tail_only": True,
        "note": (
            "This benchmark only measures the elementwise tail "
            "(s+h, sigmoid, mul) of one HybridBlock pass. The full "
            "HybridBlock includes Linear/SparseSynapse/CfC/PLIF — "
            "those are cuBLAS or already Triton-fused (see "
            "synapforge/backends/triton_block_kernel.py)."
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
