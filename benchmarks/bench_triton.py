"""bench_triton -- gpu_dense vs triton_block forward/backward speed.

Builds a 3-layer HybridBlock stack (D=256, T=256, B=64) and times:
    fwd       (just forward, no autograd)
    bwd       (backward only, given a forward already done)
    fwd+bwd   (full step)

The 'gpu_dense' baseline uses _Fp32AccumRef (per-step Python loop with fp32
accumulator; matches the Triton kernel's internal fp32 accumulator). The
'triton_block' path uses TritonHybridBlock with the fused kernel. Both
compute the same function so the rel_err < 1e-3 gate is meaningful in bf16.

Run:
    CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python /workspace/synapforge/bench_triton.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback

# Make synapforge importable when run from anywhere.
sys.path.insert(0, "/workspace")

import torch
import torch.nn as nn
import torch.nn.functional as F

import synapforge as sf
from synapforge.backends.triton_block_kernel import (
    TritonHybridBlock,
    _HAS_TRITON,
)


# ---------------------------------------------------------------------------
# Reference HybridBlock with fp32 accumulator (matches Triton kernel math).
# Same parameter shapes as TritonHybridBlock so we can copy weights between.
# ---------------------------------------------------------------------------


class _Fp32AccumRef(nn.Module):
    """Pure-PyTorch reference with fp32 internal scan accumulator.

    The fused Triton kernel upcasts (a,b,h,thr) to fp32 inside the kernel
    body and downcasts only when storing. To make a fair bf16 comparison
    we mirror that here: matmuls/elementwise stay in input dtype, but the
    scalar scan loop runs in fp32.
    """

    def __init__(self, d_in: int, d_hidden: int, alpha: float = 2.0):
        super().__init__()
        self.d_in = int(d_in)
        self.hidden_size = int(d_hidden)
        self.delta_proj = nn.Linear(d_in, d_hidden)
        self.b_proj = nn.Linear(d_in, d_hidden)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, d_hidden)))
        self.threshold = nn.Parameter(torch.ones(d_hidden))
        nn.init.constant_(self.delta_proj.bias, 0.0)
        self.out_norm = nn.LayerNorm(d_hidden)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor, h0=None):
        B, T, _ = x.shape
        D = self.hidden_size
        delta = F.softplus(self.delta_proj(x))             # input dtype
        b_step = delta * self.b_proj(x)                    # input dtype
        A = torch.exp(self.A_log)                          # fp32 (param)
        A_step = torch.exp(-delta * A.to(delta.dtype))     # input dtype

        # Run the scan in fp32 to match the kernel's accumulator.
        a_f32 = A_step.float()
        b_f32 = b_step.float()
        thr_f32 = self.threshold.float()

        h = a_f32.new_zeros(B, D) if h0 is None else h0.float().clone()
        h_post_buf = a_f32.new_empty(B, T, D)
        spikes_buf = a_f32.new_empty(B, T, D)
        for t in range(T):
            h = a_f32[:, t] * h + b_f32[:, t]
            m = h - thr_f32
            s = (m > 0).float()
            h = h * (1.0 - s)
            h_post_buf[:, t] = h
            spikes_buf[:, t] = s

        h_post = h_post_buf.to(x.dtype)
        spikes = spikes_buf.to(x.dtype)
        return self.out_norm(h_post), spikes


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------


class StackTriton(nn.Module):
    def __init__(self, d: int, n_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TritonHybridBlock(d, d, enable_stdp=False) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x, _spk = blk(x)
        return x


class StackDense(nn.Module):
    def __init__(self, d: int, n_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_Fp32AccumRef(d, d) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x, _spk = blk(x)
        return x


def _copy_weights(src: nn.Module, dst: nn.Module) -> None:
    src_layers = list(src.layers)
    dst_layers = list(dst.layers)
    assert len(src_layers) == len(dst_layers)
    with torch.no_grad():
        for s, d in zip(src_layers, dst_layers):
            d.delta_proj.weight.copy_(s.delta_proj.weight)
            d.delta_proj.bias.copy_(s.delta_proj.bias)
            d.b_proj.weight.copy_(s.b_proj.weight)
            d.b_proj.bias.copy_(s.b_proj.bias)
            d.A_log.copy_(s.A_log)
            d.threshold.copy_(s.threshold)
            d.out_norm.weight.copy_(s.out_norm.weight)
            d.out_norm.bias.copy_(s.out_norm.bias)


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def _cuda_timer():
    _sync()
    t0 = time.perf_counter()
    try:
        yield lambda: time.perf_counter() - t0
    finally:
        _sync()


def _bench_one(model, x, n_iters: int, mode: str):
    # warmup
    for _ in range(5):
        if mode == "fwd":
            with torch.no_grad():
                _ = model(x)
        else:
            x_in = x.detach().requires_grad_(True)
            y = model(x_in)
            loss = y.float().mean()
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
    _sync()

    times = []
    for _ in range(n_iters):
        if mode == "fwd":
            with _cuda_timer() as elapsed:
                with torch.no_grad():
                    _ = model(x)
            times.append(elapsed())
        elif mode == "fwdbwd":
            x_in = x.detach().requires_grad_(True)
            with _cuda_timer() as elapsed:
                y = model(x_in)
                loss = y.float().mean()
                loss.backward()
            times.append(elapsed())
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
        elif mode == "bwd":
            x_in = x.detach().requires_grad_(True)
            y = model(x_in)
            loss = y.float().mean()
            _sync()
            t0 = time.perf_counter()
            loss.backward()
            _sync()
            times.append(time.perf_counter() - t0)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
        else:
            raise ValueError(mode)
    return 1000.0 * sum(times) / len(times)


def _correctness(model_a, model_b, x):
    with torch.no_grad():
        y_a = model_a(x).float()
        y_b = model_b(x).float()
    err = (y_a - y_b).abs().max().item()
    mag = y_a.abs().mean().item() + 1e-12
    return err / mag, err


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument(
        "--out", default="/workspace/runs/synapforge_triton_bench.log",
    )
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
    dtype = dtype_map[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)

    print(f"== synapforge triton_block bench ==", flush=True)
    print(f"shape (B={args.B}, T={args.T}, D={args.D})  layers={args.n_layers}  dtype={args.dtype}")
    print(f"device={device}  torch={torch.__version__}  triton_avail={_HAS_TRITON}")
    if device.type == "cuda":
        print(f"cuda device name={torch.cuda.get_device_name(0)}", flush=True)

    triton_stack = StackTriton(args.D, args.n_layers).to(device=device, dtype=dtype)
    dense_stack = StackDense(args.D, args.n_layers).to(device=device, dtype=dtype)
    _copy_weights(triton_stack, dense_stack)

    # Smoke fwd
    x_warmup = torch.zeros(2, 8, args.D, device=device, dtype=dtype)
    try:
        _ = triton_stack(x_warmup)
        _ = dense_stack(x_warmup)
    except Exception as exc:
        print(f"FATAL: model warmup failed: {exc!r}")
        traceback.print_exc()
        return 2

    # -- Correctness check ----------------------------------------------
    x_small = torch.randn(8, 32, args.D, device=device, dtype=dtype) * 0.5
    rel_small, abs_small = _correctness(dense_stack, triton_stack, x_small)

    x = torch.randn(args.B, args.T, args.D, device=device, dtype=dtype) * 0.5
    rel_full, abs_full = _correctness(dense_stack, triton_stack, x)

    print(
        f"correctness: small_rel_err={rel_small:.3e}  full_rel_err={rel_full:.3e}  "
        f"ok={rel_full < 1e-3}",
        flush=True,
    )

    results = {
        "config": {
            "B": args.B, "T": args.T, "D": args.D,
            "n_layers": args.n_layers, "dtype": args.dtype,
            "n_iters": args.n_iters,
        },
        "correctness": {
            "small_rel_err": rel_small, "small_abs_err": abs_small,
            "full_rel_err": rel_full, "full_abs_err": abs_full,
            "ok": rel_full < 1e-3,
        },
        "ms_per_iter": {},
    }

    for mode in ("fwd", "bwd", "fwdbwd"):
        ms_d = _bench_one(dense_stack, x, args.n_iters, mode=mode)
        ms_t = _bench_one(triton_stack, x, args.n_iters, mode=mode)
        speedup = ms_d / ms_t if ms_t > 0 else float("inf")
        results["ms_per_iter"][mode] = {
            "gpu_dense": ms_d, "triton_block": ms_t, "speedup": speedup,
        }
        print(
            f"  {mode:7s}  gpu_dense={ms_d:9.2f} ms  triton_block={ms_t:9.2f} ms  "
            f"speedup={speedup:6.2f}x",
            flush=True,
        )

    print()
    print(json.dumps(results, indent=2))

    try:
        with open(args.out + ".json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as exc:
        print(f"warn: failed to write json sidecar: {exc!r}")

    return 0 if rel_full < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
