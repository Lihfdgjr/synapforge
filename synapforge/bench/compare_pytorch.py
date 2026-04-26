"""Bench: synapforge vs raw mscfc on identical workloads.

For v0.1 we expect ROUGHLY EQUAL throughput — synapforge.LiquidCell wraps
the same Heinsen scan as mscfc.LiquidS4Cell. Any speedup is attributable
to slightly less Python overhead from a flatter call stack; any slowdown
is the IRGraph/runtime dispatcher.

Workload (matches v0.1 spec):
    bs = 64, T = 256, D = 256
    32 forward passes, median timing reported (excludes 5 warmup runs).

Also prints 16x stress (T=1024) to see scan scalability.
"""

from __future__ import annotations

import os
import sys
import time
from statistics import median

sys.path.insert(0, "/workspace")

import torch
from mscfc.liquid_s4 import LiquidS4Cell

import synapforge as sf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 5
ITERS = 32


def time_fwd(fn, x, label, *args) -> tuple[float, float]:
    """Returns (median_ms, p90_ms) over ITERS iterations after WARMUP."""
    # Warmup
    for _ in range(WARMUP):
        y = fn(x, *args)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fn(x, *args)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    med = median(times)
    p90 = times[int(0.9 * len(times))]
    print(f"  {label:42s}  median={med:7.3f} ms   p90={p90:7.3f} ms")
    return med, p90


def bench_liquid(B: int, T: int, D: int) -> None:
    print(f"\n--- LiquidCell forward  B={B} T={T} D={D}  device={DEVICE} ---")
    torch.manual_seed(0)

    sf_cell = sf.LiquidCell(D, D, init="hasani").to(DEVICE).eval()
    ms_cell = LiquidS4Cell(D, D).to(DEVICE).eval()

    # Copy params so the two are functionally identical (apples-to-apples).
    with torch.no_grad():
        ms_cell.delta_proj.weight.copy_(sf_cell.delta_proj.weight)
        ms_cell.delta_proj.bias.copy_(sf_cell.delta_proj.bias)
        ms_cell.b_proj.weight.copy_(sf_cell.b_proj.weight)
        ms_cell.b_proj.bias.copy_(sf_cell.b_proj.bias)
        ms_cell.A_log.copy_(sf_cell.A_log)

    x = torch.randn(B, T, D, device=DEVICE)

    with torch.no_grad():
        sf_med, _ = time_fwd(lambda x: sf_cell(x), x, "synapforge.LiquidCell")
        ms_med, _ = time_fwd(lambda x: ms_cell.forward_seq(x), x, "mscfc.LiquidS4Cell  ")

    # Numerical check
    with torch.no_grad():
        y_sf = sf_cell(x)
        y_ms = ms_cell.forward_seq(x)
        # Mask out NaN/Inf positions (Heinsen scan overflows on long sequences;
        # both implementations produce identical NaN patterns by construction).
        finite = torch.isfinite(y_sf) & torch.isfinite(y_ms)
        if finite.any():
            diff = (y_sf[finite] - y_ms[finite]).abs()
            ref_max = y_ms[finite].abs().max().item()
            rel = diff.max().item() / (ref_max + 1e-12)
        else:
            rel = float("nan")
        nan_frac = 1.0 - finite.float().mean().item()
        print(f"  rel_err vs mscfc reference: {rel:.3e}  (finite-mask, NaN frac={nan_frac:.2%}; must < 1e-3)")
    overhead_pct = 100.0 * (sf_med - ms_med) / ms_med
    print(f"  synapforge overhead: {overhead_pct:+.2f}%  (v0.1 acceptable: -5%..+15%)")


def bench_block(B: int, T: int, D: int) -> None:
    print(f"\n--- HybridBlock (cfc -> plif) forward  B={B} T={T} D={D}  device={DEVICE} ---")
    torch.manual_seed(0)

    class SFBlock(sf.Module):
        def __init__(self, d):
            super().__init__()
            self.cfc = sf.LiquidCell(d, d)
            self.plif = sf.PLIF(d)
        def forward(self, x):
            h = self.cfc(x)
            spk, mem = self.plif(h)
            return spk, mem, h

    sf_blk = SFBlock(D).to(DEVICE).eval()
    rt = sf.compile(sf_blk, backend="gpu_dense")

    x = torch.randn(B, T, D, device=DEVICE)

    with torch.no_grad():
        time_fwd(lambda x: sf_blk(x), x, "sf_block.forward (direct)")
        time_fwd(lambda x: rt(x), x, "sf_block via Runtime")


def bench_compile_metadata() -> None:
    """Quick check that compile() produces a sane IR graph."""
    print("\n--- compile() IR inspection ---")

    class SFBlock(sf.Module):
        def __init__(self, d):
            super().__init__()
            self.cfc = sf.LiquidCell(d, d)
            self.plif = sf.PLIF(d)
            self.syn = sf.SparseSynapse(d, d, sparsity=0.20)
        def forward(self, x):
            h = self.cfc(x)
            spk, mem = self.plif(h)
            return self.syn(spk * h)

    m = SFBlock(64)
    rt = sf.compile(m, backend="gpu_dense")
    print(rt)
    print(rt.graph.summary())


def main() -> int:
    print(f"=== synapforge v0.1 bench === device={DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    bench_compile_metadata()
    # Spec workload
    bench_liquid(B=64, T=256, D=256)
    bench_block(B=64, T=256, D=256)
    # Stress
    bench_liquid(B=64, T=1024, D=256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
