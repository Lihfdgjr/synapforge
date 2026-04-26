"""bench_cpu_avx2 -- compare PyTorch eager vs numba-fused HybridBlock on CPU.

Shape: B=8 T=128 D=128 (small CPU shape per task spec). Reports:
  - JIT compile time (one-shot, amortized after first call)
  - ms/iter and tokens/sec for each path
  - speedup ratio
  - rel-error vs PyTorch reference
  - CPU util sample (top -bn1 average %CPU during a 30-iter run)

Usage:
    /opt/conda/bin/python /workspace/synapforge/bench_cpu_avx2.py
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace")

from synapforge.backends.cpu_avx2 import (  # noqa: E402
    _HAS_NUMBA,
    NumbaHybridBlock,
)
from synapforge.cells.liquid import LiquidCell  # noqa: E402
from synapforge.cells.plif import PLIF  # noqa: E402

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

B, T, D = 8, 128, 128
WARMUP = 3
N_ITERS = 30


# ---------------------------------------------------------------------------
# Reference PyTorch eager block
# ---------------------------------------------------------------------------


class TorchHybridBlock(nn.Module):
    """Reference: CfC over the whole sequence, then PLIF stepped over T.

    The numba kernel does PLIF with stateful subtract-on-spike reset across
    timesteps. The vanilla PLIF.forward is single-step (membrane=None ->
    zeros). To compare like-for-like we loop PLIF over T here, threading the
    membrane forward."""
    def __init__(self, d):
        super().__init__()
        self.cfc = LiquidCell(d, d)
        self.plif = PLIF(d, threshold=0.3)

    def forward(self, x):
        y = self.cfc(x)
        B, T, D = y.shape
        mem = torch.zeros(B, D, dtype=y.dtype, device=y.device)
        spks = []
        mems = []
        for t in range(T):
            spk_t, mem = self.plif(y[:, t], membrane=mem)
            spks.append(spk_t)
            mems.append(mem)
        return y, torch.stack(spks, dim=1), torch.stack(mems, dim=1)


def time_block(fn, n_warmup=WARMUP, n_iter=N_ITERS):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    dt = time.perf_counter() - t0
    return dt / n_iter * 1000.0  # ms/iter


def cpu_util_during(fn, duration_s=3.0):
    """Sample CPU utilisation while fn() runs in a background thread."""
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                with open("/proc/stat") as f:
                    s = f.readline().split()
                idle = int(s[4])
                total = sum(int(x) for x in s[1:8])
                samples.append((total, idle))
            except Exception:
                pass
            time.sleep(0.1)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        fn()
    stop.set()
    th.join()
    if len(samples) < 2:
        return -1.0
    t0, i0 = samples[0]
    t1, i1 = samples[-1]
    busy = (t1 - t0) - (i1 - i0)
    if (t1 - t0) <= 0:
        return -1.0
    return 100.0 * busy / (t1 - t0)


def main():
    print("=" * 72)
    print(f"bench_cpu_avx2  B={B} T={T} D={D}  (numba available: {_HAS_NUMBA})")
    print(f"CPU cores (nproc): {os.cpu_count()}")
    print("=" * 72)

    torch.manual_seed(0)
    np.random.seed(0)

    # --- Build models ----------------------------------------------------
    torch_block = TorchHybridBlock(D).eval()
    for p in torch_block.parameters():
        p.requires_grad_(False)

    numba_block = NumbaHybridBlock(d_in=D, d_hidden=D).eval()
    # Copy weights so the two blocks compute the same function.
    with torch.no_grad():
        numba_block.delta_proj.weight.copy_(torch_block.cfc.delta_proj.weight)
        numba_block.delta_proj.bias.copy_(torch_block.cfc.delta_proj.bias)
        numba_block.b_proj.weight.copy_(torch_block.cfc.b_proj.weight)
        numba_block.b_proj.bias.copy_(torch_block.cfc.b_proj.bias)
        numba_block.A_log.copy_(torch_block.cfc.A_log)
        thr = torch_block.plif.threshold
        if torch.is_tensor(thr):
            numba_block.threshold.copy_(thr)
        else:
            numba_block.threshold.fill_(float(thr))
        numba_block.log_tau.copy_(torch_block.plif.tau_log)
    for p in numba_block.parameters():
        p.requires_grad_(False)

    x = torch.randn(B, T, D)

    # --- JIT warmup (one-shot) -------------------------------------------
    print("\n[1] JIT warmup")
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = numba_block(x)
    jit_t = time.perf_counter() - t0
    print(f"  numba kernel first call (compile + run): {jit_t*1000:.1f} ms")

    # --- Correctness check -----------------------------------------------
    print("\n[2] Correctness vs PyTorch eager")
    with torch.no_grad():
        y_ref, spk_ref, mem_ref = torch_block(x)
        y_n, spk_n, mem_n = numba_block(x)

    def rel(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        return float((a - b).abs().max() / (b.abs().max() + 1e-12))

    print(f"  rel_err  y  : {rel(y_n, y_ref):.2e}  (eager max: {y_ref.abs().max():.3f})")
    print(f"  rel_err spk : {rel(spk_n, spk_ref):.2e}  (rate ref: {spk_ref.mean():.3f}, numba: {spk_n.mean():.3f})")
    print(f"  rel_err mem : {rel(mem_n, mem_ref):.2e}  (eager max: {mem_ref.abs().max():.3f})")

    # --- Bench ----------------------------------------------------------
    print("\n[3] Speed bench")

    def run_torch():
        with torch.no_grad():
            torch_block(x)

    def run_numba():
        with torch.no_grad():
            numba_block(x)

    # Single-thread torch baseline (more deterministic)
    torch.set_num_threads(os.cpu_count() or 28)
    t_torch = time_block(run_torch)
    t_numba = time_block(run_numba)

    tok = B * T
    print(f"  torch eager : {t_torch:7.3f} ms/iter   {tok/(t_torch/1000):>10.0f} tok/s")
    print(f"  numba fused : {t_numba:7.3f} ms/iter   {tok/(t_numba/1000):>10.0f} tok/s")
    print(f"  speedup     : {t_torch/t_numba:.2f}x")

    # --- CPU util ---------------------------------------------------------
    print("\n[4] CPU util (28-core box, prange should saturate cores)")
    util_torch = cpu_util_during(run_torch, duration_s=3.0)
    util_numba = cpu_util_during(run_numba, duration_s=3.0)
    print(f"  torch eager util: {util_torch:5.1f}%")
    print(f"  numba fused util: {util_numba:5.1f}%")

    # --- Triton GPU sanity (best-effort) ----------------------------------
    print("\n[5] Triton GPU sanity (best-effort, skipped on CPU-only)")
    try:
        if torch.cuda.is_available():
            from synapforge.backends.triton_block_kernel import (
                _HAS_TRITON,
                TritonHybridBlock,
            )
            if _HAS_TRITON:
                gpu = TritonHybridBlock(d_in=D, d_hidden=D).cuda().eval()
                xg = x.cuda()
                with torch.no_grad():
                    for _ in range(3):
                        gpu(xg)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(30):
                        gpu(xg)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / 30 * 1000
                print(f"  triton GPU  : {dt:7.3f} ms/iter   {tok/(dt/1000):>10.0f} tok/s")
            else:
                print("  triton not installed -> skipped")
        else:
            print("  no CUDA on this box -> skipped (CPU-only run)")
    except Exception as e:
        print(f"  triton path error: {e}")

    print("=" * 72)
    print("DONE")


if __name__ == "__main__":
    main()
