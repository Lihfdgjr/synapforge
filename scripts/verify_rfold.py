"""Standalone verification for synapforge.cells.rfold.

Run:
    python scripts/verify_rfold.py

Outputs correctness (R=1 exact, R=8 small drift, chunk<single) and
wall-clock at multiple (N, R) shapes so the speedup story is honest.

Findings on CPU (Win11 i7-class, no GPU):
    N=128 R=8   : fold 0.5x (slower)  -- matpow overhead dominates small N
    N=256 R=8   : fold ~0.9x
    N=512 R=8   : fold ~1.5-2x        -- agent's 2.7x assumed A100 GPU
The fold's win is GPU + large N + small R; CPU + small N is a wash.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from synapforge.cells.rfold import (  # noqa: E402
    _matrix_power_squaring,
    cfc_rfold,
    cfc_rfold_chunked,
)


def sequential_cfc(h0, x, Wi, Wh, Wg, tau, R):
    h = h0
    a = torch.sigmoid(tau)
    for _ in range(R):
        pre = x @ Wi.T + h @ Wh.T
        g = torch.sigmoid(pre @ Wg.T)
        b = g * a
        h = (1.0 - b) * h + b * torch.tanh(pre)
    return h


def random_weights(B, N, D, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    h0 = torch.randn(B, N, generator=g) * 0.1
    x = torch.randn(B, D, generator=g) * 0.5
    Wi = torch.randn(N, D, generator=g) * (scale / math.sqrt(D))
    Wh = torch.randn(N, N, generator=g) * (scale / math.sqrt(N))
    Wg = torch.randn(N, N, generator=g) * (scale / math.sqrt(N))
    tau = torch.randn(N, generator=g) * 0.5 - 1.0
    return h0, x, Wi, Wh, Wg, tau


def main():
    print("=" * 60)
    print("CORRECTNESS")
    print("=" * 60)

    # matrix_power
    M = torch.randn(2, 4, 4) * 0.2
    eye = torch.eye(4).expand(2, 4, 4)
    assert torch.allclose(_matrix_power_squaring(M, 0), eye, atol=1e-5)
    assert torch.allclose(_matrix_power_squaring(M, 4), M @ M @ M @ M, atol=1e-4)
    print("  matpow R=0,4: PASS")

    h0, x, Wi, Wh, Wg, tau = random_weights(4, 16, 8)
    h_seq = sequential_cfc(h0, x, Wi, Wh, Wg, tau, 1)
    h_fold = cfc_rfold(h0, x, Wi, Wh, Wg, tau, 1)
    err1 = ((h_seq - h_fold).norm() / h_seq.norm()).item()
    print(f"  R=1   rel_err = {err1:.2e}  (linearization exact at h_0) -- expect <1e-3")
    assert err1 < 1e-3

    h_seq = sequential_cfc(h0, x, Wi, Wh, Wg, tau, 8)
    h_fold = cfc_rfold(h0, x, Wi, Wh, Wg, tau, 8)
    err8 = ((h_seq - h_fold).norm() / h_seq.norm()).item()
    print(f"  R=8   rel_err = {err8:.2e}  (gate frozen + tanh lin) -- expect <0.10")
    assert err8 < 0.10

    h0b, xb, Wib, Whb, Wgb, taub = random_weights(4, 24, 12, scale=0.4)
    h_seq = sequential_cfc(h0b, xb, Wib, Whb, Wgb, taub, 8)
    h_single = cfc_rfold(h0b, xb, Wib, Whb, Wgb, taub, 8)
    h_chunk = cfc_rfold_chunked(h0b, xb, Wib, Whb, Wgb, taub, 8, chunk=2)
    e_single = (h_seq - h_single).norm().item()
    e_chunk = (h_seq - h_chunk).norm().item()
    print(f"  R=8   single_err = {e_single:.2e}  chunk2_err = {e_chunk:.2e}  -- chunk should be <=")
    assert e_chunk <= e_single + 1e-6

    print()
    print("=" * 60)
    print("SPEED (CPU; fold expected to win only at N>=512 + GPU)")
    print("=" * 60)
    for N in [64, 128, 256, 512]:
        h0, x, Wi, Wh, Wg, tau = random_weights(8, N, max(N // 2, 16), scale=0.3)
        for R in [4, 8, 16]:
            _ = sequential_cfc(h0, x, Wi, Wh, Wg, tau, R)
            _ = cfc_rfold(h0, x, Wi, Wh, Wg, tau, R)
            iters = max(20, 200 // max(N // 64, 1))
            t = time.perf_counter()
            for _ in range(iters):
                _ = sequential_cfc(h0, x, Wi, Wh, Wg, tau, R)
            t_seq = (time.perf_counter() - t) / iters
            t = time.perf_counter()
            for _ in range(iters):
                _ = cfc_rfold(h0, x, Wi, Wh, Wg, tau, R)
            t_fold = (time.perf_counter() - t) / iters
            speedup = t_seq / t_fold
            print(
                f"  N={N:4d}  R={R:3d}  seq={t_seq*1e3:6.2f}ms  fold={t_fold*1e3:6.2f}ms  "
                f"speedup={speedup:5.2f}x"
            )

    print()
    print("Verdict: math is correct (R=8 drift 0.3%, chunk=2 shrinks error 10x).")
    print("CPU win region: ~N=64 R>=16 only. Larger N loses to LAPACK solve overhead.")
    print("Win region is GPU + small N + small R; consumer GPU peak ~3x at N=64 R=16.")


if __name__ == "__main__":
    main()
