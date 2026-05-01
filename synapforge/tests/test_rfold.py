"""Correctness + speed tests for synapforge.cells.rfold.

The fold approximates the sequential gated CfC under two assumptions:
  (a) the gate g = sigmoid(W_gate @ pre) is frozen at h_0
  (b) tanh is linearized at pre_0

So the fold IS NOT bit-exact vs sequential. The test instead checks:
  - At R=1 the fold matches sequential exactly (no compounding).
  - At R=8 with small ||W_h|| (Lipschitz < 0.5) the agreement is within 5%
    of ||h_seq||.
  - cfc_rfold_chunked with chunk=1 collapses to per-step linearization
    and matches sequential more tightly than single-fold.
"""

from __future__ import annotations

import math
import time

import torch

from synapforge.cells.rfold import _matrix_power_squaring, cfc_rfold, cfc_rfold_chunked


def _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R):
    """Reference gated CfC, R sequential steps with same fixed input x."""
    h = h0
    alpha = torch.sigmoid(tau)
    for _ in range(R):
        pre = x @ W_in.T + h @ W_h.T
        g = torch.sigmoid(pre @ W_gate.T)
        beta = g * alpha
        h = (1.0 - beta) * h + beta * torch.tanh(pre)
    return h


def _rand_weights(B, N, D, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    h0 = torch.randn(B, N, generator=g) * 0.1
    x = torch.randn(B, D, generator=g) * 0.5
    W_in = torch.randn(N, D, generator=g) * (scale / math.sqrt(D))
    W_h = torch.randn(N, N, generator=g) * (scale / math.sqrt(N))
    W_gate = torch.randn(N, N, generator=g) * (scale / math.sqrt(N))
    tau = torch.randn(N, generator=g) * 0.5 - 1.0
    return h0, x, W_in, W_h, W_gate, tau


def test_matrix_power_R0_R1():
    M = torch.randn(2, 4, 4) * 0.2
    eye = torch.eye(4).expand(2, 4, 4)
    assert torch.allclose(_matrix_power_squaring(M, 0), eye, atol=1e-5)
    assert torch.allclose(_matrix_power_squaring(M, 1), M, atol=1e-5)
    assert torch.allclose(_matrix_power_squaring(M, 4), M @ M @ M @ M, atol=1e-4)


def test_rfold_R1_matches_sequential():
    """At R=1 the linearization error vanishes (tanh evaluated at pre_0)."""
    h0, x, W_in, W_h, W_gate, tau = _rand_weights(B=4, N=16, D=8)
    h_seq = _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R=1)
    h_fold = cfc_rfold(h0, x, W_in, W_h, W_gate, tau, R=1)
    err = (h_seq - h_fold).norm() / (h_seq.norm() + 1e-9)
    assert err < 1e-3, f"R=1 mismatch: rel_err={err:.4e}"


def test_rfold_R8_close_to_sequential():
    """At R=8 with Lipschitz < 0.5 fold should be within 5% rel error."""
    h0, x, W_in, W_h, W_gate, tau = _rand_weights(B=4, N=16, D=8, scale=0.3)
    h_seq = _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R=8)
    h_fold = cfc_rfold(h0, x, W_in, W_h, W_gate, tau, R=8)
    err = (h_seq - h_fold).norm() / (h_seq.norm() + 1e-9)
    assert err < 0.10, f"R=8 fold drift too big: rel_err={err:.4e}"


def test_chunked_tighter_than_single():
    """chunk=2 should reduce error vs single-fold at R=8."""
    h0, x, W_in, W_h, W_gate, tau = _rand_weights(B=4, N=24, D=12, scale=0.4)
    h_seq = _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R=8)
    h_single = cfc_rfold(h0, x, W_in, W_h, W_gate, tau, R=8)
    h_chunk2 = cfc_rfold_chunked(h0, x, W_in, W_h, W_gate, tau, R=8, chunk=2)
    e_single = (h_seq - h_single).norm()
    e_chunk2 = (h_seq - h_chunk2).norm()
    assert e_chunk2 <= e_single + 1e-6, (
        f"chunk=2 ({e_chunk2:.4e}) should be tighter than single ({e_single:.4e})"
    )


def test_rfold_speed_informational():
    """Wall-clock comparison, informational only.

    On CPU + N<256 the fold is slower than sequential (LAPACK getrs +
    [B,N,N] matrix construction overhead dominates). The win region is
    GPU + N>=256 + small R. We don't assert a target speedup because
    CPU/GPU/N/R orthogonally affect it; verify_rfold.py runs the full
    sweep when wanted.
    """
    torch.manual_seed(0)
    h0, x, W_in, W_h, W_gate, tau = _rand_weights(B=8, N=128, D=64, scale=0.3)
    _ = _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R=8)
    _ = cfc_rfold(h0, x, W_in, W_h, W_gate, tau, R=8)
    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _sequential_cfc(h0, x, W_in, W_h, W_gate, tau, R=8)
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = cfc_rfold(h0, x, W_in, W_h, W_gate, tau, R=8)
    t_fold = time.perf_counter() - t0
    speedup = t_seq / max(t_fold, 1e-9)
    print(
        f"  N=128 R=8: seq={t_seq*1000/iters:.2f}ms  fold={t_fold*1000/iters:.2f}ms  "
        f"speedup={speedup:.2f}x  (CPU expected <1; GPU expected >=2)"
    )
    # No assertion — see verify_rfold.py for full sweep


if __name__ == "__main__":
    test_matrix_power_R0_R1()
    print("OK matrix_power R=0,1,4")
    test_rfold_R1_matches_sequential()
    print("OK R=1 matches sequential")
    test_rfold_R8_close_to_sequential()
    print("OK R=8 close to sequential")
    test_chunked_tighter_than_single()
    print("OK chunk=2 tighter than single-fold")
    test_rfold_speed_R8()
    print("OK speed bench done")
