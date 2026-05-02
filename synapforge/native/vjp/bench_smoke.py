"""Minimal smoke benchmark for the VJP catalogue.

Run via:  python -m synapforge.native.vjp.bench_smoke

Times each fwd+bwd pair on a tiny batch and prints throughput. Pure
numpy; useful as a sanity check that no op regresses by 10x and as a
baseline for the CUDA / Triton adapters in sibling packages.

Output is plain text (no JSON) -- easy to eyeball and grep.
"""

from __future__ import annotations

import time
from typing import Callable, Tuple

import numpy as np

from . import cfc, cross_entropy, embed, linear, plif, rmsnorm, sew_shortcut, swiglu


def _bench(name: str, fn: Callable[[], Tuple[float, float]], n_iter: int = 50) -> None:
    # Warm up once (numpy may build BLAS kernel caches).
    fn()
    fwd_total, bwd_total = 0.0, 0.0
    for _ in range(n_iter):
        f, b = fn()
        fwd_total += f
        bwd_total += b
    fwd_us = 1e6 * fwd_total / n_iter
    bwd_us = 1e6 * bwd_total / n_iter
    print(f"{name:24s}  fwd {fwd_us:8.1f} us   bwd {bwd_us:8.1f} us")


def bench_linear() -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 32, 64)).astype(np.float32)
    W = rng.standard_normal((128, 64)).astype(np.float32) * 0.1
    b = rng.standard_normal((128,)).astype(np.float32) * 0.1
    gy = rng.standard_normal((4, 32, 128)).astype(np.float32)
    t0 = time.perf_counter()
    y = linear.linear_fwd(x, W, b)
    t1 = time.perf_counter()
    _ = linear.linear_bwd(gy, x, W, has_bias=True)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench_embed() -> Tuple[float, float]:
    rng = np.random.default_rng(1)
    V, d = 8000, 64
    ids = rng.integers(0, V, size=(4, 64)).astype(np.int64)
    W = rng.standard_normal((V, d)).astype(np.float32) * 0.1
    gy = rng.standard_normal((4, 64, d)).astype(np.float32)
    t0 = time.perf_counter()
    y = embed.embed_fwd(ids, W)
    t1 = time.perf_counter()
    _ = embed.embed_bwd(gy, ids, V)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench_rmsnorm() -> Tuple[float, float]:
    rng = np.random.default_rng(2)
    d = 128
    x = rng.standard_normal((4, 64, d)).astype(np.float32)
    g = rng.standard_normal((d,)).astype(np.float32) + 1.0
    gy = rng.standard_normal((4, 64, d)).astype(np.float32)
    t0 = time.perf_counter()
    y, rstd = rmsnorm.rmsnorm_fwd(x, g)
    t1 = time.perf_counter()
    _ = rmsnorm.rmsnorm_bwd(gy, x, g, rstd)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench_swiglu() -> Tuple[float, float]:
    rng = np.random.default_rng(3)
    d_in, h = 64, 256
    x = rng.standard_normal((4, 64, d_in)).astype(np.float32) * 0.3
    Wg = rng.standard_normal((h, d_in)).astype(np.float32) * 0.1
    Wu = rng.standard_normal((h, d_in)).astype(np.float32) * 0.1
    Wd = rng.standard_normal((d_in, h)).astype(np.float32) * 0.1
    gy = rng.standard_normal((4, 64, d_in)).astype(np.float32)
    t0 = time.perf_counter()
    y, saved = swiglu.swiglu_fwd(x, Wg, Wu, Wd)
    t1 = time.perf_counter()
    _ = swiglu.swiglu_bwd(gy, saved)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench_cfc_step() -> Tuple[float, float]:
    rng = np.random.default_rng(4)
    B, in_dim, d = 4, 64, 128
    h_prev = rng.standard_normal((B, d)).astype(np.float32) * 0.1
    x = rng.standard_normal((B, in_dim)).astype(np.float32) * 0.2
    W_in = rng.standard_normal((d, in_dim)).astype(np.float32) * 0.1
    W_h = rng.standard_normal((d, in_dim)).astype(np.float32) * 0.1
    A_log = rng.standard_normal((d,)).astype(np.float32) * 0.5
    grad_out = rng.standard_normal((B, d)).astype(np.float32)
    grad_h_next = np.zeros((B, d), dtype=np.float32)
    t0 = time.perf_counter()
    h_t, out_t, cache = cfc.cfc_step_fwd(h_prev, x, W_in, W_h, A_log)
    t1 = time.perf_counter()
    _ = cfc.cfc_step_bwd(grad_out, grad_h_next, cache)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench_plif() -> Tuple[float, float]:
    rng = np.random.default_rng(5)
    B, d = 4, 128
    v_prev = rng.standard_normal((B, d)).astype(np.float32) * 0.3
    x = rng.standard_normal((B, d)).astype(np.float32) * 0.2
    tau_log = rng.standard_normal((d,)).astype(np.float32) * 0.5
    thr = np.full((d,), 0.3, dtype=np.float32)
    grad_spike = rng.standard_normal((B, d)).astype(np.float32)
    grad_v_new = rng.standard_normal((B, d)).astype(np.float32)
    t0 = time.perf_counter()
    spike, v_new, saved = plif.plif_fwd(v_prev, x, tau_log, thr)
    t1 = time.perf_counter()
    _ = plif.plif_bwd(grad_spike, grad_v_new, saved)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench_sew() -> Tuple[float, float]:
    rng = np.random.default_rng(6)
    spike = rng.standard_normal((4, 128)).astype(np.float32)
    h_dense = rng.standard_normal((4, 128)).astype(np.float32)
    grad_y = rng.standard_normal((4, 128)).astype(np.float32)
    t0 = time.perf_counter()
    y = sew_shortcut.sew_fwd(spike, h_dense)
    t1 = time.perf_counter()
    _ = sew_shortcut.sew_bwd(grad_y)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench_ce() -> Tuple[float, float]:
    rng = np.random.default_rng(7)
    N, V = 256, 8000
    logits = rng.standard_normal((N, V)).astype(np.float32) * 0.5
    targets = rng.integers(0, V, size=(N,)).astype(np.int64)
    t0 = time.perf_counter()
    loss, saved = cross_entropy.ce_fwd(logits, targets, reduction="mean")
    t1 = time.perf_counter()
    _ = cross_entropy.ce_bwd(np.array(1.0, dtype=np.float32), saved)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def main() -> None:
    print("synapforge.native.vjp -- pure-numpy smoke benchmark")
    print("=" * 60)
    _bench("linear B4xT32xd64->128", bench_linear)
    _bench("embed V8000xd64 B4xT64", bench_embed)
    _bench("rmsnorm B4xT64xd128", bench_rmsnorm)
    _bench("swiglu B4xT64 d64->h256", bench_swiglu)
    _bench("cfc_step B4 d128", bench_cfc_step)
    _bench("plif B4 d128", bench_plif)
    _bench("sew B4 d128", bench_sew)
    _bench("ce N256 V8000", bench_ce)
    print("=" * 60)
    print("Done. Times are per-call (averaged over 50 iters after 1 warmup).")


if __name__ == "__main__":
    main()
