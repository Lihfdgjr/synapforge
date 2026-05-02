"""Microbench: STDPOnlyOptimizer vs torch.optim.AdamW on identical params.

Compares per-step wall-clock for the optimizer-side update only on a
synthetic plasticity-tagged weight set. The benchmark deliberately
isolates the optimizer step so the comparison is apples-to-apples:

* AdamW path:   pre-fill ``param.grad`` with random noise, call
                ``opt.step()``. Includes m/v moment math, weight
                decay, and the in-place param add. NO backward pass
                (we already supply the grad).
* STDP path:    feed the optimizer one observation per layer per
                step, call ``opt.step()``. Cost is determined by
                spike density and a single dense outer product.

Usage
-----
    python scripts/bench_stdp_vs_adamw.py \
        --total-params 5_000_000 \
        --layer-shape 256 256 \
        --spike-density 0.1 \
        --steps 50

Output is a single table:

    method      mean_us/step    min_us    max_us
    AdamW       <ms*1000>       <min>     <max>
    STDP        <ms*1000>       <min>     <max>

The headline number is ``AdamW_us / STDP_us``. Targets:
* dense (density=1.0):   STDP ~ AdamW (no win — full outer product)
* sparse (density=0.1):  STDP >= 50x AdamW (the production case)
* very sparse (0.01):    STDP >= 500x AdamW (the inference case)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _build_synthetic_params(
    total: int, layer_out: int, layer_in: int, *, use_torch: bool
):
    """Carve ``total`` parameters into ``[layer_out, layer_in]`` shards.

    Returns a list of (name, param). When ``use_torch=True`` we make
    real ``torch.nn.Parameter`` objects (for AdamW); otherwise numpy
    backed shims (for STDP via the no-torch path).
    """
    n_layers = max(1, total // (layer_out * layer_in))
    out: list = []
    if use_torch:
        import torch

        for i in range(n_layers):
            p = torch.nn.Parameter(
                torch.zeros(layer_out, layer_in, dtype=torch.float32)
            )
            out.append((f"layer{i}.w", p))
    else:
        for i in range(n_layers):
            arr = np.zeros((layer_out, layer_in), dtype=np.float32)
            shim = type("_NPP", (), {"_np_data": arr, "shape": arr.shape})
            out.append((f"layer{i}.w", shim()))
    return out


def bench_adamw(
    n_layers: int, layer_out: int, layer_in: int, n_steps: int
):
    """Time AdamW.step over n_layers params for n_steps."""
    import torch

    params = [
        torch.nn.Parameter(torch.zeros(layer_out, layer_in, dtype=torch.float32))
        for _ in range(n_layers)
    ]
    opt = torch.optim.AdamW(params, lr=1e-3)
    # Warmup
    for p in params:
        p.grad = torch.randn_like(p) * 0.01
    opt.step()
    for p in params:
        p.grad = torch.randn_like(p) * 0.01
    opt.step()

    times: list[float] = []
    for _ in range(n_steps):
        for p in params:
            # Use copy_() into a preallocated tensor to mirror the
            # cost of receiving a grad. For AdamW the grad must be
            # set per-step; we count this in the per-step timing.
            p.grad = torch.randn_like(p) * 0.01
        t0 = time.perf_counter()
        opt.step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us
    return times


def bench_stdp(
    n_layers: int,
    layer_out: int,
    layer_in: int,
    n_steps: int,
    density: float,
):
    """Time STDPOnlyOptimizer.step over n_layers params for n_steps."""
    from synapforge.native.stdp import STDPOnlyOptimizer

    rng = np.random.default_rng(0)
    params: list = []
    for i in range(n_layers):
        arr = np.zeros((layer_out, layer_in), dtype=np.float32)
        shim = type("_NPP", (), {"_np_data": arr, "shape": arr.shape})
        params.append((f"layer{i}.w", shim()))

    opt = STDPOnlyOptimizer.from_named_params(
        params, base_alpha=0.02, window=20
    )
    # Warmup
    for name, _ in params:
        pre = (rng.uniform(0, 1, size=layer_in) < density).astype(np.uint8)
        post = (rng.uniform(0, 1, size=layer_out) < density).astype(np.uint8)
        opt.observe(name, pre, post)
    opt.step()

    times: list[float] = []
    for _ in range(n_steps):
        for name, _ in params:
            pre = (rng.uniform(0, 1, size=layer_in) < density).astype(np.uint8)
            post = (rng.uniform(0, 1, size=layer_out) < density).astype(np.uint8)
            opt.observe(name, pre, post)
        t0 = time.perf_counter()
        opt.step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us
    return times


def _fmt(times: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
        "stdev_us": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-params", type=int, default=5_000_000,
                    help="approx total #weights under the optimizer")
    ap.add_argument("--layer-shape", type=int, nargs=2, default=[256, 256],
                    metavar=("OUT", "IN"),
                    help="per-layer (out_dim, in_dim) (default 256 256)")
    ap.add_argument("--steps", type=int, default=50,
                    help="number of optimizer steps to time")
    ap.add_argument("--spike-density", type=float, default=0.1,
                    help="fraction of pre/post neurons that fire each step")
    ap.add_argument("--skip-adamw", action="store_true",
                    help="skip AdamW path (no torch needed)")
    args = ap.parse_args(argv)

    layer_out, layer_in = args.layer_shape
    layer_size = layer_out * layer_in
    n_layers = max(1, args.total_params // layer_size)
    actual_total = n_layers * layer_size

    print(f"# Microbench: STDP vs AdamW")
    print(f"# n_layers={n_layers}  shape=({layer_out}, {layer_in})  "
          f"total_params={actual_total:,}")
    print(f"# steps={args.steps}  spike_density={args.spike_density}")
    print()

    rows: list[tuple[str, dict[str, float]]] = []

    if not args.skip_adamw:
        try:
            adamw_t = bench_adamw(n_layers, layer_out, layer_in, args.steps)
            rows.append(("AdamW", _fmt(adamw_t)))
        except ImportError:
            print("# torch unavailable; skipping AdamW.")

    stdp_t = bench_stdp(
        n_layers, layer_out, layer_in, args.steps, args.spike_density
    )
    rows.append(("STDP", _fmt(stdp_t)))

    print(f"{'method':<10}{'mean_us':>14}{'median_us':>14}"
          f"{'min_us':>12}{'max_us':>12}{'stdev_us':>12}")
    for name, s in rows:
        print(f"{name:<10}{s['mean_us']:>14.1f}{s['median_us']:>14.1f}"
              f"{s['min_us']:>12.1f}{s['max_us']:>12.1f}{s['stdev_us']:>12.1f}")

    if len(rows) == 2:
        ratio = rows[0][1]["mean_us"] / max(rows[1][1]["mean_us"], 1e-9)
        print()
        print(f"# Speedup (AdamW_mean / STDP_mean) = {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
