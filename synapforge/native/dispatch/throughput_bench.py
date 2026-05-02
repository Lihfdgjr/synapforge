"""throughput_bench.py -- sequential vs pipelined throughput.

What this measures
------------------
We compare two modes that run identical work, in identical math, on
identical inputs:

* **sequential**: HeteroPipeline(enable_pipeline=False) -- A then B
  then C, single thread, baseline.
* **pipelined**: HeteroPipeline(enable_pipeline=True) -- A and C
  overlap with B on background threads.

The wall-clock per step in sequential is ``t_A + t_B + t_C``. In
pipelined it is approximately ``max(t_A, t_B, t_C)`` after the first
step warms up the pipeline (we report both warm and cold timings).

Synthetic load
--------------
We use a fake forward (matmul-tanh-matmul, repeated ``b_iters``
times) plus a fake AdamW over ``num_param_tensors`` numpy arrays
totalling ``param_count`` params. ``b_iters=0`` triggers an
auto-balance pass that picks ``b_iters`` so ``t_B ~= t_C`` (which
gives the speedup ceiling its theoretical maximum of 2.0x).

When cupy IS available, Stage B's matmul runs on the default cuda
stream and CPU-side AdamW runs on the optim thread -- which is the
real-world configuration.

Target speedup
--------------
**>= 1.5x** when ``t_B ~= t_C``. Note that if ``t_B >> t_C`` (typical
of bf16-on-A800 with very small models) the speedup floor is ~1.0x
because the optim time was already a footnote. Conversely, if
``t_C >> t_B`` (CPU-only host with a huge model) the speedup floor
is also ~1.0x because the GPU is idle anyway. The sweet spot is
``t_B ~= t_C`` which is the configuration ZeRO-Offload Stage 0
deliberately produces.

Usage
-----
    python -m synapforge.native.dispatch.throughput_bench \
        --num-steps 50 --param-count 100000000

Outputs a markdown table and a JSON line to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from synapforge.native.dispatch.cpu_pool import (
    CpuWorkerPool,
    parallel_adamw_step,
)
from synapforge.native.dispatch.pipeline import HeteroPipeline


# ---------------------------------------------------------------------------
# Synthetic workload
# ---------------------------------------------------------------------------

@dataclass
class _SyntheticWorkload:
    """Generates batches and runs a fake forward+backward and AdamW.

    The compute volumes are chosen so a 100M-param synthetic AdamW
    takes ~30-60 ms on a modern x86 (close to a real LNN+SNN AdamW),
    and the synthetic matmul takes ~30-80 ms on the same machine
    (close to a forward+backward chunk that is *not* GPU-accelerated).
    """

    batch: int
    seq: int
    d_model: int
    num_param_tensors: int
    param_count: int
    b_iters: int = 1  # repeat the matmul-tanh-matmul k times to balance with C
    # If >0, Stage B sleeps for this many seconds instead of CPU matmul.
    # Simulates Stage B running on a real GPU (releases GIL, doesn't
    # contend for CPU cores). On a CPU-only host this is the only way
    # to get a clean speedup measurement; on a real A800 host the GPU
    # IS asynchronous w.r.t. the CPU optim thread.
    b_simulated_gpu_s: float = 0.0

    # State (allocated in __post_init__)
    params: List[np.ndarray] = None  # type: ignore[assignment]
    moments_m: List[np.ndarray] = None  # type: ignore[assignment]
    moments_v: List[np.ndarray] = None  # type: ignore[assignment]
    matmul_W: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        rng = np.random.default_rng(0)
        per = max(1, self.param_count // self.num_param_tensors)
        self.params = [
            rng.standard_normal(per, dtype=np.float32) * 0.02
            for _ in range(self.num_param_tensors)
        ]
        self.moments_m = [np.zeros_like(p) for p in self.params]
        self.moments_v = [np.zeros_like(p) for p in self.params]
        self.matmul_W = rng.standard_normal(
            (self.d_model, self.d_model), dtype=np.float32) * 0.01

    def batch_fn(self, step: int) -> Optional[Tuple[Any, Any, Dict[str, Any]]]:
        x = np.random.default_rng(step).standard_normal(
            (self.batch, self.seq, self.d_model), dtype=np.float32)
        return x, None, {}

    def fb_fn(self, x: np.ndarray, _y: Any, _extra: Dict[str, Any]
              ) -> Tuple[List[np.ndarray], float, Dict[str, Any]]:
        # Stage-B: synthetic forward+backward.
        if self.b_simulated_gpu_s > 0.0:
            # Simulate a GPU-bound stage: sleep releases the GIL so
            # Stage C can run uncontested in parallel.
            time.sleep(self.b_simulated_gpu_s)
            loss = 0.0
        else:
            h = x
            for _ in range(self.b_iters):
                h = h @ self.matmul_W
                h = np.tanh(h)
                h = h @ self.matmul_W.T
            loss = float(h.var())
        grads = [
            np.full_like(p, 1e-3 * (1.0 + 0.1 * (i % 7)))
            for i, p in enumerate(self.params)
        ]
        return grads, loss, {}

    def make_optim_step_fn(self, *, lr: float = 1e-3,
                           pool: Optional[CpuWorkerPool] = None):
        """Return a Stage-C optim_step_fn closure that mutates self.params."""
        if pool is None:
            def _serial_step(step_idx: int, grads: List[np.ndarray],
                             _extra: Dict[str, Any]) -> None:
                bc1 = 1.0 - 0.9 ** (step_idx + 1)
                bc2 = 1.0 - 0.95 ** (step_idx + 1)
                for p, g, m, v in zip(
                        self.params, grads,
                        self.moments_m, self.moments_v):
                    if g is None:
                        continue
                    np.multiply(m, 0.9, out=m)
                    m += 0.1 * g
                    np.multiply(v, 0.95, out=v)
                    v += 0.05 * (g * g)
                    m_hat = m / bc1
                    v_hat = v / bc2
                    update = m_hat / (np.sqrt(v_hat) + 1e-8) + 0.01 * p
                    p -= lr * update
            return _serial_step

        def _pool_step(step_idx: int, grads: List[np.ndarray],
                       _extra: Dict[str, Any]) -> None:
            parallel_adamw_step(
                pool,
                self.params, grads,
                self.moments_m, self.moments_v,
                step=step_idx + 1,
                lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.01,
            )
        return _pool_step


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    mode: str
    num_steps: int
    wallclock_s: float
    steps_per_second: float
    stage_a_total_s: float
    stage_b_total_s: float
    stage_c_total_s: float
    stage_a_wait_s: float
    stage_b_wait_s: float
    stage_c_wait_s: float
    overlap_ratio: float
    b_iters: int = 0


def run_bench(
    *,
    num_steps: int = 30,
    batch: int = 4,
    seq: int = 64,
    d_model: int = 512,
    num_param_tensors: int = 24,
    param_count: int = 100_000_000,
    cpu_pool_workers: int = 4,
    use_pool: bool = True,
    b_iters: int = 0,             # 0 = auto-balance vs Stage C
    b_simulated_gpu_s: float = 0.0,
    seed: int = 0,
) -> Tuple[BenchResult, BenchResult]:
    """Run sequential then pipelined bench.

    Returns ``(seq_result, pipe_result)``.

    ``b_iters`` controls Stage B's per-step matmul iteration count.
    ``0`` means auto-balance: we time one B step and one C step
    on a throwaway workload, then pick ``b_iters`` so that
    ``t_B ~= t_C``. This makes the speedup ceiling ~2.0x and
    meaningful (otherwise a light Stage B trivially dwarfs the
    pipelining benefit).
    """
    np.random.seed(seed)

    if b_simulated_gpu_s <= 0.0 and b_iters <= 0:
        b_iters = _auto_balance_b_iters(
            batch=batch, seq=seq, d_model=d_model,
            num_param_tensors=num_param_tensors,
            param_count=param_count,
            cpu_pool_workers=cpu_pool_workers,
            use_pool=use_pool,
        )

    wl_seq = _SyntheticWorkload(
        batch=batch, seq=seq, d_model=d_model,
        num_param_tensors=num_param_tensors, param_count=param_count,
        b_iters=b_iters,
        b_simulated_gpu_s=b_simulated_gpu_s)
    wl_pipe = _SyntheticWorkload(
        batch=batch, seq=seq, d_model=d_model,
        num_param_tensors=num_param_tensors, param_count=param_count,
        b_iters=b_iters,
        b_simulated_gpu_s=b_simulated_gpu_s)

    pool = CpuWorkerPool(num_workers=cpu_pool_workers, name="bench") if use_pool else None
    try:
        seq_result = _run_one(wl_seq, num_steps=num_steps, pipelined=False, pool=pool, mode_label="sequential")
        pipe_result = _run_one(wl_pipe, num_steps=num_steps, pipelined=True, pool=pool, mode_label="pipelined")
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    seq_result.b_iters = b_iters
    pipe_result.b_iters = b_iters
    return seq_result, pipe_result


def _auto_balance_b_iters(
    *, batch: int, seq: int, d_model: int,
    num_param_tensors: int, param_count: int,
    cpu_pool_workers: int, use_pool: bool,
) -> int:
    """Time one iter of B and one C step; pick b_iters so t_B ~= t_C.

    The calibration runs Stage B and Stage C with a few warmups so the
    BLAS/OPENBLAS thread pool is initialised and the matmul cache is
    warm when we measure.
    """
    wl = _SyntheticWorkload(
        batch=batch, seq=seq, d_model=d_model,
        num_param_tensors=num_param_tensors, param_count=param_count,
        b_iters=1)
    pool = CpuWorkerPool(num_workers=cpu_pool_workers, name="cal") if use_pool else None
    try:
        # Warm-up
        x = wl.batch_fn(0)[0]
        for _ in range(2):
            wl.fb_fn(x, None, {})
        # Time B with b_iters=1
        N = 5
        t0 = time.time()
        for _ in range(N):
            wl.fb_fn(x, None, {})
        t_b1 = (time.time() - t0) / N
        # Time C
        opt = wl.make_optim_step_fn(pool=pool)
        grads = wl.fb_fn(x, None, {})[0]
        # Warm-up C
        opt(0, grads, {})
        t0 = time.time()
        for s in range(1, N + 1):
            opt(s, grads, {})
        t_c = (time.time() - t0) / N
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    if t_b1 <= 0:
        return 1
    iters = max(1, int(round(t_c / t_b1)))
    return iters


def _run_one(
    workload: _SyntheticWorkload,
    *,
    num_steps: int,
    pipelined: bool,
    pool: Optional[CpuWorkerPool],
    mode_label: str,
) -> BenchResult:
    optim_step_fn = workload.make_optim_step_fn(pool=pool)
    hp = HeteroPipeline(
        batch_fn=workload.batch_fn,
        forward_backward_fn=workload.fb_fn,
        optim_step_fn=optim_step_fn,
        queue_ab_size=2,
        queue_bc_size=1,
        enable_pipeline=pipelined,
    )
    metrics = hp.run(max_steps=num_steps)
    return BenchResult(
        mode=mode_label,
        num_steps=metrics.num_steps,
        wallclock_s=metrics.wallclock_s,
        steps_per_second=metrics.steps_per_second,
        stage_a_total_s=metrics.stage_a_total_s,
        stage_b_total_s=metrics.stage_b_total_s,
        stage_c_total_s=metrics.stage_c_total_s,
        stage_a_wait_s=metrics.stage_a_wait_s,
        stage_b_wait_s=metrics.stage_b_wait_s,
        stage_c_wait_s=metrics.stage_c_wait_s,
        overlap_ratio=metrics.b_c_overlap_ratio,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(seq: BenchResult, pipe: BenchResult) -> str:
    speedup = seq.wallclock_s / pipe.wallclock_s if pipe.wallclock_s > 0 else float("nan")
    cmp = "PASS" if speedup >= 1.5 else "below-target"
    lines = [
        "## HeteroPipeline throughput bench",
        "",
        f"| metric                | sequential | pipelined |",
        f"|-----------------------|-----------:|----------:|",
        f"| steps                 | {seq.num_steps:>10} | {pipe.num_steps:>9} |",
        f"| wallclock_s           | {seq.wallclock_s:>10.3f} | {pipe.wallclock_s:>9.3f} |",
        f"| steps/s               | {seq.steps_per_second:>10.3f} | {pipe.steps_per_second:>9.3f} |",
        f"| stage_A_total_s       | {seq.stage_a_total_s:>10.3f} | {pipe.stage_a_total_s:>9.3f} |",
        f"| stage_B_total_s       | {seq.stage_b_total_s:>10.3f} | {pipe.stage_b_total_s:>9.3f} |",
        f"| stage_C_total_s       | {seq.stage_c_total_s:>10.3f} | {pipe.stage_c_total_s:>9.3f} |",
        f"| overlap_ratio (B/C)   | {seq.overlap_ratio:>10.3f} | {pipe.overlap_ratio:>9.3f} |",
        f"| b_iters (auto-bal)    | {seq.b_iters:>10} | {pipe.b_iters:>9} |",
        "",
        f"speedup (seq/pipe wallclock): **{speedup:.2f}x** -- target >=1.5x => **{cmp}**",
        "",
        "Notes:",
        f"- t_B / t_C ratio (sequential): "
        f"{(seq.stage_b_total_s / max(seq.stage_c_total_s, 1e-9)):.2f}",
        "- speedup ceiling = (t_B + t_C) / max(t_B, t_C); when t_B == t_C, ceiling is 2.0x.",
        "- t_A is excluded from the ceiling because Stage A overlaps with both B and C.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-param-tensors", type=int, default=24)
    parser.add_argument("--param-count", type=int, default=10_000_000,
                        help="Total numpy parameter count across all tensors")
    parser.add_argument("--cpu-pool-workers", type=int, default=4)
    parser.add_argument("--no-pool", action="store_true",
                        help="Disable CpuWorkerPool (single-thread Stage C)")
    parser.add_argument("--b-iters", type=int, default=0,
                        help="Stage B matmul iterations per step (0=auto-balance)")
    parser.add_argument("--b-simulated-gpu-s", type=float, default=0.0,
                        help=("Simulate Stage B as a GPU-bound stage by "
                              "sleeping for this many seconds per step. "
                              "Releases the GIL so Stage C runs uncontested. "
                              "Use this to model A800 production behaviour "
                              "on a CPU-only dev host."))
    parser.add_argument("--json-only", action="store_true",
                        help="Emit only the JSON line; suppress markdown table")
    args = parser.parse_args()

    seq, pipe = run_bench(
        num_steps=args.num_steps,
        batch=args.batch, seq=args.seq, d_model=args.d_model,
        num_param_tensors=args.num_param_tensors,
        param_count=args.param_count,
        cpu_pool_workers=args.cpu_pool_workers,
        use_pool=not args.no_pool,
        b_iters=args.b_iters,
        b_simulated_gpu_s=args.b_simulated_gpu_s,
    )
    speedup = (seq.wallclock_s / pipe.wallclock_s) if pipe.wallclock_s > 0 else float("nan")
    payload = {
        "sequential": asdict(seq),
        "pipelined": asdict(pipe),
        "speedup": speedup,
        "config": {
            "num_steps": args.num_steps,
            "batch": args.batch,
            "seq": args.seq,
            "d_model": args.d_model,
            "num_param_tensors": args.num_param_tensors,
            "param_count": args.param_count,
            "cpu_pool_workers": args.cpu_pool_workers,
            "b_iters": seq.b_iters,
        },
    }
    if not args.json_only:
        print(format_report(seq, pipe))
        print()
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
