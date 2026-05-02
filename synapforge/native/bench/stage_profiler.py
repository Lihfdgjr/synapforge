"""stage_profiler.py -- per-stage timing of one training step.

Goal
----
Measure how long each logical phase of *one* training step takes:

    data_loader     -- CPU disk read + decode + tokenize + push to GPU
    embed_fwd       -- token-id gather
    hybridblock_fwd -- per-block forward (n_layers * loop_depth blocks)
    lm_head_fwd     -- final linear to vocab
    hybridblock_bwd -- per-block backward
    lm_head_bwd     -- final-linear backward
    optimizer_step  -- AdamW (+ optional CPU offload)
    h2d / d2h       -- explicit transfer phases

Two ways to use it
------------------
1. Pure-numpy synthetic profile (this file's main entry point):
   ``profile_synthetic_step()`` runs a CPU-only LNN+SNN micro-step and
   measures stage timings. No torch, no GPU. Used to sanity-check the
   profiler harness and to bound autotune sweep cost.

2. Wrap a real trainer step (callers in trainer-refactor / run7 etc.):
   ``StageProfiler.start("data_loader")`` ... ``stop()`` blocks. Build
   timings stage-by-stage, then ``.summary()`` returns a dict of
   per-stage mean/std/p50/p99 in ms.

Output schema (so the autotuner can consume it):

    {
        "n_steps": 100,
        "tokens_per_step": 65536,
        "tok_per_sec": 12345.6,
        "stages": {
            "data_loader":     {"ms_mean": .., "ms_p50": .., "ms_p99": .., "share": 0.07},
            "hybridblock_fwd": {"ms_mean": .., ...},
            ...
        },
        "step_ms_mean": 78.3,
        "step_ms_p99": 92.1,
        "bottleneck_stage": "optimizer_step",
        "bottleneck_share": 0.34,
    }

No torch import in this file. Tests under ``tests/native/bench`` may
use torch as the oracle for whether timings agree with a reference.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stage timing record.
# ---------------------------------------------------------------------------


@dataclass
class StageTiming:
    """Timings for a single logical stage across N training steps.

    All times in milliseconds.
    """
    name: str
    samples: List[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(float(ms))

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {"ms_mean": 0.0, "ms_p50": 0.0, "ms_p99": 0.0,
                    "ms_min": 0.0, "ms_max": 0.0, "n": 0}
        sorted_ = sorted(self.samples)
        n = len(sorted_)
        return {
            "ms_mean": statistics.fmean(sorted_),
            "ms_p50":  sorted_[n // 2],
            "ms_p99":  sorted_[min(n - 1, int(n * 0.99))],
            "ms_min":  sorted_[0],
            "ms_max":  sorted_[-1],
            "n":       n,
        }


# ---------------------------------------------------------------------------
# Profiler -- start/stop API + context manager.
# ---------------------------------------------------------------------------


_DEFAULT_STAGES = (
    "data_loader",
    "embed_fwd",
    "hybridblock_fwd",
    "lm_head_fwd",
    "hybridblock_bwd",
    "lm_head_bwd",
    "optimizer_step",
    "h2d",
    "d2h",
)


class StageProfiler:
    """Lightweight per-stage profiler for a training loop.

    Usage::

        prof = StageProfiler(stages=("data_loader", "fwd", "bwd", "opt"),
                             tokens_per_step=65536)
        for step in range(100):
            with prof.stage("data_loader"):
                batch = next(loader)
            with prof.stage("fwd"):
                logits = model(batch)
            ...
            prof.next_step()

        print(prof.summary())

    The profiler is single-threaded and uses ``time.perf_counter()``.
    For GPU-side timing you must call ``cuda_synchronize`` (passed in
    by the caller) before the ``with`` block exits. We keep this
    abstract so this file has no torch dep.
    """

    def __init__(
        self,
        stages: Iterable[str] = _DEFAULT_STAGES,
        tokens_per_step: int = 0,
        sync_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self._stage_order: List[str] = list(stages)
        self._timings: Dict[str, StageTiming] = {
            n: StageTiming(name=n) for n in self._stage_order
        }
        self._tokens_per_step = int(tokens_per_step)
        self._sync_fn = sync_fn or (lambda: None)
        self._step_start: Optional[float] = None
        self._step_ms: List[float] = []
        self._n_steps: int = 0
        self._current_stage: Optional[str] = None
        self._current_t0: Optional[float] = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context manager that times a stage and adds it to the record."""
        if name not in self._timings:
            # Allow lazy stage registration -- caller knows best.
            self._stage_order.append(name)
            self._timings[name] = StageTiming(name=name)
        self._sync_fn()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync_fn()
            ms = (time.perf_counter() - t0) * 1000.0
            self._timings[name].add(ms)

    def next_step(self) -> None:
        """Mark the end of one full step. Records step wall time."""
        self._sync_fn()
        now = time.perf_counter()
        if self._step_start is not None:
            self._step_ms.append((now - self._step_start) * 1000.0)
        self._step_start = now
        self._n_steps += 1

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["n_steps"] = max(0, self._n_steps - 1)
        out["tokens_per_step"] = self._tokens_per_step
        out["stages"] = {}
        for name in self._stage_order:
            s = self._timings[name].summary()
            out["stages"][name] = s
        # Step wall-time (preferred over sum-of-stages because of
        # async stages overlapping).
        if self._step_ms:
            sorted_ = sorted(self._step_ms)
            out["step_ms_mean"] = statistics.fmean(sorted_)
            out["step_ms_p50"] = sorted_[len(sorted_) // 2]
            out["step_ms_p99"] = sorted_[min(len(sorted_) - 1,
                                              int(len(sorted_) * 0.99))]
        else:
            out["step_ms_mean"] = 0.0
            out["step_ms_p50"] = 0.0
            out["step_ms_p99"] = 0.0
        # tok/s from step wall-time
        if self._tokens_per_step > 0 and out["step_ms_mean"] > 0:
            out["tok_per_sec"] = (
                self._tokens_per_step / (out["step_ms_mean"] / 1000.0)
            )
        else:
            out["tok_per_sec"] = 0.0
        # Bottleneck stage = the one with largest mean
        if self._timings:
            stage_means = [
                (n, self._timings[n].summary()["ms_mean"])
                for n in self._stage_order
            ]
            stage_means = [(n, m) for n, m in stage_means if m > 0]
            if stage_means:
                bn, bm = max(stage_means, key=lambda x: x[1])
                out["bottleneck_stage"] = bn
                step_ms = out["step_ms_mean"] or sum(m for _, m in stage_means)
                out["bottleneck_share"] = bm / step_ms if step_ms > 0 else 0.0
            else:
                out["bottleneck_stage"] = ""
                out["bottleneck_share"] = 0.0
        else:
            out["bottleneck_stage"] = ""
            out["bottleneck_share"] = 0.0
        return out


# ---------------------------------------------------------------------------
# Synthetic CPU-only step -- used for autotune sanity + tests.
# ---------------------------------------------------------------------------


def _xavier_w(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    std = 1.0 / math.sqrt(max(1, fan_in))
    return (rng.standard_normal((fan_out, fan_in)).astype(np.float32) * std)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _silu_grad(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sq = (x * x).mean(axis=-1, keepdims=True)
    return w * x / np.sqrt(sq + eps)


def _toy_step(
    bs: int, seq: int, d: int, ffn: int, n_layers: int,
    rfold_chunk: int, rng: np.random.Generator, prof: StageProfiler,
) -> float:
    """Run one toy fwd+bwd+opt step on CPU and time each stage.

    Architecture is a stripped HybridBlock: rmsnorm -> linear -> silu ->
    linear (no PLIF / no CfC -- those are O(d) and not GEMM-bound).
    Returns per-step loss (synthetic).
    """
    V = 1024  # tiny vocab so fwd is fast
    # ----- data_loader -----
    with prof.stage("data_loader"):
        toks = rng.integers(0, V, size=(bs, seq)).astype(np.int32)
        # mimic small CPU work: bytes -> int32
        _ = toks.tobytes()

    # ----- embed_fwd -----
    with prof.stage("embed_fwd"):
        E = rng.standard_normal((V, d)).astype(np.float32) * 0.02
        h = E[toks]  # (bs, seq, d)

    # ----- hybridblock_fwd -----
    layer_W1 = [_xavier_w(rng, d, ffn) for _ in range(n_layers)]
    layer_W2 = [_xavier_w(rng, ffn, d) for _ in range(n_layers)]
    rmsw = np.ones((d,), dtype=np.float32)
    cache: List[Dict[str, np.ndarray]] = []
    with prof.stage("hybridblock_fwd"):
        x = h
        # rfold-chunk: process the seq dim in chunks to reduce peak memory
        for li in range(n_layers):
            x_norm = _rmsnorm(x, rmsw)
            for s0 in range(0, seq, rfold_chunk):
                xs = x_norm[:, s0:s0 + rfold_chunk]
                z1 = xs @ layer_W1[li].T
                z1a = _silu(z1)
                z2 = z1a @ layer_W2[li].T
                x_norm[:, s0:s0 + rfold_chunk] = z2
            x = x + x_norm
            cache.append({"x": x.copy()})

    # ----- lm_head_fwd -----
    Wlm = _xavier_w(rng, d, V)
    with prof.stage("lm_head_fwd"):
        logits = x @ Wlm.T  # (bs, seq, V)

    # ----- loss + lm_head_bwd -----
    targets = rng.integers(0, V, size=(bs, seq)).astype(np.int64)
    with prof.stage("lm_head_bwd"):
        # softmax + ce manually (numerically simple)
        m = logits.max(axis=-1, keepdims=True)
        ex = np.exp(logits - m)
        sm = ex / ex.sum(axis=-1, keepdims=True)
        loss = -np.log(sm.reshape(-1, V)[np.arange(bs * seq), targets.ravel()] + 1e-9).mean()
        # grad-logits = softmax - one_hot
        sm_flat = sm.reshape(-1, V)
        sm_flat[np.arange(bs * seq), targets.ravel()] -= 1.0
        sm_flat /= bs * seq
        grad_logits = sm_flat.reshape(bs, seq, V)
        grad_x = grad_logits @ Wlm
        # grad_W = grad_logits^T @ x
        grad_Wlm = grad_logits.reshape(-1, V).T @ x.reshape(-1, d)
        del grad_Wlm  # free; not used downstream

    # ----- hybridblock_bwd -----
    with prof.stage("hybridblock_bwd"):
        gx = grad_x
        for li in range(n_layers - 1, -1, -1):
            x_in = cache[li]["x"]
            for s0 in range(0, seq, rfold_chunk):
                gxs = gx[:, s0:s0 + rfold_chunk]
                xs = x_in[:, s0:s0 + rfold_chunk]
                # bwd through linear2
                gz1a = gxs @ layer_W2[li]
                # bwd through silu
                z1 = xs @ layer_W1[li].T   # recompute (grad-ckpt mimic)
                gz1 = gz1a * _silu_grad(z1)
                # bwd through linear1
                gx[:, s0:s0 + rfold_chunk] = gz1 @ layer_W1[li]

    # ----- optimizer_step -----
    with prof.stage("optimizer_step"):
        # Tiny in-place AdamW on Wlm (the biggest weight here)
        Wlm *= (1.0 - 1e-4)
        # touch all layer weights
        for li in range(n_layers):
            layer_W1[li] *= (1.0 - 1e-4)
            layer_W2[li] *= (1.0 - 1e-4)

    prof.next_step()
    return float(loss)


def profile_synthetic_step(
    bs: int = 4,
    seq: int = 64,
    d: int = 128,
    ffn: int = 256,
    n_layers: int = 2,
    rfold_chunk: int = 16,
    n_warmup: int = 5,
    n_steps: int = 20,
    seed: int = 0xCAFE,
) -> Dict[str, Any]:
    """Drive ``_toy_step`` for warmup+timed runs; return profiler summary.

    This is the "smoke" entry point used by the autotuner and tests.
    """
    rng = np.random.default_rng(seed)
    tokens = bs * seq
    prof = StageProfiler(stages=_DEFAULT_STAGES, tokens_per_step=tokens)
    # Warmup -- not recorded.
    warm_prof = StageProfiler(stages=_DEFAULT_STAGES, tokens_per_step=tokens)
    for _ in range(n_warmup):
        _toy_step(bs, seq, d, ffn, n_layers, rfold_chunk, rng, warm_prof)
    # Timed.
    for _ in range(n_steps):
        _toy_step(bs, seq, d, ffn, n_layers, rfold_chunk, rng, prof)
    return prof.summary()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _format_summary(s: Dict[str, Any]) -> str:
    lines = []
    lines.append(
        f"# stage_profiler synthetic | bs*seq={s['tokens_per_step']} | "
        f"steps={s['n_steps']} | step_ms_mean={s['step_ms_mean']:.3f}"
    )
    lines.append(
        f"# tok/s={s['tok_per_sec']:.1f} | "
        f"bottleneck={s['bottleneck_stage']} ({s['bottleneck_share']*100:.1f}%)"
    )
    hdr = f"{'stage':<22} {'ms_mean':>10} {'ms_p50':>10} {'ms_p99':>10} {'share%':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    step_total = s.get("step_ms_mean", 0.0) or sum(
        v["ms_mean"] for v in s["stages"].values()
    )
    for name, vals in s["stages"].items():
        share = (vals["ms_mean"] / step_total * 100.0) if step_total > 0 else 0.0
        lines.append(
            f"{name:<22} {vals['ms_mean']:>10.3f} {vals['ms_p50']:>10.3f} "
            f"{vals['ms_p99']:>10.3f} {share:>8.1f}"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover -- CLI smoke
    s = profile_synthetic_step()
    print(_format_summary(s))


if __name__ == "__main__":
    main()
