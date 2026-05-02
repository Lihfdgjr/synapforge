"""autotune.py -- find max-tok/s config that preserves quality.

Goal
----
Sweep (bs, grad_accum, rfold_chunk, n_data_threads) and pick the
config that gives the best tok/s while keeping val-loss within 1% of
the baseline.

Sweep space (defaults, override via ``AutoTuneConfig``)
-------------------------------------------------------
    bs            in {16, 32, 48, 64, 96, 128}
    grad_accum    in {1, 2, 4}
    rfold_chunk   in {8, 16, 32}
    n_data_threads in {2, 4, 8, 16}

= 6 * 3 * 3 * 4 = 216 configs. We avoid the full cross-product by
default (top-K bs only after a coarse pass) so the sweep is tractable
on a CPU box.

Quality gate
------------
We use ``profile_synthetic_step`` as the "step" so the autotuner runs
on any host (no GPU required). Each config's ``loss`` is the mean of
the last few synthetic steps. The first ``baseline`` config sets the
quality reference; subsequent configs must be within ``loss_tol`` (1%
default) or they're rejected.

In a real Run 8 wiring, the caller would pass a ``step_fn`` that
returns the *real* val loss, and the auto-tuner would gate on that
instead. The stage_profiler timings are the speed signal; this file
just wraps the search.

Output schema
-------------
    AutoTuneResult {
        baseline:   <config>,
        configs:    [ {cfg: ..., tok_per_sec: ..., val_loss: ..., kept: bool}, ... ],
        winner:     <config>,
        winner_tok_per_sec: float,
        speedup_vs_baseline: float,
        projected_run7_tok_per_sec: float  # multiplied by 2750/baseline ratio
    }

Hard rules
----------
* No torch in this file. Everything goes through ``profile_synthetic_step``.
* ``num_steps`` per config must be small enough that 216 configs finish
  in <30 minutes on a CPU box (~8s/config = 30min).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# We deliberately do NOT write
#     ``from synapforge.native.bench.stage_profiler import ...``
# because the parent ``synapforge/__init__.py`` triggers ``import torch``
# at the top, which would defeat the "no torch in production" rule for
# this module. Instead we load the sibling file by absolute path.

def _load_stage_profiler():
    """Load ``stage_profiler.py`` without going through ``synapforge``."""
    import importlib.util as _ilu
    import os as _os
    import sys as _sys

    if "_sf_stage_profiler" in _sys.modules:
        return _sys.modules["_sf_stage_profiler"]

    _here = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_here, "stage_profiler.py")
    _spec = _ilu.spec_from_file_location("_sf_stage_profiler", _path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"could not load stage_profiler at {_path}")
    _mod = _ilu.module_from_spec(_spec)
    # Register before exec so dataclasses can resolve __module__.__dict__.
    _sys.modules["_sf_stage_profiler"] = _mod
    _spec.loader.exec_module(_mod)
    return _mod


_sp = _load_stage_profiler()
_toy_step = _sp._toy_step
profile_synthetic_step = _sp.profile_synthetic_step
StageProfiler = _sp.StageProfiler
_DEFAULT_STAGES = _sp._DEFAULT_STAGES


# Run 7 baseline tok/s -- used to project a real-world speed gain.
RUN7_TOK_PER_SEC: float = 2750.0


@dataclass
class AutoTuneConfig:
    """Sweep search-space + gating thresholds.

    Defaults are tuned for a 30-min CPU sweep that covers the most
    important axes.
    """

    bs_grid: Sequence[int] = (16, 32, 48, 64, 96, 128)
    grad_accum_grid: Sequence[int] = (1, 2, 4)
    rfold_chunk_grid: Sequence[int] = (8, 16, 32)
    n_data_threads_grid: Sequence[int] = (2, 4, 8, 16)
    seq_len: int = 64                # toy seq -- fast enough for sweep
    d: int = 128
    ffn: int = 256
    n_layers: int = 2
    n_warmup: int = 3
    n_steps: int = 12                # per config; ~few seconds each
    loss_tol: float = 0.01           # within 1% of baseline
    baseline_bs: int = 32
    baseline_grad_accum: int = 1
    baseline_rfold_chunk: int = 16
    baseline_n_data_threads: int = 4
    seed: int = 0xC0FFEE
    # If set, only the bs axis is swept first; the top K bs values then
    # cross-product with the other axes.
    coarse_bs_top_k: int = 3
    # If set, skip configs whose VRAM proxy (B*T*d*4) > vram_cap_mb.
    vram_cap_mb: float = 60_000.0


@dataclass
class _ConfigResult:
    bs: int
    grad_accum: int
    rfold_chunk: int
    n_data_threads: int
    tok_per_sec: float
    val_loss: float
    step_ms_mean: float
    bottleneck_stage: str
    bottleneck_share: float
    kept: bool
    rejection_reason: str = ""


@dataclass
class AutoTuneResult:
    """Output of one autotune sweep.

    See module docstring for schema.
    """
    baseline: Dict[str, int]
    baseline_tok_per_sec: float
    baseline_val_loss: float
    configs: List[_ConfigResult] = field(default_factory=list)
    winner: Optional[Dict[str, int]] = None
    winner_tok_per_sec: float = 0.0
    winner_val_loss: float = 0.0
    speedup_vs_baseline: float = 0.0
    projected_run7_tok_per_sec: float = 0.0
    sweep_seconds: float = 0.0
    n_configs_tried: int = 0
    n_configs_kept: int = 0

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "baseline": self.baseline,
            "baseline_tok_per_sec": self.baseline_tok_per_sec,
            "baseline_val_loss": self.baseline_val_loss,
            "configs": [asdict(c) for c in self.configs],
            "winner": self.winner,
            "winner_tok_per_sec": self.winner_tok_per_sec,
            "winner_val_loss": self.winner_val_loss,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "projected_run7_tok_per_sec": self.projected_run7_tok_per_sec,
            "sweep_seconds": self.sweep_seconds,
            "n_configs_tried": self.n_configs_tried,
            "n_configs_kept": self.n_configs_kept,
        }
        return out


def _measure(
    bs: int,
    grad_accum: int,
    rfold_chunk: int,
    n_data_threads: int,
    cfg: AutoTuneConfig,
    rng_seed: int,
) -> _ConfigResult:
    """Run one config: warmup + timed steps; return _ConfigResult."""
    # VRAM-proxy: BS * SEQ * D * 4 bytes (logits roughly). Skip if above cap.
    vram_proxy_mb = (
        bs * cfg.seq_len * cfg.d * 4.0 / 1e6 +
        cfg.n_layers * cfg.d * cfg.ffn * 4.0 / 1e6
    )
    if vram_proxy_mb > cfg.vram_cap_mb:
        return _ConfigResult(
            bs=bs, grad_accum=grad_accum, rfold_chunk=rfold_chunk,
            n_data_threads=n_data_threads,
            tok_per_sec=0.0, val_loss=float("nan"),
            step_ms_mean=0.0,
            bottleneck_stage="(skipped: vram-proxy)",
            bottleneck_share=0.0, kept=False,
            rejection_reason=f"vram_proxy_mb={vram_proxy_mb:.0f} > cap",
        )

    # Honor n_data_threads via OMP/OpenBLAS env (best-effort, no torch).
    # We don't actually parallelize the data loader in this synthetic --
    # but the env knob lets a real trainer pick it up.
    os.environ["OMP_NUM_THREADS"] = str(n_data_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_data_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_data_threads)

    rng = np.random.default_rng(rng_seed)
    tokens = bs * cfg.seq_len * grad_accum
    prof = StageProfiler(stages=_DEFAULT_STAGES, tokens_per_step=tokens)
    losses: List[float] = []
    # Warmup
    for _ in range(cfg.n_warmup):
        warm_prof = StageProfiler(stages=_DEFAULT_STAGES,
                                   tokens_per_step=tokens)
        _toy_step(bs, cfg.seq_len, cfg.d, cfg.ffn, cfg.n_layers,
                  rfold_chunk, rng, warm_prof)
    # Timed
    for _ in range(cfg.n_steps):
        # mimic grad_accum: do `grad_accum` micro-steps, time as one
        for _ in range(grad_accum):
            l = _toy_step(bs, cfg.seq_len, cfg.d, cfg.ffn, cfg.n_layers,
                          rfold_chunk, rng, prof)
            losses.append(float(l))
    summ = prof.summary()
    val_loss = float(np.mean(losses[-min(8, len(losses)):])) if losses else 0.0
    return _ConfigResult(
        bs=bs, grad_accum=grad_accum, rfold_chunk=rfold_chunk,
        n_data_threads=n_data_threads,
        tok_per_sec=summ["tok_per_sec"],
        val_loss=val_loss,
        step_ms_mean=summ["step_ms_mean"],
        bottleneck_stage=summ["bottleneck_stage"],
        bottleneck_share=summ["bottleneck_share"],
        kept=True,
    )


def _enumerate_configs(cfg: AutoTuneConfig) -> List[Tuple[int, int, int, int]]:
    """Use coarse-bs-first strategy to keep the sweep tractable.

    Phase 1 (coarse): sweep bs only at default (grad_accum=1, rfold=16,
    threads=4). Pick top ``coarse_bs_top_k`` bs values by tok/s.

    Phase 2 (fine): full cross-product over (bs in top-K) x grad_accum
    x rfold_chunk x n_data_threads.

    The phase-1 results are duplicated into phase-2, just measured
    again (cheap warmup overlap is acceptable).

    Returns the *full* (bs, grad_accum, rfold_chunk, n_data_threads)
    enumeration to evaluate. Note: the returned list is consumed
    sequentially by ``autotune``.
    """
    # Phase 1: coarse bs
    coarse_pairs = [
        (b, cfg.baseline_grad_accum, cfg.baseline_rfold_chunk,
         cfg.baseline_n_data_threads)
        for b in cfg.bs_grid
    ]
    return coarse_pairs


def _refine(cfg: AutoTuneConfig,
            top_bs: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    """Phase-2 fine grid over the top-K bs values."""
    return list(itertools.product(
        top_bs,
        cfg.grad_accum_grid,
        cfg.rfold_chunk_grid,
        cfg.n_data_threads_grid,
    ))


def autotune(
    cfg: Optional[AutoTuneConfig] = None,
    *,
    on_progress: Optional[Callable[[int, int, _ConfigResult], None]] = None,
) -> AutoTuneResult:
    """Run the sweep; return the winner.

    Algorithm
    ---------
    1. Measure baseline at (baseline_bs, 1, 16, 4) -> baseline_loss /
       baseline_tok.
    2. Phase 1 coarse: sweep bs at default, pick top-K by tok/s.
    3. Phase 2 fine: cross-product (top-K bs) x grad_accum x rfold x threads.
    4. Filter: keep only configs whose val_loss is within
       ``loss_tol`` * baseline_loss.
    5. Winner = max(tok/s) among kept configs.

    Returns the full ``AutoTuneResult``.
    """
    cfg = cfg or AutoTuneConfig()
    t0 = time.perf_counter()

    # ---- Baseline
    base = _measure(
        cfg.baseline_bs, cfg.baseline_grad_accum,
        cfg.baseline_rfold_chunk, cfg.baseline_n_data_threads,
        cfg, rng_seed=cfg.seed,
    )
    res = AutoTuneResult(
        baseline=dict(
            bs=cfg.baseline_bs, grad_accum=cfg.baseline_grad_accum,
            rfold_chunk=cfg.baseline_rfold_chunk,
            n_data_threads=cfg.baseline_n_data_threads,
        ),
        baseline_tok_per_sec=base.tok_per_sec,
        baseline_val_loss=base.val_loss,
    )
    res.configs.append(base)
    base.kept = True  # baseline is always kept as reference
    if on_progress:
        on_progress(1, 1, base)

    # ---- Phase 1: coarse bs sweep
    coarse = _enumerate_configs(cfg)
    coarse_results: List[_ConfigResult] = []
    for i, (bs, ga, rc, th) in enumerate(coarse):
        if (bs, ga, rc, th) == (
            cfg.baseline_bs, cfg.baseline_grad_accum,
            cfg.baseline_rfold_chunk, cfg.baseline_n_data_threads,
        ):
            r = base
        else:
            r = _measure(bs, ga, rc, th, cfg, rng_seed=cfg.seed + i + 1)
        coarse_results.append(r)
        if r is not base:
            res.configs.append(r)
        if on_progress:
            on_progress(len(res.configs), len(coarse) + 1, r)

    # Pick top-K bs by raw tok/s (only consider non-skipped configs)
    valid = [r for r in coarse_results if not math.isnan(r.val_loss)]
    valid.sort(key=lambda r: r.tok_per_sec, reverse=True)
    top_bs_set = []
    for r in valid:
        if r.bs not in top_bs_set:
            top_bs_set.append(r.bs)
        if len(top_bs_set) >= cfg.coarse_bs_top_k:
            break
    if not top_bs_set:
        top_bs_set = [cfg.baseline_bs]

    # ---- Phase 2: fine grid over top bs
    fine = _refine(cfg, top_bs_set)
    seen = {(c.bs, c.grad_accum, c.rfold_chunk, c.n_data_threads)
            for c in res.configs}
    fine_unique = [t for t in fine if t not in seen]
    for i, (bs, ga, rc, th) in enumerate(fine_unique):
        r = _measure(bs, ga, rc, th, cfg, rng_seed=cfg.seed + 1000 + i)
        res.configs.append(r)
        if on_progress:
            on_progress(len(res.configs),
                         len(coarse) + 1 + len(fine_unique), r)

    # ---- Quality gate + winner pick
    if math.isnan(base.val_loss) or base.val_loss <= 0:
        # Synthetic loss can land at zero on tiny grids; skip the gate.
        loss_threshold = float("inf")
    else:
        loss_threshold = base.val_loss * (1.0 + cfg.loss_tol)
    for r in res.configs:
        if not r.kept:
            continue
        if math.isnan(r.val_loss):
            r.kept = False
            r.rejection_reason = r.rejection_reason or "nan_val_loss"
            continue
        if r.val_loss > loss_threshold:
            r.kept = False
            r.rejection_reason = (
                f"val_loss {r.val_loss:.4f} > threshold {loss_threshold:.4f}"
            )
            continue
        if r.tok_per_sec <= 0 or math.isnan(r.tok_per_sec):
            r.kept = False
            r.rejection_reason = r.rejection_reason or "zero_tok_per_sec"

    res.n_configs_tried = len(res.configs)
    kept = [c for c in res.configs if c.kept]
    res.n_configs_kept = len(kept)
    if kept:
        win = max(kept, key=lambda c: c.tok_per_sec)
        res.winner = dict(
            bs=win.bs, grad_accum=win.grad_accum,
            rfold_chunk=win.rfold_chunk, n_data_threads=win.n_data_threads,
        )
        res.winner_tok_per_sec = win.tok_per_sec
        res.winner_val_loss = win.val_loss
        if base.tok_per_sec > 0:
            res.speedup_vs_baseline = win.tok_per_sec / base.tok_per_sec
            res.projected_run7_tok_per_sec = (
                RUN7_TOK_PER_SEC * res.speedup_vs_baseline
            )
    res.sweep_seconds = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
# Pretty + JSON dump.
# ---------------------------------------------------------------------------


def format_autotune_report(res: AutoTuneResult) -> str:
    lines = []
    lines.append(
        f"# autotune sweep | tried={res.n_configs_tried} | "
        f"kept={res.n_configs_kept} | wall={res.sweep_seconds:.1f}s"
    )
    lines.append(
        f"# baseline {res.baseline}: {res.baseline_tok_per_sec:.0f} tok/s "
        f"loss={res.baseline_val_loss:.3f}"
    )
    if res.winner:
        lines.append(
            f"# winner   {res.winner}: {res.winner_tok_per_sec:.0f} tok/s "
            f"loss={res.winner_val_loss:.3f} "
            f"speedup={res.speedup_vs_baseline:.2f}x | "
            f"projected_run8 ~ {res.projected_run7_tok_per_sec:.0f} tok/s"
        )
    else:
        lines.append("# winner: (none -- no config passed quality gate)")
    hdr = (
        f"{'bs':>4} {'ga':>3} {'rfc':>4} {'th':>3} "
        f"{'tok/s':>10} {'loss':>8} {'step_ms':>9} "
        f"{'bottleneck':<22} {'kept':>5}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for c in sorted(res.configs, key=lambda x: -x.tok_per_sec):
        kept = "Y" if c.kept else "N"
        lines.append(
            f"{c.bs:>4} {c.grad_accum:>3} {c.rfold_chunk:>4} "
            f"{c.n_data_threads:>3} {c.tok_per_sec:>10.0f} "
            f"{c.val_loss:>8.3f} {c.step_ms_mean:>9.2f} "
            f"{c.bottleneck_stage:<22} {kept:>5}"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover -- CLI smoke
    cfg = AutoTuneConfig(
        bs_grid=(8, 16, 32, 48),
        grad_accum_grid=(1, 2),
        rfold_chunk_grid=(8, 16),
        n_data_threads_grid=(2, 4),
        n_warmup=2, n_steps=4,
        coarse_bs_top_k=2,
    )
    res = autotune(cfg)
    print(format_autotune_report(res))
    out_path = "synapforge/native/bench/_autotune_smoke.json"
    try:
        with open(out_path, "w") as f:
            json.dump(res.to_dict(), f, indent=2, default=str)
        print(f"\n[saved] {out_path}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
