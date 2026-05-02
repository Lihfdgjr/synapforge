"""synapforge.native.bench -- saturation profiler + auto-tuner.

Three primary tools:

    roofline       -- closed-form physical roofline for a given (model, hw)
                      pair (FLOPs, bytes, AI, ms/step at 100% TC, ms/step at
                      HBM-bound, ms/step at PCIe-bound).
    stage_profiler -- hooks a real training step and emits per-stage timings:
                      data-loader, fwd per HybridBlock, bwd per HybridBlock,
                      optimizer, H2D, D2H. Outputs a Pandas-style table.
    autotune       -- sweeps (bs, grad_accum, n_streams, rfold_chunk) on a
                      tiny synthetic and picks the max-tok/s config whose
                      val-loss matches the baseline within 1%.

Hard rules
----------
* No ``import torch`` anywhere in this package -- everything is numpy
  + (optional) cupy. Tests may use torch as the oracle.
* All public APIs return JSON-serialisable dicts so they can be diffed
  step-over-step without pulling in a heavy dataframe library.

Symbols are lazily re-exported via ``__getattr__`` so importing the
parent ``synapforge`` package -- which triggers a torch import -- is
not required when callers only need the bench tools.
"""

from __future__ import annotations

# Public symbols (resolved lazily through __getattr__).
__all__ = [
    "HardwareSpec",
    "ModelSpec",
    "RooflineResult",
    "A800_80GB",
    "H100_80GB",
    "A100_80GB",
    "compute_roofline",
    "format_roofline_table",
    "StageTiming",
    "StageProfiler",
    "profile_synthetic_step",
    "AutoTuneConfig",
    "AutoTuneResult",
    "autotune",
    "format_autotune_report",
    "RUN7_TOK_PER_SEC",
]


def _load(name: str):
    if name in (
        "HardwareSpec", "ModelSpec", "RooflineResult",
        "A800_80GB", "H100_80GB", "A100_80GB",
        "compute_roofline", "format_roofline_table",
    ):
        from synapforge.native.bench import roofline
        return getattr(roofline, name)
    if name in ("StageTiming", "StageProfiler", "profile_synthetic_step"):
        from synapforge.native.bench import stage_profiler
        return getattr(stage_profiler, name)
    if name in (
        "AutoTuneConfig", "AutoTuneResult", "autotune",
        "format_autotune_report", "RUN7_TOK_PER_SEC",
    ):
        from synapforge.native.bench import autotune as _at
        return getattr(_at, name)
    raise AttributeError(f"module 'synapforge.native.bench' has no attribute {name!r}")


def __getattr__(name: str):
    return _load(name)
