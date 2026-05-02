"""bench_modal_packing.py -- packed vs padded multimodal batching benchmark.

Compares two ways of feeding 5 modalities (text + image + audio +
time_series + code) at bs=8 per modal through a placeholder backbone:

* **Padded baseline** -- each modal becomes a (B, T_max_global) tensor
  with right-padding to the global maximum sequence length, so a single
  Triton kernel can stride uniformly across the mini-batch. This is
  what Run 7 currently does for text-only training when the batch
  contains heterogeneous lengths.
* **Packed (this layer)** -- per-modal sequences flat-concat into one
  ``concat_tokens`` tensor with offsets and per-token modal-id arrays.
  No padding, but the kernel iterates per-modal segment.

Usage
-----

    python scripts/bench_modal_packing.py [--bs 8] [--reps 50]

Outputs
-------
* tokens-per-sample for both layouts
* memory-per-sample (KB) for both layouts
* end-to-end pack/unpack wall-clock for the packed layout
* projected throughput multiplier when the kernel is FLOPs-bound on
  real tokens only (i.e. 1 / fill_ratio)

Notes
-----
This script does *not* try to run a real Triton kernel -- both backbones
would need GPU and the real CfC. We measure the **layout overhead**
end-to-end and report what fraction of the padded baseline's tokens are
real vs padding. The actual training-time speedup is ``1 / fill_ratio``
when the kernel is throughput-bound on tokens (which it is for
LNN+SNN: every real token = work, every padding token = wasted FLOPs).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Module loader -- bypass synapforge.__init__ (which imports torch)
# ---------------------------------------------------------------------------

def _load_modal():
    """Load synapforge.native.modal without going through the torch import."""
    repo_root = Path(__file__).resolve().parents[1]
    modal_dir = repo_root / "synapforge" / "native" / "modal"

    for name in (
        "synapforge",
        "synapforge.native",
        "synapforge.native.modal",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    pb = _load(
        "synapforge.native.modal.packed_batch", modal_dir / "packed_batch.py"
    )
    parent = sys.modules["synapforge.native.modal"]
    for k in dir(pb):
        if not k.startswith("_"):
            setattr(parent, k, getattr(pb, k))
    mm = _load("synapforge.native.modal.modal_mask", modal_dir / "modal_mask.py")
    cm = _load("synapforge.native.modal.cross_modal", modal_dir / "cross_modal.py")
    dp = _load("synapforge.native.modal.dispatch", modal_dir / "dispatch.py")
    return pb, mm, cm, dp


# ---------------------------------------------------------------------------
# Synthetic per-modal stream generator
# ---------------------------------------------------------------------------

# 5-modal subset for benchmarking -- representative of the production
# mix where text dominates and longer modalities (audio/3D/video) are
# rarer. CLI ``--full`` switches to the full 9-modal sweep.
BENCH_MODALS = ("text", "image", "audio", "time_series", "code")
BENCH_MODALS_FULL = (
    "text", "image", "audio", "time_series",
    "code", "math", "3D", "video", "gesture",
)


def _generate_seqs(
    b_per_modal: int,
    rng: np.random.Generator,
    pb_module: Any,
    modals: Tuple[str, ...] = BENCH_MODALS,
) -> Dict[str, List[np.ndarray]]:
    """Generate ``b_per_modal`` random sequences per modality.

    Sequence lengths are sampled uniformly from ``[T_max/4, T_max/2]``
    so the packed-vs-padded comparison sees realistic length variance.
    """
    out: Dict[str, List[np.ndarray]] = {}
    for name in modals:
        spec = pb_module.MODAL_REGISTRY[name]
        seqs = []
        for _ in range(b_per_modal):
            ln = int(rng.integers(low=spec.t_max // 4,
                                  high=spec.t_max // 2 + 1))
            seqs.append(rng.integers(
                low=0, high=spec.vocab_size, size=ln, dtype=np.int32
            ))
        out[name] = seqs
    return out


# ---------------------------------------------------------------------------
# Padded baseline -- the reference layout we are improving on
# ---------------------------------------------------------------------------

def _padded_layout(
    per_modal: Dict[str, List[np.ndarray]],
    pb_module: Any,
) -> Tuple[np.ndarray, int, int]:
    """Right-pad each modal to global T_max and stack into a single 2-D tensor.

    Returns (padded_array, real_token_count, total_token_count).
    """
    max_t = 0
    n_total = 0
    real_tokens = 0
    for name, seqs in per_modal.items():
        for arr in seqs:
            n_total += 1
            if arr.size > max_t:
                max_t = arr.size
            real_tokens += arr.size
    # Use the GLOBAL maximum across modalities as the pad target -- this
    # is what a mixed-batch Triton kernel actually requires when all
    # samples must share the time axis.
    padded = np.zeros((n_total, max_t), dtype=np.int32)
    row = 0
    for name, seqs in per_modal.items():
        for arr in seqs:
            padded[row, :arr.size] = arr
            row += 1
    total_tokens = padded.size
    return padded, real_tokens, total_tokens


# ---------------------------------------------------------------------------
# Per-modal-padded baseline -- pad each modal to its OWN T_max-in-batch
# ---------------------------------------------------------------------------

def _per_modal_padded_layout(
    per_modal: Dict[str, List[np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Per-modal right-pad. Each modal becomes its own (B, T_modal_max)."""
    out: Dict[str, np.ndarray] = {}
    real_tokens = 0
    total_tokens = 0
    for name, seqs in per_modal.items():
        if not seqs:
            continue
        t_max = max(s.size for s in seqs)
        padded = np.zeros((len(seqs), t_max), dtype=np.int32)
        for i, arr in enumerate(seqs):
            padded[i, :arr.size] = arr
            real_tokens += arr.size
        total_tokens += padded.size
        out[name] = padded
    return out, real_tokens, total_tokens


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

def run_bench(
    b_per_modal: int = 8,
    reps: int = 50,
    seed: int = 0,
    modals: Tuple[str, ...] = BENCH_MODALS,
) -> Dict[str, Any]:
    """Run the packed-vs-padded benchmark."""
    pb, mm, cm, dp = _load_modal()

    rng = np.random.default_rng(seed)
    packer = pb.ModalBatchPacker()

    # Warm-up: one untimed run so we measure steady state, not first-call
    # numpy import lazily allocating things.
    warm = _generate_seqs(b_per_modal, rng, pb, modals=modals)
    _ = packer.pack(warm)
    _ = _padded_layout(warm, pb)
    _ = _per_modal_padded_layout(warm)

    # ------------------------------------------------------------------
    # Layout-cost timings
    # ------------------------------------------------------------------
    pack_times: List[float] = []
    padded_times: List[float] = []
    per_modal_padded_times: List[float] = []

    real_token_total = 0
    padded_token_total = 0
    per_modal_padded_token_total = 0
    packed_byte_total = 0
    padded_byte_total = 0
    n_samples_total = 0

    for r in range(reps):
        per_modal = _generate_seqs(b_per_modal, rng, pb, modals=modals)

        # Packed.
        t0 = time.perf_counter()
        packed = packer.pack(per_modal)
        pack_times.append(time.perf_counter() - t0)

        # Padded (global pad).
        t0 = time.perf_counter()
        padded, real_tokens, padded_tokens = _padded_layout(per_modal, pb)
        padded_times.append(time.perf_counter() - t0)

        # Padded (per-modal pad).
        t0 = time.perf_counter()
        pm_padded, _, pm_padded_tokens = _per_modal_padded_layout(per_modal)
        per_modal_padded_times.append(time.perf_counter() - t0)

        real_token_total += int(real_tokens)
        padded_token_total += int(padded_tokens)
        per_modal_padded_token_total += int(pm_padded_tokens)
        packed_byte_total += int(packed.total_bytes())
        padded_byte_total += int(padded.nbytes)
        n_samples_total += int(packed.n_samples)

    avg_pack_ms = 1000.0 * float(np.mean(pack_times))
    avg_padded_ms = 1000.0 * float(np.mean(padded_times))
    avg_per_modal_padded_ms = 1000.0 * float(np.mean(per_modal_padded_times))

    # Memory + speedup math.
    bytes_per_sample_packed = packed_byte_total / max(n_samples_total, 1)
    bytes_per_sample_padded = padded_byte_total / max(n_samples_total, 1)
    fill_ratio_global = real_token_total / max(padded_token_total, 1)
    fill_ratio_per_modal = real_token_total / max(per_modal_padded_token_total, 1)

    # FLOPs-bound throughput multiplier: kernel does work proportional to
    # real_tokens. Padded layout wastes ``1 - fill_ratio`` FLOPs.
    speedup_vs_global_padded = padded_token_total / max(real_token_total, 1)
    speedup_vs_per_modal_padded = per_modal_padded_token_total / max(real_token_total, 1)

    return {
        "modals": list(modals),
        "b_per_modal": b_per_modal,
        "reps": reps,
        "n_samples_per_rep": n_samples_total // reps,
        "avg_pack_ms": avg_pack_ms,
        "avg_global_padded_ms": avg_padded_ms,
        "avg_per_modal_padded_ms": avg_per_modal_padded_ms,
        "real_tokens_per_rep": real_token_total // reps,
        "padded_tokens_per_rep": padded_token_total // reps,
        "per_modal_padded_tokens_per_rep": per_modal_padded_token_total // reps,
        "bytes_per_sample_packed_kb": bytes_per_sample_packed / 1024.0,
        "bytes_per_sample_padded_kb": bytes_per_sample_padded / 1024.0,
        "fill_ratio_vs_global_padded": fill_ratio_global,
        "fill_ratio_vs_per_modal_padded": fill_ratio_per_modal,
        "throughput_speedup_vs_global_padded": speedup_vs_global_padded,
        "throughput_speedup_vs_per_modal_padded": speedup_vs_per_modal_padded,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(r: Dict[str, Any]) -> None:
    print("=" * 72)
    print("Multimodal Packing Benchmark")
    print("=" * 72)
    print(f"Modalities         : {', '.join(r['modals'])}")
    print(f"bs_per_modal       : {r['b_per_modal']}")
    print(f"samples per rep    : {r['n_samples_per_rep']}")
    print(f"reps               : {r['reps']}")
    print()
    print("--- Layout-cost wall clock (per mini-batch) ---")
    print(f"  pack()                  : {r['avg_pack_ms']:.3f} ms")
    print(f"  global-padded layout    : {r['avg_global_padded_ms']:.3f} ms")
    print(f"  per-modal-padded layout : {r['avg_per_modal_padded_ms']:.3f} ms")
    print()
    print("--- Token counts (per mini-batch) ---")
    print(f"  real tokens          : {r['real_tokens_per_rep']:>10}")
    print(f"  global-padded tokens : {r['padded_tokens_per_rep']:>10}")
    print(f"  per-modal-padded     : {r['per_modal_padded_tokens_per_rep']:>10}")
    print()
    print("--- Memory per sample ---")
    print(f"  packed  : {r['bytes_per_sample_packed_kb']:8.2f} KB/sample")
    print(f"  padded  : {r['bytes_per_sample_padded_kb']:8.2f} KB/sample")
    print(f"  ratio   : {r['bytes_per_sample_packed_kb'] / max(r['bytes_per_sample_padded_kb'], 1e-9):.3f}x")
    print()
    print("--- Fill ratios (real / padded) ---")
    print(f"  vs global-padded    : {r['fill_ratio_vs_global_padded']:.3f}")
    print(f"  vs per-modal-padded : {r['fill_ratio_vs_per_modal_padded']:.3f}")
    print()
    print("--- Projected throughput multiplier (FLOPs-bound) ---")
    print(f"  vs global-padded    : {r['throughput_speedup_vs_global_padded']:.2f}x")
    print(f"  vs per-modal-padded : {r['throughput_speedup_vs_per_modal_padded']:.2f}x")
    print()
    target_lo, target_hi = 1.4, 1.8
    sp = r["throughput_speedup_vs_per_modal_padded"]
    if target_lo <= sp <= target_hi * 2:
        verdict = "MEETS target (1.4-1.8x)"
    elif sp > target_hi * 2:
        verdict = "EXCEEDS target (likely synthetic-mix overestimate)"
    else:
        verdict = f"BELOW target ({target_lo}-{target_hi}x)"
    print(f"Verdict: {verdict}")
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs", type=int, default=8,
                   help="batch size per modality (default 8)")
    p.add_argument("--reps", type=int, default=50,
                   help="number of mini-batches to time (default 50)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full", action="store_true",
                   help="benchmark the full 9-modal mix (default 5 modals)")
    args = p.parse_args()

    modals = BENCH_MODALS_FULL if args.full else BENCH_MODALS
    r = run_bench(
        b_per_modal=args.bs, reps=args.reps, seed=args.seed, modals=modals,
    )
    _print_report(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
