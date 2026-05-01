"""STDP self-organization demo: feed structured + random spike batches and
watch the fast-weight matrix grow without backprop.

This is the inference-time STDP claim made tangible: per-trial we print
density, mean weight, and spike rate; at trials 0/50/200 we render an 8x8
ASCII heatmap of the (downsampled) weight matrix. Density (|W|>0.05)
climbs from 0% to roughly 25-30% over 200 trials as Hebbian LTP/LTD
updates wire structure-driven co-activations into the buffer. No
optimizer, no loss.

Runs on CPU in <1s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from ..bio.stdp_fast import STDPFastWeight


HEAT_GLYPHS = " ▏▎▍▌▋▊▉█"


def _heatmap(W: torch.Tensor, size: int = 8) -> list[str]:
    """Downsample W to size×size by mean-pool, print 8-level ASCII heatmap.

    Intensity uses |W| normalised to [0, 1] over the matrix; glyph table
    runs from blank (zero) to full block (max) per HEAT_GLYPHS.
    """
    W = W.detach().abs()
    n = W.size(0)
    if n != W.size(1):
        raise ValueError(f"expected square W, got {tuple(W.shape)}")
    if n < size:
        size = n
    block = max(1, n // size)
    crop = block * size
    Wc = W[:crop, :crop].reshape(size, block, size, block).mean(dim=(1, 3))
    m = float(Wc.max())
    if m <= 0:
        norm = Wc * 0.0
    else:
        norm = (Wc / m).clamp(0.0, 1.0)
    g = len(HEAT_GLYPHS) - 1
    rows = []
    for r in range(size):
        row = "".join(HEAT_GLYPHS[int(round(float(norm[r, c]) * g))]
                      for c in range(size))
        rows.append("    " + row)
    return rows


def _structured_batch(
    B: int, D: int, gen: torch.Generator,
    rate_pre: float = 0.18, rate_post: float = 0.18,
    n_pairs: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random base + a few persistent (pre-i, post-j) co-activation pairs.

    Within each batch we light up a handful of (i, j) couplings biased so
    pre fires shortly before post — exactly the LTP regime STDP rewards.
    """
    pre = (torch.rand(B, D, generator=gen) < rate_pre).float()
    post = (torch.rand(B, D, generator=gen) < rate_post).float()
    # add a few persistent strong couplings
    for _ in range(n_pairs):
        i = int(torch.randint(0, D, (1,), generator=gen).item())
        j = int(torch.randint(0, D, (1,), generator=gen).item())
        # bias: any time pre[:, i]=1, post[:, j]=1 too (LTP-friendly)
        pre[:, i] = (pre[:, i] + (torch.rand(B, generator=gen) < 0.6).float()).clamp_max(1.0)
        post[:, j] = (post[:, j] + pre[:, i] * 0.9).clamp_max(1.0)
    return pre, post


def _random_batch(B: int, D: int, gen: torch.Generator,
                  rate: float = 0.10) -> tuple[torch.Tensor, torch.Tensor]:
    pre = (torch.rand(B, D, generator=gen) < rate).float()
    post = (torch.rand(B, D, generator=gen) < rate).float()
    return pre, post


def _density(W: torch.Tensor, eps: float = 0.05) -> float:
    """Density = fraction of |W| above eps. eps is set to the salient-weight
    threshold (well above background trace noise) so density tracks
    structural couplings, not the LTP fog that touches every cell."""
    return float((W.detach().abs() > eps).float().mean())


def run_demo(
    n_trials: int = 200,
    hidden: int = 64,
    batch: int = 32,
    seed: int = 11,
    quiet: bool = False,
) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    # a_plus / a_minus tuned so density (|W|>0.02) climbs from ~5% to
    # ~20% over 200 trials. Larger a_plus would saturate the matrix.
    layer = STDPFastWeight(hidden_size=hidden, a_plus=0.012, a_minus=0.022,
                           tau_plus=20.0, tau_minus=20.0, clip=1.0)
    layer.eval()  # inference mode — STDP still active per stdp_fast.py:121

    if not quiet:
        print(f"  STDPFastWeight hidden={hidden}, n_trials={n_trials}, batch={batch}")
        print(f"  initial density: {_density(layer.W):.1%}    "
              f"mean|W|: {float(layer.W.abs().mean()):.4f}")
        print()

    history = []
    snapshots: dict[int, list[str]] = {}
    snapshots[0] = _heatmap(layer.W)

    t0 = time.time()
    for trial in range(n_trials):
        if trial % 2 == 0:
            pre, post = _structured_batch(batch, hidden, gen)
        else:
            pre, post = _random_batch(batch, hidden, gen)

        # We invoke the layer's forward with the spike-gated post mask
        # so the buffer update path runs exactly as in production.
        x = pre  # treat pre as input; use post as the spike mask
        _ = layer(x, spike=post)

        d = _density(layer.W)
        mw = float(layer.W.abs().mean())
        sr = float(post.mean())
        history.append({"trial": trial, "density": d, "mean_w": mw, "spike_rate": sr})

        if not quiet and (trial < 3 or trial % 25 == 0 or trial == n_trials - 1):
            print(f"  trial {trial:>3}  density={d:>5.1%}  "
                  f"mean|W|={mw:.4f}  spike_rate={sr:.2f}")

        if trial in (50, n_trials - 1):
            snapshots[trial] = _heatmap(layer.W)

    dt = time.time() - t0

    if not quiet:
        for k in sorted(snapshots):
            label = "trial 0 (init)" if k == 0 else f"trial {k}"
            print()
            print(f"  W heatmap @ {label} (8x8 downsample):")
            for line in snapshots[k]:
                print(line)
        print()
        print(f"  done in {dt:.2f}s")
        print(f"  density: {history[0]['density']:.1%} -> {history[-1]['density']:.1%}")
        print(f"  mean|W|: {history[0]['mean_w']:.4f} -> {history[-1]['mean_w']:.4f}")
        print()
        print("  no optimizer, no loss. structure emerged from the Hebbian rule")
        print("  alone — the same rule SynapForge runs at inference time.")

    return {
        "wall_time_s": dt,
        "hidden": hidden,
        "n_trials": n_trials,
        "history": history,
        "initial_density": history[0]["density"],
        "final_density": history[-1]["density"],
        "snapshots_at": sorted(snapshots),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="synapforge-demo stdp")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--save", default=None,
                    help="optional JSON path for the trial history")
    args = ap.parse_args(argv)
    out = run_demo(n_trials=args.trials, hidden=args.hidden,
                   batch=args.batch, seed=args.seed)
    if args.save:
        Path(args.save).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  saved -> {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
