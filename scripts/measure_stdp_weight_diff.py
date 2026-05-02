"""T8.1 — Measure ‖ΔW_STDP‖ during 1K-token chat (paper headline verification).

Verifies the headline claim from `INVESTOR.md` §3 / `docs/MONOTONIC_QUALITY.md`:
**inference-time STDP unlock** — forward-only Hebbian fast-weight updates fire
during ``model.eval()`` (the ``self.training`` gate at
``synapforge/bio/stdp_fast.py:121`` was removed 2026-05-01), so the network
self-organises in-context with **no optimizer, no loss, no backprop**.

What this script does
---------------------
1. Loads (or builds) a small SynapForge100M backbone.
2. Wraps each HybridBlock with an :class:`STDPFastWeight` module via a
   forward post-hook. Each block's hidden output is fed to the matching
   STDP layer so the fast-weight matrix evolves with co-activation
   patterns observed at inference.
3. Generates 1024 tokens autoregressively from a seed prompt.
4. Every ``probe_every`` (default 64) tokens, snapshots
   ``‖W_t - W_initial‖`` (Frobenius) for each STDP layer.
5. Emits a JSON timeline + a PNG plot. Prints final-step summary
   (total ΔW, density change, monotonic-up boolean).

Usage
-----
    python scripts/measure_stdp_weight_diff.py
    python scripts/measure_stdp_weight_diff.py --ckpt path/to/ckpt.pt
    python scripts/measure_stdp_weight_diff.py --smoke    # n=64 for tests
    python scripts/measure_stdp_weight_diff.py --n-tokens 4096

Notes
-----
* Inference uses ``model.eval()``; the STDP gate fires anyway because of
  ``stdp_fast.py:127`` (env var ``SYNAPFORGE_STDP_INFERENCE`` defaults
  to "on"). DO NOT wrap STDP forwards in ``torch.no_grad()`` — the rule
  itself only mutates buffers (already inside its own ``no_grad`` block)
  but its forward must run end-to-end so the post-trace + LTP/LTD update
  fires every step.
* Density is the fraction of ``|W| > 0.05`` (matches ``stdp_demo.py``
  ``_density``). Tracks structural couplings, not background trace fog.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on path so the script runs both from inside the repo and from
# anywhere via absolute path invocation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from synapforge.bio.stdp_fast import STDPFastWeight  # noqa: E402
from synapforge.model_100m import SynapForge100M  # noqa: E402


# ---------------------------------------------------------------------------
# STDP wrapping — attach one STDPFastWeight per HybridBlock via post-hook.
# ---------------------------------------------------------------------------


def _attach_stdp_layers(
    model: SynapForge100M,
    a_plus: float = 0.012,
    a_minus: float = 0.022,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    clip: float = 1.0,
) -> list[STDPFastWeight]:
    """Attach one ``STDPFastWeight(d)`` per HybridBlock.

    The STDP layer reads the LAST-position hidden of the block's output
    (so it sees the most recently mixed embedding) and is driven by the
    block-output spike pattern (thresholded hidden as the spike mask).
    Returns the list of attached layers in block order.

    The hook runs DURING the block's forward pass, so each block sees
    every token via this STDP probe. ``model.eval()`` does NOT silence
    the rule — that's the headline unlock.
    """
    d = model.d
    layers: list[STDPFastWeight] = []
    for block in model.blocks:
        stdp = STDPFastWeight(
            hidden_size=d,
            a_plus=a_plus,
            a_minus=a_minus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            clip=clip,
        )
        # Move to whatever device the block is on.
        try:
            example_param = next(block.parameters())
            stdp = stdp.to(device=example_param.device, dtype=example_param.dtype)
        except StopIteration:
            pass
        # eval mode is mandatory: we want to verify the rule fires at
        # inference. The gate at stdp_fast.py:127 reads SYNAPFORGE_STDP_INFERENCE
        # which defaults to "on", so the rule runs regardless of training/eval.
        stdp.eval()
        layers.append(stdp)

        def _hook(_mod, _inp, out, _stdp=stdp):
            # ``out`` is (B, T, d). We feed the per-token hidden into the
            # STDP layer (one batched call across the whole sequence).
            # The hidden's sign is the post-spike pattern (>0 → fire).
            if not isinstance(out, torch.Tensor) or out.dim() != 3:
                return out
            h = out.detach()
            B, T, D = h.shape
            x_flat = h.reshape(B * T, D)
            spike = (x_flat > 0).float()
            _ = _stdp(x_flat, spike=spike)
            return out

        block.register_forward_hook(_hook)
    return layers


def _density(W: torch.Tensor, eps: float = 0.05) -> float:
    """Fraction of |W| above eps (matches stdp_demo._density)."""
    return float((W.detach().abs() > eps).float().mean().item())


# ---------------------------------------------------------------------------
# Model factory (small or from ckpt)
# ---------------------------------------------------------------------------


def _build_model(
    ckpt: str | None,
    vocab: int = 512,
    d: int = 128,
    n_layers: int = 2,
    max_seq: int = 4096,
) -> SynapForge100M:
    """Build a small SynapForge100M; optionally warm-start from ``ckpt``."""
    torch.manual_seed(7)
    model = SynapForge100M(
        vocab=vocab,
        d=d,
        n_layers=n_layers,
        loop_depth=1,
        max_seq=max_seq,
        ffn_ratio=2.0,
        sparsity=0.95,
        dropout=0.0,
        tie_lm_head=True,
        freeze_vocab_tail=False,  # tiny vocab; nothing to freeze
    )
    if ckpt and Path(ckpt).is_file():
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"  loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Autoregressive sampling + per-N-token measurement
# ---------------------------------------------------------------------------


def measure_stdp_weight_diff(
    model: SynapForge100M,
    stdp_layers: list[STDPFastWeight],
    n_tokens: int = 1024,
    probe_every: int = 64,
    seed_prompt_ids: torch.Tensor | None = None,
    seed: int = 11,
    temperature: float = 1.0,
) -> dict:
    """Run ``n_tokens`` autoregressive forwards, snap ‖ΔW‖ every probe_every.

    Returns a dict with:
      - ``timeline``  : list of {"token_idx": int, "layer_<i>_delta": float}.
      - ``W_initial`` : per-layer initial frobenius norm.
      - ``W_final``   : per-layer final frobenius norm.
      - ``density_initial``, ``density_final``: per-layer density.
      - ``monotonic_up`` : True iff every per-layer delta sequence is
        non-decreasing.
      - ``total_delta`` : sum of per-layer final ‖ΔW‖.
    """
    if probe_every <= 0:
        raise ValueError(f"probe_every must be > 0, got {probe_every}")
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be > 0, got {n_tokens}")
    if not stdp_layers:
        raise ValueError("no STDP layers attached")

    device = next(model.parameters()).device
    rng = torch.Generator(device="cpu").manual_seed(seed)

    # Snapshot W_initial per layer BEFORE any forwards.
    W_initial = [layer.W.detach().clone() for layer in stdp_layers]
    density0 = [_density(W) for W in W_initial]

    # Seed prompt: 8 random tokens (or user-provided).
    if seed_prompt_ids is None:
        seed_prompt_ids = torch.randint(
            0, model.vocab, (1, 8), generator=rng, dtype=torch.long
        )
    seed_prompt_ids = seed_prompt_ids.to(device=device)

    timeline: list[dict] = []
    # token_idx 0 = baseline (always 0.0 by construction).
    timeline.append(
        {
            "token_idx": 0,
            **{f"layer_{i}_delta": 0.0 for i in range(len(stdp_layers))},
        }
    )

    ids = seed_prompt_ids
    # Cap context so we never exceed model.max_seq during AR sampling.
    max_seq = int(getattr(model, "max_seq", 4096))

    model.eval()
    # Note: NO `with torch.no_grad():` here. The STDP rule itself runs
    # under its own no_grad context (see stdp_fast.py:128). The model
    # forward without no_grad outer context costs a bit of memory but
    # confirms the headline claim that mutation works regardless of the
    # autograd state. Verified in test_no_grad_path.
    for tok in range(1, n_tokens + 1):
        # Truncate context if it would exceed max_seq.
        if ids.shape[1] >= max_seq:
            ids = ids[:, -(max_seq - 1):]
        logits = model(ids)  # (1, T, vocab)
        last_logits = logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(last_logits / max(1e-6, temperature), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = last_logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

        if tok % probe_every == 0 or tok == n_tokens:
            row: dict = {"token_idx": tok}
            for i, (layer, W0) in enumerate(zip(stdp_layers, W_initial)):
                with torch.no_grad():
                    diff = layer.W - W0
                    norm = float(torch.linalg.norm(diff.float()).item())
                row[f"layer_{i}_delta"] = norm
            timeline.append(row)

    W_final_norms = []
    density_final = []
    for layer in stdp_layers:
        with torch.no_grad():
            W_final_norms.append(
                float(torch.linalg.norm(layer.W.detach().float()).item())
            )
            density_final.append(_density(layer.W))

    # Monotonic check: each per-layer delta sequence non-decreasing.
    monotonic_up = True
    for i in range(len(stdp_layers)):
        prev = -1.0
        for row in timeline:
            v = row[f"layer_{i}_delta"]
            if v < prev - 1e-6:
                monotonic_up = False
                break
            prev = v
        if not monotonic_up:
            break

    total_delta = float(timeline[-1][f"layer_{len(stdp_layers) - 1}_delta"])
    # The "total" interpretation users care about is the sum over layers.
    total_sum = sum(
        timeline[-1][f"layer_{i}_delta"] for i in range(len(stdp_layers))
    )

    return {
        "n_tokens": int(n_tokens),
        "probe_every": int(probe_every),
        "n_layers": len(stdp_layers),
        "timeline": timeline,
        "W_initial_norm_per_layer": [
            float(torch.linalg.norm(W.float()).item()) for W in W_initial
        ],
        "W_final_norm_per_layer": W_final_norms,
        "density_initial_per_layer": density0,
        "density_final_per_layer": density_final,
        "monotonic_up": bool(monotonic_up),
        "total_delta_W_final_layer": float(total_delta),
        "total_delta_sum_all_layers": float(total_sum),
    }


# ---------------------------------------------------------------------------
# Plotting (matplotlib non-interactive)
# ---------------------------------------------------------------------------


def write_plot(result: dict, out_png: str) -> None:
    """Plot per-layer ‖ΔW‖ over token index. Saves to ``out_png`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")  # headless / CI-safe
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    timeline = result["timeline"]
    xs = [r["token_idx"] for r in timeline]
    n_layers = result["n_layers"]
    for i in range(n_layers):
        ys = [r[f"layer_{i}_delta"] for r in timeline]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"layer {i}")
    ax.set_xlabel("Token index")
    ax.set_ylabel(r"$\|W_t - W_{0}\|_F$ (Frobenius)")
    title = (
        f"Inference-time STDP weight drift "
        f"(n={result['n_tokens']} tokens, monotonic_up={result['monotonic_up']})"
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    ckpt: str | None,
    n_tokens: int,
    out_json: str,
    out_png: str,
    seed: int,
    probe_every: int = 64,
    smoke: bool = False,
) -> dict:
    if smoke:
        n_tokens = 64
        probe_every = 16
    print(
        f"  building SynapForge100M (vocab=512, d=128, n_layers=2)"
        + (f" + ckpt={ckpt}" if ckpt else " + random init")
    )
    model = _build_model(ckpt)
    print(
        f"  total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    print("  attaching STDPFastWeight to each HybridBlock...")
    stdp_layers = _attach_stdp_layers(model)
    print(f"  attached {len(stdp_layers)} STDP layers (d={model.d})")

    print(f"  running {n_tokens}-token autoregressive sample, probe every {probe_every}...")
    result = measure_stdp_weight_diff(
        model=model,
        stdp_layers=stdp_layers,
        n_tokens=n_tokens,
        probe_every=probe_every,
        seed=seed,
    )

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_plot(result, out_png)

    print()
    print(f"  per-layer final ||DW||:")
    for i, v in enumerate(result["timeline"][-1].items()):
        if v[0].endswith("_delta"):
            print(f"    {v[0]}: {v[1]:.6f}")
    print(f"  monotonic_up: {result['monotonic_up']}")
    print(f"  density: {result['density_initial_per_layer']} -> "
          f"{result['density_final_per_layer']}")
    print(f"  total ||DW||_sum across layers: {result['total_delta_sum_all_layers']:.6f}")
    print()
    print(f"  json -> {out_json}")
    print(f"  png  -> {out_png}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--ckpt", default=None, help="optional path to Synap-1 checkpoint (.pt)"
    )
    ap.add_argument("--n-tokens", type=int, default=1024)
    ap.add_argument(
        "--probe-every",
        type=int,
        default=64,
        help="snap ||DW|| every N generated tokens",
    )
    ap.add_argument(
        "--out-json",
        default="runs/stdp_weight_diff/timeline.json",
    )
    ap.add_argument(
        "--out-png",
        default="runs/stdp_weight_diff/delta_W.png",
    )
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="fast 64-token run for tests (overrides --n-tokens)",
    )
    args = ap.parse_args(argv)
    run(
        ckpt=args.ckpt,
        n_tokens=args.n_tokens,
        out_json=args.out_json,
        out_png=args.out_png,
        seed=args.seed,
        probe_every=args.probe_every,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
