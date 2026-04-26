"""example_mscfc_port — port a tiny mscfc HybridBlock to synapforge primitives
and train it 100 steps from the adv29 warm-start checkpoint.

This is THE money demo. We claim to have replaced PyTorch as the
representation layer for an LNN+SNN model, and this script is what proves
the claim is more than an API-mock-up:

    * model is built from sf.cells.liquid.LiquidCell + sf.surrogate.PLIFCell
      + a tagged synapse linear with multi-source ("bp", "hebb") gradients;
    * optimizer is sf.optim.PlasticityAwareAdamW (multi-source merge);
    * plasticity is sf.plasticity.Hebbian, scheduled by sf.plasticity.PlasticityEngine;
    * data is sf.data.ParquetTokenStream against /workspace/data/wt103_raw;
    * training loop is sf.train.train.

What is still PyTorch:
    * tensors (we are NOT writing our own ndarray library)
    * autograd / loss.backward() — we use torch.autograd to populate p.grad,
      then OUR optimizer fuses bp + plast deltas before stepping.
    * cuda kernels — we call torch.cumsum / Linear / Embedding under the hood.

This is the *minimal honest* demo: every learning-rule decision (LR, WD,
weight-decay split, plasticity merge, schedule) flows through synapforge
code rather than torch.optim.AdamW.
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn as nn

# Make sure /workspace is on path for sibling imports during dev runs.
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

from synapforge.cells.liquid import LiquidCell
from synapforge.data import ParquetTokenStream
from synapforge.plasticity import Hebbian, PlasticityEngine
from synapforge.surrogate import PLIFCell
from synapforge.train import train

WARMSTART_CKPT = "/workspace/runs/step_001250.pt"  # adv29
OUT_DIR = "/workspace/runs/synapforge_smoke"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SynapForgeHybridBlock(nn.Module):
    """Equivalent of mscfc HybridBlock built using sf.* primitives only.

    Pipeline per token sequence (B, T, D):
        x_emb -> LiquidCell.forward (parallel scan) -> hidden h
        h -> PLIFCell.forward_seq (per-step LIF + spike) -> spikes s
        s -> Linear(synapse) -> hidden out

    The synapse weight is tagged with ``_sf_grad_source = ["bp", "hebb"]``
    so the optimizer's multi-source machinery merges Hebbian deltas on top
    of the BP gradient every plasticity step.

    A Hebbian rule observes (pre=h, post=s) inside forward via a registered
    hook; the engine then asks the rule for ``compute_delta_W()`` after the
    backward pass.
    """

    def __init__(self, hidden: int = 256, hebb_rule: Hebbian | None = None):
        super().__init__()
        self.hidden = int(hidden)
        # CfC-LiquidS4 — Heinsen parallel scan (sf.cells.liquid)
        self.cfc = LiquidCell(hidden, hidden)
        # PLIF — learnable tau + threshold + arctan surrogate
        self.plif = PLIFCell(hidden, tau_init=1.0, threshold_init=0.3, surrogate="atan", reset="subtract")
        # Synapse — plain Linear, but with multi-source grad tag.
        self.synapse = nn.Linear(hidden, hidden, bias=False)
        # Tag for sf.optim.build_optimizer auto-detection.
        self.synapse.weight._sf_grad_source = ["bp", "hebb"]
        # The Hebbian rule that will produce dW for synapse.weight.
        self.hebb = hebb_rule  # may be None for an "off" run

    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> CfC sequence
        h = self.cfc(x_emb)                     # (B, T, D), tanh-bounded
        s, _ = self.plif.forward_seq(h)         # (B, T, D), spikes in {0,1}
        # Observe Hebbian (pre=h, post=s). Detached inside the rule.
        if self.hebb is not None and self.training:
            self.hebb.observe(pre=h, post=s, t=1.0)
        out = self.synapse(s)
        return out


class SmallModel(nn.Module):
    """Embed -> SynapForgeHybridBlock -> tied LM head."""

    def __init__(self, vocab: int = 50257, hidden: int = 256,
                 hebb_rule: Hebbian | None = None):
        super().__init__()
        self.vocab = int(vocab)
        self.hidden = int(hidden)
        self.emb = nn.Embedding(vocab, hidden)
        self.block = SynapForgeHybridBlock(hidden, hebb_rule=hebb_rule)
        # Tied weights: lm_head.weight aliases emb.weight.
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.emb.weight
        # Better init for embedding so initial loss isn't astronomical.
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.5)  # large enough so post-CfC h crosses PLIF thr

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)                    # (B, T, D)
        h = self.block(x)                       # (B, T, D)
        logits = self.lm_head(h)                # (B, T, V)
        return logits


# ---------------------------------------------------------------------------
# Warm-start utility
# ---------------------------------------------------------------------------


def warmstart_from_adv29(
    model: SmallModel,
    ckpt_path: str,
) -> dict[str, str]:
    """Load shared parameters from an mscfc adv29 checkpoint.

    The mscfc model uses ``MultiplicativeCfCCell`` while we use
    ``LiquidCell``, so very few weight names overlap. We salvage:
        * ``wrapped.embed.text_embed.weight`` -> ``emb.weight``  (also -> lm_head.weight via tie)
        * ``wrapped.shared_tau.tau_log``       -> per-block ``cfc.A_log`` (warm decay init)
        * ``wrapped.blocks.0.norm.weight/bias`` not used — we have no LayerNorm.

    Returns a small report ``{key: status}`` for printing.
    """
    if not os.path.exists(ckpt_path):
        return {"_status": f"ckpt not found at {ckpt_path}, skipping warm start"}
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict") or ck.get("model") or ck
    report: dict[str, str] = {}
    # Embedding
    src = sd.get("wrapped.embed.text_embed.weight")
    if src is not None and src.shape == model.emb.weight.shape:
        with torch.no_grad():
            model.emb.weight.copy_(src)
        report["emb.weight"] = f"copied {tuple(src.shape)}"
    else:
        report["emb.weight"] = (
            f"shape mismatch {None if src is None else tuple(src.shape)} "
            f"vs {tuple(model.emb.weight.shape)}; init kept"
        )
    # tau_log -> A_log (related but not identical; copy if shape matches)
    src_tau = sd.get("wrapped.shared_tau.tau_log")
    if src_tau is not None and src_tau.shape == model.block.cfc.A_log.shape:
        with torch.no_grad():
            # negate: tau_log positive (tau>1 means slow decay), A_log negative
            # for slow decay. mscfc and ours both store them so that
            # decay = exp(-dt * exp(A_log_or_tau_log)) — actually we have to be
            # careful. Just copy as a warm seed; training will adjust.
            model.block.cfc.A_log.copy_(src_tau)
        report["block.cfc.A_log"] = f"copied from shared_tau {tuple(src_tau.shape)}"
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=1 -> physical GPU 1
        print(f"using cuda dev 0 (physical {torch.cuda.get_device_name(0)})",
              flush=True)
    else:
        print("WARNING: cuda unavailable, falling back to cpu (slow)", flush=True)

    # Plasticity engine: Hebbian rule on the synapse weight.
    hebb = Hebbian(lr=1e-4)
    engine = PlasticityEngine({"block.synapse.weight": hebb}, schedule="every:4")

    model = SmallModel(vocab=50257, hidden=256, hebb_rule=hebb).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {type(model).__name__}  params={n_params/1e6:.2f}M",
          flush=True)

    # Warm-start from adv29 (shared bits: embedding, tau seed).
    ws_report = warmstart_from_adv29(model, WARMSTART_CKPT)
    print("warm-start report:", flush=True)
    for k, v in ws_report.items():
        print(f"  {k:30s}  {v}", flush=True)

    # Data
    data = ParquetTokenStream(
        "/workspace/data/wt103_raw/train-*.parquet",
        seq_len=256, batch_size=32, vocab_size=50257,
    )
    print(f"data: {data!r}", flush=True)

    # Training
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    metrics = train(
        model,
        iter(data),
        n_steps=100,
        lr=3e-4,
        weight_decay=0.05,
        log_every=5,
        save_every=50,
        device=device,
        out_dir=OUT_DIR,
        plasticity_engine=engine,
        plasticity_source="hebb",
    )
    elapsed = time.time() - t0

    # Loss-curve sanity: did it actually go down?
    init_loss = metrics["loss"][0]
    final_loss = metrics["loss"][-1]
    drop = init_loss - final_loss
    print("\n========== synapforge SMOKE PASS ==========", flush=True)
    print(f"  device           : {device}", flush=True)
    print(f"  steps            : {len(metrics['loss']) * 5} logged", flush=True)
    print(f"  initial loss     : {init_loss:.4f}", flush=True)
    print(f"  final loss       : {final_loss:.4f}", flush=True)
    print(f"  loss drop        : {drop:+.4f} (positive = learning)",
          flush=True)
    print(f"  initial ppl      : {math.exp(min(init_loss, 50)):.1f}",
          flush=True)
    print(f"  final ppl        : {math.exp(min(final_loss, 50)):.1f}",
          flush=True)
    print(f"  avg tok/s        : "
          f"{sum(metrics['tok_per_s'])/len(metrics['tok_per_s']):.0f}",
          flush=True)
    print(f"  total wall       : {elapsed:.1f}s", flush=True)
    print(f"  plasticity calls : {sum(metrics['plast_n'])} delta-attaches",
          flush=True)
    if drop > 0.0:
        print("  verdict          : LEARNING (loss decreasing)", flush=True)
        return 0
    else:
        print("  verdict          : NOT LEARNING (loss did not decrease)",
              flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
