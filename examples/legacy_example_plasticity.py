"""End-to-end demo: BP gradient + sf.plasticity STDP both update SAME weight.

Tiny model: SparseSynapse (BP-trainable) + STDP rule (plasticity-trainable
on the SAME synapse weight). Trained 100 steps on a synthetic regression
target, asserting that BOTH update streams are non-zero every step.

The point: the deferred-delta execution model (observe-then-step-then-apply)
means we can mutate W in-place AFTER backward without raising autograd
version errors — even though forward read W and backward computed grads
on it.

Run:
    CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python /workspace/synapforge/example_plasticity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import synapforge as sf
from synapforge.plasticity import STDP, PlasticityEngine


def main() -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    D, BATCH, STEPS = 32, 64, 100

    # ---- Build model ----
    syn = sf.SparseSynapse(D, D, sparsity=0.5, bias=False).to(dev)

    # ---- Build plasticity rule + engine. The rule's delta will land on
    # the SAME weight (syn.weight) that BP gradients also touch. ----
    stdp = STDP(lr=1e-3, tau_pre=20.0, tau_post=20.0, decay_after_compute=0.5).to(dev)
    engine = PlasticityEngine({"syn.weight": stdp}, schedule="every:1")

    # ---- BP optimizer over the synaptic weight ----
    optim = torch.optim.AdamW(syn.parameters(), lr=3e-3, weight_decay=0.01)

    # ---- Synthetic regression target ----
    X = torch.randn(BATCH, D, device=dev)
    A_target = torch.randn(D, D, device=dev) * 0.3
    Y = X @ A_target.t()

    bp_norms, plast_norms, losses = [], [], []
    weights_dict = {"syn.weight": syn.weight}

    for step in range(STEPS):
        # ---- Forward + BP loss ----
        optim.zero_grad()
        pre = X
        post = syn(pre)                    # forward USES syn.weight
        loss = (post - Y).pow(2).mean()
        loss.backward()                    # BP grad lands on syn.weight.grad
        bp_norm = syn.weight.grad.detach().norm().item()
        bp_norms.append(bp_norm)

        # ---- Plasticity OBSERVE phase (NEVER mutates W during forward) ----
        with torch.no_grad():
            stdp.observe(pre=pre, post=post.detach(), t=1.0)

        # ---- Apply BP step (consumes the autograd graph) ----
        optim.step()

        # ---- Plasticity DELTA + APPLY phase. After optim.step the autograd
        # graph for syn.weight is dead, so in-place mutation is safe. ----
        deltas = engine.step(t=step, weight_dict=weights_dict)
        plast_norm = deltas["syn.weight"].norm().item() if "syn.weight" in deltas else 0.0
        plast_norms.append(plast_norm)
        engine.apply(deltas, weights_dict)

        losses.append(loss.item())

    # ---- Assertions ----
    avg_bp = sum(bp_norms) / len(bp_norms)
    avg_plast = sum(plast_norms) / len(plast_norms)
    assert avg_bp > 1e-4, f"BP grad stream empty (avg={avg_bp:.2e})"
    assert avg_plast > 1e-8, f"Plasticity stream empty (avg={avg_plast:.2e})"
    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not converge: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )

    print(f"[example] device                   = {dev}")
    print(f"[example] sparsity (active frac)   = {syn.density():.3f}")
    print(f"[example] avg BP grad norm         = {avg_bp:.4e}")
    print(f"[example] avg plasticity dW norm   = {avg_plast:.4e}")
    print(f"[example] loss {losses[0]:.4f}  ->  {losses[-1]:.4f}  "
          f"(drop {100 * (1 - losses[-1] / losses[0]):.1f}%)")
    print("[example] STDP delta shape         =",
          tuple(deltas["syn.weight"].shape))
    print("[example] BP + plasticity both active, no autograd version conflict.")


if __name__ == "__main__":
    main()
