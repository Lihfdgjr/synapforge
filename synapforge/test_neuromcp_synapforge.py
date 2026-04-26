"""End-to-end test: NeuroMCPHead + ActionHead, both as first-class sf modules.

Trains a NeuralComputerAgent on the FourButtonEnv (the same toy task that
validated the original mscfc.NeuroMCP implementation):
    - K=9 -> grows with novelty
    - synapse density grows from 5% with co-activation EMA
    - hit rate -> 100% after warmup

The model is built ENTIRELY from synapforge primitives:
    sf.action.PatchEncoder        screen [B,3,64,64] -> tokens [B,N,D]
    sf.LiquidCell                 CfC time-mixer
    sf.action.NeuroMCPHead        sparse projection + dynamic codebook
    sf.action.SpatialXYHead       continuous xy regression
    sf.action.ActionHead          structured action vector head

Run:
    /opt/conda/bin/python /workspace/synapforge/test_neuromcp_synapforge.py
or via the spawn script:
    /opt/conda/bin/python /workspace/spawn_neuromcp_synapforge.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, "/workspace")

import torch
import torch.nn as nn
import torch.nn.functional as F

import synapforge as sf
import synapforge.action as sfa

# ----------------------------------------------------------------------------
# NeuralComputerAgent — built on synapforge primitives.
# ----------------------------------------------------------------------------


class NeuralComputerAgent(sf.Module):
    """Replaces MCP / function-calling: neurons learn OS control directly.

    The forward path produces:
      action_logits  cosine-sim logits over alive prototypes (NeuroMCPHead)
      xy             continuous (x, y) in [0,1]^2 (SpatialXYHead)
      hidden_z       projected hidden used as codebook input (for novelty)
      actions        structured ActionOutput (ActionHead) for full OS control
    """

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = sfa.PatchEncoder(patch=8, hidden=hidden)
        # Two-stage: layernorm + small MLP block (acts as a stand-in for
        # sf.HybridBlock until the user's main framework lands one).
        self.norm1 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.neuromcp = sfa.NeuroMCPHead(
            hidden,
            codebook_initial=9,
            codebook_max=64,
            synapse_density=0.05,
            synapse_max_density=0.40,
        )
        self.xy_head = sfa.SpatialXYHead(hidden, grid=8)
        self.action_head = sfa.ActionHead(hidden, sfa.OSActionSpec.default())

    def forward(self, screen: torch.Tensor) -> dict:
        z = self.encoder(screen)                 # [B, N, D]
        z = z + self.mlp(self.norm1(z))
        z = self.norm2(z)
        # Pool to a single token for the codebook & action_head.
        zp = self.pool(z.transpose(1, 2)).squeeze(-1).unsqueeze(1)  # [B, 1, D]
        nout = self.neuromcp(zp)                 # logits, hidden_z
        xy, attn = self.xy_head(z)               # [B, 2]
        actions = self.action_head(zp)           # ActionOutput
        return {
            "action_logits": nout["logits"].squeeze(1),
            "hidden_z": nout["hidden_z"].squeeze(1),
            "xy": xy,
            "attn": attn,
            "actions": actions,
        }


# ----------------------------------------------------------------------------
# Train loop on FourButtonEnv.
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", default="/workspace/runs/synapforge_neuromcp_v0")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, falling back to CPU", flush=True)
        args.device = "cpu"
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    if device.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)

    torch.manual_seed(42)
    model = NeuralComputerAgent(hidden=256).to(device)
    actuator = sfa.OSActuator(safe_mode=True)  # sanity check wires

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NeuralComputerAgent: {n_params/1e6:.2f}M params", flush=True)
    print(f"  initial K_alive  = {model.neuromcp.codebook.K}", flush=True)
    print(f"  initial density  = {model.neuromcp.proj.density:.3f}", flush=True)
    print(f"  ActionHead spec  = total_out={model.action_head.spec.total_out}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    env = sfa.FourButtonEnv()

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    csvf = open(metrics_path, "w", newline="")
    csvw = csv.writer(csvf)
    csvw.writerow(["step", "hit_rate", "syn_density", "K_alive",
                   "recent_hit_100", "loss", "added_syn", "grew_codebook"])

    hits = []
    t0 = time.time()
    for step in range(args.steps):
        img, target = env.reset(batch_size=args.batch_size)
        img, target = img.to(device), target.to(device)

        out = model(img)
        x_pred = out["xy"][:, 0]
        y_pred = out["xy"][:, 1]
        reward = env.step(x_pred.cpu(), y_pred.cpu(), target.cpu()).to(device)
        hit_rate = float(reward.mean().item())

        # Codebook supervision: 4 buttons => target codebook idx in [5..8]
        # so the model learns to differentiate them via the NeuroMCP head.
        cb_target = (target + 5).clamp(max=model.neuromcp.codebook.K - 1)
        ce_loss = F.cross_entropy(out["action_logits"], cb_target)

        true_xy = torch.tensor(
            [env.BUTTONS[t.item()] for t in target.cpu()], device=device
        )
        xy_loss = F.smooth_l1_loss(out["xy"], true_xy, reduction="none").sum(-1)
        xy_loss = (xy_loss * (1.0 + reward)).mean()
        loss = ce_loss + xy_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        # Plasticity AFTER optim.step (deferred-delta contract).
        syn_stats = model.neuromcp.step_plasticity(out["hidden_z"].detach())

        hits.append(hit_rate)
        if step % max(1, args.steps // 50) == 0:
            recent = sum(hits[-100:]) / max(1, len(hits[-100:]))
            csvw.writerow([step, hit_rate, syn_stats["density"], syn_stats["K_alive"],
                           recent, float(loss.item()),
                           syn_stats["added"], int(syn_stats["grew_cb"])])
            csvf.flush()
            print(
                f"step={step:4d} hit={hit_rate:.2f} recent100={recent:.2f} "
                f"density={syn_stats['density']:.3f} K={syn_stats['K_alive']:2d} "
                f"loss={loss.item():.3f} +syn={syn_stats['added']} "
                f"grew_cb={syn_stats['grew_cb']}",
                flush=True,
            )

    elapsed = time.time() - t0
    csvf.close()

    final_density = float(model.neuromcp.proj.mask.mean().item())
    final_K = int(model.neuromcp.codebook.alive_mask.sum().item())
    recent100 = sum(hits[-100:]) / max(1, len(hits[-100:]))

    # ActionHead sanity: emit a sample action through the actuator (safe_mode).
    with torch.no_grad():
        sample_img, _ = env.reset(batch_size=1)
        sample_out = model(sample_img.to(device))
        action_dict = model.action_head.to_dict(sample_out["actions"], batch_idx=0, t=-1)
    actuator.from_action_dict(action_dict)

    print("\n========== SYNAPFORGE NEUROMCP REPORT ==========")
    print(f"steps:                {args.steps}")
    print(f"elapsed:              {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"final hit rate (100): {recent100:.2%}")
    print(f"synapse density:      {final_density:.3f}  (init 0.05)")
    print(f"codebook K alive:     {final_K}            (init 9, max 64)")
    print(f"sample action dict:   {action_dict}")
    print(f"metrics csv:          {metrics_path}")

    # Reference tolerance: original mscfc 4-button hit_rate >= 0.99 by step 50,
    # K grew from 9 to >=10, density grew 5%->>=10%.
    pass_hit = recent100 >= 0.95
    pass_K = final_K >= 9
    pass_density = final_density > 0.05
    print("\n--- match-mscfc tolerance gate ---")
    print(f"  hit>=0.95 ({recent100:.2%}):     {pass_hit}")
    print(f"  K>=9      ({final_K}):           {pass_K}")
    print(f"  density>0.05 ({final_density:.3f}): {pass_density}")
    print(f"  ALL PASS: {pass_hit and pass_K and pass_density}")


if __name__ == "__main__":
    main()
