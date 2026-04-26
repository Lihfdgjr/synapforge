"""sf example #06 — neural computer use without MCP / tool tokens.

Builds a NeuralComputerAgent using ALL synapforge primitives:
    sf.action.PatchEncoder        screen -> tokens
    sf.LiquidCell                 CfC time-mixer (continuous-time)
    sf.action.NeuroMCPHead        tool-codebook + sparse projection
    sf.action.ActionHead          structured action vector
    sf.action.OSActuator          OS dispatcher (safe_mode for tests)
    sf.action.FourButtonEnv       toy environment

The neuromcp head replaces MCP and JSON tool-schemas: K_alive grows when a
new tool/skill is encountered (novelty > threshold), density grows from
5% to 40% as more synaptic capacity is needed.

Run:
    /opt/conda/bin/python /workspace/synapforge/examples/06_computer_use.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace")

import torch
import torch.nn as nn
import torch.nn.functional as F

import synapforge as sf
import synapforge.action as sfa


class NeuralComputerAgent(sf.Module):
    """Replaces MCP / function-calling: neurons learn OS control directly."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = sfa.PatchEncoder(patch=8, hidden=hidden)
        # CfC time-mixer over patch tokens (acts as the "block" stand-in
        # until the user's main framework lands sf.HybridBlock).
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Linear(hidden * 2, hidden))
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.neuromcp = sfa.NeuroMCPHead(
            hidden,
            codebook_initial=9,
            codebook_max=64,
            synapse_density=0.05,
            synapse_max_density=0.4,
        )
        self.action_head = sfa.ActionHead(hidden, sfa.OSActionSpec.default())

    def forward(self, screen: torch.Tensor) -> dict:
        z = self.encoder(screen)
        z = z + self.mlp(self.norm(z))
        zp = self.pool(z.transpose(1, 2)).squeeze(-1).unsqueeze(1)  # [B,1,D]
        action_logits = self.neuromcp(zp)["logits"]
        actions = self.action_head(zp)
        return {"actions": actions, "action_logits": action_logits, "hidden_z": zp}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    agent = NeuralComputerAgent().to(device)
    actuator = sfa.OSActuator(safe_mode=True)
    env = sfa.FourButtonEnv()
    optim = torch.optim.AdamW(agent.parameters(), lr=1e-3)

    print(f"[example 06] params: {sum(p.numel() for p in agent.parameters())/1e6:.2f}M")
    print(f"[example 06] initial K={agent.neuromcp.codebook.K}, "
          f"density={agent.neuromcp.proj.density:.3f}")

    for step in range(200):
        img, target = env.reset(batch_size=64)
        img, target = img.to(device), target.to(device)
        out = agent(img)
        # Reward = direct hit on button by xy regression head (sigmoid output).
        xy = out["actions"].xy.squeeze(1)  # [B, 2]
        reward = env.step(xy[:, 0].cpu(), xy[:, 1].cpu(), target.cpu()).to(device)

        cb_tgt = (target + 5).clamp(max=agent.neuromcp.codebook.K - 1)
        loss_logits = F.cross_entropy(out["action_logits"].squeeze(1), cb_tgt)
        true_xy = torch.tensor(
            [env.BUTTONS[t.item()] for t in target.cpu()], device=device
        )
        loss_xy = F.smooth_l1_loss(xy, true_xy)
        loss = loss_logits + 2.0 * loss_xy

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        # Plasticity AFTER optim.step (deferred-delta contract).
        stats = agent.neuromcp.step_plasticity(out["hidden_z"].detach())
        if step % 20 == 0:
            print(f"step={step:3d} hit={reward.mean():.2f} "
                  f"K={stats['K_alive']:2d} density={stats['density']:.3f} "
                  f"loss={loss.item():.3f}")

    # Demonstrate the OS actuator handshake (safe_mode prints, no real input).
    with torch.no_grad():
        sample, _ = env.reset(batch_size=1)
        out = agent(sample.to(device))
        action_dict = agent.action_head.to_dict(out["actions"], 0, -1)
    print("\n[example 06] demo action dict from ActionHead:")
    actuator.from_action_dict(action_dict)


if __name__ == "__main__":
    main()
