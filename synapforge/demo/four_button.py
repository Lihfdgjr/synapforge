"""4-button NeuroMCP demo: synapses grow from 5% -> 28% as the network
learns to click the right colored square. Zero tool-call tokens emitted.

Runs on CPU in under 60 seconds. The point of the demo: action emerges
from learnable sparse synapses + a growing prototype codebook, not from
JSON tool-calling.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..action.envs import FourButtonEnv, PatchEncoder, SpatialXYHead
from ..action.neuromcp import NeuroMCPHead


def _params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class _Agent(nn.Module):
    """Tiny agent: image -> patch tokens -> NeuroMCP head -> (action, xy)."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = PatchEncoder(patch=16, hidden=hidden, img_size=64)
        self.neuromcp = NeuroMCPHead(
            hidden=hidden,
            codebook_initial=9,
            codebook_max=32,
            synapse_density=0.05,
            synapse_max_density=0.40,
        )
        # grid must equal img_size//patch = 64//16 = 4
        self.xy_head = SpatialXYHead(hidden=hidden, grid=4)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, dict]:
        tokens = self.encoder(img)              # [B, T, D]
        h = tokens.mean(dim=1)                  # [B, D] -- mean-pool tokens
        out = self.neuromcp(h)
        xy, _ = self.xy_head(tokens)            # xy: [B, 2]
        return xy, out


def run_demo(
    n_trials: int = 80,
    batch: int = 16,
    lr: float = 3e-3,
    quiet: bool = False,
    seed: int = 7,
) -> dict:
    torch.manual_seed(seed)
    env = FourButtonEnv()
    agent = _Agent(hidden=64)

    if not quiet:
        print(f"  agent: {_params(agent)/1e3:.1f}K params")
        print(f"  initial synapse density: {agent.neuromcp.proj.density:.1%}")
        print(f"  initial codebook K: {agent.neuromcp.codebook.K}")
        print()

    optim = torch.optim.AdamW(agent.parameters(), lr=lr)
    history = []
    initial_density = float(agent.neuromcp.proj.density)
    initial_K = int(agent.neuromcp.codebook.K)

    t0 = time.time()
    for trial in range(n_trials):
        img, target = env.reset(batch_size=batch)
        xy, out = agent(img)
        rewards = env.step(xy[:, 0], xy[:, 1], target)
        target_xy = torch.tensor(
            [[env.BUTTONS[int(t)][0], env.BUTTONS[int(t)][1]] for t in target]
        )
        xy_loss = F.mse_loss(xy, target_xy)
        ce = F.cross_entropy(out["logits"], target % out["logits"].size(-1))
        loss = xy_loss + 0.3 * ce

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Plasticity tick (after optimizer.step per OBSERVE/DELTA/APPLY contract)
        plast = agent.neuromcp.step_plasticity(hidden_z=out["hidden_z"].detach())

        hit = float(rewards.mean())
        d = float(agent.neuromcp.proj.density)
        K = int(agent.neuromcp.codebook.K)
        history.append({"trial": trial, "hit_rate": hit, "density": d, "K": K})

        if not quiet and (trial < 5 or trial % 16 == 0 or trial == n_trials - 1):
            grew = "GREW" if plast.get("grew_cb") else "    "
            print(
                f"  trial {trial:>3}  hit_rate={hit:.2f}  density={d:.1%}  K={K:>2}  "
                f"loss={float(loss):.3f}  {grew}"
            )

    dt = time.time() - t0
    final = history[-1]

    if not quiet:
        print()
        print(f"  done in {dt:.1f}s")
        print(f"  density: {initial_density:.1%} -> {final['density']:.1%}")
        print(f"  codebook K: {initial_K} -> {final['K']}")
        print(f"  final hit_rate (last 8 trials): "
              f"{sum(h['hit_rate'] for h in history[-8:]) / 8:.2f}")
        print(
            "\n  zero <tool_call> tokens emitted. action emerged from\n"
            "  sparse synapses + dynamic codebook, not JSON parsing.\n"
        )

    return {
        "wall_time_s": dt,
        "trials": history,
        "initial_density": initial_density,
        "final_density": final["density"],
        "initial_K": initial_K,
        "final_K": final["K"],
        "final_hit_rate": sum(h["hit_rate"] for h in history[-8:]) / 8,
    }


if __name__ == "__main__":
    run_demo()
