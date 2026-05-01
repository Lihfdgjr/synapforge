"""scripts/train_neural_web.py — PPO-style neural-action web trainer.

Pipeline (zero JSON tool calls — pure neural action):

    pixels (3,64,64) ──► PatchEncoder ──► hidden (T, D)
                                          │
                                          ├──► critic MLP ──► V(s)
                                          │
                                          └──► NeuroMCPHead.proj/codebook
                                                    │
                                                    └──► ActionHead
                                                          │
                                                          ▼
                                          {type, xy, scroll, key, text_id}
                                                          │
                                                          ▼
                                                  WebBrowserEnv.step()

Policy update: clipped PPO on a small replay window.  STDP / synaptic
growth tick runs on hidden_z under @no_grad() *after* the optimizer step,
exactly like the OBSERVE/DELTA/APPLY contract in sf.plasticity.

Curiosity bonus = NoveltyDrive(h_next) + free-energy proxy
                  (||h_next - predict(h_prev)||).  Added to the env reward.

Smoke run (no browser, no chromium install):

    python scripts/train_neural_web.py --episodes 4 --max-steps 16 \
        --task-level 1 --no-real --out /tmp/neural_web_smoke.json

Real (requires ``playwright install chromium``):

    python scripts/train_neural_web.py --episodes 50 --max-steps 24 \
        --task-level 1 --headless --out runs/neural_web.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# allow ``python scripts/...`` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapforge.action.envs import PatchEncoder
from synapforge.action.head import ActionHead, ActionOutput, OSActionSpec
from synapforge.action.neuromcp import NeuroMCPHead
from synapforge.action.web_env import (
    DEFAULT_TEXT_CODEBOOK,
    WebBrowserEnv,
    WebEnvConfig,
)
from synapforge.intrinsic import FreeEnergySurprise, NoveltyDrive
from synapforge.learn.web_curriculum import WebCurriculum, WebTask


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------


class NeuralWebActor(nn.Module):
    """Pixel-in, action-out actor-critic with neuromorphic stack.

    PatchEncoder → mean-pool → hidden → NeuroMCPHead → ActionHead + critic.
    NeuroMCPHead supplies dynamic codebook routing; ActionHead supplies the
    structured action vector.  Two outputs are fused: codebook logits decide
    *which prototype* (i.e. which skill cluster), ActionHead then emits the
    continuous fields the actuator needs.
    """

    def __init__(
        self,
        hidden: int = 128,
        text_codebook_size: int = 10,
        codebook_initial: int = 9,
        codebook_max: int = 32,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        # Patch encoder over 64x64 pixel obs.
        self.encoder = PatchEncoder(patch=8, hidden=hidden, img_size=64)
        # NeuroMCPHead: SparseSynapticLayer + DynamicCodebook (no JSON).
        self.neuro = NeuroMCPHead(
            hidden=hidden,
            codebook_initial=codebook_initial,
            codebook_max=codebook_max,
        )
        # ActionHead: hidden → (type, xy, scroll, key, text_trigger).
        self.head = ActionHead(hidden=hidden, spec=OSActionSpec.default())
        # Text-id classifier (small): chooses index into text codebook on TYPE.
        self.text_id = nn.Linear(hidden, text_codebook_size)
        # Critic.
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, obs_bcwh: torch.Tensor) -> torch.Tensor:
        """obs (B, 3, 64, 64) → hidden (B, D)."""
        z = self.encoder(obs_bcwh)        # (B, T_patches, D)
        return z.mean(dim=1)              # (B, D)

    def forward(self, obs_bcwh: torch.Tensor) -> dict:
        h = self.encode(obs_bcwh)         # (B, D)
        h_seq = h.unsqueeze(1)            # (B, 1, D) — head expects >=2 dims.
        action_out = self.head(h_seq)
        # NeuroMCPHead: provides neural-codebook routing logits.
        neuro_out = self.neuro(h_seq)
        text_logits = self.text_id(h)
        value = self.critic(h).squeeze(-1)
        return {
            "hidden": h,
            "hidden_z": neuro_out["hidden_z"],
            "neuro_logits": neuro_out["logits"],
            "action_out": action_out,
            "text_logits": text_logits,
            "value": value,
        }

    @torch.no_grad()
    def sample_action_dict(self, fwd: dict, batch_idx: int = 0) -> tuple[dict, dict]:
        """Sample an action dict + remember log-probs for PPO.

        Strategy:
          - sample type from softmax(action_out.action_type_logits)
          - xy / scroll: deterministic mean (sigmoid / tanh) — explore via
            dithering done in env loop, not in the policy.
          - key:  argmax key_logits.
          - text_id: sample from softmax(text_logits).
        """
        out: ActionOutput = fwd["action_out"]
        b, t = batch_idx, -1
        type_logits = out.action_type_logits[b, t]
        type_probs = type_logits.softmax(dim=-1)
        type_id = int(torch.multinomial(type_probs, 1).item())
        type_logp = float(torch.log(type_probs[type_id] + 1e-9).item())
        text_probs = fwd["text_logits"][b].softmax(dim=-1)
        text_id = int(torch.multinomial(text_probs, 1).item())
        text_logp = float(torch.log(text_probs[text_id] + 1e-9).item())

        spec: OSActionSpec = self.head.spec
        type_id = max(0, min(type_id, len(spec.action_types) - 1))
        action = self.head.to_dict(out, batch_idx=b, t=t)
        # Override sampled type and inject text_id.
        action["type"] = spec.action_types[type_id]
        action["text_id"] = text_id

        meta = {
            "type_id": type_id,
            "text_id": text_id,
            "logp": type_logp + text_logp,
            "value": float(fwd["value"][b].item()),
        }
        return action, meta


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    obs: torch.Tensor
    action_type_id: int
    text_id: int
    logp: float
    value: float
    reward: float
    done: bool


@dataclass
class Rollout:
    transitions: list[Transition] = field(default_factory=list)

    def append(self, tr: Transition) -> None:
        self.transitions.append(tr)

    def __len__(self) -> int:
        return len(self.transitions)


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    advs: list[float] = [0.0] * len(rewards)
    last = 0.0
    next_v = 0.0
    for t in reversed(range(len(rewards))):
        not_done = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        last = delta + gamma * lam * not_done * last
        advs[t] = last
        next_v = values[t]
    returns = [a + v for a, v in zip(advs, values)]
    return advs, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(
    actor: NeuralWebActor,
    optim: torch.optim.Optimizer,
    rollout: Rollout,
    advs: list[float],
    returns: list[float],
    clip: float = 0.2,
    epochs: int = 2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.02,
) -> dict:
    obs = torch.stack([t.obs for t in rollout.transitions])           # (N,3,64,64)
    type_ids = torch.tensor([t.action_type_id for t in rollout.transitions])
    text_ids = torch.tensor([t.text_id for t in rollout.transitions])
    old_logps = torch.tensor([t.logp for t in rollout.transitions])
    adv_t = torch.tensor(advs, dtype=torch.float32)
    ret_t = torch.tensor(returns, dtype=torch.float32)
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(epochs):
        fwd = actor(obs)
        type_logits = fwd["action_out"].action_type_logits[:, -1, :]
        text_logits = fwd["text_logits"]
        type_logp = F.log_softmax(type_logits, dim=-1).gather(
            -1, type_ids.unsqueeze(-1)
        ).squeeze(-1)
        text_logp = F.log_softmax(text_logits, dim=-1).gather(
            -1, text_ids.unsqueeze(-1)
        ).squeeze(-1)
        new_logp = type_logp + text_logp
        ratio = torch.exp(new_logp - old_logps)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(fwd["value"], ret_t)
        # Entropy bonus on type distribution
        ent = -(type_logits.softmax(-1) * F.log_softmax(type_logits, -1)).sum(-1).mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * ent
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optim.step()
        stats["policy_loss"] = float(policy_loss.detach())
        stats["value_loss"] = float(value_loss.detach())
        stats["entropy"] = float(ent.detach())
    return stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_episode(
    actor: NeuralWebActor,
    env: WebBrowserEnv,
    task: WebTask,
    novelty: NoveltyDrive,
    fes: FreeEnergySurprise,
    max_steps: int,
    intrinsic_w: float = 0.3,
) -> tuple[Rollout, dict]:
    """Roll out a single episode under the current task.

    Returns the rollout plus a summary dict with success flag, total reward,
    and growth stats from NeuroMCPHead.
    """
    env.cfg.target_text_regex = task.success_text_regex
    env.cfg.target_url_substr = task.success_url_substr
    obs = env.reset(task.url)
    rollout = Rollout()
    total_reward = 0.0
    success = False
    h_prev: torch.Tensor | None = None

    for step in range(min(max_steps, task.max_steps)):
        with torch.no_grad():
            fwd = actor(obs.unsqueeze(0))
        action, meta = actor.sample_action_dict(fwd)
        next_obs, reward, done, info = env.step(action)

        # Curiosity / FE bonus — pure pixel-side, no LM.
        with torch.no_grad():
            h_next = fwd["hidden"].detach()
            nov = float(novelty.novelty(h_next).item())
            fe = 0.0
            if h_prev is not None:
                fe = float(fes.surprise(h_prev, h_next).item())
            intrinsic = intrinsic_w * (nov + fe)
            env.attach_intrinsic(info, intrinsic)
            h_prev = h_next

        shaped = float(reward) + intrinsic
        total_reward += shaped
        rollout.append(Transition(
            obs=obs,
            action_type_id=int(meta["type_id"]),
            text_id=int(meta["text_id"]),
            logp=float(meta["logp"]),
            value=float(meta["value"]),
            reward=shaped,
            done=bool(done),
        ))
        obs = next_obs
        if done:
            success = (
                info["reward_breakdown"]["progress_text"] > 0
                or info["reward_breakdown"]["progress_url"] > 0
            )
            break

    return rollout, {
        "success": success,
        "reward": total_reward,
        "steps": len(rollout),
        "task": task.name,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="path to actor state_dict to warmstart from")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=24)
    p.add_argument("--task-level", type=int, default=1,
                   help="curriculum starting level [1..5]")
    p.add_argument("--headless", action="store_true",
                   help="real Playwright in headless mode (default true if --real)")
    p.add_argument("--real", action="store_true",
                   help="use real Playwright; default off (mock env)")
    p.add_argument("--no-real", dest="real", action="store_false")
    p.set_defaults(real=False)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--out", type=str, default="runs/neural_web_smoke.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    actor = NeuralWebActor(hidden=128, text_codebook_size=len(DEFAULT_TEXT_CODEBOOK))
    if args.ckpt and os.path.exists(args.ckpt):
        actor.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
        print(f"[ckpt] loaded {args.ckpt}")
    optim = torch.optim.AdamW(actor.parameters(), lr=args.lr)
    novelty = NoveltyDrive(hidden_size=128, ema=0.99)
    fes = FreeEnergySurprise(hidden_size=128)

    cfg = WebEnvConfig(
        real=args.real,
        headless=args.headless or args.real,
        text_codebook=DEFAULT_TEXT_CODEBOOK,
    )
    env = WebBrowserEnv(cfg)
    curric = WebCurriculum(start_level=args.task_level)

    log: dict = {
        "args": vars(args),
        "episodes": [],
        "synapse_density": [],
        "codebook_K": [],
        "loss_curve": [],
        "rolling_success": [],
    }

    t0 = time.time()
    for ep in range(args.episodes):
        task = curric.next_task()
        rollout, summary = run_episode(
            actor, env, task,
            novelty=novelty, fes=fes,
            max_steps=args.max_steps,
        )
        rewards = [tr.reward for tr in rollout.transitions]
        values = [tr.value for tr in rollout.transitions]
        dones = [tr.done for tr in rollout.transitions]
        if not rollout.transitions:
            continue
        advs, returns = compute_gae(
            rewards, values, dones, gamma=args.gamma, lam=args.lam
        )
        ppo_stats = ppo_update(actor, optim, rollout, advs, returns)

        # Plasticity tick — STDP/grow/prune, AFTER optimiser step.
        with torch.no_grad():
            obs_t = torch.stack([tr.obs for tr in rollout.transitions])
            fwd = actor(obs_t)
            plast = actor.neuro.step_plasticity(fwd["hidden_z"])
        log["synapse_density"].append(plast["density"])
        log["codebook_K"].append(plast["K_alive"])
        log["loss_curve"].append(ppo_stats)

        progress = curric.record(summary["success"])
        log["episodes"].append({
            **summary,
            "level": progress["level"],
            "promoted": progress["promoted"],
            "ppo": ppo_stats,
            "synapse_density": plast["density"],
            "K": plast["K_alive"],
        })
        log["rolling_success"].append(progress["rolling_success"])
        print(
            f"ep={ep:03d} L{progress['level']} {summary['task']:<24s} "
            f"R={summary['reward']:+.3f} steps={summary['steps']:2d} "
            f"succ={summary['success']} "
            f"ploss={ppo_stats['policy_loss']:+.3f} "
            f"vloss={ppo_stats['value_loss']:.3f} "
            f"ent={ppo_stats['entropy']:.3f} "
            f"K={plast['K_alive']} dens={plast['density']:.3f}"
        )

    env.close()
    log["wallclock_s"] = round(time.time() - t0, 2)
    log["final_density"] = log["synapse_density"][-1] if log["synapse_density"] else None
    log["final_K"] = log["codebook_K"][-1] if log["codebook_K"] else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"[done] {ep + 1} episodes in {log['wallclock_s']}s -> {out_path}")


if __name__ == "__main__":
    main()
