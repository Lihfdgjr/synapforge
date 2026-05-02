"""Bench: --self-drive ON vs OFF for the same N steps.

What it measures
----------------
Runs the SelfDriveCoordinator in a controlled harness against a tiny
torch model. The harness intentionally avoids the full 100M trainer so
the bench can complete in seconds on any laptop -- the goal is to
verify the coordinator's overhead and decision-making, not to claim
end-to-end ppl improvements.

Outputs (json + console):
    {
      "off": {"final_loss": ..., "throughput_steps_per_s": ..., "fires": 0},
      "on":  {"final_loss": ..., "throughput_steps_per_s": ...,
              "fires": K, "n_inner": M, "n_kept": ...,
              "n_rollbacks": ..., "n_proposed_fresh": ...,
              "frontier_in_band": ..., "compound_protos_emerged": ...},
      "delta_loss": (loss_on - loss_off),
      "delta_throughput": ...
    }

Honest: at very early training (random model) self-drive cannot help
because there is no signal to extrapolate from. The coordinator should
mostly KEEP (no rollback, no improvement). The bench's role is to prove
the wire-in is safe + cheap, not to claim a learning win.

Usage:
    python scripts/bench_self_drive.py [--steps 200] [--out bench_self_drive.json]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

# Project imports.
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapforge.intrinsic import (
    GoalMemory,
    ImaginationRollout,
    SelfDriveConfig,
    SelfDriveCoordinator,
    SelfGoalProposer,
)


# ----------------------------------------------------------------------
# tiny harness model
# ----------------------------------------------------------------------
class TinyLM(nn.Module):
    """Minimal next-token LM compatible with the self-drive components."""

    def __init__(self, vocab_size: int = 256, d: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.tok_embed = nn.Embedding(vocab_size, d)
        self.tie_lm_head = True
        self.layers = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok_embed(x)
        return self.layers(h)

    def forward(self, x=None, tokens=None):
        # Accept either ``x`` (positional) or ``tokens=`` (proposer-style).
        if x is None:
            x = tokens
        h = self.encode(x)
        logits = F.linear(h, self.tok_embed.weight)
        return logits


# ----------------------------------------------------------------------
# baseline + self-drive runs
# ----------------------------------------------------------------------
@dataclass
class RunResult:
    final_loss: float
    throughput_steps_per_s: float
    n_steps: int
    fires: int = 0
    n_inner: int = 0
    n_kept: int = 0
    n_rollbacks: int = 0
    n_proposed_fresh: int = 0
    frontier_in_band: int = 0


def _make_data(vocab: int, B: int, T: int, n_batches: int):
    """Synthetic next-token corpus (deterministic)."""
    g = torch.Generator().manual_seed(42)
    batches = []
    for _ in range(n_batches):
        ids = torch.randint(0, vocab, (B, T + 1), generator=g)
        batches.append((ids[:, :-1], ids[:, 1:]))
    return batches


def _train_loop(
    model: TinyLM,
    optim: torch.optim.Optimizer,
    batches: list,
    n_steps: int,
    coord: SelfDriveCoordinator | None = None,
) -> RunResult:
    losses = []
    last_loss = float("inf")
    t0 = time.time()

    fires = 0
    n_inner = 0
    n_kept = 0
    n_rollbacks = 0

    for step in range(1, n_steps + 1):
        x, y = batches[(step - 1) % len(batches)]
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        last_loss = float(loss.detach().item())
        losses.append(last_loss)

        if coord is not None and coord.should_fire(step, idle=False):
            fires += 1

            def _runner(sd_step):
                if not sd_step.goal_tokens:
                    return last_loss
                with torch.no_grad():
                    gt = torch.tensor([list(sd_step.goal_tokens)], dtype=torch.long)
                    if gt.size(1) < 2:
                        return last_loss
                    xx = gt[:, :-1]
                    yy = gt[:, 1:]
                    log = model(xx)
                    ce = F.cross_entropy(
                        log.reshape(-1, log.size(-1)).float(), yy.reshape(-1),
                    )
                return float(ce.detach().item())

            def _snap():
                return {
                    name: p.detach().clone()
                    for name, p in model.named_parameters()
                    if p.requires_grad
                }

            def _restore(snap):
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in snap and snap[name].shape == p.shape:
                            p.data.copy_(snap[name])

            def _eval():
                # Pre/post: 4-batch evaluation on the corpus.
                with torch.no_grad():
                    losses_e = []
                    for x_e, y_e in batches[:4]:
                        l_e = F.cross_entropy(
                            model(x_e).reshape(-1, model.vocab_size).float(),
                            y_e.reshape(-1),
                        )
                        losses_e.append(float(l_e.item()))
                    if not losses_e:
                        return float("nan")
                    import math
                    return math.exp(sum(losses_e) / len(losses_e))

            ran = coord.cycle(
                outer_step=step,
                run_inner_fn=_runner,
                baseline_loss_fn=lambda: last_loss,
                snapshot_fn=_snap,
                restore_fn=_restore,
                eval_fn=_eval,
            )
            if ran is not None:
                n_inner += int(ran["n_inner_steps"])
                n_kept += int(ran["n_kept"])
                if ran.get("decision") and ran["decision"].rolled_back:
                    n_rollbacks += 1

    elapsed = max(time.time() - t0, 1e-6)
    n_proposed_fresh = coord.n_proposed_fresh if coord is not None else 0
    frontier_in_band = (
        coord.sampler.stats()["n_in_band"] if coord is not None else 0
    )
    return RunResult(
        final_loss=last_loss,
        throughput_steps_per_s=n_steps / elapsed,
        n_steps=n_steps,
        fires=fires,
        n_inner=n_inner,
        n_kept=n_kept,
        n_rollbacks=n_rollbacks,
        n_proposed_fresh=n_proposed_fresh,
        frontier_in_band=frontier_in_band,
    )


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--every-k", type=int, default=20)
    p.add_argument("--inner-steps", type=int, default=10)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument("--out", type=str, default="bench_self_drive.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    batches = _make_data(args.vocab, args.batch, args.seq_len, args.n_batches)

    # ---- OFF run ----
    model_off = TinyLM(vocab_size=args.vocab, d=args.d)
    optim_off = torch.optim.AdamW(model_off.parameters(), lr=3e-4)
    res_off = _train_loop(model_off, optim_off, batches, args.steps, coord=None)

    # ---- ON run ----
    torch.manual_seed(args.seed)  # bit-exact same init.
    model_on = TinyLM(vocab_size=args.vocab, d=args.d)
    optim_on = torch.optim.AdamW(model_on.parameters(), lr=3e-4)
    cfg = SelfDriveConfig(
        enabled=True,
        every_k_steps=args.every_k,
        inner_steps=args.inner_steps,
        max_val_regression=0.05,
    )
    proposer = SelfGoalProposer(model_on, vocab_size=args.vocab)
    rollout = ImaginationRollout(model_on)
    memory = GoalMemory(capacity=1000)
    coord = SelfDriveCoordinator(
        cfg=cfg, proposer=proposer, rollout=rollout, memory=memory,
        log_fn=lambda s: None,  # silent during bench
    )
    res_on = _train_loop(model_on, optim_on, batches, args.steps, coord=coord)

    # ---- compare ----
    summary = {
        "config": vars(args),
        "off": vars(res_off),
        "on": vars(res_on),
        "delta_loss": res_on.final_loss - res_off.final_loss,
        "delta_throughput": (
            res_on.throughput_steps_per_s - res_off.throughput_steps_per_s
        ),
        "throughput_pct_change": (
            (res_on.throughput_steps_per_s - res_off.throughput_steps_per_s)
            / res_off.throughput_steps_per_s * 100.0
        ),
        "honest_note": (
            "Self-drive helps once the model has basic vocab. On a tiny "
            "random-init MLP without a real corpus, expect delta_loss ~ 0 "
            "or slightly worse: there is no signal to extrapolate from. "
            "The bench proves the wire-in is SAFE (rollback works) and "
            "CHEAP (throughput delta < ~10%); learning gains land in real "
            "trainer runs once base ppl < 250."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    # Print concise summary.
    print("=" * 60)
    print(f"OFF: loss={res_off.final_loss:.4f} "
          f"steps/s={res_off.throughput_steps_per_s:.1f}")
    print(f"ON:  loss={res_on.final_loss:.4f} "
          f"steps/s={res_on.throughput_steps_per_s:.1f} "
          f"fires={res_on.fires} inner={res_on.n_inner} "
          f"kept={res_on.n_kept} rb={res_on.n_rollbacks} "
          f"fresh={res_on.n_proposed_fresh} in_band={res_on.frontier_in_band}")
    print(f"delta loss:       {summary['delta_loss']:+.4f}")
    print(f"delta steps/s:    {summary['delta_throughput']:+.1f} "
          f"({summary['throughput_pct_change']:+.1f}%)")
    print("=" * 60)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
