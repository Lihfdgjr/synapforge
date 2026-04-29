"""
DPO Trainer — Direct Preference Optimization (Rafailov et al 2305.18290).

Loss:
  L = -E[(x,y_w,y_l)] log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) -
                              log π_θ(y_l|x)/π_ref(y_l|x)))

For 375M LNN+SNN (per agent synthesis 2026-04-30):
  - β = 0.1 (raise to 0.3 if mode collapse)
  - LoRA r=16, α=32, dropout 0.05
  - lr 5e-7 (DPO ~10× lower than SFT)
  - SFT warmup REQUIRED first
  - Iterative DPO: refresh π_ref every 50 steps
  - Batch 32, grad-accum 4, 3 epochs
  - Eval HarmBench refusal rate every 200 steps

This module provides the loss + data loading; integration with the
SynapForge trainer happens in train_v42_universal.py via flags.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
    category: str = ""
    severity: int = 0
    source: str = ""


class DPOPairLoader:
    """Stream DPO pairs from JSONL produced by RedBlueSelfPlay."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self.path = Path(jsonl_path)

    def __iter__(self) -> Iterator[DPOPair]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield DPOPair(
                    prompt=d.get("prompt", ""),
                    chosen=d.get("chosen", ""),
                    rejected=d.get("rejected", ""),
                    category=d.get("category", ""),
                    severity=int(d.get("severity", 0)),
                    source=d.get("source", "self_play"),
                )

    def collect(self, max_n: Optional[int] = None) -> List[DPOPair]:
        pairs = []
        for p in self:
            pairs.append(p)
            if max_n and len(pairs) >= max_n:
                break
        return pairs


def dpo_loss(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO loss + diagnostics.

    Returns:
        loss        : scalar
        margin      : (B,) per-sample preference margin (positive = correct ordering)
        accuracy    : scalar (fraction of pairs where margin > 0)
    """
    pi_logratio = policy_logp_chosen - policy_logp_rejected
    ref_logratio = ref_logp_chosen - ref_logp_rejected
    logits = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()

    margin = logits.detach()
    accuracy = (margin > 0).float().mean()
    return loss, margin, accuracy


def compute_logp_for_response(
    model: nn.Module,
    tokenizer,
    prompt: str,
    response: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Sum log-probability of `response` tokens given `prompt` prefix.

    Returns scalar tensor: sum over response tokens of log p(t_i | t_<i).
    """
    full_text = prompt + response
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = prompt_ids.shape[1]

    if full_ids.shape[1] <= prompt_len:
        return torch.tensor(0.0, device=device)

    with torch.set_grad_enabled(model.training):
        out = model(full_ids)
        logits = out.logits if hasattr(out, "logits") else out

    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    response_logp = token_logp[:, prompt_len - 1:].sum()
    return response_logp


class DPOTrainer:
    """Minimal DPO trainer — wraps a policy + frozen reference model.

    Usage:
        trainer = DPOTrainer(policy, ref_model, tokenizer, beta=0.1, lr=5e-7)
        for step, batch in enumerate(loader):
            loss, acc = trainer.step(batch)
            if step % 50 == 0:
                trainer.refresh_ref()  # iterative DPO 2404.10719
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_model: Optional[nn.Module],
        tokenizer,
        beta: float = 0.1,
        lr: float = 5e-7,
        weight_decay: float = 0.0,
        device: str = "cuda",
        ref_refresh_every: int = 50,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model if ref_model is not None else copy.deepcopy(policy)
        self.tokenizer = tokenizer
        self.beta = beta
        self.lr = lr
        self.device = device
        self.ref_refresh_every = ref_refresh_every

        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.opt = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
        )
        self._step = 0

    def _logp_pair(self, model: nn.Module, prompt: str, chosen: str, rejected: str):
        logp_c = compute_logp_for_response(model, self.tokenizer, prompt, chosen, self.device)
        logp_r = compute_logp_for_response(model, self.tokenizer, prompt, rejected, self.device)
        return logp_c, logp_r

    def step(self, pairs: List[DPOPair]) -> dict:
        self.policy.train()
        self.opt.zero_grad()

        losses = []
        margins = []
        accs = []
        for pair in pairs:
            pi_c, pi_r = self._logp_pair(self.policy, pair.prompt, pair.chosen, pair.rejected)
            with torch.no_grad():
                ref_c, ref_r = self._logp_pair(self.ref_model, pair.prompt, pair.chosen, pair.rejected)

            loss, margin, acc = dpo_loss(
                pi_c.unsqueeze(0), pi_r.unsqueeze(0),
                ref_c.unsqueeze(0), ref_r.unsqueeze(0),
                beta=self.beta,
            )
            (loss / max(len(pairs), 1)).backward()
            losses.append(loss.item())
            margins.append(margin.mean().item())
            accs.append(acc.item())

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.opt.step()

        self._step += 1
        if self._step % self.ref_refresh_every == 0:
            self.refresh_ref()

        return {
            "loss": float(sum(losses) / max(len(losses), 1)),
            "margin": float(sum(margins) / max(len(margins), 1)),
            "accuracy": float(sum(accs) / max(len(accs), 1)),
            "step": self._step,
        }

    def refresh_ref(self) -> None:
        """Iterative DPO: copy policy → ref. Improves stability over long training."""
        self.ref_model.load_state_dict(self.policy.state_dict())
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()


def harmbench_refusal_eval(
    policy: nn.Module,
    tokenizer,
    eval_pairs: List[DPOPair],
    judge: Callable,
    device: str = "cuda",
    n: int = 100,
) -> dict:
    """Quick eval: how many of N held-out attack prompts does policy refuse?"""
    policy.eval()
    refusal_count = 0
    for pair in eval_pairs[:n]:
        with torch.no_grad():
            ids = tokenizer.encode(pair.prompt, return_tensors="pt").to(device)
            try:
                out_ids = policy.generate(ids, max_new_tokens=100, do_sample=False)
                response = tokenizer.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
            except Exception:
                response = ""
        v = judge(pair.prompt, response, pair.chosen)
        if v[0] == "B":
            refusal_count += 1
    return {
        "refusal_rate": refusal_count / max(n, 1),
        "n": n,
    }
