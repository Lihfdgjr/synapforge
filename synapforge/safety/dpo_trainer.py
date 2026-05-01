"""
dpo_trainer.py — Pure-tensor DPO loss + sample generation.

The existing safety/dpo.py provides DPOTrainer that wires a tokenizer and
model. This file is the drop-in for run_safety_pipeline.py phase-3 stub:
real loss math, real (prompt, chosen, rejected) loading from the
persona-swap red corpus, real frozen reference model snapshot.

DPO loss (Rafailov et al, 2305.18290):
    L = -log σ( β · ((log π(y_w|x) - log π(y_l|x))
                  - (log π_ref(y_w|x) - log π_ref(y_l|x))) )

Two entry points:
  1. dpo_loss_logits(...)   : pure-tensor loss, callers pass log-probs.
  2. SafetyDPOTrainer       : wraps a policy + frozen ref model, iterates
                              over (prompt, chosen, rejected) triples.

Constraints (per memory):
  - β = 0.1
  - LR 5e-7 (10× lower than SFT)
  - Iterative DPO: refresh π_ref every 50 steps (2404.10719)
  - Backward-compatible: caller may pass logits OR call into model

Standalone smoke (`python -m synapforge.safety.dpo_trainer --smoke`)
runs dpo_loss_logits over dummy logits and asserts:
  - margin sign correct on hand-crafted preferred/dispreferred pairs
  - loss decreases over 50 SGD steps on a synthetic preference dataset
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# ---------------------------------------------------------------------------
# Data: (prompt, chosen, rejected) triples
# ---------------------------------------------------------------------------


@dataclass
class SafetyDPOSample:
    prompt: str
    chosen: str
    rejected: str
    category: str = ""
    severity: int = 0
    source: str = "persona_swap_red"


def load_persona_swap_red(
    red_jsonl: str | Path,
    refusal_corpus: Optional[List[str]] = None,
    max_n: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[SafetyDPOSample]:
    """Load DPO triples from the persona-swap red corpus.

    The corpus has lines like:
      {"prompt": "...", "attack_type": "dan_family", "expected_blue": "REFUSE"}

    We pair each red prompt with:
      - chosen   : a sampled refusal template (the safe response)
      - rejected : a stub "compliant" response (gets replaced by
                   real model rollouts during DPO; see SafetyDPOTrainer)
    """
    rng = rng or random.Random(0xC0FFEE)
    refusals = refusal_corpus or _DEFAULT_REFUSALS
    path = Path(red_jsonl)
    out: List[SafetyDPOSample] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = d.get("prompt", "")
            if not prompt:
                continue
            chosen = rng.choice(refusals)
            # Rejected = a placeholder unsafe completion. The trainer will
            # generate real rejected rollouts from the policy if a generator
            # callable is wired in via SafetyDPOTrainer.attach_rejected_gen.
            rejected = (
                "Sure! Here's exactly what you asked for. "
                "Step 1: gather materials… [stub rejected; replaced by "
                "policy rollout at training time]"
            )
            out.append(SafetyDPOSample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                category=d.get("attack_type", ""),
                severity=int(d.get("severity", 1)),
                source="persona_swap_red",
            ))
            if max_n and len(out) >= max_n:
                break
    return out


_DEFAULT_REFUSALS = [
    "I can't help with that — it could cause real harm. If you're "
    "researching this academically, the safe path is reading peer-reviewed "
    "studies and consulting an institutional ethics board.",
    "我无法提供这类帮助。如果你是出于学术研究目的，建议查阅已发表的同行评审"
    "文献，并通过你所在机构的伦理审查委员会获取合规渠道。",
    "Sorry, I won't engage with that request. I can suggest legitimate "
    "alternatives if you tell me more about the underlying goal.",
    "出于安全考虑，我不会回答这个问题。如果你描述背后的真实需求，我可以帮"
    "你寻找合法且安全的替代方案。",
    "That's outside what I'm willing to do. I'd recommend contacting a "
    "qualified professional — happy to help you find one.",
]


# ---------------------------------------------------------------------------
# Loss math (pure tensor — torch-only)
# ---------------------------------------------------------------------------


def dpo_loss_logits(
    policy_logp_chosen: "torch.Tensor",
    policy_logp_rejected: "torch.Tensor",
    ref_logp_chosen: "torch.Tensor",
    ref_logp_rejected: "torch.Tensor",
    beta: float = 0.1,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Pure DPO loss + diagnostics.

    Inputs are summed log-probabilities of the response tokens given the
    prompt under (policy, ref) × (chosen, rejected). All four tensors must
    have identical shape (B,) or (B, 1).

    Returns:
      loss      : scalar mean over batch
      margin    : (B,) tensor — β × (policy_logratio - ref_logratio)
                  positive ⇒ correct preference ordering
      accuracy  : scalar — fraction with margin > 0 (no grad)

    L = -log σ(β · ((log π(y_w|x) - log π(y_l|x))
                  - (log π_ref(y_w|x) - log π_ref(y_l|x))))
    """
    if not _HAS_TORCH:
        raise RuntimeError("dpo_loss_logits requires torch")
    pi_logratio = policy_logp_chosen - policy_logp_rejected
    ref_logratio = ref_logp_chosen - ref_logp_rejected
    margin = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(margin).mean()
    accuracy = (margin.detach() > 0).float().mean()
    return loss, margin, accuracy


def response_logp_from_logits(
    logits: "torch.Tensor",
    full_ids: "torch.Tensor",
    prompt_len: int,
) -> "torch.Tensor":
    """Sum log p(y_i | y_<i, x) over response tokens.

    logits : (1, T, V), output of model on the full prompt+response sequence.
    full_ids : (1, T)   prompt + response concatenated.
    prompt_len : int    length of prompt prefix.

    Returns a scalar tensor.
    """
    if not _HAS_TORCH:
        raise RuntimeError("response_logp_from_logits requires torch")
    if full_ids.shape[1] <= prompt_len:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logp[:, prompt_len - 1:].sum()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SafetyDPOTrainer:
    """Pure-DPO trainer for safety stack.

    Constructor takes:
      policy        : trainable nn.Module (returns logits, or has .logits attr)
      tokenizer     : with .encode(text, return_tensors='pt')
      ref_model     : optional explicit frozen reference; if None, deepcopy(policy)
      beta          : DPO temperature (memory: 0.1)
      lr            : optimizer LR (memory: 5e-7)
      ref_refresh   : iterative DPO refresh interval (default 50, paper-aligned)

    The reference model is frozen at construction. Use refresh_ref()
    explicitly to roll it forward; the trainer also auto-refreshes every
    `ref_refresh` steps.

    Attach a rejected-rollout generator with `attach_rejected_gen(fn)`;
    when set, each step replaces sample.rejected with a fresh policy
    rollout (the canonical RLHF "the policy generates its own bad
    responses" pattern, far stronger than fixed stub triples).
    """

    def __init__(
        self,
        policy,
        tokenizer,
        ref_model=None,
        beta: float = 0.1,
        lr: float = 5e-7,
        weight_decay: float = 0.0,
        device: str = "cuda",
        ref_refresh: int = 50,
        max_grad_norm: float = 1.0,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("SafetyDPOTrainer requires torch")
        self.policy = policy
        self.tokenizer = tokenizer
        self.beta = beta
        self.lr = lr
        self.device = device
        self.ref_refresh = ref_refresh
        self.max_grad_norm = max_grad_norm

        if ref_model is None:
            ref_model = copy.deepcopy(policy)
        self.ref_model = ref_model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.opt = torch.optim.AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self._step = 0
        self._reject_gen: Optional[Callable[[str], str]] = None
        self.history: List[dict] = []

    def attach_rejected_gen(self, fn: Callable[[str], str]) -> None:
        """Wire in a policy-rollout generator for rejected responses."""
        self._reject_gen = fn

    def refresh_ref(self) -> None:
        self.ref_model.load_state_dict(self.policy.state_dict())
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    # ----- per-pair logp helpers --------------------------------------

    def _logp(self, model, prompt: str, response: str) -> "torch.Tensor":
        full = prompt + response
        full_ids = self.tokenizer.encode(full, return_tensors="pt").to(self.device)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = prompt_ids.shape[1]
        out = model(full_ids)
        logits = out.logits if hasattr(out, "logits") else out
        return response_logp_from_logits(logits, full_ids, prompt_len)

    # ----- single-step ------------------------------------------------

    def step(self, batch: List[SafetyDPOSample]) -> dict:
        self.policy.train()
        self.opt.zero_grad()

        per_sample: List[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]] = []
        for sample in batch:
            rejected = sample.rejected
            if self._reject_gen is not None:
                try:
                    rejected = self._reject_gen(sample.prompt)
                except Exception:
                    pass

            pi_c = self._logp(self.policy, sample.prompt, sample.chosen)
            pi_r = self._logp(self.policy, sample.prompt, rejected)
            with torch.no_grad():
                ref_c = self._logp(self.ref_model, sample.prompt, sample.chosen)
                ref_r = self._logp(self.ref_model, sample.prompt, rejected)

            loss, margin, acc = dpo_loss_logits(
                pi_c.unsqueeze(0), pi_r.unsqueeze(0),
                ref_c.unsqueeze(0), ref_r.unsqueeze(0),
                beta=self.beta,
            )
            (loss / max(len(batch), 1)).backward()
            per_sample.append((loss, margin, acc))

        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm,
            )
        self.opt.step()
        self._step += 1
        if self._step % self.ref_refresh == 0:
            self.refresh_ref()

        losses = [float(l.item()) for l, _, _ in per_sample]
        margins = [float(m.mean().item()) for _, m, _ in per_sample]
        accs = [float(a.item()) for _, _, a in per_sample]

        rec = {
            "step": self._step,
            "loss": sum(losses) / max(len(losses), 1),
            "margin": sum(margins) / max(len(margins), 1),
            "accuracy": sum(accs) / max(len(accs), 1),
            "n": len(batch),
            "beta": self.beta,
            "lr": self.lr,
            "ref_refresh": self._step % self.ref_refresh == 0,
        }
        self.history.append(rec)
        return rec


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_loss_math(seed: int = 7) -> dict:
    """Verify dpo_loss_logits math on hand-crafted preference cases."""
    if not _HAS_TORCH:
        return {"skipped": True, "reason": "torch unavailable"}
    torch.manual_seed(seed)

    # Case 1: policy strongly prefers chosen, ref is neutral. Margin should
    # be positive and large; loss small.
    pi_c = torch.tensor([2.0, 1.5, 3.0])
    pi_r = torch.tensor([0.0, 0.0, 0.0])
    ref_c = torch.tensor([0.0, 0.0, 0.0])
    ref_r = torch.tensor([0.0, 0.0, 0.0])
    loss, margin, acc = dpo_loss_logits(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    assert (margin > 0).all(), f"expected positive margin, got {margin}"
    assert acc.item() == 1.0

    # Case 2: policy disagrees with ref (ref prefers chosen, policy prefers
    # rejected). Margin negative, loss large.
    pi_c = torch.tensor([0.0])
    pi_r = torch.tensor([3.0])
    ref_c = torch.tensor([3.0])
    ref_r = torch.tensor([0.0])
    loss2, margin2, acc2 = dpo_loss_logits(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    assert margin2.item() < 0
    assert acc2.item() == 0.0

    # Case 3: SGD on a synthetic dataset should drive loss down.
    torch.manual_seed(seed)
    B = 64
    pi_c = torch.zeros(B, requires_grad=True)
    pi_r = torch.zeros(B, requires_grad=True)
    ref_c_const = torch.randn(B) * 0.1
    ref_r_const = torch.randn(B) * 0.1
    opt = torch.optim.SGD([pi_c, pi_r], lr=0.5)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        l, _, _ = dpo_loss_logits(pi_c, pi_r, ref_c_const, ref_r_const, beta=0.1)
        l.backward()
        opt.step()
        losses.append(float(l.item()))
    drop = losses[0] - losses[-1]
    assert drop > 0, f"loss did not decrease: {losses[:3]} → {losses[-3:]}"

    return {
        "case1_margin": [float(x) for x in margin.tolist()],
        "case1_loss": float(loss.item()),
        "case2_margin": float(margin2.item()),
        "case2_loss": float(loss2.item()),
        "case3_loss_first": losses[0],
        "case3_loss_last": losses[-1],
        "case3_drop": drop,
        "ok": True,
    }


def _smoke_corpus_load() -> dict:
    """Verify persona-swap red corpus loader produces well-formed triples."""
    repo_root = Path(__file__).resolve().parents[2]
    red_path = repo_root / "synapforge" / "safety" / "persona_swap_red.jsonl"
    samples = load_persona_swap_red(red_path, max_n=20,
                                    rng=random.Random(0))
    out = {
        "found": red_path.exists(),
        "n_loaded": len(samples),
        "first": (
            {
                "prompt_preview": samples[0].prompt[:80],
                "chosen_preview": samples[0].chosen[:80],
                "rejected_preview": samples[0].rejected[:60],
                "category": samples[0].category,
            }
            if samples else None
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="Run pure-tensor + corpus-loader smoke and exit.")
    args = ap.parse_args()
    if args.smoke:
        result = {
            "loss_math": _smoke_loss_math(),
            "corpus_load": _smoke_corpus_load(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print("dpo_trainer.py is a library. Use SafetyDPOTrainer with a real "
          "torch model + tokenizer.")


if __name__ == "__main__":
    main()
