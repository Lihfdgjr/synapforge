"""
Coconut latent thinking (arxiv 2412.06769).

Original: insert <bot> ... <eot> tokens; between them, the model runs
in "continuous-thought" mode: hidden states feed back as next-step input
without going through the LM head (no token sampling).

Our adaptation:
  - We already have loop_depth=2 RDT recurrent compute in HybridBlock.
  - Coconut adds the OUTSIDE thinking loop: between input prefix and answer,
    the model takes K thinking steps where the next-step input = current
    hidden_state (a continuous vector), not embed(token).
  - This is FREE in our architecture because the CfC recurrent state is
    already a continuous time-series — we just feed it back.

Special tokens (added to Qwen vocab as new ids):
  BOT_ID = vocab_size + 0   (begin-of-thinking)
  EOT_ID = vocab_size + 1   (end-of-thinking)
  PAUSE_ID = vocab_size + 2 (pause-token, Goyal et al 2310.02226)

Training:
  - SFT data may contain <bot> ... <eot> wrappers around the chain-of-thought
    portion of an answer.
  - Loss is masked on tokens INSIDE the thinking region (no LM target).
  - Curriculum: start with K=1 thinking step; gradually increase to K=8.

Inference:
  - LatentThinker.think(prefix_hidden, k=8) returns the post-thinking hidden state.
  - The model then samples the answer normally.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LatentThinker(nn.Module):
    """Wraps a HybridBlock stack to perform K thinking steps."""

    def __init__(
        self,
        hidden: int,
        thinking_proj: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        if thinking_proj:
            self.thinking_in = nn.Linear(hidden, hidden, bias=False)
            self.thinking_out = nn.Linear(hidden, hidden, bias=False)
            nn.init.eye_(self.thinking_in.weight)
            nn.init.eye_(self.thinking_out.weight)
        else:
            self.thinking_in = nn.Identity()
            self.thinking_out = nn.Identity()

    def think_step(
        self,
        block_stack,
        h_continuous: torch.Tensor,
        h_prev_state: Optional[torch.Tensor] = None,
    ) -> tuple:
        """One thinking step: h_continuous -> next h_continuous.

        block_stack: a callable that takes (x, h_prev) and returns (out, new_state)
        h_continuous: (B, hidden) — current thought
        h_prev_state: optional CfC state to carry across thinking steps
        """
        x = self.thinking_in(h_continuous).unsqueeze(1)
        out, new_state = block_stack(x, h_prev_state)
        h_new = self.thinking_out(out.squeeze(1))
        return h_new, new_state

    def think(
        self,
        block_stack,
        prefix_hidden: torch.Tensor,
        k: int = 4,
        prev_state: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Run K thinking steps starting from prefix_hidden."""
        h = prefix_hidden
        state = prev_state
        for _ in range(k):
            h, state = self.think_step(block_stack, h, state)
        return h, state


def add_thinking_tokens(tokenizer, names: tuple = ("<bot>", "<eot>", "<pause>")) -> dict:
    """Add special thinking tokens to a HF tokenizer. Returns id mapping."""
    added = tokenizer.add_special_tokens({"additional_special_tokens": list(names)})
    ids = {n: tokenizer.convert_tokens_to_ids(n) for n in names}
    return {"added": added, "ids": ids}


def build_thinking_mask(
    input_ids: torch.Tensor,
    bot_id: int,
    eot_id: int,
) -> torch.Tensor:
    """Boolean mask, True for positions INSIDE <bot>...<eot> regions.

    Used to suppress LM loss on thinking tokens.
    """
    B, T = input_ids.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
    inside = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
    for t in range(T):
        col = input_ids[:, t]
        starts = (col == bot_id) & ~inside
        ends = (col == eot_id) & inside
        inside = inside | starts
        mask[:, t] = inside
        inside = inside & ~ends
    return mask


class CurriculumScheduler:
    """Increase k_thinking gradually during training.

    Recipe from Coconut paper:
      step 0..N1: k = 1
      step N1..N2: k = 2
      ...
    """

    def __init__(
        self,
        schedule: list[tuple[int, int]] = (
            (0, 1),
            (5_000, 2),
            (15_000, 4),
            (30_000, 6),
            (50_000, 8),
        ),
    ) -> None:
        self.schedule = sorted(schedule)

    def k_at(self, step: int) -> int:
        k = 1
        for boundary, k_val in self.schedule:
            if step >= boundary:
                k = k_val
        return k


def adaptive_k(
    context_len: int,
    retrieval_confidence: float,
    alpha: float = 1.0,
    beta: float = 4.0,
    k_min: int = 1,
    k_max: int = 8,
) -> int:
    """Confidence-scaled Coconut thinking depth.

    More context + lower retrieval confidence → deeper thinking.
    Less context or strong retrieval → shallow thinking, save compute.

    Formula (per agent synthesis 2026-05-01):
        k = clip(round(α·log₂(ctx_len) + β·(1 − conf)), k_min, k_max)

    Examples:
        ctx=1K,   conf=0.9 → α·10 + β·0.1 = 10.4  → clip 8
        ctx=1K,   conf=0.3 → α·10 + β·0.7 = 12.8  → clip 8
        ctx=100K, conf=0.9 → α·17 + β·0.1 = 17.4  → clip 8
        ctx=100,  conf=0.9 → α·6.6 + β·0.1 = 7.0  → 7
        ctx=100,  conf=0.7 → α·6.6 + β·0.3 = 7.8  → 8
        ctx=8,    conf=0.95 → α·3 + β·0.05 = 3.2  → 3

    Default α=1, β=4 is biased toward "think deep when retrieval missed."
    For paper-level claim of monotonic context scaling:
        - Short ctx: confidence high (model knows few facts well) → k=1-2
        - Long ctx: more retrieval misses possible → k=8 deep think
    """
    import math

    score = alpha * math.log2(max(context_len, 2)) + beta * (1.0 - retrieval_confidence)
    k = int(round(score))
    return max(k_min, min(k_max, k))


class PauseTokenInjector:
    """Goyal et al 2310.02226: inject <pause> tokens before answer.

    Each <pause> = an extra forward step where the model can compute
    without committing to an output. Cheap form of test-time compute.
    """

    def __init__(self, pause_id: int, n_pauses: int = 4) -> None:
        self.pause_id = pause_id
        self.n_pauses = n_pauses

    def inject(self, input_ids: torch.Tensor, answer_start: int) -> torch.Tensor:
        """Insert n_pauses pause tokens at position answer_start."""
        B = input_ids.shape[0]
        prefix = input_ids[:, :answer_start]
        suffix = input_ids[:, answer_start:]
        pauses = torch.full(
            (B, self.n_pauses), self.pause_id, dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat([prefix, pauses, suffix], dim=-1)
