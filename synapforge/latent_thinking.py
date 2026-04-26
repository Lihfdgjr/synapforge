"""Latent-space thinking — Coconut / Quiet-STaR / Pause / Ouro-LoopLM style.

The model runs a *silent* loop in hidden-state space before emitting
the next token.  No CE loss is computed at thinking positions.  This
gives the network a variable-step refinement budget without forcing
the thought into language.

Components
----------
1. ``ThinkingTokens``        — registers ``<bot>`` / ``<eot>``.
2. ``ThinkingActionTokens``  — superset that also registers
                               ``<boa>`` / ``<eoa>`` for native action
                               channels (mscfc.action_head pattern).
3. ``LatentLoopController``  — runs the silent loop on positions
                               flagged by ``thinking_mask``.
4. ``LatentConsistencyLoss`` — Quiet-STaR predictive-coding regulariser.
5. ``LatentSearchBeam``      — test-time beam search in latent space.

References
----------
- Coconut (Chain of Continuous Thought): arxiv:2412.06769
- Quiet-STaR:                            arxiv:2403.09629
- Pause Token:                           arxiv:2310.02226
- Ouro LoopLM:                           arxiv:2510.25741

No imports from other ``synapforge`` modules besides ``Module``;
``torch.jit.script`` is explicitly NOT used.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .module import Module

# ----------------------------------------------------------------------------
# 1. ThinkingTokens
# ----------------------------------------------------------------------------


class ThinkingTokens(Module):
    """Learnable ``<bot>`` / ``<eot>`` thinking tokens."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        bot_str: str = "<bot>",
        eot_str: str = "<eot>",
        init_scale: float = 0.02,
    ):
        super().__init__()
        if vocab_size <= 0 or hidden <= 0:
            raise ValueError("vocab_size and hidden must be positive")
        self.vocab_size = int(vocab_size)
        self.hidden = int(hidden)
        self.bot_str = str(bot_str)
        self.eot_str = str(eot_str)
        self.bot_id = self.vocab_size
        self.eot_id = self.vocab_size + 1
        self.extended_vocab = self.vocab_size + 2
        self.bot_embed = nn.Parameter(torch.randn(hidden) * init_scale)
        self.eot_embed = nn.Parameter(torch.randn(hidden) * init_scale)

    def embed(self, token_id: int) -> torch.Tensor:
        if token_id == self.bot_id:
            return self.bot_embed
        if token_id == self.eot_id:
            return self.eot_embed
        raise ValueError(
            f"token_id {token_id} is not a thinking token "
            f"(bot={self.bot_id}, eot={self.eot_id})"
        )

    def is_thinking(self, token_id):
        if isinstance(token_id, torch.Tensor):
            return (token_id == self.bot_id) | (token_id == self.eot_id)
        return int(token_id) in (self.bot_id, self.eot_id)

    def extend_tokenizer(self, tokenizer) -> tuple[int, int]:
        """Best-effort: add the strings to a HF-style tokenizer."""
        try:
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.bot_str, self.eot_str]}
            )
            if added > 0:
                bot = tokenizer.convert_tokens_to_ids(self.bot_str)
                eot = tokenizer.convert_tokens_to_ids(self.eot_str)
                if isinstance(bot, int) and isinstance(eot, int):
                    self.bot_id = bot
                    self.eot_id = eot
                    self.extended_vocab = max(bot, eot) + 1
        except Exception:
            pass
        return self.bot_id, self.eot_id


# ----------------------------------------------------------------------------
# 2. ThinkingActionTokens — extended with <boa>/<eoa>
# ----------------------------------------------------------------------------


class ThinkingActionTokens(Module):
    """Unified registry of <bot>/<eot>/<boa>/<eoa>."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        bot_str: str = "<bot>",
        eot_str: str = "<eot>",
        boa_str: str = "<boa>",
        eoa_str: str = "<eoa>",
        init_scale: float = 0.02,
    ):
        super().__init__()
        if vocab_size <= 0 or hidden <= 0:
            raise ValueError("vocab_size and hidden must be positive")
        self.vocab_size = int(vocab_size)
        self.hidden = int(hidden)
        self.bot_id = self.vocab_size
        self.eot_id = self.vocab_size + 1
        self.boa_id = self.vocab_size + 2
        self.eoa_id = self.vocab_size + 3
        self.extended_vocab = self.vocab_size + 4
        self.bot_str = str(bot_str)
        self.eot_str = str(eot_str)
        self.boa_str = str(boa_str)
        self.eoa_str = str(eoa_str)
        self.bot_embed = nn.Parameter(torch.randn(hidden) * init_scale)
        self.eot_embed = nn.Parameter(torch.randn(hidden) * init_scale)
        self.boa_embed = nn.Parameter(torch.randn(hidden) * init_scale)
        self.eoa_embed = nn.Parameter(torch.randn(hidden) * init_scale)

    def embed(self, token_id: int) -> torch.Tensor:
        if token_id == self.bot_id:
            return self.bot_embed
        if token_id == self.eot_id:
            return self.eot_embed
        if token_id == self.boa_id:
            return self.boa_embed
        if token_id == self.eoa_id:
            return self.eoa_embed
        raise ValueError(
            f"token_id {token_id} is not a thinking/action token "
            f"(bot={self.bot_id}, eot={self.eot_id}, boa={self.boa_id}, eoa={self.eoa_id})"
        )

    def is_thinking(self, token_id):
        if isinstance(token_id, torch.Tensor):
            return (token_id == self.bot_id) | (token_id == self.eot_id)
        return int(token_id) in (self.bot_id, self.eot_id)

    def is_action(self, token_id):
        if isinstance(token_id, torch.Tensor):
            return (token_id == self.boa_id) | (token_id == self.eoa_id)
        return int(token_id) in (self.boa_id, self.eoa_id)

    def extend_tokenizer(self, tokenizer):
        try:
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": [
                    self.bot_str, self.eot_str, self.boa_str, self.eoa_str
                ]}
            )
            if added > 0:
                ids = {
                    "bot_id": tokenizer.convert_tokens_to_ids(self.bot_str),
                    "eot_id": tokenizer.convert_tokens_to_ids(self.eot_str),
                    "boa_id": tokenizer.convert_tokens_to_ids(self.boa_str),
                    "eoa_id": tokenizer.convert_tokens_to_ids(self.eoa_str),
                }
                if all(isinstance(v, int) for v in ids.values()):
                    for k, v in ids.items():
                        setattr(self, k, v)
                    self.extended_vocab = max(ids.values()) + 1
        except Exception:
            pass
        return self.bot_id, self.eot_id, self.boa_id, self.eoa_id


# ----------------------------------------------------------------------------
# 3. LatentLoopController
# ----------------------------------------------------------------------------


class LatentLoopController(Module):
    """Run a silent latent loop at every ``thinking_mask`` position.

    Args:
        hidden:           feature dimension.
        block:            a callable / nn.Module that maps [B, T, D] -> [B, T, D].
        max_think_steps:  default number of refinement passes.
        mix_gate_init:    initial value of the sigmoid mix gate.

    Forward:
        hidden_sequence: [B, T, D]
        thinking_mask:   [B, T] bool (True = silent thinking slot)
    Returns:
        extended_hidden:    [B, T, D] with thinking positions refined
        extended_loss_mask: [B, T] bool — True where CE loss applies
    """

    def __init__(
        self,
        hidden: int,
        block: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        max_think_steps: int = 16,
        mix_gate_init: float = 0.5,
    ):
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = int(hidden)
        self.block = block
        self.max_think_steps = int(max_think_steps)
        self.mix_gate = nn.Parameter(torch.tensor(float(mix_gate_init)))
        # Make sure nn.Module-typed blocks are properly registered.
        if isinstance(block, nn.Module):
            self.add_module("inner_block", block)

    def _apply_block(self, h: torch.Tensor) -> torch.Tensor:
        out = self.block(h)
        if isinstance(out, tuple):
            out = out[0]
        if out.shape != h.shape:
            raise RuntimeError(
                f"block changed shape: {tuple(h.shape)} -> {tuple(out.shape)}"
            )
        return out

    def forward(
        self,
        hidden_sequence: torch.Tensor,
        thinking_mask: torch.Tensor,
        max_think_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_sequence.dim() != 3:
            raise ValueError(
                f"hidden_sequence must be [B, T, D], got {tuple(hidden_sequence.shape)}"
            )
        if thinking_mask.shape != hidden_sequence.shape[:2]:
            raise ValueError(
                f"thinking_mask shape {tuple(thinking_mask.shape)} "
                f"must match hidden[:2] {tuple(hidden_sequence.shape[:2])}"
            )
        steps = self.max_think_steps if max_think_steps is None else int(max_think_steps)
        if steps <= 0:
            return hidden_sequence, ~thinking_mask.bool()

        h = hidden_sequence
        B, T, D = h.shape
        tm_bool = thinking_mask.bool().to(h.device)
        mask_f = tm_bool.unsqueeze(-1).to(h.dtype)
        if mask_f.sum() == 0:
            return h, ~tm_bool

        refined = h
        gate = torch.sigmoid(self.mix_gate)
        for _ in range(steps):
            delta = self._apply_block(refined) - refined
            refined = refined + gate * mask_f * delta

        # "Think-then-speak": inject the last refined thought into the
        # first non-thinking position via a right-shift of the mask.
        speak_mask = torch.zeros_like(tm_bool)
        if T > 1:
            speak_mask[:, 1:] = tm_bool[:, :-1] & ~tm_bool[:, 1:]
        shifted = torch.zeros_like(refined)
        if T > 1:
            shifted[:, 1:] = refined[:, :-1]
        speak_f = speak_mask.unsqueeze(-1).to(h.dtype)
        extended_hidden = refined + gate * speak_f * (shifted - refined)
        extended_loss_mask = ~tm_bool
        return extended_hidden, extended_loss_mask


# ----------------------------------------------------------------------------
# 4. LatentConsistencyLoss
# ----------------------------------------------------------------------------


class LatentConsistencyLoss(Module):
    """Predictive-coding regulariser between pre-think and post-think states.

    Gaussian NLL with learnable log-sigma:

        loss = 0.5 * mse / sigma^2 + log sigma
    """

    def __init__(self, hidden: int, reduction: str = "mean"):
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction {reduction}")
        self.hidden = int(hidden)
        self.reduction = reduction
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(
        self, h_before: torch.Tensor, h_after: torch.Tensor
    ) -> torch.Tensor:
        if h_before.shape != h_after.shape:
            raise ValueError(
                f"shape mismatch: {tuple(h_before.shape)} vs {tuple(h_after.shape)}"
            )
        diff = (h_after - h_before).pow(2)
        if self.reduction == "mean":
            mse = diff.mean()
        elif self.reduction == "sum":
            mse = diff.sum()
        else:
            return diff
        sigma2 = (2.0 * self.log_sigma).exp().clamp(min=1e-6)
        return (0.5 * mse / sigma2 + self.log_sigma.squeeze()).squeeze()


# ----------------------------------------------------------------------------
# 5. LatentSearchBeam
# ----------------------------------------------------------------------------


class LatentSearchBeam:
    """Beam search directly in latent hidden-state space.

    Not an ``nn.Module`` — pure inference utility.

    Args:
        block:     callable advancing one step in latent space.
        logits_fn: optional final scoring head.
        beam_size: number of candidate hiddens to keep.
        score_fn:  optional override for in-loop scoring.
    """

    def __init__(
        self,
        block: Callable[[torch.Tensor], torch.Tensor],
        logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        beam_size: int = 4,
        score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        self.block = block
        self.logits_fn = logits_fn
        self.beam_size = int(beam_size)
        self.score_fn = score_fn
        self._anchor: torch.Tensor | None = None

    def init(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 1:
            raise ValueError(f"init hidden must be 1-D, got {tuple(h.shape)}")
        self._anchor = h.detach().clone()
        noise = torch.randn(self.beam_size, h.size(0), device=h.device, dtype=h.dtype)
        noise[0].zero_()
        return h.unsqueeze(0) + 0.01 * noise

    def _score(self, h_cand: torch.Tensor) -> torch.Tensor:
        if self.score_fn is not None:
            return self.score_fn(h_cand)
        if self._anchor is None:
            return h_cand.norm(dim=-1)
        return h_cand @ self._anchor

    def step(
        self,
        h_candidates: torch.Tensor,
        input_token_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h_candidates.dim() != 2:
            raise ValueError(
                f"h_candidates must be [k, D], got {tuple(h_candidates.shape)}"
            )
        x = h_candidates
        if input_token_embed is not None:
            if input_token_embed.dim() == 1:
                input_token_embed = input_token_embed.unsqueeze(0).expand_as(x)
            x = x + input_token_embed
        advanced = self.block(x.unsqueeze(1))
        if isinstance(advanced, tuple):
            advanced = advanced[0]
        advanced = advanced.squeeze(1)
        scores = self._score(advanced)
        k = min(self.beam_size, advanced.size(0))
        top = scores.topk(k)
        selected = advanced[top.indices]
        if k < self.beam_size:
            pad = selected[-1:].repeat(self.beam_size - k, 1)
            selected = torch.cat([selected, pad], dim=0)
        return selected

    def finalise(self, h_candidates: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self.logits_fn is not None:
            logits = self.logits_fn(h_candidates)
            score = logits.max(dim=-1).values
        else:
            score = self._score(h_candidates)
        best = int(score.argmax().item())
        return h_candidates[best], best


__all__ = [
    "ThinkingTokens",
    "ThinkingActionTokens",
    "LatentLoopController",
    "LatentConsistencyLoss",
    "LatentSearchBeam",
]
