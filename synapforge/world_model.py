"""WorldModel — predict next hidden state, MSE / NLL / reward / terminal.

Ha & Schmidhuber's "World Models" (1803.10122) and Hafner et al.'s
Dreamer V3 (2301.04104) make the case that pre-training a *next-state*
predictor (rather than a next-token predictor) gives the model a
latent-space simulator it can dream in.  V-JEPA-2-AC (2506.09985)
generalises this to "predict the next *latent*, not the next pixel".

This module ships:

  - ``WorldModelHead``        — predict next hidden + reward + terminal
                                + aleatoric variance.
  - ``WorldModelLoss``        — MSE + Gaussian NLL + reward BCE + done BCE.
  - ``HypothesisGenerator``   — propose ``N`` candidate next-hiddens.
  - ``WorldModelCritic``      — score candidates (FunSearch / Co-Scientist
                                propose-evaluate-evolve contract).

Default-disabled: byte-identical to current behaviour when ``WorldModelLoss``
is not added to the training objective.  Pure ``torch.nn``; no torch.jit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Module

# ----------------------------------------------------------------------------
# 1. WorldModelHead
# ----------------------------------------------------------------------------


@dataclass
class WorldModelOutput:
    """Structured output of :class:`WorldModelHead`."""

    next_hidden: torch.Tensor    # [B, T, D]
    reward: torch.Tensor         # [B, T, 1] in [0, 1]
    terminal: torch.Tensor       # [B, T, 1] in [0, 1]
    aleatoric_var: torch.Tensor  # [B, T, D] strictly positive


class WorldModelHead(Module):
    """Predict the next hidden state (not the next token).

    Architecture::

        h --LN--> MLP(D -> H -> 3*D + 2) --split--> (next_raw, log_var, r, done)
        next_hidden    = h + next_raw                # residual skip
        aleatoric_var  = softplus(log_var) + var_eps
        reward         = sigmoid(r)
        terminal       = sigmoid(done)

    The output linear is zero-initialised so training starts at
    "next ~= cur, r ~= 0.5, done ~= 0.5, var ~= softplus(0)" (matches
    Dreamer V3's stochastic residual init).
    """

    def __init__(
        self,
        hidden: int,
        mlp_hidden: int | None = None,
        dropout: float = 0.0,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = int(hidden)
        mlp_hidden = int(mlp_hidden) if mlp_hidden is not None else self.hidden * 2
        self.mlp_hidden = mlp_hidden
        self.var_eps = float(var_eps)
        self.out_dim = 2 * self.hidden + 2  # [next_raw | log_var | r | done]

        self.ln = nn.LayerNorm(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(mlp_hidden, self.out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor) -> WorldModelOutput:
        """h: ``[B, T, D]`` or ``[..., D]`` -> :class:`WorldModelOutput`."""
        if h.dim() < 2:
            raise ValueError(f"h must have >= 2 dims, got {tuple(h.shape)}")
        if h.shape[-1] != self.hidden:
            raise ValueError(
                f"last dim mismatch: expected {self.hidden}, got {h.shape[-1]}"
            )
        x = self.ln(h)
        flat = self.out(self.trunk(x))
        off = 0
        next_raw = flat[..., off : off + self.hidden]
        off += self.hidden
        log_var = flat[..., off : off + self.hidden]
        off += self.hidden
        r_raw = flat[..., off : off + 1]
        off += 1
        done_raw = flat[..., off : off + 1]
        off += 1
        assert off == self.out_dim

        next_hidden = h + next_raw
        aleatoric_var = F.softplus(log_var) + self.var_eps
        reward = torch.sigmoid(r_raw)
        terminal = torch.sigmoid(done_raw)
        return WorldModelOutput(
            next_hidden=next_hidden,
            reward=reward,
            terminal=terminal,
            aleatoric_var=aleatoric_var,
        )


# ----------------------------------------------------------------------------
# 2. WorldModelLoss
# ----------------------------------------------------------------------------


class WorldModelLoss(Module):
    """Composite world-model loss: MSE + Gaussian NLL + reward BCE + done BCE."""

    def __init__(
        self,
        weight_mse: float = 1.0,
        weight_nll: float = 0.5,
        weight_reward: float = 1.0,
        weight_terminal: float = 0.5,
    ):
        super().__init__()
        self.w_mse = float(weight_mse)
        self.w_nll = float(weight_nll)
        self.w_reward = float(weight_reward)
        self.w_terminal = float(weight_terminal)

    @staticmethod
    def _gaussian_nll(
        pred: torch.Tensor, var: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """0.5 * (((x-mu)^2)/sigma^2 + log sigma^2); mean-reduced."""
        var = var.clamp(min=1e-6)
        sq = (target - pred).pow(2)
        return 0.5 * (sq / var + var.log()).mean()

    def forward(
        self,
        out: WorldModelOutput,
        next_hidden_true: torch.Tensor,
        reward_signal: torch.Tensor | None = None,
        terminal_signal: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if out.next_hidden.shape != next_hidden_true.shape:
            raise ValueError(
                f"shape mismatch: pred={tuple(out.next_hidden.shape)} "
                f"vs target={tuple(next_hidden_true.shape)}"
            )
        device = out.next_hidden.device
        dtype = out.next_hidden.dtype

        def _zero() -> torch.Tensor:
            return torch.zeros((), device=device, dtype=dtype)

        # Detach target so gradients never flow through the next-state oracle.
        target = next_hidden_true.detach()

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.shape != out.next_hidden.shape[:2]:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match "
                    f"hidden[:2] {tuple(out.next_hidden.shape[:2])}"
                )
            if mask.any():
                pred = out.next_hidden[mask]
                var = out.aleatoric_var[mask]
                tgt = target[mask]
                mse = F.mse_loss(pred, tgt)
                nll = self._gaussian_nll(pred, var, tgt)
            else:
                mse = _zero()
                nll = _zero()
        else:
            mse = F.mse_loss(out.next_hidden, target)
            nll = self._gaussian_nll(out.next_hidden, out.aleatoric_var, target)

        if reward_signal is not None:
            if reward_signal.shape != out.reward.shape:
                raise ValueError(
                    f"reward shape {tuple(reward_signal.shape)} must match "
                    f"pred {tuple(out.reward.shape)}"
                )
            r_pred = out.reward.clamp(1e-7, 1 - 1e-7)
            reward_bce = F.binary_cross_entropy(r_pred, reward_signal.clamp(0.0, 1.0))
        else:
            reward_bce = _zero()

        if terminal_signal is not None:
            if terminal_signal.shape != out.terminal.shape:
                raise ValueError(
                    f"terminal shape {tuple(terminal_signal.shape)} must match "
                    f"pred {tuple(out.terminal.shape)}"
                )
            t_pred = out.terminal.clamp(1e-7, 1 - 1e-7)
            done_bce = F.binary_cross_entropy(t_pred, terminal_signal.clamp(0.0, 1.0))
        else:
            done_bce = _zero()

        total = (
            self.w_mse * mse
            + self.w_nll * nll
            + self.w_reward * reward_bce
            + self.w_terminal * done_bce
        )
        return {
            "total": total,
            "mse": mse.detach(),
            "nll": nll.detach(),
            "reward_bce": reward_bce.detach(),
            "done_bce": done_bce.detach(),
        }


# ----------------------------------------------------------------------------
# 3. HypothesisGenerator
# ----------------------------------------------------------------------------


@dataclass
class HypothesisOutput:
    hypotheses: torch.Tensor  # [B, T, N, D]
    scores: torch.Tensor      # [B, T, N]


class HypothesisGenerator(Module):
    """One forward pass produces ``N`` candidate next-hidden hypotheses.

    Implementation: ``N`` independent linear projectors (``nn.ModuleList``)
    on top of a shared trunk.  Each projector is initialised with a
    different Gaussian seed; outputs are residual-mixed with a per-head
    temperature.  At ``train()`` we add Gaussian reparameterisation noise
    so the generator is backprop-able when wrapped in actor-critic.
    """

    def __init__(
        self,
        hidden: int,
        n_hypotheses: int = 4,
        temperatures: Iterable[float] | None = None,
        trunk_hidden: int | None = None,
        noise_scale: float = 0.02,
    ):
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if n_hypotheses <= 0:
            raise ValueError("n_hypotheses must be positive")
        self.hidden = int(hidden)
        self.n_hypotheses = int(n_hypotheses)
        trunk_hidden = int(trunk_hidden) if trunk_hidden is not None else self.hidden
        self.trunk_hidden = trunk_hidden
        self.noise_scale = float(noise_scale)

        if temperatures is None:
            base = [0.3, 0.7, 1.0, 1.5]
            if n_hypotheses <= len(base):
                temps = base[: n_hypotheses]
            else:
                extra = n_hypotheses - len(base)
                step = (base[-1] - base[0]) / max(1, len(base) - 1)
                extended = base + [base[-1] + (i + 1) * step for i in range(extra)]
                temps = extended[: n_hypotheses]
        else:
            temps = list(temperatures)
            if len(temps) != n_hypotheses:
                raise ValueError(
                    f"temperatures has {len(temps)} entries but n_hypotheses={n_hypotheses}"
                )
        self.register_buffer("temperatures", torch.tensor(temps, dtype=torch.float32))

        self.ln = nn.LayerNorm(hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden, trunk_hidden), nn.GELU())
        self.projectors = nn.ModuleList(
            [nn.Linear(trunk_hidden, hidden) for _ in range(n_hypotheses)]
        )
        self.score_heads = nn.ModuleList(
            [nn.Linear(trunk_hidden, 1) for _ in range(n_hypotheses)]
        )
        self._init_diverse()

    def _init_diverse(self) -> None:
        """Initialise the N projectors with independent Gaussian perturbations."""
        for i, (proj, sh) in enumerate(zip(self.projectors, self.score_heads)):
            g = torch.Generator(device="cpu")
            g.manual_seed(1013 + 7 * i)
            for p in (proj.weight, sh.weight):
                nn.init.normal_(p, mean=0.0, std=(1.0 / p.shape[-1]) ** 0.5)
                noise = torch.empty_like(p).normal_(mean=0.0, std=0.05, generator=g)
                with torch.no_grad():
                    p.add_(noise)
            nn.init.zeros_(proj.bias)
            nn.init.zeros_(sh.bias)

    def forward(
        self,
        h: torch.Tensor,
        n_hypotheses: int | None = None,
        temperature: float = 1.0,
    ) -> HypothesisOutput:
        if h.dim() != 3:
            raise ValueError(f"h must be [B, T, D], got {tuple(h.shape)}")
        if h.shape[-1] != self.hidden:
            raise ValueError(
                f"hidden dim mismatch: expected {self.hidden}, got {h.shape[-1]}"
            )
        N = int(n_hypotheses) if n_hypotheses is not None else self.n_hypotheses
        if N <= 0 or N > self.n_hypotheses:
            raise ValueError(
                f"n_hypotheses={n_hypotheses} must be in (0, {self.n_hypotheses}]"
            )
        x = self.trunk(self.ln(h))

        hyp_list: list[torch.Tensor] = []
        score_list: list[torch.Tensor] = []
        for i in range(N):
            proj = self.projectors[i](x)
            head_temp = float(self.temperatures[i].item()) * float(temperature)
            if self.training and self.noise_scale > 0.0:
                noise = torch.randn_like(proj) * (self.noise_scale * head_temp)
                proj = proj + noise
            proj = h + head_temp * proj
            hyp_list.append(proj.unsqueeze(2))
            score_list.append(self.score_heads[i](x))
        hypotheses = torch.cat(hyp_list, dim=2)
        raw_scores = torch.cat(score_list, dim=-1)
        scores = F.softmax(raw_scores, dim=-1)
        return HypothesisOutput(hypotheses=hypotheses, scores=scores)


# ----------------------------------------------------------------------------
# 4. WorldModelCritic
# ----------------------------------------------------------------------------


class WorldModelCritic(Module):
    """Score hypotheses against a context vector.

    MLP::

        [hypothesis || context] -> trunk -> scalar score
    """

    def __init__(
        self,
        hidden: int,
        mlp_hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = int(hidden)
        mlp_hidden = int(mlp_hidden) if mlp_hidden is not None else self.hidden
        self.mlp_hidden = mlp_hidden
        self.ln = nn.LayerNorm(2 * hidden)
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        hypotheses: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if hypotheses.dim() != 4:
            raise ValueError(
                f"hypotheses must be [B, T, N, D], got {tuple(hypotheses.shape)}"
            )
        if context.dim() != 3:
            raise ValueError(f"context must be [B, T, D], got {tuple(context.shape)}")
        B, T, N, D = hypotheses.shape
        if context.shape[0] != B or context.shape[1] != T or context.shape[2] != D:
            raise ValueError(
                f"context {tuple(context.shape)} incompatible with "
                f"hypotheses {tuple(hypotheses.shape)}"
            )
        ctx = context.unsqueeze(2).expand(B, T, N, D)
        pair = torch.cat([hypotheses, ctx], dim=-1)
        flat = pair.reshape(B * T * N, 2 * D)
        flat = self.ln(flat)
        scores = self.trunk(flat).reshape(B, T, N)
        return scores

    @staticmethod
    def outcome_ce_loss(
        scores: torch.Tensor, outcome_reward: torch.Tensor
    ) -> torch.Tensor:
        """Actor-critic CE: argmax(outcome_reward) is the positive class."""
        if scores.shape != outcome_reward.shape:
            raise ValueError(
                f"shape mismatch: scores={tuple(scores.shape)} "
                f"vs outcome={tuple(outcome_reward.shape)}"
            )
        B, T, N = scores.shape
        target = outcome_reward.argmax(dim=-1)
        return F.cross_entropy(scores.reshape(B * T, N), target.reshape(B * T))


__all__ = [
    "WorldModelOutput",
    "WorldModelHead",
    "WorldModelLoss",
    "HypothesisOutput",
    "HypothesisGenerator",
    "WorldModelCritic",
]
