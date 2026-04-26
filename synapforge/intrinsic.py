"""Intrinsic-motivation modules — 7 components for self-driven learning.

Pure CPU-friendly, no torch.jit.  Modules cover curiosity, imagination,
homeostasis, and self-proposed goals.

Components
----------
1. ``FreeEnergySurprise``   — ICM-style forward-model surprise (1705.05363).
2. ``SelfGoalProposer``     — Voyager-style autocurriculum goal generation.
3. ``ImaginationRollout``   — Dreamer-style latent-space planning.
4. ``NoveltyDrive``         — running-EMA novelty signal.
5. ``HomeostaticRegulator`` — biological band-penalty.
6. ``IdleLoop``             — background watchdog for self-play.
7. ``GoalMemory``           — Self-Discover style goal-improvement bank.

Each component is independent and exported as a top-level symbol on
``synapforge.intrinsic``.  ``IntrinsicReward`` is a convenience class
that bundles 1+4+5 into one scalar reward signal.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Module

# ---------------------------------------------------------------------------
# 1. FreeEnergySurprise (ICM-style forward model)
# ---------------------------------------------------------------------------


class FreeEnergySurprise(Module):
    """Free-energy / surprise via a small forward model (ICM 1705.05363)."""

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        return self.state_predictor(h_prev)

    def surprise(self, h_prev: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        """Scalar surprise (gradient flows through the predictor)."""
        if h_prev.shape != h_next.shape:
            raise ValueError(
                f"shape mismatch: h_prev={tuple(h_prev.shape)} h_next={tuple(h_next.shape)}"
            )
        pred = self.state_predictor(h_prev)
        return (pred - h_next).pow(2).mean()


# ---------------------------------------------------------------------------
# 2. SelfGoalProposer (Voyager-style)
# ---------------------------------------------------------------------------


class SelfGoalProposer:
    """Generate token-level ``<goal>...</goal>`` sequences (Voyager 2305.16291).

    Temperature anneals from ``t_start`` to ``t_end`` across ``anneal_steps``
    proposals.  Falls back to uniform-random on logits failure.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        goal_tokens: tuple[int, int] = (0, 1),
        t_start: float = 1.5,
        t_end: float = 0.5,
        anneal_steps: int = 1000,
        max_len: int = 16,
    ) -> None:
        self.model = model
        self.vocab_size = int(vocab_size)
        self.bos, self.eos = int(goal_tokens[0]), int(goal_tokens[1])
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.anneal_steps = max(1, int(anneal_steps))
        self.max_len = int(max_len)
        self._calls = 0

    def _temperature(self) -> float:
        frac = min(1.0, self._calls / self.anneal_steps)
        return self.t_start * (1 - frac) + self.t_end * frac

    @torch.no_grad()
    def propose(self, context: torch.Tensor | None = None) -> list[int]:
        self._calls += 1
        T = self._temperature()
        seed = context if context is not None else torch.tensor([[self.bos]], dtype=torch.long)
        tokens: list[int] = [self.bos]
        cur = seed.clone()
        for _ in range(self.max_len - 1):
            try:
                out = self.model(tokens=cur)
                logits = out[0] if isinstance(out, tuple) else out
                last = logits[:, -1] if logits.dim() == 3 else logits
                probs = F.softmax(last / max(T, 1e-3), dim=-1)
                nxt = torch.multinomial(probs, 1).view(-1).item()
            except Exception:
                nxt = random.randint(0, self.vocab_size - 1)
            tokens.append(int(nxt))
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
            if nxt == self.eos:
                break
        if tokens[-1] != self.eos:
            tokens.append(self.eos)
        return tokens


# ---------------------------------------------------------------------------
# 3. ImaginationRollout
# ---------------------------------------------------------------------------


class ImaginationRollout:
    """Latent-space planning rollout (Dreamer V3 2301.04104, Plan2Explore)."""

    def __init__(
        self,
        model: nn.Module,
        max_think_steps: int = 32,
        beam: int = 4,
        reward_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.max_think_steps = int(max_think_steps)
        self.beam = max(1, int(beam))
        self.reward_fn = reward_fn or (lambda h: h.norm(dim=-1).mean())

    @torch.no_grad()
    def dream(self, goal_tokens: Sequence[int]) -> tuple[torch.Tensor, list[float]]:
        t = torch.tensor([list(goal_tokens)] * self.beam, dtype=torch.long)
        rewards: list[float] = []
        best_h = None
        best_r = -math.inf
        cur = t
        logits = None
        for _step in range(self.max_think_steps):
            try:
                out = self.model(tokens=cur)
                logits = out[0] if isinstance(out, tuple) else out
                aux = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
                h = aux.get("hidden") if isinstance(aux, dict) else None
                if h is None:
                    h = logits.mean(dim=-1, keepdim=True).expand(-1, -1, 8)
                h_last = h[:, -1] if h.dim() == 3 else h
            except Exception:
                h_last = torch.randn(self.beam, 8)
            r = float(self.reward_fn(h_last))
            rewards.append(r)
            if r > best_r:
                best_r = r
                best_h = h_last.clone()
            try:
                if logits is None:
                    break
                probs = F.softmax(logits[:, -1], dim=-1)
                nxt = torch.multinomial(probs, 1)
                cur = torch.cat([cur, nxt], dim=1)
            except Exception:
                break
        if best_h is None:
            best_h = torch.zeros(1, 8)
        return best_h, rewards


# ---------------------------------------------------------------------------
# 4. NoveltyDrive
# ---------------------------------------------------------------------------


class NoveltyDrive(Module):
    """Running-EMA novelty signal: ``||h - EMA(h)||``."""

    def __init__(self, hidden_size: int, ema: float = 0.99) -> None:
        super().__init__()
        if not 0.0 < ema < 1.0:
            raise ValueError("ema must be in (0, 1)")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.ema = float(ema)
        self.register_buffer("h_bar", torch.zeros(hidden_size))
        self._initialised = False

    def update(self, h: torch.Tensor) -> None:
        batch_mean = h.detach()
        if batch_mean.dim() > 1:
            batch_mean = batch_mean.reshape(-1, self.hidden_size).mean(dim=0)
        with torch.no_grad():
            if not self._initialised:
                self.h_bar.copy_(batch_mean)
                self._initialised = True
            else:
                self.h_bar.mul_(self.ema).add_((1 - self.ema) * batch_mean)

    def novelty(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.reshape(-1, self.hidden_size) if h.dim() > 1 else h.unsqueeze(0)
        d = (flat - self.h_bar).norm(dim=-1)
        self.update(h)
        return d.mean()


# ---------------------------------------------------------------------------
# 5. Homeostatic regulator
# ---------------------------------------------------------------------------


class HomeostaticRegulator:
    """Penalty outside biologically plausible spike/tau ranges (<= 0)."""

    def __init__(
        self,
        spike_target: tuple[float, float] = (0.05, 0.15),
        tau_range: tuple[float, float] = (0.5, 20.0),
    ) -> None:
        self.spike_lo, self.spike_hi = map(float, spike_target)
        self.tau_lo, self.tau_hi = map(float, tau_range)

    @staticmethod
    def _band_penalty(x: float, lo: float, hi: float) -> float:
        if lo <= x <= hi:
            return 0.0
        d = lo - x if x < lo else x - hi
        return -float(d * d)

    def penalty(self, current_spike_rate: float, current_tau: float) -> float:
        s = self._band_penalty(float(current_spike_rate), self.spike_lo, self.spike_hi)
        t = self._band_penalty(float(current_tau), self.tau_lo, self.tau_hi)
        return s + t


# ---------------------------------------------------------------------------
# 6. IdleLoop (background watchdog)
# ---------------------------------------------------------------------------


class IdleLoop:
    """Background thread triggering self-play when the user is idle."""

    def __init__(
        self,
        watchdog_sec: float = 30.0,
        proposer: SelfGoalProposer | None = None,
        rollout: ImaginationRollout | None = None,
        reward_combiner: Callable[[list[float]], float] | None = None,
    ) -> None:
        self.watchdog_sec = float(watchdog_sec)
        self.proposer = proposer
        self.rollout = rollout
        self.reward_combiner = reward_combiner or (lambda rs: sum(rs) / max(len(rs), 1))
        self.last_user_input_ts: float = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=256)

    def notify_user_input(self) -> None:
        self.last_user_input_ts = time.time()

    def _run(self) -> None:
        while not self._stop.is_set():
            idle_for = time.time() - self.last_user_input_ts
            if idle_for >= self.watchdog_sec and self.proposer and self.rollout:
                try:
                    goal = self.proposer.propose()
                    _hidden, rewards = self.rollout.dream(goal)
                    combined = self.reward_combiner(rewards)
                    self.events.append({
                        "ts": time.time(),
                        "goal": goal,
                        "reward": combined,
                        "n_steps": len(rewards),
                    })
                    self.notify_user_input()
                except Exception as exc:  # pragma: no cover - defensive
                    self.events.append({"ts": time.time(), "error": repr(exc)})
            self._stop.wait(max(0.05, self.watchdog_sec))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None


# ---------------------------------------------------------------------------
# 7. GoalMemory (Self-Discover style)
# ---------------------------------------------------------------------------


class GoalMemory:
    """Buffer of goal records keyed by (pre_loss - post_loss) improvement."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = int(capacity)
        self._records: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._records)

    def record(self, goal_tokens: Sequence[int], pre_loss: float, post_loss: float) -> None:
        improvement = float(pre_loss) - float(post_loss)
        self._records.append({
            "goal": list(goal_tokens),
            "pre": float(pre_loss),
            "post": float(post_loss),
            "improvement": improvement,
        })
        if len(self._records) > self.capacity:
            self._records.sort(key=lambda r: r["improvement"], reverse=True)
            self._records = self._records[: self.capacity]

    def top_k_features(self, k: int = 20) -> torch.Tensor:
        if not self._records:
            return torch.zeros(1)
        k = min(k, len(self._records))
        best = sorted(self._records, key=lambda r: r["improvement"], reverse=True)[:k]
        max_len = max(len(r["goal"]) for r in best)
        feat = torch.zeros(k, max_len)
        for i, r in enumerate(best):
            g = torch.tensor(r["goal"], dtype=torch.float32)
            feat[i, : g.numel()] = g
        return feat.mean(dim=0)


# ---------------------------------------------------------------------------
# Convenience: combined intrinsic reward
# ---------------------------------------------------------------------------


class IntrinsicReward(Module):
    """Bundle ICM surprise + Novelty + Homeostatic penalty into one scalar.

    Usage:
        ir = sf.IntrinsicReward(hidden_size=256)
        scalar = ir(h_prev, h_next, spike_rate=0.1, tau=10.0)

    Returns a positive-or-zero reward (penalty subtracted).
    """

    def __init__(
        self,
        hidden_size: int,
        weight_surprise: float = 1.0,
        weight_novelty: float = 0.5,
        weight_homeostatic: float = 0.1,
        ema: float = 0.99,
    ):
        super().__init__()
        self.surprise_mod = FreeEnergySurprise(hidden_size)
        self.novelty_mod = NoveltyDrive(hidden_size, ema=ema)
        self.homeo = HomeostaticRegulator()
        self.w_surprise = float(weight_surprise)
        self.w_novelty = float(weight_novelty)
        self.w_homeo = float(weight_homeostatic)

    def forward(
        self,
        h_prev: torch.Tensor,
        h_next: torch.Tensor,
        spike_rate: float = 0.10,
        tau: float = 10.0,
    ) -> torch.Tensor:
        s = self.surprise_mod.surprise(h_prev, h_next)
        n = self.novelty_mod.novelty(h_next)
        homeo = float(self.homeo.penalty(spike_rate, tau))
        return self.w_surprise * s + self.w_novelty * n + self.w_homeo * homeo


__all__ = [
    "FreeEnergySurprise",
    "SelfGoalProposer",
    "ImaginationRollout",
    "NoveltyDrive",
    "HomeostaticRegulator",
    "IdleLoop",
    "GoalMemory",
    "IntrinsicReward",
]
