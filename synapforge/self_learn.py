"""sf.self_learn — first-class self-learning for synapforge.

Ports the four mscfc primitives (TestTimeTraining / ExperienceReplayBuffer /
SelfPlayLoop / MAMLAdapter) into `synapforge.self_learn`, exposed as
``sf.self_learn.SelfLearnEngine`` — a single orchestrator that wires them
together with a unified ``observe / adapt / train_step`` API.

Per memory feedback_self_learning.md and feedback_self_learn_poison_defense.md,
self-learning is mandatory bedrock. When ``SelfLearnEngine`` is enabled the
matching ``synapforge.defense.DefenseStack`` MUST also be installed (a
``RuntimeWarning`` is raised otherwise).

bf16-friendly: all priorities / blends are computed in fp32, weight casts use
the parameter's own dtype.
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Module

# ---------------------------------------------------------------------------
# 1. Experience replay (DER++ style)
# ---------------------------------------------------------------------------


class ExperienceReplayBuffer:
    """Circular buffer of (input, target, reward) tuples for revisit.

    DER++-style (Buzzega et al. 2004.07211): each entry has a priority
    (default = reward + 1e-3). When capacity is exceeded the lowest-priority
    entry among the oldest half is evicted.

    Sampling blends priority-weighted with uniform via ``alpha``.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.5) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self._entries: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        input_x: torch.Tensor,
        target_y: torch.Tensor,
        reward: float = 1.0,
        logit: torch.Tensor | None = None,
    ) -> int:
        eid = self._next_id
        self._next_id += 1
        self._entries[eid] = {
            "input": input_x.detach().clone(),
            "target": target_y.detach().clone(),
            "logit": None if logit is None else logit.detach().clone(),
            "reward": float(reward),
            "priority": float(reward) + 1e-3,
        }
        self._evict_if_full()
        return eid

    def _evict_if_full(self) -> None:
        while len(self._entries) > self.capacity:
            half = max(1, len(self._entries) // 2)
            candidates = list(self._entries.items())[:half]
            victim = min(candidates, key=lambda kv: kv[1]["priority"])[0]
            self._entries.pop(victim, None)

    def update_priority(self, eid: int, priority: float) -> None:
        if eid in self._entries:
            self._entries[eid]["priority"] = float(priority)

    def sample(self, batch_size: int) -> dict[str, list[Any]]:
        if not self._entries:
            return {"ids": [], "input": [], "target": [], "logit": [], "reward": []}
        ids = list(self._entries.keys())
        priors = torch.tensor(
            [self._entries[i]["priority"] for i in ids], dtype=torch.float32
        )
        w_prior = priors.clamp(min=1e-6).pow(self.alpha)
        w_prior = w_prior / w_prior.sum()
        w_unif = torch.ones_like(w_prior) / len(w_prior)
        weights = self.alpha * w_prior + (1.0 - self.alpha) * w_unif
        k = min(batch_size, len(ids))
        idx = torch.multinomial(weights, k, replacement=(k > len(ids))).tolist()
        picked = [ids[i] for i in idx]
        return {
            "ids": picked,
            "input": [self._entries[i]["input"] for i in picked],
            "target": [self._entries[i]["target"] for i in picked],
            "logit": [self._entries[i]["logit"] for i in picked],
            "reward": [self._entries[i]["reward"] for i in picked],
        }


# ---------------------------------------------------------------------------
# 2. Test-time training (Sun et al. 1909.13231)
# ---------------------------------------------------------------------------


class TestTimeTraining(Module):
    """Adapts model weights from inference-time inputs.

    Runs ``ttt_steps`` SGD updates on a self-supervised reconstruction loss
    on the current input *before* the real forward pass. Only parameters
    matching ``params_filter`` are touched — the LM head / embeddings stay
    frozen by default.

    ``restore_after=True`` rolls weights back so inference stays stateless.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 1e-3,
        steps: int = 1,
        params_filter: Sequence[str] = ("shared_tau", "fast.w_", "ttt_"),
        mask_prob: float = 0.15,
        mask_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.inner_lr = float(inner_lr)
        self.steps = int(steps)
        self.params_filter = tuple(params_filter)
        self.mask_prob = float(mask_prob)
        self.mask_token_id = int(mask_token_id)
        self._snapshot: dict[str, torch.Tensor] | None = None

    def _filter_params(self) -> list[tuple[str, nn.Parameter]]:
        out: list[tuple[str, nn.Parameter]] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(tok in name for tok in self.params_filter):
                out.append((name, p))
        # Fallback: if filter matches nothing, adapt the *last* leaf params
        # so smoke tests don't no-op silently.
        if not out:
            tail = list(self.model.named_parameters())[-2:]
            out = [(n, p) for n, p in tail if p.requires_grad]
        return out

    def _snapshot_filtered(self) -> None:
        self._snapshot = {n: p.detach().clone() for n, p in self._filter_params()}

    def restore(self) -> None:
        if self._snapshot is None:
            return
        state = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, saved in self._snapshot.items():
                if n in state:
                    state[n].copy_(saved.to(state[n].dtype))
        self._snapshot = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self.model(tokens=x)
        except TypeError:
            out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _masked_loss(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.long, torch.int64):
            noise = torch.randn_like(x) * 0.1
            corrupted = x + noise
            pred = self._forward(corrupted)
            target = x
            if pred.shape != target.shape:
                m = min(pred.shape[-1], target.shape[-1])
                return F.mse_loss(pred[..., :m].float(), target[..., :m].float())
            return F.mse_loss(pred.float(), target.float())
        mask = torch.rand(x.shape, device=x.device) < self.mask_prob
        if not mask.any():
            mask[..., 0] = True
        masked = x.clone()
        masked[mask] = self.mask_token_id
        logits = self._forward(masked)
        if logits.dim() == 3:
            v = logits.shape[-1]
            flat_logits = logits.reshape(-1, v).float()
            flat_tgt = x.reshape(-1)
            flat_mask = mask.reshape(-1)
            idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return logits.float().sum() * 0.0
            return F.cross_entropy(flat_logits[idx], flat_tgt[idx])
        return logits.float().mean()

    def adapt(self, x: torch.Tensor, restore_after: bool = False) -> float:
        self._snapshot_filtered()
        params = [p for _, p in self._filter_params()]
        if not params:
            return 0.0
        opt = torch.optim.SGD(params, lr=self.inner_lr)
        last = 0.0
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            loss = self._masked_loss(x)
            loss.backward()
            opt.step()
            last = float(loss.detach())
        if restore_after:
            self.restore()
        return last


# ---------------------------------------------------------------------------
# 3. Self-play loop (SPIN-style, 2401.01335)
# ---------------------------------------------------------------------------


def _jsd(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = (p * (p.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    kl_qm = (q * (q.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


class SelfPlayLoop:
    """Generate -> evaluate -> replay loop (SPIN style).

    Each round the model produces candidate logits for a prompt; teachers
    score by JSD. Low-JSD (consensus) candidates are stored in ``replay``
    with high priority; high-JSD ones go to ``adversarial_buf``.
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_ensemble: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        replay: ExperienceReplayBuffer,
        num_candidates: int = 4,
        jsd_low: float = 0.2,
        jsd_high: float = 0.5,
    ) -> None:
        if not teacher_ensemble:
            raise ValueError("teacher_ensemble must be non-empty")
        self.model = model
        self.teachers = list(teacher_ensemble)
        self.replay = replay
        self.num_candidates = int(num_candidates)
        self.jsd_low = float(jsd_low)
        self.jsd_high = float(jsd_high)
        self.adversarial_buf: list[dict[str, Any]] = []

    @torch.no_grad()
    def _model_logits(self, prompt: torch.Tensor) -> torch.Tensor:
        try:
            out = self.model(tokens=prompt)
        except TypeError:
            out = self.model(prompt)
        if isinstance(out, tuple):
            out = out[0]
        return out

    @torch.no_grad()
    def round(self, prompt: torch.Tensor) -> dict[str, int]:
        counts = {"replay": 0, "adversarial": 0, "middle": 0}
        for _ in range(self.num_candidates):
            m_logits = self._model_logits(prompt)
            jsds: list[torch.Tensor] = []
            for t in self.teachers:
                t_logits = t(prompt)
                if t_logits.shape != m_logits.shape:
                    vmin = min(t_logits.shape[-1], m_logits.shape[-1])
                    jsds.append(_jsd(m_logits[..., :vmin], t_logits[..., :vmin]).mean())
                else:
                    jsds.append(_jsd(m_logits, t_logits).mean())
            j = float(torch.stack(jsds).mean())
            target = prompt
            if j < self.jsd_low:
                self.replay.add(prompt, target, reward=1.0 / (j + 1e-3), logit=m_logits)
                counts["replay"] += 1
            elif j > self.jsd_high:
                self.adversarial_buf.append(
                    {
                        "prompt": prompt.detach().clone(),
                        "logit": m_logits.detach().clone(),
                        "jsd": j,
                    }
                )
                counts["adversarial"] += 1
            else:
                counts["middle"] += 1
        return counts


# ---------------------------------------------------------------------------
# 4. MAML first-order adapter (1703.03400)
# ---------------------------------------------------------------------------


class MAMLAdapter(Module):
    """Meta-learning fast adaptation (first-order MAML).

    ``few_shot_adapt(support_set)`` runs ``inner_steps`` SGD updates and
    returns ``{name: adapted_tensor}``. Base weights are restored before
    return so the outer optimizer applies meta-grad cleanly.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_steps: int = 5,
        inner_lr: float = 1e-3,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.inner_steps = int(inner_steps)
        self.inner_lr = float(inner_lr)
        self.loss_fn = loss_fn or (
            lambda pred, tgt: F.mse_loss(pred.float(), tgt.float())
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self.model(tokens=x)
        except TypeError:
            out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def few_shot_adapt(
        self,
        support_set: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        base_state = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        params = list(self.model.parameters())
        opt = torch.optim.SGD(params, lr=self.inner_lr)
        support_list = list(support_set)
        last_loss = torch.tensor(0.0)
        for _ in range(self.inner_steps):
            for x, y in support_list:
                opt.zero_grad(set_to_none=True)
                pred = self._forward(x)
                if pred.shape != y.shape:
                    vmin = min(pred.shape[-1], y.shape[-1])
                    pred = pred[..., :vmin]
                    y = y[..., :vmin]
                loss = self.loss_fn(pred, y)
                loss.backward()
                opt.step()
                last_loss = loss.detach()
        adapted = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in base_state:
                    p.copy_(base_state[n].to(p.dtype))
        adapted["__inner_loss__"] = last_loss
        return adapted


# ---------------------------------------------------------------------------
# 5. Unified orchestrator
# ---------------------------------------------------------------------------


class SelfLearnEngine:
    """Single entry-point for all four self-learning components.

    Wires ``ExperienceReplayBuffer + TestTimeTraining + SelfPlayLoop +
    MAMLAdapter`` to a model. Public API:

        engine = SelfLearnEngine(model, replay_size=10000, ttt_steps=3,
                                  maml_inner_lr=1e-3)
        engine.observe(input_x, label_y, reward=r)   # -> replay
        engine.adapt(x)                               # -> fast TTT step
        loss = engine.train_step(batch)               # -> outer step

    A ``defense`` slot is provided so callers can attach a ``DefenseStack``.
    The constructor warns (``RuntimeWarning``) if no defense is attached —
    self-learning without poison guards is the documented anti-pattern.
    """

    def __init__(
        self,
        model: nn.Module,
        replay_size: int = 10000,
        replay_alpha: float = 0.5,
        ttt_steps: int = 3,
        ttt_lr: float = 1e-3,
        ttt_filter: Sequence[str] = ("shared_tau", "fast.w_", "ttt_"),
        maml_inner_steps: int = 5,
        maml_inner_lr: float = 1e-3,
        outer_lr: float = 5e-4,
        defense: Any | None = None,
        warn_no_defense: bool = True,
    ) -> None:
        self.model = model
        self.replay = ExperienceReplayBuffer(replay_size, replay_alpha)
        self.ttt = TestTimeTraining(model, ttt_lr, ttt_steps, ttt_filter)
        self.maml = MAMLAdapter(model, maml_inner_steps, maml_inner_lr)
        self.outer_opt = torch.optim.AdamW(model.parameters(), lr=outer_lr)
        self.defense = defense
        if defense is None and warn_no_defense:
            warnings.warn(
                "SelfLearnEngine running without DefenseStack — poison/drift "
                "is not protected. See synapforge.defense.DefenseStack and "
                "feedback_self_learn_poison_defense.md.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._step = 0
        self._self_play: SelfPlayLoop | None = None

    # ----------------------------------------------------------- observe
    def observe(
        self,
        input_x: torch.Tensor,
        label_y: torch.Tensor,
        reward: float = 1.0,
        source: str = "user",
    ) -> int:
        """Add an (input, label, reward) triple to the replay buffer."""
        if self.defense is not None:
            ok = self.defense.accept(input_x, source=source)
            if not ok:
                return -1
        return self.replay.add(input_x, label_y, reward=reward)

    # ----------------------------------------------------------- adapt
    def adapt(self, x: torch.Tensor, restore_after: bool = False) -> float:
        """Run TTT inner loop on ``x``. Returns the inner loss."""
        if self.defense is not None:
            self.defense.pre_step()
        loss = self.ttt.adapt(x, restore_after=restore_after)
        if self.defense is not None:
            self.defense.after_grads()
            if self.defense.regressed_and_rollback():
                pass  # rolled back automatically
            self.defense.tick()
        return loss

    # ----------------------------------------------------------- self-play
    def attach_self_play(
        self,
        teachers: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        num_candidates: int = 4,
        jsd_low: float = 0.2,
        jsd_high: float = 0.5,
    ) -> SelfPlayLoop:
        sp = SelfPlayLoop(
            self.model,
            teachers,
            self.replay,
            num_candidates=num_candidates,
            jsd_low=jsd_low,
            jsd_high=jsd_high,
        )
        self._self_play = sp
        return sp

    def self_play_round(self, prompt: torch.Tensor) -> dict[str, int]:
        if self._self_play is None:
            raise RuntimeError(
                "self-play not attached; call attach_self_play(teachers) first"
            )
        return self._self_play.round(prompt)

    # ----------------------------------------------------------- train_step
    def train_step(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        replay_ratio: float = 0.5,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> float:
        """Outer training step. Mixes ``batch`` with replayed samples.

        ``batch`` can be a dict {"input": ..., "target": ...} or a tuple
        (input_x, target_y). Returns the scalar loss.
        """
        if isinstance(batch, dict):
            x = batch["input"]
            y = batch["target"]
        else:
            x, y = batch

        loss_fn = loss_fn or (lambda p, t: F.mse_loss(p.float(), t.float()))
        self.outer_opt.zero_grad(set_to_none=True)

        # Forward on the live batch.
        try:
            pred = self.model(tokens=x)
        except TypeError:
            pred = self.model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.shape != y.shape:
            vmin = min(pred.shape[-1], y.shape[-1])
            pred = pred[..., :vmin]
            y_use = y[..., :vmin]
        else:
            y_use = y
        loss_live = loss_fn(pred, y_use)

        # Mix in replay if we have anything banked.
        loss_replay = torch.tensor(0.0, device=loss_live.device)
        if len(self.replay) > 0 and replay_ratio > 0.0:
            sample = self.replay.sample(max(1, x.shape[0] // 2 if x.dim() > 0 else 1))
            n = len(sample["input"])
            losses = []
            for i in range(n):
                xi = sample["input"][i]
                yi = sample["target"][i]
                try:
                    pi = self.model(tokens=xi)
                except TypeError:
                    pi = self.model(xi)
                if isinstance(pi, tuple):
                    pi = pi[0]
                if pi.shape != yi.shape:
                    vmin = min(pi.shape[-1], yi.shape[-1])
                    pi = pi[..., :vmin]
                    yi = yi[..., :vmin]
                losses.append(loss_fn(pi, yi))
            if losses:
                loss_replay = torch.stack(losses).mean()

        loss = (1.0 - replay_ratio) * loss_live + replay_ratio * loss_replay

        if self.defense is not None:
            self.defense.pre_step()

        loss.backward()

        if self.defense is not None:
            self.defense.after_grads()

        self.outer_opt.step()

        if self.defense is not None:
            if self.defense.regressed_and_rollback():
                pass
            self.defense.tick()

        self._step += 1
        return float(loss.detach())


__all__ = [
    "ExperienceReplayBuffer",
    "TestTimeTraining",
    "SelfPlayLoop",
    "MAMLAdapter",
    "SelfLearnEngine",
]
