"""sf.plasticity — first-class plasticity ops with autograd-aware execution.

Solves PyTorch's buffer-inplace + autograd-version conflict for biological
plasticity rules. Execution model:

    OBSERVE phase (in forward):
        rule.observe(pre=..., post=..., t=...)   # only updates LOCAL traces
                                                 # NEVER mutates W
    DELTA phase (after backward):
        engine.step(t, weight_dict) -> {name: dW}
                                                 # returns deltas; caller
                                                 # applies them once, atomically

    APPLY phase:
        engine.apply(deltas, weight_dict)        # in-place add, no version conflict

W is mutated EXACTLY once per engine.step() call AFTER the BP graph has
been consumed. Eligibility traces themselves can carry gradient through
tau / lr Parameters so plasticity hyper-parameters are end-to-end trainable.

Public classes
--------------
    PlasticityRule            base
    Hebbian                   ΔW = lr * post.T @ pre  (co-firing)
    STDP                      EMA-trace timing-dependent
    BCM                       pre × post × (post - theta), sliding theta
    SynaptogenesisGrowPrune   mask delta from coactivation EMA
    PlasticityEngine          orchestrator; merges all rules

Legacy aliases (kept so synapforge/__init__.py keeps importing):
    HebbianPlasticity         older Hebbian-Oja API (dim, eta, decay)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .module import Module

__all__ = [
    "PlasticityRule",
    "Hebbian",
    "STDP",
    "BCM",
    "SynaptogenesisGrowPrune",
    "PlasticityEngine",
    "HebbianPlasticity",
]


# ============================================================================
# Base
# ============================================================================


class PlasticityRule(nn.Module):
    """Base class for all plasticity rules.

    Subclasses MUST implement:
        observe(*, pre, post, t) -> None        — updates internal trace state
        compute_delta_W() -> Optional[Tensor]   — returns weight delta (or None)

    Optional API:
        reset() -> None                          — clears traces
        as_callback(get_pre, get_post)          — adapter to sf.Module hook

    Trace tensors are LAZY: allocated on first observe() so dim/device/dtype
    are auto-inferred. Avoids the need to pass dim at __init__.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ready = False

    def observe(self, *, pre: torch.Tensor, post: torch.Tensor, t: float) -> None:
        raise NotImplementedError

    def compute_delta_W(self) -> torch.Tensor | None:
        raise NotImplementedError

    def reset(self) -> None:
        self._ready = False

    def as_callback(
        self,
        get_pre: Callable[..., torch.Tensor],
        get_post: Callable[..., torch.Tensor],
    ) -> Callable[..., None]:
        """Wrap as a sf.Module.register_plasticity-compatible callback."""

        def _cb(module: nn.Module, inputs, outputs) -> None:
            pre = get_pre(module, inputs, outputs)
            post = get_post(module, inputs, outputs)
            self.observe(pre=pre, post=post, t=0.0)

        return _cb


# ============================================================================
# Hebbian
# ============================================================================


class Hebbian(PlasticityRule):
    """Simple Hebbian: ΔW = lr * (post.T @ pre) over pending observations.

    `lr_param` is a Parameter so its grad-flow can be tracked if downstream
    losses are backproped through delta_W (autograd-aware mode). Pending
    pre/post are stored DETACHED — only `lr_param` carries grad through delta.
    """

    def __init__(self, lr: float = 1e-3, max_pending: int = 256) -> None:
        super().__init__()
        self.lr_param = nn.Parameter(torch.tensor(float(lr)))
        self.max_pending = int(max_pending)
        self._pending_pre: list[torch.Tensor] = []
        self._pending_post: list[torch.Tensor] = []

    def observe(self, *, pre: torch.Tensor, post: torch.Tensor, t: float = 0.0) -> None:
        p = pre.detach().reshape(-1, pre.shape[-1])
        q = post.detach().reshape(-1, post.shape[-1])
        if p.shape[0] == 0:
            return
        self._pending_pre.append(p)
        self._pending_post.append(q)
        if len(self._pending_pre) > self.max_pending:
            self._pending_pre.pop(0)
            self._pending_post.pop(0)
        self._ready = True

    def compute_delta_W(self) -> torch.Tensor | None:
        if not self._pending_pre:
            return None
        pre_cat = torch.cat(self._pending_pre, dim=0)
        post_cat = torch.cat(self._pending_post, dim=0)
        n = pre_cat.shape[0]
        delta = (post_cat.t() @ pre_cat) / float(n)
        delta = self.lr_param * delta
        self._pending_pre.clear()
        self._pending_post.clear()
        return delta

    def reset(self) -> None:
        self._pending_pre.clear()
        self._pending_post.clear()
        super().reset()


# ============================================================================
# STDP
# ============================================================================


class STDP(PlasticityRule):
    """Spike-timing-dependent plasticity with eligibility traces.

    Trace dynamics (per-channel EMA, fresh tensor each step → no version conflict):
        pre_trace(t)  = pre_trace(t-1)  * exp(-dt/tau_pre)  + pre_spike
        post_trace(t) = post_trace(t-1) * exp(-dt/tau_post) + post_spike

    Update at compute_delta_W:
        ΔW = lr * (A_plus * outer(post_spike, pre_trace)
                  - A_minus * outer(pre_spike, post_trace))

    Pre-before-post → potentiation (positive delta).
    Post-before-pre → depression (negative delta).

    `tau_pre`, `tau_post`, `lr_param` are Parameters → learnable.
    Lazy init: trace tensor shape inferred from first observe.
    """

    def __init__(
        self,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        lr: float = 1e-4,
        eligibility_window: float = 50.0,
        a_plus: float = 1.0,
        a_minus: float = 1.0,
        decay_after_compute: float = 0.5,
        dim: int | None = None,
    ) -> None:
        """Eligibility-trace STDP. New code uses dim=None (lazy). Pass dim=D
        for legacy `stdp(pre, post)` callable form (allocates W_fast buffer
        of shape (D, D) and emits a modulation = pre @ W_fast.T).
        """
        super().__init__()
        self.tau_pre = nn.Parameter(torch.tensor(float(tau_pre)))
        self.tau_post = nn.Parameter(torch.tensor(float(tau_post)))
        self.lr_param = nn.Parameter(torch.tensor(float(lr)))
        self.eligibility_window = float(eligibility_window)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.decay_after_compute = float(decay_after_compute)

        self._trace_pre: torch.Tensor | None = None
        self._trace_post: torch.Tensor | None = None
        self._last_pre: torch.Tensor | None = None
        self._last_post: torch.Tensor | None = None
        self._steps_since_compute = 0

        # Legacy compat: when dim is given, pre-allocate W_fast for the
        # callable form `stdp(pre, post)` used by older code (e.g.
        # synapforge.example_optim).
        self.dim = int(dim) if dim is not None else None
        if dim is not None:
            self.register_buffer("W_fast", torch.zeros(int(dim), int(dim)))
        else:
            # Keep attribute present so getattr works; no buffer until used.
            self.W_fast = None  # type: ignore[assignment]

    def reset_eligibility(self) -> None:
        self._trace_pre = None
        self._trace_post = None
        self._last_pre = None
        self._last_post = None
        self._steps_since_compute = 0
        self._ready = False

    def reset(self) -> None:
        self.reset_eligibility()
        super().reset()

    def observe(self, *, pre: torch.Tensor, post: torch.Tensor, t: float = 1.0) -> None:
        """Update eligibility traces from a (..., dim) spike tensor pair.

        `t` is dt since previous observe (defaults 1.0 step). Does NOT mutate
        any weight. Stores last spikes for compute_delta_W.
        """
        pre_d = pre.detach().reshape(-1, pre.shape[-1]).mean(dim=0)
        post_d = post.detach().reshape(-1, post.shape[-1]).mean(dim=0)

        decay_pre = torch.exp(-float(t) / self.tau_pre.clamp_min(1e-3))
        decay_post = torch.exp(-float(t) / self.tau_post.clamp_min(1e-3))

        if self._trace_pre is None:
            self._trace_pre = torch.zeros_like(pre_d)
            self._trace_post = torch.zeros_like(post_d)

        # Out-of-place: traces are FRESH tensors each step (no version conflict).
        self._trace_pre = self._trace_pre.detach() * decay_pre + pre_d
        self._trace_post = self._trace_post.detach() * decay_post + post_d
        self._last_pre = pre_d
        self._last_post = post_d
        self._steps_since_compute += 1
        self._ready = True

    def compute_delta_W(self) -> torch.Tensor | None:
        """Return STDP weight update tensor; shape (post_dim, pre_dim)."""
        if not self._ready or self._trace_pre is None or self._last_pre is None:
            return None
        ltp = self.a_plus * torch.outer(self._last_post, self._trace_pre)
        ltd = self.a_minus * torch.outer(self._last_pre, self._trace_post)
        delta = self.lr_param * (ltp - ltd)
        keep = float(self.decay_after_compute)
        self._trace_pre = self._trace_pre.detach() * keep
        self._trace_post = self._trace_post.detach() * keep
        self._steps_since_compute = 0
        return delta

    @torch.no_grad()
    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Legacy callable form. Observes + applies + returns modulation.

        Only works if `dim` was passed at __init__ (W_fast pre-allocated).
        Modern code uses observe() + PlasticityEngine.step() instead.
        """
        if self.W_fast is None:
            raise RuntimeError(
                "STDP.forward() requires dim=D at construction. Use "
                "observe() + PlasticityEngine.step() for the modern API."
            )
        self.observe(pre=pre, post=post, t=1.0)
        delta = self.compute_delta_W()
        if delta is not None:
            self.W_fast.add_(delta.detach())
            self.W_fast.clamp_(-1.0, 1.0)  # soft bound, mirrors mscfc.bio
        return (pre @ self.W_fast.t()).view_as(pre)


# ============================================================================
# BCM
# ============================================================================


class BCM(PlasticityRule):
    """Bienenstock-Cooper-Munro rule with sliding-threshold homeostasis.

    Update:
        phi(post)  = post * (post - theta)
        ΔW         = lr * outer(phi(post), pre)
        theta(t+1) = theta(t) * (1-rho) + rho * (post^2).mean()

    LTP when post > theta (high activity), LTD when 0 < post < theta.
    Threshold slides up with activity → homeostatic stability.
    """

    def __init__(
        self,
        theta_init: float = 1.0,
        lr: float = 1e-4,
        theta_rate: float = 1e-3,
        max_pending: int = 256,
    ) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))
        self.lr_param = nn.Parameter(torch.tensor(float(lr)))
        self.theta_rate = float(theta_rate)
        self.max_pending = int(max_pending)
        self._pending_pre: list[torch.Tensor] = []
        self._pending_post: list[torch.Tensor] = []

    def observe(self, *, pre: torch.Tensor, post: torch.Tensor, t: float = 0.0) -> None:
        p = pre.detach().reshape(-1, pre.shape[-1])
        q = post.detach().reshape(-1, post.shape[-1])
        if p.shape[0] == 0:
            return
        self._pending_pre.append(p)
        self._pending_post.append(q)
        if len(self._pending_pre) > self.max_pending:
            self._pending_pre.pop(0)
            self._pending_post.pop(0)
        self._ready = True

    def compute_delta_W(self) -> torch.Tensor | None:
        if not self._pending_pre:
            return None
        pre_cat = torch.cat(self._pending_pre, dim=0)
        post_cat = torch.cat(self._pending_post, dim=0)
        n = pre_cat.shape[0]
        phi = post_cat * (post_cat - self.theta)
        delta = self.lr_param * (phi.t() @ pre_cat) / float(n)
        with torch.no_grad():
            mean_sq = (post_cat * post_cat).mean()
            self.theta.data.mul_(1.0 - self.theta_rate).add_(self.theta_rate * mean_sq)
        self._pending_pre.clear()
        self._pending_post.clear()
        return delta

    def reset(self) -> None:
        self._pending_pre.clear()
        self._pending_post.clear()
        super().reset()


# ============================================================================
# Synaptogenesis (structural plasticity)
# ============================================================================


class SynaptogenesisGrowPrune(PlasticityRule):
    """Connection mask grow/prune via coactivation EMA.

    Maintains coact = EMA(outer(post, pre)). Periodically:
      - GROW: add `n_grow` strongest INACTIVE entries (by |coact|).
      - PRUNE: remove `n_prune` weakest ACTIVE entries (by |W| * |coact|).

    compute_mask_delta returns a SIGNED int8 delta:
        +1 → grow connection, -1 → prune connection, 0 → no change

    `target_density` is a soft setpoint; per-step change bounded by
    `max_change_per_step`.
    """

    def __init__(
        self,
        target_density: float = 0.10,
        growth_check_every: int = 50,
        prune_check_every: int = 200,
        ema_decay: float = 0.99,
        max_change_per_step: int = 64,
    ) -> None:
        super().__init__()
        if not 0.0 < target_density <= 1.0:
            raise ValueError(f"target_density must be in (0, 1], got {target_density}")
        self.target_density = float(target_density)
        self.growth_check_every = int(growth_check_every)
        self.prune_check_every = int(prune_check_every)
        self.ema_decay = float(ema_decay)
        self.max_change_per_step = int(max_change_per_step)
        self._coact: torch.Tensor | None = None
        self._step = 0

    def observe(self, *, pre: torch.Tensor, post: torch.Tensor, t: float = 0.0) -> None:
        p = pre.detach().reshape(-1, pre.shape[-1]).mean(dim=0)
        q = post.detach().reshape(-1, post.shape[-1]).mean(dim=0)
        update = torch.outer(q, p)
        if self._coact is None:
            self._coact = update.clone()
        else:
            self._coact = self._coact * self.ema_decay + update * (1.0 - self.ema_decay)
        self._step += 1
        self._ready = True

    def compute_delta_W(self) -> torch.Tensor | None:
        # SynaptogenesisGrowPrune produces a MASK delta, not a W delta.
        return None

    def compute_mask_delta(
        self,
        current_mask: torch.Tensor,
        current_W: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return signed int8 mask delta (+1 grow, -1 prune, 0 unchanged)."""
        if self._coact is None:
            return None
        do_grow = (self._step % self.growth_check_every) == 0
        do_prune = (self._step % self.prune_check_every) == 0
        if not (do_grow or do_prune):
            return None
        delta = torch.zeros_like(current_mask, dtype=torch.int8)
        density = current_mask.float().mean().item()
        n_total = current_mask.numel()
        target_n = int(self.target_density * n_total)
        active_n = int(current_mask.sum().item())
        coact_abs = self._coact.abs()
        if do_grow and density < self.target_density:
            n_grow = min(self.max_change_per_step, target_n - active_n)
            if n_grow > 0:
                inactive = ~current_mask.bool()
                scores = coact_abs.masked_fill(~inactive, -float("inf"))
                _, idx = scores.flatten().topk(n_grow)
                delta.view(-1).scatter_(0, idx, 1)
        if do_prune and density > self.target_density:
            n_prune = min(self.max_change_per_step, active_n - target_n)
            if n_prune > 0:
                active = current_mask.bool()
                score = (current_W.detach().abs() * coact_abs).masked_fill(~active, float("inf"))
                _, idx = score.flatten().topk(n_prune, largest=False)
                d_flat = delta.view(-1)
                cur = d_flat.gather(0, idx).to(torch.int8)
                d_flat.scatter_(0, idx, cur - 1)
        return delta

    def reset(self) -> None:
        self._coact = None
        self._step = 0
        super().reset()


# ============================================================================
# PlasticityEngine
# ============================================================================


def _parse_schedule(schedule: str) -> tuple[str, int]:
    """Parse `'every:K'` or `'after_bp'` schedule string."""
    if schedule == "after_bp":
        return ("after_bp", 1)
    if schedule.startswith("every:"):
        try:
            k = int(schedule.split(":", 1)[1])
            if k <= 0:
                raise ValueError
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"invalid schedule {schedule!r}; need 'every:K' K>=1"
            ) from exc
        return ("every", k)
    raise ValueError(f"unknown schedule {schedule!r}; use 'every:K' or 'after_bp'")


class PlasticityEngine:
    """Orchestrates a set of PlasticityRule objects.

    Used by sf.optim (or directly) to compute plasticity weight deltas
    after a BP step. MERGES updates from all rules into a single
    dict[name -> delta], so the caller applies each delta atomically — no
    version conflict because W is mutated EXACTLY once per engine.step.

    Schedule:
        "every:1"   apply on every call (default).
        "every:K"   only emit deltas every K calls; trace continues to grow.
        "after_bp"  documents intent; same as every:1.

    Usage:

        rules = {"layer1.W": STDP(), "layer1.W_hebb": Hebbian(lr=1e-3)}
        engine = PlasticityEngine(rules, schedule="every:1")
        # ... in training loop:
        rules["layer1.W"].observe(pre=h_pre, post=h_post, t=1.0)
        loss.backward(); optimizer.step()
        deltas = engine.step(t=step_idx, weight_dict={"layer1.W": layer.weight})
        engine.apply(deltas, {"layer1.W": layer.weight})
    """

    def __init__(
        self,
        rules: dict[str, PlasticityRule],
        schedule: str = "every:1",
    ) -> None:
        self.rules = dict(rules)
        for name, r in self.rules.items():
            if not isinstance(r, PlasticityRule):
                raise TypeError(
                    f"engine rule {name!r} must inherit PlasticityRule, got "
                    f"{type(r).__name__}"
                )
        self.schedule_kind, self.schedule_k = _parse_schedule(schedule)
        self._step = 0

    def step(
        self,
        t: int,
        weight_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute all rule deltas. Returns {name: delta_tensor}.

        Rules whose schedule says to wait are skipped. SynaptogenesisGrowPrune
        rules surface mask deltas under name + '.mask'.
        """
        self._step += 1
        if self.schedule_kind == "every" and (self._step % self.schedule_k) != 0:
            return {}
        deltas: dict[str, torch.Tensor] = {}
        for name, rule in self.rules.items():
            try:
                if isinstance(rule, SynaptogenesisGrowPrune):
                    if name not in weight_dict:
                        continue
                    W = weight_dict[name]
                    mask_key = name + ".mask"
                    mask = weight_dict.get(mask_key)
                    if mask is None:
                        mask = torch.ones_like(W, dtype=torch.bool)
                    md = rule.compute_mask_delta(mask, W)
                    if md is not None:
                        deltas[mask_key] = md
                else:
                    d = rule.compute_delta_W()
                    if d is not None:
                        deltas[name] = d
            except Exception as exc:
                raise RuntimeError(
                    f"plasticity rule {name!r} failed: {exc}"
                ) from exc
        return deltas

    @torch.no_grad()
    def apply(
        self,
        deltas: dict[str, torch.Tensor],
        weight_dict: dict[str, torch.Tensor],
    ) -> None:
        """Atomically add each delta to its target weight (in-place)."""
        for name, d in deltas.items():
            if name not in weight_dict:
                continue
            W = weight_dict[name]
            d_use = d
            if d_use.dtype != W.dtype:
                d_use = d_use.to(W.dtype)
            if d_use.shape != W.shape:
                raise ValueError(
                    f"delta shape {tuple(d_use.shape)} != weight {name!r} "
                    f"shape {tuple(W.shape)}"
                )
            W.add_(d_use)

    def reset(self) -> None:
        for r in self.rules.values():
            r.reset()
        self._step = 0


# ============================================================================
# Legacy API (sibling-agent compat)
# ============================================================================


class HebbianPlasticity(Module):
    """Hebbian-Oja stable form. Buffer-only (no autograd path).

    Kept as v0.1-compat for synapforge/__init__.py. New code should prefer
    :class:`Hebbian` + :class:`PlasticityEngine`.
    """

    def __init__(self, dim: int, eta: float = 0.01, decay: float = 0.999) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eta = float(eta)
        self.decay = float(decay)
        self.register_buffer("W_fast", torch.zeros(dim, dim))

    @torch.no_grad()
    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        p = pre.detach().reshape(-1, self.dim)
        q = post.detach().reshape(-1, self.dim)
        if p.shape[0] == 0:
            return torch.zeros_like(pre)
        outer_qp = q.t() @ p / p.shape[0]
        oja = self.W_fast * (q.t() @ q / q.shape[0])
        self.W_fast.mul_(self.decay).add_(self.eta * (outer_qp - oja))
        return (pre @ self.W_fast.t()).view_as(pre)

    def as_callback(
        self,
        get_pre: Callable[..., torch.Tensor],
        get_post: Callable[..., torch.Tensor],
    ) -> Callable[..., None]:
        def _cb(module: nn.Module, inputs, outputs) -> None:
            self.forward(
                get_pre(module, inputs, outputs),
                get_post(module, inputs, outputs),
            )

        return _cb
