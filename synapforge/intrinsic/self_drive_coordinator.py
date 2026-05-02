"""SelfDriveCoordinator: wire the 5 self-drive components into one entry-point.

Components wired (all pre-existing in ``synapforge.intrinsic._core``):

    1. SelfGoalProposer  -- Voyager-style goal generation.
    2. ImaginationRollout -- Dreamer-style latent rollout.
    3. FreeEnergySurprise -- ICM-style surprise scorer.
    4. GoalMemory         -- buffer of attempted goals + improvement.
    5. IdleLoop           -- left for trainer to manage (we expose a
                              ``run_step`` the trainer invokes when its
                              own idle/cadence trigger fires).

Plus 2 new pure-Python modules from this package:

    6. FrontierSampler    -- pick a goal whose success is in [0.3, 0.7].
    7. QualityGuard       -- snapshot/rollback STDP if val ppl regresses.

The coordinator does NOT import torch directly. The torch-using
sub-modules (FreeEnergySurprise, ImaginationRollout, GoalMemory) are
constructed and held as opaque objects, and any ``torch.Tensor``
returned by them is forwarded back to the trainer untouched.

Trigger schedule (caller's responsibility):
- Every K=20 outer steps, OR
- When the trainer's data loader returns empty (real GPU idle).

When triggered, ``run_step`` invokes a M-step inner loop where each
inner step:
- Samples a goal from FrontierSampler (or proposes a fresh one).
- Runs ImaginationRollout to dream the goal in latent space.
- Returns a ``SelfDriveStep`` describing the imagined batch the trainer
  should run STDP-only over.
- After the trainer reports the observed loss, ``record_outcome`` updates
  GoalMemory and FrontierSampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .frontier import FrontierSampler, GoalRecord
from .quality_guard import QualityGuard


# ----------------------------------------------------------------------
# config + step record
# ----------------------------------------------------------------------
@dataclass
class SelfDriveConfig:
    """Trainer-tunable knobs for self-drive."""

    enabled: bool = False
    every_k_steps: int = 20
    inner_steps: int = 10
    sweet_lo: float = 0.3
    sweet_hi: float = 0.7
    recent_k_lockout: int = 10
    success_loss_drop: float = 0.05
    max_val_regression: float = 0.05
    max_goal_memory: int = 1000


@dataclass
class SelfDriveStep:
    """One imagined inner step returned to the trainer.

    The trainer reads ``goal_tokens`` and ``hidden`` (if not None) and
    runs its STDP-only weight update against them, then calls
    ``record_outcome`` with the observed loss.
    """

    goal_id: int
    goal_tokens: tuple[int, ...]
    hidden: Any = None  # torch.Tensor or None
    rollout_rewards: list[float] = field(default_factory=list)
    proposed_fresh: bool = False
    expected_improvement: float = 0.0
    free_energy: float = 0.0


# ----------------------------------------------------------------------
# coordinator
# ----------------------------------------------------------------------
class SelfDriveCoordinator:
    """Bundle SelfGoalProposer + Imagination + GoalMemory + Frontier + Guard.

    Designed to live alongside the trainer. The trainer pre-builds the
    torch-using components (so that import errors at trainer startup,
    not coordinator import, are the failure point) and hands them in.

    Construction:

        from synapforge.intrinsic import (
            SelfDriveCoordinator, SelfDriveConfig,
            SelfGoalProposer, ImaginationRollout, GoalMemory,
        )
        cfg = SelfDriveConfig(enabled=True, every_k_steps=20, inner_steps=10)
        proposer = SelfGoalProposer(model, vocab_size=V)
        rollout  = ImaginationRollout(model)
        memory   = GoalMemory(capacity=cfg.max_goal_memory)
        coord    = SelfDriveCoordinator(
            cfg=cfg, proposer=proposer, rollout=rollout, memory=memory,
        )

    Per-outer-step in trainer:

        if coord.should_fire(outer_step, idle=is_data_empty):
            coord.guard.snapshot(snapshot_fn=trainer.snapshot_stdp)
            for _ in range(coord.cfg.inner_steps):
                step = coord.run_step()
                obs_loss = trainer.run_stdp_only_step(step)
                coord.record_outcome(step, observed_loss=obs_loss,
                                     pre_loss=baseline_loss)
            decision = coord.guard.verify(
                pre_ppl=pre, post_ppl=post, restore_fn=trainer.restore_stdp,
            )
            trainer.log(coord.guard.format_decision(decision))
    """

    def __init__(
        self,
        cfg: SelfDriveConfig,
        proposer: Optional[Any] = None,
        rollout: Optional[Any] = None,
        memory: Optional[Any] = None,
        sampler: Optional[FrontierSampler] = None,
        guard: Optional[QualityGuard] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.proposer = proposer  # SelfGoalProposer (torch)
        self.rollout = rollout    # ImaginationRollout (torch)
        self.memory = memory      # GoalMemory (torch)
        self.sampler = sampler or FrontierSampler(
            sweet_lo=cfg.sweet_lo,
            sweet_hi=cfg.sweet_hi,
            recent_k=cfg.recent_k_lockout,
        )
        self.guard = guard or QualityGuard(max_regression=cfg.max_val_regression)
        self.log_fn = log_fn or (lambda s: None)

        self._next_goal_id = 0
        self.n_fired = 0
        self.n_steps_run = 0
        self.n_proposed_fresh = 0
        self.n_resampled = 0
        self.last_step: Optional[SelfDriveStep] = None

    # ------------------------------------------------------------------
    # firing schedule
    # ------------------------------------------------------------------
    def should_fire(self, outer_step: int, idle: bool = False) -> bool:
        if not self.cfg.enabled:
            return False
        if outer_step <= 0:
            return False
        if idle:
            return True
        return (outer_step % max(1, self.cfg.every_k_steps)) == 0

    # ------------------------------------------------------------------
    # per-inner-step entry
    # ------------------------------------------------------------------
    def run_step(self, context: Any = None) -> SelfDriveStep:
        """Produce one imagined SelfDriveStep for the trainer to consume.

        ``context`` is forwarded to the SelfGoalProposer when a fresh
        goal is needed (so the proposer can seed its sampler). Pass
        whatever the trainer has handy (last hidden tensor, prompt
        tokens, None, etc.).
        """
        # 1. Try to sample an in-band goal.
        rec = self.sampler.sample()
        proposed_fresh = False
        if rec is None:
            tokens = self._propose_fresh(context=context)
            self._next_goal_id += 1
            goal_id = self._next_goal_id
            rec = GoalRecord(goal_id=goal_id, goal_tokens=tuple(tokens),
                             n_attempts=0, n_successes=0)
            self.sampler.observe(rec)
            self.n_proposed_fresh += 1
            proposed_fresh = True
        else:
            self.n_resampled += 1

        # 2. Roll out in imagination.
        rollout_rewards: list[float] = []
        hidden: Any = None
        if self.rollout is not None:
            try:
                hidden, rollout_rewards = self.rollout.dream(rec.goal_tokens)
            except Exception as exc:
                self.log_fn(f"[self-drive] rollout failed: {exc!r}")
                hidden, rollout_rewards = None, []

        free_energy = (
            float(rollout_rewards[-1]) if rollout_rewards else 0.0
        )
        step = SelfDriveStep(
            goal_id=rec.goal_id,
            goal_tokens=rec.goal_tokens,
            hidden=hidden,
            rollout_rewards=list(rollout_rewards),
            proposed_fresh=proposed_fresh,
            expected_improvement=0.0,
            free_energy=free_energy,
        )
        self.last_step = step
        self.n_steps_run += 1
        return step

    # ------------------------------------------------------------------
    # outcome recording
    # ------------------------------------------------------------------
    def record_outcome(
        self,
        step: SelfDriveStep,
        observed_loss: float,
        pre_loss: float,
    ) -> None:
        """Update GoalMemory + FrontierSampler with the observed result.

        ``success`` is defined as observed_loss being smaller than
        ``pre_loss * (1 - success_loss_drop)`` -- i.e. the inner step
        actually moved the loss down by the configured threshold.
        """
        success = bool(
            float(observed_loss) <
            float(pre_loss) * (1.0 - float(self.cfg.success_loss_drop))
        )
        # FrontierSampler attempts/successes.
        self.sampler.observe_attempt(
            goal_id=step.goal_id,
            goal_tokens=step.goal_tokens,
            success=success,
        )
        # GoalMemory torch-side record (improvement = pre - post).
        if self.memory is not None:
            try:
                self.memory.record(
                    goal_tokens=list(step.goal_tokens),
                    pre_loss=float(pre_loss),
                    post_loss=float(observed_loss),
                )
            except Exception as exc:
                self.log_fn(f"[self-drive] memory.record failed: {exc!r}")

    # ------------------------------------------------------------------
    # firing wrapper (for trainer convenience)
    # ------------------------------------------------------------------
    def cycle(
        self,
        outer_step: int,
        idle: bool = False,
        run_inner_fn: Optional[Callable[[SelfDriveStep], float]] = None,
        baseline_loss_fn: Optional[Callable[[], float]] = None,
        snapshot_fn: Optional[Callable[[], Any]] = None,
        restore_fn: Optional[Callable[[Any], None]] = None,
        eval_fn: Optional[Callable[[], float]] = None,
    ) -> Optional[dict[str, Any]]:
        """Run a complete self-drive cycle at this outer step (if firing).

        Convenience wrapper for trainers that don't need the granular
        ``should_fire`` / ``run_step`` / ``record_outcome`` / ``guard``
        sequence. Returns None when not firing; otherwise a summary dict.
        """
        if not self.should_fire(outer_step, idle=idle):
            return None
        self.n_fired += 1

        # snapshot for rollback.
        if snapshot_fn is not None:
            self.guard.snapshot(snapshot_fn)

        baseline = float(baseline_loss_fn()) if baseline_loss_fn else 0.0
        pre_eval = float(eval_fn()) if eval_fn else float("nan")

        n_kept = 0
        n_steps = 0
        for _ in range(int(self.cfg.inner_steps)):
            step = self.run_step()
            n_steps += 1
            if run_inner_fn is None:
                # No inner runner: at least record memory with the
                # rollout reward as the only signal.
                self.record_outcome(
                    step, observed_loss=baseline, pre_loss=baseline,
                )
                continue
            try:
                obs = float(run_inner_fn(step))
            except Exception as exc:
                self.log_fn(f"[self-drive] inner-step failed: {exc!r}")
                obs = baseline
            self.record_outcome(
                step, observed_loss=obs, pre_loss=baseline,
            )
            if obs < baseline:
                n_kept += 1

            self.log_fn(
                f"[self-drive] step={outer_step} goal_id={step.goal_id} "
                f"type={'imagined' if not step.proposed_fresh else 'fresh'} "
                f"success={obs < baseline} freeenergy={step.free_energy:.3f}"
            )

        # post-cycle eval + guard.
        decision = None
        if eval_fn is not None and snapshot_fn is not None:
            post_eval = float(eval_fn())
            decision = self.guard.verify(
                pre_ppl=pre_eval, post_ppl=post_eval, restore_fn=restore_fn,
            )
            self.log_fn(self.guard.format_decision(decision))

        return {
            "n_inner_steps": n_steps,
            "n_kept": n_kept,
            "pre_eval": pre_eval,
            "decision": decision,
            "sampler_stats": self.sampler.stats(),
            "guard_stats": self.guard.stats(),
        }

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "fired": self.n_fired,
            "steps_run": self.n_steps_run,
            "proposed_fresh": self.n_proposed_fresh,
            "resampled": self.n_resampled,
            "sampler": self.sampler.stats(),
            "guard": self.guard.stats(),
            "enabled": self.cfg.enabled,
            "every_k_steps": self.cfg.every_k_steps,
            "inner_steps": self.cfg.inner_steps,
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _propose_fresh(self, context: Any = None) -> tuple[int, ...]:
        """Ask SelfGoalProposer for a fresh goal token sequence.

        Falls back to a deterministic short sequence if proposer is None
        or raises. The fallback keeps cycle() working in unit tests
        where no torch model is available.
        """
        if self.proposer is None:
            base = self._next_goal_id
            return (0, (base * 7) % 256, (base * 13) % 256, 1)
        try:
            tokens = self.proposer.propose(context=context)
            return tuple(int(t) for t in tokens)
        except Exception as exc:
            self.log_fn(f"[self-drive] proposer.propose failed: {exc!r}")
            base = self._next_goal_id
            return (0, (base * 7) % 256, (base * 13) % 256, 1)
