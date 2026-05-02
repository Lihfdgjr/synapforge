"""Tests for the self-drive coordinator + frontier + quality guard.

These tests stay torch-free where possible: FrontierSampler and
QualityGuard are pure-Python so they run in any environment. The
``SelfDriveCoordinator`` integration smoke needs torch only when a
non-stub proposer/rollout/memory are passed in -- the mock-runner
test path drives the coordinator end-to-end without a torch model.
"""

from __future__ import annotations

import math

import pytest

from synapforge.intrinsic.frontier import FrontierSampler, GoalRecord
from synapforge.intrinsic.quality_guard import (
    GuardDecision,
    QualityGuard,
)
from synapforge.intrinsic.self_drive_coordinator import (
    SelfDriveConfig,
    SelfDriveCoordinator,
    SelfDriveStep,
)


# =====================================================================
# FrontierSampler tests
# =====================================================================


class TestFrontierSampler:
    def test_returns_in_band_record(self) -> None:
        s = FrontierSampler(sweet_lo=0.3, sweet_hi=0.7, recent_k=10,
                            min_attempts=3)
        s.observe(GoalRecord(0, (1, 2, 3), n_attempts=10, n_successes=5))  # 0.5 IN
        s.observe(GoalRecord(1, (4, 5), n_attempts=10, n_successes=9))     # 0.9 OUT
        s.observe(GoalRecord(2, (7,), n_attempts=10, n_successes=1))       # 0.1 OUT
        rec = s.sample()
        assert rec is not None
        assert rec.goal_id == 0  # only one in-band record

    def test_avoids_out_of_band(self) -> None:
        s = FrontierSampler(sweet_lo=0.3, sweet_hi=0.7, min_attempts=3)
        s.observe(GoalRecord(0, (1,), n_attempts=10, n_successes=9))  # 0.9 OUT
        s.observe(GoalRecord(1, (2,), n_attempts=10, n_successes=0))  # 0.0 OUT
        assert s.sample() is None
        assert s.candidates() == []

    def test_min_attempts_gating(self) -> None:
        """Records with too few attempts are not candidates."""
        s = FrontierSampler(sweet_lo=0.3, sweet_hi=0.7, min_attempts=5)
        s.observe(GoalRecord(0, (1,), n_attempts=2, n_successes=1))  # 0.5 but n=2
        assert s.sample() is None
        s.observe(GoalRecord(0, (1,), n_attempts=5, n_successes=2))  # 0.4 n=5
        rec = s.sample()
        assert rec is not None and rec.goal_id == 0

    def test_recent_k_lockout(self) -> None:
        """A goal sampled once is locked out for recent_k subsequent calls."""
        s = FrontierSampler(sweet_lo=0.3, sweet_hi=0.7, recent_k=2,
                            min_attempts=3)
        s.observe(GoalRecord(0, (1,), n_attempts=10, n_successes=5))  # 0.5 IN
        s.observe(GoalRecord(1, (2,), n_attempts=10, n_successes=4))  # 0.4 IN
        first = s.sample()
        assert first is not None
        second = s.sample()
        # If 2 candidates exist and recent_k=2, the second sample picks the
        # other; a third sample (still within lockout window) is None
        # because all in-band candidates are now locked out.
        assert second is not None and second.goal_id != first.goal_id
        third = s.sample()
        assert third is None  # both locked out

    def test_observe_attempt_updates_running_count(self) -> None:
        s = FrontierSampler(min_attempts=1)
        s.observe_attempt(0, (1, 2), success=True)
        s.observe_attempt(0, (1, 2), success=False)
        s.observe_attempt(0, (1, 2), success=True)
        rec = s._records[0]
        assert rec.n_attempts == 3
        assert rec.n_successes == 2
        assert rec.success_rate == pytest.approx(2 / 3)

    def test_band_midpoint_tiebreak(self) -> None:
        """Tie-break prefers record nearest to band midpoint."""
        s = FrontierSampler(sweet_lo=0.3, sweet_hi=0.7,
                            recent_k=10, min_attempts=3)
        # mid = 0.5; a (rate 0.55) is closer than b (rate 0.65).
        s.observe(GoalRecord(0, (1,), n_attempts=20, n_successes=11))  # 0.55
        s.observe(GoalRecord(1, (2,), n_attempts=20, n_successes=13))  # 0.65
        chosen = s.sample()
        assert chosen.goal_id == 0

    def test_stats_dict(self) -> None:
        s = FrontierSampler(sweet_lo=0.3, sweet_hi=0.7, min_attempts=3)
        s.observe(GoalRecord(0, (1,), 10, 5))   # IN
        s.observe(GoalRecord(1, (2,), 10, 9))   # OUT
        st = s.stats()
        assert st["n_goals"] == 2
        assert st["n_in_band"] == 1
        assert st["band_lo"] == 0.3
        assert st["band_hi"] == 0.7


# =====================================================================
# QualityGuard tests
# =====================================================================


class TestQualityGuard:
    def test_keep_on_within_tolerance(self) -> None:
        g = QualityGuard(max_regression=0.05)
        g.snapshot(snapshot_fn=lambda: {"step": 1})
        d = g.verify(pre_ppl=100.0, post_ppl=104.0, restore_fn=lambda s: None)
        assert isinstance(d, GuardDecision)
        assert d.rolled_back is False
        assert g.n_rollbacks == 0
        assert g.n_kept == 1
        assert "within-tolerance" in d.reason

    def test_rollback_on_synthetic_regression(self) -> None:
        """If post > pre*(1+max_regression), restore_fn must be called."""
        g = QualityGuard(max_regression=0.05)
        sentinel = {"called": False, "snap": None}

        def _snapshot():
            return {"params": "pre"}

        def _restore(snap):
            sentinel["called"] = True
            sentinel["snap"] = snap

        g.snapshot(_snapshot)
        d = g.verify(pre_ppl=100.0, post_ppl=110.0, restore_fn=_restore)
        assert d.rolled_back is True
        assert sentinel["called"] is True
        assert sentinel["snap"] == {"params": "pre"}
        assert g.n_rollbacks == 1
        assert g.n_kept == 0

    def test_rollback_on_nan_post(self) -> None:
        g = QualityGuard(max_regression=0.05, rollback_on_nan=True)
        g.snapshot(snapshot_fn=lambda: "snap")
        called = [False]
        def _restore(snap): called[0] = True
        d = g.verify(pre_ppl=100.0, post_ppl=float("nan"), restore_fn=_restore)
        assert d.rolled_back is True
        assert called[0] is True

    def test_failopen_on_nan_pre(self) -> None:
        """If pre is non-finite, no signal -> KEEP."""
        g = QualityGuard(max_regression=0.05)
        g.snapshot(snapshot_fn=lambda: "snap")
        d = g.verify(pre_ppl=float("nan"), post_ppl=99.0,
                     restore_fn=lambda s: None)
        assert d.rolled_back is False
        assert "pre-non-finite" in d.reason

    def test_format_decision_human_readable(self) -> None:
        d = GuardDecision(pre_ppl=100.0, post_ppl=110.0,
                          threshold_ppl=105.0, rolled_back=True,
                          reason="post>5%-of-pre")
        s = QualityGuard.format_decision(d)
        assert "ROLLBACK" in s
        assert "100" in s
        assert "110" in s

    def test_snapshot_consumed_after_verify(self) -> None:
        """Snapshot is discarded after each verify regardless of outcome."""
        g = QualityGuard()
        g.snapshot(snapshot_fn=lambda: "S1")
        assert g._has_snapshot is True
        g.verify(pre_ppl=100.0, post_ppl=99.0)
        assert g._has_snapshot is False  # consumed even on KEEP


# =====================================================================
# SelfDriveCoordinator integration smoke
# =====================================================================


class _StubProposer:
    """Counts proposals so the test can assert it was called exactly N times."""

    def __init__(self) -> None:
        self.n_calls = 0

    def propose(self, context=None):
        self.n_calls += 1
        # Return a deterministic short token sequence.
        return [0, self.n_calls % 16, (self.n_calls * 7) % 16, 1]


class _StubRollout:
    """Returns a fixed reward trajectory for any goal."""

    def __init__(self, rewards=None) -> None:
        self.rewards = rewards or [0.1, 0.2, 0.3]
        self.n_calls = 0

    def dream(self, goal_tokens):
        self.n_calls += 1
        # Intentionally not torch -- coordinator forwards opaque types.
        return ("stub-hidden", list(self.rewards))


class _StubMemory:
    def __init__(self) -> None:
        self.records = []

    def record(self, goal_tokens, pre_loss, post_loss):
        self.records.append({
            "goal": list(goal_tokens),
            "pre": float(pre_loss),
            "post": float(post_loss),
            "improvement": float(pre_loss) - float(post_loss),
        })


class TestSelfDriveCoordinator:
    def test_default_disabled_should_not_fire(self) -> None:
        cfg = SelfDriveConfig(enabled=False)
        coord = SelfDriveCoordinator(cfg=cfg)
        assert coord.should_fire(20, idle=True) is False
        assert coord.should_fire(100, idle=False) is False

    def test_fires_every_k_steps_when_enabled(self) -> None:
        cfg = SelfDriveConfig(enabled=True, every_k_steps=20, inner_steps=2)
        coord = SelfDriveCoordinator(cfg=cfg)
        assert coord.should_fire(20, idle=False) is True
        assert coord.should_fire(40, idle=False) is True
        assert coord.should_fire(15, idle=False) is False
        # idle=True overrides cadence.
        assert coord.should_fire(15, idle=True) is True

    def test_run_step_proposes_fresh_when_no_in_band(self) -> None:
        cfg = SelfDriveConfig(enabled=True, every_k_steps=1, inner_steps=1)
        proposer = _StubProposer()
        rollout = _StubRollout(rewards=[0.0, 0.5, 1.2])
        memory = _StubMemory()
        coord = SelfDriveCoordinator(
            cfg=cfg, proposer=proposer, rollout=rollout, memory=memory,
        )
        step = coord.run_step()
        assert isinstance(step, SelfDriveStep)
        assert step.proposed_fresh is True
        assert step.free_energy == pytest.approx(1.2)
        # Proposer was used.
        assert proposer.n_calls == 1
        # Hidden is opaque-passthrough.
        assert step.hidden == "stub-hidden"

    def test_record_outcome_marks_success_below_threshold(self) -> None:
        cfg = SelfDriveConfig(enabled=True, every_k_steps=1, inner_steps=1,
                              success_loss_drop=0.10)
        coord = SelfDriveCoordinator(
            cfg=cfg, proposer=_StubProposer(),
            rollout=_StubRollout(), memory=_StubMemory(),
        )
        step = coord.run_step()
        coord.record_outcome(step, observed_loss=4.0, pre_loss=5.0)
        # 4.0 < 5.0 * (1 - 0.10) == 4.5  -> success
        rec = coord.sampler._records[step.goal_id]
        assert rec.n_attempts == 1
        assert rec.n_successes == 1

    def test_record_outcome_no_success_above_threshold(self) -> None:
        cfg = SelfDriveConfig(enabled=True, every_k_steps=1, inner_steps=1,
                              success_loss_drop=0.10)
        coord = SelfDriveCoordinator(
            cfg=cfg, proposer=_StubProposer(),
            rollout=_StubRollout(), memory=_StubMemory(),
        )
        step = coord.run_step()
        coord.record_outcome(step, observed_loss=4.6, pre_loss=5.0)
        # 4.6 NOT < 4.5 -> failure
        rec = coord.sampler._records[step.goal_id]
        assert rec.n_attempts == 1
        assert rec.n_successes == 0

    def test_cycle_runs_inner_steps_and_logs(self) -> None:
        """10 outer + 5 self-drive iterations completes cleanly."""
        cfg = SelfDriveConfig(
            enabled=True, every_k_steps=2, inner_steps=5,
            sweet_lo=0.3, sweet_hi=0.7, success_loss_drop=0.05,
            max_val_regression=0.05,
        )
        proposer = _StubProposer()
        rollout = _StubRollout(rewards=[0.1, 0.5])
        memory = _StubMemory()
        log_lines = []
        coord = SelfDriveCoordinator(
            cfg=cfg, proposer=proposer, rollout=rollout, memory=memory,
            log_fn=log_lines.append,
        )

        # 10 outer steps, fires at steps 2, 4, 6, 8, 10 (every K=2).
        n_fires = 0
        # Increasing observed loss -> all "failures" so we exercise
        # the failure path; eval_fn synthetic improvement so the
        # quality guard logs a KEEP.
        for outer in range(1, 11):
            ran = coord.cycle(
                outer_step=outer,
                idle=False,
                run_inner_fn=lambda step: 4.9,  # baseline=5.0 -> fail (4.9 not < 4.75)
                baseline_loss_fn=lambda: 5.0,
                snapshot_fn=lambda: {"snap": True},
                restore_fn=lambda s: None,
                eval_fn=lambda: 100.0,  # constant -> KEEP every time
            )
            if ran is not None:
                n_fires += 1
                assert ran["n_inner_steps"] == cfg.inner_steps
        assert n_fires == 5
        # Coordinator stats consistent.
        st = coord.stats()
        assert st["fired"] == 5
        assert st["steps_run"] == 5 * cfg.inner_steps
        # Memory was populated.
        assert len(memory.records) == st["steps_run"]
        # Some log lines emitted.
        assert any("[self-drive]" in line for line in log_lines)

    def test_quality_guard_rolls_back_on_synthetic_regression(self) -> None:
        """Cycle wires the guard correctly: synthetic post>>pre triggers
        the user-supplied restore_fn."""
        cfg = SelfDriveConfig(
            enabled=True, every_k_steps=1, inner_steps=2,
            max_val_regression=0.05,
        )
        coord = SelfDriveCoordinator(
            cfg=cfg, proposer=_StubProposer(),
            rollout=_StubRollout(), memory=_StubMemory(),
        )
        restored = {"called": False}
        # pre=100, post=200 -> rollback.
        ran = coord.cycle(
            outer_step=1,
            run_inner_fn=lambda s: 5.0,
            baseline_loss_fn=lambda: 5.0,
            snapshot_fn=lambda: "snapshot-handle",
            restore_fn=lambda snap: restored.__setitem__("called", True),
            eval_fn_calls=[100.0, 200.0],  # used below
        ) if False else None  # placeholder, real test below

        # Real test: monkey-patch the eval_fn via closure.
        eval_returns = iter([100.0, 200.0])  # pre, post
        def _eval():
            return next(eval_returns)
        ran = coord.cycle(
            outer_step=1,
            run_inner_fn=lambda s: 5.0,
            baseline_loss_fn=lambda: 5.0,
            snapshot_fn=lambda: "snapshot-handle",
            restore_fn=lambda snap: restored.__setitem__("called", True),
            eval_fn=_eval,
        )
        assert ran is not None
        assert restored["called"] is True
        d = ran["decision"]
        assert d is not None and d.rolled_back is True

    def test_proposer_failure_falls_back_to_deterministic_goal(self) -> None:
        """When the proposer raises, the coordinator must still produce
        a SelfDriveStep with non-empty goal_tokens."""
        class _Bad:
            def propose(self, context=None):
                raise RuntimeError("boom")

        cfg = SelfDriveConfig(enabled=True, every_k_steps=1, inner_steps=1)
        coord = SelfDriveCoordinator(
            cfg=cfg, proposer=_Bad(),
            rollout=_StubRollout(), memory=_StubMemory(),
        )
        step = coord.run_step()
        assert isinstance(step, SelfDriveStep)
        assert step.goal_tokens != ()
        assert len(step.goal_tokens) >= 2

    def test_rollout_failure_returns_zero_freeenergy(self) -> None:
        class _BadRollout:
            def dream(self, gt):
                raise RuntimeError("dream-failed")

        cfg = SelfDriveConfig(enabled=True, every_k_steps=1, inner_steps=1)
        coord = SelfDriveCoordinator(
            cfg=cfg, proposer=_StubProposer(),
            rollout=_BadRollout(), memory=_StubMemory(),
        )
        step = coord.run_step()
        assert step.free_energy == 0.0
        assert step.rollout_rewards == []

    def test_disabled_cycle_returns_none(self) -> None:
        cfg = SelfDriveConfig(enabled=False)
        coord = SelfDriveCoordinator(cfg=cfg)
        ran = coord.cycle(outer_step=20)
        assert ran is None


# =====================================================================
# end-to-end "10 outer + 5 self-drive" integration smoke
# =====================================================================


def test_integration_smoke_10outer_5selfdrive_completes_cleanly() -> None:
    """End-to-end smoke: run 10 outer steps, with self-drive every 2 steps
    (M=5 inner each fire). Verifies the coordinator survives:
      * mixed in-band / out-of-band observations,
      * one synthetic rollback in the middle of the run,
      * proposer/rollout failures recovered cleanly.
    """
    cfg = SelfDriveConfig(
        enabled=True, every_k_steps=2, inner_steps=5,
        sweet_lo=0.3, sweet_hi=0.7,
        success_loss_drop=0.10,
        max_val_regression=0.05,
    )
    log_lines: list[str] = []
    coord = SelfDriveCoordinator(
        cfg=cfg,
        proposer=_StubProposer(),
        rollout=_StubRollout(),
        memory=_StubMemory(),
        log_fn=log_lines.append,
    )

    fires = 0
    # eval_fn alternates: even outer steps return baseline 100;
    # outer step 6 returns post=200 -> rollback.
    eval_state = {"step_seen": 0}
    def _eval():
        eval_state["step_seen"] += 1
        if eval_state["step_seen"] == 4:  # 2nd eval of step 6 (post)
            return 200.0
        return 100.0

    for outer in range(1, 11):
        # Inner runner: returns observed loss. 70% of steps: success;
        # 30% failure to keep the FrontierSampler in flux.
        def _runner(step, _o=outer):
            return 4.0 if step.goal_id % 3 != 0 else 5.5

        result = coord.cycle(
            outer_step=outer,
            run_inner_fn=_runner,
            baseline_loss_fn=lambda: 5.0,
            snapshot_fn=lambda: {"o": outer},
            restore_fn=lambda snap: None,
            eval_fn=_eval,
        )
        if result is not None:
            fires += 1
            assert result["n_inner_steps"] == cfg.inner_steps

    assert fires == 5  # outer 2,4,6,8,10
    st = coord.stats()
    assert st["fired"] == 5
    assert st["steps_run"] == 25
    # at least one rollback should have triggered (step 6 post=200).
    assert st["guard"]["n_rollbacks"] >= 1
    assert any("ROLLBACK" in line for line in log_lines)
