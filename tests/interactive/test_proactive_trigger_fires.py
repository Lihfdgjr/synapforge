"""Tests for synapforge.interactive.proactive_trigger.

Covers the three rule paths:
  * Curiosity P95 — synthetic high curiosity score → trigger fires.
  * GoalMemory — high-improvement record → trigger fires.
  * Silence-relevance — quiet user + matching terms → trigger fires.
"""

from __future__ import annotations

import time

import pytest

from synapforge.interactive.proactive_trigger import (
    ProactiveTrigger,
    TriggerSource,
)


def test_curiosity_p95_fires_trigger():
    """Push 49 small scores + 1 huge score → trigger fires at next tick."""
    trigger = ProactiveTrigger(
        check_interval_steps=1,  # tick every call
        curiosity_p95_window=50,
    )

    # 49 baseline scores + 1 outlier well above P95.
    for _ in range(49):
        trigger.push_curiosity(0.10)
    trigger.push_curiosity(0.95)

    out = trigger.tick()
    cur = [t for t in out if t.source == TriggerSource.CURIOSITY]
    assert len(cur) == 1, f"expected curiosity trigger, got {out!r}"
    assert cur[0].urgency >= 0.6
    assert cur[0].confidence >= 0.7  # confidence floor enforced
    assert cur[0].relevance >= 0.5


def test_curiosity_low_history_no_trigger():
    """Too few samples → no trigger (avoids early false positives)."""
    trigger = ProactiveTrigger(
        check_interval_steps=1,
        curiosity_p95_window=50,
    )
    for _ in range(5):
        trigger.push_curiosity(0.95)
    out = trigger.tick()
    assert all(t.source != TriggerSource.CURIOSITY for t in out)


def test_goal_memory_fires_trigger():
    """A goal record with high improvement → goal_memory trigger."""

    class _FakeGoalMemory:
        def __init__(self, records):
            self._records = records

    records = [
        {"goal": [1, 2, 3], "pre": 1.0, "post": 0.5, "improvement": 0.5},
        {"goal": [4, 5], "pre": 1.0, "post": 0.95, "improvement": 0.05},
    ]
    fake = _FakeGoalMemory(records)
    trigger = ProactiveTrigger(check_interval_steps=1)
    out = trigger.tick(goal_memory=fake)
    goal_evs = [t for t in out if t.source == TriggerSource.GOAL_MEMORY]
    assert len(goal_evs) == 1
    assert goal_evs[0].goal_id == 0  # the high-improvement one
    assert goal_evs[0].confidence >= 0.7

    # Re-firing on the same record is suppressed.
    out2 = trigger.tick(goal_memory=fake)
    goal_evs2 = [t for t in out2 if t.source == TriggerSource.GOAL_MEMORY]
    assert goal_evs2 == [], "goal trigger should not re-fire on same record"


def test_silence_relevance_fires_trigger():
    """Long silence + relevant terms → silence_relevance trigger."""
    trigger = ProactiveTrigger(
        check_interval_steps=1,
        silence_threshold_s=0.01,  # immediate
        silence_relevance_threshold=0.1,
    )
    # Record a recent user interaction long enough ago to clear the threshold.
    trigger.note_user_interaction(ts=time.time() - 1.0)
    out = trigger.tick(
        recent_user_terms=["alpha", "beta", "gamma"],
        h_t_summary="alpha beta is interesting",
        idle_thought_proposer=lambda: "By the way, alpha-beta...",
    )
    sil = [t for t in out if t.source == TriggerSource.SILENCE_RELEVANCE]
    assert len(sil) == 1
    assert sil[0].suggested_text.startswith("By the way")
    assert sil[0].confidence >= 0.7
    assert sil[0].relevance >= 0.5


def test_quality_guards_drop_low_confidence():
    """Low-confidence/low-relevance triggers are filtered out."""
    trigger = ProactiveTrigger(
        check_interval_steps=1,
        min_confidence=0.95,
        min_relevance=0.95,
        curiosity_p95_window=10,
    )
    # Build a curiosity event that would naturally have confidence/relevance
    # well below the very-high min thresholds.
    for _ in range(10):
        trigger.push_curiosity(0.10)
    trigger.push_curiosity(0.20)  # mild excess
    out = trigger.tick()
    assert out == []


def test_check_interval_steps():
    """Trigger tick body only runs every K steps."""
    trigger = ProactiveTrigger(check_interval_steps=5)
    # 4 ticks → no rule evaluation
    for _ in range(4):
        assert trigger.tick() == []
    # 5th tick evaluates rules but with no data → still empty
    assert trigger.tick() == []
