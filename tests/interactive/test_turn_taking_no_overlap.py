"""Tests for synapforge.interactive.turn_taking.TurnTakingPolicy.

Verifies the four spec-mandated rules:
  1. Never overlap unless model_urgency > interrupt_threshold.
  2. Wait politeness_pause_ms after user EOT before model speaks.
  3. User starts speaking while model speaking → model aborts.
  4. Proactive trigger while user is mid-typing → queue for later.
"""

from __future__ import annotations

import time

import pytest

from synapforge.interactive.turn_taking import (
    TurnAction,
    TurnTakingPolicy,
)


def test_user_speech_aborts_active_model():
    policy = TurnTakingPolicy()
    decision = policy.decide_on_user_speech(currently_generating=True)
    assert decision.action == TurnAction.ABORT_MODEL


def test_user_speech_no_op_when_model_idle():
    policy = TurnTakingPolicy()
    decision = policy.decide_on_user_speech(currently_generating=False)
    assert decision.action == TurnAction.STAY_SILENT


def test_user_eot_speak_now_with_politeness_pause():
    policy = TurnTakingPolicy(politeness_pause_ms=300)
    decision = policy.decide_on_user_eot(currently_generating=False)
    assert decision.action == TurnAction.SPEAK_NOW
    assert decision.delay_ms == 300


def test_user_eot_aborts_then_speaks():
    """If model is mid-stream when EOT arrives, abort first."""
    policy = TurnTakingPolicy()
    decision = policy.decide_on_user_eot(currently_generating=True)
    assert decision.action == TurnAction.ABORT_MODEL


def test_proactive_default_threshold_does_not_interrupt():
    """At the default 0.95 threshold, urgency 0.9 cannot interrupt."""
    policy = TurnTakingPolicy(interrupt_threshold=0.95)
    decision = policy.decide_on_proactive(
        urgency=0.9, currently_generating=True
    )
    assert decision.action == TurnAction.QUEUE_FOR_LATER


def test_proactive_interrupts_only_above_threshold():
    """Urgency must exceed interrupt_threshold to actually interrupt."""
    policy = TurnTakingPolicy(interrupt_threshold=0.95)
    decision = policy.decide_on_proactive(
        urgency=0.97, currently_generating=True
    )
    assert decision.action == TurnAction.INTERRUPT_MODEL
    assert decision.interrupt_marker == "[interrupt]"


def test_proactive_queues_while_user_mid_typing():
    """User actively typing → proactive trigger goes to queue."""
    policy = TurnTakingPolicy(user_typing_grace_s=2.0)
    policy.on_user_partial()  # user started typing now
    decision = policy.decide_on_proactive(
        urgency=0.5, currently_generating=False
    )
    assert decision.action == TurnAction.QUEUE_FOR_LATER


def test_proactive_speaks_after_user_typing_grace():
    """User typed >2 s ago → proactive can speak."""
    policy = TurnTakingPolicy(user_typing_grace_s=0.01)
    policy.on_user_partial()
    time.sleep(0.05)  # exceed user_typing_grace_s
    decision = policy.decide_on_proactive(
        urgency=0.5, currently_generating=False
    )
    assert decision.action == TurnAction.SPEAK_NOW


def test_aggressive_flag_lowers_threshold():
    """--interrupt-aggressive drops threshold to 0.7."""
    policy = TurnTakingPolicy(interrupt_threshold=0.95)
    policy.set_aggressive()
    assert policy.interrupt_threshold == 0.7
    decision = policy.decide_on_proactive(
        urgency=0.75, currently_generating=True
    )
    assert decision.action == TurnAction.INTERRUPT_MODEL


def test_user_and_model_never_overlap_default():
    """Exactly one of (user_speaking, model_speaking) wins per rule."""
    policy = TurnTakingPolicy()
    policy.on_user_partial()
    # User speech beats model — model must abort.
    d = policy.decide_on_user_speech(currently_generating=True)
    assert d.action == TurnAction.ABORT_MODEL


def test_state_transitions_track_speakers():
    policy = TurnTakingPolicy()
    assert policy.state.user_speaking is False
    assert policy.state.model_speaking is False
    policy.on_user_partial()
    assert policy.state.user_speaking is True
    policy.on_user_eot()
    assert policy.state.user_speaking is False
    policy.on_model_token()
    assert policy.state.model_speaking is True
    policy.on_model_finished()
    assert policy.state.model_speaking is False
