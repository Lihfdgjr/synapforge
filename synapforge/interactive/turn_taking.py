"""TurnTakingPolicy — speak / queue / abort / silent decisions.

Stateful policy that tracks who's talking and applies the four rules from
the task spec:

  * Never overlap unless ``model_urgency > interrupt_threshold`` (default
    0.95). Then interrupt politely with an ``[interrupt]`` marker.
  * Wait ``politeness_pause_ms`` (default 300 ms) after user EOT before
    the model speaks.
  * If the user starts speaking while the model is mid-sentence, the
    model immediately calls ``cancel_partial()`` and listens.
  * If the model's proactive trigger fires while the user is mid-typing,
    wait for EOT or pause >2 s.

This module *makes a decision*; the actual cancel / speak is performed
by the kernel using the model's :class:`StreamingGen` handle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TurnAction(Enum):
    SPEAK_NOW = "speak_now"
    QUEUE_FOR_LATER = "queue_for_later"
    ABORT_MODEL = "abort_model"
    INTERRUPT_MODEL = "interrupt_model"
    STAY_SILENT = "stay_silent"


@dataclass
class TurnTakingState:
    user_speaking: bool = False
    model_speaking: bool = False
    model_pending: bool = False
    last_user_token_time: float = 0.0
    last_model_token_time: float = 0.0
    last_user_eot_time: float = 0.0


@dataclass
class TurnDecision:
    action: TurnAction
    reason: str = ""
    interrupt_marker: str = ""
    delay_ms: int = 0


class TurnTakingPolicy:
    """Stateful turn manager. The kernel feeds it events via the
    ``on_*`` methods and asks for decisions via ``decide_*``.

    All thresholds are tunables. Defaults are conservative:
      * ``politeness_pause_ms = 300`` matches typical human turn-taking.
      * ``interrupt_threshold = 0.95`` means the model essentially
        never interrupts; bug bounty / chat alike show users hate it.
      * ``user_typing_grace_s = 2.0`` is the pause before a proactive
        trigger fires if the user is still typing.

    The ``--interrupt-aggressive`` CLI flag flips ``interrupt_threshold``
    down to 0.7, which the spec calls out as a user override.
    """

    def __init__(
        self,
        politeness_pause_ms: int = 300,
        interrupt_threshold: float = 0.95,
        user_typing_grace_s: float = 2.0,
    ) -> None:
        self.politeness_pause_ms = int(politeness_pause_ms)
        self.interrupt_threshold = float(interrupt_threshold)
        self.user_typing_grace_s = float(user_typing_grace_s)
        self.state = TurnTakingState()

    # ------------------------------------------------------------------
    # Inputs from the kernel
    # ------------------------------------------------------------------

    def on_user_partial(self) -> None:
        now = time.time()
        self.state.user_speaking = True
        self.state.last_user_token_time = now

    def on_user_eot(self) -> None:
        now = time.time()
        self.state.user_speaking = False
        self.state.last_user_eot_time = now

    def on_model_token(self) -> None:
        now = time.time()
        self.state.model_speaking = True
        self.state.last_model_token_time = now

    def on_model_finished(self) -> None:
        self.state.model_speaking = False

    def on_model_pending(self, pending: bool = True) -> None:
        self.state.model_pending = bool(pending)

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def decide_on_user_speech(
        self, currently_generating: bool
    ) -> TurnDecision:
        """User just started speaking. The kernel asks: what do I do
        with the current model output?"""
        if currently_generating:
            return TurnDecision(
                action=TurnAction.ABORT_MODEL,
                reason="user_speech_started",
            )
        return TurnDecision(
            action=TurnAction.STAY_SILENT, reason="model_idle"
        )

    def decide_on_user_eot(
        self,
        currently_generating: bool,
    ) -> TurnDecision:
        """User just finished a turn (EOT). Decide whether the model
        should speak after the politeness pause."""
        if currently_generating:
            # Model is still draining a previous response — abort it,
            # then speak (the spec says: user EOT is a hard gate).
            return TurnDecision(
                action=TurnAction.ABORT_MODEL,
                reason="user_eot_while_model_speaking",
                delay_ms=self.politeness_pause_ms,
            )
        return TurnDecision(
            action=TurnAction.SPEAK_NOW,
            reason="user_eot",
            delay_ms=self.politeness_pause_ms,
        )

    def decide_on_proactive(
        self,
        urgency: float,
        currently_generating: bool,
    ) -> TurnDecision:
        """A proactive trigger fired. The kernel asks: speak / queue / silent?

        Rule table (matches docs/ASYNC_INTERACTIVE_KERNEL.md):

        | user_speaking | model_speaking | urgency | action          |
        |---------------|----------------|---------|-----------------|
        | True          | False          | <0.95   | queue_for_later |
        | True          | False          | >=0.95  | queue_for_later |
        | False (typing pause >2s) | False | any | speak_now      |
        | False         | True           | <0.95   | queue_for_later |
        | False         | True           | >=0.95  | interrupt_model |
        | False         | False          | any     | speak_now       |
        """
        urgency = float(urgency)
        now = time.time()
        seconds_since_user = (
            now - self.state.last_user_token_time
            if self.state.last_user_token_time
            else 1e9
        )

        # User mid-typing — wait for EOT or 2s pause.
        if (
            self.state.user_speaking
            and seconds_since_user < self.user_typing_grace_s
        ):
            return TurnDecision(
                action=TurnAction.QUEUE_FOR_LATER,
                reason=(
                    f"user_typing(elapsed={seconds_since_user:.2f}s)"
                ),
            )

        if currently_generating or self.state.model_speaking:
            if urgency >= self.interrupt_threshold:
                return TurnDecision(
                    action=TurnAction.INTERRUPT_MODEL,
                    reason=f"high_urgency={urgency:.2f}",
                    interrupt_marker="[interrupt]",
                )
            return TurnDecision(
                action=TurnAction.QUEUE_FOR_LATER,
                reason=f"model_generating_urgency={urgency:.2f}",
            )

        return TurnDecision(action=TurnAction.SPEAK_NOW, reason="idle_channel")

    # ------------------------------------------------------------------
    # Tunables
    # ------------------------------------------------------------------

    def set_aggressive(self) -> None:
        """``--interrupt-aggressive`` flag — drop threshold to 0.7."""
        self.interrupt_threshold = 0.7

    def stats(self) -> dict:
        now = time.time()
        return {
            "user_speaking": self.state.user_speaking,
            "model_speaking": self.state.model_speaking,
            "model_pending": self.state.model_pending,
            "seconds_since_user_token": (
                now - self.state.last_user_token_time
                if self.state.last_user_token_time
                else 0.0
            ),
            "seconds_since_user_eot": (
                now - self.state.last_user_eot_time
                if self.state.last_user_eot_time
                else 0.0
            ),
            "interrupt_threshold": self.interrupt_threshold,
            "politeness_pause_ms": self.politeness_pause_ms,
        }
