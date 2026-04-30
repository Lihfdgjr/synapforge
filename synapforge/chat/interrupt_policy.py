"""
InterruptPolicy — speak / interrupt-self / stay-silent decisions.

Real-time conversation has 4 critical decisions per event:

  A. User just typed something. Should I respond now?
     - Respond if turn complete + meaningful content
     - Wait if user still typing (TurnTakingDetector)
     - Backchannel ("嗯", "OK"...) if long monologue but not done

  B. I'm currently generating. New event arrived. Should I interrupt myself?
     - Interrupt if user typed contradicting info
     - Interrupt if user explicitly asked to stop
     - Continue if event is informational (cron tick, log update)
     - Continue if my output is near completion (< N tokens left)

  C. User has been silent for a while. Should I say something proactive?
     - Yes if there's a pending insight from web (proactive_messenger)
     - Yes if user asked me to remind them about X at time Y
     - No if user explicitly muted
     - No if proactive frequency cap exceeded (default: 1/min)

  D. User is in the middle of a sentence. Backchannel?
     - Useful in long-form audio, less so in text. Default: NO backchannel.

Per Anthropic interruptible response design (2024-2025): proactive messages
must be tagged `[Proactive]` so user knows it wasn't requested.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional


class DecisionKind(Enum):
    SPEAK = "speak"
    INTERRUPT_SELF = "interrupt_self"
    SILENCE = "silence"
    BACKCHANNEL = "backchannel"


@dataclass
class SpeakDecision:
    kind: DecisionKind = DecisionKind.SPEAK
    reason: str = ""
    is_proactive: bool = False


@dataclass
class InterruptDecision:
    kind: DecisionKind = DecisionKind.INTERRUPT_SELF
    reason: str = ""


@dataclass
class SilenceDecision:
    kind: DecisionKind = DecisionKind.SILENCE
    reason: str = ""


@dataclass
class _ProactiveHistory:
    sent_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=100))


class InterruptPolicy:
    """Stateful policy. Track recent proactive emissions, user mute state."""

    def __init__(
        self,
        proactive_max_per_hour: int = 30,
        proactive_max_per_minute: int = 1,
        backchannel_enabled: bool = False,
        urgency_interrupt_threshold: float = 0.7,
        near_completion_token_threshold: int = 8,
    ) -> None:
        self.proactive_max_per_hour = proactive_max_per_hour
        self.proactive_max_per_minute = proactive_max_per_minute
        self.backchannel_enabled = backchannel_enabled
        self.urgency_interrupt_threshold = urgency_interrupt_threshold
        self.near_completion_token_threshold = near_completion_token_threshold

        self.history = _ProactiveHistory()
        self.user_muted: bool = False
        self.user_mute_until_ts: float = 0.0
        self.last_user_interaction_ts: float = time.time()

    def set_mute(self, until_seconds_from_now: Optional[float] = None) -> None:
        if until_seconds_from_now is None:
            self.user_muted = True
            self.user_mute_until_ts = float("inf")
        else:
            self.user_mute_until_ts = time.time() + until_seconds_from_now
            self.user_muted = True

    def unmute(self) -> None:
        self.user_muted = False
        self.user_mute_until_ts = 0.0

    @property
    def is_muted(self) -> bool:
        if not self.user_muted:
            return False
        if self.user_mute_until_ts < time.time():
            self.user_muted = False
            self.user_mute_until_ts = 0.0
            return False
        return True

    def note_user_interaction(self) -> None:
        self.last_user_interaction_ts = time.time()

    def can_proactive(self) -> tuple[bool, str]:
        if self.is_muted:
            return False, "user_muted"
        now = time.time()
        recent_minute = sum(1 for t in self.history.sent_timestamps if now - t < 60)
        recent_hour = sum(1 for t in self.history.sent_timestamps if now - t < 3600)
        if recent_minute >= self.proactive_max_per_minute:
            return False, f"per_minute_cap ({self.proactive_max_per_minute})"
        if recent_hour >= self.proactive_max_per_hour:
            return False, f"per_hour_cap ({self.proactive_max_per_hour})"
        return True, ""

    def note_proactive_sent(self) -> None:
        self.history.sent_timestamps.append(time.time())

    def decide_on_user_msg(
        self,
        turn_state: str,
        msg_text: str,
        currently_generating: bool,
    ) -> SpeakDecision | InterruptDecision | SilenceDecision:
        """User just sent (or partially sent) something."""
        self.note_user_interaction()

        if currently_generating:
            urgency = self._compute_user_urgency(msg_text)
            if urgency >= self.urgency_interrupt_threshold:
                return InterruptDecision(reason=f"user_urgency={urgency:.2f}")
            return SilenceDecision(reason="model_currently_speaking")

        if turn_state in ("complete", "explicit_submit"):
            return SpeakDecision(reason="turn_complete")
        if turn_state == "pause":
            if self.backchannel_enabled and len(msg_text) > 80:
                return SpeakDecision(
                    kind=DecisionKind.BACKCHANNEL,
                    reason="long_monologue_pause",
                )
            return SilenceDecision(reason="pause_too_short")
        return SilenceDecision(reason="user_still_typing")

    def decide_on_proactive(
        self,
        urgency: float,
        currently_generating: bool,
        seconds_since_user_idle: float,
    ) -> SpeakDecision | SilenceDecision:
        """A proactive trigger fired (cron / web event / idle)."""
        ok, reason = self.can_proactive()
        if not ok:
            return SilenceDecision(reason=reason)

        if currently_generating:
            return SilenceDecision(reason="busy_generating")

        if seconds_since_user_idle < 10:
            return SilenceDecision(reason="user_recently_active")

        if urgency < 0.3:
            return SilenceDecision(reason=f"low_urgency_{urgency:.2f}")

        return SpeakDecision(
            reason=f"proactive_urgency={urgency:.2f}",
            is_proactive=True,
        )

    def decide_on_self_interrupt(
        self,
        n_tokens_remaining_estimate: int,
        new_event_urgency: float,
    ) -> InterruptDecision | SilenceDecision:
        """I'm generating; new event arrived; should I cut myself off?"""
        if n_tokens_remaining_estimate <= self.near_completion_token_threshold:
            return SilenceDecision(reason="near_completion_let_finish")
        if new_event_urgency >= self.urgency_interrupt_threshold:
            return InterruptDecision(reason=f"new_urgency={new_event_urgency:.2f}")
        return SilenceDecision(reason="event_not_urgent_enough")

    def _compute_user_urgency(self, msg_text: str) -> float:
        urgency = 0.0
        text = msg_text.lower()
        urgency_markers = [
            "stop", "wait", "停", "等等", "别", "no",
            "不对", "错了", "wrong", "actually",
            "!", "！", "?", "？",
        ]
        for marker in urgency_markers:
            if marker in text:
                urgency += 0.2
        if len(msg_text) < 10:
            urgency += 0.2
        return min(1.0, urgency)

    def stats(self) -> dict:
        now = time.time()
        return {
            "muted": self.is_muted,
            "mute_until": self.user_mute_until_ts if self.user_muted else None,
            "proactive_last_minute": sum(1 for t in self.history.sent_timestamps if now - t < 60),
            "proactive_last_hour": sum(1 for t in self.history.sent_timestamps if now - t < 3600),
            "seconds_since_user": now - self.last_user_interaction_ts,
        }
