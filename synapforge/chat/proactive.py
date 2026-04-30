"""
ProactiveMessenger — outbound triggers that come from outside the user.

Sources of "I should say something unprompted":

  1. **IdleLoop**: user has been quiet for N minutes; model has something
     interesting from intrinsic motivation (FreeEnergy / Novelty)
  2. **Cron / time-of-day**: user said "remind me at 3pm"; the timer fires
  3. **Web event**: autonomous_daemon found something the user cares about
     (matches user's recent topics in retrieval cache)
  4. **System notification**: training crashed, ckpt saved, etc.
  5. **Relevance bump**: user said something at T-100 turns and now
     contextually related info is relevant again

Each trigger has urgency [0, 1]. InterruptPolicy.decide_on_proactive() decides
whether to actually emit, given mute state, frequency caps, recent activity.

ALL proactive messages tagged `[Proactive]` per Anthropic interruptible
response design.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, List, Optional


class TriggerSource(Enum):
    IDLE_LOOP = "idle_loop"
    CRON = "cron"
    WEB_EVENT = "web_event"
    SYSTEM = "system"
    RELEVANCE_BUMP = "relevance_bump"


@dataclass
class ProactiveTrigger:
    source: TriggerSource
    urgency: float
    suggested_message: str = ""
    payload: dict = field(default_factory=dict)
    fire_after_ts: float = field(default_factory=time.time)
    handled: bool = False


class IdleLoopTrigger:
    """Fires when user has been silent for N minutes AND model has high
    intrinsic-motivation signal (FreeEnergy / Novelty above threshold).

    Drives the "agent has its own goals" behavior — the model thinks
    autonomously and surfaces interesting findings.
    """

    def __init__(
        self,
        idle_threshold_seconds: float = 600.0,  # 10 min
        check_interval_seconds: float = 60.0,
    ) -> None:
        self.idle_threshold_seconds = idle_threshold_seconds
        self.check_interval_seconds = check_interval_seconds
        self._last_proposal_ts: float = 0.0
        self._consecutive_idle_checks: int = 0

    def should_fire(self, seconds_since_user: float) -> bool:
        if seconds_since_user < self.idle_threshold_seconds:
            self._consecutive_idle_checks = 0
            return False
        if time.time() - self._last_proposal_ts < self.idle_threshold_seconds:
            return False
        self._consecutive_idle_checks += 1
        return self._consecutive_idle_checks >= 1

    def note_fired(self) -> None:
        self._last_proposal_ts = time.time()


class WebEventTrigger:
    """Fires when autonomous_daemon's web_cache.jsonl gets new entries that
    match terms in the user's recent retrieval cache."""

    def __init__(self, relevance_threshold: float = 0.5) -> None:
        self.relevance_threshold = relevance_threshold

    def maybe_trigger(
        self,
        new_entry: dict,
        user_recent_terms: List[str],
    ) -> Optional[ProactiveTrigger]:
        text = (new_entry.get("a") or "") + " " + (new_entry.get("q") or "")
        if not text or not user_recent_terms:
            return None
        text_lower = text.lower()
        hits = sum(1 for t in user_recent_terms if t.lower() in text_lower)
        relevance = hits / max(len(user_recent_terms), 1)
        if relevance < self.relevance_threshold:
            return None
        return ProactiveTrigger(
            source=TriggerSource.WEB_EVENT,
            urgency=min(1.0, relevance + 0.2),
            suggested_message=f"我刚看到一个相关的: {text[:120]}",
            payload={"web_entry": new_entry, "matched_terms": hits},
        )


class CronTrigger:
    """User said 'remind me at 3pm'. Trigger fires at scheduled time."""

    def __init__(self) -> None:
        self.scheduled: List[ProactiveTrigger] = []

    def schedule(
        self,
        fire_at_ts: float,
        message: str,
        urgency: float = 0.5,
    ) -> None:
        self.scheduled.append(
            ProactiveTrigger(
                source=TriggerSource.CRON,
                urgency=urgency,
                suggested_message=message,
                fire_after_ts=fire_at_ts,
            )
        )

    def fire_due(self) -> List[ProactiveTrigger]:
        now = time.time()
        ready = [t for t in self.scheduled if t.fire_after_ts <= now and not t.handled]
        for t in ready:
            t.handled = True
        return ready


class ProactiveMessenger:
    """Runs all sub-triggers and surfaces ProactiveTrigger objects to the
    ConversationKernel. Kernel decides actual emission via InterruptPolicy.
    """

    def __init__(
        self,
        idle_threshold_seconds: float = 600.0,
        web_relevance_threshold: float = 0.5,
    ) -> None:
        self.idle = IdleLoopTrigger(idle_threshold_seconds=idle_threshold_seconds)
        self.web = WebEventTrigger(relevance_threshold=web_relevance_threshold)
        self.cron = CronTrigger()
        self._user_recent_terms: List[str] = []
        self._goal_proposer: Optional[Callable[[], str]] = None

    def set_goal_proposer(self, fn: Callable[[], str]) -> None:
        """Pluggable: called when IdleLoop fires to ask 'what would the
        model say if it had to speak now?'"""
        self._goal_proposer = fn

    def update_user_recent_terms(self, terms: List[str]) -> None:
        self._user_recent_terms = terms[-50:]

    def poll(self, seconds_since_user: float) -> List[ProactiveTrigger]:
        """Single tick: check all sub-triggers, return list of ready triggers."""
        out: List[ProactiveTrigger] = []

        out.extend(self.cron.fire_due())

        if self.idle.should_fire(seconds_since_user):
            msg = ""
            if self._goal_proposer:
                try:
                    msg = self._goal_proposer()
                except Exception:
                    msg = ""
            urgency = 0.4
            out.append(ProactiveTrigger(
                source=TriggerSource.IDLE_LOOP,
                urgency=urgency,
                suggested_message=msg,
            ))
            self.idle.note_fired()

        return out

    def on_web_event(self, new_entry: dict) -> Optional[ProactiveTrigger]:
        return self.web.maybe_trigger(new_entry, self._user_recent_terms)
