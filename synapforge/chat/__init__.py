"""
Async conversation kernel — like a real person, not request-response.

Modules:
  event_loop.py        ConversationKernel main async loop
  interrupt_policy.py  When to speak, interrupt, stay silent
  streaming.py         Cancellable token-by-token generation
  turn_taking.py       Detect partial vs complete user message
  proactive.py         Outbound messages from cron / web / idle

Pattern (replaces synchronous request-response):

  inbox queue ◀──── user types char-by-char (partial msg events)
              ◀──── cron tick (idle / time-of-day / web event)
              ◀──── internal IdleLoop self-trigger

  ConversationKernel
    ├── TurnTakingDetector: is the user done typing?
    ├── InterruptPolicy: should I speak now?
    └── StreamingGenerator: emit tokens; can be cancelled mid-stream

  outbox queue ───▶ user sees streaming reply
                 ───▶ proactive message arrives unbidden
                 ───▶ self-interrupt: model stops mid-sentence
                       to address something more urgent

Industry precedent: Anthropic's *interruptible response* design (2024-2025
client UX), real-time agents (Inflection Pi, OpenAI Realtime API).
"""

from __future__ import annotations

from .event_loop import ConversationKernel, ConversationEvent, EventKind
from .interrupt_policy import (
    InterruptPolicy,
    SpeakDecision,
    InterruptDecision,
    SilenceDecision,
)
from .streaming import StreamingGenerator, GenerationHandle, CancellationToken
from .turn_taking import TurnTakingDetector, TurnState
from .proactive import ProactiveMessenger, ProactiveTrigger

__all__ = [
    "ConversationKernel",
    "ConversationEvent",
    "EventKind",
    "InterruptPolicy",
    "SpeakDecision",
    "InterruptDecision",
    "SilenceDecision",
    "StreamingGenerator",
    "GenerationHandle",
    "CancellationToken",
    "TurnTakingDetector",
    "TurnState",
    "ProactiveMessenger",
    "ProactiveTrigger",
]
