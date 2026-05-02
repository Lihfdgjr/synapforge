"""synapforge.interactive — async chat kernel (streaming + listening + interrupt + proactive).

Why this is a strict win for LNN+SNN
------------------------------------
Continuous-time CfC + event-driven PLIF are *natively* streaming:

* No KV cache to grow — per-token forward is O(1) at any context length.
* Per-token hidden state ``(h_cfc, v_plif)`` fully summarises the
  conversation so far; we can pause / resume / cancel cleanly at any
  token boundary.
* Pace control is automatic: the model can emit one token per asyncio
  scheduler tick without backpressure.

This is the "real-person" chat path the task spec asks for:

  inbox  ─▶ TurnTakingPolicy  ─▶ InterruptPolicy / Proactive Trigger  ─▶ StreamingGen
                                                                          │
                                                                          ▼
                                                                    cancellable
                                                                    token stream

Hard constraints (per task spec):
  * 0 imports of ``torch`` at module scope in any file under this
    package.  Any torch usage is lazy-imported inside the model adapter
    code path.
  * Doesn't break the synchronous ``synapforge.demo.chat_demo`` —
    that one stays as the request-response fallback.
  * Default policy is conservative: model never interrupts unless
    urgency >= 0.95.

Public API
----------
* :class:`StreamingGen` / :class:`GenerationHandle` / :class:`CancellationToken`
* :class:`InputListener` (+ ``InputEvent``, ``InputEventKind``)
* :class:`ProactiveTrigger` (+ ``TriggerEvent``, ``TriggerSource``)
* :class:`TurnTakingPolicy` (+ ``TurnAction``, ``TurnDecision``)
* :class:`AsyncChatSession` (+ ``ChatChunk``)

Back-compat
-----------
``synapforge.chat`` (the original module name) is kept for legacy callers
that import ``ConversationKernel`` directly.  New code should import
from :mod:`synapforge.interactive`.
"""

from __future__ import annotations

from .async_chat import AsyncChatSession, ChatChunk
from .listener import InputEvent, InputEventKind, InputListener
from .proactive_trigger import (
    ProactiveTrigger,
    TriggerEvent,
    TriggerSource,
)
from .streaming import (
    CancellationToken,
    GenerationHandle,
    StreamingGen,
    StreamingGenerator,
)
from .turn_taking import TurnAction, TurnDecision, TurnTakingPolicy

__all__ = [
    "AsyncChatSession",
    "ChatChunk",
    "InputEvent",
    "InputEventKind",
    "InputListener",
    "ProactiveTrigger",
    "TriggerEvent",
    "TriggerSource",
    "CancellationToken",
    "GenerationHandle",
    "StreamingGen",
    "StreamingGenerator",
    "TurnAction",
    "TurnDecision",
    "TurnTakingPolicy",
]
