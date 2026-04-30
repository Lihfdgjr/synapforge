"""
ConversationKernel — async event loop replacing request-response chat.

The kernel runs forever. Three event sources feed an inbox queue:
  1. User messages (per-keystroke or per-message)
  2. Cron / time-based triggers
  3. Internal idle / proactive triggers
  4. Web events (autonomous_daemon → relevant new content)

Outputs go to an outbox queue (consumed by client UI / WebSocket).

Each event is dispatched through:
  TurnTakingDetector  →  InterruptPolicy  →  StreamingGenerator (or silence)

Critical properties:
  - User CAN type while model is generating (concurrent inbox)
  - Model can be cancelled mid-generation by user typing
  - Model can speak unprompted (proactive triggers, with [Proactive] tag)
  - Model can speak twice in a row (no strict turn alternation)
  - User can mute model entirely (UI button → InterruptPolicy.set_mute)

This is the chat kernel. Wire it to a WebSocket / SSE for browser, to a
terminal for CLI, to PyQt for desktop.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Dict, List, Optional

from .interrupt_policy import (
    DecisionKind,
    InterruptPolicy,
    InterruptDecision,
    SilenceDecision,
    SpeakDecision,
)
from .proactive import ProactiveMessenger, ProactiveTrigger, TriggerSource
from .streaming import GenerationHandle, StreamingGenerator
from .turn_taking import TurnState, TurnTakingDetector


class EventKind(Enum):
    USER_CHAR = "user_char"
    USER_SUBMIT = "user_submit"
    USER_MUTE = "user_mute"
    USER_UNMUTE = "user_unmute"
    PROACTIVE = "proactive"
    TICK = "tick"


@dataclass
class ConversationEvent:
    kind: EventKind
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class _ConversationState:
    user_buffer: str = ""
    user_messages: List[str] = field(default_factory=list)
    model_messages: List[dict] = field(default_factory=list)
    current_handle: Optional[GenerationHandle] = None


class ConversationKernel:
    """The async event loop. One instance per conversation.

    Usage (sketch):

        kernel = ConversationKernel(
            generator=StreamingGenerator(model, tokenizer),
            policy=InterruptPolicy(),
            turn_taker=TurnTakingDetector(),
            proactive=ProactiveMessenger(),
        )

        # Publisher (called by client on user input):
        await kernel.send_user_char('h')
        await kernel.send_user_char('i')
        await kernel.send_user_submit()

        # Consumer (renders model output to UI):
        async for chunk in kernel.outbox_stream():
            print(chunk['text'], end='', flush=True)
    """

    def __init__(
        self,
        generator: StreamingGenerator,
        policy: Optional[InterruptPolicy] = None,
        turn_taker: Optional[TurnTakingDetector] = None,
        proactive: Optional[ProactiveMessenger] = None,
        max_history_chars: int = 16384,
        tick_interval_s: float = 5.0,
    ) -> None:
        self.generator = generator
        self.policy = policy or InterruptPolicy()
        self.turn_taker = turn_taker or TurnTakingDetector()
        self.proactive = proactive or ProactiveMessenger()
        self.max_history_chars = max_history_chars
        self.tick_interval_s = tick_interval_s

        self.inbox: asyncio.Queue[ConversationEvent] = asyncio.Queue()
        self.outbox: asyncio.Queue[dict] = asyncio.Queue()
        self.state = _ConversationState()
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks.append(asyncio.create_task(self._main_loop()))
        self._tasks.append(asyncio.create_task(self._tick_loop()))

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    async def send_user_char(self, c: str) -> None:
        self.turn_taker.on_char(c)
        self.state.user_buffer += c
        self.policy.note_user_interaction()
        await self.inbox.put(ConversationEvent(kind=EventKind.USER_CHAR, payload={"char": c}))

    async def send_user_text(self, text: str) -> None:
        for c in text:
            self.turn_taker.on_char(c)
        self.state.user_buffer += text
        self.policy.note_user_interaction()
        await self.inbox.put(ConversationEvent(kind=EventKind.USER_CHAR, payload={"text": text}))

    async def send_user_submit(self) -> None:
        self.turn_taker.on_explicit_submit()
        self.policy.note_user_interaction()
        await self.inbox.put(ConversationEvent(kind=EventKind.USER_SUBMIT))

    async def send_mute(self, until_seconds: Optional[float] = None) -> None:
        await self.inbox.put(ConversationEvent(
            kind=EventKind.USER_MUTE,
            payload={"until_seconds": until_seconds},
        ))

    async def send_unmute(self) -> None:
        await self.inbox.put(ConversationEvent(kind=EventKind.USER_UNMUTE))

    async def emit_proactive(self, trigger: ProactiveTrigger) -> None:
        await self.inbox.put(ConversationEvent(
            kind=EventKind.PROACTIVE,
            payload={"trigger": trigger},
        ))

    async def outbox_stream(self) -> AsyncIterator[dict]:
        while True:
            chunk = await self.outbox.get()
            yield chunk

    async def _main_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await self._handle_event(event)
            except Exception as e:
                await self.outbox.put({
                    "kind": "error",
                    "text": f"kernel error: {e!r}",
                    "ts": time.time(),
                })

    async def _tick_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.tick_interval_s)
            triggers = self.proactive.poll(
                seconds_since_user=time.time() - self.policy.last_user_interaction_ts,
            )
            for t in triggers:
                await self.emit_proactive(t)

    async def _handle_event(self, event: ConversationEvent) -> None:
        if event.kind == EventKind.USER_CHAR:
            await self._on_user_char(event)
        elif event.kind == EventKind.USER_SUBMIT:
            await self._on_user_submit(event)
        elif event.kind == EventKind.USER_MUTE:
            self.policy.set_mute(event.payload.get("until_seconds"))
        elif event.kind == EventKind.USER_UNMUTE:
            self.policy.unmute()
        elif event.kind == EventKind.PROACTIVE:
            await self._on_proactive(event)
        elif event.kind == EventKind.TICK:
            pass

    async def _on_user_char(self, event: ConversationEvent) -> None:
        currently_generating = self.state.current_handle is not None and not self.state.current_handle.finished
        turn_state = self.turn_taker.state(self.state.user_buffer)
        decision = self.policy.decide_on_user_msg(
            turn_state=turn_state.value,
            msg_text=self.state.user_buffer,
            currently_generating=currently_generating,
        )

        if isinstance(decision, InterruptDecision):
            if self.state.current_handle:
                await self.state.current_handle.cancel(decision.reason)
                await self.outbox.put({
                    "kind": "interrupt",
                    "text": "[interrupted by user]",
                    "ts": time.time(),
                })

    async def _on_user_submit(self, event: ConversationEvent) -> None:
        msg = self.state.user_buffer.strip()
        if not msg:
            return
        self.state.user_messages.append(msg)
        self.state.user_buffer = ""
        self.turn_taker.reset()
        self.proactive.update_user_recent_terms(msg.split()[-50:])

        if self.state.current_handle and not self.state.current_handle.finished:
            await self.state.current_handle.cancel("user_new_message")

        await self._start_response(msg)

    async def _start_response(self, user_msg: str) -> None:
        prompt = self._build_prompt(user_msg)
        handle = await self.generator.start(prompt)
        self.state.current_handle = handle
        asyncio.create_task(self._consume_handle(handle, is_proactive=False))

    async def _consume_handle(self, handle: GenerationHandle, is_proactive: bool) -> None:
        prefix = "[Proactive] " if is_proactive else ""
        first_chunk = True
        text_collected = ""
        async for chunk in handle.stream_chars():
            if first_chunk and is_proactive:
                await self.outbox.put({"kind": "model", "text": prefix, "ts": time.time(), "proactive": True})
                first_chunk = False
            await self.outbox.put({"kind": "model", "text": chunk, "ts": time.time(), "proactive": is_proactive})
            text_collected += chunk

        self.state.model_messages.append({
            "text": text_collected,
            "proactive": is_proactive,
            "ts": time.time(),
            "finish_reason": handle.finish_reason,
        })

        if not is_proactive:
            self.state.current_handle = None

    async def _on_proactive(self, event: ConversationEvent) -> None:
        trigger: ProactiveTrigger = event.payload["trigger"]
        currently_generating = self.state.current_handle is not None and not self.state.current_handle.finished
        seconds_since_user = time.time() - self.policy.last_user_interaction_ts

        decision = self.policy.decide_on_proactive(
            urgency=trigger.urgency,
            currently_generating=currently_generating,
            seconds_since_user_idle=seconds_since_user,
        )

        if isinstance(decision, SpeakDecision):
            self.policy.note_proactive_sent()
            prompt = self._build_proactive_prompt(trigger)
            handle = await self.generator.start(prompt)
            asyncio.create_task(self._consume_handle(handle, is_proactive=True))

    def _build_prompt(self, user_msg: str) -> str:
        history = ""
        chars = 0
        for m in reversed(self.state.user_messages[-10:] + [user_msg]):
            piece = f"<|im_start|>user\n{m}<|im_end|>\n"
            chars += len(piece)
            if chars > self.max_history_chars:
                break
            history = piece + history
        return history + "<|im_start|>assistant\n"

    def _build_proactive_prompt(self, trigger: ProactiveTrigger) -> str:
        seed = trigger.suggested_message or "I want to share something."
        return (
            f"<|im_start|>system\n"
            f"You are SynapForge. You have something proactive to say to the user "
            f"(source: {trigger.source.value}). Open with [Proactive] tag.\n<|im_end|>\n"
            f"<|im_start|>assistant\n[Proactive] {seed}"
        )

    def stats(self) -> dict:
        return {
            "user_messages": len(self.state.user_messages),
            "model_messages": len(self.state.model_messages),
            "currently_generating": (
                self.state.current_handle is not None
                and not self.state.current_handle.finished
            ),
            "policy": self.policy.stats(),
            "turn_taker": self.turn_taker.stats(),
        }
