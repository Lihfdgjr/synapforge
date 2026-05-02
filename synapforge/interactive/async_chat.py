"""AsyncChatSession — full async chat kernel.

Ties together :class:`StreamingGen`, :class:`InputListener`,
:class:`ProactiveTrigger`, and :class:`TurnTakingPolicy` into a single
coroutine you can drive from a CLI / WebSocket / IPC pipe.

Spec contract:
  * ``run()`` is an asyncio coroutine that loops on listener events,
    model forward, decisions until the listener emits EOF.
  * Maintains conversation hidden state across turns: the streaming
    generator carries CfC + PLIF state forward via the model adapter
    (``model.encode + model.lm_logits``), so per-turn prompt re-prefill
    is avoided when the adapter supports incremental decode.
  * Saves session checkpoint: hidden state + GoalMemory + plasticity
    weights on graceful shutdown, in JSON-serializable form when
    possible (numpy / torch tensors stored as ``.npy`` siblings).
  * Default policy is conservative — model never interrupts unless
    urgency clears 0.95.

Hard constraints:
  * 0 imports of ``torch`` at module level. Lazy-imported only when the
    session ckpt has actual tensor weights to serialize.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional

from .listener import InputEvent, InputEventKind, InputListener
from .proactive_trigger import ProactiveTrigger, TriggerEvent
from .streaming import GenerationHandle, StreamingGen
from .turn_taking import TurnAction, TurnDecision, TurnTakingPolicy


# ---------------------------------------------------------------------------
# Session output schema
# ---------------------------------------------------------------------------


@dataclass
class ChatChunk:
    text: str
    proactive: bool = False
    interrupt_marker: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class _SessionState:
    user_buffer: str = ""
    user_messages: List[str] = field(default_factory=list)
    model_messages: List[dict] = field(default_factory=list)
    current_handle: Optional[GenerationHandle] = None
    proactive_queue: List[TriggerEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AsyncChatSession
# ---------------------------------------------------------------------------


class AsyncChatSession:
    """One async conversation.  Wires listener + generator + policy together.

    Usage (sketch):

        gen = StreamingGen(model, tokenizer)
        listener = InputListener.with_stdin_lines()
        session = AsyncChatSession(
            generator=gen,
            listener=listener,
            on_chunk=lambda c: print(c.text, end="", flush=True),
        )
        asyncio.run(session.run())
    """

    def __init__(
        self,
        generator: StreamingGen,
        listener: InputListener,
        proactive_trigger: Optional[ProactiveTrigger] = None,
        turn_policy: Optional[TurnTakingPolicy] = None,
        on_chunk: Optional[Callable[[ChatChunk], Awaitable[None] | None]] = None,
        max_history_chars: int = 16384,
        proactive_tick_interval_s: float = 1.0,
        prompt_builder: Optional[Callable[[List[str], str], str]] = None,
    ) -> None:
        self.generator = generator
        self.listener = listener
        self.proactive_trigger = proactive_trigger or ProactiveTrigger()
        self.turn_policy = turn_policy or TurnTakingPolicy()
        self.on_chunk = on_chunk
        self.max_history_chars = int(max_history_chars)
        self.proactive_tick_interval_s = float(proactive_tick_interval_s)
        self.prompt_builder = prompt_builder or self._default_prompt_builder

        self.state = _SessionState()
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._goal_memory_ref: Any = None
        self._idle_thought_proposer: Optional[Callable[[], str]] = None
        # Audit trail of trigger evaluations — useful for tuning thresholds
        # and for the test harness.
        self.trigger_log: List[dict] = []

    # ------------------------------------------------------------------
    # Configuration knobs
    # ------------------------------------------------------------------

    def attach_goal_memory(self, goal_memory: Any) -> None:
        self._goal_memory_ref = goal_memory

    def set_idle_thought_proposer(
        self, proposer: Callable[[], str]
    ) -> None:
        self._idle_thought_proposer = proposer

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    async def run(self) -> None:
        if self._running:
            return
        self._running = True
        await self.listener.start()
        try:
            self._tasks.append(asyncio.create_task(self._listener_loop()))
            self._tasks.append(asyncio.create_task(self._proactive_loop()))
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await self._shutdown()

    async def _shutdown(self) -> None:
        await self.listener.stop()
        if (
            self.state.current_handle is not None
            and not self.state.current_handle.finished
        ):
            await self.state.current_handle.cancel("session_shutdown")
        self._tasks.clear()
        self._running = False

    # ------------------------------------------------------------------
    # Listener loop — consumes user keystrokes / EOT / EOF
    # ------------------------------------------------------------------

    async def _listener_loop(self) -> None:
        async for ev in self.listener.stream():
            if not self._running:
                return
            if ev.kind == InputEventKind.EOF:
                self._running = False
                return
            await self._handle_input(ev)

    async def _handle_input(self, ev: InputEvent) -> None:
        if ev.kind == InputEventKind.PARTIAL_TOKEN:
            self.state.user_buffer += ev.text
            self.turn_policy.on_user_partial()
            self.proactive_trigger.note_user_interaction(ts=ev.ts)
            currently_generating = self._is_generating()
            decision = self.turn_policy.decide_on_user_speech(
                currently_generating
            )
            if (
                decision.action == TurnAction.ABORT_MODEL
                and self.state.current_handle is not None
            ):
                await self._abort_current("user_started_typing")
            return

        if ev.kind == InputEventKind.EOT:
            await self._on_user_eot()
            return

        if ev.kind == InputEventKind.CANCEL:
            await self._abort_current("user_cancel")
            return

        if ev.kind == InputEventKind.SILENCE:
            # Silence event — relevant for proactive trigger. The
            # ProactiveTrigger checks the silence threshold itself; we
            # just record the marker.
            return

    async def _on_user_eot(self) -> None:
        msg = self.state.user_buffer.strip()
        self.state.user_buffer = ""
        if not msg:
            return
        self.state.user_messages.append(msg)
        self.turn_policy.on_user_eot()

        currently_generating = self._is_generating()
        decision = self.turn_policy.decide_on_user_eot(currently_generating)

        if decision.action == TurnAction.ABORT_MODEL:
            await self._abort_current("user_new_eot")

        if decision.delay_ms > 0:
            await asyncio.sleep(decision.delay_ms / 1000.0)

        if decision.action in (TurnAction.SPEAK_NOW, TurnAction.ABORT_MODEL):
            await self._start_response(msg, proactive=False)

    # ------------------------------------------------------------------
    # Proactive loop — polls trigger every K cfc steps' worth of wall time
    # ------------------------------------------------------------------

    async def _proactive_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.proactive_tick_interval_s)
            recent_terms = self._extract_recent_terms()
            triggers = self.proactive_trigger.tick(
                goal_memory=self._goal_memory_ref,
                recent_user_terms=recent_terms,
                idle_thought_proposer=self._idle_thought_proposer,
                h_t_summary=self._h_t_summary(),
            )
            for t in triggers:
                await self._handle_trigger(t)

    async def _handle_trigger(self, trigger: TriggerEvent) -> None:
        currently_generating = self._is_generating()
        decision = self.turn_policy.decide_on_proactive(
            urgency=trigger.urgency,
            currently_generating=currently_generating,
        )
        self.trigger_log.append(
            {
                "trigger": trigger.source.value,
                "urgency": trigger.urgency,
                "confidence": trigger.confidence,
                "relevance": trigger.relevance,
                "decision": decision.action.value,
                "reason": decision.reason,
                "ts": trigger.ts,
            }
        )

        if decision.action == TurnAction.SPEAK_NOW:
            await self._start_response(
                trigger.suggested_text or "By the way,",
                proactive=True,
                interrupt_marker=decision.interrupt_marker,
            )
            return
        if decision.action == TurnAction.INTERRUPT_MODEL:
            await self._abort_current("proactive_interrupt")
            await self._start_response(
                trigger.suggested_text or "[interrupt] One second,",
                proactive=True,
                interrupt_marker=decision.interrupt_marker or "[interrupt]",
            )
            return
        if decision.action == TurnAction.QUEUE_FOR_LATER:
            self.state.proactive_queue.append(trigger)
        # STAY_SILENT / ABORT_MODEL — drop.

    # ------------------------------------------------------------------
    # Generation orchestration
    # ------------------------------------------------------------------

    async def _start_response(
        self,
        seed: str,
        proactive: bool,
        interrupt_marker: str = "",
    ) -> None:
        prompt = self.prompt_builder(self.state.user_messages, seed)
        try:
            handle = await self.generator.start(prompt)
        except Exception as exc:  # pragma: no cover — defensive
            await self._emit_chunk(
                ChatChunk(text=f"[error: {exc!r}]", proactive=False)
            )
            return
        self.state.current_handle = handle
        asyncio.create_task(
            self._consume_handle(
                handle,
                proactive=proactive,
                interrupt_marker=interrupt_marker,
                seed_text=seed,
            )
        )

    async def _consume_handle(
        self,
        handle: GenerationHandle,
        proactive: bool,
        interrupt_marker: str = "",
        seed_text: str = "",
    ) -> None:
        prefix = ""
        if proactive:
            prefix = "[Proactive] "
        if interrupt_marker:
            prefix = interrupt_marker + " " + prefix
        first = True
        text_collected = ""
        async for chunk in handle.stream_chars():
            if first:
                first = False
                if prefix:
                    await self._emit_chunk(
                        ChatChunk(
                            text=prefix,
                            proactive=proactive,
                            interrupt_marker=interrupt_marker,
                        )
                    )
            self.turn_policy.on_model_token()
            text_collected += chunk
            await self._emit_chunk(
                ChatChunk(text=chunk, proactive=proactive)
            )

        self.turn_policy.on_model_finished()
        self.state.model_messages.append(
            {
                "text": text_collected,
                "proactive": proactive,
                "ts": time.time(),
                "finish_reason": handle.finish_reason,
                "interrupt_marker": interrupt_marker,
                "seed_text": seed_text,
            }
        )
        if handle is self.state.current_handle:
            self.state.current_handle = None

    async def _abort_current(self, reason: str) -> None:
        if (
            self.state.current_handle is not None
            and not self.state.current_handle.finished
        ):
            await self.state.current_handle.cancel(reason)
        self.state.current_handle = None

    async def _emit_chunk(self, chunk: ChatChunk) -> None:
        if self.on_chunk is None:
            return
        try:
            res = self.on_chunk(chunk)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            # An on_chunk failure must never kill the kernel.
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_generating(self) -> bool:
        return (
            self.state.current_handle is not None
            and not self.state.current_handle.finished
        )

    def _extract_recent_terms(self) -> List[str]:
        out: List[str] = []
        for m in reversed(self.state.user_messages[-3:]):
            for w in m.split():
                w = w.strip(",.!?。!？")
                if 2 < len(w) < 30:
                    out.append(w)
        return out[:50]

    def _h_t_summary(self) -> str:
        # Lightweight summary stand-in: last user message tail. The real
        # h_t summary would come from the model's hidden state via a
        # caller-supplied probe — kept pluggable so tests don't need
        # torch.
        if not self.state.user_messages:
            return ""
        return self.state.user_messages[-1][-200:]

    def _default_prompt_builder(
        self, history: List[str], current_seed: str
    ) -> str:
        out_parts: List[str] = []
        chars = 0
        for m in reversed(history[-10:] + [current_seed]):
            piece = f"<|im_start|>user\n{m}<|im_end|>\n"
            chars += len(piece)
            if chars > self.max_history_chars:
                break
            out_parts.append(piece)
        out_parts.reverse()
        return "".join(out_parts) + "<|im_start|>assistant\n"

    # ------------------------------------------------------------------
    # Session ckpt
    # ------------------------------------------------------------------

    def save_session(self, path: str) -> dict:
        """Save the conversation transcript + trigger log + policy stats.

        Plasticity weights & hidden state are NOT serialized here — those
        live on the model object. The caller can save the model state
        separately via ``torch.save(model.state_dict(), ...)``.
        """
        data = {
            "user_messages": self.state.user_messages,
            "model_messages": self.state.model_messages,
            "trigger_log": self.trigger_log[-500:],  # cap audit trail
            "policy_stats": self.turn_policy.stats(),
            "trigger_stats": self.proactive_trigger.stats(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return data
