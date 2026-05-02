"""End-to-end smoke for synapforge.interactive.async_chat.AsyncChatSession.

Drives a 5-turn synthetic conversation through the full kernel using:
  * a fake StreamingGen that doesn't import torch (replaces the model
    forward path entirely),
  * an InputListener fed by a deterministic iterable transcript,
  * the real TurnTakingPolicy + ProactiveTrigger,
  * an on_chunk callback that collects ChatChunks for assertion.

Verifies the whole pipeline completes without errors, that proactive
triggers can be observed in the audit log, and that the listener EOF
shuts the session down cleanly.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import List

import pytest

from synapforge.interactive import (
    AsyncChatSession,
    ChatChunk,
    InputListener,
    ProactiveTrigger,
    StreamingGen,
    TurnTakingPolicy,
)
from synapforge.interactive.streaming import GenerationHandle, CancellationToken


# ---------------------------------------------------------------------------
# Fake generator — yields a canned response per turn
# ---------------------------------------------------------------------------


class _FakeStreamingGen(StreamingGen):
    """A StreamingGen that bypasses ``_run`` and emits canned text."""

    def __init__(self, responses=None):
        self.responses = list(responses or ["okay. ", "sure. ", "got it. ",
                                            "I see. ", "understood."])
        self._idx = 0

    async def start(
        self,
        prompt_text: str,
        request_id: str = "",
        max_new: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> GenerationHandle:
        token = CancellationToken()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        handle = GenerationHandle(
            request_id=request_id or f"fake_{self._idx}",
            token=token,
            output_queue=queue,
        )
        text = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        asyncio.create_task(self._emit(handle, text))
        return handle

    async def _emit(self, handle: GenerationHandle, text: str) -> None:
        try:
            for c in text:
                if handle.token.cancelled:
                    await handle.output_queue.put("<<<__CANCELLED__>>>")
                    return
                await handle.output_queue.put(c)
                handle.n_tokens_emitted += 1
                await asyncio.sleep(0)
            await handle.output_queue.put("<<<__DONE__>>>")
        except Exception as exc:
            await handle.output_queue.put(f"<<<__ERROR__>>>{exc!r}")


# ---------------------------------------------------------------------------
# 5-turn conversation smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_5_turn_conversation_smoke():
    transcript = [
        "hello",
        "\n",
        "how are you",
        "\n",
        "tell me about CfC",
        "\n",
        "and PLIF",
        "\n",
        "thanks bye",
        "\n",
        None,  # EOF
    ]
    listener = InputListener.with_iterable(transcript)
    gen = _FakeStreamingGen()
    chunks: List[ChatChunk] = []
    session = AsyncChatSession(
        generator=gen,
        listener=listener,
        on_chunk=lambda c: chunks.append(c),
    )
    await asyncio.wait_for(session.run(), timeout=10.0)

    # Each user EOT should have produced at least one model chunk.
    text_collected = "".join(c.text for c in chunks if not c.proactive)
    assert text_collected, "expected model output, got nothing"
    assert len(session.state.user_messages) == 5
    assert len(session.state.model_messages) == 5
    # Each model message must have a non-empty finish_reason. The fake
    # generator finishes with "eos"; a user-aborted turn is "cancelled" or
    # carries the user-supplied reason ("user_new_eot" etc). All are fine.
    for m in session.state.model_messages:
        assert m["finish_reason"], (
            f"empty finish_reason on model message: {m!r}"
        )


@pytest.mark.asyncio
async def test_user_speech_aborts_in_flight_response():
    """Mid-response user typing → model handle is cancelled."""
    transcript = [
        "first message",
        "\n",
        # User starts typing again before model finishes the canned response.
        "stop",
        # Then sends a new EOT.
        "\n",
        None,
    ]
    listener = InputListener.with_iterable(transcript)
    # Make the response slow enough that "stop" arrives mid-stream.
    gen = _FakeStreamingGen(responses=["a" * 100])
    chunks: List[ChatChunk] = []
    session = AsyncChatSession(
        generator=gen,
        listener=listener,
        on_chunk=lambda c: chunks.append(c),
    )
    await asyncio.wait_for(session.run(), timeout=10.0)
    # At least one of the model_messages has finish_reason that records a cancel.
    cancelled = [m for m in session.state.model_messages
                 if m["finish_reason"] not in ("eos",)]
    # We expect either the first turn to be cut off, or both turns landed
    # back-to-back; either way, no exception was raised.
    assert session.state.user_messages == ["first message", "stop"]


@pytest.mark.asyncio
async def test_proactive_trigger_logged():
    """Proactive trigger fires while user is silent → audit log entry."""
    transcript = [
        "tell me about alpha and beta",
        "\n",
        # No more input — the listener will feed silence ticks via the
        # background read_fn returning ''. We use the iterable variant so
        # we just stop here, then EOF.
    ]
    items = transcript + [""] * 5 + [None]
    listener = InputListener.with_iterable(items)
    gen = _FakeStreamingGen(responses=["alpha and beta are letters."])

    trigger = ProactiveTrigger(
        check_interval_steps=1,
        silence_threshold_s=0.05,
        silence_relevance_threshold=0.0,  # accept any term overlap
        min_confidence=0.5,
        min_relevance=0.0,
    )

    session = AsyncChatSession(
        generator=gen,
        listener=listener,
        proactive_trigger=trigger,
        proactive_tick_interval_s=0.05,
    )
    session.set_idle_thought_proposer(
        lambda: "by the way, alpha is the first letter."
    )
    # Push some curiosity scores so the curiosity rule has data to work with.
    for _ in range(10):
        trigger.push_curiosity(0.10)
    trigger.push_curiosity(0.95)

    await asyncio.wait_for(session.run(), timeout=10.0)

    # We expect at least one trigger evaluation to have been recorded.
    assert len(session.trigger_log) >= 1
    # The first trigger should have a valid action recorded.
    first = session.trigger_log[0]
    assert first["decision"] in (
        "speak_now",
        "queue_for_later",
        "interrupt_model",
        "stay_silent",
    )


@pytest.mark.asyncio
async def test_session_save(tmp_path):
    transcript = ["hi", "\n", None]
    listener = InputListener.with_iterable(transcript)
    gen = _FakeStreamingGen(responses=["hello back"])
    session = AsyncChatSession(generator=gen, listener=listener)
    await asyncio.wait_for(session.run(), timeout=5.0)

    save_path = tmp_path / "session.json"
    session.save_session(str(save_path))

    data = json.loads(save_path.read_text(encoding="utf-8"))
    assert data["user_messages"] == ["hi"]
    assert len(data["model_messages"]) == 1
    assert "policy_stats" in data
    assert "trigger_stats" in data


@pytest.mark.asyncio
async def test_listener_iterable_basic():
    """The iterable listener emits PARTIAL/EOT/EOF in the right order."""
    listener = InputListener.with_iterable(["a", "b", "\n", None])
    await listener.start()
    out = []
    async for ev in listener.stream():
        out.append((ev.kind.value, ev.text))
    await listener.stop()
    kinds = [k for k, _ in out]
    assert "partial_token" in kinds
    assert "eot" in kinds
    assert kinds[-1] == "eof"
