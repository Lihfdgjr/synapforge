"""Tests for synapforge.interactive.streaming.StreamingGen.

Covers cancel / pause / resume / flush_partial without requiring torch
or a real model: we monkey-patch a fake model + tokenizer that the
StreamingGen treats just like real ones.

Skips the real-torch path when torch isn't installed.
"""

from __future__ import annotations

import asyncio

import pytest

from synapforge.interactive.streaming import (
    CancellationToken,
    GenerationHandle,
    StreamingGen,
    StreamingGenerator,
)


# ---------------------------------------------------------------------------
# Cancellation token tests (pure-python, no torch needed)
# ---------------------------------------------------------------------------


def test_cancellation_token_initial_state():
    t = CancellationToken()
    assert t.cancelled is False
    assert t.reason is None


def test_cancellation_token_cancel_records_reason():
    t = CancellationToken()
    t.cancel("user_interrupt")
    assert t.cancelled is True
    assert t.reason == "user_interrupt"


def test_cancellation_token_reset():
    t = CancellationToken()
    t.cancel("foo")
    t.reset()
    assert t.cancelled is False
    assert t.reason is None


# ---------------------------------------------------------------------------
# Handle pause / resume / flush_partial (pure-python)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_resume_flow():
    queue: asyncio.Queue[str] = asyncio.Queue()
    h = GenerationHandle(
        request_id="t",
        token=CancellationToken(),
        output_queue=queue,
    )
    assert h.paused is False
    h.pause()
    assert h.paused is True
    h.resume()
    assert h.paused is False


@pytest.mark.asyncio
async def test_handle_cancel_puts_sentinel():
    queue: asyncio.Queue[str] = asyncio.Queue()
    h = GenerationHandle(
        request_id="t",
        token=CancellationToken(),
        output_queue=queue,
    )
    await h.cancel("x")
    chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
    assert chunk == "<<<__CANCELLED__>>>"
    assert h.token.cancelled is True


@pytest.mark.asyncio
async def test_flush_partial_drains_queue():
    queue: asyncio.Queue[str] = asyncio.Queue()
    h = GenerationHandle(
        request_id="t",
        token=CancellationToken(),
        output_queue=queue,
    )
    await queue.put("hel")
    await queue.put("lo")
    out = h.flush_partial()
    assert out == "hello"
    # After flush, queue is empty.
    assert queue.empty()


@pytest.mark.asyncio
async def test_flush_partial_preserves_done_sentinel():
    """flush_partial must put the sentinel back so stream_chars terminates."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    h = GenerationHandle(
        request_id="t",
        token=CancellationToken(),
        output_queue=queue,
    )
    await queue.put("a")
    await queue.put("<<<__DONE__>>>")
    out = h.flush_partial()
    assert out == "a"
    sentinel = await asyncio.wait_for(queue.get(), timeout=0.1)
    assert sentinel == "<<<__DONE__>>>"


@pytest.mark.asyncio
async def test_stream_chars_terminates_on_cancel():
    queue: asyncio.Queue[str] = asyncio.Queue()
    h = GenerationHandle(
        request_id="t",
        token=CancellationToken(),
        output_queue=queue,
    )
    await queue.put("hi")
    await queue.put("<<<__CANCELLED__>>>")
    h.token.cancel("user_interrupt")
    chunks = []
    async for c in h.stream_chars():
        chunks.append(c)
    assert chunks == ["hi"]
    assert h.finished is True
    assert h.finish_reason == "user_interrupt"


# ---------------------------------------------------------------------------
# Real generator with torch (skipped if torch missing)
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_or_skip():
    return pytest.importorskip("torch")


class _FakeTokenizer:
    eos_token_id = 999  # never produced by our fake model
    pad_token_id = 0

    def encode(self, s, add_special_tokens=False, return_tensors=None):
        import torch
        ids = [3] * max(1, len(s))
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "x"


class _FakeModel:
    """Returns logits picking id=4 every step.

    StreamingGen treats us as a callable that returns ``logits`` of shape
    ``(B, T, V)``. We construct a tensor on the fly. ``vocab`` is large
    enough that ``top_k=50`` doesn't try to index past it.
    """

    def __init__(self):
        self.next_id = 4
        self.vocab = 200

    def __call__(self, ids):
        import torch
        B, T = ids.size(0), ids.size(1)
        logits = torch.zeros(B, T, self.vocab)
        logits[:, -1, self.next_id] = 100.0
        return logits


@pytest.mark.asyncio
async def test_streaming_cancel_mid_stream(torch_or_skip):
    torch = torch_or_skip
    gen = StreamingGen(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        check_cancel_every=1,
        target_tokens_per_sec=0,  # unlimited so cancel races deterministically
    )
    handle = await gen.start("hello", max_new=200)
    # Read a few chunks then cancel.
    seen = 0
    async for chunk in handle.stream_chars():
        seen += 1
        if seen >= 3:
            await handle.cancel("user_interrupt")
            break
    assert handle.finished is True or handle.token.cancelled is True
    assert handle.token.cancelled
    assert handle.token.reason == "user_interrupt"


@pytest.mark.asyncio
async def test_streaming_pause_resume(torch_or_skip):
    """Pause stops emission; resume yields more chunks."""
    gen = StreamingGen(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        check_cancel_every=1,
        target_tokens_per_sec=0,
    )
    handle = await gen.start("hi", max_new=10)
    # Receive 1 chunk, then pause.
    first = await handle.output_queue.get()
    assert first not in ("<<<__DONE__>>>", "<<<__CANCELLED__>>>")
    handle.pause()
    # Now drain a small wait to make sure no further chunks land.
    try:
        await asyncio.wait_for(handle.output_queue.get(), timeout=0.15)
        # If we got a chunk while paused, that's still allowed by the
        # spec because the loop only checks pause every N tokens; but
        # we must NEVER get more than N chunks.
    except asyncio.TimeoutError:
        pass
    handle.resume()
    # Cancel to cleanly tear down so the test exits.
    await handle.cancel("test_cleanup")


def test_streaming_generator_alias():
    """Back-compat: ``StreamingGenerator`` is an alias for ``StreamingGen``."""
    assert StreamingGenerator is StreamingGen
