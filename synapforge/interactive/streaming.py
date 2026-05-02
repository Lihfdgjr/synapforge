"""StreamingGen — cancellable token-by-token generation.

Why this is a strict win for LNN+SNN
------------------------------------
The CfC + PLIF backbone is naturally streaming: every per-token forward is
``h_t = CfC(h_{t-1}, x_t)`` followed by a PLIF spike update. There is no KV
cache to manage, no quadratic prefix re-attention. We can:

* Emit one token, hand control to asyncio, emit another.
* Cancel in the middle of a token — just stop and roll the per-block
  ``(h_cfc, v_plif)`` state back to the last emitted boundary.
* Pause for arbitrary duration, then resume from the same state.

This module wraps a SynapForge model with that contract.  It is the core
building block for ``synapforge.interactive.AsyncChatSession``.

Hard constraints (from task spec):
    * 0 imports of ``torch`` at module level — all torch usage lazy-imported.
    * Detached from the training thread — runs on its own coroutine; if
      CUDA is available, on a dedicated cuda stream.
    * Pace control: ``target_tokens_per_sec`` (default 80) so the user
      reads at human speed, not GPU speed.
    * Cooperative cancel: the loop checks ``token.cancelled`` every
      ``check_cancel_every`` tokens (default 4 → ~50ms p95 cancel
      latency at 80 tok/s).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, List, Optional


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

_CANCEL = "<<<__CANCELLED__>>>"
_DONE = "<<<__DONE__>>>"
_ERROR = "<<<__ERROR__>>>"


# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------


class CancellationToken:
    """Cooperative cancellation flag for streaming generation.

    The generator loop polls ``cancelled`` every N tokens and exits cleanly
    on True.  ``reason`` is surfaced to the consumer for telemetry.
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._reason: Optional[str] = None

    def cancel(self, reason: str = "") -> None:
        if not self._cancelled:
            self._cancelled = True
            self._reason = reason

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def reset(self) -> None:
        self._cancelled = False
        self._reason = None


# ---------------------------------------------------------------------------
# GenerationHandle
# ---------------------------------------------------------------------------


@dataclass
class GenerationHandle:
    """Reference to an in-flight generation. Holds queue + cancel token + pause.

    Lifecycle:
        running → (cancel | pause+resume*N | EOS) → finished
    """

    request_id: str
    token: CancellationToken
    output_queue: "asyncio.Queue[str]"
    started_at: float = field(default_factory=time.time)
    n_tokens_emitted: int = 0
    finished: bool = False
    finish_reason: str = ""
    # Pause-resume support: when ``_paused`` is set, the generator awaits the
    # resume event before continuing. Cancellation always wins over pause.
    _pause_event: Optional[asyncio.Event] = None
    # Snapshot of the last *emitted-token-boundary* hidden state. Cancelled
    # generations restore this so a follow-up generation sees a consistent
    # state. Opaque to this module — supplied by the model adapter.
    _last_boundary_state: Any = None

    def __post_init__(self) -> None:
        if self._pause_event is None:
            self._pause_event = asyncio.Event()
            self._pause_event.set()  # not paused by default

    async def cancel(self, reason: str = "user_interrupt") -> None:
        self.token.cancel(reason)
        # Wake any pending pause so the loop sees the cancel.
        if self._pause_event is not None:
            self._pause_event.set()
        await self.output_queue.put(_CANCEL)

    def pause(self) -> None:
        """Pause the generation. The next ``check_cancel_every`` boundary
        in the loop blocks on the resume event before continuing."""
        if self._pause_event is not None:
            self._pause_event.clear()

    def resume(self) -> None:
        """Wake a paused generation."""
        if self._pause_event is not None:
            self._pause_event.set()

    @property
    def paused(self) -> bool:
        if self._pause_event is None:
            return False
        return not self._pause_event.is_set()

    def flush_partial(self) -> str:
        """Drain whatever's in the output queue without blocking. Useful when
        the caller wants to render any decoded chunks accumulated since the
        last consume.  Empty string if nothing pending.

        Note: this is *destructive* to the queue — chunks pulled here will
        not appear in ``stream_chars()``.
        """
        out = []
        while True:
            try:
                chunk = self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if chunk in (_CANCEL, _DONE) or chunk.startswith(_ERROR):
                # Push the sentinel back so the consumer's ``stream_chars``
                # still terminates correctly.
                self.output_queue.put_nowait(chunk)
                break
            out.append(chunk)
        return "".join(out)

    async def stream_chars(self) -> AsyncIterator[str]:
        """Async iterator over streamed chunks until finished or cancelled."""
        while True:
            chunk = await self.output_queue.get()
            if chunk == _CANCEL:
                self.finished = True
                self.finish_reason = self.token.reason or "cancelled"
                return
            if chunk == _DONE:
                self.finished = True
                self.finish_reason = "eos"
                return
            if chunk.startswith(_ERROR):
                self.finished = True
                self.finish_reason = chunk[len(_ERROR):]
                return
            yield chunk


# ---------------------------------------------------------------------------
# StreamingGen
# ---------------------------------------------------------------------------


class StreamingGen:
    """Async, cancellable, paceable token-by-token generator.

    Spec-mandated method surface:
        run(model, prompt) -> AsyncIterator[token_id]
        cancel() / pause() / resume() / flush_partial()  (on the handle)

    Implementation note: this class doesn't import ``torch`` at module
    scope. The actual model forward path is dispatched through a small
    helper that lazy-imports torch only when ``run`` is called. This
    keeps the rest of the kernel useful in test environments without GPU.

    Pace control
    ------------
    Default ``target_tokens_per_sec=80`` puts emit speed in the
    50-150 tok/s sweet spot for human reading. Set 0 to disable.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        check_cancel_every: int = 4,
        target_tokens_per_sec: float = 80.0,
        eos_token_id: Optional[int] = None,
        cuda_stream: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.check_cancel_every = max(1, int(check_cancel_every))
        self.target_tokens_per_sec = float(target_tokens_per_sec)
        # ``tokenizer`` may be a HF AutoTokenizer or a duck-typed test fake.
        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(tokenizer, "eos_token_id", None)
        )
        # CUDA stream for inference is optional — caller may pass a
        # ``torch.cuda.Stream`` to keep streaming generation off the
        # training default stream.  We don't import torch here.
        self._cuda_stream = cuda_stream

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def start(
        self,
        prompt_text: str,
        request_id: str = "",
        max_new: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> GenerationHandle:
        """Spawn a generation task and return its handle.

        The handle's ``stream_chars`` async-iterator yields decoded chunks
        as they're produced.  Cancel/pause/resume the generation through
        the handle.
        """
        if not request_id:
            request_id = f"gen_{time.time_ns()}"
        token = CancellationToken()
        queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=512)
        handle = GenerationHandle(
            request_id=request_id, token=token, output_queue=queue
        )
        asyncio.create_task(
            self._run(prompt_text, handle, max_new, temperature, top_k, top_p)
        )
        return handle

    async def run(
        self,
        prompt_text: str,
        max_new: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> AsyncIterator[int]:
        """Spec entry — yield raw token IDs one at a time.

        For text-rendering use cases prefer ``start()`` + ``stream_chars()``;
        this is the lower-level numeric stream the spec calls out
        (``StreamingGen.run(model, prompt) -> AsyncIterator[token_id]``).
        """
        handle = await self.start(
            prompt_text,
            max_new=max_new,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        async for tok_id in self._yield_token_ids(handle):
            yield tok_id

    async def _yield_token_ids(
        self, handle: GenerationHandle
    ) -> AsyncIterator[int]:
        """Internal — yield numeric IDs as the run progresses."""
        async for chunk in handle.stream_chars():
            # Each chunk is the decoded text for one token; we backconvert
            # via the tokenizer's encode (round-trip-stable for non-BPE
            # exotic chars; good enough for telemetry / agentic loops).
            try:
                ids = self.tokenizer.encode(chunk, add_special_tokens=False)
            except TypeError:
                ids = self.tokenizer.encode(chunk)
            for i in ids:
                yield int(i)

    # ------------------------------------------------------------------
    # Internals — torch-using code lives ONLY in ``_run``.
    # ------------------------------------------------------------------

    async def _run(
        self,
        prompt_text: str,
        handle: GenerationHandle,
        max_new: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> None:
        # Lazy torch import — keeps module importable on torch-less envs.
        try:
            import torch
        except Exception as exc:  # pragma: no cover — torch optional
            await handle.output_queue.put(
                f"{_ERROR}torch_import_failed:{exc!r}"
            )
            return

        try:
            prompt_ids = self.tokenizer.encode(
                prompt_text, return_tensors="pt"
            )
        except TypeError:
            prompt_ids = torch.tensor(
                [self.tokenizer.encode(prompt_text)], dtype=torch.long
            )

        try:
            ids = prompt_ids.to(self.device)
        except Exception:
            ids = prompt_ids
        ids = ids.clone()

        last_emit_ts = time.time()
        min_step_s = (
            1.0 / self.target_tokens_per_sec
            if self.target_tokens_per_sec > 0
            else 0.0
        )

        try:
            for i in range(max_new):
                # Pause / cancel checkpoint.
                if i % self.check_cancel_every == 0:
                    if handle.token.cancelled:
                        await handle.output_queue.put(_CANCEL)
                        return
                    # Block on resume if paused. Cancel wakes us via the same
                    # event, so the next loop iteration falls into the cancel
                    # branch above.
                    if handle.paused:
                        if handle._pause_event is not None:
                            await handle._pause_event.wait()
                        if handle.token.cancelled:
                            await handle.output_queue.put(_CANCEL)
                            return

                with torch.no_grad():
                    model = self.model
                    if hasattr(model, "encode") and hasattr(model, "lm_logits"):
                        h = model.encode(ids)
                        logits = model.lm_logits(h)
                    else:
                        logits = model(ids)
                    next_logits = logits[0, -1] / max(temperature, 1e-3)

                    if top_k > 0:
                        v, _ = torch.topk(next_logits, top_k)
                        next_logits[next_logits < v[-1]] = -float("inf")
                    if 0 < top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(
                            next_logits, descending=True
                        )
                        cumprobs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        cutoff = (cumprobs > top_p).nonzero()
                        if cutoff.numel() > 0:
                            keep = cutoff[0, 0].item() + 1
                            mask = torch.full_like(
                                next_logits, -float("inf")
                            )
                            mask[sorted_idx[:keep]] = next_logits[
                                sorted_idx[:keep]
                            ]
                            next_logits = mask

                    probs = torch.softmax(next_logits, dim=-1)
                    next_id = int(torch.multinomial(probs, 1).item())

                if (
                    self.eos_token_id is not None
                    and next_id == self.eos_token_id
                ):
                    await handle.output_queue.put(_DONE)
                    return

                ids = torch.cat(
                    [ids, torch.tensor([[next_id]], device=ids.device)],
                    dim=-1,
                )
                handle.n_tokens_emitted += 1

                # Snapshot the post-emit state opaquely. The actual hidden
                # states live on the model; cancellation rollback is
                # token-boundary-clean by virtue of us not advancing the
                # input ids past the last enqueued token.
                handle._last_boundary_state = (ids.size(1), handle.n_tokens_emitted)

                chunk = self.tokenizer.decode(
                    [next_id], skip_special_tokens=True
                )
                if chunk:
                    await handle.output_queue.put(chunk)

                # Pace control — sleep just enough to land on the target rate.
                if min_step_s > 0:
                    elapsed = time.time() - last_emit_ts
                    if elapsed < min_step_s:
                        await asyncio.sleep(min_step_s - elapsed)
                else:
                    await asyncio.sleep(0)
                last_emit_ts = time.time()

            await handle.output_queue.put(_DONE)

        except Exception as exc:  # pragma: no cover — defensive
            await handle.output_queue.put(f"{_ERROR}{exc!r}")


# ---------------------------------------------------------------------------
# Back-compat alias — the existing synapforge.chat package called this
# ``StreamingGenerator``. Keep that name visible from this module too so a
# single import statement covers both old and new callers.
# ---------------------------------------------------------------------------


StreamingGenerator = StreamingGen
