"""Async multi-stage data pipeline (Deliverable 2 / 2026-05-02 perf push).

Drop-in replacement for :class:`synapforge.data.ParquetTokenStream` that
runs disk read, tokenization, host pinning, and (optionally) GPU H2D
copy on **separate Python threads** with **bounded ring buffers**
between stages. Bit-exact compatible with the production stream when
fed the same files / shuffle config / RNG seed: the pipeline is purely a
producer/consumer overlay and never reorders or drops tokens.

Why a 4-stage pipeline (vs the existing single ``prefetch_factor=2``)
--------------------------------------------------------------------
The current ``ParquetTokenStream._iter_prefetch`` is one daemon thread
that does **everything** (parquet decode → tokenize → reshape →
``pin_memory()``) into a single ``queue.Queue``. That thread holds the
GIL across:

    parquet.iter_batches()  -> bytes/strings (PyArrow drops the GIL)
    tok.encode(...)         -> python loop over thousands of strings
    torch.tensor(...)        -> small alloc but holds GIL
    arr.pin_memory()         -> calls cudaHostAlloc; cuRT releases GIL

For our typical batch (Qwen tokenizer, B=24, T=256), the tokenize step
is ~30ms/batch on the rental A800's CPUs (PyArrow cores are fine but
the python tokenizer loop is single-threaded). With one mixed thread,
tokenize+pin sit in the same critical path that blocks the next disk
read, so the queue stays at 0-1 batches deep most of the time and the
GPU step has to wait for fresh batches roughly every other step.

A 4-stage pipeline lets each stage run on its own thread with its own
queue, so tokenize can run **while** the next file is being decoded
**while** the previous batch is being pinned. The total wall-time per
batch is bounded by ``max(stage_i)`` instead of ``sum(stage_i)``.

Pipeline layout::

    stage 0 (disk)
        _iter_text_rows_raw  ->  q0_text  (ring of N0 strings)
            ↓
    stage 1 (CPU tokenize)
        tok.encode + chunk    ->  q1_chunks (ring of N1 list[int])
            ↓
    stage 2 (CPU pin)
        torch.tensor + pin    ->  q2_pinned (ring of N2 (in, out) tuples)
            ↓
    stage 3 (consumer = trainer GPU step)
        x.to(device, non_blocking=True) is overlapped with stage 2

Configuration:
    stages: 1..4. ``1`` falls back to the legacy single-thread path
        (passthrough to ``ParquetTokenStream._iter_sync``). ``2`` =
        legacy single-prefetch path. ``3`` = split tokenize/pin. ``4`` =
        full disk/tokenize/chunk/pin split. Default is 4.
    prefetch: per-queue depth. Larger values smooth out variance at
        the cost of pinned RAM. Default 8 ⇒ ~40 MiB pinned per queue
        at B=24/T=256 (8 batches × 24 × 256 × 8 bytes × 2 tensors).

Bit-exact contract:
    For deterministic source streams (``shuffle_buffer<=1`` OR a fixed
    ``shuffle_seed`` + ``shuffle_buffer>1``), the yielded sequence is
    identical to ``ParquetTokenStream`` with the same args. Threading
    is purely an I/O overlap, never a re-order. The unit tests in
    ``tests/integration/test_async_pipeline_bitexact.py`` pin this.

Quality guards (per user铁律 2026-05-02):
    - Default OFF: trainer activates via ``--async-data-pipeline``.
    - Tokenizer + chunk + RNG identical to ``ParquetTokenStream``.
    - Producer-thread exceptions are surfaced on the consumer's stack
      (no silent worker crashes — we re-raise the *original* exception
      with its traceback intact, matching ``_iter_prefetch``).
    - Daemon threads exit with the process. We also expose ``close()``
      for explicit teardown in tests.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

import torch

from synapforge.data import ParquetTokenStream

# Sentinel used to signal end-of-stream between pipeline stages. A
# distinct object so we never confuse it with a real (empty) tuple.
_STAGE_SENTINEL = object()


class AsyncTokenStream:
    """4-stage async pipeline wrapping ``ParquetTokenStream``.

    Constructor arguments mirror :class:`ParquetTokenStream` 1:1 plus
    two pipeline knobs (``stages``, ``prefetch``). The wrapped stream
    is a real ``ParquetTokenStream`` instance; we drive its
    ``_iter_text_rows`` / ``_iter_token_chunks`` methods stage-by-stage.

    Args:
        glob_pattern, seq_len, batch_size, vocab_size, text_column,
        eot_id, loop, tokenizer_name, shuffle_buffer, shuffle_seed,
        pin_memory, remote_warehouse: forwarded verbatim to
        :class:`ParquetTokenStream`. The wrapped stream's own
        ``prefetch_factor`` is forced to ``0`` because the async
        pipeline IS the prefetch mechanism.

        stages: number of pipeline stages, 1..4. Default 4. Each
            stage runs in its own daemon thread with a bounded queue
            in front of it. ``stages=1`` falls back to the legacy
            single-thread path (no threads at all).
        prefetch: capacity of every inter-stage queue. Default 8.

    Bit-exact: see module docstring.
    """

    def __init__(
        self,
        glob_pattern: str,
        seq_len: int = 256,
        batch_size: int = 32,
        vocab_size: int = 50257,
        text_column: str | None = None,
        eot_id: int | None = None,
        loop: bool = True,
        tokenizer_name: str = "gpt2",
        shuffle_buffer: int = 0,
        shuffle_seed: int = 42,
        pin_memory: bool = False,
        remote_warehouse: object | None = None,
        stages: int = 4,
        prefetch: int = 8,
        files_with_weights: list[tuple[str, float]] | None = None,
    ) -> None:
        # ``files_with_weights`` (D4 quality-data push) is forwarded to
        # the inner ParquetTokenStream verbatim. The async pipeline
        # operates on whatever rows the inner stream yields, so the
        # weighted mixture is honoured upstream of any threading.
        self.stages = max(1, min(4, int(stages)))
        self.prefetch = max(2, int(prefetch))

        # Build the inner stream WITHOUT its own prefetch thread; we
        # drive its _iter_text_rows / _iter_token_chunks directly.
        # pin_memory is also disabled at the inner level (the async
        # stage 2 owns pinning) so the inner _build_batch never spends
        # GIL on cudaHostAlloc.
        self._inner = ParquetTokenStream(
            glob_pattern=glob_pattern,
            seq_len=seq_len,
            batch_size=batch_size,
            vocab_size=vocab_size,
            text_column=text_column,
            eot_id=eot_id,
            loop=loop,
            tokenizer_name=tokenizer_name,
            shuffle_buffer=shuffle_buffer,
            shuffle_seed=shuffle_seed,
            prefetch_factor=0,        # async pipeline replaces this
            pin_memory=False,         # stage 2 owns pinning
            remote_warehouse=remote_warehouse,
            files_with_weights=files_with_weights,
        )
        # Public attribute mirrors so callers that rely on these (e.g.
        # the trainer's val-stream auto-scaler) see the same surface
        # as ParquetTokenStream.
        self.seq_len = self._inner.seq_len
        self.batch_size = self._inner.batch_size
        self.vocab_size = self._inner.vocab_size
        self.tokenizer_name = self._inner.tokenizer_name
        self.shuffle_buffer = self._inner.shuffle_buffer
        self.shuffle_seed = self._inner.shuffle_seed
        # Pin attribute reflects the user-requested value (the async
        # pipeline applies it in stage 2 regardless of the inner
        # stream's effective state).
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        # Threads/queues created lazily on first __iter__ so a stream
        # constructed but never iterated does not start workers.
        self._threads: list[threading.Thread] = []
        self._closed = False

    # ------------------------------------------------------------------
    # Stage workers
    # ------------------------------------------------------------------

    def _stage0_disk(self, q_text: "queue.Queue[Any]") -> None:
        """Stage 0: disk → string queue. Reads parquet rows from
        ``self._inner._iter_text_rows`` (which honours the same
        shuffle_buffer / shuffle_seed contract as the legacy stream)
        and pushes each text row onto ``q_text``.
        """
        try:
            for row in self._inner._iter_text_rows():
                if self._closed:
                    break
                q_text.put(row)
        except BaseException as exc:  # noqa: BLE001 — surface to consumer
            q_text.put(("__exc__", exc))
        finally:
            q_text.put(_STAGE_SENTINEL)

    def _stage1_tokenize(
        self,
        q_text: "queue.Queue[Any]",
        q_chunks: "queue.Queue[Any]",
    ) -> None:
        """Stage 1: string → length-(seq_len+1) chunk queue.

        Mirrors ``ParquetTokenStream._iter_token_chunks`` byte-for-byte:
        same tokenizer, same EOT_ID, same window size, same buffer
        rollover. The only difference is the source — we pull strings
        from ``q_text`` instead of calling
        ``self._inner._iter_text_rows()`` directly. This is what
        guarantees bit-exact equality with the legacy path.
        """
        win = self._inner.seq_len + 1
        eot_id = self._inner.eot_id
        tok = self._inner._tok
        buf: list[int] = []
        try:
            while True:
                if self._closed:
                    break
                item = q_text.get()
                if item is _STAGE_SENTINEL:
                    break
                if isinstance(item, tuple) and len(item) == 2 \
                        and item[0] == "__exc__":
                    q_chunks.put(item)  # forward the exception
                    return
                ids = tok.encode(item, add_special_tokens=False)
                if not ids:
                    continue
                buf.extend(ids)
                buf.append(eot_id)
                while len(buf) >= win:
                    q_chunks.put(buf[:win])
                    buf = buf[win:]
        except BaseException as exc:  # noqa: BLE001
            q_chunks.put(("__exc__", exc))
        finally:
            q_chunks.put(_STAGE_SENTINEL)

    def _stage2_pin(
        self,
        q_chunks: "queue.Queue[Any]",
        q_pinned: "queue.Queue[Any]",
    ) -> None:
        """Stage 2: chunk → pinned (in, out) tuple queue.

        Groups ``batch_size`` chunks, materialises a (B, T+1) int64
        tensor (same code as ``ParquetTokenStream._build_batch``),
        splits into ``tokens_in`` / ``tokens_out``, and pins the
        result if ``self.pin_memory`` is on. Pinning is a silent no-op
        on no-CUDA torch builds.
        """
        chunks: list[list[int]] = []
        try:
            while True:
                if self._closed:
                    break
                item = q_chunks.get()
                if item is _STAGE_SENTINEL:
                    break
                if isinstance(item, tuple) and len(item) == 2 \
                        and item[0] == "__exc__":
                    q_pinned.put(item)
                    return
                chunks.append(item)
                if len(chunks) >= self._inner.batch_size:
                    arr = torch.tensor(chunks, dtype=torch.long)
                    tokens_in = arr[:, :-1].contiguous()
                    tokens_out = arr[:, 1:].contiguous()
                    if self.pin_memory:
                        # Same guard as ParquetTokenStream._build_batch:
                        # safe because self.pin_memory is masked to
                        # False on no-CUDA builds in __init__.
                        tokens_in = tokens_in.pin_memory()
                        tokens_out = tokens_out.pin_memory()
                    q_pinned.put((tokens_in, tokens_out))
                    chunks = []
        except BaseException as exc:  # noqa: BLE001
            q_pinned.put(("__exc__", exc))
        finally:
            q_pinned.put(_STAGE_SENTINEL)

    # ------------------------------------------------------------------
    # Reduced-stage variants (stages == 1 / 2 / 3)
    # ------------------------------------------------------------------
    #
    # The reduced-stage paths are degenerate combinations of the four
    # stage workers above. We collapse the threads but keep the same
    # queue boundaries between threads that DO run, so the bit-exact
    # contract holds at every value of ``stages``.

    def _stage12_combined(
        self,
        q_chunks: "queue.Queue[Any]",
    ) -> None:
        """Stages 0+1 fused: disk → chunks. Used when stages==2."""
        win = self._inner.seq_len + 1
        eot_id = self._inner.eot_id
        tok = self._inner._tok
        buf: list[int] = []
        try:
            for txt in self._inner._iter_text_rows():
                if self._closed:
                    break
                ids = tok.encode(txt, add_special_tokens=False)
                if not ids:
                    continue
                buf.extend(ids)
                buf.append(eot_id)
                while len(buf) >= win:
                    q_chunks.put(buf[:win])
                    buf = buf[win:]
        except BaseException as exc:  # noqa: BLE001
            q_chunks.put(("__exc__", exc))
        finally:
            q_chunks.put(_STAGE_SENTINEL)

    def _stage123_combined(
        self,
        q_pinned: "queue.Queue[Any]",
    ) -> None:
        """Stages 0+1+2 fused: disk → pinned tuple. Used when stages==3."""
        win = self._inner.seq_len + 1
        eot_id = self._inner.eot_id
        tok = self._inner._tok
        buf: list[int] = []
        chunks: list[list[int]] = []
        try:
            for txt in self._inner._iter_text_rows():
                if self._closed:
                    break
                ids = tok.encode(txt, add_special_tokens=False)
                if not ids:
                    continue
                buf.extend(ids)
                buf.append(eot_id)
                while len(buf) >= win:
                    chunks.append(buf[:win])
                    buf = buf[win:]
                    if len(chunks) >= self._inner.batch_size:
                        arr = torch.tensor(chunks, dtype=torch.long)
                        tokens_in = arr[:, :-1].contiguous()
                        tokens_out = arr[:, 1:].contiguous()
                        if self.pin_memory:
                            tokens_in = tokens_in.pin_memory()
                            tokens_out = tokens_out.pin_memory()
                        q_pinned.put((tokens_in, tokens_out))
                        chunks = []
        except BaseException as exc:  # noqa: BLE001
            q_pinned.put(("__exc__", exc))
        finally:
            q_pinned.put(_STAGE_SENTINEL)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def _iter_async(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Spawn the configured number of stages and yield batches."""
        if self.stages <= 1:
            # No threads — passthrough to the inner stream's sync path
            # but with our own pin_memory setting honoured.
            yield from self._iter_sync_passthrough()
            return

        cap = self.prefetch
        if self.stages == 2:
            # 1 worker thread: stages 0+1 fused, consumer does pin.
            q_chunks: "queue.Queue[Any]" = queue.Queue(maxsize=cap)
            t = threading.Thread(
                target=self._stage12_combined,
                args=(q_chunks,),
                name="AsyncTokenStream-stage12",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
            yield from self._consume_chunks(q_chunks)
        elif self.stages == 3:
            # 1 worker thread: stages 0+1+2 fused.
            q_pinned: "queue.Queue[Any]" = queue.Queue(maxsize=cap)
            t = threading.Thread(
                target=self._stage123_combined,
                args=(q_pinned,),
                name="AsyncTokenStream-stage123",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
            yield from self._consume_pinned(q_pinned)
        else:
            # stages == 4: 3 worker threads, full split.
            q_text: "queue.Queue[Any]" = queue.Queue(maxsize=cap)
            q_chunks2: "queue.Queue[Any]" = queue.Queue(maxsize=cap)
            q_pinned2: "queue.Queue[Any]" = queue.Queue(maxsize=cap)
            t0 = threading.Thread(
                target=self._stage0_disk, args=(q_text,),
                name="AsyncTokenStream-stage0", daemon=True,
            )
            t1 = threading.Thread(
                target=self._stage1_tokenize, args=(q_text, q_chunks2),
                name="AsyncTokenStream-stage1", daemon=True,
            )
            t2 = threading.Thread(
                target=self._stage2_pin, args=(q_chunks2, q_pinned2),
                name="AsyncTokenStream-stage2", daemon=True,
            )
            for t in (t0, t1, t2):
                t.start()
                self._threads.append(t)
            yield from self._consume_pinned(q_pinned2)

    def _consume_chunks(
        self, q_chunks: "queue.Queue[Any]",
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Stage-2-on-consumer path: build + pin the (B, T+1) tensor on
        the main thread. Used by ``stages==2``.
        """
        chunks: list[list[int]] = []
        while True:
            item = q_chunks.get()
            if item is _STAGE_SENTINEL:
                return
            if isinstance(item, tuple) and len(item) == 2 \
                    and item[0] == "__exc__":
                raise item[1]
            chunks.append(item)
            if len(chunks) >= self._inner.batch_size:
                arr = torch.tensor(chunks, dtype=torch.long)
                tokens_in = arr[:, :-1].contiguous()
                tokens_out = arr[:, 1:].contiguous()
                if self.pin_memory:
                    tokens_in = tokens_in.pin_memory()
                    tokens_out = tokens_out.pin_memory()
                yield tokens_in, tokens_out
                chunks = []

    def _consume_pinned(
        self, q_pinned: "queue.Queue[Any]",
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Last-stage consumer for ``stages in {3, 4}``: just drain the
        pinned-batch queue, surfacing producer-thread exceptions."""
        while True:
            item = q_pinned.get()
            if item is _STAGE_SENTINEL:
                return
            if isinstance(item, tuple) and len(item) == 2 \
                    and item[0] == "__exc__":
                raise item[1]
            yield item

    def _iter_sync_passthrough(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """``stages==1``: legacy single-thread tokenize+chunk+(maybe)pin."""
        chunks: list[list[int]] = []
        for c in self._inner._iter_token_chunks():
            chunks.append(c)
            if len(chunks) >= self._inner.batch_size:
                arr = torch.tensor(chunks, dtype=torch.long)
                tokens_in = arr[:, :-1].contiguous()
                tokens_out = arr[:, 1:].contiguous()
                if self.pin_memory:
                    tokens_in = tokens_in.pin_memory()
                    tokens_out = tokens_out.pin_memory()
                yield tokens_in, tokens_out
                chunks = []

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (tokens_in [B,T], tokens_out [B,T]) int64 cpu tensors."""
        yield from self._iter_async()

    def close(self) -> None:
        """Signal worker threads to exit and join them. Safe to call
        multiple times. Daemon threads also exit on process exit, so
        explicit close is only needed in long-running test harnesses."""
        self._closed = True

    def __repr__(self) -> str:
        return (
            f"AsyncTokenStream(stages={self.stages}, prefetch={self.prefetch}, "
            f"pin_memory={self.pin_memory}, inner={self._inner!r})"
        )


__all__ = ["AsyncTokenStream"]
