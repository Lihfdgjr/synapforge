"""``NativeJsonlStream`` -- torch-free JSONL -> token-pair iterator.

Mirror of ``NativeParquetStream`` for JSONL inputs. Used for KD-distill
data (``runs/kd_distill_*.jsonl``) and instruction-tuning corpora
(``alpaca-zh.jsonl`` and friends). Same pipeline, same emit shape:

  jsonl rows -> string text -> NativeTokenizer.encode -> (B, T) numpy.

JSONL row schemas supported (in priority order)
-----------------------------------------------
1. ``{"text": "..."}``                  -- standard pretrain.
2. ``{"content": "..."}``               -- alt name (HF datasets style).
3. ``{"messages": [{role, content}, ...]}`` -- chat / SFT format. We
   serialise each message as ``<|role|>content`` so that the legacy
   chat tokenizer + our tokenizer agree on the prefix tokens. Multi-turn
   conversations join with ``\\n``.
4. ``{"prompt": "...", "completion": "..."}`` -- alpaca-zh style.
   Joined with ``\\n``.

Files may be plaintext ``*.jsonl`` or gzip-compressed ``*.jsonl.gz``.
Each line is parsed with ``json.loads``; lines that are blank or fail
to parse are silently skipped (real corpora always have one or two
short lines from line-buffered writes).

Constraints
-----------
* Zero ``import torch``.
* No HF or pyarrow required for this module's input path; only the
  tokenizer pulls those in (and only when it picks the corresponding
  backend).
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import queue
import random
import threading
from collections.abc import Iterator
from typing import Optional

import numpy as np

from synapforge.native.data.tokenizer import NativeTokenizer, get_tokenizer

# Reuse the exact same pinned-host helper as the parquet stream so the
# two iterators have identical memory characteristics. Importing here
# avoids duplicating the cupy probing logic.
from synapforge.native.data.parquet_stream import _alloc_pinned_int64


# ---------------------------------------------------------------------------
# row-iterator helper (also used by parquet_stream as a fallback for
# .jsonl files mixed into a parquet-list input).
# ---------------------------------------------------------------------------


def _iter_jsonl_text(path: str) -> Iterator[str]:
    """Yield row-text strings from one jsonl (or jsonl.gz) file.

    See module docstring for supported schemas. Empty / malformed lines
    are skipped, matching the legacy ``ParquetTokenStream._open_jsonl_iter``
    behaviour (real KD shards have a few stragglers).
    """
    if path.lower().endswith(".gz"):
        opener = gzip.open
    else:
        opener = open  # type: ignore[assignment]
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 1. {"text": ...}
            txt = obj.get("text") if isinstance(obj, dict) else None
            if isinstance(txt, str) and txt:
                yield txt
                continue
            # 2. {"content": ...}
            content = obj.get("content") if isinstance(obj, dict) else None
            if isinstance(content, str) and content:
                yield content
                continue
            # 3. {"messages": [...]}
            msgs = obj.get("messages") if isinstance(obj, dict) else None
            if isinstance(msgs, list):
                bits: list[str] = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role", "") or "")
                    ctn = m.get("content", "")
                    if isinstance(ctn, str) and ctn:
                        bits.append(f"<|{role}|>{ctn}")
                if bits:
                    yield "\n".join(bits)
                    continue
            # 4. {"prompt": ..., "completion": ...}
            if isinstance(obj, dict):
                prom = obj.get("prompt")
                comp = obj.get("completion")
                if isinstance(prom, str) and isinstance(comp, str):
                    yield f"{prom}\n{comp}"
                    continue


# ---------------------------------------------------------------------------
# main streamer
# ---------------------------------------------------------------------------


class NativeJsonlStream:
    """Iterator over JSONL files emitting (tokens_in, tokens_out) batches.

    Same constructor surface as ``NativeParquetStream``; see that class
    for argument-by-argument docs. The only difference is that
    ``glob_pattern`` matches ``*.jsonl`` / ``*.jsonl.gz``.
    """

    def __init__(
        self,
        glob_pattern: str,
        seq_len: int = 256,
        batch_size: int = 32,
        *,
        eot_id: Optional[int] = None,
        loop: bool = True,
        tokenizer: "str | NativeTokenizer" = "gpt2",
        shuffle_buffer: int = 0,
        shuffle_seed: int = 42,
        prefetch_factor: int = 0,
        num_workers: int = 4,
        cuda: bool = False,
        files_with_weights: "Optional[list[tuple[str, float]]]" = None,
    ) -> None:
        if files_with_weights is not None and len(files_with_weights) > 0:
            paths = [p for p, _ in files_with_weights]
            weights = [float(w) for _, w in files_with_weights]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(
                    "files_with_weights path(s) do not exist: "
                    + ", ".join(missing[:5])
                    + (f" (and {len(missing)-5} more)" if len(missing) > 5 else "")
                )
            total = sum(max(0.0, w) for w in weights)
            if total <= 0:
                raise ValueError(
                    f"files_with_weights total weight must be > 0; got {weights!r}"
                )
            self.files_with_weights = [
                (p, max(0.0, w) / total) for p, w in zip(paths, weights)
            ]
            self.files = list(paths)
        else:
            self.files = sorted(glob.glob(glob_pattern))
            if not self.files:
                raise FileNotFoundError(f"glob {glob_pattern!r} matched no files")
            self.files_with_weights = None

        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.loop = bool(loop)
        if isinstance(tokenizer, str):
            self._tok: NativeTokenizer = get_tokenizer(tokenizer)
            self.tokenizer_name = tokenizer
        else:
            self._tok = tokenizer
            self.tokenizer_name = getattr(tokenizer, "name", "<custom>")
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.shuffle_seed = int(shuffle_seed)
        self.prefetch_factor = int(prefetch_factor or 0)
        self.num_workers = max(1, int(num_workers))
        # Cuda flag: cupy-pinned allocations only when cupy is importable.
        try:
            import cupy as _cp  # noqa: F401
            self.cuda = bool(cuda)
        except ImportError:
            self.cuda = False

        if eot_id is None:
            eos = getattr(self._tok, "eos_token_id", None)
            self.eot_id = int(eos) if eos is not None else 50256
        else:
            self.eot_id = int(eot_id)

    # ------------------------------------------------------------------ raw

    def _iter_text_rows_raw(self) -> Iterator[str]:
        """Iterate row-text strings across all files, with epoch shuffle."""
        if self.files_with_weights is not None:
            yield from self._iter_text_rows_weighted()
            return

        epoch = 0
        while True:
            if self.shuffle_buffer > 1:
                epoch_rng = random.Random(self.shuffle_seed + epoch)
                files = list(self.files)
                epoch_rng.shuffle(files)
            else:
                files = self.files
            for path in files:
                yield from _iter_jsonl_text(path)
            epoch += 1
            if not self.loop:
                return

    def _iter_text_rows_weighted(self) -> Iterator[str]:
        """Weighted N-file mixture (mirror of NativeParquetStream)."""
        rng = random.Random(self.shuffle_seed)
        files = [p for p, _ in self.files_with_weights]  # type: ignore[union-attr]
        weights = [w for _, w in self.files_with_weights]  # type: ignore[union-attr]
        gens: list[Iterator[str] | None] = [_iter_jsonl_text(p) for p in files]
        seen_eof: set[int] = set()
        while True:
            idx = rng.choices(range(len(files)), weights=weights, k=1)[0]
            try:
                if gens[idx] is None:
                    gens[idx] = _iter_jsonl_text(files[idx])
                yield next(gens[idx])  # type: ignore[arg-type]
            except StopIteration:
                seen_eof.add(idx)
                gens[idx] = None
                if not self.loop and len(seen_eof) >= len(files):
                    return
                gens[idx] = _iter_jsonl_text(files[idx])
                try:
                    yield next(gens[idx])
                except StopIteration:
                    weights[idx] = 0.0
                    if all(w <= 0 for w in weights):
                        return

    def _iter_text_rows(self) -> Iterator[str]:
        """Apply Fisher-Yates streaming shuffle on top of the raw iterator."""
        raw_iter = self._iter_text_rows_raw()
        if self.shuffle_buffer <= 1:
            yield from raw_iter
            return
        rng = random.Random(self.shuffle_seed)
        buffer: list[str] = []
        K = int(self.shuffle_buffer)
        for row in raw_iter:
            if len(buffer) < K:
                buffer.append(row)
                continue
            idx = rng.randrange(K)
            yield buffer[idx]
            buffer[idx] = row
        rng.shuffle(buffer)
        yield from buffer

    def _iter_token_chunks(self) -> Iterator[list[int]]:
        """Yield length-(seq_len+1) integer windows."""
        win = self.seq_len + 1
        buf: list[int] = []
        for txt in self._iter_text_rows():
            ids = self._tok.encode(txt)
            if not ids:
                continue
            buf.extend(ids)
            buf.append(self.eot_id)
            while len(buf) >= win:
                yield buf[:win]
                buf = buf[win:]

    # ------------------------------------------------------------------ batch

    def _build_batch(
        self, chunks: "list[list[int]]"
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Same logic as NativeParquetStream._build_batch."""
        B = len(chunks)
        T1 = self.seq_len + 1
        if self.cuda:
            buf = _alloc_pinned_int64((B, T1))
        else:
            buf = None
        if buf is None:
            buf = np.empty((B, T1), dtype=np.int64)
        for i, c in enumerate(chunks):
            if len(c) == T1:
                buf[i] = c
            else:
                row = list(c)
                if len(row) < T1:
                    row.extend([self.eot_id] * (T1 - len(row)))
                buf[i] = row[:T1]
        tokens_in = np.ascontiguousarray(buf[:, :-1])
        tokens_out = np.ascontiguousarray(buf[:, 1:])
        return tokens_in, tokens_out

    # ------------------------------------------------------------------ iter

    def _iter_sync(self) -> Iterator["tuple[np.ndarray, np.ndarray]"]:
        chunks: list[list[int]] = []
        for c in self._iter_token_chunks():
            chunks.append(c)
            if len(chunks) >= self.batch_size:
                yield self._build_batch(chunks)
                chunks = []

    def _iter_prefetch(self) -> Iterator["tuple[np.ndarray, np.ndarray]"]:
        """Mirror of NativeParquetStream._iter_prefetch."""
        q_chunks: queue.Queue = queue.Queue(maxsize=self.num_workers * 2)
        q_batches: queue.Queue = queue.Queue(maxsize=max(2, self.prefetch_factor))
        _SENTINEL = object()

        class _ExcWrap:
            __slots__ = ("exc",)

            def __init__(self, exc: BaseException) -> None:
                self.exc = exc

        def _chunk_producer() -> None:
            try:
                chunks: list[list[int]] = []
                for c in self._iter_token_chunks():
                    chunks.append(c)
                    if len(chunks) >= self.batch_size:
                        q_chunks.put(chunks)
                        chunks = []
            except BaseException as exc:  # noqa: BLE001
                q_chunks.put(_ExcWrap(exc))
            finally:
                for _ in range(self.num_workers):
                    q_chunks.put(_SENTINEL)

        def _batch_worker() -> None:
            try:
                while True:
                    item = q_chunks.get()
                    if item is _SENTINEL:
                        q_batches.put(_SENTINEL)
                        return
                    if isinstance(item, _ExcWrap):
                        q_batches.put(item)
                        return
                    chunks_local = item  # type: ignore[assignment]
                    batch = self._build_batch(chunks_local)
                    q_batches.put(batch)
            except BaseException as exc:  # noqa: BLE001
                q_batches.put(_ExcWrap(exc))

        prod = threading.Thread(
            target=_chunk_producer, name="NativeJSONLProducer", daemon=True,
        )
        workers = [
            threading.Thread(
                target=_batch_worker,
                name=f"NativeJSONLWorker-{i}",
                daemon=True,
            )
            for i in range(self.num_workers)
        ]
        prod.start()
        for w in workers:
            w.start()

        sentinels_seen = 0
        try:
            while True:
                item = q_batches.get()
                if item is _SENTINEL:
                    sentinels_seen += 1
                    if sentinels_seen >= self.num_workers:
                        return
                    continue
                if isinstance(item, _ExcWrap):
                    raise item.exc
                yield item
        finally:
            for q in (q_chunks, q_batches):
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass

    def __iter__(self) -> Iterator["tuple[np.ndarray, np.ndarray]"]:
        if self.prefetch_factor >= 2:
            yield from self._iter_prefetch()
        else:
            yield from self._iter_sync()

    def __repr__(self) -> str:
        return (
            f"NativeJsonlStream(files={len(self.files)}, "
            f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"tokenizer={self.tokenizer_name!r}, "
            f"shuffle_buffer={self.shuffle_buffer}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"num_workers={self.num_workers}, "
            f"cuda={self.cuda})"
        )
