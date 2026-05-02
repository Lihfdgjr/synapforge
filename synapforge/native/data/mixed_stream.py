"""``WeightedMixedStream`` -- weighted round-robin over heterogeneous streams.

Replaces the ``--data-files PATH:W,PATH:W,...`` parsing logic of
``synapforge.data.ParquetTokenStream.files_with_weights`` with a torch-free
native version. Designed for two use cases:

1. **Quality-corpus pretrain** -- mix three parquets (FineWeb-Edu,
   WikiText-103) and one JSONL (KD-distill) at fixed weights so the
   realised token mixture matches the requested mixture regardless of
   shard size.
2. **Heterogeneous inputs** -- a single ``WeightedMixedStream`` can mix
   ``NativeParquetStream`` and ``NativeJsonlStream`` instances together,
   which the legacy ``ParquetTokenStream.files_with_weights`` does only
   indirectly (by lifting jsonl reads into its own ``_open_file_iter``).

Construction modes
------------------
A. **Path-with-weights mode** (matches legacy behaviour):

   >>> mix = WeightedMixedStream.from_paths_with_weights(
   ...     [("data/fwe.parquet", 0.4), ("data/wt103.parquet", 0.3),
   ...     ("data/kd.jsonl", 0.3)],
   ...     seq_len=256, batch_size=32, tokenizer="gpt2",
   ... )

   Each path is auto-routed to ``NativeParquetStream`` or
   ``NativeJsonlStream`` based on extension; per-source streams are
   then mixed at the row level using ``random.choices`` weighted by
   the requested weights.

B. **Pre-built sources mode** (advanced):

   >>> p = NativeParquetStream("...parquet", seq_len=256, ...)
   >>> j = NativeJsonlStream("...jsonl", seq_len=256, ...)
   >>> mix = WeightedMixedStream([(p, 0.6), (j, 0.4)],
   ...                          seq_len=256, batch_size=32)

   The caller controls the per-source streams entirely. Weights are
   applied at the *batch* row level so the mixing is the same as Mode
   A but over user-supplied iterators.

Constraints
-----------
* Zero ``import torch``.
* Mixing happens at the **row** level, not at the batch level. This
  guarantees the realised token mixture matches the requested weights
  even if the per-source row sizes differ wildly (FineWeb-Edu rows are
  ~10x longer than WikiText-103 rows).
* Sources are looped infinitely by default (``loop=True``); when one
  source's underlying stream raises StopIteration we re-open it. With
  ``loop=False`` the iterator stops once *every* source has signalled
  EOF at least once.
"""

from __future__ import annotations

import os
import queue
import random
import threading
from collections.abc import Iterator
from typing import Optional, Union

import numpy as np

from synapforge.native.data.jsonl_stream import NativeJsonlStream
from synapforge.native.data.parquet_stream import (
    NativeParquetStream,
    _alloc_pinned_int64,
)
from synapforge.native.data.tokenizer import NativeTokenizer, get_tokenizer

# Forward type alias for the per-source stream union; the actual streams
# only need to expose ``_iter_token_chunks()`` and ``loop`` -- duck typing.
_Stream = Union[NativeParquetStream, NativeJsonlStream]


class WeightedMixedStream:
    """Weighted row-level mixture over multiple sub-streams.

    Args
    ----
    sources
        ``[(source, weight), ...]``. ``source`` may be a pre-built
        ``NativeParquetStream`` / ``NativeJsonlStream``. ``weight``
        floats are normalised internally; they need not sum to 1.0.
    seq_len, batch_size, eot_id, tokenizer, prefetch_factor, num_workers,
    cuda
        Same as ``NativeParquetStream``. **Important**: when ``sources``
        carries pre-built streams, the streams' own ``seq_len`` and
        ``batch_size`` are IGNORED -- this stream resamples token chunks
        from each source at the row level then re-batches at THIS
        stream's ``batch_size``. If you want batch-level mixing, build a
        single ``NativeParquetStream`` / ``NativeJsonlStream`` with
        ``files_with_weights`` instead (cheaper).
    shuffle_buffer, shuffle_seed
        Streaming Fisher-Yates buffer applied AFTER row-level mixing
        but BEFORE batch assembly. Default 0 (no shuffle).
    """

    def __init__(
        self,
        sources: "list[tuple[_Stream, float]]",
        seq_len: int = 256,
        batch_size: int = 32,
        *,
        eot_id: Optional[int] = None,
        tokenizer: "Optional[Union[str, NativeTokenizer]]" = None,
        shuffle_buffer: int = 0,
        shuffle_seed: int = 42,
        prefetch_factor: int = 0,
        num_workers: int = 4,
        cuda: bool = False,
    ) -> None:
        if not sources:
            raise ValueError("WeightedMixedStream needs at least one source")
        weights = [float(w) for _, w in sources]
        total = sum(max(0.0, w) for w in weights)
        if total <= 0:
            raise ValueError(f"weights must sum > 0; got {weights!r}")
        self.sources: list[tuple[_Stream, float]] = [
            (s, max(0.0, w) / total) for s, w in sources
        ]
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.shuffle_seed = int(shuffle_seed)
        self.prefetch_factor = int(prefetch_factor or 0)
        self.num_workers = max(1, int(num_workers))
        # cupy probe -- pinned-host alloc is best-effort.
        try:
            import cupy as _cp  # noqa: F401
            self.cuda = bool(cuda)
        except ImportError:
            self.cuda = False
        # Resolve eot_id: prefer explicit; else first source's eot; else
        # tokenizer eos; else GPT-2 magic.
        if eot_id is not None:
            self.eot_id = int(eot_id)
        else:
            first_eot = getattr(self.sources[0][0], "eot_id", None)
            if first_eot is not None:
                self.eot_id = int(first_eot)
            elif tokenizer is not None:
                tok = (
                    get_tokenizer(tokenizer)
                    if isinstance(tokenizer, str)
                    else tokenizer
                )
                self.eot_id = int(getattr(tok, "eos_token_id", 50256))
            else:
                self.eot_id = 50256

    # ------------------------------------------------------------------

    @classmethod
    def from_paths_with_weights(
        cls,
        files_with_weights: "list[tuple[str, float]]",
        seq_len: int = 256,
        batch_size: int = 32,
        *,
        tokenizer: "str | NativeTokenizer" = "gpt2",
        shuffle_buffer: int = 0,
        shuffle_seed: int = 42,
        prefetch_factor: int = 0,
        num_workers: int = 4,
        cuda: bool = False,
        loop_sources: bool = True,
    ) -> "WeightedMixedStream":
        """Construct from ``[(path, weight), ...]`` -- the ``--data-files`` shape.

        Each path is auto-routed:
          * ``*.jsonl`` / ``*.jsonl.gz`` -> ``NativeJsonlStream``
          * ``*.parquet`` (or anything else) -> ``NativeParquetStream``

        Per-source streams are built with the same ``seq_len``,
        ``shuffle_buffer``, etc. as the parent mix.
        """
        if isinstance(tokenizer, str):
            tok_obj = get_tokenizer(tokenizer)
        else:
            tok_obj = tokenizer
        sources: list[tuple[_Stream, float]] = []
        for path, w in files_with_weights:
            if w <= 0:
                continue
            low = path.lower()
            if low.endswith(".jsonl") or low.endswith(".jsonl.gz"):
                src: _Stream = NativeJsonlStream(
                    path,
                    seq_len=seq_len,
                    batch_size=batch_size,  # unused at sub-stream level
                    tokenizer=tok_obj,
                    shuffle_buffer=shuffle_buffer,
                    shuffle_seed=shuffle_seed,
                    loop=loop_sources,
                    cuda=False,  # sub-streams allocate via parent
                )
            else:
                src = NativeParquetStream(
                    path,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    tokenizer=tok_obj,
                    shuffle_buffer=shuffle_buffer,
                    shuffle_seed=shuffle_seed,
                    loop=loop_sources,
                    cuda=False,
                )
            sources.append((src, float(w)))
        if not sources:
            raise ValueError(
                f"all weights are zero; got {files_with_weights!r}"
            )
        return cls(
            sources,
            seq_len=seq_len,
            batch_size=batch_size,
            eot_id=tok_obj.eos_token_id,
            tokenizer=tok_obj,
            shuffle_buffer=0,  # sub-streams already shuffle; mixing
                              # is independent
            shuffle_seed=shuffle_seed,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
            cuda=cuda,
        )

    # ------------------------------------------------------------------

    def _iter_chunks_weighted(self) -> Iterator[list[int]]:
        """Round-robin token chunks from sources at the requested weights.

        Each step picks a source via ``random.choices(sources, weights)``,
        pulls one token-chunk from that source's
        ``_iter_token_chunks()``, yields it. When a sub-iterator is
        exhausted (only happens with ``loop=False``), we drop its weight
        to 0 and continue with the rest; the iterator terminates when
        all weights are zero.
        """
        rng = random.Random(self.shuffle_seed)
        gens: list[Optional[Iterator[list[int]]]] = [
            iter(s._iter_token_chunks()) for s, _ in self.sources
        ]
        weights = [w for _, w in self.sources]
        while True:
            # All sources exhausted -> done.
            if all(w <= 0 for w in weights):
                return
            idx = rng.choices(
                range(len(self.sources)), weights=weights, k=1
            )[0]
            try:
                if gens[idx] is None:
                    return
                yield next(gens[idx])  # type: ignore[arg-type]
            except StopIteration:
                # Sub-source signalled EOF. If looping is on at the
                # sub-source level the original would never raise; this
                # branch only fires when the user passed loop=False.
                gens[idx] = None
                weights[idx] = 0.0

    def _iter_chunks(self) -> Iterator[list[int]]:
        """Apply optional Fisher-Yates buffer over the weighted chunk stream."""
        raw = self._iter_chunks_weighted()
        if self.shuffle_buffer <= 1:
            yield from raw
            return
        rng = random.Random(self.shuffle_seed + 1)
        buf: list[list[int]] = []
        K = int(self.shuffle_buffer)
        for c in raw:
            if len(buf) < K:
                buf.append(c)
                continue
            i = rng.randrange(K)
            yield buf[i]
            buf[i] = c
        rng.shuffle(buf)
        yield from buf

    # ------------------------------------------------------------------ batch

    def _build_batch(
        self, chunks: "list[list[int]]"
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Identical logic to NativeParquetStream._build_batch."""
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
        for c in self._iter_chunks():
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
                for c in self._iter_chunks():
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
            target=_chunk_producer, name="MixedProducer", daemon=True,
        )
        workers = [
            threading.Thread(
                target=_batch_worker,
                name=f"MixedWorker-{i}",
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
        head = ", ".join(
            f"{type(s).__name__}(files={len(getattr(s, 'files', []))}):{w:.2f}"
            for s, w in self.sources[:3]
        )
        extra = "" if len(self.sources) <= 3 else f"+{len(self.sources)-3} more"
        return (
            f"WeightedMixedStream(sources=[{head}{extra}], "
            f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"shuffle_buffer={self.shuffle_buffer}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"num_workers={self.num_workers}, cuda={self.cuda}, "
            f"eot_id={self.eot_id})"
        )


# Convenience -- mirror the ``--data-files`` parsing in train_100m_kd.py
# so the trainer can swap drop-in replace.
def parse_data_files_arg(
    arg: str,
) -> "list[tuple[str, float]]":
    """Parse a ``--data-files`` string into ``[(path, weight), ...]``.

    Format: ``PATH1:W1,PATH2:W2,...`` -- exactly what
    ``train_100m_kd.py`` accepts on the command line. Whitespace inside
    each part is stripped; empty parts are skipped. Weights must be
    positive floats.

    >>> parse_data_files_arg("a.parquet:0.4, b.parquet:0.6")
    [('a.parquet', 0.4), ('b.parquet', 0.6)]
    """
    out: list[tuple[str, float]] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"--data-files entry {part!r} missing ':WEIGHT' suffix"
            )
        path, w = part.rsplit(":", 1)
        path = path.strip()
        try:
            w_f = float(w.strip())
        except ValueError as e:
            raise ValueError(
                f"--data-files entry {part!r}: weight {w!r} not a float"
            ) from e
        if w_f <= 0:
            raise ValueError(
                f"--data-files entry {part!r}: weight {w_f!r} must be > 0"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--data-files entry {path!r} does not exist"
            )
        out.append((path, w_f))
    if not out:
        raise ValueError(f"--data-files arg {arg!r} parsed to empty list")
    return out
