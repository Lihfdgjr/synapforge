"""``NativeParquetStream`` -- torch-free parquet -> token-pair iterator.

Logical port of ``synapforge.data.ParquetTokenStream`` that emits
**numpy** arrays instead of torch tensors. Multi-threaded pre-fetch with
a bounded queue. Optional pinned-host-memory allocation through
``cupy.cuda.runtime.hostAlloc`` so the trainer can issue async H2D
copies without needing torch.

Pipeline
--------
parquet rows (string text)
  -> NativeTokenizer encode (no specials)
  -> append eot_id between docs
  -> slide a length-(seq_len+1) window
  -> emit (tokens_in [B,T], tokens_out [B,T]) numpy int64 arrays
     (cupy when ``cuda=True`` and cupy is importable)

Constraints
-----------
* Zero ``import torch``.
* No HF transformers required for the streaming path -- that's the
  tokenizer's responsibility (``synapforge.native.data.tokenizer``).
* Pinned-memory path is opt-in (``cuda=True``) AND silent-no-op when
  cupy is unavailable, so unit tests work on dev boxes without GPU.
"""

from __future__ import annotations

import glob
import os
import queue
import random
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from synapforge.native.data.tokenizer import NativeTokenizer, get_tokenizer

if TYPE_CHECKING:
    pass


# Lazy cupy + pyarrow imports so the module loads on dev boxes without
# either installed. Both are required for the actual streaming path; the
# tests that only touch the API can mock them.
try:
    import pyarrow.parquet as _pq
except ImportError:
    _pq = None  # type: ignore[assignment]

try:
    import cupy as _cp  # type: ignore[import-not-found]
except ImportError:
    _cp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pinned-host buffer helper (optional cupy)
# ---------------------------------------------------------------------------


def _alloc_pinned_int64(shape: "tuple[int, int]") -> Optional[np.ndarray]:
    """Allocate a pinned-host ``np.ndarray`` via ``cupy.cuda.runtime.hostAlloc``.

    Returns the numpy view backed by the pinned region, or ``None`` if
    cupy is unavailable. The buffer is not auto-freed -- callers must
    keep a reference and let GC do it (cupy detaches the underlying
    pinned region when the array goes out of scope via the
    ``base.alloc.free`` chain). For the bounded-queue prefetch pattern
    here, that's fine: at most ``prefetch_factor`` buffers exist at any
    moment and they get reused as the consumer pops from the queue.

    Pinning is required for ``cudaMemcpyAsync`` to actually be async
    (Driver API silently falls back to sync on pageable host memory).
    """
    if _cp is None:
        return None
    n = int(shape[0]) * int(shape[1])
    if n <= 0:
        return None
    nbytes = n * 8  # int64
    # cuda.runtime.hostAlloc returns a raw uintptr_t; we wrap it as a
    # numpy array. Cupy 9+ has ``cupy.cuda.alloc_pinned_memory`` that
    # returns a managed object -- prefer that when present so the
    # underlying pointer is auto-freed when the buffer is released.
    try:
        mem = _cp.cuda.alloc_pinned_memory(nbytes)
        # ``mem`` is a ``cupy.cuda.PinnedMemoryPointer``; numpy can wrap
        # the underlying buffer via __array_interface__.
        # Safest path: ``np.frombuffer`` with the memoryview-able obj.
        arr = np.frombuffer(mem, dtype=np.int64).reshape(shape)
        # Hold a back-ref to ``mem`` so it isn't GC'd while ``arr`` is
        # alive; numpy's ``base`` slot is exactly the right hook.
        arr._pinned_base = mem  # type: ignore[attr-defined]
        return arr
    except Exception:
        # Older cupy or driver issue -- fall back to a regular numpy buf.
        # The trainer's H2D copy will still work; just synchronously.
        return None


# ---------------------------------------------------------------------------
# main streamer
# ---------------------------------------------------------------------------


class NativeParquetStream:
    """Infinite iterator over (tokens_in, tokens_out) numpy/cupy batches.

    Args
    ----
    glob_pattern
        Shell-style glob of parquet files. Each file must have a single
        string column (we auto-pick the first string-typed col, with
        preference for ``text/content/raw_content/document/code``).
    seq_len
        Tokens per training example T. We slide a length-T+1 window so
        we can split into in/out cleanly.
    batch_size
        Examples per batch B.
    text_column
        Optional explicit column name; auto-detect if None.
    eot_id
        End-of-text token id used as document separator. If None
        (default) auto-derives from the tokenizer's ``eos_token_id``.
    loop
        If True, loop over files forever (training loop never raises
        StopIteration). Each loop pass also re-shuffles the file order
        when ``shuffle_buffer > 1``.
    tokenizer
        Either a string (passed to ``NativeTokenizer``) or a
        pre-constructed ``NativeTokenizer`` instance. Default is
        ``"gpt2"`` so the bench / smoke path works without a Qwen
        download.
    shuffle_buffer
        Streaming Fisher-Yates reservoir size. Default 0 (legacy
        deterministic order). Same semantics as
        ``ParquetTokenStream.shuffle_buffer``.
    shuffle_seed
        Deterministic seed for the reservoir RNG. Default 42.
    prefetch_factor
        Number of batches to pre-fetch in background threads when
        ``> 0``. ``0`` keeps the legacy synchronous path. ``>= 2``
        spawns ``num_workers`` producer threads + bounded
        ``queue.Queue(maxsize=prefetch_factor)``.
    num_workers
        Number of background threads (only used when
        ``prefetch_factor >= 2``). Default 4.
    cuda
        When True, allocate yielded arrays in cupy-pinned host memory
        for async H2D copy. Silent no-op when cupy is unavailable.
    """

    def __init__(
        self,
        glob_pattern: str,
        seq_len: int = 256,
        batch_size: int = 32,
        *,
        text_column: Optional[str] = None,
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
        if _pq is None:
            raise ImportError(
                "pyarrow is required by NativeParquetStream; pip install pyarrow"
            )
        # Either explicit list (with weights) or a glob.
        self.files_with_weights: "Optional[list[tuple[str, float]]]" = None
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

        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.text_column = text_column
        self.loop = bool(loop)
        # Tokenizer: accept name or pre-built instance.
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
        self.cuda = bool(cuda) and (_cp is not None)

        # EOT default = tokenizer eos
        if eot_id is None:
            eos = getattr(self._tok, "eos_token_id", None)
            self.eot_id = int(eos) if eos is not None else 50256
        else:
            self.eot_id = int(eot_id)

    # ------------------------------------------------------------------ raw

    def _open_file_iter(self, path: str) -> Iterator[str]:
        """Open one parquet (or jsonl) and yield rows as strings."""
        low = path.lower()
        if low.endswith(".jsonl") or low.endswith(".jsonl.gz"):
            yield from self._open_jsonl_iter(path)
            return
        pf = _pq.ParquetFile(path)  # type: ignore[union-attr]
        col = self.text_column
        if col is None:
            for f in pf.schema_arrow:
                if str(f.type) == "string" and f.name in (
                    "text", "content", "raw_content", "document", "code",
                ):
                    col = f.name
                    break
            if col is None:
                for f in pf.schema_arrow:
                    if str(f.type) == "string":
                        col = f.name
                        break
            if col is None:
                raise RuntimeError(
                    f"no string column in {path}; got {pf.schema_arrow}"
                )
        for batch in pf.iter_batches(batch_size=64, columns=[col]):
            for s in batch.column(col).to_pylist():
                if s:
                    yield s

    def _open_jsonl_iter(self, path: str) -> Iterator[str]:
        """Iterator helper for jsonl files when mixed with parquets."""
        # Re-implemented in NativeJsonlStream; just redirect to keep code
        # paths consistent. Avoids duplicate maintenance of the schema
        # extraction.
        from synapforge.native.data.jsonl_stream import _iter_jsonl_text  # local imp.
        yield from _iter_jsonl_text(path)

    def _iter_text_rows_raw(self) -> Iterator[str]:
        """Raw row iterator across all files. Re-shuffled per-epoch when
        ``shuffle_buffer > 1`` so deterministic-data-order P24 divergence
        cannot recur (see ``feedback_data_ordering_divergence_2026q2``).
        """
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
                yield from self._open_file_iter(path)
            epoch += 1
            if not self.loop:
                return

    def _iter_text_rows_weighted(self) -> Iterator[str]:
        """Weighted multi-file mixture iterator (same semantics as
        ``ParquetTokenStream._iter_text_rows_weighted``)."""
        rng = random.Random(self.shuffle_seed)
        files = [p for p, _ in self.files_with_weights]  # type: ignore[union-attr]
        weights = [w for _, w in self.files_with_weights]  # type: ignore[union-attr]
        gens: list[Iterator[str] | None] = [self._open_file_iter(p) for p in files]
        seen_eof: set[int] = set()
        while True:
            idx = rng.choices(range(len(files)), weights=weights, k=1)[0]
            try:
                if gens[idx] is None:
                    gens[idx] = self._open_file_iter(files[idx])
                yield next(gens[idx])  # type: ignore[arg-type]
            except StopIteration:
                seen_eof.add(idx)
                gens[idx] = None
                if not self.loop and len(seen_eof) >= len(files):
                    return
                gens[idx] = self._open_file_iter(files[idx])
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
        """Yield length-(seq_len+1) integer windows from the token stream."""
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
        """Materialise a (B, T+1) array and split into (in, out)."""
        B = len(chunks)
        T1 = self.seq_len + 1
        # Pinned-host alloc (cupy) when cuda=True; else plain numpy.
        if self.cuda:
            buf = _alloc_pinned_int64((B, T1))
        else:
            buf = None
        if buf is None:
            buf = np.empty((B, T1), dtype=np.int64)
        for i, c in enumerate(chunks):
            # Each chunk may be slightly shorter on the LAST batch of a
            # non-looping iter; pad with eot_id on the right (numpy will
            # error if we try to broadcast a short list into a row).
            if len(c) == T1:
                buf[i] = c
            else:
                # Defensive: real chunks are always T1; this only triggers
                # if the caller hand-feeds a partial batch.
                row = list(c)
                if len(row) < T1:
                    row.extend([self.eot_id] * (T1 - len(row)))
                buf[i] = row[:T1]
        # Slice -> (B, T) views; copy() to ensure both halves are
        # contiguous (sliced views share the parent buf otherwise; the
        # trainer's downstream copy-to-device handles either, but
        # downstream codepaths sometimes assume independent buffers).
        tokens_in = np.ascontiguousarray(buf[:, :-1])
        tokens_out = np.ascontiguousarray(buf[:, 1:])
        return tokens_in, tokens_out

    # ------------------------------------------------------------------ iter

    def _iter_sync(self) -> Iterator["tuple[np.ndarray, np.ndarray]"]:
        """Single-thread iterator -- each next() blocks on parquet+tokenizer."""
        chunks: list[list[int]] = []
        for c in self._iter_token_chunks():
            chunks.append(c)
            if len(chunks) >= self.batch_size:
                yield self._build_batch(chunks)
                chunks = []

    def _iter_prefetch(self) -> Iterator["tuple[np.ndarray, np.ndarray]"]:
        """N-worker producer + bounded queue consumer.

        We run a SINGLE chunk producer (token windows are inherently
        sequential -- partial-buffer state spans rows) but let multiple
        worker threads concurrently call ``_build_batch`` to amortise
        the numpy copy + pinned alloc. The queue caps at
        ``prefetch_factor`` to bound memory.

        Termination: the producer pushes one ``_SENTINEL`` per worker
        when ``loop=False`` exhausts the row iterator; consumer counts
        sentinels.

        Exception channel: producer / worker exceptions are wrapped in
        a dedicated ``_ExcWrap`` instance so the consumer can dispatch
        on ``isinstance`` (NOT tuple-equality, which would trip numpy's
        elementwise-compare warning when a normal ``(x_arr, y_arr)``
        batch flows through the queue).
        """
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
                # Send one sentinel per worker so they all exit.
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
            target=_chunk_producer, name="NativePQProducer", daemon=True,
        )
        workers = [
            threading.Thread(
                target=_batch_worker,
                name=f"NativePQWorker-{i}",
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
            # Drain so producer's last put() unblocks; daemon threads
            # exit with the process anyway.
            for q in (q_chunks, q_batches):
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass

    def __iter__(self) -> Iterator["tuple[np.ndarray, np.ndarray]"]:
        """Yield (tokens_in [B,T], tokens_out [B,T]) int64 numpy arrays."""
        if self.prefetch_factor >= 2:
            yield from self._iter_prefetch()
        else:
            yield from self._iter_sync()

    def __repr__(self) -> str:
        ww = ""
        if self.files_with_weights is not None:
            top = ", ".join(
                f"{os.path.basename(p)}:{w:.2f}"
                for p, w in self.files_with_weights[:3]
            )
            extra = "" if len(self.files_with_weights) <= 3 else \
                f"...+{len(self.files_with_weights) - 3} more"
            ww = f", files_with_weights=[{top}{extra}]"
        return (
            f"NativeParquetStream(files={len(self.files)}, "
            f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"tokenizer={self.tokenizer_name!r}, "
            f"shuffle_buffer={self.shuffle_buffer}, "
            f"shuffle_seed={self.shuffle_seed}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"num_workers={self.num_workers}, "
            f"cuda={self.cuda}{ww})"
        )
