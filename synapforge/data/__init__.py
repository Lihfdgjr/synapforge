"""sf.data — minimal parquet -> token-pair stream for synapforge training.

Goal: feed sf.train without pulling in the torch.utils.data DataLoader
multi-process machinery (which interacts badly with our plasticity hooks
and CUDA contexts in v0.1). Just a generator yielding (in, out) tensors.

Pipeline:
  parquet rows (string text)
    -> AutoTokenizer encode (any HF-compatible tokenizer; vocab is
       whatever the tokenizer reports, e.g. 50257 for gpt2 / 151643 for
       qwen2.5)
    -> append EOT token between docs (auto-derived from the tokenizer's
       eos_token_id; no hard-coded 50256)
    -> slide a window of seq_len+1 tokens, batch up B of them
    -> emit (tokens_in[B,T], tokens_out[B,T]) where out = in shifted left by 1.

Files are streamed row-by-row (pyarrow batch_size=64) so memory stays bounded
even on the 900k-row WT-103 train shard. The vocab_size argument is purely
documentation; the actual vocab is whatever the tokenizer reports.

Public API:
    >>> ds = ParquetTokenStream("/workspace/data/wt103_raw/train-*.parquet",
    ...                         seq_len=256, batch_size=32)
    >>> it = iter(ds)
    >>> x, y = next(it)              # both (32, 256) int64 cpu tensors
    >>> assert (x[:, 1:] == y[:, :-1]).all()  # next-token target shape
"""

from __future__ import annotations

import glob
import random
from collections.abc import Iterator

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None
import torch

_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(name: str = "gpt2"):
    """Lazy-load tokenizer (any HF AutoTokenizer-compatible); per-name cached.

    Cached PER tokenizer name -- the previous single-global cache returned
    a stale tokenizer when the trainer switched between GPT-2 (vocab=50257)
    and Qwen (vocab=151643) within the same process, silently mis-tokenising
    the second stream.
    """
    cached = _TOKENIZER_CACHE.get(name)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    _TOKENIZER_CACHE[name] = tok
    return tok


class ParquetTokenStream:
    """Infinite-loop iterator over (tokens_in, tokens_out) batches.

    Args:
        glob_pattern: shell-style glob of parquet files. Each file must have
            a single string column (we auto-pick the first string-typed col).
        seq_len: tokens per training example T. We slide a length-T+1 window
            so we can split into in/out cleanly.
        batch_size: examples per batch B.
        vocab_size: documentation only; tokenizer vocab is authoritative.
        text_column: optional explicit column name; auto-detect if None.
        eot_id: end-of-text token id used as document separator. If None
            (default) auto-derives from the tokenizer's eos_token_id so
            we don't accidentally inject GPT-2's 50256 magic into a
            different vocab (e.g. Qwen).
        loop: if True, loop over files forever (training loop never raises
            StopIteration). Each loop pass also re-shuffles the file
            order when ``shuffle_buffer > 1`` so consecutive epochs
            don't replay the same lexical sequence.
        shuffle_buffer: P24 (MASTER_PLAN.md §6) — size of the streaming
            reservoir used in ``_iter_text_rows``. ``> 1`` activates
            Fisher-Yates streaming shuffle: fill a fixed-size buffer of
            K rows then yield-then-replace at a random index. ``<= 1``
            (or 0) yields rows in raw lexical order (LEGACY broken
            default). Three Run 3a/3b/3c divergences at step ~2500 were
            traced to deterministic ordering; default is now 10000 in
            ``train_100m_kd.py`` so callers don't have to opt in.
        shuffle_seed: deterministic seed for the reservoir RNG and the
            per-epoch file-order shuffle. Default 42. Same seed +
            same buffer size ⇒ identical yield sequence.
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
    ) -> None:
        self.files = sorted(glob.glob(glob_pattern))
        if not self.files:
            raise FileNotFoundError(f"glob {glob_pattern!r} matched no files")
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.vocab_size = int(vocab_size)
        self.text_column = text_column
        self.loop = bool(loop)
        self.tokenizer_name = tokenizer_name
        # P24: shuffle config -- accept None as "off" for resilience to
        # callers that build kwargs from dicts.
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.shuffle_seed = int(shuffle_seed)
        self._tok = _get_tokenizer(tokenizer_name)
        # Default EOT to the tokenizer's eos_token_id, NOT the GPT-2 magic
        # number 50256 -- using a hard-coded id with a Qwen tokenizer would
        # inject a spurious mid-vocab token between docs and corrupt
        # training. eot_id can still be passed explicitly to override.
        if eot_id is None:
            eos = getattr(self._tok, "eos_token_id", None)
            if eos is None:
                # Fall back to GPT-2 magic only if the tokenizer truly has
                # no eos (vanishingly rare).
                eos = 50256
            self.eot_id = int(eos)
        else:
            self.eot_id = int(eot_id)

    # ---------------------------------------------------------------- internal

    def _iter_text_rows_raw(self) -> Iterator[str]:
        """Raw row iterator: yield text strings from parquets in (possibly
        per-epoch shuffled) file order. With ``shuffle_buffer <= 1`` this
        is the LEGACY deterministic path.

        With ``shuffle_buffer > 1`` and ``loop=True``, the file list is
        re-shuffled per epoch using a ``Random(shuffle_seed + epoch)``
        instance so different passes through the corpus never replay the
        same lexical sequence (the deterministic-ordering divergence
        root cause logged as P24 in MASTER_PLAN.md §6).
        """
        epoch = 0
        while True:
            if self.shuffle_buffer > 1:
                epoch_rng = random.Random(self.shuffle_seed + epoch)
                files = list(self.files)
                epoch_rng.shuffle(files)
            else:
                files = self.files
            for path in files:
                pf = pq.ParquetFile(path)
                col = self.text_column
                if col is None:
                    # auto-pick first string column
                    for f in pf.schema_arrow:
                        if str(f.type) == "string":
                            col = f.name
                            break
                    if col is None:
                        raise RuntimeError(
                            f"no string column in {path}; "
                            f"got {pf.schema_arrow}"
                        )
                # batch_size here is parquet-row batches, not training batches
                for batch in pf.iter_batches(batch_size=64, columns=[col]):
                    for s in batch.column(col).to_pylist():
                        if s:
                            yield s
            epoch += 1
            if not self.loop:
                return

    def _iter_text_rows(self) -> Iterator[str]:
        """Yield text strings from all parquet files.

        With ``shuffle_buffer <= 1`` this is a passthrough of
        ``_iter_text_rows_raw`` (legacy deterministic order).

        With ``shuffle_buffer > 1`` this implements a streaming
        Fisher-Yates reservoir shuffle: fill a list of K rows, then on
        each subsequent input row pop a random slot, yield it, and put
        the new row in its place. Constant memory O(K) and decorrelates
        consecutive yields. At iter end the reservoir is drained in
        random order so no rows are lost. The per-iter RNG is seeded by
        ``shuffle_seed`` so the same seed yields the same sequence.
        """
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
        # Drain remaining buffer at end (only happens when loop=False).
        rng.shuffle(buffer)
        yield from buffer

    def _iter_token_chunks(self) -> Iterator[list[int]]:
        """Yield length-(seq_len+1) integer windows from the token stream."""
        win = self.seq_len + 1
        buf: list[int] = []
        for txt in self._iter_text_rows():
            ids = self._tok.encode(txt, add_special_tokens=False)
            if not ids:
                continue
            buf.extend(ids)
            buf.append(self.eot_id)
            while len(buf) >= win:
                yield buf[:win]
                buf = buf[win:]

    # ---------------------------------------------------------------- public

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (tokens_in [B,T], tokens_out [B,T]) int64 cpu tensors."""
        chunks: list[list[int]] = []
        for c in self._iter_token_chunks():
            chunks.append(c)
            if len(chunks) >= self.batch_size:
                arr = torch.tensor(chunks, dtype=torch.long)  # (B, T+1)
                tokens_in = arr[:, :-1].contiguous()
                tokens_out = arr[:, 1:].contiguous()
                yield tokens_in, tokens_out
                chunks = []

    def __repr__(self) -> str:
        return (
            f"ParquetTokenStream(files={len(self.files)}, "
            f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"tokenizer={self.tokenizer_name!r}, "
            f"shuffle_buffer={self.shuffle_buffer}, "
            f"shuffle_seed={self.shuffle_seed})"
        )


# ---------------------------------------------------------------------------
# val-set split for honest TTT-leak-free reporting (P3 in MASTER_PLAN.md §6)
# ---------------------------------------------------------------------------


class _RowSubsetStream:
    """Iterates a parent ``ParquetTokenStream`` and emits only chunks whose
    deterministic row index is in ``keep_set``.

    Chunks (length seq_len+1 token windows) are filtered by an integer
    "row index" -- the position in the parent stream's chunk iterator.
    With a denominator D and a keep-set ``K subset {0..D-1}`` (e.g.
    K={0,1,2,3} for the TTT side, K={4} for the holdout side at D=5),
    every D-th chunk is routed deterministically to one side.

    Why row-level (not batch-level): the TTT mixin trains on the *chunks*
    inside a batch; if we split at batch granularity the TTT side would
    train on chunks the holdout side will never see only because they
    happened to be in the same batch -- not a true disjoint split.
    Row-level filtering is the cleanest no-overlap guarantee.

    The iterator re-batches the kept chunks into ``batch_size`` groups so
    callers can use ``next(iter(side))`` exactly like the parent stream.
    """

    def __init__(
        self,
        parent: "ParquetTokenStream",
        keep_indices: "set[int]",
        denom: int,
        side: str = "ttt",
    ) -> None:
        self._parent = parent
        self._keep = set(int(i) for i in keep_indices)
        self._denom = int(denom)
        self._side = str(side)
        if self._denom <= 0:
            raise ValueError("denom must be > 0")
        if not self._keep:
            raise ValueError("keep_indices must be non-empty")
        if any(i < 0 or i >= self._denom for i in self._keep):
            raise ValueError(
                f"keep_indices {sorted(self._keep)!r} must all lie in "
                f"[0, {self._denom})"
            )

    @property
    def batch_size(self) -> int:
        return self._parent.batch_size

    @property
    def seq_len(self) -> int:
        return self._parent.seq_len

    @property
    def keep_indices(self) -> set[int]:
        return set(self._keep)

    @property
    def denom(self) -> int:
        return self._denom

    @property
    def side(self) -> str:
        return self._side

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        chunks: list[list[int]] = []
        for row_idx, chunk in enumerate(self._parent._iter_token_chunks()):
            if (row_idx % self._denom) not in self._keep:
                continue
            chunks.append(chunk)
            if len(chunks) >= self._parent.batch_size:
                arr = torch.tensor(chunks, dtype=torch.long)  # (B, T+1)
                tokens_in = arr[:, :-1].contiguous()
                tokens_out = arr[:, 1:].contiguous()
                yield tokens_in, tokens_out
                chunks = []

    def __repr__(self) -> str:
        return (
            f"_RowSubsetStream(side={self._side!r}, "
            f"keep={sorted(self._keep)!r}/{self._denom}, "
            f"parent={self._parent!r})"
        )


def split_val_stream(
    parent: "ParquetTokenStream",
    ttt_fraction: float = 0.8,
    denom: int = 5,
) -> tuple["_RowSubsetStream", "_RowSubsetStream"]:
    """Split a val ``ParquetTokenStream`` into disjoint TTT and holdout sides.

    The split is **row-deterministic**: each length-(seq_len+1) chunk in
    the parent stream's chunk iterator is routed to exactly one side
    based on ``row_idx % denom``. The default ``denom=5`` with
    ``ttt_fraction=0.8`` puts 4 of every 5 chunks on the TTT side and 1
    on the holdout side.

    P3 in ``docs/MASTER_PLAN.md`` §6: ``SelfLearnMixin.adapt_on_failures``
    runs an inner step on the TTT side, which artificially drops val ppl
    on that set. The holdout side is **never touched by TTT** so its ppl
    is the honest signal phase_manager should gate on.

    Returns:
        ``(val_ttt, val_holdout)`` -- two ``_RowSubsetStream`` objects
        whose row sets are disjoint and whose union equals the parent.

    Args:
        parent: a ``ParquetTokenStream`` (or any object with the same
            ``_iter_token_chunks()`` + ``batch_size`` interface).
        ttt_fraction: fraction of chunks to route to the TTT side. Must
            lie in (0, 1). Default 0.8.
        denom: denominator of the integer split. ``num = round(
            ttt_fraction * denom)`` chunks per ``denom`` go to TTT. The
            default ``denom=5`` gives an exact 4:1 split for
            ``ttt_fraction=0.8``.
    """
    if not (0.0 < ttt_fraction < 1.0):
        raise ValueError(
            f"ttt_fraction must be strictly in (0, 1); got {ttt_fraction!r}"
        )
    denom = int(denom)
    if denom < 2:
        raise ValueError(f"denom must be >= 2; got {denom!r}")
    num_ttt = int(round(ttt_fraction * denom))
    if num_ttt <= 0 or num_ttt >= denom:
        raise ValueError(
            f"ttt_fraction={ttt_fraction!r} with denom={denom} yields "
            f"num_ttt={num_ttt}; must be in (0, {denom})"
        )
    ttt_idx = set(range(num_ttt))               # {0..num_ttt-1}
    holdout_idx = set(range(num_ttt, denom))    # {num_ttt..denom-1}
    ttt_stream = _RowSubsetStream(parent, ttt_idx, denom, side="ttt")
    holdout_stream = _RowSubsetStream(
        parent, holdout_idx, denom, side="holdout"
    )
    return ttt_stream, holdout_stream


__all__ = ["ParquetTokenStream", "split_val_stream"]
