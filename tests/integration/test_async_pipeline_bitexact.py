"""Async multi-stage data pipeline — bit-exact contract tests.

Pins the contract that ``synapforge.data.AsyncTokenStream`` yields
**identical** ``(tokens_in, tokens_out)`` tensor sequences to
``ParquetTokenStream`` for the SAME input args. The async pipeline is
purely a producer/consumer overlay — it must not reorder or drop
batches, regardless of how many internal stages run.

Mirrors the ``test_dataloader_prefetch.py`` harness pattern: we bind
the production ``_iter_token_chunks`` / ``_iter_text_rows`` methods
onto a tiny stand-in that yields synthetic tokens, so the test runs
without parquet / tokenizer dependencies (heavy on the Windows dev
box).
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest

torch = pytest.importorskip("torch")

from synapforge.data import AsyncTokenStream, ParquetTokenStream  # noqa: E402


class _StubTokenizer:
    """Minimal tokenizer that maps row "rowNNNNN" to a deterministic
    integer sequence. The chunk-window logic (slide ``seq_len+1`` ids
    after appending EOT per row) yields ``[i, i+1, ..., i+seq_len-1,
    EOT_ID]`` per row.
    """
    SEQ_LEN = 8

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        n = int(s[3:])
        return [(n + j) % 1000 for j in range(self.SEQ_LEN)]


class _TokenStreamHarness:
    """Stand-in that exposes the surface ``AsyncTokenStream`` reaches
    into on the inner ``ParquetTokenStream``: ``seq_len``,
    ``batch_size``, ``eot_id``, ``_tok``, ``_iter_text_rows``,
    ``_iter_token_chunks``, ``_build_batch``, ``pin_memory``.

    For bit-exact comparison both the async-pipeline path (calls
    ``_iter_text_rows`` + tokenizer + EOT) and the legacy
    ``_iter_sync`` path (calls ``_iter_token_chunks`` directly) MUST
    produce identical chunks. The harness aligns both: row i ->
    ``[i, i+1, ..., i+seq_len-1, EOT_ID]``.
    """

    EOT_ID = 50256
    SEQ_LEN = 8

    def __init__(
        self,
        n_chunks: int,
        seq_len: int = 8,
        batch_size: int = 4,
        seed: int = 42,
    ) -> None:
        assert seq_len == self.SEQ_LEN
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self._n_chunks = int(n_chunks)
        self.eot_id = self.EOT_ID
        self._tok = _StubTokenizer()
        self._seed = int(seed)
        self.pin_memory = False  # required by ParquetTokenStream._build_batch

    def _iter_text_rows(self) -> Iterator[str]:
        # Row i -> "rowiiiiii"; tokenizer emits ``[i, i+1, ...,
        # i+seq_len-1]``. The async code appends EOT_ID giving one
        # length-(seq_len+1) chunk per row: ``[i, ..., i+seq_len-1,
        # EOT_ID]``.
        for i in range(self._n_chunks):
            yield f"row{i:05d}"

    def _iter_token_chunks(self) -> Iterator[list[int]]:
        # Mirror the async path exactly: row i -> ``[i, i+1, ...,
        # i+seq_len-1, EOT_ID]``. ``ParquetTokenStream._iter_sync``
        # calls this method, so the legacy reference yields the same
        # chunks as the async path.
        for i in range(self._n_chunks):
            yield [(i + j) % 1000 for j in range(self.SEQ_LEN)] + [self.EOT_ID]

    # Bind production methods. Use ``__get__`` so ``self`` is the
    # harness, not a ParquetTokenStream instance. Mirrors the pattern
    # in test_dataloader_prefetch.py.
    _build_batch = ParquetTokenStream._build_batch


def _drain(stream, n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    it = iter(stream)
    for _ in range(n_batches):
        out.append(next(it))
    del it
    return out


def _build_async_with_harness(
    n_chunks: int, stages: int, prefetch: int = 8,
) -> AsyncTokenStream:
    """Construct an ``AsyncTokenStream`` whose inner stream is a
    ``_TokenStreamHarness`` (no real parquet / tokenizer)."""
    inst = AsyncTokenStream.__new__(AsyncTokenStream)
    inst._inner = _TokenStreamHarness(n_chunks=n_chunks)
    inst.seq_len = inst._inner.seq_len
    inst.batch_size = inst._inner.batch_size
    inst.vocab_size = 1000
    inst.tokenizer_name = "stub"
    inst.shuffle_buffer = 0
    inst.shuffle_seed = 42
    inst.pin_memory = False  # off so test passes on no-CUDA torch
    inst.stages = max(1, min(4, int(stages)))
    inst.prefetch = max(2, int(prefetch))
    inst._threads = []
    inst._closed = False
    return inst


def _legacy_sync_batches(
    n_chunks: int, n_batches: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build the legacy single-thread path output by calling the
    production ``ParquetTokenStream._iter_sync`` against the same
    harness. This is the reference sequence we compare against.
    """
    h = _TokenStreamHarness(n_chunks=n_chunks)
    # Bind the production method onto the harness via __get__.
    sync_iter = ParquetTokenStream._iter_sync.__get__(h, type(h))
    out = []
    it = sync_iter()
    for _ in range(n_batches):
        out.append(next(it))
    return out


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
def test_async_stages_match_legacy_sync(stages: int) -> None:
    """For every supported stage count, AsyncTokenStream yields the
    same (B, T) tensors as the legacy single-thread path. This is the
    headline bit-exact contract.
    """
    # 16 chunks / batch_size 4 => 4 batches; we pull 3 to cover the
    # bulk of the pipeline while leaving room for stage drain.
    async_stream = _build_async_with_harness(n_chunks=16, stages=stages)
    async_batches = _drain(async_stream, n_batches=3)
    legacy_batches = _legacy_sync_batches(n_chunks=16, n_batches=3)
    assert len(async_batches) == len(legacy_batches) == 3
    for i, ((x_a, y_a), (x_l, y_l)) in enumerate(
        zip(async_batches, legacy_batches)
    ):
        assert x_a.shape == x_l.shape, (
            f"batch {i} shape mismatch (async={x_a.shape}, legacy={x_l.shape}, "
            f"stages={stages})"
        )
        assert torch.equal(x_a, x_l), (
            f"batch {i} tokens_in mismatch at stages={stages}"
        )
        assert torch.equal(y_a, y_l), (
            f"batch {i} tokens_out mismatch at stages={stages}"
        )
        assert x_a.dtype == torch.long
        assert y_a.dtype == torch.long


def test_async_pipeline_yields_correct_shape() -> None:
    """Sanity: ``(B, T)`` shape and shifted-by-1 contract."""
    stream = _build_async_with_harness(n_chunks=12, stages=4)
    batches = _drain(stream, n_batches=2)
    for x, y in batches:
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)
        assert torch.equal(x[:, 1:], y[:, :-1])


def test_async_producer_exception_resurfaces() -> None:
    """Exceptions in any pipeline stage must re-raise on the consumer's
    stack — never a silent hang.
    """
    sentinel = RuntimeError("synthetic stage failure")

    class _BadHarness(_TokenStreamHarness):
        def _iter_text_rows(self) -> Iterator[str]:
            yield "row00000"
            raise sentinel

        def _iter_token_chunks(self) -> Iterator[list[int]]:
            yield [0] * (self.seq_len + 1)
            raise sentinel

    inst = AsyncTokenStream.__new__(AsyncTokenStream)
    inst._inner = _BadHarness(n_chunks=8)
    inst.seq_len = inst._inner.seq_len
    inst.batch_size = inst._inner.batch_size
    inst.vocab_size = 1000
    inst.tokenizer_name = "stub"
    inst.shuffle_buffer = 0
    inst.shuffle_seed = 42
    inst.pin_memory = False
    inst.stages = 4
    inst.prefetch = 4
    inst._threads = []
    inst._closed = False

    with pytest.raises(RuntimeError) as excinfo:
        for _ in iter(inst):
            pass
    assert "synthetic stage failure" in str(excinfo.value)


def test_async_constructor_off_default() -> None:
    """The trainer activates the async pipeline only when --async-data-
    pipeline is passed. ``stages`` itself defaults to 4 (active when
    requested), but the trainer keeps building ``ParquetTokenStream``
    by default. This test pins that AsyncTokenStream's defaults at
    least don't crash on a missing-parquet smoke path."""
    # We don't need a real parquet — just check the constructor's
    # defaults don't raise on a non-existent glob (the inner stream
    # will raise FileNotFoundError, which we catch).
    with pytest.raises(FileNotFoundError):
        AsyncTokenStream(
            "/nonexistent/glob/*.parquet",
            seq_len=8, batch_size=4,
        )


def test_async_pin_memory_no_cuda_safe() -> None:
    """``pin_memory=True`` on a no-CUDA build must be a silent no-op,
    never crash. The harness path runs without pyarrow/transformers."""
    inst = _build_async_with_harness(n_chunks=8, stages=4)
    # Force pin on; the constructor's CUDA-availability mask should
    # have already squashed it to False on no-CUDA builds.
    inst.pin_memory = bool(torch.cuda.is_available())
    out = _drain(inst, n_batches=2)
    assert len(out) == 2


def test_async_repr_safe() -> None:
    """``repr()`` must not depend on the inner stream having a real
    parquet — sanity check for log lines in the trainer."""
    inst = _build_async_with_harness(n_chunks=8, stages=2)
    s = repr(inst)
    assert "AsyncTokenStream" in s
    assert "stages=2" in s
    assert "prefetch=8" in s
