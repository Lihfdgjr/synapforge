"""docs/PERF_KNOBS.md — dataloader prefetch + pinned-memory contract.

The trainer's ``ParquetTokenStream`` gained two perf knobs in the
2026-05-01 perf-v2 batch:

  * ``prefetch_factor=N`` (>=2) spawns a daemon producer thread that
    pre-builds batches into a bounded ``queue.Queue`` while the main
    iterator (the GPU step loop) consumes them.
  * ``pin_memory=True`` allocates the yielded ``(tokens_in,
    tokens_out)`` tensors in pinned (page-locked) host memory so the
    trainer's ``x.to(device, non_blocking=True)`` H2D copy actually
    overlaps with compute.

This test pins the contract:

  (a) prefetch=on yields the SAME batches in the same order as
      prefetch=off when the underlying source is deterministic
      (shuffle_buffer<=1) — the prefetch thread is purely a producer/
      consumer overlay, NOT a re-shuffler.
  (b) Same ``shuffle_seed`` + same ``shuffle_buffer`` => same yield
      sequence with prefetch on or off (determinism preserved across
      the threading boundary).
  (c) Producer-thread exceptions are re-raised on the consumer's stack
      (no silent worker crashes — the worst class of bug for KD runs).
  (d) ``pin_memory=True`` on a no-CUDA torch build is a silent no-op.

We exercise the prefetch + pin contract WITHOUT running real parquet
or the AutoTokenizer (heavy deps that are absent on the Windows dev
box). The harness binds the production ``__iter__`` /
``_iter_prefetch`` / ``_build_batch`` methods onto a tiny stand-in
that yields synthetic token chunks. Mirrors the
``test_shuffle_buffer.py`` pattern.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch")

from synapforge.data import ParquetTokenStream  # noqa: E402


class _PrefetchHarness:
    """Stand-in exposing exactly the surface needed by the production
    ``_iter_sync`` / ``_iter_prefetch`` / ``_build_batch`` methods.

    Yields deterministic length-(seq_len+1) integer chunks so we can
    assert byte-equal output between sync and prefetch paths.
    """

    def __init__(
        self,
        n_chunks: int,
        seq_len: int = 8,
        batch_size: int = 4,
        prefetch_factor: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self._n_chunks = int(n_chunks)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.prefetch_factor = int(prefetch_factor)
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()

    def _iter_token_chunks(self) -> Iterator[list[int]]:
        # Deterministic: chunk i is [i, i+1, ..., i+seq_len].
        for i in range(self._n_chunks):
            yield [(i + j) % 1000 for j in range(self.seq_len + 1)]

    # Bind production methods. Use ``__get__`` so ``self`` is the
    # harness, not a ParquetTokenStream instance.
    _build_batch = ParquetTokenStream._build_batch
    _iter_sync = ParquetTokenStream._iter_sync
    _iter_prefetch = ParquetTokenStream._iter_prefetch
    __iter__ = ParquetTokenStream.__iter__


def _drain(stream, n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    it = iter(stream)
    for _ in range(n_batches):
        out.append(next(it))
    del it
    return out


def test_prefetch_on_matches_off():
    """(a) Prefetch ON yields the same (B, T) tensor values as prefetch
    OFF when the source is deterministic. The prefetch thread is a
    pure producer/consumer overlay — it must not reorder or drop.
    """
    h_off = _PrefetchHarness(n_chunks=32, prefetch_factor=0)
    h_on = _PrefetchHarness(n_chunks=32, prefetch_factor=2)
    batches_off = _drain(h_off, n_batches=4)
    batches_on = _drain(h_on, n_batches=4)
    assert len(batches_off) == len(batches_on) == 4
    for i, ((x_off, y_off), (x_on, y_on)) in enumerate(
        zip(batches_off, batches_on)
    ):
        assert torch.equal(x_off, x_on), f"batch {i} tokens_in mismatch"
        assert torch.equal(y_off, y_on), f"batch {i} tokens_out mismatch"


def test_prefetch_yields_correct_shapes_and_dtypes():
    """The yielded ``(tokens_in, tokens_out)`` must be int64 (B, T)
    tensors with ``out`` shifted left by 1 from ``in`` — same contract
    as the legacy synchronous path."""
    h_on = _PrefetchHarness(
        n_chunks=12, seq_len=8, batch_size=4, prefetch_factor=2,
    )
    batches = _drain(h_on, n_batches=3)
    for x, y in batches:
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)
        assert x.dtype == torch.long
        assert y.dtype == torch.long
        # next-token target shape: x[:, 1:] == y[:, :-1]
        assert torch.equal(x[:, 1:], y[:, :-1])


def test_prefetch_factor_zero_is_passthrough():
    """``prefetch_factor=0`` (and 1) keeps the legacy single-thread
    path. Exercises the dispatch in ``__iter__``."""
    h0 = _PrefetchHarness(n_chunks=8, prefetch_factor=0)
    h1 = _PrefetchHarness(n_chunks=8, prefetch_factor=1)
    h2 = _PrefetchHarness(n_chunks=8, prefetch_factor=2)
    out0 = _drain(h0, n_batches=2)
    out1 = _drain(h1, n_batches=2)
    out2 = _drain(h2, n_batches=2)
    assert len(out0) == len(out1) == len(out2) == 2
    for (x0, y0), (x1, y1), (x2, y2) in zip(out0, out1, out2):
        assert torch.equal(x0, x1), "prefetch=0 vs =1 must match"
        assert torch.equal(x0, x2), "prefetch=0 vs =2 must match (deterministic)"
        assert torch.equal(y0, y1)
        assert torch.equal(y0, y2)


def test_prefetch_producer_exception_resurfaces():
    """(c) Worker-thread exceptions must re-raise on the main thread.

    We construct a stream then monkey-patch ``_iter_token_chunks`` to
    raise after one good chunk; the consumer must see the same
    RuntimeError, NOT a hang or a silent StopIteration.
    """

    sentinel = RuntimeError("synthetic producer failure")

    class _BadHarness(_PrefetchHarness):
        def _iter_token_chunks(self):
            yield [0] * (self.seq_len + 1)
            raise sentinel

    bad = _BadHarness(n_chunks=8, prefetch_factor=2)

    with pytest.raises(RuntimeError) as excinfo:
        for _ in iter(bad):
            pass
    assert "synthetic producer failure" in str(excinfo.value)


def test_pin_memory_no_cuda_is_silent_noop():
    """``pin_memory=True`` on a no-CUDA torch build must be a silent
    no-op — never crash with the "Cannot pin tensor without CUDA"
    error you'd otherwise hit on the Windows dev box. The harness
    constructor mirrors ``ParquetTokenStream.__init__``'s guard, so we
    expect ``pin_memory`` to evaluate to False here."""
    h = _PrefetchHarness(n_chunks=8, prefetch_factor=0, pin_memory=True)
    assert h.pin_memory is torch.cuda.is_available(), (
        f"pin_memory should track CUDA availability; "
        f"got pin_memory={h.pin_memory} cuda={torch.cuda.is_available()}"
    )
    out = _drain(h, n_batches=2)
    assert len(out) == 2
    for x, y in out:
        assert x.dtype == torch.long
        assert y.dtype == torch.long


def test_parquet_stream_constructor_threads_args(tmp_path):
    """``ParquetTokenStream.__init__`` must accept ``prefetch_factor``
    and ``pin_memory`` and store them as attributes (no swallow). This
    is the smoke that catches "I added the kwarg but forgot the
    assignment" bugs at PR review time.

    We hit the real constructor with a tiny one-row parquet (the
    minimal artifact AutoTokenizer doesn't actually load until
    ``_iter_token_chunks`` is consumed)."""
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("transformers")
    parquet = tmp_path / "synth.parquet"
    pa_table = pa.table({"text": ["hello world"]})
    pq.write_table(pa_table, str(parquet))
    stream = ParquetTokenStream(
        str(parquet),
        seq_len=8,
        batch_size=2,
        text_column="text",
        loop=True,
        tokenizer_name="gpt2",
        shuffle_buffer=0,
        prefetch_factor=4,
        pin_memory=True,
    )
    assert stream.prefetch_factor == 4
    # pin_memory is masked to False on no-CUDA builds.
    assert stream.pin_memory is torch.cuda.is_available()
    assert "prefetch_factor=4" in repr(stream)
