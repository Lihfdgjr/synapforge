"""P3 (MASTER_PLAN.md §6) -- secondary val_holdout for honest TTT-leak-free
reporting.

Mocks a 100-row val parquet with deterministically numbered rows, builds the
TTT/holdout split via ``synapforge.data.split_val_stream``, and asserts:

* the two streams' chunk sets are DISJOINT (no leak),
* their union equals the full parent stream,
* ttt_fraction=0.8 puts ~80% of chunks on the TTT side,
* the bad ``ttt_fraction`` values are rejected.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from synapforge.data import (  # noqa: E402
    ParquetTokenStream,
    _RowSubsetStream,
    split_val_stream,
)


class _CountingChunkStream:
    """Tiny stand-in for ``ParquetTokenStream`` that yields N numbered chunks.

    ``_RowSubsetStream`` only requires the parent expose ``_iter_token_chunks``
    plus a ``batch_size``; this avoids the parquet/tokenizer round-trip in
    the unit test (the real ``ParquetTokenStream`` is exercised by the
    end-to-end smoke).
    """

    def __init__(self, n_chunks: int, seq_len: int = 4, batch_size: int = 1):
        self.n_chunks = int(n_chunks)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)

    def _iter_token_chunks(self):
        # Each chunk is uniquely identified by its first token == row index;
        # the rest is filler. Length seq_len+1 mirrors ParquetTokenStream.
        for i in range(self.n_chunks):
            yield [i] + [0] * self.seq_len


def _collected_row_ids(stream: _RowSubsetStream) -> list[int]:
    """Drain a side and return the list of first-tokens (== parent row idx)."""
    rows: list[int] = []
    for tokens_in, _tokens_out in stream:
        # tokens_in shape (B, seq_len). First column is the chunk's first
        # token, which is its parent-stream row index.
        rows.extend(int(v) for v in tokens_in[:, 0].tolist())
    return rows


def test_split_disjoint_and_complete():
    parent = _CountingChunkStream(n_chunks=100, batch_size=1)
    ttt, hold = split_val_stream(parent, ttt_fraction=0.8, denom=5)
    assert isinstance(ttt, _RowSubsetStream)
    assert isinstance(hold, _RowSubsetStream)
    ttt_rows = set(_collected_row_ids(ttt))
    hold_rows = set(_collected_row_ids(hold))
    assert ttt_rows.isdisjoint(hold_rows), (
        f"TTT ∩ holdout must be empty; overlap={ttt_rows & hold_rows!r}"
    )
    assert ttt_rows | hold_rows == set(range(100)), (
        "TTT ∪ holdout must equal the full parent set"
    )


def test_split_ratio_is_approximately_eighty_twenty():
    parent = _CountingChunkStream(n_chunks=100, batch_size=1)
    ttt, hold = split_val_stream(parent, ttt_fraction=0.8, denom=5)
    n_ttt = len(_collected_row_ids(ttt))
    n_hold = len(_collected_row_ids(hold))
    # denom=5 with ttt_fraction=0.8 is an exact 4:1 split for n_chunks
    # divisible by 5 (100 here), so we can require equality.
    assert n_ttt == 80, f"expected ~80 TTT rows, got {n_ttt}"
    assert n_hold == 20, f"expected ~20 holdout rows, got {n_hold}"
    assert n_ttt + n_hold == 100


def test_split_keep_indices_metadata():
    parent = _CountingChunkStream(n_chunks=10, batch_size=1)
    ttt, hold = split_val_stream(parent, ttt_fraction=0.8, denom=5)
    assert ttt.keep_indices == {0, 1, 2, 3}
    assert hold.keep_indices == {4}
    assert ttt.denom == hold.denom == 5
    assert ttt.side == "ttt"
    assert hold.side == "holdout"


@pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5])
def test_split_rejects_bad_fractions(bad):
    parent = _CountingChunkStream(n_chunks=10)
    with pytest.raises(ValueError):
        split_val_stream(parent, ttt_fraction=bad, denom=5)


def test_split_with_real_parquet_stream(tmp_path):
    """End-to-end with a real (tiny) parquet to exercise the actual
    ``ParquetTokenStream._iter_token_chunks`` path the trainer uses."""
    pytest.importorskip("transformers")
    # Build 200 short rows of repeated text; tokenizer yields well-defined
    # chunks deterministically. Use the gpt2 tokenizer (default) so no
    # network/HF mirror dependency.
    rows = [f"row {i:04d} alpha beta gamma delta" for i in range(200)]
    table = pa.table({"text": rows})
    parquet_path = tmp_path / "val.parquet"
    pq.write_table(table, parquet_path)
    try:
        ds = ParquetTokenStream(
            str(tmp_path / "val.parquet"),
            seq_len=8, batch_size=2, loop=False, tokenizer_name="gpt2",
        )
    except Exception as exc:
        pytest.skip(f"gpt2 tokenizer unavailable in this env: {exc}")
    ttt, hold = split_val_stream(ds, ttt_fraction=0.8, denom=5)
    n_ttt = sum(x.shape[0] for x, _ in ttt)
    n_hold = sum(x.shape[0] for x, _ in hold)
    # The ratio is exactly 4:1 over the full pre-batch chunk stream.
    # Because the parent yields chunks in lockstep and we batch with
    # batch_size=2, the totals can fluctuate by < batch_size due to a
    # trailing partial batch being dropped on each side.
    assert n_ttt > 0 and n_hold > 0, (n_ttt, n_hold)
    if n_ttt + n_hold >= 5:
        assert n_ttt > n_hold, "ttt_fraction=0.8 must produce more TTT rows"
