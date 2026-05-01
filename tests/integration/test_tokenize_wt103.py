"""Integration tests for ``scripts/tokenize_wikitext103.py`` (T3.7).

Covers the four contracts the queue spec for T3.7 calls out:

1. ``test_smoke_writes_pkl``           -- ``--smoke`` writes a pickle with a
   list of int token-id lists, all entries int-castable.
2. ``test_smoke_writes_parquet``       -- ``--smoke`` writes a parquet with
   exactly one column ``input_ids`` (``list<int32>``-shaped) plus a
   companion ``.manifest.json``.
3. ``test_missing_files_clear_error``  -- with a glob that matches no files
   the script exits ``1`` and writes a discoverable diagnostic to stderr.
4. ``test_token_count_in_smoke``       -- ``--smoke`` produces exactly 100
   sequences (the documented mock output size).

All tests run on CPU and do NOT require ``transformers``, ``torch``, or
network access. They DO require ``pyarrow`` (already a hard dep of the
trainer's ``ParquetTokenStream``).
"""
from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import pytest


# Make the repo's ``scripts/`` dir importable identically to how
# tests/integration/conftest.py does it -- this also works when this test
# is collected stand-alone (e.g. via ``pytest tests/integration/test_tokenize_wt103.py``).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def wt_module():
    pytest.importorskip("pyarrow")
    if "tokenize_wikitext103" in sys.modules:
        return importlib.reload(sys.modules["tokenize_wikitext103"])
    return importlib.import_module("tokenize_wikitext103")


# ---------------------------------------------------------------------------
# 1. --smoke writes a pickle file with int token IDs.
# ---------------------------------------------------------------------------
def test_smoke_writes_pkl(tmp_path, wt_module):
    """``--smoke`` round-trips through pickle to a list-of-int-lists."""
    out_pkl = tmp_path / "wt103_smoke.pkl"
    out_parquet = tmp_path / "wt103_smoke.parquet"
    rc = wt_module.main([
        "--smoke",
        "--output-pkl", str(out_pkl),
        "--output-parquet", str(out_parquet),
    ])
    assert rc == 0
    assert out_pkl.exists(), "smoke run did not write the pkl"

    with open(out_pkl, "rb") as f:
        data = pickle.load(f)

    # Contract: list of list[int].
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(seq, list) for seq in data)
    # Every entry must be int-castable (no numpy oddities leaked).
    for seq in data:
        for tok in seq:
            assert isinstance(tok, int)
            assert tok >= 0


# ---------------------------------------------------------------------------
# 2. --smoke writes a parquet with the correct schema.
# ---------------------------------------------------------------------------
def test_smoke_writes_parquet(tmp_path, wt_module):
    """``--smoke`` writes a parquet with a single ``input_ids`` column."""
    import pyarrow.parquet as pq

    out_pkl = tmp_path / "wt103_smoke.pkl"
    out_parquet = tmp_path / "wt103_smoke.parquet"
    rc = wt_module.main([
        "--smoke",
        "--output-pkl", str(out_pkl),
        "--output-parquet", str(out_parquet),
    ])
    assert rc == 0
    assert out_parquet.exists(), "smoke run did not write the parquet"

    # Schema contract: one column named "input_ids".
    table = pq.read_table(str(out_parquet))
    assert table.column_names == ["input_ids"], (
        f"expected single 'input_ids' column; got {table.column_names}"
    )
    assert table.num_rows > 0

    # First row is a list of ints (parquet list<int>).
    row0 = table.column("input_ids")[0].as_py()
    assert isinstance(row0, list)
    assert all(isinstance(x, int) for x in row0)

    # Companion manifest must exist + record the right kind/tokenizer.
    manifest_path = Path(str(out_parquet) + ".manifest.json")
    assert manifest_path.exists(), "missing companion manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "wikitext103_qwen_tokenized"
    assert manifest["tokenizer"] == "Qwen/Qwen2.5-0.5B"
    assert manifest["rows"] == table.num_rows


# ---------------------------------------------------------------------------
# 3. Missing files: clear error + non-zero exit.
# ---------------------------------------------------------------------------
def test_missing_files_clear_error(tmp_path, wt_module, capsys):
    """A glob that matches nothing yields a clear stderr message + exit 1."""
    out_pkl = tmp_path / "wt103_no_input.pkl"
    out_parquet = tmp_path / "wt103_no_input.parquet"
    bogus_glob = str(tmp_path / "definitely_does_not_exist" / "*.txt")

    rc = wt_module.main([
        "--input-glob", bogus_glob,
        "--output-pkl", str(out_pkl),
        "--output-parquet", str(out_parquet),
    ])
    assert rc == 1, f"expected exit 1 on missing files; got {rc}"

    captured = capsys.readouterr()
    # Diagnostic must be loud enough to grep.
    assert "FATAL" in captured.err
    assert "no wikitext-103 source files found" in captured.err
    # The glob is echoed via repr() inside a list; on Windows that
    # double-escapes backslashes. Match on the leaf-name fragment instead
    # which is identical across platforms.
    assert "definitely_does_not_exist" in captured.err
    assert "*.txt" in captured.err

    # No output files should have been created.
    assert not out_pkl.exists()
    assert not out_parquet.exists()


# ---------------------------------------------------------------------------
# 4. Smoke mode emits exactly 100 sequences (documented contract).
# ---------------------------------------------------------------------------
def test_token_count_in_smoke(tmp_path, wt_module):
    """The mock output size is 100 sequences -- not more, not fewer."""
    out_pkl = tmp_path / "wt103_count.pkl"
    out_parquet = tmp_path / "wt103_count.parquet"
    rc = wt_module.main([
        "--smoke",
        "--output-pkl", str(out_pkl),
        "--output-parquet", str(out_parquet),
    ])
    assert rc == 0

    with open(out_pkl, "rb") as f:
        data = pickle.load(f)
    assert len(data) == 100, (
        f"smoke mode contract: 100 sequences; got {len(data)}"
    )

    # Parquet must agree.
    import pyarrow.parquet as pq
    table = pq.read_table(str(out_parquet))
    assert table.num_rows == 100, (
        f"parquet rows must equal pickle rows ({len(data)}); "
        f"got {table.num_rows}"
    )

    # Each sequence is non-empty + within max_length=512 (default).
    for seq in data:
        assert 1 <= len(seq) <= 512


# ---------------------------------------------------------------------------
# Bonus: helper-level coverage so the tests don't depend solely on main().
# ---------------------------------------------------------------------------
def test_find_input_files_priority(tmp_path, wt_module):
    """Default-glob mode short-circuits on the first non-empty match."""
    # Build two layouts; the first should win.
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "train.txt").write_text("hello", encoding="utf-8")
    (b / "train.txt").write_text("world", encoding="utf-8")

    found = wt_module.find_input_files([
        str(a / "*.txt"),
        str(b / "*.txt"),
    ])
    # First non-empty pattern wins (priority order).
    assert len(found) == 1
    assert found[0].endswith("train.txt")
    assert "a" in found[0]


def test_chunk_into_sequences_drops_short_tail(wt_module):
    """The tail < 8 tokens must be dropped (avoid noisy gradient)."""
    tokens = list(range(100))
    chunks = wt_module._chunk_into_sequences(tokens, max_length=30)
    # 30 + 30 + 30 + 10 = 100; last chunk (10) >= 8 stays.
    assert len(chunks) == 4
    assert len(chunks[-1]) == 10

    # If the tail is < 8, it gets dropped.
    chunks2 = wt_module._chunk_into_sequences(tokens, max_length=33)
    # 33 + 33 + 33 + 1 -> tail of 1 dropped; 3 chunks total.
    assert len(chunks2) == 3
    assert sum(len(c) for c in chunks2) == 99
