"""Integration tests for ``scripts/curriculum_sort.py`` (T8.3).

Curriculum learning sorts training rows by reference-model perplexity
ascending so the student trainer sees easy samples first. Tests below
cover the four contract guarantees the trainer relies on:

1. ``test_smoke``                 -- ``main(['--smoke'])`` writes 5 rows
                                     with ppl + curriculum_idx populated.
2. ``test_monotonic_after_sort``  -- ``ref_ppl`` column is monotonically
                                     non-decreasing after the sort.
3. ``test_curriculum_idx_zero_to_n`` -- the new index column is exactly
                                     ``[0, 1, ..., N-1]``.
4. ``test_preserves_input_columns`` -- every original column survives
                                     the sort (rows reordered, values
                                     unchanged).

Plus three round-trip helpers covering the pure-Python sort core; all
tests run on CPU without torch / transformers (mock ppl_fn).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def cur_module():
    pytest.importorskip("pyarrow")
    pytest.importorskip("numpy")
    if "curriculum_sort" in sys.modules:
        return importlib.reload(sys.modules["curriculum_sort"])
    return importlib.import_module("curriculum_sort")


# ----- contract test 1: smoke run ------------------------------------------
def test_smoke(tmp_path, cur_module):
    """``main(['--smoke', '--output', X])`` writes 5 rows + manifest."""
    out = tmp_path / "smoke.parquet"
    rc = cur_module.main(["--smoke", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    assert (Path(str(out) + ".manifest.json")).exists()

    import pyarrow.parquet as pq
    table = pq.read_table(str(out))
    assert table.num_rows == 5
    cols = set(table.column_names)
    assert {"input_ids", "ref_ppl", "curriculum_idx"}.issubset(cols)


# ----- contract test 2: monotonic after sort -------------------------------
def test_monotonic_after_sort(tmp_path, cur_module):
    """The ``ref_ppl`` column must be monotonically non-decreasing."""
    out = tmp_path / "mono.parquet"
    cur_module.main(["--smoke", "--output", str(out)])

    import pyarrow.parquet as pq
    table = pq.read_table(str(out))
    ppls = table.column("ref_ppl").to_pylist()
    assert len(ppls) == 5
    for a, b in zip(ppls, ppls[1:]):
        assert a <= b, f"ppl not monotonic: {a} > {b} (sequence={ppls})"


# ----- contract test 3: curriculum_idx is 0..N-1 ---------------------------
def test_curriculum_idx_zero_to_n(tmp_path, cur_module):
    """``curriculum_idx`` is exactly ``[0, 1, ..., N-1]`` after sort."""
    out = tmp_path / "idx.parquet"
    cur_module.main(["--smoke", "--output", str(out)])

    import pyarrow.parquet as pq
    table = pq.read_table(str(out))
    idxs = table.column("curriculum_idx").to_pylist()
    assert idxs == list(range(table.num_rows))


# ----- contract test 4: original columns preserved -------------------------
def test_preserves_input_columns(tmp_path, cur_module):
    """All input columns survive (rows reordered, *values* identical set)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build an input parquet with two extra columns (input_ids + meta1 + meta2)
    # to verify the sort carries them along with the input_ids re-order.
    in_path = tmp_path / "in.parquet"
    rows = [
        [10, 20, 30],                    # len 3
        [11],                            # len 1 (hardest)
        [12, 22, 32, 42, 52, 62, 72],    # len 7 (easiest)
        [13, 23],                        # len 2
        [14, 24, 34, 44, 54],            # len 5
    ]
    meta1 = ["a", "b", "c", "d", "e"]
    meta2 = [101, 102, 103, 104, 105]
    pq.write_table(
        pa.table({"input_ids": rows, "meta1": meta1, "meta2": meta2}),
        str(in_path),
    )

    # Use the pure-python sorter directly with the smoke ppl_fn so we don't
    # need torch.
    table = pq.read_table(str(in_path))
    sorted_table = cur_module.sort_table_by_ppl(table, cur_module._smoke_ppl_fn)

    # Original columns must survive (and exactly match the source set).
    assert "input_ids" in sorted_table.column_names
    assert "meta1" in sorted_table.column_names
    assert "meta2" in sorted_table.column_names
    assert "ref_ppl" in sorted_table.column_names
    assert "curriculum_idx" in sorted_table.column_names

    # The set of meta1 values must be unchanged (just reordered).
    assert set(sorted_table.column("meta1").to_pylist()) == set(meta1)
    assert set(sorted_table.column("meta2").to_pylist()) == set(meta2)

    # And the per-row binding survives: row that originally had meta1='c' is
    # the longest one (len 7), so it must be first after sort (lowest ppl).
    sorted_meta1 = sorted_table.column("meta1").to_pylist()
    sorted_input = sorted_table.column("input_ids").to_pylist()
    assert sorted_meta1[0] == "c", (
        f"len-7 row should be easiest -> first; got meta1={sorted_meta1}"
    )
    assert sorted_input[0] == [12, 22, 32, 42, 52, 62, 72]
    # And the shortest (len 1, meta1='b') must be last (hardest).
    assert sorted_meta1[-1] == "b", (
        f"len-1 row should be hardest -> last; got meta1={sorted_meta1}"
    )


# ----- helper unit tests (pure-python core, no torch) ----------------------
def test_compute_curriculum_order_stable(cur_module):
    """Stable sort: equal-ppl rows keep their input order."""
    # All rows return ppl=42; order must be [0,1,2,3,4].
    rows = [[i] for i in range(5)]
    order = cur_module.compute_curriculum_order(rows, lambda r: 42.0)
    assert order == [0, 1, 2, 3, 4]


def test_compute_curriculum_order_ascending(cur_module):
    """Distinct ppls sort ascending: row with ppl 1.0 is first."""
    rows = [[0], [1], [2], [3]]
    order = cur_module.compute_curriculum_order(
        rows, lambda r: float(10 - r[0])  # row 3 -> ppl 7, row 0 -> ppl 10
    )
    # ppls are [10, 9, 8, 7]; ascending order is [3, 2, 1, 0].
    assert order == [3, 2, 1, 0]


def test_smoke_ppl_fn_inverse_length(cur_module):
    """Documented contract: ``_smoke_ppl_fn`` is monotone-decreasing in len."""
    a = cur_module._smoke_ppl_fn([1])
    b = cur_module._smoke_ppl_fn([1, 2])
    c = cur_module._smoke_ppl_fn([1, 2, 3, 4, 5])
    assert a > b > c


def test_sort_table_requires_input_ids_column(cur_module, tmp_path):
    """Missing 'input_ids' column raises a clear KeyError, not IndexError."""
    import pyarrow as pa
    bad = pa.table({"text": ["a", "b"]})
    with pytest.raises(KeyError, match="input_ids"):
        cur_module.sort_table_by_ppl(bad, cur_module._smoke_ppl_fn)


def test_main_arg_parse(cur_module):
    """``--smoke`` flag flips bool; default ref-model is None."""
    ns = cur_module._parse_args(["--smoke"])
    assert ns.smoke is True
    assert ns.ref_model is None
    assert ns.input_ids_col == "input_ids"
