"""Integration tests for ``scripts/tokenize_arc.py`` (T3.9).

Covers the four contracts the queue spec for T3.9 calls out:

1. ``test_smoke_writes_parquet``        -- ``--smoke`` writes a parquet
   with the expected ARC eval columns + companion manifest.
2. ``test_format_prompt_with_4_choices`` -- prompt formatter renders
   ``Question:``/``A.``/``B.``/``C.``/``D.``/``Answer:`` exactly.
3. ``test_extract_answer_letter``       -- ``B`` -> 1, ``A`` -> 0,
   numeric labels also resolve correctly.
4. ``test_smoke_5_examples_count``      -- ``--smoke`` produces exactly
   5 rows (the documented mock output size).

All tests run on CPU and do NOT require ``transformers`` or network
access. They DO require ``pyarrow`` (already a hard dep of the trainer's
``ParquetTokenStream``).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


# Make the repo's ``scripts/`` dir importable identically to how
# tests/integration/conftest.py does it.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def arc_module():
    pytest.importorskip("pyarrow")
    if "tokenize_arc" in sys.modules:
        return importlib.reload(sys.modules["tokenize_arc"])
    return importlib.import_module("tokenize_arc")


# ---------------------------------------------------------------------------
# 1. --smoke writes a parquet with the correct schema.
# ---------------------------------------------------------------------------
def test_smoke_writes_parquet(tmp_path, arc_module):
    """``--smoke`` writes the multi-choice eval parquet + manifest."""
    import pyarrow.parquet as pq

    out = tmp_path / "arc_smoke.parquet"
    rc = arc_module.main(["--smoke", "--out", str(out)])
    assert rc == 0
    assert out.exists(), "smoke run did not write parquet"

    table = pq.read_table(str(out))
    expected_cols = {
        "task", "split", "id", "question", "choices", "answerKey",
        "answer_idx", "prompt_input_ids", "prompt_plus_answer_ids",
    }
    assert expected_cols.issubset(set(table.column_names)), (
        f"missing columns; got {table.column_names}"
    )
    assert table.num_rows > 0

    # Each row's `choices` is a list of 4 strings.
    row0 = table.to_pylist()[0]
    assert isinstance(row0["choices"], list)
    assert len(row0["choices"]) == 4
    # `prompt_plus_answer_ids` shape: list of 4 lists.
    plus = row0["prompt_plus_answer_ids"]
    assert isinstance(plus, list)
    assert len(plus) == 4
    # Both Easy and Challenge rows must be present.
    tasks = set(table.column("task").to_pylist())
    assert tasks == {"Easy", "Challenge"}

    # Manifest contract.
    manifest_path = Path(str(out) + ".manifest.json")
    assert manifest_path.exists(), "missing companion manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "arc_tokenized"
    assert manifest["rows"] == table.num_rows
    assert sorted(manifest["tasks"]) == ["Challenge", "Easy"]


# ---------------------------------------------------------------------------
# 2. Prompt formatter renders the canonical multiple-choice layout.
# ---------------------------------------------------------------------------
def test_format_prompt_with_4_choices(arc_module):
    """``format_prompt`` lays out Question + A.. D + Answer: exactly."""
    q = "What color is the sky on a clear day?"
    choices = ["Red", "Blue", "Green", "Yellow"]
    out = arc_module.format_prompt(q, choices)

    expected = (
        "Question: What color is the sky on a clear day?\n"
        "A. Red\n"
        "B. Blue\n"
        "C. Green\n"
        "D. Yellow\n"
        "Answer:"
    )
    assert out == expected, f"prompt mismatch:\n{out!r}"

    # Pads to 4 choices when fewer are supplied.
    short = arc_module.format_prompt("Q?", ["only", "two"])
    lines = short.split("\n")
    assert lines[0] == "Question: Q?"
    assert lines[1] == "A. only"
    assert lines[2] == "B. two"
    assert lines[3] == "C. "
    assert lines[4] == "D. "
    assert lines[5] == "Answer:"


# ---------------------------------------------------------------------------
# 3. Letter -> index resolution (B -> 1, etc.)
# ---------------------------------------------------------------------------
def test_extract_answer_letter(arc_module):
    """``B`` -> 1, ``A`` -> 0, with numeric labels also working."""
    labels_abc = ["A", "B", "C", "D"]
    assert arc_module.extract_answer_idx("A", labels_abc) == 0
    assert arc_module.extract_answer_idx("B", labels_abc) == 1
    assert arc_module.extract_answer_idx("C", labels_abc) == 2
    assert arc_module.extract_answer_idx("D", labels_abc) == 3
    # Whitespace tolerance.
    assert arc_module.extract_answer_idx(" B ", labels_abc) == 1

    # Numeric labels (some ARC rows are 1..4 not A..D).
    labels_num = ["1", "2", "3", "4"]
    assert arc_module.extract_answer_idx("3", labels_num) == 2
    assert arc_module.extract_answer_idx("1", labels_num) == 0

    # Malformed: empty key, unknown key.
    assert arc_module.extract_answer_idx("", labels_abc) is None
    assert arc_module.extract_answer_idx("Z", labels_abc) is None


# ---------------------------------------------------------------------------
# 4. Smoke mode emits exactly 5 rows (documented contract).
# ---------------------------------------------------------------------------
def test_smoke_5_examples_count(tmp_path, arc_module):
    """The mock output size is 5 examples — not more, not fewer."""
    import pyarrow.parquet as pq

    out = tmp_path / "arc_count.parquet"
    rc = arc_module.main(["--smoke", "--out", str(out)])
    assert rc == 0

    table = pq.read_table(str(out))
    assert table.num_rows == 5, (
        f"smoke mode contract: 5 rows; got {table.num_rows}"
    )

    # Sanity: each row's answer_idx is within [0, 3] and matches the
    # smoke-record ground truth ordering (B->1, C->2, B->1, "3" w/ numeric
    # labels -> 2).
    rows = table.to_pylist()
    expected_idx = [1, 2, 2, 1, 2]
    actual_idx = [int(r["answer_idx"]) for r in rows]
    assert actual_idx == expected_idx, (
        f"answer_idx ordering mismatch: got {actual_idx}, want {expected_idx}"
    )

    # Each row's `choices` always pads to len 4.
    for r in rows:
        assert len(r["choices"]) == 4, (
            f"choices not padded to 4 on row {r['id']}: {r['choices']!r}"
        )
