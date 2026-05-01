"""Integration tests for ``scripts/tokenize_swebench_mini.py`` (T3.10).

Covers the four contracts the queue spec for T3.10 calls out:

1. ``test_smoke_writes_parquet``        -- ``--smoke`` writes a parquet
   with the documented columns (instance_id, repo, prompt, patch_target,
   prompt_input_ids, target_input_ids) plus a companion ``.manifest.json``.
2. ``test_format_prompt_includes_repo`` -- the prompt template includes
   the repo string and the problem statement (smoke-test for the format
   contract used by the code-fix demo).
3. ``test_truncation_long_patch``       -- a very long patch fed through
   the encoder is hard-truncated to ``max_length`` (no overflow).
4. ``test_n_50_default``                -- the ``--n`` argparse default
   is 50, matching the queue spec.

All tests run on CPU and do NOT require ``transformers``, ``torch``, or
network access. They DO require ``pyarrow`` (already a hard dep of the
trainer's ``ParquetTokenStream``).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import List

import pytest


# Make the repo's ``scripts/`` dir importable identically to how
# tests/integration/conftest.py does it -- this also works when this test
# is collected stand-alone.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def swe_module():
    pytest.importorskip("pyarrow")
    if "tokenize_swebench_mini" in sys.modules:
        return importlib.reload(sys.modules["tokenize_swebench_mini"])
    return importlib.import_module("tokenize_swebench_mini")


class _MockTokenizer:
    """Deterministic char-id tokenizer for tests; honours ``max_length``.

    Returns one int per char in the text, capped to ``max_length`` when the
    encoder is invoked with that kwarg. We deliberately match what the real
    HF tokenizer surface looks like (encode + add_special_tokens kwarg).
    """

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids: List[int] = [(ord(c) % 50000) + 1 for c in (text or "")]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids


# ---------------------------------------------------------------------------
# 1. --smoke writes a parquet with all 6 documented columns + manifest.
# ---------------------------------------------------------------------------
def test_smoke_writes_parquet(tmp_path, swe_module):
    """``--smoke`` writes a parquet with the documented schema."""
    import pyarrow.parquet as pq

    out = tmp_path / "swebench_smoke.parquet"
    rc = swe_module.main(["--smoke", "--out", str(out)])
    assert rc == 0, f"smoke run exited {rc}"
    assert out.exists(), "smoke run did not write the parquet"

    # Schema contract: 6 documented columns.
    table = pq.read_table(str(out))
    assert set(table.column_names) == {
        "instance_id",
        "repo",
        "prompt",
        "patch_target",
        "prompt_input_ids",
        "target_input_ids",
    }, f"unexpected columns: {table.column_names}"
    # Smoke ships exactly 5 mocked records (subject to --n cap).
    assert table.num_rows == 5, (
        f"smoke contract is 5 rows; got {table.num_rows}"
    )

    # First row sanity: instance_id non-empty, prompt has 'Repo:', patch
    # target starts with the unified-diff '---' marker.
    row0_iid = table.column("instance_id")[0].as_py()
    row0_prompt = table.column("prompt")[0].as_py()
    row0_patch = table.column("patch_target")[0].as_py()
    assert row0_iid.startswith("smoke__demo-")
    assert "Repo:" in row0_prompt
    assert row0_patch.startswith("---")

    # Token id columns are list<int> (smoke uses no real tokenizer -> []).
    row0_prompt_ids = table.column("prompt_input_ids")[0].as_py()
    row0_target_ids = table.column("target_input_ids")[0].as_py()
    assert isinstance(row0_prompt_ids, list)
    assert isinstance(row0_target_ids, list)
    for x in row0_prompt_ids + row0_target_ids:
        assert isinstance(x, int)

    # Companion manifest must exist + record the right kind/tokenizer.
    manifest_path = Path(str(out) + ".manifest.json")
    assert manifest_path.exists(), "missing companion manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "swebench_lite_mini_tokenized"
    assert manifest["rows"] == 5
    assert manifest["truncation"] is True
    assert manifest["max_length"] == 4096
    assert manifest["dataset"] == "princeton-nlp/SWE-bench_Lite"


# ---------------------------------------------------------------------------
# 2. format_prompt embeds repo + problem_statement in the documented order.
# ---------------------------------------------------------------------------
def test_format_prompt_includes_repo(swe_module):
    """The prompt must surface the repo and end with the diff-prompt."""
    p = swe_module.format_prompt(
        "django/django",
        "AttributeError when foo() is called with None.",
    )
    # Documented format:
    #   Repo: <repo>\nProblem:\n<problem_statement>\n\nProvide a unified diff patch:\n
    assert p.startswith("Repo: django/django\n")
    assert "Problem:" in p
    assert "AttributeError when foo() is called with None." in p
    assert "Provide a unified diff patch:" in p
    # Repo line precedes the problem section.
    assert p.index("Repo:") < p.index("Problem:")
    # Problem section precedes the call-to-action.
    assert p.index("Problem:") < p.index("Provide a unified diff patch:")


# ---------------------------------------------------------------------------
# 3. Long patches truncate cleanly to max_length on both prompt + target.
# ---------------------------------------------------------------------------
def test_truncation_long_patch(swe_module):
    """A patch longer than ``max_length`` chars must hard-truncate."""
    tok = _MockTokenizer()
    long_patch = "x" * 10000  # 10k chars, > 4096 max_length
    long_problem = "y" * 8000  # 8k chars, > 4096 too

    rec = {
        "instance_id": "trunc__test-001",
        "repo": "trunc/test",
        "base_commit": "000",
        "problem_statement": long_problem,
        "patch": long_patch,
    }
    rows = swe_module.build_rows(
        iter([rec]),
        tokenizer=tok,
        n=1,
        max_length=4096,
    )
    assert len(rows) == 1
    r = rows[0]

    # Both prompt and target must be capped at max_length tokens.
    assert len(r["prompt_input_ids"]) <= 4096
    assert len(r["target_input_ids"]) <= 4096
    # And specifically: with 10k-char patch + 1-char/token mock, target
    # should hit the cap exactly.
    assert len(r["target_input_ids"]) == 4096

    # Stored raw fields are preserved (not truncated) — the parquet keeps
    # the canonical text for audit.
    assert len(r["patch_target"]) == 10000
    assert long_problem in r["prompt"]


# ---------------------------------------------------------------------------
# 4. argparse default --n is 50 (queue spec contract).
# ---------------------------------------------------------------------------
def test_n_50_default(swe_module):
    """Default ``--n`` must be 50 (matches DEEP_MAINT_QUEUE T3.10 spec)."""
    args = swe_module._parse_args([])
    assert args.n == 50, f"queue spec is 50; got {args.n}"
    # And smoke flag default off; max-length default 4096.
    assert args.smoke is False
    assert args.max_length == 4096
    # Out path default is set to the canonical workspace destination.
    assert "swebench_mini_qwen.parquet" in args.out


# ---------------------------------------------------------------------------
# Bonus: helper-level coverage so the tests don't depend solely on main().
# ---------------------------------------------------------------------------
def test_build_rows_sorts_by_instance_id(swe_module):
    """Records are deterministically sorted by ``instance_id`` before the cap."""
    recs = [
        {"instance_id": "z__last", "repo": "z/r", "base_commit": "x",
         "problem_statement": "p3", "patch": "diff3"},
        {"instance_id": "a__first", "repo": "a/r", "base_commit": "x",
         "problem_statement": "p1", "patch": "diff1"},
        {"instance_id": "m__middle", "repo": "m/r", "base_commit": "x",
         "problem_statement": "p2", "patch": "diff2"},
    ]
    rows = swe_module.build_rows(iter(recs), tokenizer=None, n=10)
    ids = [r["instance_id"] for r in rows]
    assert ids == ["a__first", "m__middle", "z__last"]


def test_build_rows_caps_at_n(swe_module):
    """``--n`` caps the row count even when more records are fed in."""
    recs = [
        {"instance_id": f"{i:04d}", "repo": "a/b", "base_commit": "x",
         "problem_statement": f"prob {i}", "patch": "diff"}
        for i in range(20)
    ]
    rows = swe_module.build_rows(iter(recs), tokenizer=None, n=7)
    assert len(rows) == 7
    # Sorted by instance_id (zero-padded) -> first 7.
    assert [r["instance_id"] for r in rows] == [f"{i:04d}" for i in range(7)]
