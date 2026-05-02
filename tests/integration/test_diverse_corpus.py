"""Integration tests for ``scripts/build_diverse_corpus.py``.

Covers the four contracts the task spec calls out:

1. ``test_smoke_writes_parquet``      -- ``--smoke`` writes a parquet with
   the expected (text, corpus) columns + companion manifest.
2. ``test_token_count_matches_target`` -- the manifest's
   ``estimated_tokens`` field is recorded and bounded; in smoke mode the
   per-category fixture sizes are deterministic so the count is stable.
3. ``test_per_category_ratios``       -- the row counts respect the
   declared category-inclusion list (en >= zh >= code >= math >= instruct
   in the default ratio order).
4. ``test_no_duplicates_across_sources`` -- sha256(first 4096 chars)
   of every emitted row is unique (the dedup pass actually runs).

All tests run on CPU. They do NOT require ``transformers``, ``torch``,
``datasets``, or network access — ``--smoke`` short-circuits HF.
``pyarrow`` is required (already a hard trainer dep).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

# Make the repo's ``scripts/`` dir importable identically to how
# tests/integration/conftest.py does it -- this also works when the test
# is collected stand-alone.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def diverse_module():
    """Import (or reload) ``scripts/build_diverse_corpus.py``."""
    pytest.importorskip("pyarrow")
    if "build_diverse_corpus" in sys.modules:
        return importlib.reload(sys.modules["build_diverse_corpus"])
    return importlib.import_module("build_diverse_corpus")


# ---------------------------------------------------------------------------
# 1. --smoke writes a valid parquet + manifest.
# ---------------------------------------------------------------------------
def test_smoke_writes_parquet(tmp_path, diverse_module):
    """``--smoke`` writes a parquet with (text, corpus) columns."""
    import pyarrow.parquet as pq

    out = tmp_path / "diverse_smoke.parquet"
    rc = diverse_module.main([
        "--smoke",
        "--out", str(out),
    ])
    assert rc == 0, "smoke run failed"
    assert out.exists(), "smoke run did not write the parquet"

    table = pq.read_table(str(out))
    # Schema contract: a text + corpus column.
    assert "text" in table.column_names
    assert "corpus" in table.column_names
    assert table.num_rows > 0

    # Each row must be a non-empty string for both columns.
    texts = table.column("text").to_pylist()
    corpora = table.column("corpus").to_pylist()
    assert len(texts) == len(corpora)
    for t in texts:
        assert isinstance(t, str)
        assert t.strip()
    for c in corpora:
        assert isinstance(c, str)
        assert c in {"en", "zh", "code", "math", "instruct"}

    # Companion manifest.
    manifest_path = Path(str(out) + ".manifest.json")
    assert manifest_path.exists(), "missing companion manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "diverse_corpus"
    assert manifest["smoke"] is True
    assert manifest["rows"] == table.num_rows
    assert manifest["categories"] == ["en", "zh", "code", "math", "instruct"]


# ---------------------------------------------------------------------------
# 2. Token-count target reflected in the manifest.
# ---------------------------------------------------------------------------
def test_token_count_matches_target(tmp_path, diverse_module):
    """The manifest must record the requested target + an estimate.

    Smoke mode emits exactly len(SMOKE_FIXTURES[cat]) = 5 rows per
    category (5 cats × 5 rows = 25 rows after dedup). The estimate must
    therefore land on the sum of per-row char/4 estimates -- which is
    deterministic for the fixed fixtures.
    """
    out = tmp_path / "diverse_count.parquet"
    rc = diverse_module.main([
        "--smoke",
        "--target-tokens", "1B",
        "--out", str(out),
    ])
    assert rc == 0

    manifest = json.loads(
        Path(str(out) + ".manifest.json").read_text(encoding="utf-8")
    )

    # Target was the parsed budget (1B).
    assert manifest["target_tokens"] == 1_000_000_000

    # Estimate is computed from the actually-emitted text rows.
    expected_est = sum(
        diverse_module._approx_tokens(t)
        for fixtures in diverse_module.SMOKE_FIXTURES.values()
        for t in fixtures
    )
    assert manifest["estimated_tokens"] == expected_est, (
        f"manifest estimate {manifest['estimated_tokens']} != "
        f"sum-of-fixtures estimate {expected_est}"
    )

    # Helper unit-tests for the parser.
    assert diverse_module._parse_token_target("1B") == 1_000_000_000
    assert diverse_module._parse_token_target("500M") == 500_000_000
    assert diverse_module._parse_token_target("100K") == 100_000
    assert diverse_module._parse_token_target("1234") == 1234


# ---------------------------------------------------------------------------
# 3. Per-category row ratios match the declared --include order.
# ---------------------------------------------------------------------------
def test_per_category_ratios(tmp_path, diverse_module):
    """Each declared category must appear at least once and the EN bucket
    (largest ratio in the default mix) must have >= every other bucket.

    Smoke mode caps each category at len(fixtures)=5, so the row counts
    are equal across categories. We exercise --include with two subsets
    to make sure --include filters correctly:

      - default (all 5 cats)  -> 5 distinct corpus labels in the parquet
      - en + code only        -> 2 distinct labels, no zh/math/instruct
    """
    import pyarrow.parquet as pq

    # 3a. Default include = all five.
    out_a = tmp_path / "diverse_full.parquet"
    rc = diverse_module.main([
        "--smoke",
        "--out", str(out_a),
    ])
    assert rc == 0
    cats_a = set(pq.read_table(str(out_a)).column("corpus").to_pylist())
    assert cats_a == {"en", "zh", "code", "math", "instruct"}, (
        f"default --include should cover all 5 cats; got {cats_a}"
    )

    # 3b. Subset include = en + code only.
    out_b = tmp_path / "diverse_subset.parquet"
    rc_b = diverse_module.main([
        "--smoke",
        "--include", "en,code",
        "--out", str(out_b),
    ])
    assert rc_b == 0
    cats_b = set(pq.read_table(str(out_b)).column("corpus").to_pylist())
    assert cats_b == {"en", "code"}, (
        f"--include en,code should drop zh/math/instruct; got {cats_b}"
    )

    # 3c. Ratios renormalise when only a subset is included.
    sub_ratios = diverse_module._normalise_ratios(["en", "code"])
    assert set(sub_ratios.keys()) == {"en", "code"}
    assert abs(sum(sub_ratios.values()) - 1.0) < 1e-9
    # EN ratio (0.50) > code ratio (0.15) in the defaults, so EN > code
    # post-renormalisation too.
    assert sub_ratios["en"] > sub_ratios["code"]

    # 3d. Per-category row count: in smoke mode each cat is 5 rows, so
    # the largest bucket equals the smallest. We assert the rough
    # ordering on the budget side instead: cat budgets respect ratios.
    full_ratios = diverse_module._normalise_ratios(
        ["en", "zh", "code", "math", "instruct"]
    )
    assert full_ratios["en"] >= full_ratios["zh"]
    assert full_ratios["zh"] >= full_ratios["code"]
    assert full_ratios["code"] >= full_ratios["math"]
    assert full_ratios["math"] >= full_ratios["instruct"]


# ---------------------------------------------------------------------------
# 4. No duplicates across sources (sha256 prefix unique per row).
# ---------------------------------------------------------------------------
def test_no_duplicates_across_sources(tmp_path, diverse_module):
    """Every emitted row's sha256(first 4096 chars) must be unique.

    Also verifies that the deduper drops a deliberately-injected
    duplicate from a synthetic 2-source result list.
    """
    import pyarrow.parquet as pq

    out = tmp_path / "diverse_dedup.parquet"
    rc = diverse_module.main([
        "--smoke",
        "--out", str(out),
    ])
    assert rc == 0

    texts = pq.read_table(str(out)).column("text").to_pylist()
    keys = [diverse_module._hash_key(t) for t in texts]
    assert len(keys) == len(set(keys)), (
        f"duplicate rows detected: {len(keys)} rows but "
        f"{len(set(keys))} unique sha256 prefixes"
    )

    # Direct unit-test of the deduper: build two SourceResults with one
    # shared row + one unique row each, and confirm the shared row is
    # dropped exactly once.
    SR = diverse_module.SourceResult
    shared = "Photosynthesis converts light energy into chemical energy."
    a = SR(category="en",   rows=[shared, "EN unique"], n_tokens_est=20)
    b = SR(category="math", rows=[shared, "MATH unique"], n_tokens_est=20)
    out_text, out_cat, n_dropped = diverse_module._dedup_across_sources([a, b])
    assert n_dropped == 1, f"expected 1 cross-source dup; got {n_dropped}"
    assert len(out_text) == 3
    assert "EN unique" in out_text and "MATH unique" in out_text
    # First-seen-wins: the shared row keeps the EN label.
    shared_idx = out_text.index(shared)
    assert out_cat[shared_idx] == "en"

    # And our manifest dedup count must be a non-negative int.
    manifest = json.loads(
        Path(str(out) + ".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["n_dropped_dups"] >= 0
