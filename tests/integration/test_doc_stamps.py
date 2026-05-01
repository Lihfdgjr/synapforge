"""Integration test for ``scripts/check_doc_stamps.py``.

Goal: CI fails only when the doc-stamp tool itself is broken (unparseable
output / non-{0,1} return code), not when docs are merely stale. Stale-doc
discovery is informational; humans react to it.

Resolves P10 in docs/MASTER_PLAN.md.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_doc_stamps.py"


pytestmark = pytest.mark.docs


def _run_checker(*flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *flags],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_exists():
    assert SCRIPT.exists(), f"missing: {SCRIPT}"


def test_stamp_file_exists():
    assert (REPO_ROOT / "docs" / "_stamp.json").exists()


def test_returncode_is_zero_or_one():
    """Anything other than 0 (no STALE) or 1 (STALE) means the tool itself broke."""
    result = _run_checker()
    assert result.returncode in (0, 1), (
        f"unexpected returncode {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_output_is_parseable_table():
    """Output must be a 3-column markdown table with one row per doc."""
    result = _run_checker()
    out = result.stdout.strip()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) >= 3, f"too few lines:\n{out}"
    # Header.
    assert lines[0].startswith("| doc"), f"bad header: {lines[0]!r}"
    assert lines[1].startswith("|---"), f"bad separator: {lines[1]!r}"
    # Body rows: each must have exactly 3 |-separated cells.
    body = lines[2:]
    for row in body:
        cells = [c for c in row.split("|") if c.strip()]
        assert len(cells) == 3, f"bad row (need 3 cells): {row!r}"
    # At least one row must reference docs/ to ensure the script saw the dir.
    assert any("docs/" in ln for ln in body), "no docs/ references in body"


def test_status_values_are_known():
    """Every status cell must be one of the documented set."""
    result = _run_checker()
    valid = {"fresh", "auto-fresh", "MAYBE STALE", "STALE"}
    rx = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")
    for line in result.stdout.splitlines():
        if not line.startswith("| docs/"):
            continue
        m = rx.search(line)
        assert m is not None, f"can't parse row: {line!r}"
        status = m.group(2).strip()
        assert status in valid, f"unknown status {status!r} in row: {line!r}"


def test_json_mode_round_trip():
    """``--json`` must emit a JSON array with doc/status/reason keys."""
    import json as _json

    result = _run_checker("--json")
    assert result.returncode in (0, 1)
    payload = _json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload, "empty payload"
    keys = {"doc", "status", "reason"}
    for entry in payload:
        assert keys <= set(entry), f"missing keys in {entry!r}"
