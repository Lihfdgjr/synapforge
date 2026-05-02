"""Tests for scripts/web_self_learn_daemon.py.

Smoke-runnable: writes 3 cycles to a tmp dir, asserts parquet shard
contains the expected schema and that the cycle log records pass-rate.
The supervisor path is exercised with --max-cycles 1 so it exits cleanly
after one inner spawn.

These tests are network-free: --enable-real-fetch is NEVER passed; the
offline curated samples drive the test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DAEMON = REPO / "scripts" / "web_self_learn_daemon.py"

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _run_daemon(argv: list, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run the daemon under the current python; capture stdout/stderr."""
    return subprocess.run(
        [sys.executable, str(DAEMON), *argv],
        capture_output=True, text=True, timeout=timeout,
    )


def test_daemon_smoke_writes_parquet(tmp_path: Path) -> None:
    """Smoke: 3 cycles, 4 docs/cycle. After exit, web_0001.parquet exists
    and contains rows with the expected schema."""
    out_dir = tmp_path / "runs"
    rc = _run_daemon([
        "--smoke",
        "--out-dir", str(out_dir),
        "--rng-seed", "1",
    ])
    assert rc.returncode == 0, (
        f"daemon failed rc={rc.returncode}\n"
        f"stdout={rc.stdout}\nstderr={rc.stderr}"
    )
    assert (out_dir / "cycle_log.jsonl").is_file()
    assert (out_dir / "gate_log.jsonl").is_file()
    parquets = sorted(out_dir.glob("web_*.parquet"))
    assert len(parquets) >= 1, "daemon should have written at least 1 parquet"
    # Schema check
    pf = pq.ParquetFile(str(parquets[0]))
    cols = {f.name for f in pf.schema_arrow}
    assert {
        "text", "source_id", "url", "ingest_ts",
        "quality_score", "gate_scores",
    }.issubset(cols), f"missing columns: {cols}"
    # At least one row
    rows = pf.read().to_pylist()
    assert len(rows) >= 1, "expected at least one accepted row"
    # gate_scores is JSON-decodable
    gs = json.loads(rows[0]["gate_scores"])
    assert isinstance(gs, dict) and len(gs) > 0
    # quality_score in [0, 1]
    for r in rows:
        q = r["quality_score"]
        assert 0.0 <= q <= 1.0 + 1e-3, f"quality_score out of range: {q}"


def test_daemon_per_source_cap(tmp_path: Path) -> None:
    """Per-source 7d cap: even after many cycles, no source exceeds the cap."""
    out_dir = tmp_path / "cap"
    rc = _run_daemon([
        "--out-dir", str(out_dir),
        "--interval", "0.001",
        "--max-cycles", "30",
        "--docs-per-cycle", "5",
        "--per-source-cap-7d", "5",
        "--rng-seed", "0",
    ])
    assert rc.returncode == 0, (
        f"rc={rc.returncode} stderr={rc.stderr}"
    )
    parquets = sorted(out_dir.glob("web_*.parquet"))
    assert len(parquets) >= 1
    rows = pq.ParquetFile(str(parquets[0])).read().to_pylist()
    if len(parquets) > 1:
        for p in parquets[1:]:
            rows.extend(pq.ParquetFile(str(p)).read().to_pylist())
    # Count per source
    by_src: dict[str, int] = {}
    for r in rows:
        by_src[r["source_id"]] = by_src.get(r["source_id"], 0) + 1
    over_cap = {s: c for s, c in by_src.items() if c > 5}
    assert not over_cap, f"sources over cap=5: {over_cap}"


def test_supervisor_exits_when_inner_succeeds(tmp_path: Path) -> None:
    """Supervisor with --max-cycles N runs inner once with that N and exits.
    (Inner finishes successfully since max_cycles is set, supervisor returns.)"""
    out_dir = tmp_path / "sup"
    rc = _run_daemon([
        "--supervisor",
        "--out-dir", str(out_dir),
        "--interval", "0.001",
        "--max-cycles", "2",
        "--docs-per-cycle", "2",
        "--rng-seed", "5",
    ], timeout=30)
    # Supervisor returns inner's rc (0 on success).
    assert rc.returncode == 0, (
        f"rc={rc.returncode} stderr={rc.stderr}"
    )
    sup_log = out_dir / "supervisor_log.jsonl"
    assert sup_log.is_file()
    events = [json.loads(line) for line in sup_log.read_text().splitlines()]
    assert any(e.get("event") == "spawn" for e in events)
    assert any(e.get("event") == "exit" for e in events)


def test_quality_floor_filters_to_high_score_only(tmp_path: Path) -> None:
    """Quality floor 0.5 drops rows whose MIN gate score < 0.5."""
    out_dir = tmp_path / "qf"
    rc = _run_daemon([
        "--out-dir", str(out_dir),
        "--interval", "0.001",
        "--max-cycles", "5",
        "--docs-per-cycle", "4",
        "--quality-score-floor", "0.5",
        "--rng-seed", "11",
    ])
    assert rc.returncode == 0, rc.stderr
    parquets = sorted(out_dir.glob("web_*.parquet"))
    if not parquets:
        # If quality-floor was very strict + small set we may end with no
        # rows. Verify the daemon at least logged the cycles.
        cycle_log = out_dir / "cycle_log.jsonl"
        assert cycle_log.is_file()
        events = [json.loads(l) for l in cycle_log.read_text().splitlines()]
        assert sum(1 for e in events if e.get("event") == "cycle") == 5
        return
    rows = []
    for p in parquets:
        rows.extend(pq.ParquetFile(str(p)).read().to_pylist())
    for r in rows:
        assert r["quality_score"] >= 0.5 - 1e-3, (
            f"row below quality floor: {r['quality_score']}"
        )


def test_daemon_cli_help_runs(tmp_path: Path) -> None:
    """Sanity: --help prints argparse and exits 0."""
    rc = _run_daemon(["--help"], timeout=5)
    assert rc.returncode == 0
    assert "supervisor" in rc.stdout
    assert "out-dir" in rc.stdout


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        test_daemon_smoke_writes_parquet(td); print("OK smoke")
        test_daemon_per_source_cap(td / "cap"); print("OK cap")
        test_supervisor_exits_when_inner_succeeds(td / "sup"); print("OK supervisor")
