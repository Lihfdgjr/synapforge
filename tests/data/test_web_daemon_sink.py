"""Tests for synapforge.data.web_daemon_sink."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from synapforge.data.web_daemon_sink import (  # noqa: E402
    WebDaemonSink,
    append_decision,
)


def test_basic_append_and_close(tmp_path: Path) -> None:
    sink = WebDaemonSink(
        out_dir=tmp_path, write_batch_size=2, rotate_max_rows=10,
    )
    assert sink.append(
        text="hello world", source_id="web:wikipedia.zh",
        url="https://wikipedia.zh/x", quality_score=0.9,
        gate_scores={"G1": 0.9, "G2": 0.95},
    )
    assert sink.append(
        text="goodbye world", source_id="web:arxiv.org",
        url=None, quality_score=0.6, gate_scores={"G1": 0.6},
    )
    sink.close()
    files = list(tmp_path.glob("web_*.parquet"))
    assert len(files) == 1, files
    rows = pq.ParquetFile(str(files[0])).read().to_pylist()
    assert len(rows) == 2
    assert rows[0]["text"] == "hello world"
    assert rows[0]["source_id"] == "web:wikipedia.zh"
    assert rows[0]["url"] == "https://wikipedia.zh/x"
    assert rows[0]["quality_score"] == pytest.approx(0.9, rel=1e-3)
    assert rows[1]["url"] == ""  # None -> empty string
    # gate_scores is JSON-encoded
    assert json.loads(rows[0]["gate_scores"]) == {"G1": 0.9, "G2": 0.95}


def test_empty_text_rejected(tmp_path: Path) -> None:
    sink = WebDaemonSink(out_dir=tmp_path)
    assert not sink.append(text="", source_id="x", quality_score=1.0)
    sink.close()
    # No file should have been opened.
    files = list(tmp_path.glob("web_*.parquet"))
    assert files == []


def test_quality_floor_drops_low_scores(tmp_path: Path) -> None:
    sink = WebDaemonSink(out_dir=tmp_path, quality_score_floor=0.5)
    assert sink.append(text="below", source_id="x", quality_score=0.4) is False
    assert sink.append(text="above", source_id="x", quality_score=0.7) is True
    sink.close()
    files = list(tmp_path.glob("web_*.parquet"))
    rows = pq.ParquetFile(str(files[0])).read().to_pylist()
    assert len(rows) == 1
    assert rows[0]["text"] == "above"


def test_rotation_by_max_rows(tmp_path: Path) -> None:
    """Trigger rotation by exceeding rotate_max_rows."""
    sink = WebDaemonSink(
        out_dir=tmp_path, write_batch_size=1, rotate_max_rows=3,
    )
    for i in range(7):
        ok = sink.append(
            text=f"row{i}", source_id="web:src", quality_score=0.8,
        )
        assert ok, f"row {i} should be accepted"
    sink.close()
    files = sorted(tmp_path.glob("web_*.parquet"))
    # 7 rows / rotate_max_rows=3 ~ 3 shards (3+3+1)
    assert len(files) >= 2, files
    total = sum(pq.ParquetFile(str(f)).metadata.num_rows for f in files)
    assert total == 7, f"expected 7 total rows, got {total}"


def test_rotation_by_age(tmp_path: Path) -> None:
    """rotate_seconds=0 forces rotation on every append after first."""
    sink = WebDaemonSink(
        out_dir=tmp_path, write_batch_size=1, rotate_seconds=0.0001,
    )
    sink.append(text="a", source_id="x", quality_score=0.9)
    time.sleep(0.005)
    sink.append(text="b", source_id="x", quality_score=0.9)
    time.sleep(0.005)
    sink.append(text="c", source_id="x", quality_score=0.9)
    sink.close()
    files = sorted(tmp_path.glob("web_*.parquet"))
    # We should see >=2 shards since rotate_seconds is essentially 0.
    assert len(files) >= 2, [str(f) for f in files]


def test_restart_continues_index(tmp_path: Path) -> None:
    """On restart, sink picks up after the highest existing shard index."""
    s1 = WebDaemonSink(out_dir=tmp_path, write_batch_size=1)
    s1.append(text="a", source_id="x", quality_score=1.0)
    s1.close()
    files1 = sorted(tmp_path.glob("web_*.parquet"))
    assert len(files1) == 1
    assert files1[0].name == "web_0001.parquet"

    s2 = WebDaemonSink(out_dir=tmp_path, write_batch_size=1)
    s2.append(text="b", source_id="x", quality_score=1.0)
    s2.close()
    files2 = sorted(tmp_path.glob("web_*.parquet"))
    assert len(files2) == 2
    assert files2[1].name == "web_0002.parquet"


def test_shard_glob_format(tmp_path: Path) -> None:
    sink = WebDaemonSink(out_dir=tmp_path)
    glob = sink.shard_glob()
    assert "web_*.parquet" in glob
    assert str(tmp_path) in glob
    sink.close()


def test_append_decision_translates_gates(tmp_path: Path) -> None:
    """``append_decision`` shape mirrors continual_daemon.IngestDecision."""

    class _G:
        def __init__(self, name, score, accept=True):
            self.name = name
            self.score = score
            self.accept = accept

    class _D:
        def __init__(self, accept, gates):
            self.accept = accept
            self.gates = gates

    sink = WebDaemonSink(out_dir=tmp_path, write_batch_size=1)
    decision = _D(
        accept=True,
        gates=[
            _G("G1_source_trust", 0.7),
            _G("G3_perplexity", 0.92),
            _G("G7_trak", 0.55),
        ],
    )
    ok = append_decision(
        sink, decision,
        text="real example", source_id="web:wikipedia.en",
        url="https://en.wiki/x",
    )
    assert ok
    sink.close()
    files = list(tmp_path.glob("web_*.parquet"))
    rows = pq.ParquetFile(str(files[0])).read().to_pylist()
    assert len(rows) == 1
    # quality_score = MIN over gates (weakest link)
    assert rows[0]["quality_score"] == pytest.approx(0.55, rel=1e-3)
    gs = json.loads(rows[0]["gate_scores"])
    assert gs == {
        "G1_source_trust": 0.7,
        "G3_perplexity": 0.92,
        "G7_trak": 0.55,
    }


def test_append_decision_rejected_does_nothing(tmp_path: Path) -> None:
    class _D:
        accept = False
        gates: list = []

    sink = WebDaemonSink(out_dir=tmp_path, write_batch_size=1)
    out = append_decision(
        sink, _D(), text="nope", source_id="web:bad",
    )
    assert out is False
    sink.close()
    files = list(tmp_path.glob("web_*.parquet"))
    assert files == []


def test_stats_keys(tmp_path: Path) -> None:
    sink = WebDaemonSink(out_dir=tmp_path)
    s = sink.stats()
    for key in (
        "out_dir", "next_index", "open_writer_path",
        "open_writer_rows", "buffer_rows", "rotate_seconds",
        "rotate_max_rows", "rotate_max_bytes",
        "quality_score_floor",
    ):
        assert key in s, f"stats missing {key}: {s}"
    sink.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        test_basic_append_and_close(td); print("OK basic append")
        test_quality_floor_drops_low_scores(td / "qf"); print("OK quality floor")
        test_rotation_by_max_rows(td / "rot"); print("OK rotation by rows")
