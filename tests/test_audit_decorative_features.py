"""Tests for scripts/audit_decorative_features.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "audit_decorative_features.py"


def _run(extra_argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra_argv],
        capture_output=True, text=True, timeout=20, cwd=cwd,
    )


def test_audit_exists_and_runs(tmp_path: Path) -> None:
    """Audit script exists and exits 0 on the working tree (all activated)."""
    out_md = tmp_path / "DECORATIVE_AUDIT.md"
    rc = _run(["--out-md", str(out_md), "--json"])
    assert rc.returncode == 0, (
        f"audit failed: rc={rc.returncode}\nstdout={rc.stdout}\n"
        f"stderr={rc.stderr}"
    )
    assert out_md.is_file()
    body = out_md.read_text(encoding="utf-8")
    assert "Decorative" in body
    assert "STDP-only routing" in body
    assert "Multimodal byte-patch" in body
    assert "Web daemon" in body
    assert "R-fold" in body


def test_audit_json_schema(tmp_path: Path) -> None:
    out_md = tmp_path / "AUDIT.md"
    rc = _run(["--out-md", str(out_md), "--json"])
    assert rc.returncode == 0, rc.stderr
    json_path = out_md.with_suffix(".json")
    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    # Six features
    assert len(data) == 6, f"expected 6 features, got {len(data)}"
    # Required keys per entry
    for entry in data:
        for key in ("name", "file_path", "tag_or_flag", "file_exists",
                    "tag_set", "flag_in_launcher",
                    "actually_active", "blocked_reason", "notes"):
            assert key in entry, f"missing {key} in {entry.get('name')}"


def test_audit_marks_ternary_blocked(tmp_path: Path) -> None:
    """Ternary should be blocked (deferred per spec)."""
    out_md = tmp_path / "AUDIT.md"
    _run(["--out-md", str(out_md), "--json"])
    json_path = out_md.with_suffix(".json")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    ternary = next(
        e for e in data if "Ternary" in e["name"]
    )
    assert ternary["blocked_reason"] is not None
    assert "DEFERRED" in ternary["blocked_reason"]


def test_audit_marks_plif_blocked(tmp_path: Path) -> None:
    out_md = tmp_path / "AUDIT.md"
    _run(["--out-md", str(out_md), "--json"])
    json_path = out_md.with_suffix(".json")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    plif = next(
        e for e in data if "PLIF" in e["name"]
    )
    assert plif["blocked_reason"] is not None
    assert "step 4000" in plif["blocked_reason"] or "DEPENDENCY" in plif["blocked_reason"]


def test_audit_log_probe_unknown_when_no_log(tmp_path: Path) -> None:
    out_md = tmp_path / "AUDIT.md"
    _run(["--out-md", str(out_md), "--json"])
    json_path = out_md.with_suffix(".json")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for entry in data:
        # Without --log, actually_active is None (unknown)
        assert entry["actually_active"] is None


def test_audit_log_probe_with_synthetic_log(tmp_path: Path) -> None:
    """Pass a synthetic training log that mentions the activation tags;
    audit should report actually_active=True for the matched features."""
    log = tmp_path / "fake_train.log"
    log.write_text(
        "[mixin] MultimodalMixin enabled: ('image', 'audio')\n"
        "[run8-full] modal: image,audio,time_series dir=/x alpha=0.1\n"
        "loaded liquid_rfold; rfold=True\n"
        "[mixin] NeuroMCPHead enabled\n"
        "wrote /workspace/data/web_self_learn/web_0001.parquet\n",
        encoding="utf-8",
    )
    out_md = tmp_path / "AUDIT.md"
    rc = _run(["--out-md", str(out_md), "--json", "--log", str(log)])
    # rc 0 even with log probe (we do not fail the script if a few
    # features didn't show up in the log)
    json_path = out_md.with_suffix(".json")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # All four features should show actually_active=True now
    by_name = {e["name"]: e for e in data}
    assert by_name["Multimodal byte-patch (image/audio/time_series)"]["actually_active"] is True
    assert by_name["R-fold parallel scan in TRAINING (not just inference)"]["actually_active"] is True
    assert by_name["STDP-only routing on SparseSynapticLayer"]["actually_active"] is True
    assert by_name["Web daemon -> trainer parquet pipe"]["actually_active"] is True


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        test_audit_exists_and_runs(td); print("OK runs")
        test_audit_json_schema(td); print("OK schema")
        test_audit_marks_ternary_blocked(td); print("OK ternary blocked")
        test_audit_marks_plif_blocked(td); print("OK plif blocked")
