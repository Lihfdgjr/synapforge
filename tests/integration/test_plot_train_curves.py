"""Smoke tests for ``scripts/plot_train_curves.py`` (T5.5).

Covers:
  - regex parses for step / VAL / spike lines (real rental-log shape)
  - graceful behaviour when fields are missing (`[stl=...]` truncated)
  - end-to-end PNG emission on a synthetic 50-line mock log

All tests run on CPU with matplotlib's ``Agg`` backend (set by the script
itself before any pyplot import). No GPU / torch / rental ssh required.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# scripts/ is added to sys.path by tests/integration/conftest.py.
import plot_train_curves as ptc  # noqa: E402

# matplotlib is a soft dep at the repo level. Skip the PNG test cleanly
# when it isn't installed (the parsing tests don't need it).
try:
    import matplotlib  # noqa: F401  pylint: disable=unused-import
    _HAVE_MPL = True
except Exception:  # pragma: no cover -- only hits on minimal CI
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# Mock log fixtures
# ---------------------------------------------------------------------------
def _mock_lines(n_steps: int = 50,
                start_step: int = 4000,
                step_stride: int = 10,
                with_stl: bool = True,
                val_every: int = 25,
                spike_every: int = 5) -> list[str]:
    """Generate a synthetic train.log mirroring the rental format exactly.

    Default 50 step events + 2 VAL events + 10 spike events.
    """
    lines: list[str] = []
    for i in range(n_steps):
        step = start_step + i * step_stride
        ce = round(9.5 - (i / max(1, n_steps)) * 4.0, 4)  # 9.5 -> ~5.5
        kd = round(0.0 + (i / max(1, n_steps)) * 0.5, 4)
        z = round(94.0 + (i % 7) * 0.1, 3)
        lr = "3.0e-04"
        step_ms = round(50.0 + (i % 5), 2)
        tok_s = round(8000 + (i % 11) * 50, 1)
        mem = round(60.0 + (i % 3) * 0.5, 2)
        stl = round(0.05 + (i % 4) * 0.01, 4)
        line = (
            f"[01:35:{i % 60:02d}] step {step} loss={ce + 0.01:.4f} "
            f"ce={ce} kd={kd:.4f} z={z} lr={lr} "
            f"step_ms={step_ms} tok/s={tok_s} mem_GB={mem}"
        )
        if with_stl:
            line += f" stl={stl}"
        lines.append(line)

        if val_every and i and (i % val_every == 0):
            ppl_ttt = round(2000.0 - (i / max(1, n_steps)) * 1500.0, 2)
            ppl_ho = round(ppl_ttt * 1.05, 2)
            lines.append(
                f"[01:35:{i % 60:02d}] VAL step {step}: "
                f"val_ppl_ttt={ppl_ttt} val_ppl_holdout={ppl_ho} (honest)"
            )

        if spike_every and (i % spike_every == 0):
            mean = round(0.05 + (i / max(1, n_steps)) * 0.10, 4)
            dead = max(0, 5 - (i // 10))
            sat = max(0, (i // 25))
            lines.append(
                f"[01:35:{i % 60:02d}] spike: mean={mean} "
                f"range=[0.000, 0.500] dead={dead}/10 sat={sat}/10"
            )
    return lines


# ---------------------------------------------------------------------------
# Regex tests
# ---------------------------------------------------------------------------
class TestParseStepLine:
    """`step N ... ce=... kd=... z=... lr=... step_ms=... tok/s=... mem_GB=... [stl=...]`"""

    def test_full_line_with_stl(self):
        line = (
            "[01:35:12] step 4110 loss=9.4689 ce=9.458 kd=0.000 z=94.611 "
            "lr=3.0e-04 step_ms=52.34 tok/s=8050.0 mem_GB=60.5 stl=0.072"
        )
        ev = ptc.parse_step_line(line)
        assert ev is not None
        assert ev["step"] == 4110.0
        assert ev["ce"] == pytest.approx(9.458)
        assert ev["kd"] == pytest.approx(0.0)
        assert ev["z"] == pytest.approx(94.611)
        assert ev["lr"] == pytest.approx(3.0e-04)
        assert ev["step_ms"] == pytest.approx(52.34)
        assert ev["tok_s"] == pytest.approx(8050.0)
        assert ev["mem_gb"] == pytest.approx(60.5)
        assert ev["stl"] == pytest.approx(0.072)

    def test_line_without_stl(self):
        """Older logs lack the `[stl=...]` field -- must still parse."""
        line = (
            "[01:35:12] step 4110 loss=9.4689 ce=9.458 kd=0.000 z=94.611 "
            "lr=3.0e-04 step_ms=52.34 tok/s=8050.0 mem_GB=60.5"
        )
        ev = ptc.parse_step_line(line)
        assert ev is not None
        assert ev["step"] == 4110.0
        assert ev["ce"] == pytest.approx(9.458)
        assert "stl" not in ev  # gracefully absent

    def test_minimal_line(self):
        """Bare `step N ce=C` -- everything else should be optional."""
        line = "step 1 ce=2.5"
        ev = ptc.parse_step_line(line)
        assert ev is not None
        assert ev["step"] == 1.0
        assert ev["ce"] == pytest.approx(2.5)
        # No kd/z/lr/etc. should appear in the dict.
        for k in ("kd", "z", "lr", "step_ms", "tok_s", "mem_gb", "stl"):
            assert k not in ev

    def test_non_step_line_returns_none(self):
        assert ptc.parse_step_line("VAL step 100: val_ppl_ttt=500 val_ppl_holdout=600") is None
        assert ptc.parse_step_line("spike: mean=0.05 range=[0,1] dead=2/10 sat=0/10") is None
        assert ptc.parse_step_line("random log line") is None
        assert ptc.parse_step_line("") is None

    def test_negative_and_scientific_numbers(self):
        line = "step 50 ce=-1.5e-2 kd=0.0 z=0 lr=1e-5 step_ms=100 tok/s=1.5e3 mem_GB=10"
        ev = ptc.parse_step_line(line)
        assert ev is not None
        assert ev["ce"] == pytest.approx(-1.5e-2)
        assert ev["lr"] == pytest.approx(1e-5)
        assert ev["tok_s"] == pytest.approx(1500.0)


class TestParseValLine:
    def test_canonical(self):
        line = "[01:50:00] VAL step 4500: val_ppl_ttt=350.42 val_ppl_holdout=388.10 (honest)"
        ev = ptc.parse_val_line(line)
        assert ev is not None
        assert ev["step"] == 4500.0
        assert ev["ppl_ttt"] == pytest.approx(350.42)
        assert ev["ppl_holdout"] == pytest.approx(388.10)

    def test_no_honest_suffix(self):
        line = "VAL step 100: val_ppl_ttt=999 val_ppl_holdout=1000"
        ev = ptc.parse_val_line(line)
        assert ev is not None
        assert ev["step"] == 100.0
        assert ev["ppl_ttt"] == pytest.approx(999.0)
        assert ev["ppl_holdout"] == pytest.approx(1000.0)

    def test_step_line_returns_none(self):
        assert ptc.parse_val_line("step 4110 ce=9.458") is None
        assert ptc.parse_val_line("") is None
        assert ptc.parse_val_line("VAL step 100") is None  # missing fields


class TestParseSpikeLine:
    def test_canonical(self):
        line = "[01:35:12] spike: mean=0.085 range=[0.000, 0.450] dead=2/10 sat=1/10"
        ev = ptc.parse_spike_line(line, step_hint=4110)
        assert ev is not None
        assert ev["mean"] == pytest.approx(0.085)
        assert ev["range_lo"] == pytest.approx(0.0)
        assert ev["range_hi"] == pytest.approx(0.45)
        assert ev["dead"] == 2.0
        assert ev["total"] == 10.0
        assert ev["sat"] == 1.0
        assert ev["step"] == 4110.0

    def test_no_step_hint_drops_step(self):
        line = "spike: mean=0.05 range=[0, 0.5] dead=2/10 sat=0/10"
        ev = ptc.parse_spike_line(line, step_hint=None)
        # ev is non-None but has no `step` key; parse_log() filters those out.
        assert ev is not None
        assert "step" not in ev

    def test_non_spike_returns_none(self):
        assert ptc.parse_spike_line("step 100 ce=3.0", step_hint=100) is None
        assert ptc.parse_spike_line("VAL step 100: val_ppl_ttt=10 val_ppl_holdout=11",
                                   step_hint=100) is None


# ---------------------------------------------------------------------------
# parse_log() walks all 3 streams + maintains step_hint correctly
# ---------------------------------------------------------------------------
class TestParseLog:
    def test_walks_all_streams(self):
        lines = _mock_lines(n_steps=50)
        parsed = ptc.parse_log(lines)
        assert parsed.n_step == 50
        assert parsed.n_val == 1  # i=25 only (i=0 skipped by `if i and ...`)
        assert parsed.n_spike == 10  # i in {0,5,10,...,45}

    def test_spike_step_alignment(self):
        """Spike lines should inherit the step from the most recent `step N` line."""
        lines = [
            "step 100 ce=3.0",
            "spike: mean=0.05 range=[0, 0.5] dead=1/10 sat=0/10",
            "step 200 ce=2.5",
            "spike: mean=0.07 range=[0, 0.5] dead=0/10 sat=0/10",
        ]
        parsed = ptc.parse_log(lines)
        assert parsed.n_spike == 2
        assert parsed.spike_events[0]["step"] == 100
        assert parsed.spike_events[1]["step"] == 200

    def test_orphan_spike_dropped(self):
        """A spike line BEFORE any step line has no step_hint -> dropped."""
        lines = [
            "spike: mean=0.05 range=[0, 0.5] dead=1/10 sat=0/10",
            "step 50 ce=4.0",
        ]
        parsed = ptc.parse_log(lines)
        assert parsed.n_step == 1
        assert parsed.n_spike == 0  # orphan dropped

    def test_empty_log(self):
        parsed = ptc.parse_log([])
        assert parsed.n_step == parsed.n_val == parsed.n_spike == 0


# ---------------------------------------------------------------------------
# End-to-end PNG emission
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
class TestEmitPng:
    def test_writes_nonempty_png(self, tmp_path: Path):
        log = tmp_path / "train_run3test.log"
        log.write_text("\n".join(_mock_lines(n_steps=50)) + "\n", encoding="utf-8")
        out = tmp_path / "curves.png"
        rc = ptc.main(["--log", str(log), "--out", str(out), "--title", "test-fixture"])
        assert rc == 0
        assert out.exists(), "expected output PNG"
        size = out.stat().st_size
        assert size > 5000, f"PNG suspiciously small ({size} bytes); render likely failed"
        # Confirm magic bytes (PNG header is `\x89PNG\r\n\x1a\n`).
        head = out.read_bytes()[:8]
        assert head[:4] == b"\x89PNG", f"not a PNG file (header={head!r})"

    def test_handles_log_without_stl(self, tmp_path: Path):
        """Older logs lacking the [stl=...] field must still emit a valid PNG."""
        log = tmp_path / "old_train.log"
        log.write_text(
            "\n".join(_mock_lines(n_steps=20, with_stl=False)) + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "old.png"
        rc = ptc.main(["--log", str(log), "--out", str(out)])
        assert rc == 0
        assert out.stat().st_size > 5000

    def test_empty_log_still_emits(self, tmp_path: Path):
        """Zero-event logs should still produce a "(no data)" placeholder PNG, not crash."""
        log = tmp_path / "empty.log"
        log.write_text("just\nrandom\nnoise\n", encoding="utf-8")
        out = tmp_path / "empty.png"
        rc = ptc.main(["--log", str(log), "--out", str(out)])
        assert rc == 0
        assert out.stat().st_size > 1000  # placeholder text + axes still take some bytes

    def test_missing_log_returns_2(self, tmp_path: Path):
        rc = ptc.main(["--log", str(tmp_path / "does_not_exist.log"),
                       "--out", str(tmp_path / "x.png")])
        assert rc == 2

    def test_default_out_path_inside_docs(self, tmp_path: Path, monkeypatch):
        """When --out is omitted, default path is `<repo>/docs/CURVES_<run>.png`."""
        log = tmp_path / "train_run3z.log"
        log.write_text("\n".join(_mock_lines(n_steps=10)) + "\n", encoding="utf-8")
        # Compute expected destination via the same helper main() uses.
        expected = ptc._default_out_for(log)
        # Clean up if a prior test left it -- we'll recreate.
        if expected.exists():
            expected.unlink()
        try:
            rc = ptc.main(["--log", str(log)])
            assert rc == 0
            assert expected.exists(), f"default path {expected} not created"
            assert expected.stat().st_size > 5000
        finally:
            if expected.exists():
                expected.unlink()
