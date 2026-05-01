"""P22: phase_auto_relauncher.sh smoke + dry-run integration tests.

Resolves §6 P22 (auto-relaunch on phase threshold crossing). The bash
script polls ``<run_dir>/.phase`` and SIGTERMs+restarts the trainer with
the new phase's flags appended, replacing the manual "human reads JSON
then restarts" step.

These tests run the script in dry-run mode (``PAR_DRY_RUN=1`` +
``PAR_TEST_MODE=1``) so we never actually fork a trainer or kill any
real PIDs. We mock:
  * the run_dir as a tmp_path,
  * a fake ``.phase`` JSON written by hand (mirrors phase_manager.py output),
  * fake ``step_*.pt`` files (zero-byte; ``PAR_SKIP_STRIP=1`` keeps torch
    out of the loop),
  * the trainer PID via ``PAR_FAKE_PID`` and its argv via
    ``PAR_FAKE_CMDLINE_FILE``.

Asserts:
  * ``bash -n`` returns 0 on the script (syntax).
  * Dry-run on a phase=1 ``.phase`` JSON logs "phase change detected: -1 -> 1"
    and "DRY_RUN=1; not killing or spawning".
  * The rebuilt argv includes the phase-1 flags
    (``--self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05 --phase-aware``).
  * The argv drops ``--no-warmstart`` and points ``--warmstart`` at the
    stripped ckpt path.
  * If trainer PID is missing (no PAR_FAKE_PID, no real pgrep match) the
    script logs "trainer not running" and refuses to relaunch.
  * The ``flags_for_phase`` table for phase 1 matches phase_manager.py
    PHASES table exactly (no drift).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "phase_auto_relauncher.sh"
_BASH = shutil.which("bash") or "bash"


def _bash_path(p):
    r"""Forward-slashed path bash can read (works on Linux + git-bash on Windows).

    Mingw64 / git-bash treats backslash as escape, so a Windows path like
    ``D:\foo\bar`` gets mangled to ``D:foobar``. Replacing with forward
    slashes (``D:/foo/bar``) is accepted by both git-bash and rental Linux.
    """
    return str(p).replace("\\", "/")


def _have_bash() -> bool:
    return shutil.which("bash") is not None


pytestmark = pytest.mark.skipif(
    not _have_bash(), reason="bash not on PATH; required to run the relauncher script"
)


def test_script_exists():
    assert _SCRIPT.is_file(), f"missing {_SCRIPT}"


def test_bash_syntax():
    """bash -n is the spec's mandatory shellcheck stand-in."""
    rc = subprocess.run(
        [_BASH, "-n", _bash_path(_SCRIPT)], capture_output=True, text=True
    )
    assert rc.returncode == 0, f"bash -n failed: {rc.stderr}"


def _write_phase_json(run_dir: Path, phase_id: int) -> None:
    """Mirror phase_manager.write_phase_signal output."""
    payload = {
        "phase_id": phase_id,
        "phase_name": "intrinsic" if phase_id == 1 else f"phase{phase_id}",
        "ts": 1714567890.0,
        "state": {"last_step": 4000, "best_val_ppl": 240.0},
        "next_phase_flags": [
            "--self-learn-ttt",
            "--self-learn-k 8",
            "--curiosity-weight 0.05",
        ],
    }
    (run_dir / ".phase").write_text(json.dumps(payload, indent=2))


def _run_relauncher(run_dir: Path, env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Defaults that keep tests hermetic.
    env.update({
        "PAR_DRY_RUN": "1",
        "PAR_TEST_MODE": "1",
        "PAR_THRASH_SEC": "0",
        "PAR_SKIP_STRIP": "1",
        # T6.5: keep the real CHANGELOG.md untouched in dry-run tests.
        "PAR_SKIP_CHANGELOG": "1",
        "PAR_REPO_DIR": _bash_path(run_dir),
    })
    env.update(env_overrides)
    return subprocess.run(
        [
            _BASH, _bash_path(_SCRIPT),
            "--run-dir", _bash_path(run_dir),
            "--interval", "1",
        ],
        env=env, capture_output=True, text=True, timeout=20,
    )


def test_phase_1_dry_run_detects_change_and_builds_correct_argv(tmp_out_dir):
    run_dir = tmp_out_dir
    _write_phase_json(run_dir, phase_id=1)
    # Fake step ckpt the script will pick.
    (run_dir / "step_002500.pt").write_bytes(b"")
    # Fake the trainer's current cmdline so build_new_argv has input.
    cmdline = run_dir / "fake_cmdline"
    cmdline.write_text(
        f"python3 train_100m_kd.py --warmstart /old/path/step_001000.pt "
        f"--no-warmstart --teacher Qwen/Qwen2.5-0.5B "
        f"--out {run_dir} --batch-size 64 --steps 30000"
    )
    rc = _run_relauncher(run_dir, {
        "PAR_FAKE_PID": "99999",
        "PAR_FAKE_CMDLINE_FILE": _bash_path(cmdline),
    })
    out = rc.stdout + rc.stderr
    assert rc.returncode == 0, f"non-zero rc; out=\n{out}"

    # phase change detected
    assert "phase change detected: -1 -> 1" in out, out

    # dry-run did not actually spawn anything
    assert "DRY_RUN=1; not killing or spawning" in out, out

    # phase-1 flags appended verbatim (from phase_manager PHASES[1])
    assert "--self-learn-ttt" in out, out
    assert "--self-learn-k 8" in out, out
    assert "--curiosity-weight 0.05" in out, out
    assert "--phase-aware" in out, out

    # --no-warmstart stripped from rebuilt argv
    restart_log = (run_dir / "phase_restart.log").read_text()
    argv_line = [ln for ln in restart_log.splitlines() if "argv=" in ln]
    assert argv_line, f"no argv= line in restart log:\n{restart_log}"
    argv = argv_line[-1].split("argv=", 1)[1]
    assert "--no-warmstart" not in argv, f"--no-warmstart should be dropped, got: {argv}"

    # --warmstart points at the stripped (no_optim) ckpt, not the old one
    assert "step_002500_no_optim.pt" in argv, f"warmstart not redirected: {argv}"
    assert "/old/path/step_001000.pt" not in argv, f"old warmstart leaked: {argv}"

    # last-phase state file persisted
    assert (run_dir / ".par_last_phase").read_text().strip() == "1"


def test_no_trainer_pid_does_not_relaunch(tmp_out_dir):
    """Spec safety guard: if no PID, log + skip; do not start trainer."""
    run_dir = tmp_out_dir
    _write_phase_json(run_dir, phase_id=1)
    (run_dir / "step_001000.pt").write_bytes(b"")
    # PAR_FAKE_PID="" -> find_trainer_pid falls through to pgrep, which
    # on this hermetic test box has nothing matching the run_dir path.
    rc = _run_relauncher(run_dir, {"PAR_FAKE_PID": ""})
    out = rc.stdout + rc.stderr
    assert rc.returncode == 0
    assert "trainer not running" in out, out
    # last-phase NOT advanced (relaunch failed safely)
    assert not (run_dir / ".par_last_phase").exists() or \
        (run_dir / ".par_last_phase").read_text().strip() != "1"


def test_no_phase_file_no_action(tmp_out_dir):
    """No .phase yet -> single poll iteration -> exit cleanly under TEST_MODE."""
    rc = _run_relauncher(tmp_out_dir, {"PAR_FAKE_PID": "1"})
    out = rc.stdout + rc.stderr
    assert rc.returncode == 0
    assert "phase change detected" not in out
    assert "TEST_MODE=1; no phase event" in out


def test_phase_1_flags_match_phase_manager_table():
    """Drift guard: phase_auto_relauncher's phase-1 flag string must match
    phase_manager.PHASES[1]['flags']. We parse the script source rather
    than `source`-ing it (the arg parser exits without --run-dir).
    """
    text = _SCRIPT.read_text(encoding="utf-8")
    # The phase 1 line is uniquely tagged; assert the canonical flag set.
    expected = "--self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05 --phase-aware"
    assert expected in text, (
        f"phase 1 flags drift: expected {expected!r} in script text"
    )

    # Cross-reference against phase_manager.PHASES (skip if torch not
    # importable since phase_manager.py imports nothing heavy, this is
    # pure stdlib so it should always work).
    import importlib.util
    pm_path = _REPO_ROOT / "scripts" / "phase_manager.py"
    spec = importlib.util.spec_from_file_location("phase_manager", pm_path)
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    pm_phase1 = next(p for p in pm.PHASES if p["id"] == 1)
    assert "--self-learn-ttt" in pm_phase1["flags"]
    assert "--self-learn-k 8" in pm_phase1["flags"]
    assert "--curiosity-weight 0.05" in pm_phase1["flags"]
