"""phase_signal — atomic .phase file plumbing between phase_manager and trainer.

`scripts/phase_manager.py` watches `train.log`, decides when to enter the next
training phase, and writes a `.phase` JSON into the run directory. The trainer
should poll for that file and, on transition, save a checkpoint and exit with
a relauncher-friendly status code (101) so an outer wrapper can re-spawn it
with the new flag set.

Public API:
    read_phase(out_dir)     -- non-destructive read, returns dict | None
    consume_phase(out_dir)  -- atomic rename to .phase.consumed.<ts>; idempotent

Defensive guarantees:
* Missing dir, missing file, malformed JSON -> returns None (never raises).
* `consume_phase` uses os.replace so the rename is atomic even on Windows;
  a second consume in the same instant returns None instead of double-firing.
* The consumed copy is preserved for audit; the trainer can confirm what it
  saw by reading `.phase.consumed.<ts>`.

Smoke entry point at the bottom builds a tempdir, writes a fake payload,
and verifies read + consume + idempotency.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

PHASE_FILE = ".phase"
CONSUMED_PREFIX = ".phase.consumed."


def _phase_path(out_dir: Path) -> Path:
    return Path(out_dir) / PHASE_FILE


def read_phase(out_dir: Path | str) -> Optional[dict[str, Any]]:
    """Return the parsed `.phase` payload, or None on missing/malformed/race."""
    try:
        p = _phase_path(Path(out_dir))
    except Exception:
        return None
    try:
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        # FS race: file deleted between exists() and read_text().
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def consume_phase(out_dir: Path | str) -> Optional[dict[str, Any]]:
    """Atomically rename `.phase` -> `.phase.consumed.<ts>` and return the payload.

    Idempotent: if the file is already gone (e.g. another worker consumed it),
    returns None without raising.
    """
    try:
        out_dir = Path(out_dir)
    except Exception:
        return None
    src = _phase_path(out_dir)
    if not src.exists():
        return None
    payload = read_phase(out_dir)
    if payload is None:
        # Malformed file -- still try to move it out of the way so we don't
        # spin on a poison pill. Use a distinct suffix for forensics.
        try:
            ts = int(time.time())
            os.replace(src, out_dir / f".phase.malformed.{ts}")
        except OSError:
            pass
        return None
    ts = int(time.time() * 1000)
    dst = out_dir / f"{CONSUMED_PREFIX}{ts}"
    try:
        os.replace(src, dst)
    except (OSError, FileNotFoundError):
        # Race: someone else moved/deleted .phase already.
        return None
    return payload


def _smoke() -> int:
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="phase_signal_smoke_"))
    try:
        # 1. Missing dir / file -> None.
        assert read_phase(tmp / "does_not_exist") is None
        assert read_phase(tmp) is None
        assert consume_phase(tmp) is None

        # 2. Malformed JSON -> None + moved aside.
        (tmp / PHASE_FILE).write_text("not-json", encoding="utf-8")
        assert read_phase(tmp) is None
        assert consume_phase(tmp) is None
        assert any(
            n.name.startswith(".phase.malformed.") for n in tmp.iterdir()
        ), "malformed file should have been moved aside"

        # 3. Round-trip happy path.
        payload = {"phase_id": 1, "phase_name": "intrinsic", "ts": time.time()}
        (tmp / PHASE_FILE).write_text(json.dumps(payload), encoding="utf-8")
        got = read_phase(tmp)
        assert got is not None and got["phase_id"] == 1, got

        consumed = consume_phase(tmp)
        assert consumed is not None and consumed["phase_id"] == 1, consumed
        assert not (tmp / PHASE_FILE).exists(), "consume should have moved file"
        assert any(
            n.name.startswith(CONSUMED_PREFIX) for n in tmp.iterdir()
        ), "consumed file should be present"

        # 4. Idempotency: a second consume sees nothing.
        assert consume_phase(tmp) is None
        assert read_phase(tmp) is None

        print("phase_signal smoke OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(_smoke())
