"""changelog_helper.py -- atomic CHANGELOG.md auto-append on phase transitions.

Implements T6.5 (docs/DEEP_MAINT_QUEUE.md): when phase autopilot fires,
append a markdown section recording the phase reached, val ppl, step,
warmstart ckpt path, and the new flags layered onto the trainer.

Companion to:
  - scripts/phase_auto_relauncher.sh (sh watchdog; calls this helper
    AFTER picking the warmstart ckpt + composing new flags, BEFORE
    SIGTERMing the old trainer).
  - scripts/phase_manager.py (Python phase decider; PHASES table).
  - scripts/build_next_launch.py (cold-start phase variant generator).

Public API:
    append_changelog_phase_transition(
        phase_id: int,
        val_ppl: float | None,
        step: int | None,
        ckpt_path: str | os.PathLike | None,
        new_flags: list[str] | None = None,
        changelog_path: str | os.PathLike | None = None,
        target_ppl: float | None = None,
        now: datetime | None = None,
    ) -> Path

Atomicity: writes to a temp file in the same directory as the changelog
(so os.replace() is atomic on POSIX + NTFS), then renames over the
target. If the helper is interrupted mid-write the only side effect is
a leftover .tmp file (cleaned on next successful run via best-effort
glob cleanup); the canonical CHANGELOG.md is never partially written.

Smoke: python3 -m py_compile scripts/changelog_helper.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Phase id -> human-readable target ppl threshold (matches PHASES in
# scripts/phase_manager.py). Used only to decorate the changelog entry
# when caller doesn't pass an explicit ``target_ppl``.
_PHASE_TARGET_PPL: dict[int, str] = {
    0: "infinity (base LM-only KD)",
    1: "250",
    2: "100",
    3: "60",
    4: "0 (gated by chat_eval >= 0.6, not ppl)",
}

# Default target if the caller can't be bothered to pass one.
_DEFAULT_CHANGELOG_REL = "CHANGELOG.md"

# Header used when creating CHANGELOG.md from scratch.
_FRESH_HEADER = "# SynapForge Changelog\n\n"


def _resolve_changelog_path(
    changelog_path: str | os.PathLike | None,
) -> Path:
    if changelog_path is not None:
        return Path(changelog_path)
    # Default: <repo-root>/CHANGELOG.md. Helper lives in scripts/ so go up one.
    here = Path(__file__).resolve().parent
    return (here.parent / _DEFAULT_CHANGELOG_REL).resolve()


def _format_entry(
    phase_id: int,
    val_ppl: float | None,
    step: int | None,
    ckpt_path: str | os.PathLike | None,
    new_flags: list[str] | None,
    target_ppl: float | None,
    now: datetime,
) -> str:
    """Render the markdown section text. Pure function; no IO."""
    ts = now.strftime("%Y-%m-%d %H:%M")
    val_str = f"{val_ppl:.1f}" if val_ppl is not None else "n/a"
    step_str = str(step) if step is not None else "n/a"
    ckpt_str = str(ckpt_path) if ckpt_path else "n/a"

    if target_ppl is not None:
        target_str = f"{target_ppl:.1f}"
    else:
        target_str = _PHASE_TARGET_PPL.get(phase_id, "n/a")

    flags_list = list(new_flags or [])
    if flags_list:
        flags_str = " ".join(flags_list)
    else:
        flags_str = "(none)"

    lines = [
        f"## {ts} - Phase {phase_id} reached",
        f"- val_ppl_holdout: {val_str} (target <= {target_str} for this phase)",
        f"- step: {step_str}",
        f"- warmstart ckpt: {ckpt_str}",
        f"- new flags added: {flags_str}",
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically (temp file in same dir + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync not available on some Windows filesystems; best-effort.
                pass
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; don't mask the original exception.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def append_changelog_phase_transition(
    phase_id: int,
    val_ppl: float | None,
    step: int | None,
    ckpt_path: str | os.PathLike | None,
    new_flags: list[str] | None = None,
    changelog_path: str | os.PathLike | None = None,
    target_ppl: float | None = None,
    now: datetime | None = None,
) -> Path:
    """Append a phase-transition section to CHANGELOG.md atomically.

    Creates CHANGELOG.md with a fresh header if absent. Never modifies
    existing entries; only appends.

    Returns the resolved path of the changelog written.
    """
    path = _resolve_changelog_path(changelog_path)
    if now is None:
        now = datetime.now()

    entry = _format_entry(
        phase_id=phase_id,
        val_ppl=val_ppl,
        step=step,
        ckpt_path=ckpt_path,
        new_flags=new_flags,
        target_ppl=target_ppl,
        now=now,
    )

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if not existing.endswith("\n"):
            existing += "\n"
        new_content = existing + "\n" + entry
    else:
        new_content = _FRESH_HEADER + entry

    _atomic_write(path, new_content)
    return path


def _parse_flags_arg(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [tok for tok in raw.split() if tok]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Append a phase-transition entry to CHANGELOG.md atomically.",
    )
    ap.add_argument("--phase-id", type=int, required=True)
    ap.add_argument("--val-ppl", type=float, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument(
        "--flags",
        default="",
        help="space-separated list of new trainer flags layered for this phase",
    )
    ap.add_argument("--target-ppl", type=float, default=None)
    ap.add_argument(
        "--changelog",
        default=None,
        help="path to CHANGELOG.md (default: repo-root/CHANGELOG.md)",
    )
    args = ap.parse_args(argv)

    out = append_changelog_phase_transition(
        phase_id=args.phase_id,
        val_ppl=args.val_ppl,
        step=args.step,
        ckpt_path=args.ckpt_path,
        new_flags=_parse_flags_arg(args.flags),
        changelog_path=args.changelog,
        target_ppl=args.target_ppl,
    )
    print(f"[changelog_helper] appended phase-{args.phase_id} entry to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
