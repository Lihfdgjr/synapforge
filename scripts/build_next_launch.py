"""build_next_launch.py -- generate scripts/launch_qwen3<letter>.sh for next phase.

Companion to phase_auto_relauncher.sh and phase_manager.py. The relauncher
already knows how to kill+respawn an in-flight trainer with new flags
(via /proc/<pid>/cmdline rewrite). This script handles the *cold start*
case: take the latest dead-PLIF-revival template (launch_qwen3m.sh) and
emit a phase-N variant that the user (or autopilot on threshold cross)
can invoke directly.

Phase flag table mirrors scripts/phase_manager.py PHASES + flags_for_phase
in scripts/phase_auto_relauncher.sh. Single source of truth: this file
imports nothing dynamic; PHASE_FLAGS below stays in lock-step with those
two manually.

Usage:
    # Generate phase-1 variant from current template (default base = 3m).
    python3 scripts/build_next_launch.py --phase 1 --letter n
    # -> writes scripts/launch_qwen3n.sh with phase-1 flags appended.

    # Use a different base (e.g. when later runs add more flags).
    python3 scripts/build_next_launch.py --phase 1 --letter n \
        --base scripts/launch_qwen3m.sh

    # Dry run -- just print what would be written.
    python3 scripts/build_next_launch.py --phase 1 --letter n --dry-run

Phase-flag policy (kept identical to phase_auto_relauncher.sh::flags_for_phase):
    0  ""  (no extra flags; phase 0 is base LM-only KD)
    1  --self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05 --phase-aware
    2  --modal-list image,audio,time_series --phase-aware
    3  --sft-data /workspace/data/alpaca_zh/alpaca_zh.json --response-only-loss --lr 1e-4 --phase-aware
    4  --rl-grpo --rl-verifier sympy --rl-rollouts 8 --phase-aware

Note: --phase-aware is already in launch_qwen3m.sh; we don't double-add it.

Smoke: python3 -m py_compile scripts/build_next_launch.py
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from pathlib import Path

# Mirror of scripts/phase_auto_relauncher.sh::flags_for_phase. If you
# change this, change BOTH (and scripts/phase_manager.py PHASES).
PHASE_FLAGS: dict[int, list[str]] = {
    0: [],
    1: [
        "--self-learn-ttt",
        "--self-learn-k", "8",
        "--curiosity-weight", "0.05",
    ],
    2: [
        "--modal-list", "image,audio,time_series",
    ],
    3: [
        "--sft-data", "/workspace/data/alpaca_zh/alpaca_zh.json",
        "--response-only-loss",
        "--lr", "1e-4",
    ],
    4: [
        "--rl-grpo",
        "--rl-verifier", "sympy",
        "--rl-rollouts", "8",
    ],
}

# Marker pattern: the trainer invocation block in launch_qwen3m.sh ends
# with "  --phase-aware \\" then "  > '${LOG_FILE}' 2>&1" -- we splice
# the phase flags between those two lines. Keeps existing flags intact.
PHASE_AWARE_RE = re.compile(r"^(\s*)--phase-aware\s*\\\s*$", re.MULTILINE)
LOG_REDIRECT_RE = re.compile(r"^\s*>\s*'\$\{LOG_FILE\}'\s*2>&1", re.MULTILINE)
LOG_FILE_LINE_RE = re.compile(
    r'^LOG_FILE="\$\{LOG_FILE:-\$\{RUN_DIR\}/train_run3([a-z]+)\.log\}"',
    re.MULTILINE,
)
COMMENT_HDR_RE = re.compile(
    r"^# launch_qwen3([a-z]+)\.sh -- ", re.MULTILINE,
)


def render_flag_lines(flags: list[str], indent: str) -> str:
    """Convert flags list into bash-multi-line-arg block.

    PHASE_FLAGS are flat: ['--name', 'value', '--bool-flag', ...]. We
    re-pair them: --name value -> "--name value \\" on one line,
    standalone --bool-flag -> "--bool-flag \\".
    """
    if not flags:
        return ""
    lines: list[str] = []
    i = 0
    while i < len(flags):
        tok = flags[i]
        if not tok.startswith("--"):
            # Should not happen given PHASE_FLAGS shape, but be safe.
            lines.append(f"{indent}{tok} \\")
            i += 1
            continue
        # Look ahead: is the next token a value or another flag?
        if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
            lines.append(f"{indent}{tok} {flags[i + 1]} \\")
            i += 2
        else:
            lines.append(f"{indent}{tok} \\")
            i += 1
    return "\n".join(lines)


def patch_template(template: str, letter: str, phase: int) -> str:
    """Return the new script body with phase flags spliced + log file
    renamed to train_run3<letter>.log + header comment letter swapped."""
    flags = PHASE_FLAGS.get(phase, [])

    # 1. Update LOG_FILE default so train_run3m.log -> train_run3<letter>.log.
    new_template = LOG_FILE_LINE_RE.sub(
        f'LOG_FILE="${{LOG_FILE:-${{RUN_DIR}}/train_run3{letter}.log}}"',
        template,
        count=1,
    )

    # 2. Update header comment so it identifies the new letter.
    new_template = COMMENT_HDR_RE.sub(
        f"# launch_qwen3{letter}.sh -- ",
        new_template,
        count=1,
    )

    # 3. Splice phase flags before the log redirect line. We anchor on
    # the trailing `--phase-aware \\` line and insert AFTER it.
    if not flags:
        return new_template
    m = PHASE_AWARE_RE.search(new_template)
    if not m:
        # Template doesn't have --phase-aware; bail by appending before log redirect.
        m_log = LOG_REDIRECT_RE.search(new_template)
        if not m_log:
            raise RuntimeError(
                "template missing both '--phase-aware \\\\' and "
                "'> ${LOG_FILE} 2>&1' anchors; refusing to patch")
        indent = "  "
        block = render_flag_lines(flags, indent) + "\n"
        return new_template[:m_log.start()] + block + new_template[m_log.start():]
    indent = m.group(1) or "  "
    block = render_flag_lines(flags, indent) + "\n"
    insert_at = m.end() + 1  # after the newline that follows --phase-aware
    return new_template[:insert_at] + block + new_template[insert_at:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--phase", type=int, required=True,
                    choices=sorted(PHASE_FLAGS.keys()),
                    help="phase id (0..4) to layer onto the base template")
    ap.add_argument("--letter", required=True,
                    help="single letter for output script name "
                         "(e.g. 'n' -> scripts/launch_qwen3n.sh)")
    ap.add_argument("--base", default="scripts/launch_qwen3m.sh",
                    help="path to base template (default: launch_qwen3m.sh)")
    ap.add_argument("--out-dir", default="scripts",
                    help="directory for the generated script")
    ap.add_argument("--dry-run", action="store_true",
                    help="print rendered body to stdout, do not write")
    ap.add_argument("--repo-root", default=None,
                    help="repo root override (default: cwd)")
    args = ap.parse_args()

    if not re.fullmatch(r"[a-z]", args.letter):
        print(f"[build_next_launch] --letter must be a single a-z char, "
              f"got '{args.letter}'", file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root or os.getcwd()).resolve()
    base_path = (repo_root / args.base).resolve()
    if not base_path.exists():
        print(f"[build_next_launch] base template not found: {base_path}",
              file=sys.stderr)
        return 1

    template = base_path.read_text(encoding="utf-8")
    body = patch_template(template, args.letter, args.phase)

    out_path = (repo_root / args.out_dir /
                f"launch_qwen3{args.letter}.sh").resolve()

    if args.dry_run:
        sys.stdout.write(body)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    # chmod +x.
    st = out_path.stat()
    out_path.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[build_next_launch] wrote {out_path} (phase={args.phase}, "
          f"flags={PHASE_FLAGS.get(args.phase)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
