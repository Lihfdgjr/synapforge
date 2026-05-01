"""rental_health.py — local-machine query tool for the mohuanfang watchdog.

Reads the watchdog's heartbeat/state files from mohuanfang.com over SSH
and prints a colored health summary. Also exposes a manual emergency
recovery path that rsyncs all best*.pt out of the rental into a local
folder.

Designed for the operator's laptop (Windows or Linux). The watchdog
itself runs on mohuanfang and is documented in docs/RENTAL_RECOVERY.md.

Subcommands:

    status                       one-shot summary, exit 0 on healthy
    --watch                      poll every minute, colored OK/WARN/DEAD
    emergency-recover --to PATH  manual rsync of best*.pt to PATH

Config priority: CLI flag > env var > built-in default.

    --mohuanfang-host  $MOHUANFANG_HOST       default mohuanfang.com
    --mohuanfang-user  $MOHUANFANG_USER       default liu
    --backup-dir       $WATCHDOG_BACKUP_DIR   default /home/liu/synapforge_backup
    --rental-host      $WATCHDOG_RENTAL_HOST  default 117.74.66.77
    --rental-port      $WATCHDOG_RENTAL_PORT  default 41614
    --rental-user      $WATCHDOG_RENTAL_USER  default root
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Optional


# -- color helpers ----------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, msg: str) -> str:
    if not _USE_COLOR:
        return msg
    return f"\033[{code}m{msg}\033[0m"


def green(m: str) -> str:
    return _c("32", m)


def yellow(m: str) -> str:
    return _c("33", m)


def red(m: str) -> str:
    return _c("31;1", m)


def dim(m: str) -> str:
    return _c("90", m)


# -- ssh runners ------------------------------------------------------------


def ssh_cat(args: argparse.Namespace, remote_path: str, timeout: int = 8) -> Optional[str]:
    """Cat a file from mohuanfang via ssh. Returns text or None on failure."""
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "StrictHostKeyChecking=no",
        f"{args.mohuanfang_user}@{args.mohuanfang_host}",
        f"cat {remote_path} 2>/dev/null || true",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 5
        )
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        print(red("ssh not found in PATH"), file=sys.stderr)
        return None
    if out.returncode != 0:
        return None
    return out.stdout if out.stdout else None


def parse_state(text: Optional[str]) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def age_seconds(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    try:
        # Python 3.7+ accepts +HH:MM, but Linux date -Is uses +0000 sometimes.
        ts = _dt.datetime.fromisoformat(iso.strip())
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.timezone.utc)
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    return int((now - ts).total_seconds())


# -- status -----------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    backup = args.backup_dir
    state_text = ssh_cat(args, "/tmp/synapforge_rental_health.json")
    heartbeat_text = ssh_cat(args, f"{backup}/.healthy")
    dead_text = ssh_cat(args, "/tmp/.rental_dead_ts")

    state = parse_state(state_text)
    fails = state.get("consecutive_failures", -1)
    last_check = state.get("last_check")
    last_status = state.get("last_status", "?")
    hb_age = age_seconds(heartbeat_text.strip() if heartbeat_text else None)
    chk_age = age_seconds(last_check)

    print(f"mohuanfang: {args.mohuanfang_user}@{args.mohuanfang_host}")
    print(f"rental:     {args.rental_user}@{args.rental_host}:{args.rental_port}")
    print()

    if state_text is None:
        print(red("DEAD CONTROL PLANE — cannot ssh to mohuanfang"))
        print(dim("  watchdog itself or mohuanfang are unreachable"))
        return 3

    print(f"watchdog state file:   {'present' if state else red('missing or unreadable')}")
    print(f"  consecutive_failures: {fails}")
    print(f"  last_status:          {last_status}")
    print(f"  last_check:           {last_check}  ({chk_age}s ago)" if chk_age is not None else f"  last_check:           {last_check}")
    print(f"  heartbeat age:        {hb_age}s" if hb_age is not None else f"  heartbeat:            {red('never written')}")

    if dead_text:
        print()
        print(red("RENTAL FLAGGED DEAD"))
        for line in dead_text.strip().splitlines():
            print(f"  {line}")
        print(dim("  see docs/RENTAL_RECOVERY.md for runbook"))
        return 2

    if isinstance(fails, int) and fails >= 3:
        print()
        print(yellow(f"WARN: {fails} consecutive failures (escalation in progress)"))
        return 1

    if hb_age is not None and hb_age > 900:  # >15 min stale
        print()
        print(yellow(f"WARN: heartbeat stale ({hb_age}s) — watchdog may not be running"))
        return 1

    print()
    print(green("OK"))
    return 0


# -- watch ------------------------------------------------------------------


def cmd_watch(args: argparse.Namespace) -> int:
    print(dim("polling every 60s; Ctrl+C to exit"))
    try:
        while True:
            print()
            print(dim(f"--- {_dt.datetime.now().isoformat(timespec='seconds')} ---"))
            cmd_status(args)
            time.sleep(60)
    except KeyboardInterrupt:
        print()
        return 0


# -- emergency-recover ------------------------------------------------------


def cmd_emergency_recover(args: argparse.Namespace) -> int:
    dest = args.to
    if not dest:
        print(red("ERROR: --to PATH is required"), file=sys.stderr)
        return 2
    os.makedirs(dest, exist_ok=True)
    print(f"emergency rsync of best*.pt -> {dest}")

    if not shutil.which("rsync") and not shutil.which("scp"):
        print(red("ERROR: neither rsync nor scp found in PATH"), file=sys.stderr)
        return 2

    remote_glob = args.remote_glob
    rental_user = args.rental_user
    rental_host = args.rental_host
    rental_port = args.rental_port

    base_ssh = f"ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 -p {rental_port}"

    # Try rsync first (resumable, preserves perms); fall back to scp.
    if shutil.which("rsync"):
        cmd = [
            "rsync", "-av", "--partial", "--timeout=60",
            "-e", base_ssh,
            f"{rental_user}@{rental_host}:{remote_glob}",
            f"{dest}/",
        ]
        print(f"running: {' '.join(cmd)}")
        try:
            rc = subprocess.run(cmd, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            print(red("rsync timed out after 30 min; partial files in $dest"))
            return 1
        if rc == 0:
            print(green(f"OK — best*.pt evacuated to {dest}"))
            return 0
        print(yellow(f"rsync rc={rc}; falling back to scp"))

    cmd = [
        "scp", "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-P", str(rental_port),
        f"{rental_user}@{rental_host}:{remote_glob}",
        f"{dest}/",
    ]
    print(f"running: {' '.join(cmd)}")
    try:
        rc = subprocess.run(cmd, timeout=1800).returncode
    except subprocess.TimeoutExpired:
        print(red("scp timed out; check $dest for any partial files"))
        return 1
    if rc == 0:
        print(green(f"OK — best*.pt evacuated to {dest}"))
        return 0
    print(red(f"scp failed rc={rc} — rental likely fully unreachable"))
    return 1


# -- argparse ---------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mohuanfang-host",
        default=os.environ.get("MOHUANFANG_HOST", "mohuanfang.com"),
    )
    p.add_argument(
        "--mohuanfang-user",
        default=os.environ.get("MOHUANFANG_USER", "liu"),
    )
    p.add_argument(
        "--backup-dir",
        default=os.environ.get("WATCHDOG_BACKUP_DIR", "/home/liu/synapforge_backup"),
    )
    p.add_argument(
        "--rental-host",
        default=os.environ.get("WATCHDOG_RENTAL_HOST", "117.74.66.77"),
    )
    p.add_argument(
        "--rental-port",
        type=int,
        default=int(os.environ.get("WATCHDOG_RENTAL_PORT", "41614")),
    )
    p.add_argument(
        "--rental-user",
        default=os.environ.get("WATCHDOG_RENTAL_USER", "root"),
    )
    p.add_argument(
        "--remote-glob",
        default=os.environ.get(
            "WATCHDOG_REMOTE_GLOB", "/workspace/runs/v24h_qwen/best*.pt"
        ),
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Query the mohuanfang watchdog for rental health, or trigger manual emergency recovery."
    )
    sub = parser.add_subparsers(dest="cmd")

    p_status = sub.add_parser("status", help="one-shot health summary")
    _add_common(p_status)
    p_status.add_argument("--watch", action="store_true", help="poll every minute")

    p_recover = sub.add_parser(
        "emergency-recover", help="manual rsync of best*.pt to local path"
    )
    _add_common(p_recover)
    p_recover.add_argument("--to", required=True, help="local destination directory")

    args = parser.parse_args(argv)

    if args.cmd == "status":
        if getattr(args, "watch", False):
            return cmd_watch(args)
        return cmd_status(args)
    if args.cmd == "emergency-recover":
        return cmd_emergency_recover(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
