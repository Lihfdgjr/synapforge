"""monitor_run.py — structured polling monitor for a remote SynapForge run.

Connects to the rental box over SSH every N seconds, scrapes structured
status (trainer PID, latest step line, spike rates, val ppl, GPU util,
CPU/RAM, backup daemon, ckpt count, disk usage), and appends one JSON
object per cycle to `--out`.

Designed to run on the operator's laptop. CPU-only. Uses `paramiko` if
installed; otherwise shells out to local `ssh` (which already has the
rental in `~/.ssh/known_hosts`). The password may be supplied via
`$RENTAL_SSH_PASSWORD` for paramiko fallback, but key auth is preferred.

A single-line summary is also printed to stdout so the operator can watch
the tail in another terminal:

    [19:40] step=1520 ce=5.78 ppl=461 spike=0.0 dead=10/10 gpu=78% disk=28% backup=#3-syncing

Watched alerts (printed to stderr, also recorded as `alerts: [...]` in the
JSONL row):

    CE_RISING        ce went up across the last 3 step samples
    PLIF_STUCK       all PLIF cells dead for >2000 training steps
    BACKUP_FAILED    last backup cycle wrote `status: failed`
    DISK_FULL_RISK   df -h / shows >=90% full

CLI:
    python scripts/monitor_run.py \
        --host 117.74.66.77 --port 41614 --user root \
        --run-dir /workspace/runs/v24h_qwen3 \
        --interval 300 \
        --out monitor.jsonl

Cross-refs: docs/MONITOR_AND_CPU_JOBS.md, docs/RENTAL_OPS.md (P8),
docs/AUTO_EVAL.md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Optional paramiko
# ---------------------------------------------------------------------------

try:
    import paramiko  # type: ignore
    _HAS_PARAMIKO = True
except ImportError:
    paramiko = None  # type: ignore
    _HAS_PARAMIKO = False


# ---------------------------------------------------------------------------
# SSH abstraction
# ---------------------------------------------------------------------------


class SSHRunner:
    """Run commands on a remote host. Tries paramiko first, then `ssh` CLI."""

    def __init__(self, host: str, port: int, user: str, password: Optional[str] = None) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password or os.environ.get("RENTAL_SSH_PASSWORD")
        self._client: Any = None
        if _HAS_PARAMIKO:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                kwargs: Dict[str, Any] = dict(
                    hostname=host, port=port, username=user, timeout=10,
                    banner_timeout=10, auth_timeout=10,
                )
                if self.password:
                    kwargs["password"] = self.password
                    kwargs["look_for_keys"] = False
                    kwargs["allow_agent"] = False
                client.connect(**kwargs)
                self._client = client
            except Exception as exc:  # noqa: BLE001
                print(f"[monitor] paramiko connect failed ({exc}); falling back to ssh CLI",
                      file=sys.stderr)
                self._client = None

    def run(self, cmd: str, timeout: int = 15) -> str:
        """Run `cmd` on remote and return stdout text. Returns "" on failure."""
        if self._client is not None:
            try:
                _stdin, stdout, _stderr = self._client.exec_command(cmd, timeout=timeout)
                data = stdout.read().decode("utf-8", errors="replace")
                return data
            except Exception:  # noqa: BLE001
                return ""
        # ssh CLI fallback
        argv = [
            "ssh",
            "-p", str(self.port),
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={min(timeout, 15)}",
            "-o", "StrictHostKeyChecking=no",
            f"{self.user}@{self.host}",
            cmd,
        ]
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 5)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""
        return out.stdout or ""

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches:  step  1520 loss=5.7806 ce=5.781 kd=0.000 z=0.001 lr=0.00006 step_ms=320.4 tok/s=200 mem=4.7GB
_STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+loss=(?P<loss>[\d.]+)\s+ce=(?P<ce>[\d.]+)\s+"
    r"kd=(?P<kd>[\d.]+)\s+z=(?P<z>[\d.eE+-]+)\s+lr=(?P<lr>[\d.eE+-]+)"
    r"(?:\s+step_ms=(?P<step_ms>[\d.]+))?"
    r"(?:\s+tok/s=(?P<toks>[\d.]+))?"
    r"(?:\s+mem=(?P<mem>[\d.]+)GB)?"
)

# Matches:  spike: mean=0.000 range=[0.000, 0.000] dead=10/10 sat=0/10
_SPIKE_RE = re.compile(
    r"spike:\s+mean=(?P<mean>[\d.]+)\s+range=\[(?P<lo>[\d.]+),\s*(?P<hi>[\d.]+)\]\s+"
    r"dead=(?P<dead>\d+)/(?P<denom>\d+)\s+sat=(?P<sat>\d+)/\d+"
)

# Matches:  VAL step 1500: val_ppl_ttt=355.21 val_ppl_holdout=461.34 (honest)
_VAL_HOLDOUT_RE = re.compile(r"val_ppl_holdout=(?P<ppl>[\d.]+)")
# Legacy fallback:  ppl=461.34
_VAL_LEGACY_RE = re.compile(r"\bppl=(?P<ppl>[\d.]+)")

# Matches a backup.log line like:  [2026-05-01T19:30:01] cycle #3 status=syncing
_BACKUP_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+cycle\s+#?(?P<cycle>\d+)\s+status=(?P<status>\S+)"
)


def parse_step_lines(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        m = _STEP_RE.search(line)
        if m:
            d = m.groupdict()
            rows.append({
                "step": int(d["step"]),
                "ce": float(d["ce"]),
                "kd": float(d["kd"]),
                "z": float(d["z"]),
                "lr": float(d["lr"]),
                "tok_s": float(d["toks"]) if d.get("toks") else None,
                "mem_gb": float(d["mem"]) if d.get("mem") else None,
            })
    return rows


def parse_spike_lines(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        m = _SPIKE_RE.search(line)
        if m:
            d = m.groupdict()
            rows.append({
                "mean": float(d["mean"]),
                "range_lo": float(d["lo"]),
                "range_hi": float(d["hi"]),
                "dead": int(d["dead"]),
                "denom": int(d["denom"]),
                "sat": int(d["sat"]),
            })
    return rows


def parse_val_ppl(text: str) -> Optional[float]:
    # Prefer P3 honest holdout if present.
    m = None
    for line in reversed(text.splitlines()):
        m = _VAL_HOLDOUT_RE.search(line)
        if m:
            return float(m.group("ppl"))
    # Fall back to legacy ppl=
    for line in reversed(text.splitlines()):
        m = _VAL_LEGACY_RE.search(line)
        if m:
            return float(m.group("ppl"))
    return None


def parse_gpu_smi(text: str) -> Dict[str, Any]:
    # nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    # e.g. "78 %, 77345 MiB, 81920 MiB"
    line = text.strip().splitlines()[0] if text.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    out: Dict[str, Any] = {"util_pct": None, "mem_used_mib": None, "mem_total_mib": None}
    if len(parts) >= 3:
        try:
            out["util_pct"] = int(parts[0].split()[0])
            out["mem_used_mib"] = int(parts[1].split()[0])
            out["mem_total_mib"] = int(parts[2].split()[0])
        except (ValueError, IndexError):
            pass
    return out


def parse_loadavg(text: str) -> Optional[float]:
    # /proc/loadavg: "1.20 1.10 0.95 ..."
    try:
        return float(text.strip().split()[0])
    except (IndexError, ValueError):
        return None


def parse_meminfo(text: str) -> Dict[str, Any]:
    # /proc/meminfo
    total = avail = None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1])  # KiB
        elif line.startswith("MemAvailable:"):
            avail = int(line.split()[1])
    out: Dict[str, Any] = {"total_gb": None, "used_gb": None, "pct": None}
    if total is not None:
        out["total_gb"] = round(total / 1024 / 1024, 1)
    if total is not None and avail is not None:
        used = total - avail
        out["used_gb"] = round(used / 1024 / 1024, 1)
        out["pct"] = int(100 * used / total)
    return out


def parse_df(text: str) -> Optional[int]:
    # `df -h /` second line, 5th col = "28%"
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None
    parts = lines[1].split()
    for p in parts:
        if p.endswith("%"):
            try:
                return int(p[:-1])
            except ValueError:
                continue
    return None


def parse_backup_tail(text: str) -> Dict[str, Any]:
    """Find latest cycle line in backup.log."""
    for line in reversed(text.splitlines()):
        m = _BACKUP_LINE_RE.search(line)
        if m:
            return {
                "cycle": int(m.group("cycle")),
                "status": m.group("status"),
                "ts": m.group("ts"),
            }
    return {"cycle": None, "status": None, "ts": None}


# ---------------------------------------------------------------------------
# Sample one cycle
# ---------------------------------------------------------------------------


def sample(ssh: SSHRunner, run_dir: str) -> Dict[str, Any]:
    rd = shlex.quote(run_dir)
    train_log = f"{run_dir}/train.log"
    backup_log = f"{run_dir}/backup.log"

    # Trainer PID alive?
    pid_text = ssh.run(
        f"pgrep -af 'train_100m_kd.py.*{run_dir}' || pgrep -af train_100m_kd.py | head -1"
    ).strip()
    pid = None
    if pid_text:
        try:
            pid = int(pid_text.split()[0])
        except (ValueError, IndexError):
            pid = None

    # Last 200 lines of train.log -- enough to find latest 3 step + spike + val.
    log_tail = ssh.run(f"tail -n 400 {shlex.quote(train_log)} 2>/dev/null || true")
    step_rows = parse_step_lines(log_tail)[-3:]
    spike_rows = parse_spike_lines(log_tail)[-3:]
    val_ppl = parse_val_ppl(log_tail)

    # nvidia-smi
    gpu = parse_gpu_smi(ssh.run(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader 2>/dev/null || true"
    ))

    # CPU loadavg + meminfo
    load = parse_loadavg(ssh.run("cat /proc/loadavg 2>/dev/null || true"))
    mem = parse_meminfo(ssh.run("cat /proc/meminfo 2>/dev/null || true"))

    # Backup daemon
    backup_tail = parse_backup_tail(ssh.run(
        f"tail -n 50 {shlex.quote(backup_log)} 2>/dev/null || true"
    ))

    # Ckpt count + latest step
    ckpt_listing = ssh.run(
        f"ls -1 {rd}/step_*.pt 2>/dev/null | sort | tail -n 1; "
        f"ls -1 {rd}/step_*.pt 2>/dev/null | wc -l"
    ).strip().splitlines()
    latest_ckpt = None
    ckpt_step = None
    ckpt_count = 0
    if ckpt_listing:
        if len(ckpt_listing) >= 1 and ckpt_listing[0].strip().endswith(".pt"):
            latest_ckpt = ckpt_listing[0].strip()
            m = re.search(r"step_(\d+)\.pt", latest_ckpt)
            if m:
                ckpt_step = int(m.group(1))
        if len(ckpt_listing) >= 2:
            try:
                ckpt_count = int(ckpt_listing[-1].strip())
            except ValueError:
                ckpt_count = 0

    # Disk
    disk_pct = parse_df(ssh.run("df -h / 2>/dev/null || true"))

    return {
        "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "trainer_pid": pid,
        "trainer_alive": pid is not None,
        "step_lines": step_rows,
        "spike_lines": spike_rows,
        "val_ppl": val_ppl,
        "gpu": gpu,
        "cpu_loadavg_1m": load,
        "ram": mem,
        "backup": backup_tail,
        "ckpt_count": ckpt_count,
        "latest_ckpt": latest_ckpt,
        "latest_ckpt_step": ckpt_step,
        "disk_pct": disk_pct,
    }


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


def detect_alerts(rec: Dict[str, Any], history: List[Dict[str, Any]]) -> List[str]:
    alerts: List[str] = []

    # CE rising over last 3 step samples (use rec's last 3, or accumulated history)
    ce_seq: List[float] = [r["ce"] for r in rec.get("step_lines", []) if r.get("ce") is not None]
    if len(ce_seq) >= 3 and ce_seq[-1] > ce_seq[-2] > ce_seq[-3]:
        alerts.append("CE_RISING")

    # PLIF stuck dead -- all dead in latest spike line, and it has been so for
    # >2000 training steps. We approximate this by scanning back through
    # `history` cycles.
    spike = rec.get("spike_lines", [])
    if spike:
        last = spike[-1]
        if last["denom"] > 0 and last["dead"] == last["denom"]:
            # Find earliest cycle in history where dead==denom; use its step.
            cur_step = rec["step_lines"][-1]["step"] if rec.get("step_lines") else None
            earliest_step = cur_step
            for h in history:
                hs = h.get("spike_lines") or []
                hsteps = h.get("step_lines") or []
                if not hs or not hsteps:
                    continue
                if hs[-1]["denom"] > 0 and hs[-1]["dead"] == hs[-1]["denom"]:
                    earliest_step = min(earliest_step or hsteps[-1]["step"], hsteps[-1]["step"])
                else:
                    break
            if cur_step is not None and earliest_step is not None and (cur_step - earliest_step) > 2000:
                alerts.append("PLIF_STUCK")

    # Backup failed
    bs = (rec.get("backup") or {}).get("status")
    if bs and "fail" in bs.lower():
        alerts.append("BACKUP_FAILED")

    # Disk
    disk = rec.get("disk_pct")
    if disk is not None and disk >= 90:
        alerts.append("DISK_FULL_RISK")

    return alerts


# ---------------------------------------------------------------------------
# Pretty print one-line summary
# ---------------------------------------------------------------------------


def summary_line(rec: Dict[str, Any]) -> str:
    hh = _dt.datetime.now().strftime("%H:%M")
    parts: List[str] = [f"[{hh}]"]
    if rec.get("step_lines"):
        last = rec["step_lines"][-1]
        parts.append(f"step={last['step']}")
        parts.append(f"ce={last['ce']:.2f}")
    if rec.get("val_ppl") is not None:
        parts.append(f"ppl={rec['val_ppl']:.0f}")
    if rec.get("spike_lines"):
        ls = rec["spike_lines"][-1]
        parts.append(f"spike={ls['mean']:.1f}")
        parts.append(f"dead={ls['dead']}/{ls['denom']}")
    if rec.get("gpu", {}).get("util_pct") is not None:
        parts.append(f"gpu={rec['gpu']['util_pct']}%")
    if rec.get("disk_pct") is not None:
        parts.append(f"disk={rec['disk_pct']}%")
    bk = rec.get("backup") or {}
    if bk.get("cycle") is not None:
        parts.append(f"backup=#{bk['cycle']}-{bk.get('status') or '?'}")
    if not rec.get("trainer_alive"):
        parts.append("DEAD")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--user", default="root")
    p.add_argument("--run-dir", required=True, help="absolute remote run dir, e.g. /workspace/runs/v24h_qwen3")
    p.add_argument("--interval", type=int, default=300, help="seconds between samples (default 300)")
    p.add_argument("--out", default="monitor.jsonl", help="JSONL output file")
    p.add_argument("--once", action="store_true", help="sample once and exit")
    p.add_argument("--history-window", type=int, default=12,
                   help="how many recent cycles to retain in memory for alert detection")
    args = p.parse_args(argv)

    ssh = SSHRunner(args.host, args.port, args.user)
    history: List[Dict[str, Any]] = []

    try:
        while True:
            try:
                rec = sample(ssh, args.run_dir)
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }

            alerts = detect_alerts(rec, history) if "error" not in rec else []
            rec["alerts"] = alerts

            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(summary_line(rec), flush=True)
            for a in alerts:
                print(f"  WARN {a}", file=sys.stderr, flush=True)

            history.append(rec)
            history = history[-args.history_window:]

            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        ssh.close()


if __name__ == "__main__":
    sys.exit(main())
