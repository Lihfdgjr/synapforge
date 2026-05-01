"""auto_pretrain_to_sft -- the grand orchestrator: pretrain -> SFT -> chat eval -> release.

Stages
------
0  WATCH    Tail pretrain `train.log`. When `VAL step N: ppl=X.YY` shows
            `ppl <= --ppl-target`, advance to stage 1.
1  TRANSITION  Send SIGTERM to the pretrain trainer process. Wait `--save-grace`
            seconds for it to flush its final ckpt. Locate the best ckpt.
2  SFT     Spawn `train_100m_sft.py --warmstart <best.pt> --out <sft_out>` ...
            Wait for it to exit cleanly OR until `--sft-max-hours` elapses.
3  GATE    Run `chat_eval_gate.py --ckpt <sft_best> --out <sft_out>/chat_eval`.
            If `pass_rate >= 0.6` -> stage 4. Else iterate (re-run SFT with
            different lr / steps / data mix; max `--max-sft-iters` times).
4  RELEASE Copy ckpt + chat_eval results + git rev-parse + repro commands
            to `<release-dir>/v0.1.0/`. Write release.json with manifest.

Each stage appends a JSONL record to `<out>/auto_pretrain_to_sft.jsonl` so the
whole pipeline is auditable + replayable. SIGTERM/SIGINT to the orchestrator
saves state to `<out>/auto_pretrain_to_sft.state.json` and exits cleanly.

Honest constraints (matches user request 2026-05-01):
* The pretrain trainer must already be running (we don't launch it; we just
  watch its log). This avoids the fragility of dual-launching across rentals.
* `--ppl-target` defaults to 60.0 — matches phase_manager.py phase 3 boundary.
* `--smoke` mode runs the whole state machine on a fake log + fake ckpt so
  we can verify the orchestrator end-to-end without real training.

CLI:
    python scripts/auto_pretrain_to_sft.py \
        --pretrain-out /workspace/runs/v24h_qwen \
        --pretrain-pidfile /workspace/runs/v24h_qwen/trainer.pid \
        --sft-cmd "python train_100m_sft.py --backend triton_block --batch-size 16 \\
                   --steps 4000 --lr 1e-5 \\
                   --tokenizer-path /workspace/teachers/qwen2.5-0.5b \\
                   --sft-parquet /workspace/data/sft/alpaca_combined.parquet" \
        --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
        --sft-out /workspace/runs/v24h_sft \
        --release-dir ~/.synapforge/release/v0.1.0 \
        --ppl-target 60 \
        --max-sft-iters 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Repo root on sys.path so `import synapforge` works during smoke
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VAL_PPL_RE = re.compile(r"VAL\s+step\s+(\d+).*?ppl[=:]\s*([\d.]+)", re.IGNORECASE)
BEST_CKPT_RE = re.compile(r"best_step_(\d+)\.pt$")
STEP_CKPT_RE = re.compile(r"step_(\d+)\.pt$")


# ---------------------------------------------------------------------------
# JSONL audit log
# ---------------------------------------------------------------------------

class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, **kw: Any) -> None:
        rec = {"ts": time.time(), "stage": stage, **kw}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[orchestrator {time.strftime('%H:%M:%S')}] [{stage}] "
              f"{ {k: v for k, v in kw.items() if k != 'detail'} }")


# ---------------------------------------------------------------------------
# stage 0: watch pretrain log for ppl gate
# ---------------------------------------------------------------------------

def parse_latest_val_ppl(log_path: Path) -> tuple[int | None, float | None]:
    """Return (last_step, last_val_ppl) from train.log, or (None, None)."""
    if not log_path.exists():
        return None, None
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, None
    last_step, last_ppl = None, None
    for m in VAL_PPL_RE.finditer(text):
        last_step = int(m.group(1))
        last_ppl = float(m.group(2))
    return last_step, last_ppl


def watch_pretrain_log(
    log_path: Path,
    ppl_target: float,
    audit: AuditLog,
    poll_s: int = 60,
    deadline_ts: float | None = None,
) -> tuple[int, float]:
    """Block until val_ppl <= ppl_target. Return (step, ppl). Raises TimeoutError."""
    audit.emit("watch_start", log=str(log_path), ppl_target=ppl_target,
               poll_s=poll_s)
    last_seen_ppl = None
    last_seen_step = None
    while True:
        if deadline_ts is not None and time.time() > deadline_ts:
            raise TimeoutError(
                f"watch_pretrain_log: deadline exceeded; last_ppl={last_seen_ppl}"
            )
        step, ppl = parse_latest_val_ppl(log_path)
        if ppl is not None and (last_seen_ppl != ppl or last_seen_step != step):
            audit.emit("watch_progress", step=step, ppl=ppl)
            last_seen_ppl = ppl
            last_seen_step = step
        if ppl is not None and ppl <= ppl_target:
            audit.emit("watch_gate_crossed", step=step, ppl=ppl,
                       target=ppl_target)
            return step or 0, ppl
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# stage 1: signal pretrain to flush + locate best ckpt
# ---------------------------------------------------------------------------

def read_pid(pidfile: Path) -> int | None:
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None


def signal_pretrain_and_wait(
    pidfile: Path,
    audit: AuditLog,
    grace_s: int = 90,
) -> bool:
    """SIGTERM the pretrain process; poll until it exits or grace expires."""
    pid = read_pid(pidfile)
    if pid is None:
        audit.emit("transition_no_pidfile", pidfile=str(pidfile))
        return False
    audit.emit("transition_sigterm", pid=pid, grace_s=grace_s)
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as exc:
        audit.emit("transition_kill_skip", pid=pid, error=repr(exc))
        return True  # already dead
    deadline = time.time() + grace_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)  # alive check
        except ProcessLookupError:
            audit.emit("transition_pretrain_exited", pid=pid)
            return True
        time.sleep(2)
    audit.emit("transition_grace_exhausted", pid=pid)
    return False


def find_best_ckpt(out_dir: Path) -> Path | None:
    """Find the highest-step `best_step_*.pt` (else newest `step_*.pt`)."""
    bests = sorted(
        (p for p in out_dir.glob("best_step_*.pt")),
        key=lambda p: int(BEST_CKPT_RE.search(p.name).group(1))
        if BEST_CKPT_RE.search(p.name) else 0,
    )
    if bests:
        return bests[-1]
    steps = sorted(
        (p for p in out_dir.glob("step_*.pt")),
        key=lambda p: int(STEP_CKPT_RE.search(p.name).group(1))
        if STEP_CKPT_RE.search(p.name) else 0,
    )
    if steps:
        return steps[-1]
    finals = list(out_dir.glob("final.pt"))
    if finals:
        return finals[0]
    return None


# ---------------------------------------------------------------------------
# stage 2: launch SFT
# ---------------------------------------------------------------------------

def launch_sft(
    sft_cmd: str,
    warmstart: Path,
    sft_out: Path,
    iter_idx: int,
    audit: AuditLog,
    extra_args: list[str] | None = None,
    timeout_s: int | None = None,
) -> tuple[int, Path | None]:
    """Run SFT trainer to completion. Returns (rc, sft_log_path)."""
    sft_out.mkdir(parents=True, exist_ok=True)
    log_path = sft_out / f"sft_iter{iter_idx}.log"
    cmd = shlex.split(sft_cmd) + [
        "--warmstart", str(warmstart),
        "--out", str(sft_out),
    ] + (extra_args or [])
    audit.emit("sft_launch", iter=iter_idx, cmd=cmd, log=str(log_path))
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parent.parent,
            )
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGTERM)
                try:
                    rc = proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = -9
                audit.emit("sft_timeout", iter=iter_idx, rc=rc,
                           timeout_s=timeout_s)
        audit.emit("sft_exit", iter=iter_idx, rc=rc, log=str(log_path))
        return rc, log_path
    except FileNotFoundError as exc:
        audit.emit("sft_launch_failed", iter=iter_idx, error=repr(exc))
        return 127, log_path


def iter_strategies(iter_idx: int) -> list[str]:
    """Different SFT hyperparam tweaks per retry. Index 0 = original cmd."""
    if iter_idx == 0:
        return []
    if iter_idx == 1:
        # Iter 1: longer + higher LR
        return ["--steps", "6000", "--lr", "3e-5"]
    if iter_idx == 2:
        # Iter 2: lower LR, very long
        return ["--steps", "8000", "--lr", "5e-6"]
    return ["--steps", "5000", "--lr", "1e-5"]


# ---------------------------------------------------------------------------
# stage 3: chat eval gate
# ---------------------------------------------------------------------------

def run_chat_eval_gate(
    sft_ckpt: Path,
    tokenizer_path: str,
    out_dir: Path,
    threshold: float,
    audit: AuditLog,
    smoke: bool = False,
) -> dict[str, Any] | None:
    """Invoke chat_eval_gate.py as a subprocess; return parsed JSON report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "chat_eval_gate.py"),
        "--out", str(out_dir),
        "--threshold", str(threshold),
    ]
    if smoke:
        cmd.append("--smoke")
    else:
        cmd += ["--ckpt", str(sft_ckpt), "--tokenizer-path", tokenizer_path]
    audit.emit("gate_launch", cmd=cmd)
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    except subprocess.TimeoutExpired:
        audit.emit("gate_timeout")
        return None
    rc = proc.returncode
    audit.emit("gate_exit", rc=rc,
               stdout_tail=proc.stdout.decode(errors="replace")[-500:])
    report_path = out_dir / "chat_eval_gate.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# stage 4: release dump
# ---------------------------------------------------------------------------

def git_rev_parse(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.decode().strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def release_dump(
    sft_ckpt: Path,
    chat_eval_report: dict[str, Any],
    release_dir: Path,
    repo_root: Path,
    pretrain_cmd: str,
    sft_cmd: str,
    audit: AuditLog,
) -> Path:
    """Copy ckpt + chat eval JSON + repro instructions to release_dir."""
    release_dir.mkdir(parents=True, exist_ok=True)
    # Copy ckpt
    dst_ckpt = release_dir / sft_ckpt.name
    if sft_ckpt.exists():
        shutil.copy2(sft_ckpt, dst_ckpt)
    # Copy chat eval
    chat_eval_dst = release_dir / "chat_eval_gate.json"
    chat_eval_dst.write_text(
        json.dumps(chat_eval_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Repro
    repro = {
        "git_head": git_rev_parse(repo_root),
        "pretrain_cmd": pretrain_cmd,
        "sft_cmd": sft_cmd,
        "ckpt_filename": sft_ckpt.name,
        "ckpt_size_bytes": sft_ckpt.stat().st_size if sft_ckpt.exists() else 0,
        "chat_eval_pass_rate": chat_eval_report.get("pass_rate"),
        "chat_eval_passed": chat_eval_report.get("passed"),
        "ts": time.time(),
        "release_version": "v0.1.0",
        "honest_about_gate": chat_eval_report.get("honest_about_heuristic", ""),
    }
    (release_dir / "release.json").write_text(
        json.dumps(repro, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    # Run script
    run_demo = release_dir / "run_demo.sh"
    run_demo.write_text(_RUN_DEMO_SH.format(ckpt=dst_ckpt.name), encoding="utf-8")
    try:
        os.chmod(run_demo, 0o755)
    except OSError:
        pass
    audit.emit("release_dump_done", path=str(release_dir),
               ckpt=str(dst_ckpt), pass_rate=chat_eval_report.get("pass_rate"))
    return release_dir


_RUN_DEMO_SH = """#!/usr/bin/env bash
# Investor demo entrypoint — generated by auto_pretrain_to_sft.py.
# Loads the shipped ckpt and starts an interactive chat REPL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "[demo] loading {ckpt}"
python -m pip install -r "$HERE"/requirements.txt 2>/dev/null || true
python "$HERE"/scripts/chat_repl.py \\
    --ckpt "$HERE/{ckpt}" \\
    --tokenizer-path "${{TOKENIZER_PATH:-Qwen/Qwen2.5-0.5B}}" \\
    --temperature 0.7 \\
    --max-new 80
"""


# ---------------------------------------------------------------------------
# state save/restore
# ---------------------------------------------------------------------------

def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, state_path)


def load_state(state_path: Path) -> dict | None:
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(out / "auto_pretrain_to_sft.jsonl")
    state_path = out / "auto_pretrain_to_sft.state.json"

    state: dict[str, Any] = load_state(state_path) or {"stage": "watch"}
    audit.emit("pipeline_start", state=state, args=vars(args))

    def _sigterm(sig, frame):
        audit.emit("sigterm_received", sig=int(sig))
        save_state(state_path, state)
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigterm)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sigterm)

    # ---- stage 0: WATCH ----
    pretrain_log = Path(args.pretrain_out) / "train.log"
    if state.get("stage") == "watch":
        if args.smoke:
            audit.emit("smoke_skipping_watch")
            state.update({"watch_step": 999, "watch_ppl": 55.0,
                          "stage": "transition"})
        else:
            try:
                step, ppl = watch_pretrain_log(
                    pretrain_log, args.ppl_target, audit,
                    poll_s=args.watch_poll_s,
                )
            except TimeoutError as exc:
                audit.emit("watch_timeout", error=str(exc))
                return 11
            state.update({"watch_step": step, "watch_ppl": ppl,
                          "stage": "transition"})
        save_state(state_path, state)

    # ---- stage 1: TRANSITION ----
    if state.get("stage") == "transition":
        pidfile = Path(args.pretrain_pidfile) if args.pretrain_pidfile else None
        if not args.smoke and pidfile is not None:
            ok = signal_pretrain_and_wait(pidfile, audit,
                                           grace_s=args.save_grace)
            audit.emit("transition_complete", ok=ok)
        else:
            audit.emit("transition_skipped",
                       reason="smoke" if args.smoke else "no-pidfile")
        # Find best ckpt
        if args.smoke:
            fake = Path(args.pretrain_out) / "best_step_smoke.pt"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.write_bytes(b"\x00" * 1024)
            best = fake
        else:
            best = find_best_ckpt(Path(args.pretrain_out))
        if best is None or not best.exists():
            audit.emit("transition_no_ckpt", out=args.pretrain_out)
            return 12
        state["pretrain_best_ckpt"] = str(best)
        state["stage"] = "sft"
        save_state(state_path, state)

    # ---- stage 2-3: SFT loop with chat-eval gate ----
    sft_iter = state.get("sft_iter", 0)
    sft_out = Path(args.sft_out)
    while True:
        if sft_iter >= args.max_sft_iters:
            audit.emit("sft_max_iters_reached", iter=sft_iter)
            return 13
        state.update({"stage": "sft", "sft_iter": sft_iter})
        save_state(state_path, state)

        # Stage 2: SFT (skip in smoke)
        sft_ckpt: Path | None = None
        if args.smoke:
            audit.emit("smoke_skipping_sft", iter=sft_iter)
            fake = sft_out / f"best_step_smoke_iter{sft_iter}.pt"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.write_bytes(b"\x00" * 1024)
            sft_ckpt = fake
        else:
            extra = iter_strategies(sft_iter)
            iter_sft_out = sft_out / f"iter{sft_iter}"
            iter_sft_out.mkdir(parents=True, exist_ok=True)
            rc, _logp = launch_sft(
                args.sft_cmd, Path(state["pretrain_best_ckpt"]),
                iter_sft_out, sft_iter, audit,
                extra_args=extra,
                timeout_s=args.sft_timeout_s,
            )
            if rc != 0:
                audit.emit("sft_iter_failed", iter=sft_iter, rc=rc)
                # Continue to next iteration (training might have produced
                # a partial ckpt that's still worth gating).
            sft_ckpt = find_best_ckpt(iter_sft_out)

        if sft_ckpt is None or not sft_ckpt.exists():
            audit.emit("sft_iter_no_ckpt", iter=sft_iter)
            sft_iter += 1
            continue

        state["sft_best_ckpt"] = str(sft_ckpt)
        state["stage"] = "gate"
        save_state(state_path, state)

        # Stage 3: gate
        gate_out = sft_ckpt.parent / "chat_eval"
        report = run_chat_eval_gate(
            sft_ckpt, args.tokenizer_path, gate_out,
            threshold=args.gate_threshold, audit=audit, smoke=args.smoke,
        )
        if report is None:
            audit.emit("gate_no_report", iter=sft_iter)
            sft_iter += 1
            continue
        passed = bool(report.get("passed"))
        pass_rate = report.get("pass_rate")
        audit.emit("gate_result", iter=sft_iter, passed=passed,
                   pass_rate=pass_rate)
        if passed:
            state["stage"] = "release"
            state["chat_eval_report_path"] = str(gate_out / "chat_eval_gate.json")
            save_state(state_path, state)
            break
        # Failed gate -> next SFT iter
        sft_iter += 1

    # ---- stage 4: RELEASE ----
    if state.get("stage") == "release":
        sft_ckpt = Path(state["sft_best_ckpt"])
        report = json.loads(
            Path(state["chat_eval_report_path"]).read_text(encoding="utf-8")
        )
        release_dir = Path(args.release_dir).expanduser()
        repo_root = Path(__file__).resolve().parent.parent
        release_dump(
            sft_ckpt, report, release_dir, repo_root,
            pretrain_cmd=args.pretrain_cmd_record,
            sft_cmd=args.sft_cmd,
            audit=audit,
        )
        state["stage"] = "done"
        state["release_dir"] = str(release_dir)
        save_state(state_path, state)

    audit.emit("pipeline_done", final_state=state)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/auto_p2sft",
                    help="orchestrator audit dir (jsonl + state)")
    ap.add_argument("--pretrain-out", required=True,
                    help="dir of running pretrain trainer (has train.log)")
    ap.add_argument("--pretrain-pidfile", default="",
                    help="file with pretrain trainer PID (for SIGTERM)")
    ap.add_argument("--pretrain-cmd-record", default="",
                    help="pretrain cmd string for the release manifest")
    ap.add_argument("--sft-cmd", default="",
                    help='full SFT base cmd; --warmstart and --out injected. '
                         'e.g. "python train_100m_sft.py --backend triton_block ..."')
    ap.add_argument("--sft-out", default="runs/v24h_sft",
                    help="dir for SFT runs (each iter -> iter<N>/)")
    ap.add_argument("--tokenizer-path", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--release-dir", default="~/.synapforge/release/v0.1.0")
    ap.add_argument("--ppl-target", type=float, default=60.0)
    ap.add_argument("--watch-poll-s", type=int, default=60)
    ap.add_argument("--save-grace", type=int, default=90,
                    help="seconds to wait for pretrain SIGTERM->save->exit")
    ap.add_argument("--max-sft-iters", type=int, default=3)
    ap.add_argument("--sft-timeout-s", type=int, default=None,
                    help="hard timeout for one SFT iter (default no timeout)")
    ap.add_argument("--gate-threshold", type=float, default=0.6)
    ap.add_argument("--smoke", action="store_true",
                    help="end-to-end self-test with fake log/ckpt; no real proc")
    args = ap.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
