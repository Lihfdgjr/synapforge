"""Full-volume triple-path backup daemon — never lose another training run.

Watches a runs directory and EVERY `interval` seconds rsyncs the WHOLE
directory (all ckpts, skill_log, train.log, config snapshots, anything
the trainer dumps) to up to 3 off-rental destinations in parallel:

  1. mohuanfang.com:/home/liu/synapforge_backup/<run>/        (rsync, primary)
  2. GitHub Release `Lihfdgjr/synapforge:auto-<run>-<date>`   (gh, top-K best ckpts only — release size cap)
  3. HuggingFace dataset `Lihfdgjr/synapforge-ckpts`          (huggingface_hub, all ckpts < 5GB)

The 2026-04-30 v4.1 disaster (7h training lost when SSH died) is
structurally impossible after this: every step worth keeping lands on
at least one off-rental store within `interval` seconds.

Usage (standalone):
    python scripts/triple_backup_daemon.py \
        --watch /workspace/runs/synapforge_v42_universal \
        --interval 600

Usage (auto-launched by trainer): trainer's main() should spawn this as
a subprocess.Popen at startup, kill on SIGTERM/sys.exit.

Environment:
    GH_TOKEN              GitHub PAT with `repo` scope (for release uploads)
    HF_TOKEN              HuggingFace token with write
    MOHUANFANG_HOST       default mohuanfang.com
    MOHUANFANG_USER       default liu
    MOHUANFANG_BASE       default /home/liu/synapforge_backup

Failure semantics: at least 1 of 3 success per cycle = OK. Total failure
prints loud WARN and re-tries next cycle. Never crashes — daemon keeps
running across transient network/auth errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[backup {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def push_mohuanfang_full(watch_dir: Path, run_name: str) -> bool:
    """rsync ENTIRE watch_dir to mohuanfang. Idempotent + incremental."""
    host = os.environ.get("MOHUANFANG_HOST", "mohuanfang.com")
    user = os.environ.get("MOHUANFANG_USER", "liu")
    base = os.environ.get("MOHUANFANG_BASE", "/home/liu/synapforge_backup")
    dest = f"{user}@{host}:{base}/{run_name}/"
    try:
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}", f"mkdir -p {base}/{run_name}"],
            check=True, capture_output=True, timeout=30,
        )
    except Exception as e:
        _log(f"  mohuanfang mkdir fail: {e}")
        return False
    cmd = [
        "rsync", "-a", "--partial", "--inplace", "--timeout=300",
        "--info=stats0",
        "-e", "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30",
        f"{watch_dir}/", dest,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
        size = sum(p.stat().st_size for p in watch_dir.rglob("*") if p.is_file())
        _log(f"  mohuanfang OK ({size/1e9:.2f}GB synced)")
        return True
    except subprocess.TimeoutExpired:
        _log("  mohuanfang TIMEOUT (3600s) -- partial may be on disk")
        return False
    except subprocess.CalledProcessError as e:
        _log(f"  mohuanfang FAIL: {e.stderr.decode()[:200]}")
        return False


def push_github_top_best(watch_dir: Path, run_name: str, max_assets: int = 3) -> bool:
    """Push top-K best_*.pt to a single GitHub Release (size limits, free public)."""
    if not os.environ.get("GH_TOKEN"):
        return False
    bests = sorted(watch_dir.glob("best*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_assets]
    if not bests:
        return False
    tag = f"auto-{run_name}-{time.strftime('%Y%m%d')}"
    title = f"auto-backup {run_name} {time.strftime('%Y-%m-%d')}"
    notes = "\n".join(
        f"- {p.name} sha256[:16]={_sha256_short(p)} size={p.stat().st_size}"
        for p in bests
    )
    create = subprocess.run(
        ["gh", "release", "create", tag, *[str(p) for p in bests],
         "--title", title, "--notes", notes,
         "--repo", "Lihfdgjr/synapforge"],
        capture_output=True, timeout=1200,
    )
    if create.returncode == 0:
        _log(f"  github OK new release {tag} ({len(bests)} assets)")
        return True
    upload = subprocess.run(
        ["gh", "release", "upload", tag, *[str(p) for p in bests],
         "--clobber", "--repo", "Lihfdgjr/synapforge"],
        capture_output=True, timeout=1200,
    )
    if upload.returncode == 0:
        _log(f"  github OK upload to {tag}")
        return True
    _log(f"  github FAIL: {(create.stderr or upload.stderr).decode()[:200]}")
    return False


def push_huggingface_full(watch_dir: Path, run_name: str) -> bool:
    """Upload entire watch_dir as a folder to HF dataset repo."""
    if not os.environ.get("HF_TOKEN"):
        return False
    try:
        from huggingface_hub import HfApi
    except ImportError:
        _log("  hf skip: huggingface_hub not installed")
        return False
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = "Lihfdgjr/synapforge-ckpts"
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(watch_dir),
            path_in_repo=run_name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"auto: {run_name} @ {time.strftime('%Y-%m-%d %H:%M')}",
            ignore_patterns=["*.tmp", "*.pyc", "__pycache__/*", ".rsync-partial/*"],
        )
        _log("  hf OK (full folder)")
        return True
    except Exception as e:
        _log(f"  hf FAIL: {e}")
        return False


def cycle_once(watch_dir: Path, run_name: str) -> dict[str, bool]:
    return {
        "mohuanfang": push_mohuanfang_full(watch_dir, run_name),
        "github":     push_github_top_best(watch_dir, run_name),
        "hf":         push_huggingface_full(watch_dir, run_name),
    }


def watch_loop(watch_dir: Path, interval: int, run_name: str | None):
    if run_name is None:
        run_name = watch_dir.name
    _log(f"FULL-VOLUME backup daemon: watch={watch_dir}, every {interval}s, run={run_name}")
    _log("targets: mohuanfang (rsync entire dir) + github (top-3 best.pt) + hf (full folder)")

    n = 0
    while True:
        n += 1
        try:
            files = list(watch_dir.rglob("*"))
            files = [p for p in files if p.is_file()]
            total_mb = sum(p.stat().st_size for p in files) / 1e6
            _log(f"cycle #{n}: {len(files)} files, {total_mb:.1f}MB")
            results = cycle_once(watch_dir, run_name)
            ok = sum(results.values())
            tag = f"{ok}/3 paths"
            ok_paths = ",".join(k for k, v in results.items() if v) or "NONE"
            (watch_dir / ".backup_status.json").write_text(json.dumps({
                "cycle": n, "ts": time.time(), "ok": ok, "paths": results,
            }))
            if ok >= 1:
                _log(f"  cycle #{n} success: {ok_paths}")
            else:
                _log(f"  cycle #{n} ALL 3 FAILED -- will retry in {interval}s")
        except Exception as e:
            _log(f"cycle exception: {e}")
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="dir to watch (will be backed up in full)")
    ap.add_argument("--interval", type=int, default=600,
                    help="seconds between full-volume cycles (default 600 = 10 min)")
    ap.add_argument("--run-name", default=None, help="override run-name in dest path")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = ap.parse_args()
    watch_dir = Path(args.watch)
    watch_dir.mkdir(parents=True, exist_ok=True)

    def _term(sig, frame):
        _log(f"got signal {sig}, exiting cleanly")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    if args.once:
        results = cycle_once(watch_dir, args.run_name or watch_dir.name)
        _log(f"once-mode result: {results}")
        sys.exit(0 if any(results.values()) else 1)
    watch_loop(watch_dir, args.interval, args.run_name)


if __name__ == "__main__":
    main()
