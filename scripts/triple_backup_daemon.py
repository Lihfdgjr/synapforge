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


def _sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 100MB-cap chunked split fallback for GitHub Releases
#
# GitHub silently truncates assets > 100MB with `validation failed`. For our
# 151M-param ckpts (~600MB w/ optim_state) we shred to 50MB chunks plus a
# manifest, upload as separate assets, and document the recovery path:
#
#     cat best.pt.chunk-000 best.pt.chunk-001 ... > best.pt
#     sha256sum -c best.pt.manifest.json   # checks every chunk + the assembled file
# ---------------------------------------------------------------------------

# GitHub asset cap is 2GB but anything > 100MB starts to fail-silently in
# the gh CLI under intermittent conditions. 50MB chunks stay well under that.
GITHUB_ASSET_CAP_MB = 100


def _chunked_split_fallback(
    src_path: Path, chunk_mb: int = 50, out_dir: Path | None = None
) -> tuple[list[Path], Path] | None:
    """Split src_path into chunk-NNN files of `chunk_mb` MB; emit manifest.json.

    Returns (chunk_paths, manifest_path) on success, None on failure.
    Manifest schema:
        {
            "src_name": "best.pt",
            "src_size": 632819712,
            "src_sha256": "<hex>",
            "chunk_size_bytes": 52428800,
            "n_chunks": 13,
            "chunks": [
                {"name": "best.pt.chunk-000", "size": 52428800, "sha256": "<hex>"},
                ...
            ],
            "recovery": "cat best.pt.chunk-* > best.pt && python -c \"...verify...\""
        }
    """
    src_path = Path(src_path)
    if not src_path.exists() or not src_path.is_file():
        _log(f"  split: source missing {src_path}")
        return None
    out_dir = out_dir or src_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = chunk_mb * 1024 * 1024
    src_size = src_path.stat().st_size
    chunks: list[Path] = []
    chunk_records: list[dict] = []
    h_full = hashlib.sha256()
    try:
        with open(src_path, "rb") as src:
            idx = 0
            while True:
                buf = src.read(chunk_size)
                if not buf:
                    break
                h_full.update(buf)
                cname = f"{src_path.name}.chunk-{idx:03d}"
                cpath = out_dir / cname
                cpath.write_bytes(buf)
                chunks.append(cpath)
                chunk_records.append({
                    "name": cname,
                    "size": len(buf),
                    "sha256": hashlib.sha256(buf).hexdigest(),
                })
                idx += 1
    except OSError as exc:
        _log(f"  split FAIL writing chunks: {exc}")
        # Best-effort cleanup of partial outputs.
        for c in chunks:
            try:
                c.unlink()
            except OSError:
                pass
        return None
    manifest = {
        "src_name": src_path.name,
        "src_size": src_size,
        "src_sha256": h_full.hexdigest(),
        "chunk_size_bytes": chunk_size,
        "n_chunks": len(chunks),
        "chunks": chunk_records,
        "recovery": (
            f"cat {src_path.name}.chunk-* > {src_path.name} && "
            f"python -c \"import hashlib,json,sys;m=json.load(open('"
            f"{src_path.name}.manifest.json'));"
            f"h=hashlib.sha256(open(m['src_name'],'rb').read()).hexdigest();"
            f"sys.exit(0 if h==m['src_sha256'] else 1)\""
        ),
    }
    manifest_path = out_dir / f"{src_path.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _log(f"  split: {src_path.name} -> {len(chunks)} chunks @ {chunk_mb}MB "
         f"(manifest {manifest_path.name})")
    return chunks, manifest_path


def _assemble_chunks(manifest_path: Path, out_path: Path | None = None) -> Path | None:
    """Reverse `_chunked_split_fallback`: read manifest, concat chunks, verify sha256.

    Returns the assembled path on success, None on failure (also leaves the
    bad output in place for inspection).
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        _log(f"  assemble: manifest missing {manifest_path}")
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _log(f"  assemble: manifest parse fail: {exc}")
        return None
    out_path = out_path or manifest_path.parent / manifest["src_name"]
    base = manifest_path.parent
    h = hashlib.sha256()
    try:
        with open(out_path, "wb") as dst:
            for rec in manifest["chunks"]:
                cpath = base / rec["name"]
                if not cpath.exists():
                    _log(f"  assemble FAIL: missing chunk {cpath}")
                    return None
                buf = cpath.read_bytes()
                if hashlib.sha256(buf).hexdigest() != rec["sha256"]:
                    _log(f"  assemble FAIL: sha mismatch on {cpath}")
                    return None
                h.update(buf)
                dst.write(buf)
    except OSError as exc:
        _log(f"  assemble FAIL writing {out_path}: {exc}")
        return None
    if h.hexdigest() != manifest["src_sha256"]:
        _log(f"  assemble FAIL: full-file sha256 mismatch (got {h.hexdigest()[:16]} "
             f"vs expected {manifest['src_sha256'][:16]})")
        return None
    _log(f"  assemble OK -> {out_path} ({manifest['src_size']:,} B verified)")
    return out_path


# Module-level state for "watch dir is empty for too many cycles" detection.
_EMPTY_CYCLES_TRIPWIRE = 5
_empty_cycles_seen = 0


def _warn_if_persistently_empty(watch_dir: Path, file_count: int) -> None:
    """Loud-log if watch_dir has been empty for `_EMPTY_CYCLES_TRIPWIRE` consecutive cycles.

    Catches the post-pivot footgun of pointing the daemon at the wrong path
    after manually editing the systemd unit on a rental box.
    """
    global _empty_cycles_seen
    if file_count == 0:
        _empty_cycles_seen += 1
        if _empty_cycles_seen == _EMPTY_CYCLES_TRIPWIRE:
            _log(
                f"WARNING: backup daemon watching empty dir for "
                f"{_EMPTY_CYCLES_TRIPWIRE} cycles ({watch_dir!s}); "
                f"did you point to the wrong path?"
            )
        elif _empty_cycles_seen > _EMPTY_CYCLES_TRIPWIRE and _empty_cycles_seen % 5 == 0:
            _log(
                f"WARNING: still empty after {_empty_cycles_seen} cycles "
                f"({watch_dir!s}); check --watch arg & systemd unit"
            )
    else:
        if _empty_cycles_seen >= _EMPTY_CYCLES_TRIPWIRE:
            _log(f"  recovered: watch dir now has files after "
                 f"{_empty_cycles_seen} empty cycles")
        _empty_cycles_seen = 0


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
    # rsync -a is already incremental (skip same mtime+size). --inplace
    # patches in place (same dest path each cycle, no temp files), so a
    # stable file isn't re-uploaded.
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
        # e.stderr may be bytes or None depending on capture_output state.
        msg = (e.stderr or b"").decode(errors="replace")[:200]
        _log(f"  mohuanfang FAIL: {msg}")
        return False


_GH_PUSHED_CACHE: dict[str, str] = {}  # path -> sha256[:16]


def _gh_release_upload(
    fresh_paths: list[Path],
    bests: list[Path],
    cache: dict[str, str],
    cache_file: Path,
    tag: str,
    title: str,
    notes: str,
) -> bool:
    """Try create-then-upload to GitHub Releases. Shared between flat + split paths."""
    create = subprocess.run(
        ["gh", "release", "create", tag, *[str(p) for p in fresh_paths],
         "--title", title, "--notes", notes,
         "--repo", "Lihfdgjr/synapforge"],
        capture_output=True, timeout=1200,
    )
    if create.returncode == 0:
        _log(f"  github OK new release {tag} ({len(fresh_paths)} fresh assets, "
             f"{len(bests)-len(fresh_paths)} skipped)")
        cache_file.write_text(json.dumps(cache))
        return True
    upload = subprocess.run(
        ["gh", "release", "upload", tag, *[str(p) for p in fresh_paths],
         "--clobber", "--repo", "Lihfdgjr/synapforge"],
        capture_output=True, timeout=1200,
    )
    if upload.returncode == 0:
        _log(f"  github OK upload {len(fresh_paths)} to {tag} "
             f"({len(bests)-len(fresh_paths)} skipped)")
        cache_file.write_text(json.dumps(cache))
        return True
    # Both stderrs may be None on certain subprocess paths -- coerce to b"".
    err = (create.stderr or upload.stderr or b"").decode(errors="replace")[:200]
    _log(f"  github FAIL: {err}")
    return False


def push_github_top_best(watch_dir: Path, run_name: str, max_assets: int = 3) -> bool:
    """Push top-K best_*.pt to a single GitHub Release (size limits, free public).

    Dedup: skip files whose sha256[:16] hasn't changed since last successful
    push. Avoids re-uploading 600MB ckpts every cycle when they haven't moved.

    Files exceeding GITHUB_ASSET_CAP_MB (100MB) are split via
    `_chunked_split_fallback` and uploaded as `<name>.chunk-NNN` + a manifest
    under a `-split` tag suffix so recovery is unambiguous.
    """
    if not os.environ.get("GH_TOKEN"):
        return False
    bests = sorted(watch_dir.glob("best*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_assets]
    if not bests:
        return False
    # dedup
    cache_file = watch_dir / ".backup_gh_cache.json"
    cache: dict[str, str] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            cache = {}
    fresh = []
    for p in bests:
        s = _sha256_short(p)
        if cache.get(p.name) != s:
            fresh.append(p)
            cache[p.name] = s
    if not fresh:
        _log(f"  github skip: {len(bests)} files unchanged since last push")
        return True  # treat as success — content is already there

    # Split fresh files into "fits flat" vs "needs split" buckets.
    cap_bytes = GITHUB_ASSET_CAP_MB * 1024 * 1024
    flat_paths = [p for p in fresh if p.stat().st_size <= cap_bytes]
    big_paths = [p for p in fresh if p.stat().st_size > cap_bytes]

    flat_ok = True
    if flat_paths:
        tag = f"auto-{run_name}-{time.strftime('%Y%m%d')}"
        title = f"auto-backup {run_name} {time.strftime('%Y-%m-%d')}"
        notes = "\n".join(
            f"- {p.name} sha256[:16]={_sha256_short(p)} size={p.stat().st_size}"
            for p in bests if p.stat().st_size <= cap_bytes
        )
        flat_ok = _gh_release_upload(
            flat_paths, [p for p in bests if p.stat().st_size <= cap_bytes],
            cache, cache_file, tag, title, notes,
        )

    if not big_paths:
        return flat_ok

    # Split-and-upload path for ckpts > 100MB. One -split release per cycle
    # holds chunks for every big file (manifest carries file boundaries).
    split_dir = watch_dir / ".split_staging"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_tag = f"auto-{run_name}-{time.strftime('%Y%m%d')}-split"
    split_title = f"auto-backup {run_name} {time.strftime('%Y-%m-%d')} (split)"
    split_assets: list[Path] = []
    split_notes_lines = [
        "Files >100MB are split to 50MB chunks. Recovery:",
        "    cat <name>.chunk-* > <name>",
        "    python -c \"import hashlib,json,sys;m=json.load(open('<name>.manifest.json'));"
        "h=hashlib.sha256(open(m['src_name'],'rb').read()).hexdigest();"
        "sys.exit(0 if h==m['src_sha256'] else 1)\"",
        "",
    ]
    for p in big_paths:
        result = _chunked_split_fallback(
            p, chunk_mb=50, out_dir=split_dir,
        )
        if result is None:
            _log(f"  github split FAIL for {p.name}; will retry next cycle")
            # On split failure remove the cache entry so we re-try cleanly.
            cache.pop(p.name, None)
            continue
        chunks, manifest = result
        split_assets.extend(chunks)
        split_assets.append(manifest)
        split_notes_lines.append(
            f"- {p.name}: {len(chunks)} chunks, "
            f"sha256={_sha256_full(p)[:16]} size={p.stat().st_size}"
        )

    split_ok = True
    if split_assets:
        split_ok = _gh_release_upload(
            split_assets, big_paths, cache, cache_file,
            split_tag, split_title, "\n".join(split_notes_lines),
        )

    return flat_ok and split_ok


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
    # Defensive: if the operator hand-edited the systemd unit and the path
    # is now bogus, mkdir parents-of-parents so we at least don't crash.
    watch_dir.mkdir(parents=True, exist_ok=True)
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
            _warn_if_persistently_empty(watch_dir, len(files))
            # If the dir is empty we still let the cycle run -- it's harmless
            # and keeps cycle counts/.backup_status.json in sync. The warn
            # above is what flags the misconfiguration.
            results = cycle_once(watch_dir, run_name)
            ok = sum(results.values())
            tag = f"{ok}/3 paths"
            ok_paths = ",".join(k for k, v in results.items() if v) or "NONE"
            (watch_dir / ".backup_status.json").write_text(json.dumps({
                "cycle": n, "ts": time.time(), "ok": ok, "paths": results,
                "n_files": len(files),
                "empty_cycles_seen": _empty_cycles_seen,
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
