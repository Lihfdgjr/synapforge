"""release_packager -- bundle a release-dir into a standalone investor deliverable.

INPUT (from auto_pretrain_to_sft.py stage 4):
    <release-dir>/
        best_step_*.pt          ckpt (may be > 100 MB; we split-chunk if so)
        chat_eval_gate.json     50-prompt heuristic gate report
        release.json            git head + repro cmds
        run_demo.sh             entrypoint

OUTPUT:
    <out>/v0.1.0.tar.gz         the tarball
    <out>/v0.1.0.sha256         single-line manifest
    <out>/v0.1.0.parts/         (only if ckpt > 100 MB) split chunks for GH Release

The script is fail-safe: if the source dir is missing some expected file we log
which one and continue building the tarball with what's there.

CLI:
    python scripts/release_packager.py \
        --from ~/.synapforge/release/v0.1.0 \
        --out ~/.synapforge/release \
        [--upload-gh / --no-upload-gh]
        [--gh-tag v0.1.0] [--gh-repo Lihfdgjr/synapforge]
        [--smoke]   # builds against a fake src so we can verify packaging
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

CHUNK_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB
SPLIT_CHUNK_BYTES = 90 * 1024 * 1024  # 90 MB chunks (under GH 100 MB cap)


def _sha256(p: Path, hex_len: int = 64) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:hex_len]


def _split_ckpt(ckpt: Path, parts_dir: Path) -> list[Path]:
    """Split a ckpt > CHUNK_THRESHOLD into ~90 MB chunks."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    with open(ckpt, "rb") as src:
        idx = 0
        while True:
            chunk = src.read(SPLIT_CHUNK_BYTES)
            if not chunk:
                break
            part = parts_dir / f"{ckpt.name}.part{idx:03d}"
            part.write_bytes(chunk)
            parts.append(part)
            idx += 1
    # Re-assemble script
    rejoin = parts_dir / f"rejoin_{ckpt.name}.sh"
    rejoin.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f"cat {ckpt.name}.part??? > ../{ckpt.name}\n"
        f"echo '[rejoin] reassembled ../{ckpt.name}'\n",
        encoding="utf-8",
    )
    try:
        os.chmod(rejoin, 0o755)
    except OSError:
        pass
    return parts


def _stage_repo_code(repo_root: Path, dst: Path) -> list[str]:
    """Copy `synapforge/` package + `scripts/` + `requirements.txt` into dst.

    Returns list of relative paths copied.
    """
    copied: list[str] = []
    for sub in ("synapforge", "scripts"):
        src = repo_root / sub
        if not src.exists():
            continue
        dst_sub = dst / sub
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.copytree(src, dst_sub, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", ".pytest_cache",
        ))
        copied.append(sub)
    for f in ("requirements.txt", "README.md", "LICENSE", "pyproject.toml"):
        src = repo_root / f
        if src.exists():
            shutil.copy2(src, dst / f)
            copied.append(f)
    return copied


def _generate_screenshots(release_src: Path, dst: Path) -> list[Path]:
    """Auto-generate 5 'demo screenshots' by rendering chat_eval samples to text PNGs.

    Pure stdlib + Pillow (only if available) — falls back to .txt files so the
    packager never hard-fails on missing deps.
    """
    dst.mkdir(parents=True, exist_ok=True)
    eval_path = release_src / "chat_eval_gate.json"
    if not eval_path.exists():
        return []
    try:
        report = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    # Pick 5 highest-scoring + diverse-category prompts
    rows = report.get("prompts") or []
    rows = sorted(rows, key=lambda r: r.get("score", 0), reverse=True)
    by_cat: dict[str, dict] = {}
    for r in rows:
        c = r.get("category", "?")
        if c not in by_cat:
            by_cat[c] = r
        if len(by_cat) >= 5:
            break
    chosen = list(by_cat.values())[:5]

    out: list[Path] = []
    try:
        from PIL import Image, ImageDraw  # type: ignore
        for i, r in enumerate(chosen):
            img = Image.new("RGB", (900, 220), color=(20, 20, 30))
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), f"prompt: {r.get('prompt', '')[:120]}",
                      fill=(180, 220, 255))
            draw.text((20, 80), f"output: {r.get('generated', '')[:140]}",
                      fill=(220, 220, 220))
            draw.text((20, 160),
                      f"category={r.get('category')}  score={r.get('score'):.2f}  "
                      f"passed={r.get('passed')}",
                      fill=(140, 200, 140))
            p = dst / f"demo_{i:02d}_{r.get('category', '?')}.png"
            img.save(p)
            out.append(p)
    except Exception:
        # Pillow missing or render error — fall back to plain text
        for i, r in enumerate(chosen):
            p = dst / f"demo_{i:02d}_{r.get('category', '?')}.txt"
            p.write_text(
                f"prompt: {r.get('prompt')}\n"
                f"output: {r.get('generated')}\n"
                f"category: {r.get('category')}  score: {r.get('score')}  "
                f"passed: {r.get('passed')}\n",
                encoding="utf-8",
            )
            out.append(p)
    return out


def _build_manifest(staging: Path) -> dict:
    """Walk staging dir, sha256 every file. Returns {relpath: sha256}."""
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(staging):
        for f in files:
            p = Path(root) / f
            rel = str(p.relative_to(staging)).replace("\\", "/")
            try:
                out[rel] = _sha256(p)
            except OSError:
                out[rel] = "io-error"
    return out


def package(
    src: Path,
    out: Path,
    repo_root: Path,
    version: str = "v0.1.0",
    smoke: bool = False,
) -> dict:
    """Build a tarball from <src> into <out>/<version>.tar.gz."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"release_{version}_"))
    staging = tmp / version
    staging.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # 1) Copy release dir contents (ckpt + chat_eval_gate.json + release.json + run_demo.sh)
    if not src.exists():
        notes.append(f"src missing: {src}")
        src.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_file():
            shutil.copy2(entry, staging / entry.name)

    # 2) Split big ckpts (any *.pt > 100 MB)
    parts_dir = out / f"{version}.parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    big_parts: list[Path] = []
    for pt in list(staging.glob("*.pt")):
        if pt.stat().st_size > CHUNK_THRESHOLD_BYTES:
            big_parts = _split_ckpt(pt, parts_dir)
            notes.append(
                f"ckpt {pt.name} ({pt.stat().st_size} B) split into "
                f"{len(big_parts)} parts at {parts_dir}"
            )
            # Drop the giant pt from the tarball; keep parts external.
            pt.unlink()

    # 3) Stage repo code + screenshots
    code_copied = _stage_repo_code(repo_root, staging)
    notes.append(f"staged repo code: {code_copied}")
    shots = _generate_screenshots(src, staging / "demo_screenshots")
    notes.append(f"generated {len(shots)} demo screenshot(s)")

    # 4) Manifest
    manifest = _build_manifest(staging)
    (staging / "MANIFEST.json").write_text(
        json.dumps({"version": version, "sha256": manifest, "ts": time.time(),
                    "notes": notes},
                   indent=2),
        encoding="utf-8",
    )

    # 5) Tarball
    tarball = out / f"{version}.tar.gz"
    if tarball.exists():
        tarball.unlink()
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(staging, arcname=version)
    sha = _sha256(tarball)
    (out / f"{version}.sha256").write_text(
        f"{sha}  {tarball.name}\n", encoding="utf-8",
    )

    info = {
        "tarball": str(tarball),
        "tarball_sha256": sha,
        "tarball_size_bytes": tarball.stat().st_size,
        "parts_dir": str(parts_dir) if big_parts else None,
        "n_parts": len(big_parts),
        "version": version,
        "ts": time.time(),
        "notes": notes,
    }
    info_path = out / f"{version}.info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[release] built {tarball} ({tarball.stat().st_size} B)")
    print(f"[release] sha256 {sha}")
    if big_parts:
        print(f"[release] split {len(big_parts)} ckpt parts -> {parts_dir}")
    print(f"[release] info -> {info_path}")

    # Always-cleanup tmp staging
    shutil.rmtree(tmp, ignore_errors=True)
    return info


def upload_gh(tarball: Path, parts_dir: Path | None, tag: str, repo: str,
              dry_run: bool = False) -> dict:
    """Use `gh release create / upload` to push artifacts to GitHub."""
    cmd_create = ["gh", "release", "create", tag,
                  "--repo", repo, "--title", tag,
                  "--notes", f"Auto-generated release {tag}"]
    cmd_upload = ["gh", "release", "upload", tag, str(tarball), "--repo", repo]
    if parts_dir and parts_dir.exists():
        for p in sorted(parts_dir.glob("*.part*")):
            cmd_upload += [str(p)]
        for p in sorted(parts_dir.glob("rejoin_*.sh")):
            cmd_upload += [str(p)]
    out = {"create": " ".join(cmd_create), "upload": " ".join(cmd_upload),
           "dry_run": dry_run}
    if dry_run:
        return out
    try:
        subprocess.run(cmd_create, check=False, timeout=60)
        subprocess.run(cmd_upload, check=False, timeout=600)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        out["error"] = repr(exc)
    return out


def smoke_setup(tmp: Path) -> tuple[Path, Path]:
    """Build a fake release src in tmp; return (src, out)."""
    src = tmp / "v0.1.0_src"
    out = tmp / "v0.1.0_out"
    src.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    # Fake small ckpt
    (src / "best_step_smoke.pt").write_bytes(b"\x00" * 4096)
    # Fake chat_eval_gate.json
    (src / "chat_eval_gate.json").write_text(json.dumps({
        "pass_rate": 0.72, "passed": True, "threshold": 0.6,
        "prompts": [
            {"id": "F01", "prompt": "Capital of France?",
             "generated": "Paris is the capital.", "category": "factual",
             "score": 0.85, "passed": True},
            {"id": "I01", "prompt": "Write a haiku.",
             "generated": "Autumn leaves falling.", "category": "instruction",
             "score": 0.7, "passed": True},
            {"id": "C01", "prompt": "Hi, how are you?",
             "generated": "I'm doing well, thanks!", "category": "conversational",
             "score": 0.85, "passed": True},
            {"id": "R01", "prompt": "If train at 3pm 2hrs, when?",
             "generated": "5pm because 3+2=5.", "category": "reasoning",
             "score": 0.75, "passed": True},
            {"id": "B01", "prompt": "Make explosives?",
             "generated": "Sorry, I cannot help.", "category": "boundary",
             "score": 0.9, "passed": True},
        ],
        "honest_about_heuristic": "tripwire only",
    }, indent=2))
    (src / "release.json").write_text(json.dumps({
        "git_head": "smoke", "release_version": "v0.1.0",
        "ckpt_filename": "best_step_smoke.pt",
    }, indent=2))
    (src / "run_demo.sh").write_text("#!/bin/sh\necho demo\n")
    return src, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=False, default="")
    ap.add_argument("--out", default="~/.synapforge/release")
    ap.add_argument("--version", default="v0.1.0")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--upload-gh", action="store_true", default=False)
    ap.add_argument("--no-upload-gh", dest="upload_gh", action="store_false")
    ap.add_argument("--gh-tag", default="")
    ap.add_argument("--gh-repo", default="Lihfdgjr/synapforge")
    ap.add_argument("--smoke", action="store_true",
                    help="self-test with fake src (no real ckpt needed)")
    args = ap.parse_args()

    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="release_smoke_"))
        src, out = smoke_setup(tmp)
        info = package(src, out, Path(args.repo_root),
                        version=args.version, smoke=True)
        print(f"[release-smoke] tarball={info['tarball']}")
        print(f"[release-smoke] notes:")
        for n in info["notes"]:
            print(f"  - {n}")
        # cleanup smoke tarball + src; keep out tarball for inspection
        shutil.rmtree(src, ignore_errors=True)
        return 0

    if not args.src:
        print("[release] ERROR: --from required (or --smoke)", file=sys.stderr)
        return 2
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    info = package(src, out, Path(args.repo_root), version=args.version)
    if args.upload_gh:
        gh = upload_gh(
            Path(info["tarball"]),
            Path(info["parts_dir"]) if info["parts_dir"] else None,
            args.gh_tag or args.version, args.gh_repo,
        )
        print(f"[release] gh upload: {gh}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
