"""auto_ckpt_backup — periodically picks best-ppl ckpt per run dir,
copies to /workspace/best_ckpts/ (preserved by algoshell on stop),
optionally compresses + uploads to GitHub release every N hours.

Strategy:
1. Scan /workspace/runs/synapforge_*/step_*.pt
2. Parse metrics.json for val ppl per step
3. Promote best ckpt per run -> /workspace/best_ckpts/<run>_best.pt
4. Every N hours: tar.gz the 3 most recent runs' best ckpts, push to a github release.
"""
import os, glob, json, time, shutil, subprocess, hashlib, sys

INTERVAL_S = 1800  # 30 min
RELEASE_INTERVAL_S = 3600 * 4  # tar+release every 4h
RUNS_GLOB = "/workspace/runs/synapforge_*"
BEST_DIR = "/workspace/best_ckpts"
RELEASE_REPO = "Lihfdgjr/synapforge-runs"
TOKEN = os.environ.get("GH_TOKEN", "")
LAST_RELEASE_FILE = "/tmp/auto_ckpt_last_release"

os.makedirs(BEST_DIR, exist_ok=True)


def parse_best_per_run():
    """For each run dir, return (run_name, best_step, best_ppl, ckpt_path) or None."""
    out = []
    for run in sorted(glob.glob(RUNS_GLOB)):
        if not os.path.isdir(run):
            continue
        name = os.path.basename(run)
        # Find ckpts
        ckpts = sorted(glob.glob(os.path.join(run, "step_*.pt")))
        if not ckpts:
            continue
        # Find ppl from metrics.json
        metrics_path = os.path.join(run, "metrics.json")
        best_step = None
        best_ppl = float("inf")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
                ppls = m.get("ppl_eval", {})
                for step_str, ppl in ppls.items():
                    try:
                        ppl = float(ppl)
                    except (TypeError, ValueError):
                        continue
                    if ppl == ppl and ppl < best_ppl:  # NaN check
                        best_ppl = ppl
                        best_step = int(step_str)
            except Exception:
                pass
        # If no metrics, fall back to latest ckpt
        if best_step is None:
            latest = ckpts[-1]
            try:
                fname = os.path.basename(latest)  # step_NNNNNN.pt
                best_step = int(fname.split("_")[1].split(".")[0])
            except Exception:
                continue
        # Find the ckpt closest to best_step
        target = os.path.join(run, f"step_{best_step:06d}.pt")
        if not os.path.exists(target):
            # Take whichever ckpt step is <= best_step (eval at step N may run after ckpt at N-something)
            avail_steps = []
            for c in ckpts:
                fn = os.path.basename(c)
                try:
                    s = int(fn.split("_")[1].split(".")[0])
                    avail_steps.append((s, c))
                except Exception:
                    continue
            avail_steps.sort()
            cand = [c for s, c in avail_steps if s <= best_step]
            if cand:
                target = cand[-1]
            elif avail_steps:
                target = avail_steps[-1][1]
            else:
                continue
        out.append((name, best_step, best_ppl, target))
    return out


def file_md5(path: str, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def promote_best():
    info = parse_best_per_run()
    promoted = []
    for name, step, ppl, src in info:
        dst = os.path.join(BEST_DIR, f"{name}_best.pt")
        marker = os.path.join(BEST_DIR, f"{name}_best.json")
        # Skip if same file already promoted (md5 match)
        if os.path.exists(marker):
            try:
                with open(marker) as f:
                    prev = json.load(f)
                if prev.get("src_md5") == file_md5(src):
                    continue
            except Exception:
                pass
        shutil.copy2(src, dst)
        with open(marker, "w") as f:
            json.dump({
                "run": name,
                "best_step": step,
                "best_ppl": ppl,
                "src": src,
                "src_md5": file_md5(src),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        promoted.append((name, step, ppl, src, dst))
    return promoted


def tar_release_best():
    """tar.gz best_ckpts/ + upload as a github release asset."""
    if not os.listdir(BEST_DIR):
        return False
    # Skip if recent release (< RELEASE_INTERVAL_S)
    if os.path.exists(LAST_RELEASE_FILE):
        try:
            last = float(open(LAST_RELEASE_FILE).read().strip())
            if time.time() - last < RELEASE_INTERVAL_S:
                return False
        except Exception:
            pass
    tag = "best-" + time.strftime("%Y%m%d-%H%M")
    archive = f"/tmp/{tag}.tar.gz"
    subprocess.run(["tar", "-czf", archive, "-C", "/workspace", "best_ckpts"],
                   check=False)
    if not os.path.exists(archive):
        return False
    # Create release via GitHub API (requires `gh` cli or curl)
    rel_body = json.dumps({
        "tag_name": tag,
        "name": f"Best checkpoints {tag}",
        "body": "Auto-promoted best-val-ppl ckpt per run, packed by auto_ckpt_backup.py",
        "draft": False,
        "prerelease": True,
    })
    # Create release
    r = subprocess.run([
        "curl", "-sS", "-XPOST",
        "-H", f"Authorization: token {TOKEN}",
        "-H", "Accept: application/vnd.github+json",
        "-H", "Content-Type: application/json",
        f"https://api.github.com/repos/{RELEASE_REPO}/releases",
        "-d", rel_body,
    ], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"[release create FAIL] rc={r.returncode}: {r.stderr[:300]}", flush=True)
        return False
    try:
        rel = json.loads(r.stdout)
        upload_url = rel["upload_url"].split("{")[0]
        rel_url = rel["html_url"]
    except Exception as e:
        print(f"[release create parse FAIL] {e}: {r.stdout[:300]}", flush=True)
        return False
    # Upload asset
    sz = os.path.getsize(archive)
    if sz > 2 * 1024**3:  # GitHub 2GB asset limit
        print(f"[release skip] archive {sz/1e9:.1f}GB > 2GB GitHub asset limit", flush=True)
        os.remove(archive)
        return False
    fname = os.path.basename(archive)
    up = subprocess.run([
        "curl", "-sS", "-XPOST", "--data-binary", f"@{archive}",
        "-H", f"Authorization: token {TOKEN}",
        "-H", "Content-Type: application/gzip",
        f"{upload_url}?name={fname}",
    ], capture_output=True, text=True, timeout=600)
    os.remove(archive)
    if up.returncode != 0:
        print(f"[asset upload FAIL] rc={up.returncode}: {up.stderr[:300]}", flush=True)
        return False
    with open(LAST_RELEASE_FILE, "w") as f:
        f.write(str(time.time()))
    print(f"[release OK] {rel_url}  asset={fname} ({sz/1e6:.1f}MB)", flush=True)
    return True


def main():
    print(f"[start] auto_ckpt_backup interval={INTERVAL_S}s release_every={RELEASE_INTERVAL_S}s", flush=True)
    while True:
        try:
            ts = time.strftime("%H:%M:%S")
            promoted = promote_best()
            if promoted:
                for name, step, ppl, src, dst in promoted:
                    print(f"[{ts}] promote {name} step={step} ppl={ppl:.2f} -> {dst}", flush=True)
            else:
                print(f"[{ts}] no new best-ckpt promotions", flush=True)
            tar_release_best()
        except Exception as e:
            print(f"[ERR] {e}", flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
