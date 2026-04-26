"""Periodic backup: ONLY logs/metrics/samples to synapforge-runs GitHub repo.
.pt ckpts stay on rental's persistent /workspace (algoshell preserves system disk).

v2 (2026-04-26): glob-scan of /workspace/runs/synapforge_* so new runs are picked up.
"""
import subprocess, time, os, shutil, glob

INTERVAL_S = 600
REPO_DIR = "/workspace/runs_repo"
GITHUB_URL = "https://github.com/Lihfdgjr/synapforge-runs.git"
RUNS_GLOB = "/workspace/runs/synapforge_*"
TEXT_EXTS = (".log", ".json", ".txt", ".csv", ".md")
MAX_FILE_BYTES = 5 * 1024 * 1024


def setup():
    if not os.path.exists(f"{REPO_DIR}/.git"):
        os.makedirs(REPO_DIR, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=REPO_DIR, check=False)
        subprocess.run(["git", "checkout", "-b", "main"], cwd=REPO_DIR, capture_output=True)
        subprocess.run(["git", "config", "user.email", "rental@synapforge.local"], cwd=REPO_DIR)
        subprocess.run(["git", "config", "user.name", "rental"], cwd=REPO_DIR)
        subprocess.run(["git", "remote", "add", "origin", GITHUB_URL], cwd=REPO_DIR, capture_output=True)


def discover_srcs():
    return sorted(set(d for d in glob.glob(RUNS_GLOB) if os.path.isdir(d)))


def sync():
    ts = time.strftime("%H:%M:%S")
    srcs = discover_srcs()
    n_files_copied = 0
    for src in srcs:
        name = os.path.basename(src)
        dst = f"{REPO_DIR}/{name}"
        os.makedirs(dst, exist_ok=True)
        try:
            entries = os.listdir(src)
        except OSError:
            continue
        for f in entries:
            if not f.endswith(TEXT_EXTS):
                continue
            src_f = f"{src}/{f}"
            try:
                if os.path.getsize(src_f) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            dst_f = f"{dst}/{f}"
            try:
                if not os.path.exists(dst_f) or os.path.getmtime(src_f) > os.path.getmtime(dst_f):
                    shutil.copy2(src_f, dst_f)
                    n_files_copied += 1
            except OSError as e:
                print(f"[{ts}] copy err {src_f}: {e}", flush=True)
    subprocess.run(["git", "add", "-A"], cwd=REPO_DIR, capture_output=True)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_DIR)
    if r.returncode == 0:
        print(f"[{ts}] no changes ({len(srcs)} dirs)", flush=True)
        return
    subprocess.run(["git", "commit", "-m", f"auto: {ts} ({n_files_copied} files, {len(srcs)} dirs)"],
                   cwd=REPO_DIR, capture_output=True)
    push = subprocess.run(["git", "push", "-u", "origin", "main", "--force"],
                          cwd=REPO_DIR, capture_output=True, text=True, timeout=120)
    print(f"[{ts}] push rc={push.returncode}: {push.stderr.strip()[:200]}", flush=True)


def main():
    setup()
    print(f"[start] auto_runs_backup v2 every {INTERVAL_S}s", flush=True)
    while True:
        try: sync()
        except Exception as e: print(f"[ERR] {e}", flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
