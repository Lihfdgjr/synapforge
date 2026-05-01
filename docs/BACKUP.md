# Backup Strategy — never lose another training run

After the 2026-04-30 v4.1 disaster (7 hours of training lost when the
rental's sshd died), **every** worth-keeping training artifact is now
mirrored off-rental within 10 minutes. This is structural, not policy.

## The 3-path triple backup

The trainer (`synapforge/train_v42_universal.py`) auto-spawns
`scripts/triple_backup_daemon.py` on rank 0 at startup. The daemon does
a **full-volume sync** of `--out-dir` every `--backup-interval` seconds
(default 600s = 10 min) to up to 3 destinations in parallel:

| Path | Target | What | Cap | Use |
|---|---|---|---|---|
| 1 | `mohuanfang.com:/home/liu/synapforge_backup/<run>/` | rsync entire run dir, incremental | 1.5T disk | primary, always on |
| 2 | GitHub Release `Lihfdgjr/synapforge:auto-<run>-<date>` | top-3 most recent `best*.pt` | 100MB/asset, 2GB total | public mirror, recovery via `gh release download` |
| 3 | HuggingFace dataset `Lihfdgjr/synapforge-ckpts` (private) | full folder upload | unlimited | tertiary, archival |

**Success criterion**: at least 1 of 3 succeeds per cycle. Total failure
prints loud WARN but never crashes the trainer — the daemon retries next
cycle.

## What gets backed up (full-volume)

Everything under `--out-dir` is rsynced as-is:

- All `step_*.pt` and `best*.pt` checkpoints
- `train.log` (training log)
- `skill_log.json` (NeuroMCP persisted skills — hard to regenerate)
- `events.out.tfevents.*` (TensorBoard if used)
- `config.json` / `args.json` (the trainer flags this run was launched with)
- `optimizer_state.pt` (resume-able if present)
- `.backup_status.json` (the daemon's own heartbeat — useful to verify it's alive)

## Configuration

```bash
# trainer auto-spawn (default behavior)
python -m synapforge.train_v42_universal \
  --out-dir /workspace/runs/synapforge_v42_universal \
  --backup-interval 600     # default 10 min, raise to 3600 for slow links

# disable for debug runs
python -m synapforge.train_v42_universal --disable-backup-daemon ...

# standalone (e.g. backing up a finished run)
python scripts/triple_backup_daemon.py \
  --watch /workspace/runs/synapforge_v42_universal \
  --once    # one cycle then exit
```

Required env vars (each path is independent — if a token is missing,
that path is skipped, the others still run):

```bash
export GH_TOKEN=ghp_...                # for GitHub Releases
export HF_TOKEN=hf_...                 # for Hugging Face
export MOHUANFANG_HOST=mohuanfang.com  # default; override for staging
export MOHUANFANG_USER=liu             # default
```

ssh key for `liu@mohuanfang.com` must be in the rental's
`/root/.ssh/authorized_keys` of mohuanfang (and the rental's pubkey on
mohuanfang's authorized_keys for incoming auth). See
`reference_mohuanfang_backup.md` in memory.

## Recovery

```bash
# from mohuanfang (preferred)
ssh liu@mohuanfang.com 'ls /home/liu/synapforge_backup/<run>/'
rsync -av liu@mohuanfang.com:/home/liu/synapforge_backup/<run>/ /workspace/runs/<run>/

# from GitHub Release (top-3 best.pt only)
gh release download auto-<run>-YYYYMMDD --repo Lihfdgjr/synapforge

# from HuggingFace (full folder)
huggingface-cli download Lihfdgjr/synapforge-ckpts --repo-type dataset \
  --include "<run>/*" --local-dir /workspace/runs/<run>/
```

## Honest perf notes

- mohuanfang link: tested 7 MB/s rental↔mohuanfang on initial recovery.
  Full 4.3GB rsync takes ~10 min; incremental cycles are <1 min unless
  a new full ckpt was just written.
- GitHub Releases: 100MB/asset cap means our 375M models (1.5GB) **cannot**
  be uploaded as raw release assets. Workaround: GitHub LFS (already
  configured in this repo's `.gitattributes`) or split file. Currently
  daemon uploads `best*.pt` (typically 200-600M for smaller variants);
  the 375M v4.x best.pt would need LFS or split.
- HuggingFace: `upload_folder` re-uploads only changed files (LFS-aware
  hashing); large runs amortize well across cycles.
- Daemon is single-threaded but pushes to all 3 paths within one cycle;
  ~5 min upload time for a 600MB best.pt to 3 destinations is typical.

## When the daemon DOESN'T save you

- Same-region failure mode: if mohuanfang and the rental are both in
  Sichuan/CN and the regional ISP is down, both unreachable. **HF** is
  global, so always keep `HF_TOKEN` set.
- Token expiry: GH/HF tokens expire silently. The daemon logs FAIL each
  cycle but doesn't email — check `.backup_status.json` periodically.
- Trainer crash before first cycle: `--backup-interval 600` means the
  first 10 minutes of training have no off-rental copy. For very valuable
  warmstart points, run `--once` manually right after warmstart load.
