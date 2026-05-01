# Reliability — silent killers in the 24h LNN+SNN pipeline

The 24-hour Qwen-vocab run on rental burns >$160 of GPU and a non-recoverable
calendar window. Four classes of bug have killed past runs *silently* — i.e.
training kept producing log lines + ckpts, but at the end of the run the
artifact we wanted didn't exist where we expected. This doc catalogues each
killer, the fix landed in this commit, and how to verify the fix is working.

If you change anything here, also re-read the user-memory entry
"Don't hallucinate that training is going up" — the metaphor for these
killers is exactly that: the curve looks fine while the project is dying.

---

## 1. Backup daemon watching the wrong directory

### Symptom
- `triple_backup_daemon.py` is alive in `systemctl --user list-units`.
- Cycle log lines roll by every 600s ("`cycle #N: 0 files, 0.0MB`").
- mohuanfang.com /home/liu/synapforge_backup/<run>/ is empty.
- After the rental dies, no off-rental copy exists.

### Mechanism
Operator hand-edited the rental's systemd unit (or `--watch` flag in
`launch_train.sh`) to point at `/workspace/runs/v24h_qwen/` after the
Qwen-vocab pivot, but the trainer is still writing into the *old* path
(e.g. `/workspace/runs/synapforge_100m`). Because rsync of an empty
directory is a no-op success, the daemon prints `cycle #N success: ...`
every 10 minutes and nobody notices for hours.

### Fix
Two things:

1. **Loud warning when the watch dir has been empty for 5 cycles.**
   `_warn_if_persistently_empty(watch_dir, file_count)` is called once per
   cycle in `watch_loop`. If 5 consecutive cycles see `len(files) == 0`
   it logs:
   ```
   WARNING: backup daemon watching empty dir for 5 cycles (<path>);
   did you point to the wrong path?
   ```
   It re-warns every 5 cycles after that. Counter resets to 0 the first
   time files appear.
2. **`mkdir(parents=True, exist_ok=True)` at startup** so an operator who
   wrote a path with a missing parent at least gets a non-crashing daemon
   plus the loud warning above.

The cycle count, file count, and `empty_cycles_seen` are also persisted
into `<watch_dir>/.backup_status.json` for outside monitors / dashboards.

### Verification
```bash
# 1. point daemon at a tmpdir, leave it empty
mkdir -p /tmp/empty_watch
python scripts/triple_backup_daemon.py --watch /tmp/empty_watch --interval 5
# expect after ~25s:
#     WARNING: backup daemon watching empty dir for 5 cycles ...
```

### Common false positive
"Daemon prints success every cycle" — that's just rsync syncing 0 bytes.
The only correct success signal is: `mohuanfang.com:/home/liu/synapforge_backup/<run>/`
contains a `step_*.pt` file with a recent mtime.

---

## 2. GitHub Release silent 100MB asset cap

### Symptom
- `gh release upload` exits 0 in our daemon.
- `gh release view <tag>` shows the asset row but with size = 0 / 100MB
  truncated, or `validation failed` if you check the API directly.
- Recovery tries `gh release download` and gets a corrupt `best.pt`
  (sha256 mismatch). Trainer with `--warmstart=$DOWNLOAD` then crashes
  inside `torch.load`.

### Mechanism
For our 151M-param model + Adam optimizer state, `best.pt` is ~600 MB.
GitHub Releases nominally allow 2 GB per asset, but the gh CLI silently
fails on uploads >100 MB under intermittent network conditions, and
that failure mode has bitten us during prior runs.

### Fix
`_chunked_split_fallback(src_path, chunk_mb=50, out_dir=...)` shreds any
`.pt` file >100 MB into 50 MB `chunk-NNN` files plus a `manifest.json`
that carries:

- `src_size` and `src_sha256` — full-file digest for end-to-end verify
- `chunks: [{name, size, sha256}, ...]` — per-chunk digest + ordering
- `recovery` — a one-liner copy-pasteable into bash for restoration

`push_github_top_best` segregates `fresh` ckpts into "fits flat
(<=100MB)" and "needs split". Big ckpts go to a separate release with
the `-split` tag suffix so recovery is unambiguous; the manifest lives
in the same release.

### Verification
```bash
# 1. force a >100MB file, dry-run the split pipeline
dd if=/dev/urandom of=/tmp/fake_best.pt bs=1M count=200
python -c "
from pathlib import Path
import sys; sys.path.insert(0, 'scripts')
import triple_backup_daemon as tbd
chunks, manifest = tbd._chunked_split_fallback(Path('/tmp/fake_best.pt'), chunk_mb=50, out_dir=Path('/tmp/split_out'))
print(f'{len(chunks)} chunks, manifest at {manifest}')
restored = tbd._assemble_chunks(manifest, Path('/tmp/restored.pt'))
print(f'restored OK at {restored}')
"
# expect: 4 chunks + manifest, then 'restored OK ... (200,000,000 B verified)'
```

Recovery on a fresh box (after the rental dies):
```bash
gh release download auto-v24h_qwen-20260501-split --repo Lihfdgjr/synapforge
cat best.pt.chunk-* > best.pt
python -c "
import hashlib, json, sys
m = json.load(open('best.pt.manifest.json'))
h = hashlib.sha256(open(m['src_name'], 'rb').read()).hexdigest()
sys.exit(0 if h == m['src_sha256'] else 1)
"
echo $?  # 0 = bytes match
```

### Common false positive
`gh release view` showing the right number of asset rows, *without*
checking the size column or running sha256sum, "looks like it worked."
Always verify against the manifest's `src_sha256` end-to-end — chunk-level
hashes alone don't catch a missing tail chunk.

---

## 3. `phase_manager.py` writes `.phase` but trainer ignores it

### Symptom
- `scripts/phase_manager.py` running for hours, writes
  `<out_dir>/.phase` JSON with `phase_id=1, phase_name=intrinsic`.
- `train.log` keeps logging Phase 0 KD-only steps; intrinsic / self-learn
  / STDP-novelty never come on.
- ppl plateau at 240ish for 6+ hours. Training "succeeds" but produces
  a Phase 0 model.

### Mechanism
The phase manager design assumed a human-in-the-loop reading
`<run>/.phase` and re-launching the trainer. In a 24h unattended run
nobody reads it; the file is permanently stale.

### Fix
1. **`synapforge/phase_signal.py`** is the canonical reader/consumer:
   - `read_phase(out_dir)` — non-destructive read, defensive against
     malformed JSON / FS races / missing dir.
   - `consume_phase(out_dir)` — atomic `os.replace` of `.phase` to
     `.phase.consumed.<ts>`. Idempotent (a second consume returns None).
2. **`train_100m_kd.py --phase-aware`** (default OFF, *does not change
   the running command*). When on:
   - Reads initial `.phase` at startup to set `current_phase_id`.
   - Every 100 steps, calls `phase_signal.consume_phase`.
   - On phase change: saves
     `<out_dir>/phase_change_step_NNNNNN.pt`, persists `metrics.json`,
     and `sys.exit(101)`.
   - The outer relauncher (a shell wrapper, not in this repo) should
     watch for exit 101 and re-spawn the trainer with the new phase
     flags appended.

The exit code 101 is chosen so it doesn't collide with 0 (clean exit),
1 (Python error), or 2 (argparse error).

### Verification
```bash
python synapforge/phase_signal.py
# expect: phase_signal smoke OK
```

End-to-end (requires torch on rental):
```bash
mkdir -p /tmp/run_pa
echo '{"phase_id":1,"phase_name":"intrinsic"}' > /tmp/run_pa/.phase
python train_100m_kd.py --out /tmp/run_pa --phase-aware --steps 200 ...
# expect logs to show:
#   [phase-aware] enabled; initial phase_id=None
#   [phase-aware] phase change None -> 1: 'intrinsic' flags=[...]
#   [phase-aware] saved ckpt /tmp/run_pa/phase_change_step_000100.pt
#   [phase-aware] sys.exit(101) -- relauncher should re-spawn
```

### Common false positive
"`--phase-aware` flag accepted at startup" — the flag default is OFF.
The default training command never goes through this code path, so a
green smoke of the default command says nothing about whether
`--phase-aware` would actually work. Always test with `--phase-aware`
explicitly toggled on against a hand-written `.phase` file.

---

## 4. `honest_eval_hook` exists but never wired into the trainer

### Symptom
- `scripts/honest_eval_hook.py` is on disk, `import HonestEvalHook`
  succeeds.
- `train.log` shows `VAL step 1500: ppl=87.4`.
- Nobody has actually looked at chat output. Sampling later shows the
  model emits English-grammar-shaped wikitext n-grams with zero
  semantic content, and zero Chinese despite the Qwen-vocab pivot.

### Mechanism
The eval hook was authored separately from the trainer and never
imported. `train_100m_kd.py` only logged numerical ppl, which encodes
"the training-distribution next-token statistic looks better" — it
does *not* encode "the model can generate coherent text", let alone
"the model can do what we shipped Qwen vocab for, namely Chinese."

### Fix
1. **`train_100m_kd.py` imports `HonestEvalHook`** with a try/except
   guard so a missing `scripts/` directory just disables the hook
   instead of crashing the trainer.
2. **`--honest-eval` (default True)** with the explicit
   `--no-honest-eval` opt-out so historic launch scripts can disable
   if needed.
3. **Inside the EVAL_EVERY block**, after the existing val-ppl print,
   the trainer now calls `eval_hook.maybe_eval(step, ppl)` inside a
   `try/except` that swallows everything. Eval cannot kill training.
4. The hook uses the **Qwen tokenizer** loaded a few lines above
   (`tok = load_tokenizer("/workspace/teachers/qwen2.5-0.5b")`), which
   means EN + ZH prompts both decode through the right vocab.
5. Output is appended to `<out_dir>/honest_eval.jsonl` plus a stdout
   pretty-print of the first two samples per cycle.

### Verification
```bash
python scripts/honest_eval_hook.py
# expect: smoke OK; jsonl: /tmp/honest_smoke/honest_eval.jsonl
tail -1 /tmp/honest_smoke/honest_eval.jsonl
# expect a single JSON line: {"step":1, "ts":..., "ppl":88.0, "verdict_heuristic":"...", "samples":[...]}
```

End-to-end on rental, after a 500-step EVAL boundary:
```bash
ls /workspace/runs/v24h_qwen/honest_eval.jsonl
# file exists, one record per eval step
jq -c '{step, verdict: .verdict_heuristic, ppl, first_sample: .samples[0]}' \
    /workspace/runs/v24h_qwen/honest_eval.jsonl | tail -5
```

### Common false positive
"`verdict_heuristic` says GRAMMAR_OK" — that just means the model
emits punctuation in plausible places. Always read the actual
`samples[*].generated` text. The verdict is a fast hint; the source of
truth is the original strings. (See user-memory entry on "Don't trust
ppl, read the original output.")

---

## How the four fixes interact

The four killers are deliberately ordered by upstream-ness:

1. The backup daemon must work, or the next 3 don't matter (the run
   ends with everything on a dead rental disk).
2. GitHub release chunking must work, or the backup-daemon-works
   illusion holds but the artifact pulled in recovery is corrupt.
3. The phase signal must be honored, or the artifact is the wrong
   model (Phase 0 instead of Phase 2).
4. The honest eval must run, or even the right-phase model could be
   Phase-0-quality wikitext-noise and we wouldn't know.

Each layer adds a different signal to `train.log` /
`<out_dir>/.backup_status.json` / `<out_dir>/honest_eval.jsonl`, so
post-mortem analysis can pinpoint which layer failed.

## What's *not* covered by this doc

- Spike rate drift (`spike_target` warnings) — already covered.
- Teacher load failure → kd-weight=0 fallback — already covered.
- Triton kernel bf16 fallback — already covered (see CHANGELOG).
- Phase-aware *relauncher script* (the wrapper that catches exit 101
  and re-spawns) — out of scope here; lives in `launch_train.sh` (TODO).
