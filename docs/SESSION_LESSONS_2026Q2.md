# SESSION_LESSONS_2026Q2 — durable lessons codified to memory

5 lessons learned during Run 3l/3m/3n (2026-05-02) that future training
sessions must respect. Each has a `feedback_*.md` memory entry with full
rule + why + how-to-apply detail; this doc is the next-session pickup.

## Lesson 1 — lm-head spectral_norm warmstart cost

`feedback_spectral_norm_warmstart_cost.md`. T2.6 `--lm-head-spectral-norm`
wraps `lm_head.weight` as `weight_orig` + `weight_v` + `weight_u` (PyTorch
spectral_norm standard). Loading a non-spectral-norm ckpt into a
spectral-norm model triggers 71 missing keys / 183 unexpected keys, and
the entire lm-head goes random-init. Run 3m paid ~5h training time
recovering from this. Rule: if shipping spectral_norm to production,
either (a) train from scratch with the flag enabled from step 0, or
(b) train a "bridge" run (1-2k steps with old ckpt warmstart + new
spectral_norm flag) until val ppl returns to baseline, then save that
as the new warmstart anchor. Never treat spectral_norm as a mid-run
toggle; it changes module topology.

## Lesson 2 — cosine LR decay too low after warmstart → plateau then divergence

`feedback_cosine_lr_warmstart_replateau.md`. Run 3m at LR 5e-5 + cosine
schedule continued from step ~10000 reached lr=3e-5 by step 14000 then
val ppl re-diverged from 4406 → 25965. Root cause: cosine schedule
continuation puts you in the LR trough before the warmstarted model
has re-converged on the new architecture (in this case, the freshly
reset spectral_norm lm-head). Run 3n switched to LR 2e-5 constant and
showed clean stability through step ~3000+. Rule: when warmstarting,
don't blindly continue cosine schedule from `step_N`. Either re-warmup
LR linearly from zero over 500-1000 steps, or use constant LR until you
confirm val ppl drops monotonically for ≥2k steps. Only THEN switch to
cosine decay. Encode in trainer defaults; don't rely on launch-script
discipline.

## Lesson 3 — mtime-newest ≠ latest-trained ckpt

`feedback_warmstart_explicit_step_not_mtime.md`. `step_004000.pt`
(Run 3e from days earlier) was newer-mtime than `step_002000.pt`
(Run 3l's anchor) when Run 3l started at 23:34 — but Run 3l's first
in-run ckpt save didn't land until 23:59. Any cron job during that
gap that picked latest by `ls -t step_*.pt | head -1` got Run 3e's
artifact, not Run 3l's. Run 3l's chat samples / bench results during
that window thus reflected the wrong model state. Rule: warmstart and
all downstream cron jobs must reference an EXPLICIT step path
(`/workspace/runs/v24h_qwen3/step_002000.pt` written verbatim into the
launch script), or read the `best_*.pt` symlink (T5.4-shipped). NEVER
sort by mtime/`ls -t` in multi-run shared directories. Add `--run-tag`
to ckpt names if multiple runs share a dir.

## Lesson 4 — Run 3c-class divergence threshold = 7000 ppl, kill don't wait

`feedback_run3c_divergence_threshold.md`. Three independent
divergence events confirmed the empirical threshold: Run 3c step 2500
(ppl 1886, initial divergence from ~600), Run 3l step 5500 (ppl 2522),
Run 3m step 15000 (ppl 25965). When val ppl crosses ~7000 within 500
steps of a previous lower value, OR jumps ≥3× the prior best within
1000 steps, the run is dead. Adam m/v second-moment is polluted by
extreme gradients; "let it self-correct" doesn't work — three runs
gave 0/3 recovery. Total wasted GPU time across these "maybe it'll
recover" delays: ~10 hours. Rule: encode the threshold in the cron
H1 health check; SIGTERM the trainer immediately and rollback to the
best ckpt (which requires Lesson 5).

## Lesson 5 — best-ckpt symlink saves emergency relaunches

`feedback_best_ckpt_track_mandatory.md`. Run 3m's actual best ckpt was
`step_013500.pt` (val ppl 4406). The rotation policy `keep-last-5` ran
on every save and evicted step_013500 once steps 14000/14500/15000
landed (since by then best wasn't in the last-5 window). When the kill
fired at step 15000, the best ckpt to rollback to was already gone —
forcing rollback to step_011000 (a worse ppl) and losing ~1h of
post-best training. T5.4 best-ckpt-track (just shipped) creates a
symlink `best_run3<letter>.pt → step_NNNN.pt` and the rotation policy
skips symlink-targets. Rule: enable `--best-ckpt-track` (default True)
on every trainer launch, ALWAYS. Cron's H1 check additionally verifies
the symlink isn't dangling. Cost: zero. EV: ~¥350-1400/month in
recovered A800 rental time at 5-10 divergence events per month.

## Cross-references

- `feedback_no_random_init_use_warmstart.md` (older sibling rule) — warmstart fundamentals.
- `feedback_data_ordering_divergence_2026q2.md` — Run 3a/3b/3c data-shuffle root cause, separate from these 5.
- `feedback_training_root_causes_2026q2.md` — Run 1-3h 8-issue summary; this doc extends it with Run 3l/3m/3n's three further failure modes.
- `MEMORY.md` index entries (5 new lines added 2026-05-02) — one-line summaries for cross-session lookup.

## Application checklist for next-session trainer launch

```
[ ] If using --lm-head-spectral-norm, confirm warmstart ckpt was already
    spectral_norm-trained. If not, run a bridge-ckpt step first.
[ ] LR for warmstart resumes is constant (not cosine continuation) until
    val ppl is stable for ≥2k steps.
[ ] Warmstart path is explicit step (`step_NNNN.pt`), not `ls -t`
    or mtime-based selection.
[ ] H1 cron health check enforces: val ppl > 7000 within 500 steps OR
    val ppl > 3× best within 1000 steps → SIGTERM + relaunch from best.
[ ] --best-ckpt-track is set (default True). Cron verifies symlink valid
    on each fire.
```
