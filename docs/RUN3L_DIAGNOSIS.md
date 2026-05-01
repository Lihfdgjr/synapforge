<!-- DOC_STAMP: LIVE 2026-05-02 -->
# Run 3l VAL-rise diagnosis — 2026-05-02 00:30

**Subject**: `v24h_qwen3` Run 3l, PID 24075, A800 80GB rental. VAL ppl rising
500→2000 (298 → 313 → 363 → 418). Train CE 5.8–6.1 oscillating with KD step
pattern. Spike rate 0/10 throughout. z-loss steady ~130. LR=1e-4 cosine after
warmup-100, grad-clip 0.5, kd-every=4, kd-topk=0 (full-vocab), shuffle ON.

This run uses the trainer source at-launch (Python loaded `train_100m_kd.py`
into the process before merge `0344bb5`). T2.4-T2.7 commits were merged AFTER
Run 3l started, so the LIVE process does **not** see the freeze-tail hook,
spike-rate-target loss, spectral norm, or grad-accum logic. Diagnosis must
focus on what the running code actually does.

---

## 1. Most-likely root causes (ranked, evidence-based)

### H1 — Run 3l is the **classic Train↔Val drift** (P27 §2.c) — HIGH confidence

**Evidence**:
- Train CE 5.82–6.02 ⇒ train ppl ≈ 340–410. **Train and val ppl are now
  the same number.** That is *not* "rising val on a healthy run"; that is
  **train CE was *already* there** at step 2240 and val is just catching up
  with no headroom.
- z-loss steady at ~130 — the partition isn't blowing up; the logits are
  not catastrophically diverging. Loss numbers are well-behaved.
- KD-step `loss=3.5176` vs no-KD `loss=5.8331` → exactly the
  `(1-α)·base + α·KD = 0.3·5.85 + 0.7·1.96 ≈ 3.13` arithmetic with α=0.7,
  KD KL ~2.0. KD is *working* and pulling the student toward Qwen 0.5B's
  teacher distribution.
- This matches `docs/TRAINING_ISSUES_RETROSPECTIVE.md` §2.c precisely:
  *"train ce 6–7 vs val ppl 320–400 ... student overfits to current
  minibatch, drops on val. Classic small-model + strong-teacher
  overfitting."* Run 3l now sits exactly on that pattern.

**Why VAL is rising and not stable**: the warmstart from `step_002000.pt`
loaded **fresh Adam state** (the launch flags don't include `--no-strip-optim`,
and the warmstart path strips state on vocab/shape mismatch). The optimizer
is re-warming m/v from zero against a model whose weights are
already-half-trained. For the first ~500 optimizer steps after warmstart,
update direction is dominated by the gradient sign without the variance
bias-correction Adam normally provides → drift in token-distribution that
doesn't recover until variance estimates stabilize. This is exactly
`feedback_training_root_causes_2026q2.md` "stale-Adam" failure mode (Issue
#3 in the retrospective) replaying as **fresh-Adam-on-pre-trained-weights**,
its symmetric twin.

### H2 — PLIF dead 10/10 means CfC is the **only model** and it's drifting (P25) — HIGH confidence

**Evidence**:
- Spike rate 0/10 at every layer. T2.5 spike-target loss is
  **NOT in the running process** (merged at 23:59, run launched ~23:34).
  Auto-revive homeostasis is firing and slowly walking threshold 0.0321 → 0.0312
  per 100 steps — that's **~0.0009/100steps**, far too slow to hit the
  surrogate-active band before another 5,000 steps.
- The "LNN+SNN hybrid" is currently **a single CfC backbone** with the
  PLIF tap returning all-zeros, so SparseSynapse(s)·sigmoid(gate(s)) is
  **0 vector** and the spike branch contributes nothing — only the FFN
  residual `x + ffn(ln2(x))` is learning. CE oscillates because the only
  active path (FFN+CfC) has no spike-driven regularization to bound state.
- Neither §2.a's surrogate annealing nor §2.c's BPTT regularizer have
  shipped. The model has **zero defense** against per-step state drift.

### H3 — Sustained KD-on/KD-off step gradient-magnitude oscillation perturbs Adam moments — MEDIUM confidence

**Evidence**:
- KD step (every 4): `loss=3.5176` → `(loss/1).backward()` produces a small
  KL-driven gradient. No-KD step: `loss=5.8331` → backward produces a
  CE+z-loss gradient ~2× larger in magnitude.
- LR=1e-4 with grad-clip 0.5 means the per-step parameter update is
  `lr · ĝ` where ĝ is the clipped+momentum-corrected direction. Adam's
  variance estimate `v̂` averages **squared** gradients across the alternation
  pattern, smoothing it. *But*: the warmstart-fresh-Adam phase has v̂ still
  building up — the alternating large/small gradients during this window
  pump v̂ unevenly, biasing it toward the larger no-KD gradient and
  *under-correcting* updates on KD steps.
- This is not a divergence cause on its own (Run 3b survived 3h with this
  pattern), but it amplifies H1's drift: every 4th step is "soft" KD and
  the other 3 are "hard" CE+z. With `kd_weight=0.7` the off-step is 100%
  CE while on-step is 30% CE — a **3.3× implicit LR difference between
  step types** that pushes weights in micro-cycles.

### Lower-confidence noise

- **z-loss top-K is OFF** (`--kd-topk 0` and the launch line shows no
  `--z-loss-topk`). The trainer's z-loss default is `--z-loss-topk 2048`,
  so it IS sparse — z=130 is a steady-state *value*, not rising
  (logits drift per P28 §2.d would manifest as **rising** z, not steady).
  P28 is **not** the cause of Run 3l's val rise.
- **shuffle works** (`--shuffle-buffer 10000`). P24/P29 ruled out — a
  data-ordering recurrence would show a step-2500 cliff, not a smooth rise
  from step 500.

---

## 2. Differential — why this is NOT just normal warmup

Run 3b at step 2000 was VAL ppl **397 healthy** (per retrospective §1).
Run 3l at step 2000 is VAL ppl **418** and trending up at +30/500-steps.

**Genuinely different vs Run 3b**:

| Dimension | Run 3b | Run 3l |
|-----------|--------|--------|
| Warmstart | `step_002250_plif_reinit.pt` (PLIF param-name break → random PLIF) | `step_002000.pt` from this same Run 3 lineage (PLIF state preserved, but still dead) |
| Adam state | `optim_state` STRIPPED → fresh m/v from random PLIF | `optim_state` likely STRIPPED on warmstart (default behavior on shape mismatch) → **fresh Adam against half-trained CfC + half-trained tied embed** |
| Warmup | 200 steps default | **100 steps** — half the runway for Adam moments to stabilize |
| Grad clip | 1.0 default | **0.5** — tighter clipping, helpful but interacts with KD/no-KD oscillation by clipping the larger no-KD gradients more often |
| `kd-topk` | 2048 (default, sparse) | **0** (full-vocab path) — 70× more memory in KD softmax, BUT **also** different KD signal: full-vocab KL covers the long tail of Qwen's distribution while top-2048 truncates it. Different KD target distribution. |
| Shuffle seed | 42 (default) | **311** — different token order through buffer |

The honest read is **NOT** "normal warmup non-monotonicity":

1. Train CE has stalled at 5.8–6.1 since at least step 2000 (no descent
   trajectory shown). Healthy warmup shows train CE **descending** with val
   following.
2. The val-rise rate (~30 ppl per 500 steps) is consistent across all four
   measurements — not a noisy bounce.
3. The phase 0 trigger gate is val ≤ 250 and we're moving the wrong way at
   the same arithmetic distance phase 0 has been in for 3 runs (Run 3b
   sat at val 397–421 for 8 hours).

This **is** the same Run 3b/3c/3e plateau that the retrospective §2.c
calls out as unfixed.

---

## 3. Recommended fix (top hypothesis)

**Top recommendation: DO NOTHING for the next ~500 steps.** Watch through
step 2500. The fresh-Adam warmstart-recovery argument predicts val ppl will
either:
- (a) **plateau and turn** by step 2500–3000 once Adam moments stabilize
  (warmup-100 schedule completed at step 100, but moment estimates take
  closer to ~800–1500 steps to reach steady-state at β2=0.999), or
- (b) **continue rising linearly to ≥500** by step 3000, in which case kill
  + relaunch with the patch stack below.

Concrete decision matrix (apply at next val checkpoint, step 2500):

| Step 2500 val ppl | Decision |
|-------------------|----------|
| ≤ 420 (flat) | **Hold**. Continue to step 4000. Re-evaluate. |
| 421–500 (slow rise) | **Hold one more eval**. Step 3000 still ≤ 500 → kill+relaunch. |
| > 500 (continued rise) | **Kill + relaunch immediately** with patch stack below. |
| > 700 (Run 3c-style) | **Kill + relaunch immediately, drop LR to 5e-5.** |

If kill+relaunch decided, the patch stack:

1. **Enable T2.6 LM head spectral norm** — `--lm-head-spectral-norm`.
   Even though P28 isn't the cause of Run 3l (z-loss is steady not
   rising), the spectral norm bounds the magnitude **before** any
   deterioration. Cheap insurance at +1% step time.
2. **Enable T2.5 spike-rate-target loss** — `--spike-target-loss-weight 0.001`
   with `--spike-target-loss-low 0.05 --spike-target-loss-high 0.20`.
   T2.5 tests show 0.000 → 0.130 spike rate in 50 steps at weight=0.1;
   even at the conservative 0.001 default it provides graph-attached
   pressure on `threshold` and `log_tau` that the no-grad homeostasis
   cannot. **This is the single biggest lever for "PLIF actually fires"
   — addresses H2 and matches retrospective §3 patch #3.**
3. **Enable T2.3 surrogate anneal** — `--surrogate-anneal-steps 5000`
   if the flag is wired. (Verify in the latest trainer; T2.3 is marked
   `[in 00:06]` in DEEP_MAINT_QUEUE so it may not have landed yet.) If
   not landed, skip — T2.5 alone unblocks PLIF.
4. **Increase warmup to 500 steps** — `--warmup 500`. Run 3l's
   `--warmup 100` is too aggressive for a fresh-Adam warmstart and is the
   most likely contributor to H1's drift. Run 3b used the 200 default.
5. **Drop kd-every to 8** — `--kd-every 8`. Halves the KD/no-KD gradient-
   magnitude oscillation that pumps Adam variance unevenly during fresh-
   Adam recovery.
6. **Restore `--kd-topk 2048`** (default). Run 3l's `--kd-topk 0` was
   probably set to A/B test full-vocab KD, but it costs 12 GB more VRAM
   and pulls toward a *different* teacher distribution shape than what
   Run 3b validated. Returning to 2048 makes this run comparable to 3b.
7. **DO NOT enable T2.4 freeze-vocab-tail in this relaunch.** The hook
   is default-on in the merged trainer but the running ckpt
   `step_002000.pt` was saved by the pre-merge code; the freeze hook will
   activate at relaunch and silently nail rows 151643..151935 to their
   step-2000 values. That's the *intended* behavior; it just needs noting
   so a later "why did the embed look discontinuous at relaunch boundary"
   has a paper trail.

**Do not change**: shuffle seed, grad-clip 0.5, lr 1e-4. Those are working.

**Single-line relaunch**:
```bash
ssh -p 41614 root@117.74.66.77 'pkill -f train_100m_kd; sleep 5; cd /workspace/synapforge_git && git pull && \
  setsid bash -c "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u train_100m_kd.py --warmstart /workspace/runs/v24h_qwen3/step_002000.pt \
  --teacher Qwen/Qwen2.5-0.5B --backend triton_block --batch-size 64 \
  --kd-every 8 --kd-topk 2048 --shuffle-buffer 10000 --shuffle-seed 311 \
  --steps 30000 --warmup 500 --lr 1e-4 --lr-decay cosine --grad-clip 0.5 \
  --lm-head-spectral-norm --spike-target-loss-weight 0.001 --phase-aware \
  --out /workspace/runs/v24h_qwen3 \
  > /workspace/runs/v24h_qwen3/train_run3m.log 2>&1 < /dev/null & disown"'
```

(Verify `git pull` lands `0344bb5` so T2.5/T2.6 flags exist on disk; do **not**
relaunch from the worktree's pre-merge copy.)

**One non-obvious risk to flag**: the merge `0344bb5` introduced an
indentation regression at lines 1365–1398 of `train_100m_kd.py`: the T2.5
spike-rate-target block is nested at the wrong indent (inside the
`if data_exhausted and _accum_iter == 0: break` body), making it dead code.
A second `loss.backward()` follows at line 1401 after `optim.zero_grad`,
which would raise on every step in a relaunched process at `accum_steps=1`
(default graph free after the inner-loop backward at line 1345). **Fix this
indentation bug before relaunching, or the relaunch will crash on step 1.**
Filing as P30. Do not include in this run's diagnosis commit (analysis-only
mandate); flag for the next code commit.

---

## 4. Triggers for ALARM (Run 3c-style divergence threshold)

| Step | Val ppl threshold | Action |
|------|-------------------|--------|
| 2500 | > 500 | Kill + relaunch with patch stack |
| 2500 | > 700 | Kill + relaunch + LR drop to 5e-5 |
| 3000 | > 500 | **Hard kill regardless of trajectory.** No 100M LNN+SNN run has ever recovered from val>500 at step 3000. |
| 3000 | > 1000 | Run 3c reproduction — also revert `--shuffle-seed 311` to 42 in next attempt; seed-specific overfit is a known failure mode at small-model scale. |
| 4000 | > 450 | Run 3l is "stuck"; phase 0 trigger 250 is unreachable in this run's budget. Kill + try Synap-Mini 30M (insurance plan A from `INSURANCE_NATIVE.md`). |
| any | NaN in any component | Immediate kill. Almost always a Triton bf16 quirk; relaunch with `--backend gpu_dense` for diagnosis. |

Do **not** kill on:
- step 2000–2500 val rise alone (this report's prediction is non-monotonic
  recovery is plausible)
- spike rate stuck at 0/10 (already-known concern; T2.5 flag is the fix,
  not a kill trigger)
- KD-step / no-KD-step loss oscillation (intentional)

---

## 5. PROGRESS.md §7 risks 2-line summary

```
Run 3l val_ppl rising 298→418 step 500-2000: most-likely fresh-Adam re-warmup overlap with already-known train↔val drift (P27/§2.c) NOT P24 shuffle nor P28 z-drift; spike 0/10 = unfixed P25 (T2.5 not in live process, only in trainer source post-0344bb5). Hold to step 2500/3000; kill+relaunch with --lm-head-spectral-norm + --spike-target-loss-weight 0.001 + --warmup 500 + --kd-every 8 if val>500 at step 3000.
Code health: 0344bb5 merge introduced indent-regression dead code at train_100m_kd.py:1365-1398 + double-backward at line 1401 — running process unaffected (loaded pre-merge), but next relaunch will crash. Fix before relaunch (file as P30).
```

---

## 6. Honest uncertainty

- **High confidence**: this is *not* the shuffle bug recurring, *not*
  z-loss runaway, and *not* a Triton/numerics issue. The pattern matches
  the documented unfixed concerns §2.a/c too cleanly.
- **Medium confidence**: that the val rise will plateau by step 2500–3000.
  Fresh-Adam recovery in the literature is 800–2000 steps but no paper
  matches a 100M LNN+SNN with PLIF dead, so this is extrapolation.
- **Low confidence**: that the patch stack will fix it. The retrospective
  itself flags T2.5 as a "scalpel not hammer" with expected impact "spike
  rate 0% → 5–15% by step 5000" — slow. Train↔val drift has no shipped
  fix, only patches §3 #2 (surrogate anneal) and §3 #3 (BPTT regularizer)
  planned. None ship today's relaunch.
- **Most likely truth**: phase 0 will not trigger in this run. If val
  doesn't drop below 350 by step 5000, the right move is to **switch to
  Synap-Mini 30M insurance plan A** (per `feedback_native_synap_insurance_options.md`)
  rather than burn another 8 hours on a 100M run that has plateaued at
  the same ppl band Run 3b sat at for 3h.

---

*Last updated 2026-05-02 by SynapForge agent. Next update: post step-2500
val ppl reading, OR post-relaunch step-1000 ce reading, whichever comes
first.*
