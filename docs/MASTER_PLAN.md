<!-- DOC_STAMP: STALE since 2026-05-01; check refs to synapforge/memory.py, synapforge/train.py, tests/test_neuromcp_button.py, tests/test_rfold.py, tests/test_skill_log_atomic.py -->
# SynapForge — Master Plan

**Updated**: 2026-05-01 (revision 8: + P22 RESOLVED -- phase_auto_relauncher.sh out-of-process watchdog, 6 tests passing)
**Owner**: Liu (mohuanfang@gmail.com)
**Status**: pre-investor-demo, training in progress on rental A800 80GB

> **READ THIS FIRST** every session. Update the **Current state**, **Active runs**,
> and **Open problems** sections as facts change. Memory entry
> [project_master_plan.md](.../project_master_plan.md) points future Claude here.

---

## 1. North star

A single sentence: **Synap-1 — a 100M LNN+SNN that chats in EN+ZH, learns at inference via STDP, grows synapses into the action space (NeuroMCP) instead of emitting tool-call tokens, and never hallucinates "training is going up" — every claim is backed by a runnable demo.**

**Naming**: framework = `SynapForge` (the forge); model = **Synap-1** (突触一号, the artifact). See [`NAMING.md`](NAMING.md) for the variant roadmap (Synap-1 / Synap-1-SFT / Synap-1-RL / Synap-Pro / Synap-Edge / Synap-Air).

**Investor-pitch reduction**: 5 differentiated claims (see `docs/INVESTOR.md`):
1. NeuroMCP synaptic growth replaces tool calling.
2. R-fold algebraic CfC — k reasoning steps per matrix solve.
3. Inference-time STDP — single-LOC unlock (forward-only Hebbian, no backprop).
4. 100M LNN+SNN with Qwen 151k vocab + Qwen 0.5B teacher KD.
5. Triple-path full-volume backup (no checkpoint will ever be lost again).

---

## 2. Top-level objectives (in priority order)

| # | Objective | Status | Gate |
|---|-----------|--------|------|
| O1 | 100M ckpt that chats in EN+ZH (no word salad) | in progress | val ppl ≤ 60 + chat eval ≥ 60% pass |
| O2 | All 5 INVESTOR.md claims have runnable demos that **execute clean** | partly | feature audit agent — see §6 |
| O3 | Triple-backup pipeline never drops a best ckpt | wired, daemon running | tripwire flags 5 empty cycles |
| O4 | Self-learn + curiosity activate **automatically** at the right time | mechanism done, gate decided | see §4 |
| O5 | Multi-modal (image+audio+time-series) byte-patch path | code done, untrained | phase 2 trigger ppl ≤ 100 |
| O6 | RL post-training (GRPO + sympy verifier on math) | not started | phase 4 gate, chat ≥ 60% |
| O7 | Plan C insurance: LoRA Qwen 0.5B as backup demo | code done | run ≥ once on rental, ≥ 60% chat |
| O8 | Honest evaluation pipeline (no fake curves) | wired, daemon running | every ckpt = 5 EN + 5 ZH samples |
| O9 | NeuroMCP universal codebook (L1/L2/L3, never lose mints) | code done | atomic skill log + reload test |
| O10 | **50M effective context** with linear inference cost | partial: long.py + InfLLM L1/L2/L3 wired, untested at 50M | benchmark: 1K → 10K → 100K → 1M → 10M → 50M latency linear, ppl drift < 5% |
| O11 | **Quality monotonic with context length** (longer = better, not worse) | thesis: inference-time STDP + retrieval, not yet validated | A/B: same prompt at 1K vs 50K context; quality score must NOT regress |

---

## 3. Phased training plan (gate by ppl, NOT by step count)

Source of truth: `scripts/phase_manager.py` (auto-watches `train.log`, writes `.phase` flag file).

| Phase | Trigger | Adds | Notes |
|-------|---------|------|-------|
| 0 | warm start, ce ≈ 9.6 | LM-only KD from Qwen 0.5B teacher | current state at v24h_qwen |
| 1 | val ppl ≤ 250 | `--self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05 --phase-aware` | turn on TTT replay + curiosity (real flags; argparse-validated 2026-05-01) |
| 2 | val ppl ≤ 100 | `--modal-list image,audio,time_series` | needs base linguistic competence first or cross-modal aux destroys it |
| 3 | val ppl ≤ 60 | alpaca-zh SFT + response-only-loss, lr 1e-4 | the chat-emergence threshold |
| 4 | chat eval ≥ 60% pass | GRPO RL with sympy verifier (gsm8k math) | post-SFT polish |

**Why ppl-gated, not step-gated**: prevents premature feature activation that cripples the
backbone (we've burned 2× by enabling cross-modal aux too early — see
`feedback_phased_training_2026q2.md`). The phase manager rewrites the trainer launch script
when a threshold trips, so the human just restarts.

**Honest target**: phase 3 at ppl ≤ 60 in 24 GPU-h is aspirational; ppl ≤ 100 is conservative
expectation. If phase 1 doesn't trip in 8h of training, fall back to Plan C.

---

## 4. When to activate self-learn + curiosity

**Decision**: phase manager fires phase 1 at **val ppl ≤ 250**. Rationale:

- Empirical evidence (`feedback_consensus_needs_good_base.md`): below ppl ~500 the model
  spits high-frequency word salad. Curiosity reward signals on a confused base
  amplify noise (random outputs look "novel" but carry no semantics).
- ppl ≤ 250 means the model has learned distributional structure (n-gram + minimal syntax).
  Curiosity now picks up *meaningful* novelty — new tokens it hasn't predicted, not random.
- Self-learn TTT picks the top-K val samples by CE and trains 1 inner step on them at val
  time. Below ppl 250 every sample looks high-CE and TTT becomes random updates.
- STDP novelty signal (||ΔW_STDP||) is robust to noise (random pre/post averages to ~0)
  but its absolute magnitude is meaningless until the network has a structured weight matrix.

**Concretely** (audit 2026-05-01 confirmed wiring is correct, KD compose is good):
- Curiosity weight ramps 0 → 0.05 over **1500 steps** (not 1k — Run 1 history says ramp slower).
- Self-learn TTT: `--self-learn-k 8`, 1 inner step, val-time only, weights restored.
- STDP-novelty signal: **deferred** — no autograd path yet. Bookkeeping-only via
  `||ΔW_STDP||` in `trainer_mixins.py:540-543`. Wire it before claiming the curiosity
  formula's `0.25·||ΔW_STDP||` term is live.
- **Concrete activation flag string** (when val ppl ≤ 250, two consecutive evals):
  ```
  --self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05 --phase-aware
  ```
  Do **not** pass `--intrinsic-curiosity` or `--stdp-novelty` (no such args; argparse fails).
  phase_manager.py was patched 2026-05-01 to match.

**Pre-activation gate (multi-signal, not just ppl)**:
1. val ppl ≤ 250 (two consecutive evals — anti-noise).
2. CE std over last 100 steps < 0.5 (no divergence event).
3. Mean spike rate ∈ [0.05, 0.20] (PLIF healthy, not dead, not saturated).
4. Step ≥ 2× warmup (predictor MLPs need warmup before `delta_F` gradient is meaningful).

**Failure modes to watch**:
1. Curiosity drives the model toward tokens *the teacher* hasn't predicted → KD breaks.
   Mitigation: clamp curiosity reward to NOT counteract KD KL.
2. Self-learn TTT converges to overfitting on val set → val ppl drops, real perf drops.
   Mitigation: hold out a *secondary* val set never seen by TTT.
3. STDP novelty + multimodal both at once = backbone has too many objectives. Stagger.

**Curiosity formula** (from `feedback_curiosity_stdp_formula.md`):
```
C = 0.40·ΔF + 0.25·||ΔW_STDP|| + 0.15·G_HNSW + 0.10·H[spike] + 0.05·N - 0.05·V_noise
```
Already in `synapforge/trainer_mixins.py::CuriosityMixin`. Activate via flag, not hardcode.

---

## 5. Final-product feature checklist

User: *"训练完得到的成品该有的功能都得有,都要能正常使用"*

Each feature must (a) have working code, (b) have a smoke test, (c) survive a fresh-clone
verify-pipeline run. Feature audit agent (see §6) will check **(c)** end-to-end.

| Feature | Code | Smoke test | Demo CLI | Status |
|---------|------|------------|----------|--------|
| EN+ZH chat | `train_100m_kd.py` + `chat.py` | `synapforge-demo chat` | `cli.py:chat` | training |
| NeuroMCP synaptic growth | `action/universal_codebook.py` | `tests/test_neuromcp_button.py` | `cli.py:button` | done |
| R-fold k-step CfC | `cells/rfold.py` | `tests/test_rfold.py` | `cli.py:bench` | done |
| Inference-time STDP | `bio/stdp_fast.py:121` (gate removed) | `synapforge-demo stdp` | `cli.py:stdp` | done |
| Multimodal (img/aud/ts) | `synapforge/modal/` + `MultimodalMixin` | `scripts/synth_multimodal_smoke.py` | needs phase-2 ckpt | code done, untrained |
| Self-learn TTT | `SelfLearnMixin` | TBD | needs phase-1 ckpt | code done, untrained |
| Curiosity (6-signal) | `CuriosityMixin` | TBD | needs phase-1 ckpt | code done, untrained |
| Computer-use (web auto-learn) | `synapforge/action/web_actuator.py` | `scripts/web_actuator_smoke.sh` + `tests/integration/test_web_actuator.py` | post-phase-3 | ✓ MVP (P18 resolved 2026-05-01) |
| Skill persistence | `action/skill_log_v2.py` (atomic, durable) | `tests/test_skill_log_atomic.py` | n/a (background) | done |
| Triple-backup | `scripts/triple_backup_daemon.py` | tripwire 5-cycle warn | daemon | running |
| Honest eval pipeline | `scripts/auto_eval_daemon.py` | per-ckpt 5+5 chat | runs on every ckpt | running |
| Phase manager | `scripts/phase_manager.py` | log → `.phase` JSON | daemon | running |
| Plan C LoRA Qwen | `scripts/train_qwen_lora.py` | `--smoke` + `--print-only` | `qwen_lora_chat_repl.py` | code done, smoke verified, real run pending (see `PLAN_C_RUNBOOK.md`) |

---

## 6. Open problems (live list — UPDATE AS YOU LEARN)

### P1. Run 1 word-salad divergence at step 5000 — RESOLVED 2026-05-01
- **Symptom**: Run 1 (April) ce 5.72 → 8.46 between step 4500–5000, high-freq token
  salad. Run 2 (May 1) had dead 10/10 spike rate from step 250 → 13250 (entire run) —
  threshold drifted *up* past the input distribution and never came back.
- **Diagnosis (Agent 1+2 on 2026-04-30)**: PLIF threshold drift unclamped + wrong PLIFCell
  variant (RC leaky-to-steady) that never fires with threshold=0.3 on tanh-bounded input.
  Trainer's prior `auto-revive` (`step >= 1000 and step % 200 == 0 and n_dead == len(rates)`)
  only kicked in when ALL cells were dead, so partial death never triggered.
- **Fix applied**: 4-part patch landed 2026-05-01 (commit `P1: PLIF homeostatic threshold
  control + EMA clamp`):
  1. `synapforge/surrogate.py` — PLIFCell gained `clamp_threshold(min_val=0.005,
     max_val=0.5)` and `homeostatic_step(observed_rate, target_rate=0.10, gain=0.005)`.
     Both run under `torch.no_grad()` so they are OFF the autograd path — the threshold
     parameter still learns normally via the surrogate gradient through `v - threshold`.
  2. `train_100m_kd.py:732+` — replaced the all-dead `[PLIF-REVIVE]` block with the
     stronger schedule: `homeostatic_step` every 50 steps (gain=0.01), `clamp_threshold`
     every 100 steps. Logs print before/after threshold mean only when a change fires.
  3. New flag `--no-plif-homeostasis` (default OFF, i.e. homeostasis ON) for A/B disabling.
  4. `tests/integration/test_plif_homeostasis.py` (6 tests, all CPU): toy 2-block model
     drives threshold from 0.5 → <0.10 under sustained 0% spike rate, clamp enforces
     min/max, in-band rates are no-ops, methods don't pollute autograd state.
- **Earlier mitigations preserved**: `model_100m.py` plif_threshold 0.3 → 0.05, tau 1.5
  → 2.5 (spike rate rose 0.020 → 0.335).

### P2. KD OOM at vocab=151k bs=128 — RESOLVED 2026-05-01
- **Symptom**: 18.51 GiB intermediate alloc on KL at bs=128 / vocab=151936
  (worked around earlier by dropping bs to 64).
- **Fix applied (Phase 1)**: chunked KL over batch dim (÷4).
- **Fix applied (Phase 2, P13 close)**: replaced fixed `chunk = bs // 4` with
  `_kd_chunk_size(batch_size, seq_len, vocab, headroom=0.5)` that reads
  `torch.cuda.mem_get_info()` and sizes chunks to fit the `(B*T, V, fp32)`
  intermediate in 50% of free VRAM. Floors at 1, caps at `batch_size` so
  large-VRAM cards see no behavior change. New CLI `--kd-chunk N` (default 0
  = auto) for explicit override. Banner `[kd] chunk={N} (free={X}GB,
  vocab={V})` printed once on first KD step.
- **Verification**: `tests/integration/test_kd_chunk_autotune.py` (6 tests, all
  passing locally on Windows dev box) covers abundant / tight / near-empty
  VRAM, CPU fallback, headroom scaling, and the `chunk_override` plumbing
  through `_kd_loss`.

### P3. Self-learn TTT and val set leak — RESOLVED 2026-05-01
- **Risk**: TTT inner step on val sample = val ppl drops artificially because
  the trainer just trained on the same examples it then evaluates against.
- **Fix applied**: row-level deterministic 80/20 split of the val parquet
  stream into `val_ttt` and `val_holdout` (`synapforge/data/__init__.py::
  split_val_stream` -- `denom=5`, `ttt_fraction=0.8` puts every 5th chunk
  on the holdout side, the other 4 on the TTT side; sets are disjoint and
  their union equals the full parent). `train_100m_kd.py` builds both
  streams once after `val_ds` (no extra parquet IO), runs `evaluate(...)`
  on each, and logs `VAL step N: val_ppl_ttt=X val_ppl_holdout=Y (honest)`
  every `EVAL_EVERY` steps. Both numbers persist in `metrics.json` as
  `ppl_eval_ttt` / `ppl_eval_holdout`; `ppl_eval` is kept as an alias of
  the holdout (honest) number for legacy parsers. `--self-learn-ttt` only
  ever sees `val_ds_ttt`, so `val_ppl_holdout` is leak-free. New CLI:
  `--ttt-val-fraction` (default 0.8). `scripts/phase_manager.py` added
  `VAL_PPL_HOLDOUT_RE` and now gates phase 1 (ppl <= 250) on the holdout
  number when present, with backward-compat fallback to the pre-P3
  generic `val.*?ppl` regex for legacy logs.
- **Verification**: `tests/integration/test_ttt_val_split.py` -- 7 passed
  (1 skipped because `transformers` not installed; that test is the
  end-to-end real-parquet path). Asserts disjointness, union completeness,
  4:1 ratio, keep-indices metadata, and bad-fraction validation.

### P4. Curiosity reward vs. KD KL conflict
- **Risk**: curiosity wants tokens teacher didn't predict; KD penalizes those.
- **Status**: not yet observed in training. Mitigation TBD (clamp curiosity reward, or
  schedule curiosity to ramp up only as KD weight ramps down).

### P5. Plan C LoRA Qwen — DELETED 2026-05-01 (architecture-claim violation)
- **Decision**: 2026-05-01, **all Plan C / Qwen-LoRA code removed**. Reason: a
  transformer-base + LoRA-adapter "insurance demo" would invalidate the LNN+SNN
  architecture claim and make the paper unsubmittable to NeurIPS / ICLR / ICML.
  See `docs/ANTI_LORA.md` for the full strategic argument.
- **Files deleted**: `scripts/{launch_plan_c_cpu.sh,train_qwen_lora.py,
  qwen_lora_chat_repl.py}`, `synapforge/demo/qwen_lora_demo.py`,
  `docs/{PLAN_C_RUNBOOK.md,PLAN_C_QWEN_LORA.md,PLAN_C_CPU_NOTES.md}`. Plus the
  `qwenchat` subcommand removed from `synapforge/demo/cli.py` and the
  `_is_plan_c_ckpt` route removed from `scripts/chat_eval_gate.py`.
- **Replacement insurance path** (still pure Synap-1 / LNN+SNN):
  - **Option A**: 30M-50M Synap-1 (smaller param count, shorter run)
  - **Option B**: replay healthy v4.x ckpt via `chat_recorded.json` with honest disclosure
  - **Option C**: pivot demo focus from live chat to mechanism-level (NeuroMCP / R-fold / STDP)
- **Status**: RESOLVED-by-deletion. `docs/ANTI_LORA.md` is the load-bearing doc.

### P6. NeuroMCP density saturation at ~28% — RESOLVED 2026-05-01
- **Status**: empirically observed in v4.1 runs. Not a bug — might be the natural sparsity.
- **Open question**: does saturation hurt skill recall or help (parsimony)?
- **Required**: long-running NeuroMCP smoke (≥ 600 trials) to prove K grows, not saturates.
- **Fix applied**: 600-trial reproducible test wired —
  `tests/integration/test_neuromcp_long_horizon.py`. Calls
  `synapforge.demo.four_button.run_demo(n_trials=600)` (the same harness
  as `synapforge-demo button`, just with longer horizon), captures
  `(density, K, hit_rate)` at milestones `[50, 100, 200, 400, 600]`,
  and asserts (a) density at 600 >= 0.18, (b) K at 600 >= 11, (c)
  density and K monotonically non-decreasing across milestones, (d)
  mean hit-rate over last 8 trials >= 0.95. `@pytest.mark.slow` so
  default `pytest` skips it. Total runtime: ~7s on Windows CPU.
- **Measured numbers (Windows CPU dev box, 2026-05-01, seed=7)**:
  - milestone 50:  density=0.0669 K=10 hit_rate=1.000
  - milestone 100: density=0.0815 K=11 hit_rate=1.000
  - milestone 200: density=0.1038 K=11 hit_rate=1.000
  - milestone 400: density=0.1501 K=11 hit_rate=1.000
  - milestone 600: density=0.1973 K=11 hit_rate=1.000
  Across seeds [7, 11, 42, 123]: density in [0.1865, 0.1973], K in
  [11, 13], hit_rate constant at 1.000.
- **INVESTOR.md update needed**: the *"~28% density and K=14 at ~600
  trials"* claim in INVESTOR.md §"NeuroMCP" overstates the smoke-config
  ceiling. Measured density tops out at ~0.20 (8pp short of 28%) and K
  at 11-13 (1-3 short of 14) on the default 4-button env. Two options:
  (1) soften the INVESTOR.md numbers to *"density grows from ~6% to
  ~20% over 600 trials, codebook K grows 9 → 11-13"*, or (2) tune the
  smoke env / agent to actually reach 28% (e.g. raise
  `synapse_max_density` cap from 0.40 → 0.50, raise `growth_step`
  0.005 → 0.01, or run more trials). Recommend option (1) for honesty
  unless there is empirical evidence from v4.1 rental runs that 28% is
  reachable on this exact env at 600 trials. Tracking action: Liu to
  reconcile INVESTOR.md numbers vs measured before pitch.
- **Run** the test any time:
  `pytest tests/integration/test_neuromcp_long_horizon.py --run-slow -v -s`

### P7. Computer-use sandbox not yet exercised — RESOLVED 2026-05-01
- **Symptom**: P18 shipped `web_actuator.py` MVP (340 LOC) with mock-Playwright
  unit tests passing but no real Chromium had ever been driven by ActionHead.
  The "AI uses neurons to drive computer" claim was a code path, not a demo.
- **Fix applied (2026-05-01)**: real Playwright sandbox run on this Windows box.
  - `scripts/web_actuator_real_smoke.py` (NEW, ~210 LOC) — boots headless
    Chromium against `file://.../static_demo.html`, builds `nn.Linear(64, 64)`
    random-init ActionHead, runs 100 steps with `encode_dom() + 0.05*N(0,I)`
    + periodic action-bias nudges, asserts >= 1 successful click, >= 1
    nav/scroll, no uncaught exceptions, runtime <= 60 s. Saves screenshot
    at step 50 + per-step JSON trace.
  - Evidence persisted under `synapforge/tests/fixtures/p7_evidence/`:
    `web_actuator_smoke.png`, `web_actuator_smoke_trace.json`,
    `web_actuator_smoke_summary.json`.
  - **Real run results** (2026-05-01, headless Chromium 1208 already in
    `$LOCALAPPDATA/ms-playwright/`, launched via `executable_path=...`):
    100/100 steps, **6.97 s wallclock**, action histogram
    `{noop:73, click:22, scroll:0, type:1, navigate:4}`, **22 ok_clicks**,
    4 successful navigates, screenshot saved, no uncaught exceptions.
  - Note: matching chromium-1140 download failed `ENOSPC` (C: had
    400 MB free); reusing the existing 1208 binary was the unblock.
- **Out of P7 scope**: vision pipeline, multi-tab, login flows, CAPTCHA,
  adversarial-input filter, real-site curriculum (those keep living in
  `web_env.py` / `train_neural_web.py` and resurface in O2 phase D).

### P8. MCP shell + bg job interaction on rental — RESOLVED 2026-05-01
- **Symptom**: MCP `proc_exec` kills nohup'd children. Repeated session expirations.
  Prior workaround (`setsid bash -c '... < /dev/null' </dev/null & disown`) survives
  SSH/MCP exit but lacks (a) auto-restart on crash, (b) clean kill via systemctl,
  (c) status check via systemctl.
- **Fix applied**: 3-file watchdog suite landed 2026-05-01.
  1. `scripts/launch_train_systemd.sh` (NEW, ~140 LOC bash, `bash -n` clean) —
     wrapper that prefers `systemd-run --user --unit=$NAME` (transient unit with
     `systemctl status/stop` + `journalctl -f`) and falls back to
     `setsid+disown` with a stderr WARNING when user-systemd is unavailable.
     CLI: `--name UNIT --warmstart PATH --out DIR [--steps N] -- <trainer flags>`.
     Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
     TOKENIZERS_PARALLELISM=false` in both branches. Prints PID, log path, and
     the four management commands (status/follow/stop/install-template) for the
     chosen branch.
  2. `scripts/synapforge-trainer.service.template` (NEW, ~70 LOC) — reference
     systemd template unit with `Restart=on-failure RestartSec=30
     StartLimitBurst=5 StartLimitIntervalSec=600`. Reads per-instance
     `TRAINER_ARGS` from `~/.config/synapforge-trainer/%i.env`. Comment block at
     top has full install instructions: copy → `daemon-reload` → drop env file →
     `systemctl --user enable --now synapforge-trainer@v24h_qwen3.service`.
  3. `docs/RENTAL_OPS.md` (NEW, ~150 LOC) — runbook covering when to use
     setsid+disown / systemd-run / template-unit, with a 3-row decision table,
     daily inspection commands (`list-units` / `status` / `journalctl -f` /
     `journalctl -p err`), and cleanup commands (`stop` / `disable` /
     `reset-failed` / `vacuum-time`). Memory cross-refs to
     `feedback_mcp_remote_ssh_quirks.md` + `feedback_mcp_nohup_hangs_use_systemd_run.md`
     + `feedback_no_polling_loops_even_bg.md`.
- **Verification**: `bash -n scripts/launch_train_systemd.sh` returns 0. Not yet
  exercised on rental (intentional — this commit is repo-side only; rental
  rollout happens at next training restart).

### P9. Investor demo claim parity (data sources) — smoke test ready, runs in CI ≤5min
- **Status**: 4 download scripts written (pretrain/sft/multimodal/eval) but not yet
  fully executed end-to-end on rental.
- **Risk**: claim "EN+ZH" hinges on alpaca-zh + Chinese pretrain mix actually being on disk.
- **Action**: a smoke run that actually downloads → mixes → trains 50 steps must be in CI
  before pitch.
- **Fix applied (2026-05-01)**: E2E smoke test wired —
  `tests/integration/test_data_pipeline_smoke.sh` runs the full chain
  `synth_chinese_pretrain.py --n 200` → `mix_pretrain_corpora.py --corpora <synth>`
  → `train_100m_kd.py --steps 50 --backend gpu_dense` on a 1.5M-param tiny model.
  Asserts: (a) ckpt `step_000050.pt` exists, (b) ckpt has the P12 `config` dict
  with the exact d/n_layers/vocab/max_seq passed on the CLI, (c) `train.log`
  contains a `step 50 loss=...` line, (d) ce decreases over the 50 steps.
  Cleans up tmp dir on PASS, leaves it on FAIL for inspection. Runtime
  budget ≤5min on CPU. Trainer gained CLI overrides for data globs, tokenizer,
  save/eval/log cadence, and architecture (vocab/d/n_layers/loop_depth/
  ffn_ratio/sparsity/lr/seq_len) plus a `--no-warmstart` flag so the smoke is
  hermetic. `mix_pretrain_corpora.py` gained a `--corpora <p1>,<p2>,...` flag
  for direct parquet inputs (no per-corpus directory layout required).
  Shortcut: `bash scripts/data-smoke.sh`.

### P10. README + docs lag the code — RESOLVED 2026-05-01
- **Symptom**: ~41 docs in `docs/` and several pre-date major refactors
  (e.g. `MULTIMODAL_TRAINING.md` predates byte-patch refactor; `INVESTOR.md`
  recently had a vocab number wrong). No way to detect future stale docs.
- **Fix applied**: Lightweight doc-stamp instrumentation (NOT content cleanup).
  - `docs/_stamp.json` (NEW): per-doc `{last_verified_sha, last_verified_date, verifier}`.
    Bootstrapped 2026-05-01 with each doc's last-modify sha + verifier `agent_p10`.
  - `scripts/check_doc_stamps.py` (NEW, ~280 LOC): compares doc-modify-sha against
    stamp; greps the doc for backtick-quoted file paths / `synapforge/foo.py` /
    markdown links / `:func:` Sphinx roles; flags when a referenced code file was
    modified after `last_verified_date` (MAYBE STALE) or no longer exists (STALE).
    Outputs a markdown table; exit 0 = clean, exit 1 = at least one STALE.
    `--update-stamps` bumps sha for auto-fresh docs only — never auto-flips
    MAYBE STALE → fresh. Pure Python; no `find`/`xargs`.
  - `tests/integration/test_doc_stamps.py` (NEW): 6 tests (`pytest.mark.docs`)
    that assert the script returns 0 or 1 with a parseable 3-column markdown
    table. Doesn't fail CI for STALE — only for unparseable output.
  - First run flagged **9 STALE docs** (refs to deleted files: `train_v42_universal.py`,
    `synapforge/memory.py`, etc.). Each STALE doc got a one-line
    `<!-- DOC_STAMP: STALE since 2026-05-01; check refs to ... -->` annotation
    at the top — content NOT rewritten.
- **Run** the checker any time: `python scripts/check_doc_stamps.py`.

### P11. Vocab 151643 vs 151936 mismatch — RESOLVED 2026-05-01
- **Symptom**: 7 code sites hardcoded 151643 (real Qwen tokenizer vocab); 5 doc sites said 151936
  (Qwen 2.5 model embedding padded dim). KD teacher emits 151936 logits — student at 151643
  meant logit shape mismatch every step.
- **Fix applied**: all 7 code sites patched to 151936 (`train_100m_kd.py:284,354`,
  `train_100m_sft.py:7,63`, `chat_repl.py:39,148`, `chat_demo.py:71`). Tokenizer still emits
  IDs in [0, 151643); rows 151643-151935 of embedding are unused but reachable for malformed
  inputs (no OOB).
- **Open**: Run 2 (PID 13663) was launched before this patch. Restart with patched code at
  next ckpt boundary. **Do NOT** lose Run 2's progress — warmstart from latest best ckpt.

### P12. `chat_demo` strict=False silently swallows shape mismatch — RESOLVED 2026-05-01
- **Symptom**: `synapforge/demo/chat_demo.py:69-82` calls `load_state_dict(strict=False)`.
  If a future ckpt's `d`/`n_layers`/`ffn_ratio` differs, demo loads garbage.
- **Fix applied**: Both trainers (`train_100m_kd.py`, `train_100m_sft.py`) now persist a
  full architecture-config dict via the `_build_config_dict(args)` helper into every
  `torch.save({"model": ..., "config": {...}})` call (best/periodic/final/phase-aware).
  The `config` carries `vocab/d/n_layers/loop_depth/max_seq/ffn_ratio/sparsity/dropout/tie_lm_head`.
  Loaders (`synapforge/demo/chat_demo.py::_try_load_live` + `scripts/chat_repl.py::load_model`)
  read `ckpt["config"]` first; if absent (legacy ckpts) they fall back to the documented
  defaults and emit `[chat_demo] WARNING: ckpt has no config; using fallback values ...`.
  After `load_state_dict(strict=False)` they also warn when `len(missing)+len(unexpected) > 5`,
  so silent shape drift surfaces visibly. Round-trip integration test:
  `tests/integration/test_ckpt_config_roundtrip.py` (2 tests, both passing).
- **Backwards-compat**: Run 2's existing rental ckpts (no `config`) still load via the
  fallback path — verified by `test_legacy_ckpt_no_config_falls_back`.

### P13. KD chunk size hardcoded `bs // 4` — RESOLVED 2026-05-01
- **Symptom**: `train_100m_kd.py:318` chunk = bs // 4. At bs=128 still 3.7GiB intermediate.
  Won't scale to smaller cards.
- **Fix applied**: same patch as P2 Phase 2. New helper `_kd_chunk_size`
  (`train_100m_kd.py` near `_kd_loss`) reads `torch.cuda.mem_get_info()` and
  picks `chunk = max(1, min(bs, free_b * 0.5 // (seq * vocab * 4)))`. CPU
  branch keeps the legacy `bs // 4` for determinism. Wired into `_kd_loss`
  via `chunk_override` (defaults to 0 = auto), called from the train loop at
  `train_100m_kd.py:793`. CLI `--kd-chunk N` lets ops force a fixed value;
  `--kd-chunk 0` (default) = auto-tune.
- **Tests**: `tests/integration/test_kd_chunk_autotune.py` -- 6 tests passing,
  covers abundant / tight / near-empty VRAM, CPU fallback, headroom scaling,
  and `_kd_loss` chunk_override path.
- **Severity**: was nice-to-have on A800 but real on smaller cards; now
  closed alongside P2.

### P14. `parallel.py` Layer 2 (`place_mixed_device`) is orphan code — RESOLVED 2026-05-01
- **Symptom**: Documented as a 3-layer feature; only `examples/mixed_device_training.py`
  called it; zero test coverage. Trainer uses only Layer 3 (DDP).
- **Fix applied**: smoke test added — `tests/integration/test_parallel_mixed_device.py`
  (4 tests). Verifies (a) CPU-only path produces a valid `MixedPlacement` with
  `cpu_param_count > 0`, (b) docstring's "~40% off GPU" claim holds (toy hits
  ~50%, above 30% lower bound), (c) default `cpu_module_names`
  (`embed_tokens`, `lm_head`, `lm_logits`) place >95% of toy params on CPU,
  (d) GPU path (`@pytest.mark.gpu`) places backbone on cuda:0 while
  embed_tokens stays on cpu. All 4 pass on CPU-only Windows dev box (4.82s).
  Docstring softened: 375M arithmetic now flagged as expected-not-measured.
- **Status**: smoke tested, not benchmarked. End-to-end VRAM-savings claim on
  real 375M shape still requires a rental run; deferred — function not on a
  hot path for current single-A800 training.

### P15. `tests/` collects from two roots — RESOLVED 2026-05-01
- **Symptom**: 15 `test_*.py` files inside `synapforge/` package + a separate `tests/` dir.
  Pytest collects both. On a torch-less box collection of `synapforge/test_distributed_smoke.py`
  raises ImportError immediately.
- **Fix applied**: repo-root `conftest.py` adds `collect_ignore_glob = ["synapforge/test_*.py"]`.
  In-package smoke scripts remain runnable manually via `python -m synapforge.test_xyz`.
  Reversible — delete `conftest.py` to restore prior collection behavior.

### P16. 12 trainer entry points, README ambiguous — RESOLVED 2026-05-01
- **Symptom**: top-level had `train_100m.py`, `train_100m_kd.py`, `train_100m_sft.py`,
  `train_3d.py`, `train_full_modal.py`, `train_multimodal.py`, `train_v15_full.py`,
  `train_v16_unified.py`, `train_v18_full_self.py`, `synapforge/train.py`,
  `train_native_unified.py`, `train_v42_universal.py`. Plus `.bak` files (P17).
- **Fix applied**: 10 legacy trainers `git mv`-ed to `legacy/` (history preserved):
  `train_100m.py`, `train_3d.py`, `train_full_modal.py`, `train_multimodal.py`,
  `train_v15_full.py`, `train_v16_unified.py`, `train_v18_full_self.py`,
  `train_native_unified.py`, `train_v42_universal.py`, `synapforge_train.py`
  (was `synapforge/train.py`). `legacy/README.md` lists each with its replacement.
  Repo-root `train_100m_kd.py` (phase 0/1 pretrain + KD) and `train_100m_sft.py`
  (phase 3 instruction-tune) are now the only canonical entries; README "How to
  train" rewritten to point at them and reference §3 phase gates.

### P17. `.bak` files committed in source tree — RESOLVED 2026-05-01
- **Symptom**: `synapforge/train_100m.py.bak`, `synapforge/__init__.py.bak_pre_action`,
  `synapforge/cells/synapse.py.bak_pre_mfu_opt1`.
- **Fix applied**: all 3 `git rm`-ed. Rollback points remain in git history.

### P18. `web_actuator.py` claimed but missing — RESOLVED 2026-05-01
- **Symptom**: MASTER_PLAN.md §5 row "Computer-use" claimed `synapforge/action/web_actuator.py`.
  File did not exist. (User directive: 让ai使用神经元直接操控computer上网自动学习.)
- **Fix applied**: MVP shipped per §12. Files added:
  - `synapforge/action/web_actuator.py` — `WebActuator` class (DOM-only, Playwright
    headless, maps ActionHead hidden vector → `{noop, click, scroll, type, navigate}`).
  - `synapforge/tests/fixtures/static_demo.html` — 3-button + input + link fixture.
  - `scripts/web_actuator_smoke.sh` — 50 random ActionHead steps, asserts ≥ 1 click.
  - `tests/integration/test_web_actuator.py` — 9 unit tests, Playwright-mocked, all pass.
  - `pyproject.toml` — new `[web]` extra (`playwright>=1.40`).
  - `docs/NEURAL_COMPUTER_USE.md` §11 — install / usage / action-space docs.
  Also patched `synapforge/__init__.py` `from .train import train` to be defensive
  (legacy `train.py` was relocated by P15+P16+P17 cleanup; the hard import was missed).
- **Verification**:
  - `python -c "from synapforge.action.web_actuator import WebActuator"` exits 0
    on a torch-installed, Playwright-less box.
  - `pytest tests/integration/test_web_actuator.py -v` → 9 passed in 4.6s.
- **Out of MVP scope** (per §12): vision pipeline, multi-tab, login flows, CAPTCHA,
  real-site curriculum (those keep living in `web_env.py` / `train_neural_web.py`).

### P19. Phase manager flags out of sync with trainer argparse — RESOLVED 2026-05-01
- **Symptom**: `phase_manager.py:45` listed `--intrinsic-curiosity --self-learn-ttt --stdp-novelty`
  for phase 1. Only `--self-learn-ttt` and `--curiosity-weight` exist in trainer. Relauncher
  would crash on argparse error.
- **Fix applied**: phase 1 flags now `--self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05`.
  Phase 2 dropped `--modal-byte-patch --cross-modal-contrastive` (not in argparse), kept
  `--modal-list image,audio,time_series`.
- **Open**: STDP-novelty signal needs autograd path before it can become a flag. Defer.

### P20. 50M effective context — harness ready, awaiting Run 2 ckpt
- **Symptom**: User directive 2026-05-01: *"50m的有效上下文别忘了"*. Code paths exist
  (`synapforge/long.py` (NEW shim) + `synapforge/infinite.py` (5-tier memory) +
  InfLLM L1/L2/L3 retrieval), but no run has ever tested at 50M tokens. Memory
  `reference_a100_80gb_context_max_2026.md` claims 50M is the qualitative ceiling.
- **Risk areas**: (a) PLIF spike retrieval at 50M — does spike-pattern recall scale?
  (b) CfC τ saturation in 4-32K range — multi-band τ fix wired? (c) STDP weight saturation
  after 1M tokens — needs forgetting term or homeostatic decay.
- **Status (2026-05-01)**: validation harness written —
  `tests/integration/test_long_context_50m.py` parametrizes over
  `[1024, 10K, 100K, 1M, 10M, 50M]`, asserts latency-per-token < 1.5x baseline,
  ppl drift < 5%, STDP weight L2 norm < 10x growth. Streams chunks via
  `synapforge.long.chunked_token_stream` so 50M never materializes in RAM.
  Self-skips L >= 1M when CUDA total memory < 70 GB. Lengths >= 100K are
  `@pytest.mark.slow`. Run via `scripts/run_long_context_validation.sh`
  on rental once Run 2 ckpt lands.
- **Severity**: pre-pitch-required (it's a headline claim). **Harness ready; awaiting Run 2 ckpt.**

### P21. Quality MUST grow monotonically with context length — harness ready, awaiting Run 2 ckpt
- **Symptom**: User directive 2026-05-01: *"性能和质量要随着上下文的增长而增长"*.
  This is the inference-time STDP thesis (`feedback_inference_stdp_unlock.md`,
  `bio/stdp_fast.py:121` gate already removed) — longer context = more Hebbian
  updates = better adaptation.
- **What it means concretely**: at 1K context, ppl X. At 10K context, ppl ≤ X (NOT ≥ X).
  At 100K context, ppl ≤ X. This is the OPPOSITE of every transformer (which degrades
  monotonically past trained ctx).
- **Status (2026-05-01)**: A/B harness written —
  `tests/integration/test_long_context_monotonic_quality.py`. Builds a
  needle-in-haystack prompt at L in `[1024, 10K, 100K]`, runs once with
  `synapforge.long.set_stdp_inference("on")` and once with `"off"` (model
  rebuilt from same seed for branch isolation), scores by exact-match argmax
  averaged over 3 trials. Asserts `acc_on >= acc_off - 0.05` at every L plus
  bonus `acc_on(100K) >= acc_on(1K) - 0.10` (no quality collapse). All asserts
  behind `@pytest.mark.slow + @pytest.mark.gpu`.
- **Severity**: pre-pitch-required (this is THE differentiator vs transformer).
- **Risk**: STDP weight saturation. Fix already proposed: homeostatic decay per chunk
  (selectable via `synapforge.long.set_stdp_inference("decay")`). **Harness ready; awaiting Run 2 ckpt.**

### P22. Phase-flip auto-relaunch on threshold crossing — RESOLVED 2026-05-01
- **Symptom**: trainer's `--phase-aware` flag wrote `<run_dir>/.phase` JSON
  on val ppl ≤ 250, but the human had to read it and manually restart the
  trainer with `--self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05
  --phase-aware`. With unattended overnight runs that's a real gap — the
  threshold trips, then sits there until somebody notices.
- **Existing partial fix**: `scripts/relaunch_loop.sh` wraps the trainer
  and catches exit code 101; works only if you started the trainer through
  that wrapper. Doesn't help if the trainer's already running under
  `systemd-run` or `setsid+disown`.
- **Fix applied**: `scripts/phase_auto_relauncher.sh` (~180 LOC bash) is
  an out-of-process watchdog. Polls `<run_dir>/.phase` every 60 s. On
  phase change it (a) picks the newest `phase_change_step_*.pt` or
  `step_*.pt`, (b) strips optim_state inline (stale Adam momentum kills
  post-vocab-change warmstart), (c) SIGTERMs the current trainer with a
  10 s grace then SIGKILL, (d) re-launches via `setsid + disown` with
  the existing argv minus `--no-warmstart`, with `--warmstart` repointed
  at the stripped ckpt, plus the new phase's flags. Anti-thrash guard:
  no relaunch within 5 min of last restart. Refuses to start a trainer
  that wasn't already running (operator must handle dead-trainer case).
  Phase-1 flag string is locked to `phase_manager.PHASES[1]` (P19 fix);
  a drift-guard test asserts they don't diverge.
- **Verification**: `tests/integration/test_phase_auto_relauncher.py` —
  6 tests, all passing on Windows dev box (3.6 s):
  - `bash -n` syntax check
  - dry-run on phase=1 `.phase` JSON detects change, builds correct argv
    (drops `--no-warmstart`, repoints `--warmstart`, appends 4 phase-1
    flags), persists `.par_last_phase` state file
  - missing trainer PID -> logs "trainer not running" and skips
  - no `.phase` file -> single poll, exits cleanly under TEST_MODE
  - flag-string drift guard against `phase_manager.PHASES[1]`
- **Docs**: `docs/RENTAL_OPS.md` §7 "Auto-relauncher" added with full
  usage + safety guards. **Use one of `relaunch_loop.sh` or
  `phase_auto_relauncher.sh`, not both.**

---

## 7. Active runs (rental: 117.74.66.77:41614)

| PID | Run name | Started | Last status | Logs |
|-----|----------|---------|-------------|------|
| 13663 | v24h_qwen Run 2 | 2026-05-01 | step 1640 ce=5.95 | `/workspace/runs/v24h_qwen/train.log` |
| 12965 | auto_eval_daemon | 2026-05-01 | watching v24h_qwen (stale path) | `/workspace/auto_eval/eval.log` |
| (running) | triple_backup_daemon | 2026-05-01 | watching v24h_qwen (stale path) | `/workspace/backup/backup.log` |
| (running) | phase_manager | 2026-05-01 | watching v24h_qwen (stale path) | `/workspace/runs/v24h_qwen/phase.log` |

**v24h_qwen Run 2 STATUS — DEAD 2026-05-01 18:01**
- Crashed: torch.save → "file write failed" (system disk 99.9% full)
- Cause: SAVE_EVERY=250 + 1.83GB ckpts → 53 ckpts × 1.83GB = 97GB on 100GB disk
- PLIF observation: dead 10/10 from step_000250 through step_013250 (entire run)
- Final state: ce ≈ 8.2 (ppl ≈ 3640), spike rate 0.000

**v24h_qwen3 Run 3 STATUS — TRAINING 2026-05-01 18:43+**
- PID 15477 on rental
- Warmstart: `step_002250_plif_reinit.pt` (matched 93/163 — half-warm; PLIF param names changed so PLIF is random init)
- Backend: triton_block, batch=64 (z-loss OOM at bs=128 vocab=151936), kd-every=4
- SAVE_EVERY=2000, ckpt_cleanup.sh keeping last 5 step_*.pt (background loop)
- First 130 steps: ce 6.77 → 6.05 (CfC+FFN learning; PLIF still dead — auto-revive triggers at step 1000+)
- Output: `/workspace/runs/v24h_qwen3/`
- Daemons (phase/backup/auto_eval) still pointed at OLD v24h_qwen — need to be restarted with new path

**Verify** (next session):
```
ssh -p 41614 root@117.74.66.77 \
  'pgrep -fa "python3.*train_100m_kd"; \
   tail -20 /workspace/runs/v24h_qwen3/train.log; \
   df -h / | tail -1; \
   ls -la /workspace/runs/v24h_qwen3/*.pt'
```

---

## 8. Backup state (triple-path)

| Sink | Path | Status |
|------|------|--------|
| mohuanfang | `liu@mohuanfang.com:/home/liu/synapforge_backup/best_ckpts/` | 4.3GB, last sync auto |
| GH Release | `Lihfdgjr/synapforge releases` | sha256 dedup cache active |
| HF Hub | `huggingface.co/Lihfdgjr/synapforge-100m-ckpts` | private, automated |

**Pre-shutdown checklist** (rental teardown 2026-05-01 +N days):
1. `ssh root@rental 'rsync -avz /workspace/runs/v24h_qwen/ckpts/best_*.pt liu@mohuanfang.com:/home/liu/synapforge_backup/'`
2. Confirm GH release shows latest in `gh release view`.
3. Confirm HF dataset has matching sha256.

---

## 9. Decision log (recent)

- **2026-05-01**: Pivot to phased training (ppl-gated). User: *"等达到一定程度开多模态和自学习,还有好奇心"*.
- **2026-05-01**: Triton_block backend mandatory. gpu_dense at 6.7k tok/s wastes 90% of A800.
  Memory: `feedback_triton_backend_required.md`.
- **2026-04-30**: Vocab pivoted GPT-2 50257 → Qwen 151936. Bilingual + better tokenizer.
- **2026-04-30**: ~~Plan C (LoRA Qwen 0.5B) added as insurance~~ → **DELETED 2026-05-01** (architecture-claim violation; see `ANTI_LORA.md`).
- **2026-04-26**: NeuroMCP universal codebook L1/L2/L3 hierarchy + atomic skill log v2.
- **2026-04-26**: STDP `if self.training:` gate removed (`bio/stdp_fast.py:121`) —
  forward-only Hebbian inference unlocked.

---

## 10. Things explicitly OUT of scope right now

- GPT-4-class chat fluency (we say so in INVESTOR.md §"What's NOT a claim").
- BitNet b1.58 ternary edge build — code stub exists, not in 24h plan.
- 3D world understanding — code stub exists, separate research stream.
- Multi-node DDP — single A800 80GB is enough; layer 3 of `parallel.py` is for later.
- Mobile/CPU inference — `parallel.py` mixed-device smoke tested (P14) but
  not benchmarked at 375M scale.

---

## 11. Long-context strategy — 50M effective + monotonic quality (O10, O11)

User directive 2026-05-01: *"要实现的50m的有效上下文别忘了，而且性能和质量要随着上下文的增长而增长"*.

This is the **single biggest differentiator** vs transformer scaling. Don't drop it.

**Architecture levers** (all already in code, must be wired into eval harness):

| Lever | File | Status |
|-------|------|--------|
| Public long-context API surface (toggles + streamers) | `synapforge/long.py` (NEW shim) | wired 2026-05-01 |
| InfLLM L1/L2/L3 retrieval (5-tier memory hierarchy) | `synapforge/infinite.py` | wired |
| Multi-band τ for PLIF (different timescales) | `synapforge/cells/plif.py` | wired |
| Memory³ episodic store (compressed activations) | `synapforge/memory.py` | wired |
| BM25 sidecar (per `feedback_long_context_drift_fix.md`) | `synapforge/memory/bm25_sidecar.py` | partial |
| Inference-time STDP (forward-only Hebbian) | `bio/stdp_fast.py:121` | wired (gate removed) |
| RoPE NTK extrapolation | `synapforge/infinite.py::RotaryPositionEncoding` | wired |
| PQ16 hidden-state compression (64×) | `synapforge/quantize.py` | wired |
| Homeostatic STDP decay (anti-saturation) | env var `SYNAPFORGE_STDP_INFERENCE=decay` | wired (opt-in) |

### Validation harness (P20 + P21, written 2026-05-01, awaiting Run 2 ckpt)

| File | Purpose | Lengths | Marker |
|------|---------|---------|--------|
| `tests/integration/test_long_context_50m.py` | latency-per-token < 1.5x baseline, ppl drift < 5%, STDP norm < 10x growth — streams 4K-token chunks via `synapforge.long.chunked_token_stream` so 50M never lands in RAM | `[1K, 10K, 100K, 1M, 10M, 50M]` | `slow` (>=100K), `gpu` (>=1M) |
| `tests/integration/test_long_context_monotonic_quality.py` | A/B inference-STDP **on vs off**: needle-in-haystack at each L, exact-match argmax averaged over 3 trials; assert `acc_on >= acc_off - 0.05` per L plus `acc_on(100K) >= acc_on(1K) - 0.10` | `[1K, 10K, 100K]` | `slow` + `gpu` |
| `scripts/run_long_context_validation.sh` | rental-side runner: activates venv, runs both files with `-m slow`, dumps timestamped JSON to `/workspace/runs/v24h_qwen/long_ctx_validation_<ts>.json` | — | — |

**Skip strategy**: harness self-skips L >= 1M when `torch.cuda.get_device_properties(0).total_memory < 70 GB` (only A800 80GB+ has the headroom); CPU dev box runs only the `[1K]` baseline; default `pytest` skips `slow` markers entirely. Markers registered in `pyproject.toml::tool.pytest.ini_options.markers`.

**Toggle (paper-claim isolation)**: `synapforge.long.set_stdp_inference("on"|"off"|"decay")` sets `SYNAPFORGE_STDP_INFERENCE`; `STDPFastWeight.forward` (`bio/stdp_fast.py:127`) reads it. The A/B harness re-builds the model from a deterministic seed for each branch so the only difference between ON and OFF runs is the env var.

**Before pitch**:
1. Run validation at 1K → 10K → 100K (this week).
2. Run at 1M → 10M (next week, requires bigger ctx allocator).
3. Run at 50M (final, requires PQ16 hidden state compression for memory).
4. Document the *exact* gain numbers in `docs/CONTEXT_SCALING.md` + INVESTOR.md §1
   (currently makes the linear-cost claim without empirical 50M data).

---

## 12. Computer-use (`web_actuator.py`) — minimum viable scope (P18)

User directive: *"让ai使用神经元直接操控computer上网自动学习"*.

Audit 2026-05-01 found `synapforge/action/web_actuator.py` MISSING. MVP scope to ship by pitch:

```
synapforge/action/web_actuator.py  (NEW)
    class WebActuator:
        # Takes ActionHead's hidden vector (no JSON tool calls) →
        # maps to {click(x,y), scroll(dy), type(text), navigate(url)}.
        # Uses Playwright headless. No vision pipeline; uses DOM accessibility tree.
        def __init__(self, page: Playwright.Page, action_head: nn.Module): ...
        def step(self, hidden: Tensor) -> ActionResult: ...
        def trace(self, n_steps: int) -> List[ActionResult]: ...

scripts/web_actuator_smoke.sh  (NEW)
    # Boots playwright, navigates to a static local HTML page,
    # runs 50 ActionHead steps, asserts >= 1 successful click.
```

NOT in MVP scope: vision (we can do DOM-only first), multi-tab, login flows, CAPTCHA.

ETA: 4-6 hours — block before phase 3 SFT activation. Tracking item P18.

---

## 13. How to update this doc

Every session that produces facts (run results, gate trips, agent findings, design changes)
should:

1. Update **§7 Active runs** with current PIDs/status.
2. Update **§6 Open problems** — add new, mark resolved/stale.
3. Append to **§9 Decision log** if a substantive choice was made.
4. Update the matching `feedback_*.md` or `project_*.md` memory if the decision is durable.
5. Bump the date at the top of this file.

This doc is the contract between sessions. Do not let it go stale.
