# SynapForge — Master Plan

**Updated**: 2026-05-01
**Owner**: Liu (mohuanfang@gmail.com)
**Status**: pre-investor-demo, training in progress on rental A800 80GB

> **READ THIS FIRST** every session. Update the **Current state**, **Active runs**,
> and **Open problems** sections as facts change. Memory entry
> [project_master_plan.md](.../project_master_plan.md) points future Claude here.

---

## 1. North star

A single sentence: **a 100M LNN+SNN that chats in EN+ZH, learns at inference via STDP, grows synapses into the action space (NeuroMCP) instead of emitting tool-call tokens, and never hallucinates "training is going up" — every claim is backed by a runnable demo.**

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

---

## 3. Phased training plan (gate by ppl, NOT by step count)

Source of truth: `scripts/phase_manager.py` (auto-watches `train.log`, writes `.phase` flag file).

| Phase | Trigger | Adds | Notes |
|-------|---------|------|-------|
| 0 | warm start, ce ≈ 9.6 | LM-only KD from Qwen 0.5B teacher | current state at v24h_qwen |
| 1 | val ppl ≤ 250 | `--intrinsic-curiosity --self-learn-ttt --stdp-novelty` | turn on internal-reward + TTT replay |
| 2 | val ppl ≤ 100 | `--modal-byte-patch --cross-modal-contrastive --modal-list image,audio,time_series` | needs base linguistic competence first or cross-modal aux destroys it |
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

**Concretely**:
- Curiosity weight starts at 0, ramps to 0.05 over 1k steps after gate. Don't slam it on.
- Self-learn TTT runs on top-K=8 high-CE samples, 1 inner step each, every K=200 outer steps.
- STDP novelty contributes to curiosity reward only after STDP has >100 mints in the codebook.

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
| Computer-use (web auto-learn) | `synapforge/action/web_actuator.py` | needs sandbox | post-phase-3 | code done, gated |
| Skill persistence | `action/skill_log_v2.py` (atomic, durable) | `tests/test_skill_log_atomic.py` | n/a (background) | done |
| Triple-backup | `scripts/triple_backup_daemon.py` | tripwire 5-cycle warn | daemon | running |
| Honest eval pipeline | `scripts/auto_eval_daemon.py` | per-ckpt 5+5 chat | runs on every ckpt | running |
| Phase manager | `scripts/phase_manager.py` | log → `.phase` JSON | daemon | running |
| Plan C LoRA Qwen | `scripts/train_qwen_lora.py` | `--smoke` flag | needs rental run | code done, untrained |

---

## 6. Open problems (live list — UPDATE AS YOU LEARN)

### P1. Run 1 word-salad divergence at step 5000 — REAL ROOT CAUSE
- **Symptom**: ce 5.72 → 8.46 between step 4500–5000. Output: high-freq token salad.
- **Diagnosis (Agent 1+2 on 2026-04-30)**: PLIF threshold drift unclamped + wrong PLIFCell
  variant (RC leaky-to-steady) that never fires with threshold=0.3 on tanh-bounded input.
- **Fix applied**: `model_100m.py` plif_threshold 0.3 → 0.05, tau 1.5 → 2.5. Spike rate
  rose 0.020 → 0.335 verified.
- **Open**: should we EMA-clamp threshold during training to prevent re-drift?

### P2. KD OOM at vocab=151k bs=128
- **Symptom**: 18.51 GiB intermediate alloc on KL.
- **Fix applied**: chunked KL over batch dim (÷4).
- **Open**: chunk size hardcoded — should be auto-tuned per VRAM headroom.

### P3. Self-learn TTT and val set leak
- **Risk**: TTT inner step on val sample = val ppl converges artificially.
- **Status**: not yet validated. Need to hold a *secondary* val set for honest reporting.

### P4. Curiosity reward vs. KD KL conflict
- **Risk**: curiosity wants tokens teacher didn't predict; KD penalizes those.
- **Status**: not yet observed in training. Mitigation TBD (clamp curiosity reward, or
  schedule curiosity to ramp up only as KD weight ramps down).

### P5. Plan C LoRA Qwen — never run on rental
- **Risk**: insurance demo is theoretical. If 100M training fails phase 3, no fallback.
- **Action**: ≤ 1 hour LoRA smoke run on rental this week.

### P6. NeuroMCP density saturation at ~28%
- **Status**: empirically observed in v4.1 runs. Not a bug — might be the natural sparsity.
- **Open question**: does saturation hurt skill recall or help (parsimony)?
- **Required**: long-running NeuroMCP smoke (≥ 600 trials) to prove K grows, not saturates.

### P7. Computer-use sandbox not yet exercised
- **Status**: code complete (`web_actuator.py`), but no real session has run.
- **Risk**: claim "AI uses neurons to drive computer" is currently a code path, not a demo.
- **Action**: 30-min sandbox session post-phase-3 with adversarial input filter active.

### P8. MCP shell + bg job interaction on rental
- **Symptom**: MCP `proc_exec` kills nohup'd children. Repeated session expirations.
- **Mitigation**: push commands FROM mohuanfang side (real Linux shell), or use
  `systemd-run --user --unit=X` (logs in `feedback_mcp_shell_kills_bg_jobs.md`).
- **Open**: should we add a watchdog that restarts the trainer if it dies during MCP
  reconnect?

### P9. Investor demo claim parity (data sources)
- **Status**: 4 download scripts written (pretrain/sft/multimodal/eval) but not yet
  fully executed end-to-end on rental.
- **Risk**: claim "EN+ZH" hinges on alpaca-zh + Chinese pretrain mix actually being on disk.
- **Action**: a smoke run that actually downloads → mixes → trains 50 steps must be in CI
  before pitch.

### P10. README + docs lag the code
- **Status**: ~30 docs in `docs/` but several are stale (e.g., `MULTIMODAL_TRAINING.md`
  predates byte-patch refactor).
- **Action**: doc-sync agent on next push, or add a per-doc `_stamp.json` so we know what
  was last verified.

---

## 7. Active runs (rental: 117.74.66.77:41614)

| PID | Run name | Started | Last status | Logs |
|-----|----------|---------|-------------|------|
| 13663 | v24h_qwen Run 2 | 2026-05-01 | step 1640 ce=5.95 | `/workspace/runs/v24h_qwen/train.log` |
| 12965 | auto_eval_daemon | 2026-05-01 | watching v24h_qwen | `/workspace/auto_eval/eval.log` |
| (running) | triple_backup_daemon | 2026-05-01 | watching v24h_qwen | `/workspace/backup/backup.log` |
| (running) | phase_manager | 2026-05-01 | watching v24h_qwen, currently phase 0 | `/workspace/runs/v24h_qwen/phase.log` |

**Verify** (next session): `ssh -p 41614 root@117.74.66.77 'ls -la /workspace/runs/v24h_qwen/ckpts/ | head; tail -5 /workspace/runs/v24h_qwen/train.log'`

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
- **2026-04-30**: Plan C (LoRA Qwen 0.5B) added as insurance. Documented `PLAN_C_QWEN_LORA.md`.
- **2026-04-26**: NeuroMCP universal codebook L1/L2/L3 hierarchy + atomic skill log v2.
- **2026-04-26**: STDP `if self.training:` gate removed (`bio/stdp_fast.py:121`) —
  forward-only Hebbian inference unlocked.

---

## 10. Things explicitly OUT of scope right now

- GPT-4-class chat fluency (we say so in INVESTOR.md §"What's NOT a claim").
- BitNet b1.58 ternary edge build — code stub exists, not in 24h plan.
- 3D world understanding — code stub exists, separate research stream.
- Multi-node DDP — single A800 80GB is enough; layer 3 of `parallel.py` is for later.
- Mobile/CPU inference — `parallel.py` mixed-device exists but not validated.

---

## 11. How to update this doc

Every session that produces facts (run results, gate trips, agent findings, design changes)
should:

1. Update **§7 Active runs** with current PIDs/status.
2. Update **§6 Open problems** — add new, mark resolved/stale.
3. Append to **§9 Decision log** if a substantive choice was made.
4. Update the matching `feedback_*.md` or `project_*.md` memory if the decision is durable.
5. Bump the date at the top of this file.

This doc is the contract between sessions. Do not let it go stale.
