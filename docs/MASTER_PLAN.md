# SynapForge — Master Plan

**Updated**: 2026-05-01 (revision 2: + agent audit P11-P21, vocab fix, 50M context O10-O11)
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
| O10 | **50M effective context** with linear inference cost | partial: long.py + InfLLM L1/L2/L3 wired, untested at 50M | benchmark: 1K → 10K → 100K → 1M → 10M → 50M latency linear, ppl drift < 5% |
| O11 | **Quality monotonic with context length** (longer = better, not worse) | thesis: inference-time STDP + retrieval, not yet validated | A/B: same prompt at 1K vs 50K context; quality score must NOT regress |

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
| Computer-use (web auto-learn) | `synapforge/action/web_actuator.py` | needs sandbox | post-phase-3 | code done, gated |
| Skill persistence | `action/skill_log_v2.py` (atomic, durable) | `tests/test_skill_log_atomic.py` | n/a (background) | done |
| Triple-backup | `scripts/triple_backup_daemon.py` | tripwire 5-cycle warn | daemon | running |
| Honest eval pipeline | `scripts/auto_eval_daemon.py` | per-ckpt 5+5 chat | runs on every ckpt | running |
| Phase manager | `scripts/phase_manager.py` | log → `.phase` JSON | daemon | running |
| Plan C LoRA Qwen | `scripts/train_qwen_lora.py` | `--smoke` + `--print-only` | `qwen_lora_chat_repl.py` | code done, smoke verified, real run pending (see `PLAN_C_RUNBOOK.md`) |

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

### P5. Plan C LoRA Qwen — runbook done, real run pending
- **Risk**: insurance demo was theoretical. If 100M training fails phase 3, no fallback.
- **Status (2026-05-01)**: trainer hardened — `--print-only` flag added, peft+inline LoRA
  fallback both verified on smoke, unified `final.pt` ckpt now saved with `{model, config,
  framework, lora, vocab}` payload (matches §6 P12 contract), `--lora-r/--lora-alpha/
  --output-dir` aliases match rental orchestration scripts, `chat_eval_gate.py` auto-detects
  Plan C ckpts via `framework` field and routes to `qwen_lora_chat_repl` loader. Local
  smoke verified: `python scripts/train_qwen_lora.py --smoke --print-only` and
  `--smoke` (5 steps, no GPU/peft/qwen weights needed). See `docs/PLAN_C_RUNBOOK.md`.
- **Action**: ≤ 1 hour LoRA real run on rental — runbook has 4 steps (print-only
  sanity → smoke → 200-step real → chat_eval_gate ≥ 0.6) plus a fix-and-rerun
  decision tree. Once `pass_rate >= 0.6` and triple-backup picks up `final.pt`,
  mark P5 RESOLVED.

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

### P13. KD chunk size hardcoded `bs // 4`
- **Symptom**: `train_100m_kd.py:318` chunk = bs // 4. At bs=128 still 3.7GiB intermediate.
  Won't scale to smaller cards.
- **Fix**: `chunk = max(1, int(vram_free * 0.5 / (seq * vocab * 4)))` from
  `torch.cuda.mem_get_info()`.
- **Severity**: nice-to-have on A800.

### P14. `parallel.py` Layer 2 (`place_mixed_device`) is orphan code
- **Symptom**: Documented as a 3-layer feature; only `examples/mixed_device_training.py`
  calls it. Trainer uses only Layer 3 (DDP).
- **Fix**: Either delete + update `INVESTOR.md` honesty section, or add 1-step smoke test.
- **Severity**: nice-to-have / honesty.

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

### P18. `web_actuator.py` claimed but missing
- **Symptom**: MASTER_PLAN.md §5 row "Computer-use" claims `synapforge/action/web_actuator.py`.
  File does not exist. (User directive: 让ai使用神经元直接操控computer上网自动学习.)
- **Fix**: Either (a) implement minimal web actuator (Playwright wrapper that takes ActionHead
  output vector, maps to click/scroll/type), or (b) remove the claim and scope to phase 4
  post-pitch. Decision: implement minimal version this week — see §12.
- **Severity**: pre-pitch-required (user-explicit feature).

### P19. Phase manager flags out of sync with trainer argparse — RESOLVED 2026-05-01
- **Symptom**: `phase_manager.py:45` listed `--intrinsic-curiosity --self-learn-ttt --stdp-novelty`
  for phase 1. Only `--self-learn-ttt` and `--curiosity-weight` exist in trainer. Relauncher
  would crash on argparse error.
- **Fix applied**: phase 1 flags now `--self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05`.
  Phase 2 dropped `--modal-byte-patch --cross-modal-contrastive` (not in argparse), kept
  `--modal-list image,audio,time_series`.
- **Open**: STDP-novelty signal needs autograd path before it can become a flag. Defer.

### P20. 50M effective context — claimed, not validated
- **Symptom**: User directive 2026-05-01: *"50m的有效上下文别忘了"*. Code paths exist
  (`synapforge/long.py`, `infinite.py`, InfLLM L1/L2/L3 retrieval), but no run has
  ever tested at 50M tokens. Memory `reference_a100_80gb_context_max_2026.md` claims
  50M is the qualitative ceiling.
- **Risk areas**: (a) PLIF spike retrieval at 50M — does spike-pattern recall scale?
  (b) CfC τ saturation in 4-32K range — multi-band τ fix wired? (c) STDP weight saturation
  after 1M tokens — needs forgetting term or homeostatic decay.
- **Action**: write `tests/integration/test_long_context_50m.py` that streams 50M tokens
  through inference and reports (i) latency vs context length, (ii) ppl drift vs context,
  (iii) STDP weight L2 norm growth curve. Target: latency linear, ppl drift < 5%, STDP norm
  not exploding.
- **Severity**: pre-pitch-required (it's a headline claim).

### P21. Quality MUST grow monotonically with context length
- **Symptom**: User directive 2026-05-01: *"性能和质量要随着上下文的增长而增长"*.
  This is the inference-time STDP thesis (`feedback_inference_stdp_unlock.md`,
  `bio/stdp_fast.py:121` gate already removed) — longer context = more Hebbian
  updates = better adaptation.
- **What it means concretely**: at 1K context, ppl X. At 10K context, ppl ≤ X (NOT ≥ X).
  At 100K context, ppl ≤ X. This is the OPPOSITE of every transformer (which degrades
  monotonically past trained ctx).
- **Action**: A/B harness at 1K / 10K / 100K / 1M context with inference-STDP ON vs OFF.
  Quality metrics: needle-in-haystack at varying depths, conversation-coherence on long
  multi-turn, factuality on long-context QA. ON > OFF must hold at all four lengths.
- **Severity**: pre-pitch-required (this is THE differentiator vs transformer).
- **Risk**: STDP weight saturation. Fix already proposed: homeostatic decay per chunk.

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

## 11. Long-context strategy — 50M effective + monotonic quality (O10, O11)

User directive 2026-05-01: *"要实现的50m的有效上下文别忘了，而且性能和质量要随着上下文的增长而增长"*.

This is the **single biggest differentiator** vs transformer scaling. Don't drop it.

**Architecture levers** (all already in code, must be wired into eval harness):

| Lever | File | Status |
|-------|------|--------|
| InfLLM L1/L2/L3 retrieval (1K/10K/100K) | `synapforge/long.py` | wired |
| Multi-band τ for PLIF (different timescales) | `synapforge/cells/plif.py` | wired |
| Memory³ episodic store (compressed activations) | `synapforge/memory.py` | wired |
| BM25 sidecar (per `feedback_long_context_drift_fix.md`) | `synapforge/retrieval/bm25.py` | partial |
| Inference-time STDP (forward-only Hebbian) | `bio/stdp_fast.py:121` | wired (gate removed) |
| RoPE NTK extrapolation | `synapforge/cells/rope.py` | wired |
| PQ16 hidden-state compression (64×) | `synapforge/quantize.py` | wired |
| Homeostatic STDP decay (anti-saturation) | NOT YET — add per-chunk decay term | **TODO** |

**Validation harness needed** (P20):
```python
# tests/integration/test_long_context_50m.py (NEW)
ctx_lengths = [1024, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]
for L in ctx_lengths:
    latency_ms = measure_inference_latency(L)
    ppl_drift = ppl(L) / ppl(1024)
    stdp_norm = ||ΔW_STDP||(at end of L tokens) / ||W_initial||
    # Assertions:
    assert latency_ms / L < linear_baseline * 1.1   # near-linear, not quadratic
    assert ppl_drift < 1.05                          # ≤5% drift
    assert stdp_norm < 10.0                          # not exploding
```

**Quality monotonic harness (P21)** — A/B inference-STDP on/off at each context length.
Quality metric = composite (needle-in-haystack accuracy + conversation coherence + long-context
QA factuality). ON > OFF expected at every length; gap should *grow* with length.

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
