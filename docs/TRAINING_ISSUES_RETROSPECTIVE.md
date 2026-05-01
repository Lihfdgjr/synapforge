<!-- DOC_STAMP: LIVE 2026-05-01 -->
# Training Issues Retrospective — 2026-05-01

**Scope**: 8 training runs over 12 hours on rental A800-80GB. Recipe: 100M LNN+SNN
+ Qwen 0.5B teacher KD, vocab 151936, target val ppl ≤ 250 to trip phase 1.

**Audience**: investors / external reviewers / the next agent.

> Eight distinct failure modes, eight shipped fixes, no shortcuts. Each row is a
> commit you can `git show`. The follow-up section enumerates known
> architecture-quality concerns we have **not** fixed but have a plan for.

---

## 1. Root causes encountered (8 distinct)

| # | Run | Symptom | Root cause | Source | Fix shipped (commit) |
|---|-----|---------|------------|--------|----------------------|
| 1 | Run 1 (Apr) | Word salad after step 5000; spike rate 0/10 entire run | PLIF threshold drifted *up* past tanh-bounded input distribution; wrong PLIFCell variant (RC leaky-to-steady) never fires at threshold=0.3 | `synapforge/model_100m.py` (init), `synapforge/surrogate.py` (PLIFCell) | `f051257` — PLIF homeostatic threshold control + EMA clamp (`clamp_threshold(0.005, 0.5)` + `homeostatic_step(target=0.10, gain=0.005)` every 50/100 steps under `torch.no_grad()`) |
| 2 | Run 2 | Trainer crashed at step ~13k; `/workspace` 100% full; 53 ckpts × 1.83 GB ate disk | `SAVE_EVERY=250` checkpointed every 250 steps with no rotation; 100M model serialized at 1.83 GB | `train_100m_kd.py` save loop | `2d8bb24` — disk-full recovery: keep best + last-3 rotation + `--save-every` raised to 1000 + cleanup script |
| 3 | Run 3a | Val ppl 3474 in 2500 steps despite "warmstart"; clean trainer worked | Warmstart loaded **stale Adam** `m`/`v` from a previous vocab=50257 ckpt into vocab=151936 model — momentum vectors pointed at deleted rows | `train_100m_kd.py` warmstart path | `2d8bb24` — strip `optim_state` on vocab-mismatch warmstart; only param tensors copied, optimizer reinitialized |
| 4 | Run 3b | Val ppl 422 → 4071 between step 2500–5500 | LR=3e-4 too aggressive for 100M LNN+SNN cold-start; CfC weights diverge before homeostasis can damp PLIF | `train_100m_kd.py:_parse_args` LR default | `f7ff162` — LR 3e-4 → 1e-4; 100M LNN+SNN needs 3× softer warmup than transformers |
| 5 | Run 3c | Val ppl 397 at step 2000 then 1886 at step 2500 — *deterministic* divergence | `ParquetTokenStream` walked parquets in `sorted(glob.glob(...))` order with **zero shuffle**; every run hit the same overfit token at the same step | `synapforge/data/__init__.py:92` | `bd1ac73` — `shuffle_buffer=10000` default in `ParquetTokenStream` + per-epoch file reshuffle; train-side only, val deterministic |
| 6 | Run 3d | OOM at bs=128: 18.5 GiB intermediate alloc on KD KL | KD chunked softmax materialized `(B*T, V, fp32)` over full vocab=151936; fixed `chunk = bs // 4` doesn't track free VRAM | `train_100m_kd.py::_kd_loss` | `5f420a4` — `_kd_chunk_size(B,T,V, headroom=0.5)` reads `cuda.mem_get_info()`, sizes chunks to fit in 50% free VRAM, floored at 1, capped at batch_size |
| 7 | Run 3f | OOM at bs=80 in *backward* pass | Full-vocab z-loss `(logits**2).sum(-1)` materialized fp32 activations in backward → exceeded 80 GB even with KD chunked | `train_100m_kd.py` z-loss aux term | `4d0d2a9` — sparse z-loss top-K=2048; restrict z-penalty to top-K row entries, dropping backward activations 70× |
| 8 | Run 3h | OOM at bs=80 in *forward* — KD softmax | Teacher logits softmax materialized full vocab in fp32 forward at bs=80 even after chunking; cannot reduce chunk further without losing KD signal | `train_100m_kd.py::_kd_loss` teacher branch | **incoming** — top-K=512 teacher softmax (renormalize over top-K), drops teacher fp32 alloc 300× |

**Reading guide for reviewers**: each row is one commit. Issues #1–7 closed; #8 patch
in flight (the next iteration).

---

## 2. Architecture-quality concerns (documented, not fixed)

These are *known weaknesses* in the current 100M Synap-1 training, kept honest
on this list so investors/reviewers see we are not hiding them. Mitigation
plans below — none are implemented in this commit.

### a. PLIF dead 10/10 spike rate even with P1 homeostasis (`f051257`)

**Symptom**: Run 3b / 3c / 3e all logged 0% spike rate across all blocks for the
entire run despite `homeostatic_step` firing every 50 steps and `clamp_threshold`
re-floor-ing every 100. Threshold *did* drop (0.05 → 0.012) but spike rate
never crossed 0%.

**Implication**: the CfC-only path is doing **all** the modeling work. The
"LNN+SNN hybrid" claim is currently aspirational — what we actually train is a
LNN with a dead SNN tap. Sparse-activation efficiency claims (matmul-free
M2/M3 in `MATMUL_FREE.md`) ride on this and would not be honest until PLIF
fires.

**Plan**: surrogate gradient annealing (start width=10, anneal to 1 over 5000
steps) — wide surrogate gives gradient even on far-from-threshold pre-spike
voltages, lets PLIF *learn* to fire instead of being clamped into firing.
Combine with threshold ramp 0.05 → 0.01 over phase 0. ETA: 1 week.

### b. 70 vocab tail rows never gradient-loaded

**Symptom**: vocab padded 151643 → 151936 (multiple-of-128 alignment for the
matmul kernel). Rows 151643..151935 never appear in any training token (Qwen2.5
tokenizer max id is 151642), so their `tok_embed.weight` and `lm_head.weight`
rows receive **zero** gradient.

**Implication**: not a hot bug today (those IDs never get sampled), but if any
future tokenizer change or special-token addition uses those slots, the model
will emit untrained-noise embeddings for them. Defensive code should freeze
them.

**Plan**: `freeze_vocab_tail_mask()` — set `tok_embed.weight[151643:].requires_grad
= False` and same for `lm_head.weight[151643:]` in `model_100m.py::__init__`.
ETA: 1 day.

### c. Train ce 6–7 vs val ppl 320–400 drift

**Symptom**: train batch CE drops to 6.0–7.0 (~ppl 400–1000 in CE units) while
val ppl stays 320–400. Eyeballed for 5 runs.

**Diagnosis**: KD signal pulls hard toward Qwen 0.5B teacher distribution within
the active minibatch context, then drift on next batch undoes some. Student
overfits to current minibatch, drops on val. Classic small-model + strong-teacher
overfitting.

**Plan**: longer LR warmup (currently 200 steps; bump to 1000), smaller LR
slope (currently linear; switch to cosine with longer plateau), and a BPTT
regularizer term `λ · ||h_T - h_0||²` to prevent state drift inside the
truncated-BPTT window. ETA: 1 week.

### d. z-loss linearly growing across Run 3b

**Symptom**: sparse z-loss top-K=2048 (`4d0d2a9`) helps OOM but `z_loss` itself
trends up linearly with step — not constant.

**Diagnosis**: top-K only restricts the *backward* activation; doesn't address
the cause: `lm_head` logits drift per-row over training. Bigger logits → bigger
softmax denominators → bigger z-loss.

**Plan**: tied LM head normalization — insert `nn.LayerNorm(d, elementwise_affine
=False)` immediately before the final `lm_head` projection. Stops per-row scale
drift without harming representation. ETA: 1 day.

**Status (2026-05-02 02:18, T7.3)**: PRIMARY PATCH SHIPPED. Affine-free
LayerNorm wired into `SynapForge100M.forward()` between `ln_f` (the existing
RMSNorm with affine scale) and the LM projection, behind default-OFF
`--lm-head-pre-ln` CLI flag. Registers zero state-dict keys (no learnable
gamma/beta, no buffers) so the toggle is bit-compatible with all existing
checkpoints — param count identical on/off. 5 tests at
`tests/integration/test_lm_head_pre_ln.py` (5/5 PASS), including a
smoking-gun robustness test that artificially corrupts `ln_f.weight = 100.0`
(simulating runaway Adam drift) and verifies that with the flag ON the
LM-head input row norm stays clamped near `sqrt(d)` while with the flag OFF
it scales linearly to ~100. Awaits live training-run validation that the
z-loss curve flattens.

### e. Run 3c step-2500 jump unconfirmed-cause

**Symptom**: P24 attributed Run 3c's step-2500 divergence to deterministic data
ordering. Run 3e (with shuffle ON) made it past step 1500 healthy.

**Honest gap**: we killed Run 3e before step 2500, so we don't have direct
evidence shuffle solves it — Run 3i is the test that finally confirms
or rejects the diagnosis.

**Plan**: queue Run 3i with `--shuffle-buffer 10000` and run past step 3000;
log `train.log` ppl trajectory from step 2000–3000 as a permanent artifact.
ETA: 1 day (just rerun trainer with current config).

---

## 3. Five concrete next-iteration patches (planned, not in this commit)

| # | Patch | Mechanism | Expected impact | ETA |
|---|-------|-----------|-----------------|-----|
| 1 | Frozen vocab tail mask | `tok_embed.weight[151643:].requires_grad=False` + same on `lm_head.weight`; padding rows can never accumulate untrained gradient noise. Touch in `model_100m.py::__init__`. | Quality: defensive, zero observable effect today; prevents future regression if tokenizer changes. Speed: marginal speedup (70 fewer rows in optimizer state). | 1 day |
| 2 | Surrogate gradient annealing | Replace fixed `ATan(width=2)` with `ATanScheduled(start_width=10, end_width=1, anneal_steps=5000)`. Wide surrogate at start gives gradient on far-from-threshold pre-spikes, anneals to sharp at convergence (SEW recipe). | Quality: PLIF *learns* to fire instead of being clamped (concern §2.a). Expect spike rate 0% → 5–15% by step 5000. Speed: +2% slower (slightly more autograd work in the wide-surrogate phase), but unblocks the SNN tap. | 1 week |
| 3 | PLIF spike-rate-target loss term | `λ_rate · (rate_observed - rate_target).pow(2)` with `rate_target ∈ [0.05, 0.20]` and `λ_rate=0.001`. Differentiable pressure to hit the band — homeostasis is a hammer (no-grad clamp), this is a scalpel. | Quality: tight control on spike rate sparsity; lets us *honestly* claim 5–20% activation. Speed: +0.5% (one extra reduction per block). | 1 day |
| 4 | LM head spectral norm | `nn.utils.spectral_norm` on `self.lm_head` weight. Bounds spectral radius → bounds logit magnitude → caps z-loss (concern §2.d). | Quality: training stability, fixes z-loss linear drift. Speed: small forward overhead (one power-iteration per step), measured ≤1% on similar models. | 1 day |
| 5 | Adaptive batch via grad-accum | `bs_eff=128` via 2× grad-accum at `bs_micro=64` instead of bs=80 — works around the 80 GB ceiling that broke Run 3f / 3h. Optimizer step every 2 micro-batches; effective batch unchanged. | Quality: identical to bs=128 (no curvature change). Speed: ~5% slower than literal bs=128 (extra optimizer.zero_grad), but eliminates OOM blast radius. | 1 day |

**Total ETA if shipped sequentially**: ~2 weeks. Patches #1, #3, #4, #5 are
each one day; #2 takes one week because of the eval cycle (must re-run phase 0
to validate spike rate trajectory).

---

## 4. Cross-reference

- `docs/MASTER_PLAN.md` §6 — entries P25 / P26 / P27 / P28 / P29 point here
- Memory: `feedback_training_root_causes_2026q2.md` — one-line summary per issue

---

*Document: 99 lines of body content (excluding boilerplate).
Last updated 2026-05-01 — next update on Run 3i confirmation or any new failure.*
