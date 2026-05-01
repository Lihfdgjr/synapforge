# Phased training -- when to flip on each opt-in component

The 100M LNN+SNN model (`SynapForge100M`: 10x HybridBlock with CfC +
PLIF + sparse synapse) trains in five phases. Each phase opens up an
extra capability once the base model is healthy enough that the
new signal is helpful (not noise).

The default `train_100m_kd.py` command runs **Phase 0** -- LM CE plus
GPT-2 KD distillation, no extra mixins. Every other phase is gated by
a CLI flag and only enabled after the matching ppl threshold is
crossed. `scripts/phase_manager.py` watches `train.log`, tags
threshold crossings, and writes `.phase` JSON files describing the
flags that should be appended on the next launch.

**The opt-in contract is firm: no flag ever set means the default
training trajectory is byte-identical to the pre-mixin behaviour.**

---

## Phase 0 -- LM CE + KD (default)

**Trigger:** ppl > 250 (i.e., on cold/warm start until base LM is
fluent).

**Launch command:**

```bash
python train_100m_kd.py \
    --backend gpu_dense \
    --warmstart /workspace/runs/step_001250.pt \
    --steps 5000 --batch-size 32 \
    --teacher gpt2 --kd-weight 0.7 --kd-temperature 4.0 --kd-every 4
```

**Loss combine:**

```
loss = (1 - alpha_KD) * (CE + 1e-4 * z_loss) + alpha_KD * KL(student||teacher)
```

**Expected effects:**
* Train ce drops 9.6 -> ~5.5 in ~3000 steps (warmstart from v16)
* Spike rate stabilises in [0.05, 0.20] band (PLIF healthy region)
* KD loss tapers from ~25 to ~6 as student catches up

**Metrics to watch:**
* `loss` -- the reported scalar; should monotonic-decrease modulo noise
* `ce`, `kd`, `z` -- the three components
* `spike: mean=...` -- if mean drifts more than 0.05 from `--spike-target`,
  console will WARN; persistent drift > 100 steps is a real bug
* `VAL ... ppl=...` every 500 steps; healthy is monotonic-decreasing

**Known failure modes:**
* `kd=...` stays at 0: HF teacher load failed; `--teacher-fallback-ckpt`
  not set or invalid. Inspect early `[teacher]` log lines.
* Spike rate collapses to 0 (`dead=N/N`) -- bootstrap is dying;
  drop `--kd-weight` to 0.5 and rerun. Documented in
  `feedback_plif_dead_bootstrap.md`.
* `loss` stalls > 5.0 for > 1000 steps: warmstart was bogus; check
  `warmstart matched=...` log line; should be > 80% of target params.

---

## Phase 1 -- + curiosity + self-learn (ppl < 250)

**Trigger:** Validation ppl drops below 250 (typical: step 3000-5000
of Phase 0 with warmstart). `phase_manager.py` writes `.phase`
file with `phase_id=1` to signal this.

**Launch command** (append on top of Phase 0 flags):

```bash
python train_100m_kd.py \
    ... (all Phase 0 flags) ... \
    --curiosity-weight 0.05 \
    --self-learn-ttt --self-learn-k 8
```

**What turns on:**

1. **CuriosityMixin** wires the 6-signal STDP-driven reward at every
   training step:
   ```
   C = 0.40*delta_F + 0.25*||delta W_STDP|| + 0.15*G_HNSW
       + 0.10*H[spike] + 0.05*N - 0.05*V_noise
   ```
   Only `delta_F` (ICM forward-model surprise) gets autograd; the
   other five signals are bookkeeping for offline analysis. The
   trainer adds `--curiosity-weight * delta_F_loss` on top of the LM
   loss so the predictor learns to model state transitions.

2. **SelfLearnMixin** at every val (`EVAL_EVERY=500`) probes the model:
   it picks the top-K hardest val samples, runs a 1-step TTT update on
   filtered params (`tau`, `threshold`, `shared_tau`, `fast.w_`,
   `ttt_`), measures CE-after, restores weights. Reports lift in
   `metrics["self_learn_lift"][step]`.

**Why ppl<250 is the right gate:**
* Below ppl 250, the FreeEnergySurprise predictor's MSE has signal --
  it's distinguishing real transitions from noise. Above ppl 250 the
  predictor is just copying random embeddings; curiosity reduces to
  unhelpful noise.
* TTT lift is measurable only when the base model has enough structure
  for a single step to fix something. At ppl > 250, TTT lift is in
  the rounding error and the probe just adds noise to logs.

**Expected effects:**
* `cur=...` tracked in step log; should be small (~0.01-0.05) and
  decreasing as the predictor learns. Spikes during distribution
  shifts are expected.
* `[self-learn] lift=...` reported each eval. Healthy: positive
  (between +0.001 and +0.05 ce-units). Drifting negative = TTT is
  hurting; reduce `--self-learn-k` or pause this phase.

**Metrics to watch:**
* `cur` per step
* `lift` per eval -- should be positive on average. Track
  `metrics["self_learn_lift"]` over the run.
* Wallclock per step: curiosity adds one MLP forward + a backward pass
  through it; expect ~5% slowdown. SelfLearn only fires at val so it
  does not affect train throughput.

**Known failure modes:**
* `cur=` reads `nan` -- FreeEnergySurprise predictor diverged. Drop
  `--curiosity-weight` to 0.01 and warmstart from the last good ckpt.
* `lift` consistently negative -- TTT is over-fitting the K samples.
  Solutions: lower `--self-learn-k` to 4, or expand `params_filter`
  (more params smooths the update). Avoid the entire phase if PLIF
  spike rates are still drifting.

---

## Phase 2 -- + multimodal contrastive (ppl < 100)

**Trigger:** Validation ppl drops below 100. By now the text-only
hidden representation is structured enough that aligning it with image
or audio embeddings is a useful auxiliary signal.

**Launch command** (append on top of Phase 1 flags):

```bash
python train_100m_kd.py \
    ... (all Phase 1 flags) ... \
    --modal-list image,audio \
    --modal-data-dir /workspace/data/multimodal/ \
    --modal-alpha 0.05
```

**Data layout** required at `--modal-data-dir`:

```
/workspace/data/multimodal/
    image/   *.pt (each tensor is (3, H, W) float, H,W % 8 == 0)
    audio/   *.pt (each tensor is (samples,) float, mono 16kHz)
```

The trainer pre-loads up to 64 files per modality and cycles through
them with index `_sample_idx % len(files)`. Pre-tokenise once with
`scripts/cache_modal.py` (or hand-roll). Tensors smaller than the
batch size are tiled.

**What turns on:**

The `MultimodalMixin` runs each modality's byte-patch encoder
(`ImagePatchEmbed` 8x8 patches, `AudioPatchEmbed` 20ms raw chunks --
both Fuyu/Chameleon-style, NOT LLaVA-style frozen vision). Mean-pools
modal hidden over time, mean-pools text hidden, computes InfoNCE
(temperature 0.1, in-batch negatives) and adds
`alpha * InfoNCE_loss` to the total. Default `alpha=0.05`.

**Why ppl<100 is the right gate:**
* Above ppl 100 the text hidden reps are still highly noisy; aligning
  them with image embeddings teaches the encoder to match noise.
* Once ppl<100, the text hidden carries actual semantics ("dog" tokens
  cluster, "color" tokens cluster) and the contrastive signal aligns
  modal embeddings with those clusters -- valuable.

**Expected effects:**
* `modal=...` tracked in step log. Reasonable: 0.02 - 0.20 with the
  default alpha. Decreasing trend means alignment is improving.
* No effect on PPL initially, may even hurt by 2-5 PPL for the first
  500 steps; pure-text PPL should recover within 1500 steps as text
  hidden gains modal-grounded structure.

**Metrics to watch:**
* `modal` log component
* `ppl_eval` -- short-term regression of 2-5 ppl is expected; if it
  doesn't recover within 1500 steps, drop `--modal-alpha` to 0.02 or
  pause the phase.

**Known failure modes:**
* `modal=0.0000` always -- check `[mixin] MultimodalMixin enabled:`
  log line at startup; data dir missing or no `.pt` files. Mixin
  fails open (no-op); training continues without modal aux.
* PPL diverges -- modal aux dominating; drop `--modal-alpha` to 0.02.
* Out of memory -- image encoder's per-sample tensor is large at
  high HxW. Either pre-resize images to <= 128x128 or drop image and
  keep only `--modal-list audio`.

---

## Phase 3 -- SFT response-only chat (ppl < 60)

**Trigger:** Validation ppl drops below 60. Base model is fluent
enough that instruction-following SFT bites without destroying
language ability (lesson from the v2.5 -> v2.6 wikitext leak: SFT on
top of a too-weak base bakes leakage rather than instruction-following).

**Launch command** -- this is a different trainer entry point:

```bash
python train_100m_sft.py \
    --base-ckpt /workspace/runs/synapforge_100m/step_XXXXXX.pt \
    --sft-data /workspace/data/alpaca_zh/alpaca_zh.json \
    --response-only-loss \
    --lr 1e-5 --steps 2000 --batch-size 16
```

**What changes from train_100m_kd:**

* CE loss masked to instruction-response delimiters (the
  `--response-only-loss` flag in the SFT trainer)
* Lower LR (`1e-5`, 30x smaller than Phase 0/1/2)
* No KD (`alpha_KD = 0`)
* No mixins (curiosity/self-learn/multimodal are training-phase tools;
  SFT is finishing-phase)
* Linear LR decay rather than cosine

**Expected effects:**
* CE on instruction-response pairs drops 2.0 -> 0.6 in ~1500 steps
* Chat eval (`chat_v25.py`): from raw-text completion to actual
  instruction following

**Metrics to watch:**
* SFT-CE on alpaca-zh held-out
* Chat eval pass rate on a 50-sample manual rubric (recipe in
  `examples/chat_eval.py`)

**Known failure modes:**
* Wikitext leakage -- chat outputs include training prompts verbatim;
  caused by `--response-only-loss` flag missing or improperly masking.
  Verify with `cat metrics.json | jq '.samples[-1]'`.
* Catastrophic forgetting of base LM -- if SFT runs > 5000 steps the
  base model loses fluency. Stop at the first val PPL minimum on a
  held-out language slice.

---

## Phase 4 -- RL GRPO with verifier (chat eval > 60% pass)

**Trigger:** Manual chat eval rubric passes more than 60% on a
50-sample test set. RL is fundamentally different from gradient-
descent KD/SFT; it can both lift and shatter the model so it only
makes sense once the base model is reliably correct.

**Launch command:**

```bash
python train_100m_rl.py \
    --base-ckpt /workspace/runs/sft/best.pt \
    --rl-grpo --rl-verifier sympy --rl-rollouts 8 \
    --reward-format math --steps 1000
```

**What turns on:**
* GRPO (Group Relative Policy Optimisation) -- 8 rollouts per prompt,
  reward computed by a SymPy/AST verifier (correctness only, no LLM
  judge)
* No KD, no mixins, no SFT loss

**Expected effects:**
* GSM8K accuracy 25% -> 40-50% in 500 steps
* MATH-500 accuracy 8% -> 15-22%

**Known failure modes:**
* Reward hacking -- rollouts find weird format that the SymPy verifier
  accepts; the model collapses to the format. Solution: stricter
  reward, require a final numeric answer in `\boxed{}`.
* KL drift -- rollouts diverge from base policy. Add KL penalty to
  base SFT model (DPO-style), beta=0.1.

---

## phase_manager.py integration

Run alongside the trainer to write the `.phase` signal at each
threshold crossing:

```bash
python scripts/phase_manager.py \
    --watch /workspace/runs/synapforge_100m \
    --interval 60 &
```

The manager polls `train.log` every 60s. On each ppl threshold
crossing it:

1. Reads `step` and `last_train_ppl` (and best `val_ppl` if any).
2. Decides phase id by walking the `PHASES` list in `phase_manager.py`.
3. If the phase changed, prints `>>> THRESHOLD CROSSED <<<` and
   writes `<watch_dir>/.phase`:

   ```json
   {
     "phase_id": 1,
     "phase_name": "intrinsic",
     "ts": 1714568400.0,
     "state": {"last_step": 4250, "last_train_ppl": 217.4, ...},
     "next_phase_flags": [
       "--intrinsic-curiosity",
       "--self-learn-ttt",
       "--stdp-novelty"
     ],
     "instructions": "Restart trainer with these flags appended to
                      launch_train.sh to enable the next phase. Existing
                      run continues; ckpt is reused."
   }
   ```

4. Phase 0 -> 1 is the only fully-automatic transition the
   manager supports today. Phase 1 -> 2 (multimodal) requires manual
   confirmation that the modal data dir is populated and the encoders
   are smoke-tested. Phase 2 -> 3 (SFT) and 3 -> 4 (RL) require manual
   review because they switch trainer scripts entirely.

The trainer itself does NOT poll `.phase` to live-toggle flags --
that would risk silent state corruption mid-run. Instead, the
contract is: `.phase` exists -> human reads it -> next launch picks
up the additional flags. This matches `feedback_no_train_hallucination.md`:
threshold crossings are observed, not assumed.

**Mapping flag <-> mixin** for cross-reference:

| flag                    | mixin            | active in        |
|-------------------------|------------------|------------------|
| `--curiosity-weight F`  | CuriosityMixin   | every train step |
| `--self-learn-ttt`      | SelfLearnMixin   | every val (probe only) |
| `--modal-list ...` + `--modal-data-dir ...` | MultimodalMixin | every train step |
| `--response-only-loss`  | (in SFT trainer) | Phase 3 only |
| `--rl-grpo`             | (in RL trainer)  | Phase 4 only |

All flags are default-off. Removing all flags reverts to Phase 0
behaviour byte-for-byte (verified by the smoke-test contract in
`synapforge/trainer_mixins.py::_run_all_smokes`).

---

## Quick references

* **Live phase decision** -- `python scripts/phase_manager.py
  --watch <run_dir>` prints current phase to stdout every 60s.
* **Manual phase override** -- write your own `.phase` JSON with
  the desired flags before launching the trainer; the trainer
  itself does not validate phase transitions.
* **Smoke-test mixins** -- `python -m synapforge.trainer_mixins`
  runs all 3 smokes on dummy tensors; exit code 0 = green.
* **Where to read more** --
  * `synapforge/trainer_mixins.py` for mixin internals
  * `synapforge/intrinsic.py` for free-energy / novelty / homeostatic
  * `synapforge/curiosity.py` for the 6-signal scorer
  * `synapforge/self_learn.py` for TTT, replay, MAML
  * `synapforge/modal/__init__.py` for the byte-patch encoders
  * `feedback_phased_training_2026q2.md` (memory) for the original
    user instruction that defined these phases.
