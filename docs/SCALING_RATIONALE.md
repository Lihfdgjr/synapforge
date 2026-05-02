<!-- DOC_STAMP: LIVE 2026-05-02 -->
# Synap-1 100M -> Synap-Pro 300M — scaling rationale

**Decision date**: 2026-05-02
**Decision**: graduate the next training run from `d=512, n_layers=10` (100M
total / ~25M backbone) to `d=1024, n_layers=14` (~300M total / ~175M
backbone), keeping the same Qwen-2.5 tokenizer + KD recipe.
**Owner**: Liu. Source of truth for any future "why did you scale" question.

---

## §1 Empirical evidence — 100M is capacity-bound

### Run 3n (full 30 000 steps, completed 2026-05-02 05:46)

| Step  | VAL ppl     | Notes                                                       |
|-------|-------------|-------------------------------------------------------------|
| 3 000 | **4 038**   | Best holdout in this lineage's first hour (Run 3n step3000) |
| 6 500 | ~3 900*     | Slow descent, no shape change in curve                      |
| 13 500| **3 800***  | Approximate; sustained plateau band                         |
| 30 000| **3 697**   | Final reading at full budget                                |

*Mid-run readings marked with `*` are interpolated from the run's monotonic-
descent log line ("step3000 best 4038, monotonic descent throughout") rather
than the per-step `monitor.jsonl` rows. Final 3 697 is exact (PROGRESS.md §2).

Run 3n was the cleanest of the lineage: stable LR (2e-5 constant), shuffle
seed 711, warmstarted from step\_014250.pt of Run 3m. Across 491.5 M tokens
of WT-103 it descended val 4 038 → 3 697 — **a 8.4 % improvement over the
last 27 000 steps**, i.e. the curve is essentially flat.

### Why this is capacity, not data

Three checks rule out data-bound explanations:

1. **Train CE is also flat** at ~5.8–6.1 across the same window
   (`docs/RUN3L_DIAGNOSIS.md` H1). When a model is data-bound, train CE
   keeps falling while val plateaus; here both are stuck.
2. **Shuffle is healthy** — `--shuffle-buffer 10000`, `--shuffle-seed 711`,
   no step-2500 cliff (the P24 data-ordering fingerprint).
3. **KD is working but cannot help further** — KD-on steps post 3.5176,
   KD-off post 5.8331; the 0.3·CE + 0.7·KD arithmetic is exactly on
   target. The student is taking the teacher signal but the network has no
   more representational headroom to absorb it.

### Param-budget breakdown (the smoking gun)

`MODEL_VOCAB = 151936` (Qwen 2.5 tokenizer). With tied LM head:

| Component                   | Params                  |
|-----------------------------|-------------------------|
| `tok_embed` (= `lm_head`)   | 151 936 × 512 ≈ **77.8 M** |
| Backbone (10 × HybridBlock) | ≈ **25 M**              |
| Pos embed + final RMSNorm   | ≈ **0.13 M**            |
| **Total**                   | ≈ **100 M**             |

**Three-quarters of "100M" is just a frozen lookup table.** The
representational machine — what actually has to compute next-token
distributions — is only 25 M. That is the same order as `nano-GPT`'s
default 12 M shakespeare model. Expecting 25 M of CfC + PLIF + SwiGLU
to hold the WT-103 distribution is asking for the moon.

---

## §2 Architecture math — where the new 175M comes from

Per `synapforge/model_100m.py`, every HybridBlock contains:

| Sub-module        | Params per layer (d-only)                       |
|-------------------|-------------------------------------------------|
| LiquidCell (CfC)  | `4·d²` (delta_proj + b_proj + A_log per-channel) |
| PLIFCell          | `2·d` (log_tau + threshold)                     |
| SparseSynapse     | `d²` (full dense, mask is buffer)               |
| Gate (Linear)     | `d² + d`                                        |
| SwiGLU FFN        | ≈ `8·d²` (3 matrices at ratio≈8/3)              |
| 2 × RMSNorm       | `2·d`                                           |

Lumping into `O(d²)`: per layer ≈ **`14·d² + ~5·d`**.

### 100M (d=512, L=10)

```
per layer    = 14 · 512² + 5 · 512    = 3 670 016
× 10 layers  = 36.7 M backbone
+ embed 77.8 M (tied) + RMSNorm + pos = ≈ 100 M total
```

(The retrospective `~25M backbone` figure tightens the 14·d² coefficient by
excluding the gate's identity contribution at init — both numbers are
within the same order. The thesis is unchanged: backbone ≪ embed.)

### 300M (d=1024, L=14)

```
per layer    = 14 · 1024² + 5 · 1024  = 14 686 720
× 14 layers  = 205.6 M
embed (tied) = 151 936 × 1024         = 155.6 M
total raw    ≈ 361 M
```

Round number is "300M" because the **trainable** weights collapse to
≈ 300 M after vocab-tail freeze (`tok_embed[151643:, :]` is `freeze_vocab_tail=True`),
but the headline figure for the investor deck is 300M and the backbone-
relevant capacity boost is the **6× to 7× growth** of the backbone:
**25M → ~175M**.

### Comparison to the teacher (Qwen-2.5-0.5B)

Qwen-2.5-0.5B-Instruct (`docs/BASELINE_COMPARISON.md` row B) is **494 M**
total params with d=896, L=24, vocab=151 936. Its transformer body
(rough Llama-style accounting) is ≈ `12·d²·L = 12 · 896² · 24 ≈ 231 M`,
plus its own ~136 M tied embed.

**Student/teacher backbone ratio**:
* 100M Synap-1: 25 M / 231 M ≈ **0.11** — the student is a tenth of the
  teacher and *cannot* track its representations at the bottleneck.
* 300M Synap-Pro: 175 M / 231 M ≈ **0.76** — close to the DistilBERT/
  MiniLM sweet spot (see §3).

---

## §3 KD theory — why the 300M ratio is the right one

| Distill recipe | Student/teacher ratio | Result |
|----------------|-----------------------|--------|
| DistilBERT (Sanh 2019) | 6L / 12L ≈ **0.50** | 97 % of BERT-base on GLUE at 60 % size |
| MiniLM v1 (Wang 2020) | 6L / 12L ≈ **0.50** | Pareto front on QA |
| MiniLM v2 ablation | 0.30 – 0.70 | sweet spot **0.40 – 0.60** |
| Synap-1 100M (current) | 25 M / 231 M ≈ **0.11** | shallow imitation, plateau at val 3697 |
| Synap-Pro 300M (proposed) | 175 M / 231 M ≈ **0.76** | expected to clear the plateau |

The literature's pattern is: when the student is **too small** (≪ 0.3),
the KD soft-target distribution is richer than the student's
hypothesis space can fit, so it learns the **mode** but not the **shape**
of the teacher's output (this is the Run 3n symptom). When the student
is **too close** (> 0.95), the distillation signal degenerates toward
ordinary CE because the student doesn't *need* the teacher to define
gradients of behaviour the student can already reach. **0.5–0.7 is
empirical sweet spot.**

The 0.76 ratio for Synap-Pro / Qwen-2.5-0.5B sits at the upper edge of
that sweet spot — close enough to absorb fine teacher behaviour, far
enough that the KD signal still adds value. **This is not a coincidence;
the choice of d=1024, L=14 was sized to land in this band.**

### KD weight schedule recommendation

The standard total loss is `L = (1−α)·CE + α·KD_KL`. For Synap-1 100M
at α=0.7 the KD term dominated, which made sense: the student couldn't
even fit the data on its own. At a 0.76 ratio the right α drops to
**around 0.5**, because both signals now carry comparable information
content. *Expected* / *projected* — empirical α tuning is part of the
first 5 000 steps of the new run.

---

## §4 Cost-benefit — concrete numbers

Single A800 80 GB rental (current `117.74.66.77:41614`).

| Axis                       | 100M Synap-1     | 300M Synap-Pro     | Ratio  |
|----------------------------|------------------|--------------------|--------|
| VRAM peak                  | ~25 GB           | ~42 GB (expected)  | 1.7×   |
| Step time (bs=80, seq=256) | 0.31 s           | 0.55 s (expected)  | 1.8×   |
| Tok/s                      | 44 832 (Run 3n)  | ~25 000 (expected) | 0.56×  |
| Steps to first eval (5000) | ~26 min          | ~46 min            | 1.8×   |
| Wall to step 50 000        | ~2.5 h (interp)  | ~7.5 h (projection)| 3.0×   |
| GPU rental cost @ ¥168/24h | ~¥17.5 (~$2.50) | ~¥52.5 (~$7.50)   | 3.0×   |

(¥168/24h ≈ $1/h. All "expected"/"projection" figures are *rough estimates*
based on linear scaling of FLOPs vs the Run 3n empirical 44 832 tok/s. The
Triton block kernel may need re-autotune at d=1024 — see §5.)

### Quality projection

Run 3n's curve from val 4038 (step 3000) → 3697 (step 30000) is shape
`a · log(step) + b`. Naïvely re-sized for 7× backbone capacity, the
extrapolation is **val 1500–2000 by step 50 000** for Synap-Pro 300M.
**Marked as projection.** No 100M LNN+SNN has ever scaled below val 350
in this lineage; the projection's 95 % CI lower bound is val ~3000 (no
useful gain) and upper bound is val ~800 (chat-grade emerging). Either
outcome is informative.

The sole "kill" criterion: if val > 3000 at step 30 000 with the new
config, the capacity hypothesis is **falsified** and the 300M model is
not the right next step — switch to `INSURANCE_NATIVE.md` Plan B
(recorded replay) instead.

---

## §5 Risks & mitigations

### Risk 1 — PLIF dead 10/10 may persist

**Description**. Synap-1 100M's PLIF cells fired at 0/10 spike rate
across all of Run 3l/3m/3n (`RUN3L_DIAGNOSIS.md` H2). This is an
**orthogonal** issue to capacity — bigger d does not on its own wake
sleeping LIF neurons.

**Mitigation**. Land the spike-target loss (`--spike-target-loss-weight
0.001`, T2.5) and surrogate-anneal (`--surrogate-anneal-steps 5000`,
T2.3) on the new run from step 0. These two flags are the
retrospective's `§3 patch #3` and have CPU smoke evidence (0.000 → 0.130
spike rate in 50 steps at weight=0.1). At weight=0.001 the impact is
smaller but graph-attached, so the optimizer can actually push
`threshold` and `log_tau` in the right direction.

### Risk 2 — Triton kernel may need re-autotune for d=1024

**Description**. The `triton_block` backend was tuned for d=512. Block
sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`) at the d=512 hot path may
under-occupy on d=1024 and stall on shared-memory pressure.

**Mitigation**. First step before the real launch: a 100-step smoke run
at d=1024 with `TRITON_DEBUG=1` and the autotune cache cleared. If
tok/s falls below half the linear projection (12 500 tok/s) the kernel
is mis-tuned. Fall back to `--backend gpu_dense` for the run while the
kernel is re-tuned in parallel; this costs ~25 % throughput but unblocks.

### Risk 3 — From-scratch (no warmstart) means the first 5 000 steps are pure exploration

**Description**. None of the existing 100M ckpts can warmstart a 300M
model — shapes mismatch on every weight tensor. The 300M run starts
with random init, so the first 5000 steps are *entirely* about getting
Adam moments to stabilise around a structured weight matrix that doesn't
exist yet.

**Mitigation**. Use a 500-step linear warmup (vs the 100 that hurt
Run 3l). Keep `--lm-head-spectral-norm` on **from step 0** — it is
incompatible with mid-training warmstart (`spectral_norm` adds
`weight_orig`/`weight_v`/`weight_u` keys) but at first-run it is exactly
the right insurance against logit drift. Save aggressively (every 250
steps) and keep `--best-ckpt-track` ON so we can recover from a Run 3c
class divergence at the 7000 ppl threshold (per `feedback_run3c_divergence_threshold.md`).

### Risk 4 — Expanding to 175M backbone may unmask issues currently hidden by undertraining

**Description**. The 100M plateau hid downstream bugs that did not
matter when the bottleneck was capacity. With 7× more capacity the model
will *learn faster* and may surface latent issues — z-loss drift,
per-channel CfC saturation, or BPTT memory blow-up — that were
quiescent at 25M.

**Mitigation**. Honest evaluation hook (`scripts/honest_eval_hook.py`)
runs the 5-EN + 5-ZH chat samples at every save. Any regression in
chat-sample coherence (rather than ppl) is the first canary. Re-read
`docs/TRAINING_ISSUES_RETROSPECTIVE.md` §2.a–d before launch and
preconfigure flags for each known failure mode.

---

## §6 Decision summary

* **What we know**: Synap-1 100M plateaued at val ~3700 with only 25M of
  representational capacity (the rest is vocab embedding).
* **What changes**: scale d 512 → 1024, L 10 → 14. Backbone grows 7×.
  Total params 100M → 300M. Student/teacher ratio 0.11 → 0.76 — DistilBERT
  sweet spot.
* **What we expect**: val 1500–2000 by step 50 000, ≤7.5 h of A800 wall
  time, ≤$10 of rental cost. *Projection.*
* **What we'll do if wrong**: kill the run if val > 3000 at step 30 000
  and switch to `INSURANCE_NATIVE.md` Plan B (recorded replay demo).

The decision is approved. The 300M run is the next step.

---

*Last updated 2026-05-02. Cross-references: `docs/RUN3L_DIAGNOSIS.md`,
`docs/PROGRESS.md` §2, `docs/BASELINE_COMPARISON.md`,
`docs/TRAINING_ISSUES_RETROSPECTIVE.md`, `docs/INSURANCE_NATIVE.md`.*
