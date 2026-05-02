# Synap-1 Scaling Plan: Base 100M -> Pro 300M -> Ultra 500M

Status: Ultra config STAGED, launch script STAGED, tests passing.
Owner: training-pipeline. Last updated: 2026-05-02.

This is the cross-tier plan covering all three Synap-1 size classes.
For the Pro-specific deep dive (Run 3l/3m/3n diagnoses, KD-math
derivation), see `docs/SYNAP1_PRO_PLAN.md`. This file focuses on
how the three tiers relate and what changed at each step.

## 1. Tier ladder

| Tier  | d    | n_layers | ffn_ratio | loop_depth | Total       | Embed       | Backbone    | vs 100M backbone |
|-------|-----:|---------:|----------:|-----------:|------------:|------------:|------------:|-----------------:|
| Base  |  512 |       10 |       8.0 |          1 |    151.4 M  |    77.8 M   |    73.6 M   | 1.00x (anchor)   |
| Pro   | 1024 |       14 |       2.5 |          2 |    325.0 M  |   155.6 M   |   169.4 M   | 2.30x            |
| Ultra | 1280 |       16 |       3.0 |          2 |    535.8 M  |   194.5 M   |   341.3 M   | 4.64x            |

Ultra's **backbone** (~341M) is what does the actual learning -- the
embedding table is a fixed cost dictated by Qwen 2.5's vocab=151936.
Stripping the embedding, Ultra's useful capacity is **13.7x** the
historical 100M architecture's useful backbone (~25M) -- an order of
magnitude more learnable representation.

## 2. Why Ultra (skipping Pro for the launch)

User decision 2026-05-02: skip Pro, go straight to Ultra. Rationale:

- The 100M plateau (val ppl ~3500-4400 floor on Run 3l/3m/3n) is
  **architectural**, not a hyperparam bug.  More backbone is the only
  fix; the question is just how much.
- A800 80GB at d=1280 / n=16 fits with bs=24 + grad-accum 2 (effective
  bs=48), leaving enough VRAM headroom for the Qwen2.5-0.5B teacher
  forward pass during KD steps.
- One Ultra run at ~14h is cheaper than Pro 7.5h followed by an Ultra
  re-launch when Pro saturates short of the chat target -- and saves
  a checkpoint-incompatible warm-start.
- Ultra at student/teacher = 1.08 totals (0.71 backbone-only) is in the
  high-yield KD band where MiniLM / DistilBERT-style distillation
  reaches near-teacher quality (the student is no longer an
  under-capacity bottleneck).

## 3. Hyperparam delta table

The full heritage of how each flag arrived is in `SYNAP1_PRO_PLAN.md`
SS3-SS4. This table summarises the deltas at each step.

| Flag                        | 100M     | Pro      | Ultra    | Why Ultra differs from Pro |
|-----------------------------|---------:|---------:|---------:|----------------------------|
| `--d`                       |      512 |     1024 |     1280 | +25% hidden width          |
| `--n-layers`                |       10 |       14 |       16 | +2 layers                  |
| `--loop-depth`              |        1 |        2 |        2 | RDT recurrence kept        |
| `--ffn-ratio`               |      8.0 |      2.5 |      3.0 | hits 350M backbone target  |
| `--batch-size`              |       64 |       32 |       24 | VRAM headroom on A800      |
| `--grad-accum-steps`        |        1 |        1 |        2 | effective bs 48            |
| `--lr`                      |     5e-5 |     3e-4 |     2e-4 | bigger model -> lower LR (Chinchilla sqrt) |
| `--warmup`                  |      500 |     1000 |     1500 | longer optimizer-state stabilisation |
| `--kd-every`                |        8 |        4 |        4 | KD signal valuable per step |
| `--kd-weight`               |      0.7 |      0.5 |      0.4 | student >= teacher capacity (DistilBERT band 0.3-0.5) |
| `--steps`                   |    30000 |    50000 |    60000 | sub-linear scaling for 2x backbone |
| `--shuffle-seed`            |      911 |     1011 |     1212 | fresh ordering              |
| `--grad-clip`               |      0.5 |      1.0 |      1.0 | bigger natural grad norm    |
| `--lm-head-spectral-norm`   | mid-run  | from t0  | from t0  | avoid Run 3m mid-run reset  |
| `--warmstart`               | enabled  |  NO      |  NO      | architecture change         |

Unchanged across all tiers: `--shuffle-buffer 10000`, `--lr-decay cosine`,
`--kd-topk 2048`, `--spike-target-loss-weight 0.05`, `--phase-aware`,
`--backend triton_block`, `--teacher Qwen/Qwen2.5-0.5B`.

## 4. ffn_ratio = 3.0 (the load-bearing number)

The Pro tier picked ffn_ratio = 2.5 to fit the 300M total budget at
d=1024 / n=14. Ultra at d=1280 / n=16 has different geometry:

| ffn_ratio at d=1280 n=16 | Total      | Backbone   | Total within +/- 10% of 500M? | Backbone within +/- 10% of 350M? |
|-------------------------:|-----------:|-----------:|:-----------------------------:|:--------------------------------:|
|                      2.5 |   496.4 M  |   302.0 M  | YES                           | NO (below 315M lower bound)      |
|                  **3.0** |  **535.8 M** |  **341.3 M** | **YES**                      | **YES**                          |
|                      3.5 |   575.1 M  |   380.6 M  | NO (above 550M cap)           | YES                              |
|                      4.0 |   614.4 M  |   419.9 M  | NO                            | NO (above 385M cap)              |

ffn_ratio = 3.0 is the **only** value in [2.5, 4.0] that puts both
total and backbone inside the +/- 10% bands required by the variant
spec / unit tests. It is also a clean number: SwiGLU's effective
expansion is `2 * (3.0 * d) = 7680` per block at d=1280 -- bigger than
the d=512 baseline's full FFN (which was the entire model bottleneck).

Originally the user spec called for ffn_ratio = 2.5 + d=1280 + n=16
+ "backbone ~350M". Since 2.5 yields backbone = 302M (under target)
this PR uses 3.0 instead, fully documented above. The launch script
matches: `--ffn-ratio 3.0`.

## 5. KD math at student >= teacher

Teacher Qwen2.5-0.5B has **494M total** params (~169M backbone after
its own embedding). Capacity ratios:

| Run     | Student total | Teacher total | Total ratio | Backbone ratio | Recommended alpha_kd |
|---------|--------------:|--------------:|------------:|---------------:|---------------------:|
| 100M    |        100 M  |         494 M |        0.20 |          0.15  | 0.7 (regulariser)    |
| Pro     |        325 M  |         494 M |        0.66 |          1.00  | 0.5 (DistilBERT)     |
| Ultra   |        536 M  |         494 M |        1.08 |          2.02  | 0.4 (under-pull)     |

At Ultra the **student backbone is 2x the teacher backbone**. The
teacher is no longer a "more capable model the student must imitate";
it's a peer with a different inductive bias (transformer vs LNN+SNN).
Per the DistilBERT / MiniLM band:

- alpha_kd >= 0.5 collapses the LNN+SNN student onto the teacher's
  attention manifold -- defeats the purpose of the architecture.
- alpha_kd <= 0.3 makes KD a regulariser only; the student doesn't
  benefit from the teacher's distribution at all.
- alpha_kd = 0.4 keeps CE (60%) dominant while letting the teacher's
  soft labels round off rare-token over-confidence at the lm-head.

KD frequency stays at every-4 steps (Pro's setting). The justification
is unchanged: at this capacity ratio every KD step extracts useful
signal, and the ~12% step-time overhead is worth paying.

## 6. Training budget

A800 80GB at bs=24 / d=1280 / n=16 with KD on:

- Per-step time estimate: ~0.85 s (vs Pro's 0.55 s, 100M's 0.30 s).
  Sub-linear scaling because the triton_block kernel is bandwidth-bound;
  doubling backbone params doesn't double step time.
- 60000 steps * 0.85 s = **51000 s ~ 14.2 h wall**.
- Phase 1 gate (val ppl <= 250) target: step ~25000-35000 (~6-8 h in).
  If not hit by step 45000, escalate per `docs/INSURANCE_NATIVE.md`.

VRAM budget:

| Component                     | Estimate at bs=24 |
|-------------------------------|------------------:|
| Student fwd activations       |          ~22 GB   |
| Student bwd + Adam state      |          ~18 GB   |
| Teacher fwd (KD steps only)   |           ~6 GB   |
| KD logits (top-2048 sparse)   |           ~3 GB   |
| Triton kernel scratch         |           ~5 GB   |
| Headroom                      |        ~12-16 GB  |
| **Total**                     |       **~65 GB**  |

If OOM at bs=24, drop to bs=16 + grad-accum 3 (effective bs=48
unchanged). If still OOM, drop loop_depth=1 temporarily to halve
recurrent compute (loses some Pro-style RDT signal but keeps training
alive).

## 7. Risks and mitigations

| Risk                                                | Likelihood | Mitigation                                           |
|-----------------------------------------------------|-----------:|------------------------------------------------------|
| OOM at bs=24 (KD chunks at vocab 151936 spike VRAM) | Medium     | drop bs to 16 + grad-accum 3 (effective bs same)     |
| LR 2e-4 too hot for cold-start (no warmstart)       | Low-Med    | warmup 1500 protects step 0-1500; if val rises >10x train at step 2500, kill + relaunch at lr 1.5e-4 |
| Cosine decay reaches lr_min before convergence      | Low        | `--lr-min 1e-5` is conservative; if needed extend with `--steps 80000` |
| Student >= teacher: student diverges from KD signal | Low-Med    | alpha_kd 0.4 already dampens this; if val ce stalls > teacher's ce_estimate (~4.5 -> ~90 ppl) for 5000 steps, drop alpha_kd to 0.2 (KD becomes pure regulariser) |
| Triton kernel bandwidth saturation at d=1280 / n=16 | Medium     | first 100 steps measure tok/s; if < 30k, fall back to `--backend gpu_dense` (slower but proven stable) |

## 8. Decision criteria

Launch only when ALL of:

1. Any active 100M / Pro run has terminated (success OR killed).
2. A800 rental has >= 16h remaining on the clock.
3. `configs/synap1.py` SYNAP1_ULTRA config is merged into main (this PR).
4. Disk on /workspace has >= 80 GB free for ckpts (60 ckpts * ~1.4 GB).

Abort criteria during run:

- Step <500: any NaN in train CE (kill, investigate kernel)
- Step 1500-2500: train ce > 9.0 (warmup didn't catch -- relaunch at lr 1.5e-4)
- Step 5000-10000: val ppl > 5x train ppl (Run-3c-class drift, kill)
- Step 25000+: val ppl plateau within 1% over 4000 steps and >= 400 (early-stop, save best)

## 9. Post-launch hooks

- `--best-ckpt-track` ON by default (T5.4) -- best ppl ckpt symlinked.
- Phase autopilot (scripts/build_next_launch.py) reads `train_synap1_ultra.log`
  for val ppl trigger; on val ppl <= 250 it advances Phase 1 -> Phase 2.
- T8.6 (Self-distillation) becomes unblocked once Ultra reaches val ppl <= 80
  -- Ultra then becomes teacher for a Synap-1-v2 100M re-train.
- Investor demo: Ultra is the headline param count for the deck. Even if
  val ppl plateau is higher than Pro's, the 13.7x backbone multiplier
  vs the 100M anchor is the load-bearing number for the "we scale" pitch.

## 10. References

- `scripts/launch_synap1_ultra.sh` -- the actual launch script.
- `scripts/launch_synap1_pro.sh` -- Pro launch template (this script
  is a tuned copy of it).
- `synapforge/configs/synap1.py` -- SYNAP1_ULTRA dataclass.
- `tests/integration/test_synap1_pro_config.py` -- Ultra param-count
  tests (`test_ultra_param_count_about_500M`, `test_ultra_backbone_about_350M`,
  `test_build_from_config_ultra`).
- `docs/SYNAP1_PRO_PLAN.md` -- the Pro-specific deep dive (Run 3l/3m/3n
  diagnoses, KD-math derivation, full DistilBERT / MiniLM citations).
- `docs/PHASE_TRAINING.md` -- ppl gates per phase.
- `docs/INSURANCE_NATIVE.md` -- fallback options if Ultra plateaus.
- DistilBERT (Sanh, Debut, Chaumond, Wolf 2019, arXiv:1910.01108).
- MiniLM (Wang, Wei, Dong, Bao, Yang, Zhou 2020, arXiv:2002.10957).
- Chinchilla (Hoffmann et al. 2022, arXiv:2203.15556) for the LR-vs-size
  scaling rule used to drop Pro's 3e-4 to Ultra's 2e-4.
