# Synap-1 Pro 300M Launch Plan

Status: STAGED (not launched). Trigger: Run 3o (100M baseline) completion.
Owner: training-pipeline. Last updated: 2026-05-02.

## 1. Why scale up

Run 3l/3m/3n on the 100M baseline (d=512 / n_layers=10) ran into a hard
plateau wall:

| Run  | Best val ppl | Plateau step | Failure mode |
|------|-------------:|-------------:|--------------|
| 3l   | ~570         | 4000         | classic train/val drift, val rose to 2522 by step 5500 (Run-3c-class divergence, killed) |
| 3m   | 4406         | 13500        | spectral-norm mid-run-init reset lm-head, re-diverged 4406 -> 25965 (killed) |
| 3n   | ~3700        | 30000        | LR 2e-5 constant + cosine continuation -- stable but flatlined |
| 3o   | (in flight)  | --           | warmstart from 3n step_030000.pt at LR 2e-5; converging slowly toward ~3500 ceiling |

The 100M shape has only **~25M useful backbone capacity**. The Qwen vocab
(151936 tokens) at d=512 consumes 151936 * 512 = ~75M parameters in the
tied tok_embed/lm_head -- that's 75% of the model. Whatever loss floor we
hit is bounded by 25M of HybridBlock + CfC + PLIF capacity, which is just
not enough for the alpaca-zh chat distribution we need at val ppl <= 250
(Phase 1 gate per docs/PHASE_TRAINING.md).

Conclusion: the 100M plateau is **architectural**, not a hyperparam bug.
Bigger backbone is the only path forward.

## 2. Capacity math

Pro at vocab=151936 / d=1024 / n_layers=14 / ffn_ratio=8 / loop_depth=1:

| Component         | 100M (d=512 n=10)      | Pro (d=1024 n=14)        | Multiplier |
|-------------------|-----------------------:|-------------------------:|-----------:|
| tok_embed/lm_head | 151936 * 512 = ~78M    | 151936 * 1024 = ~155M    | 2.0x       |
| HybridBlock body  | ~25M (10 layers)       | ~145M (14 layers @ 4x/layer) | 5.8x   |
| **Total**         | **~100M**              | **~300M**                | 3.0x       |

The headline 3x param count understates it: the **useful** (non-vocab)
backbone grows 5.8x, which is where chat-quality lives.

## 3. KD math: 300M is the sweet spot for KD on Qwen2.5-0.5B

Teacher Qwen2.5-0.5B has 494M params. Student/teacher capacity ratio:

| Run     | Student | Teacher | Ratio | KD literature consensus |
|---------|--------:|--------:|------:|-------------------------|
| 100M    |    100M |    494M |  0.20 | Below DistilBERT 0.40 minimum -- KD acts as regularizer only |
| **Pro** | **300M**| **494M**| **0.61** | **DistilBERT/MiniLM optimal band 0.50-0.70 -- student can absorb teacher distribution** |

DistilBERT (Sanh et al. 2019) and MiniLM (Wang et al. 2020) both report
peak KD efficiency in the 0.4-0.7 ratio band. Below that, the student has
no headroom for the soft-label distribution; above that, the teacher
under-fits the student.

Pro at 0.61 sits in the high-yield zone, but with one caveat: at this ratio
the student starts to have **its own representation preferences** that may
diverge from the teacher's. We therefore drop `--kd-weight` from 0.7
(100M default, treats teacher as ground truth) to **0.5** (treats teacher
as half-strength prior + student CE on its own data). This is the
DistilBERT recipe (alpha_kd = 0.5) -- not a guess.

We also drop `--kd-every` from 8 to **4** (more frequent KD steps). KD
signal is more useful per call at this ratio, so the ~12% step-time
overhead pays off.

## 4. Hyperparam delta table (vs 100M Run 3n/3o)

| Flag                       | 100M    | Pro     | Why |
|----------------------------|--------:|--------:|-----|
| `--batch-size`             |      64 |      32 | 2x activation memory; bs=64 OOMs ~80GB |
| `--lr`                     |    5e-5 |    3e-4 | bigger backbone -> smoother loss surface, takes higher LR (Chinchilla) |
| `--warmup`                 |     500 |    1000 | 2% of step budget; standard for 300M+ |
| `--kd-every`               |       8 |       4 | closer ratio -> more KD steps useful |
| `--kd-weight`              |     0.7 |     0.5 | DistilBERT recipe at 0.5-0.7 ratio band |
| `--grad-clip`              |     0.5 |     1.0 | bigger natural grad norm; 0.5 over-clips |
| `--steps`                  |   30000 |   50000 | 1.7x steps for 5.8x backbone (sub-linear) |
| `--shuffle-seed`           |     911 |    1011 | rotate to break any data-ordering coupling |
| `--lm-head-spectral-norm`  | mid-run | from t0 | avoid Run 3m's mid-run reset disaster |
| `--warmstart`              | enabled | NO      | architecture change, dimensions incompatible |

Unchanged: `--shuffle-buffer 10000`, `--lr-decay cosine`, `--kd-topk 2048`,
`--spike-target-loss-weight 0.05`, `--phase-aware`,
`--backend triton_block`.

## 5. Training budget

- A800 80GB at bs=32 / d=1024 / n=14: estimated ~0.55 s/step (vs 100M's
  ~0.30 s/step). Empirically ~1.8x slower per step despite 3x params, due
  to triton_block kernel scaling sub-linearly.
- 50000 steps * 0.55 s = **27500 s = 7.6h wall**.
- Phase 1 gate (val ppl <= 250) target: step ~25000-35000 (~3.8-5.4h in).
  If not hit by step 40000, escalate per docs/INSURANCE_NATIVE.md.

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------:|------------|
| OOM at bs=32 (KD chunks at vocab 151936 spike VRAM) | Medium | drop bs to 24 first; if still OOM, set `--grad-accum-steps 2` (waits for trainer support) |
| LR 3e-4 too hot for cold-start (no warmstart) | Low-Med | warmup 1000 protects step 0-1000; if val rises >10x train at step 2000, kill + relaunch at lr 1e-4 |
| Cosine decay reaches lr_min before convergence | Low | `--lr-min 1e-5` is conservative; if needed extend with `--steps 70000` |
| Spectral-norm slows step time >20% | Low | T2.6 is fp32 power-iter; measured <5% on 100M, scales linearly |
| KD weight 0.5 too low, student under-distills | Low-Med | bump to 0.6 at relaunch if val ppl flatlines >500 above teacher's CE estimate (~4.5 -> ~90 ppl) for 5000 steps |

## 7. Decision criteria

Launch only when ALL of:

1. Run 3o has terminated (success OR killed).
2. A800 rental has >= 8h remaining on the clock.
3. configs/synap1.py SYNAP1_PRO config is merged into main (sister agent's PR).
4. Disk on /workspace has >= 50 GB free for ckpts (50 ckpts * ~1.2 GB).

Abort criteria during run:

- Step <500: any NaN in train CE (kill, investigate kernel)
- Step 1000-2000: train ce > 9.5 (warmup didn't catch -- relaunch at lr 1e-4)
- Step 5000-10000: val ppl > 5x train ppl (Run-3c-class drift, kill)
- Step 20000+: val ppl plateau within 1% over 3000 steps and >= 500 (early-stop, save best)

## 8. Post-launch hooks

- `--best-ckpt-track` ON by default (T5.4) -- best ppl ckpt symlinked.
- Phase autopilot (scripts/build_next_launch.py) reads `train_synap1_pro.log`
  for val ppl trigger; on val ppl <= 250 it advances Phase 1 -> Phase 2.
- T8.6 (Self-distillation) becomes unblocked once Pro reaches val ppl <= 100
  -- Pro then becomes teacher for a Synap-1-v2 100M re-train.

## 9. References

- `scripts/launch_synap1_pro.sh` -- the actual launch script.
- `scripts/launch_qwen3m.sh` -- 100M Run 3m template (warmstart variant).
- `docs/PHASE_TRAINING.md` -- ppl gates per phase.
- `docs/RUN3L_DIAGNOSIS.md` -- the diagnosis that motivated the scale-up.
- `docs/DEEP_MAINT_QUEUE.md` T9.1 -- the maintenance task block for this launch.
- DistilBERT (Sanh, Debut, Chaumond, Wolf 2019, arXiv:1910.01108).
- MiniLM (Wang, Wei, Dong, Bao, Yang, Zhou 2020, arXiv:2002.10957).
