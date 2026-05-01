# Multi-modal training -- 9 modalities, native byte-patch, anti-fakery enforced

This document is the runbook for `train_multimodal.py`. The bet: a SynapForge100M
backbone, warmstarted from a text-only ckpt, can be extended to **all 9 modalities**
in roughly **24-36 GPU-h on a single A100** by following the phase schedule below.

The non-negotiable contract (memory `feedback_native_multimodal_required.md`):

- No frozen vision encoder (no LLaVA/CLIP/ViT projection-only adapter).
- Every modality enters the backbone via `synapforge.modal.UnifiedEmbed`,
  which is a **byte-patch + Linear** projection -- Fuyu/Chameleon style.
- The backbone (CfC + PLIF + SwiGLU + sparse synapse, 10-layer 512-dim) does
  ALL cross-modal mixing on its unified token stream.
- An **anti-fakery probe** runs every 500 steps. If zeroing the modal hidden
  tensors does NOT collapse caption quality, the run is flagged FAIL and you
  must keep training -- the model isn't actually using the modality yet.

## Quick smoke

About 30-60 s on CPU -- validates code-path end-to-end without real data:

```
python scripts/prep_multimodal_data.py --smoke --n-train 32 --n-val 4
python train_multimodal.py --smoke --steps 50 --bs 2
```

## Phase schedule

The honest budget for a real run:

| Phase | What | GPU-h | Eval gate |
|-------|------|-------|-----------|
| 0 | Warmstart text ckpt + freeze backbone, train modal encoders + LM head row only | 4 | InfoNCE drops monotonically across all 9 modalities |
| 1 | Unfreeze last 4 HybridBlocks, joint multitask `LM + Σ_m α·contrastive` | 12 | Anti-fakery ratio > 1.5x for >= 7 of 9 modalities |
| 2 | SFT on LLaVA-Instruct-150K + M3IT 50K (or smaller subsets) | 8 | Caption BLEU > 0.05 on val captions |
| 3 | Eval (MMMU/MathVista/AudioBench/VideoMME) + final anti-fakery sweep | 1 | Targets below |
| **total** |  | **~25** |  |

A "5T-effective" full run (memory `feedback_5T_effective_target.md`) adds 311 more
GPU-h (LLaVA + ScienceQA + Video-Instruct + ECG + ZINC SFT). Both fit on one A100
80 GB with `--use-grad-checkpoint`.

### Phase 0 -- modal encoders only (4 GPU-h)

```
python train_multimodal.py \
  --warmstart runs/synapforge_100m/best.pt \
  --phase 0 \
  --steps 5000 --bs 4 --seq 512 \
  --data data/multimodal --out runs/multimodal_phase0
```

What is trained: `UnifiedEmbed.*PatchEmbed.*` (per-modality linear projections,
markers, position encodings) + the **first 9 rows of the LM head** corresponding
to the new `<|image|>...<|biosignal|>` special tokens (vocab 50257 -> 50266).
Backbone is frozen.

Why: gives the modal encoders something cheap to do before unleashing the
backbone, prevents the backbone from collapsing to pure text language modelling
when modal tokens flood in. Lr 2e-4, cosine to 1e-5, warmup 200 steps.

Expected losses at end of phase 0:

- `cap_loss`: drops from ~10.5 to ~4.5 (next-token CE on captions).
- `con_loss`: drops from ~2.0 (random) to ~0.8 (8 modalities x InfoNCE bs=4).

### Phase 1 -- unfreeze last 4 backbone blocks (12 GPU-h)

```
python train_multimodal.py \
  --warmstart runs/multimodal_phase0/best.pt \
  --phase 1 \
  --steps 24000 --bs 4 --seq 512 \
  --data data/multimodal --out runs/multimodal_phase1
```

The full multitask loss. Each step picks one of 9 modalities round-robin and
adds an InfoNCE term between text-hidden and modal-hidden, alpha=0.05 per
modality (so 9*0.05 = 0.45 on top of the ~4.5 LM CE -- aux dominates only if
LM CE has already converged).

Why only last 4 blocks: unfreezing all 10 blocks at once causes the backbone to
forget text in the first 200 steps. Last-4 is the SAME recipe as Chameleon and
LLaVA-1.5; the lower 6 layers stay frozen as a "frozen text feature extractor".

Hyperparams: `lr=1e-4`, `bs=4`, `seq=512`, `wd=0.01`. Anti-fakery probe
every 500 steps; expected to hit ratio > 1.5x on **image and audio first**,
then graph and time_series, then video (slowest because 4-frame temporal
patches double the per-step compute).

### Phase 2 -- multimodal SFT (8 GPU-h)

Real LLaVA-Instruct-150K + M3IT (or 50K subsets):

```
python train_multimodal.py \
  --warmstart runs/multimodal_phase1/best.pt \
  --phase 2 --steps 16000 --bs 8 --seq 1024 \
  --data data/multimodal_sft --out runs/multimodal_phase2
```

Phase 2 unfreezes everything. The data prep script needs an SFT branch
(not yet wired in `prep_multimodal_data.py` -- left as a follow-up because
the 150K download is ~80 GB).

### Phase 3 -- eval + sign-off (1 GPU-h)

| Benchmark | Modality | Target (memory `feedback_5T_effective_target.md`) |
|-----------|----------|---------------------------------------------------|
| MMMU | image+text reasoning | >= 30% |
| MathVista | image+math | >= 25% |
| AudioBench-MMAU | audio comprehension | >= 35% |
| VideoMME | video QA (short) | >= 30% |
| **Anti-fakery** | every modality | ratio >= 1.5x in >= 7 of 9 |

The eval suite ships in `synapforge/eval/` (gpqa.py, mmlu.py existing; modal
suites are stubbed; real grading is left as `bench/eval_modal.py` follow-up).

## Anti-fakery contract -- the load-bearing test

```
ratio = caption_loss(zeroed_modal_hidden) / caption_loss(real_modal_hidden)
PASS  iff ratio >= 1.5
```

If the model is FAKING (i.e. ignoring the modality and just guessing captions
from text priors), zeroing the modal embeds will not change loss meaningfully.
If the model is REAL, zeroing collapses caption quality.

Threshold rationale: 1.5x is the minimum that survives noise. LLaVA-1.5
hits ~3.0x on image. Chameleon hits ~5.0x. Below 1.2x = the modality
is doing nothing. Between 1.2 and 1.5x = weak signal -- keep training.

If the probe FAILS at end of phase 1 for any modality:

1. **Train more on that modality**: bias the round-robin sampler so the
   weak modality is sampled 2-4x more often (`--modal-weights '...:4.0'`,
   not yet implemented; manually duplicate parquet rows).
2. **Inspect the encoder**: a runtime bug in `*PatchEmbed` (e.g. wrong
   shape unpack) will silently produce zeros that the contrastive loss
   tolerates because all batch members are zero.
3. **Backbone has degenerated**: if ALL modalities fail at once,
   the backbone has collapsed to text-only. Roll back to phase 0 ckpt
   and re-do phase 1 with smaller `lr` (try 5e-5) or larger `alpha` (try 0.10).

`train_multimodal.py` writes `antifakery_step{N}.json` per-probe so the
trainer can be inspected post-hoc without re-running.

## Per-modality data sources

Why byte-patch and not VQ tokens / VAE latents:

- VQ codebooks fix vocabulary at training time; you can't add a new modality
  without retraining the codebook.
- Frozen vision encoders (CLIP, ViT) bottleneck through 196-token-per-image
  sequences and lock you into a specific resolution.
- Byte-patch (`Linear(C * patch_h * patch_w -> hidden)`) is a single matmul
  that learns IN the multimodal model. Fuyu-Heavy paper (Adept 2024) showed
  this matches CLIP-init performance with 4x less data.

Concrete sources used by `scripts/prep_multimodal_data.py`:

| Modality | Real source | Synthetic fallback | Smoke uses |
|----------|-------------|--------------------|-----------:|
| image | CC12M (HF Hub `Lin-Chen/CC12M`, 1k subset) | RGB shapes + colour caption | synthetic |
| audio | LibriSpeech mel memmap on `mohuanfang.com:/home/liu/synapforge_backup/librispeech_mel.memmap` | 1s tones with vowel labels | synthetic |
| video | (10 dummy clips smoke; YouCook2 for real) | numpy gradient + circle motion | synthetic |
| biosignal | (PhysioNet ECG MIT-BIH for real) | sine + 1/f noise + HR caption | synthetic |
| graph | ZINC 5k subset (random adj, real shape) | ring + random extra edges | synthetic |
| time_series | ETH-USD 1m OHLCV (offline parquet) | geometric Brownian motion | synthetic |
| screen | (NaviGui subset for real) | dark window + button + cursor | synthetic |
| point_cloud | re-uses `scripts/prep_3d_data.py` smoke output | random sphere/cube surfaces | synthetic |
| spatial_3d | same as point_cloud + per-row pinhole intrinsics for `PluckerRayEmbed` | same | synthetic |

All synthetic generators are seeded so a re-run produces byte-identical
parquets on the same machine. `--smoke` forces synthetic everywhere
(no network). Real-data switch is a one-liner per modality (TODO marker
in each `gen_*` function in `scripts/prep_multimodal_data.py`).

## Code paths exercised

`train_multimodal.py` exercises the following synapforge code paths during
each step:

- `synapforge.modal.UnifiedEmbed` (orchestrator, `<|sep|>` insertion).
- `synapforge.modal.{Image,Audio,Video,Screen,PointCloud,TimeSeries,Graph,BioSignal}PatchEmbed`.
- `synapforge.model_100m.SynapForge100M.forward_from_z` (skips text token embed).
- `synapforge.cells.LiquidCell` + `synapforge.surrogate.PLIFCell` + `synapforge.cells.SparseSynapse` (HybridBlock).
- `synapforge.modal.spatial_3d.PluckerRayEmbed` (when `spatial_3d` rows include intrinsics; trainer wires this via the `point_cloud` branch + a side-loaded Plucker token sequence -- followup work in `train_multimodal.py:phase>=2`).

## Honest budget

A complete run at the targets above needs roughly **336 GPU-h** on A100 80 GB
(~14 days at 24 h/day on one card; or 24 h on a 14-card pod). At the rental
prices in `reference_rental_a100x2_ssh.md` (~¥168 / 24h), the full run is
**~¥600 RMB ($80 USD)**.

The scaffold validates in **30 minutes on CPU** (smoke + 50 steps), so the
risk surface is decoupled from the bill: `train_multimodal.py --smoke`
exercises all 9 modal encoders + UnifiedEmbed + backbone + anti-fakery probe
without spending a yuan.

## Cross-references

- `synapforge/modal/spatial_3d.py` -- 3D Plucker + EGNN + DUSt3R teacher (Task #236).
- `scripts/prep_3d_data.py` -- existing CLEVR-3D synthetic generator,
  reused by the spatial_3d branch of `prep_multimodal_data.py`.
- `synapforge/trainer_mixins.py:MultimodalMixin` -- the InfoNCE contrastive
  aux mixin that this trainer reuses (extended from 2 modalities to 9).
- `synapforge/test_multimodal.py` -- existing smoke test for UnifiedEmbed.
