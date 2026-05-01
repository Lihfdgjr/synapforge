# 3D World Understanding -- Implementation Plan (Task #236)

Status: SCAFFOLD READY (CPU code-path validated). Real training deferred until
GPU rental is available.

This document is the implementation companion to `docs/3D.md` (research plan).
It covers (a) what's been built, (b) reuse of `synapforge.modal.point_cloud`,
(c) eval gates with falsification triggers, and (d) deferred decisions that
must be revisited when the 140 GPU-h budget unlocks.

## Files added

| Path | Lines | Purpose |
|------|-------|---------|
| `synapforge/modal/spatial_3d.py` | ~330 | PluckerRayEmbed + EGNNAdapter + DUSt3RTeacher stub. Exported via `synapforge.modal`. |
| `scripts/prep_3d_data.py` | ~250 | Synthetic CLEVR-3D-style stereo dataset (CPU raycast, no Blender). 1000 examples in seconds. |
| `train_3d.py` | ~360 | 4-phase scaffold trainer with backbone freeze/unfreeze + multitask loss (0.4 LM + 0.3 PM + 0.2 view + 0.1 QA). Smoke verified on CPU/GPU. |
| `docs/3D_PLAN.md` | this file | Plan + falsification gates. |

Total: ~960 LOC (spec said ~750; the gap is decoder helpers + collate + the
4-phase plumbing the spec asked for).

## Why this combination (recap from docs/3D.md)

- **DUSt3R as data engine, not as model.** We don't ship DUSt3R weights or run
  it on the hot path of our model. Instead the teacher is a frozen
  pseudo-label generator: 2 photos -> per-pixel pointmap, used as a
  regression target for our small head + as a seed for the EGNN graph. Saves
  ~500 GPU-h vs training a 3D encoder from scratch.

- **EGNN over per-frame ConvNets / ViTs.** Satorras 2102.09844 gives us
  rotation/translation invariance in the architecture. Empirically reported
  ~10x data efficiency on N-body and molecular regression. Critical when our
  budget is 1000-50000 examples, not millions.

- **Plucker rays as conditioning, not as output.** Sitzmann's LFN
  (1909.06443) trick. We feed (origin x dir, dir) per-pixel-grid into a tiny
  MLP and let the LNN+SNN backbone learn the geometry-language binding.
  ~13K params; cheap.

## Why not (rejected alternatives)

- **NeRF / 3D Gaussian Splatting.** Both are scene-fitting representations:
  every new scene needs minutes-to-hours of optimisation. Scaling them to a
  large pre-training mix is infeasible at our budget. DUSt3R is feed-forward
  and amortises across scenes.

- **Voxel grids (e.g. PointPillars / VoxNet).** Already covered by
  `point_cloud.py` (V=8 voxel hash). The new EGNN adapter is COMPLEMENTARY:
  point_cloud handles dense indoor scans, EGNN handles sparse multi-view
  triangulation graphs.

- **Zero-1-to-3 / single-image 3D.** Hallucinated geometry. We want
  metrically correct stereo, which DUSt3R provides for free at inference.

- **3D-LLM 2307.12981 architecture port.** That model is 3B params + uses a
  full vision transformer. We are 100M-375M and stick with our LNN+SNN
  backbone (memory `feedback_not_training_transformer.md`: TEACHER may be
  transformer for KD, STUDENT is not). Architecture parity is rejected.

## What `point_cloud.py` already does (reuse plan)

`synapforge/modal/point_cloud.py` provides `PointCloudEmbed` which:
- Takes (B, N, F>=3) points + optional mask.
- Voxel-hashes into a fixed (V x V x V) grid (default V=8 -> 512 tokens).
- Max-pools per-voxel features.
- Linear-projects to hidden dim, adds 3D sinusoidal positional encoding,
  prepends a `<|3d|>` learnable marker.

Our spatial_3d additions stack ON TOP of this:
- `EGNNAdapter` consumes the **same** (B, N, F) layout and emits
  `(B, 1+N, hidden)` with refined per-node tokens. It is a sibling, not a
  replacement.
- For dense indoor reconstructions (Habitat/ScanNet) we use
  `PointCloudEmbed` (voxel-hashed, fixed token count, fast).
- For sparse multi-view triangulation graphs (DUSt3R output, Objaverse
  pairs) we use `EGNNAdapter` (variable N, equivariant edges).

The `train_3d.py` model concatenates streams: text + image + plucker + egnn.
`PointCloudEmbed` is not currently wired in `Spatial3DModel` -- adding it
when ScanNet data lands is a one-line change inside `Spatial3DModel.forward`.

## Eval gates (must hit each before next phase starts)

Same as docs/3D.md, restated with falsification triggers:

| Gate | Phase | Pass | Soft fail (re-cook) | Hard fail (kill thread) |
|------|-------|------|---------------------|--------------------------|
| G1 pointmap MSE | post-Phase 1 | < 0.15 | 0.15-0.30 | > 0.50 |
| G2 CLEVR-3D acc | post-Phase 2 | >= 65% | 50-65% | < 40% |
| G3 ScanQA val EM | post-Phase 3 | >= 18% | 12-18% | < 8% |
| G4 WikiText regression | always | < 8% | 8-15% | > 20% |

**Hard-fail action:** if pointmap MSE > 0.5 after 10 GPU-h of phase 0, abort.
We are NOT spending 130 more GPU-h on a thread that doesn't even fit
synthetic depth. Document the negative result, ship a one-page note,
re-allocate GPU-h to other tasks (memory `feedback_audit_drought_2026q2.md`:
deep time per target, no speculative continuation).

**Soft-fail action:** drop to phase 0.5 with reduced LR (1/3 peak), increase
KD weight from teacher pseudo-labels, retry phase 1.

## 4-phase schedule (140 GPU-h on A800, mirror of docs/3D.md)

| Phase | Hours | Action | Notes |
|-------|-------|--------|-------|
| 0 | 10 | Freeze backbone; train EGNN + Plucker only. Observe PLIF spike rate. | Risk #1: PLIF spike-rate collapse; mitigation = `pointmap branch dense-CfC only, PLIF observe-only` for first 30 GPU-h. |
| 1 | 60 | Unfreeze last 4 CfC blocks + adapter. lr=2e-4 cosine, bs=32, seq=2048. | G1 must pass at end. |
| 2 | 50 | Full unfreeze. lr=5e-5. Add CLEVR-3D + ScanQA mix. | G2 must pass. |
| 3 | 20 | SFT on ScanQA + 3DSRBench train, lr=1e-5. | G3 must pass. |

For the **24h smoke budget** (current scaffold) we run only:
- Phase 0 ~10h on synthetic CLEVR-3D
- Phase 1 ~14h truncated -- enough to observe whether pointmap MSE is
  trending below 0.5 (anti-fakery).

If phase-0 metrics on synthetic data look reasonable, we apply for the next
GPU rental window and run the full schedule on Habitat + Objaverse.

## Falsifiability and anti-fakery

Per docs/3D.md anti-fakery: zero out the image embedding (`view_a`) and
retest. Caption + QA must collapse to <50% of the unzeroed baseline. If the
model still answers, we are dataset-bias cheating; whole thread is abandoned.

Implementation hook (in `train_3d.py`): add `--anti-fakery-check` flag for an
eval-only pass that zeros `batch["image_left"]` before forward. Currently
**TODO** -- low priority until we have at least one passing G1.

## Deferred decisions (revisit when GPU lands)

1. **Real DUSt3R weights.** `DUSt3RTeacher` is a stub. The expected ckpt
   path is `/workspace/teachers/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`.
   Download from https://github.com/naver/dust3r/releases. Replace the
   `_stub_proj` with the real model load + add `from dust3r.model import
   AsymmetricCroCo3DStereo` import. The forward signature
   `(image_a, image_b) -> {pointmap, confidence}` is already aligned with
   the real DUSt3R API.

2. **Real CLEVR-3D / ScanQA / 3DSRBench downloads.** Currently
   `prep_3d_data.py` synthesises the dataset from raycast. Real download:
   - CLEVR-3D: arxiv 2403.13554 (3DSRBench), HF hub `CLEVR-3D/clevr3d`.
   - ScanQA: arxiv 1908.04340. Train + val splits at https://github.com/ATR-DBI/ScanQA.
   - Habitat-Sim trajectories: https://aihabitat.org/datasets/hm3d.

3. **PLIF bypass during pointmap regression** (docs/3D.md risk #1). The
   scaffold currently does NOT toggle PLIF observe-only mode. Wire this in
   `Spatial3DModel.forward`: when `kind == "pointmap"`, route through dense
   CfC only and skip PLIF spike emission. ~20 LOC change.

4. **STDP pause on pointmap loss steps** (docs/3D.md risk #3). When STDP
   plasticity is active in the trainer, pause it on `pointmap`-loss steps
   to avoid Hebbian co-firing memorising 2D image priors. The current
   scaffold doesn't run STDP; this is wired up in v4.2 main trainer.

5. **EGNN coordinate update detach** (docs/3D.md risk #2). Already
   addressed: `_EGNNLayer.forward` emits `x_new = x + delta` (NOT in-place).
   Caller may detach if shared with CfC fast-weights.

6. **View-augmentation rotation 0-360 degrees** (docs/3D.md risk #3). Add
   to data loader as a transform on `(image_left, image_right, pointmap)`.
   Rotate point cloud and recompute extrinsics. Currently TODO.

7. **Anti-fakery eval flag.** See above.

## Cross-references

- Plan source: `docs/3D.md`
- Architecture: `docs/ARCHITECTURE.md` (existing modal subsystem)
- Reuse: `synapforge/modal/point_cloud.py` (voxel-hash branch, complementary)
- Constraints: memory `feedback_lnn_snn_small_data_beat_large.md` (small
  model + small data), `feedback_torch_buffer_inplace.md` (no in-place
  buffer mutation), `feedback_not_training_transformer.md` (student is NOT
  transformer; teacher MAY be).
- Anchor papers (no changes from docs/3D.md): DUSt3R 2312.14132, MASt3R
  2406.09756, EGNN 2102.09844, 3D-LLM 2307.12981, ScanQA 1908.04340,
  3DSRBench 2403.13554, Habitat 1904.01201, Objaverse 2212.08051.

## Smoke-run checklist (verified)

- [x] `python -m synapforge.modal.spatial_3d` prints OK
- [x] `python scripts/prep_3d_data.py --n-examples 10 --out /tmp/x.parquet`
      writes parquet
- [x] `python train_3d.py --help` shows flags
- [x] 2-step training loop completes without NaN on synthetic data
      (loss order-of-magnitude is high because pointmap GT is in world
      coords with no scale normalisation -- this is expected for the
      scaffold; real training adds per-scene scale normalisation as a
      preprocessing step)

## Next step (when GPU rental is up)

1. Download real DUSt3R ckpt; replace stub.
2. Download Habitat-Sim HM3D + 1k Objaverse pairs.
3. Run phase 0 (10 GPU-h). Check G4 (no WikiText regression > 8%).
4. Decide on phase 1 based on phase-0 trend.

If phase 0 pointmap MSE > 0.5 -> abort, document, move on.
