# Changelog

All notable changes to **synapforge** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
## [1.5.0-dev] — 2026-04-26 (in progress on rental)

Web-augmented self-learning training. **Goal: WT103 val ppl < 50** (still tracking; 24h run).

### Added

- **`train_v15_full.py`** — joint full-9-modality + KD + intrinsic + semantic trainer.
  - Stronger teacher: **Qwen2.5-0.5B** rep-level KD (vocab-agnostic; mean-pool teacher
    hidden -> Linear projection -> MSE; bypasses GPT-2 50257 vs Qwen 151643 mismatch).
  - Web-augmented data: WT103 + agent-generated Q/A + FineWeb-Edu chunks
    (downloaded one-shot at training prep, **not** runtime tool-calling — preserves
    NeuroMCP rule that the model itself never schema-calls anything).
  - Semantic understanding: triplet (anchor/synonym/antonym), definition modeling,
    char-level aux head, cross-modal contrastive (image+caption).
  - Intrinsic exploration: every N steps the model self-rolls a `Q:` continuation
    via its own LM and logs it; future versions will feed the rollout back into the
    NoveltyDrive replay buffer.
- **`train_100m_kd.py`** — text-only KD trainer with frozen GPT-2 teacher (logit
  KL distillation, batch-mean reduction with `T*T` scaling).
- **`train_full_modal.py`** — joint native training across all 9 modalities + action +
  NeuroMCP. Used as warm-start basis for v1.5.
- **PLIFCell DA-LIF tau init** — `tau_init={float, "bimodal", ("bimodal", fast, slow),
  "log_uniform"}`. The bimodal default starts half the channels fast (tau~3) and half
  slow (tau~30) to break the degenerate uniform-tau symmetry of the Fang 2021 PLIF
  init. Verified: bimodal split of 64 channels gives 32 @ tau=3.0, 32 @ tau=30.0.
- **z-loss + label smoothing** in `train_100m.py` — `loss = ce + α·logsumexp(logits)²`,
  PaLM/Gemma standard. `--z-loss-weight` (default 1e-4), `--label-smoothing` (default 0.0).
- **Per-layer PLIF spike-rate monitor** — every 50 train steps prints
  `spike: mean=… range=… dead=N/M sat=N/M` from `last_spike_rate` buffers.
  Detects dead (rate<0.005) and saturated (rate>0.5) channels.
- **Optimizer state save/restore** in checkpoints — warm-start now keeps Adam m/v
  momentum across runs. Direct fix for the v1.2 b80 divergence (loss 5.45 → 6.34
  in 500 steps because momentum was cold-restarted).
- **Gradient checkpointing flag** (`--grad-checkpoint`) on `SynapForge100M.forward` —
  wraps each `blk(x)` call in `torch.utils.checkpoint(use_reentrant=False)` for
  memory relief at B>=128. Trades 2x compute for ~50% activation memory.
- **Auto checkpoint backup daemon** (`auto_ckpt_backup.py`) — periodically scans
  `/workspace/runs/synapforge_*` for the best-val-ppl checkpoint per run, promotes
  to `/workspace/best_ckpts/<run>_best.pt`, and tar.gz uploads to a GitHub release
  every 4 hours. **Token read from `GH_TOKEN` env var**, never hardcoded.
- **Auto metrics rsync** (`auto_rsync_ckpt.py`) — glob-scans `/workspace/runs/synapforge_*`
  for `.log/.json/.csv/.md` and pushes to `Lihfdgjr/synapforge-runs` every 10 min.
  Replaces a hard-coded 3-dir list that missed every run after v1.1.
- **GitHub push retry daemon** (`github_push_retry.py`) — handles intermittent
  github.com filtering on Chinese-cloud rentals.

### Fixed

- **Triton 2.1 bf16 MLIR encoding bug** in fused HybridBlock kernel. Three patches:
  (1) upcast bf16/fp16 to fp32 inside `_triton_block_forward` / `_triton_block_backward`;
  (2) force fp32 elig buffer (`atomic_add` doesn't support bf16);
  (3) always pass `ENABLE_STDP=False` to the kernel, do full STDP in PyTorch wrapper.
  Result: standalone B=64 T=256 D=512 bf16 fwd+bwd peak 0.51 GB (vs 4.28 GB on the
  unpatched PyTorch fallback). v1.3 KD trains at **24,000 tok/s on A100** (vs v1.2 b80's
  silently-fallback 4,712 tok/s, a 6.1× speedup).
- **`train_100m.py --lr-decay` was dead** — flag was parsed but never wired to the
  optimizer. LR stayed at 3e-4 for the entire run, contributing to the v1.2 b80
  divergence. Now `lr_at(step, peak, warmup, total, kind)` is called per step with
  proper warmup + cosine decay.
- **GitHub PUSH PROTECTION** triggered by hardcoded GitHub PAT in early commit —
  sanitised, switched to env-var read, repo unblocked.
- **`sys.path` shadowing** in `train_100m.py` — script directory was prepended ahead
  of `/workspace`, causing the nested `synapforge/synapforge/` package to win over
  the patched outer `synapforge/`. Now strips script dir from `sys.path` at startup.

### Removed

- All hardcoded GitHub tokens from source — read `GH_TOKEN` from env. (Repo CI / local
  dev unchanged; affected only the rental backup scripts.)

### Pending

- v1.5 full launch with both A100s in DDP (NV12 NVLink topology confirmed; expected
  1.8-1.9x scaling with `synapforge.distributed.wrap_model`).
- Cross-modal contrastive integration (data downloaded; loss wired but pending real
  COCO/AudioCaps batches).
- TestTimeTraining inference-time adaptation (`sf.self_learn.TestTimeTraining`).

to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-04-26

First public release. Code base ~7,100 LOC. Eleven of twelve milestones (M1-M11)
landed; v1.0 gates on Loihi-2 hardware verification.

### Added

- **Core API** — `sf.Module`, `sf.LiquidCell`, `sf.PLIF`, `sf.SparseSynapse`.
- **Plasticity** — `Hebbian`, `STDP`, `BCM`, `SynaptogenesisGrowPrune`,
  `PlasticityEngine`. All rules are local and gradient-free.
- **Surrogate-gradient registry** — six built-ins (atan, sigmoid, super-spike,
  fast-sigmoid, triangular, multi-Gaussian) plus `register()` for custom rules.
- **Backends** — `gpu_dense` (PyTorch eager, numerical-equivalent to
  `mscfc.LiquidS4Cell`), `triton_block` (fused parallel scan, ~29x forward),
  `cpu_event` (numba CSR raster for sparse <5% inference), `lava_export`
  (Loihi-2; code path complete, hardware verification pending).
- **Optimizer** — `PlasticityAwareAdamW` with `MultiSourceParam` to merge
  gradient and plastic-delta streams.
- **Distributed** — `sf.wrap_model` DDP wrapper with `PlasticBufferSync`
  (1.78x throughput on 2x A800).
- **Quantization** — Ternary BitNet 1.58 post-training quantization
  (~20x weight compression, <2pp accuracy loss on tested workloads).
- **CUDA-graph runtime** (`runtime_cuda_graph.py`) — ~2x throughput on small
  recurrent models.
- **Hugging Face adapter** (`huggingface_adapter.py`) — drop-in tokenizer +
  dataset wrapper for plug-and-play training.
- **Examples** — five runnable scripts in `examples/` covering hello-world,
  LNN training, plasticity, Triton speedup, and 2-GPU DDP.
- **Tests** — full pytest suite under `tests/` with CPU-friendly skips for
  GPU/Triton tests.
- **CI** — GitHub Actions matrix (Linux/macOS/Windows × Python 3.10/3.11)
  on push/PR; PyPI auto-publish on `v*.*.*` tags.

### Pending for v1.0

- Loihi-2 numerical equivalence verification (M11 hardware path).
- Full neuromorphic + analog co-deployment story (M12).
- Plasticity-aware autograd (currently plasticity is gradient-free; v1.0 will
  thread plastic deltas through autograd for end-to-end differentiable rules).

### Notes

- Core install is `torch + numpy` only. `triton`, `lava`, `pyarrow`, and
  `transformers` are opt-in extras.
- Tested on torch 2.0/2.1/2.2, Python 3.10/3.11/3.12.
