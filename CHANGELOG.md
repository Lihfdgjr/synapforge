# Changelog

All notable changes to **synapforge** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
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
