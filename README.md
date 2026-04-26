<div align="center">

# synapforge

**Train dense, deploy async — purpose-built ML for liquid + spiking + plastic networks.**

[![PyPI version](https://img.shields.io/pypi/v/synapforge.svg)](https://pypi.org/project/synapforge/)
[![Python](https://img.shields.io/pypi/pyversions/synapforge.svg)](https://pypi.org/project/synapforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/Lihfdgjr/synapforge/actions/workflows/test.yml/badge.svg)](https://github.com/Lihfdgjr/synapforge/actions/workflows/test.yml)

</div>

---

## Why synapforge

PyTorch is a tensor framework. Liquid Neural Networks (LNN), Spiking Neural Networks (SNN),
and synaptic plasticity break almost every assumption it was built on:

| Assumption                                | Breaks because                                                |
|-------------------------------------------|---------------------------------------------------------------|
| Forward writes activations once           | Plasticity rewrites weights *during* forward                  |
| Backprop is the only learning signal      | Hebbian / STDP / BCM are local, gradient-free                 |
| Layers are differentiable                 | Spikes are non-differentiable; need surrogate gradients       |
| Dense matmul is the bottleneck            | Spike rasters are <5% sparse — dense waste >95% FLOPs         |
| Time is a leading dim, not a state        | Liquid ODEs need true continuous-time dynamics                |
| All optimizers see the same `param.grad`  | Plasticity-aware AdamW must merge gradient + plastic-delta    |

**synapforge** is a thin layer on top of PyTorch that fixes all seven, while keeping
the friendly `nn.Module`-style ergonomics. Train a hybrid model with `sf.LiquidCell` +
one call.

```python
import synapforge as sf

class HybridBlock(sf.Module):
    def __init__(self, d):
        super().__init__()
        self.cfc  = sf.LiquidCell(d, d)            # liquid (continuous-time)
        self.plif = sf.PLIF(d, threshold=0.3)      # spiking with surrogate
        self.syn  = sf.SparseSynapse(d, d, sparsity=0.9)
        self.stdp = sf.STDP(self.syn, tau_pre=20., tau_post=20.)

    def forward(self, x):
        h        = self.cfc(x)
        spk, mem = self.plif(h)
        out      = self.syn(spk)
        self.stdp(spk_pre=spk, spk_post=out)       # plastic update, gradient-free
        return out

model = HybridBlock(256)
y     = rt(torch.randn(32, 100, 256, device="cuda"))
```

---

## Install

```bash
pip install synapforge
```

Optional extras:

```bash
pip install "synapforge[triton]"     # Triton-fused parallel scan (Linux+CUDA)
pip install "synapforge[data,hf]"    # Parquet streaming + HF tokenizer adapter
pip install "synapforge[dev]"        # pytest, ruff, mypy, build, twine
```

Core dependencies are minimal: `torch>=2.0` and `numpy`. Everything else is opt-in.

---

## Quickstart (10 lines)

```python
import torch, synapforge as sf

class M(sf.Module):
    def __init__(self): super().__init__(); self.cell = sf.LiquidCell(16, 16)
    def forward(self, x): return self.cell(x)

m  = M().cuda()
rt = sf.compile(m, backend="gpu_dense")
y  = rt(torch.randn(4, 32, 16, device="cuda"))
print(y.shape)  # torch.Size([4, 32, 16])
```

That's it. No custom training loop, no manual surrogate registration, no boilerplate.

---

## Architecture

```
+----------------------------------------------------------------------+
|                         User code (sf.Module)                        |
+----------------------------------------------------------------------+
|  LiquidCell  |  PLIF   |  SparseSynapse  |  Plasticity  |  Optim     |
|  (cells/)    | (cells/)| (cells/)        | (rules)      | (PA-AdamW) |
+----------------------------------------------------------------------+
|                     IR (graph / compiler / passes)                   |
+----------------------------------------------------------------------+
+----------------------------------------------------------------------+
+----------------------------------------------------------------------+
```

- **cells/** — primitives (LiquidCell, PLIF, SparseSynapse). Every cell exposes
  a deterministic, differentiable forward and a backend-agnostic IR descriptor.
- **plasticity.py** — `Hebbian`, `STDP`, `BCM`, `SynaptogenesisGrowPrune`, glued
  by `PlasticityEngine`. All rules are local, gradient-free, and JIT-friendly.
- **surrogate.py** — pluggable surrogate-gradient registry. Built-ins: arctan,
  sigmoid, super-spike, fast-sigmoid, triangular, multi-Gaussian. `register()`
  to add custom.
- **optim.py** — `PlasticityAwareAdamW` merges Adam moments with `Param`
  plastic-delta tensors; `MultiSourceParam` lets one tensor carry both gradient
  and plastic gradient streams.
- **runtime.py / runtime_cuda_graph.py** — compiles a `sf.Module` into one of
  the four backends. CUDA-graph runtime delivers ~2x throughput on small RNNs.
- **distributed.py** — DDP wrapper with `PlasticBufferSync`, which all-reduces
  plasticity buffers (synaptic weights mutated in forward) on a configurable
  cadence so plastic state stays consistent across ranks.
- **quantize.py** — BitNet-style ternary {-1, 0, +1} post-training quantization;
  ~20x weight compression, <2pp accuracy loss on tested workloads.
  pending).

The IR layer (`ir/graph.py`, `ir/compiler.py`) is the hinge: it lets one model
description target wildly different runtimes without rewriting kernels.

---

## Benchmarks

Numbers measured on `/workspace/synapforge` rental box (2x A800-80G, torch 2.1.2,
triton 2.1, fp32):

| Component                            | Baseline                  | synapforge        | Speedup        |
|--------------------------------------|---------------------------|-------------------|----------------|
| LiquidCell forward (B=8, T=128, D=256)  | PyTorch eager unfused     | Triton block scan | **29.0x**      |
| PlasticityAwareAdamW step (10M params)  | torch AdamW + manual sync | fused step kernel | 1.8x           |
| Multi-GPU DDP throughput (2x A800)      | torch DDP                 | sf.wrap_model     | **1.78x**      |
| Ternary post-training quantization      | fp32 weights              | ternary {-1,0,+1} | **20.1x compression** |
| CPU event-driven inference (sparse <5%) | torch.eager fp32 dense    | numba CSR raster  | 6.4x           |
| Surrogate fwd+bwd (atan, B=64, D=512)   | naive autograd            | fused atan kernel | 4.2x           |

Reproduce: `python benchmarks/bench_triton.py`,
`python benchmarks/bench_surrogate.py`, `python benchmarks/bench_ternary.py`.

---

## Roadmap

| Milestone | Status     | What                                                        |
|-----------|------------|-------------------------------------------------------------|
| M1        | done       | sf.Module, LiquidCell, PLIF, SparseSynapse                  |
| M2        | done       | Hebbian / STDP / BCM / synaptogenesis                       |
| M3        | done       | Surrogate-gradient registry (6 built-ins)                   |
| M4        | done       | gpu_dense backend (numerical-equivalent to mscfc)           |
| M5        | done       | PlasticityAwareAdamW + MultiSourceParam                     |
| M6        | done       | Triton block-fused parallel scan (29x fwd)                  |
| M7        | done       | DDP wrapper + PlasticBufferSync (1.78x on 2-GPU)            |
| M8        | done       | Ternary BitNet 1.58 PTQ (~20x compression)                  |
| M9        | done       | CPU event-driven inference (numba raster)                   |
| M10       | done       | CUDA-graph runtime (~2x small-model throughput)             |

hardware. Today's release is `v0.5.0`.

---

## Examples

The `examples/` directory ships 5 runnable scripts:

| File                              | Demonstrates                                       |
|-----------------------------------|----------------------------------------------------|
| `examples/01_hello.py`            | Minimal `sf.Module` end-to-end (10 lines)          |
| `examples/02_train_lnn.py`        | Train a `LiquidCell` stack on a toy regression    |
| `examples/03_plasticity.py`       | Hebbian + STDP rules with `PlasticityEngine`       |
| `examples/04_triton_speedup.py`   | Bench gpu_dense vs triton_block forward            |
| `examples/05_distributed.py`      | 2-GPU DDP smoke with `PlasticBufferSync`           |

```bash
python examples/01_hello.py
torchrun --nproc-per-node 2 examples/05_distributed.py
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/ --cov=synapforge --cov-report=term
```

Tests are CPU-friendly where possible; GPU/Triton tests skip cleanly when
hardware is missing.

---

## Citing synapforge

If you use synapforge in academic work:

```bibtex
@software{synapforge_2026,
  author    = {Liu},
  title     = {synapforge: Train dense, deploy async — a framework for liquid + spiking + plastic networks},
  year      = {2026},
  version   = {0.5.0},
  url       = {https://github.com/Lihfdgjr/synapforge}
}
```

---

## Contributing

We welcome PRs. See [CONTRIBUTING.md](CONTRIBUTING.md). Quick checklist:

- `pip install -e ".[dev]"`
- `ruff check .` (lint)
- `pytest tests/ -v` (must stay green on CPU)
- Open issues before large refactors

---

## License

[MIT](LICENSE) © 2026 Liu

---

<div align="center">
<sub>synapforge is purpose-built. PyTorch for tensors, synapforge for synapses.</sub>
</div>

---

## v1.5 Roadmap (in progress, 2026-04-26)

`train_v15_full.py` joint trains the 9-modal model with KD + intrinsic exploration:

### Components

- **9-modality byte-patch** — text / image / audio / video / screen / point_cloud /
  time_series / graph / biosignal, all via `sf.modal.UnifiedEmbed` (Fuyu/Chameleon-style
  reshape + Linear, **no convolutional encoders**).
- **CfC + PLIF backbone** — 2 layers; PLIF tau init defaults to **DA-LIF bimodal**
  (half fast tau~3, half slow tau~30) for heterogeneous time constants from step 0.
- **Knowledge distillation** — Qwen2.5-0.5B teacher, **rep-level** (mean-pooled hidden
  state -> Linear projection -> MSE). Vocab-agnostic so any teacher tokenizer works.
- **Web-augmented data** — `/workspace/data/` mixes WT103 + agent-generated Q/A pairs
  + FineWeb-Edu chunks. Data pulled one-shot at training prep, **never** as runtime
  tool-calls — preserves the rule that the model itself only emits actions through
  `ActionHead` / `NeuroMCPHead` (neuronal direct, no schema).
- **Semantic understanding aux losses**:
  - Triplet (anchor / synonym / antonym) — InfoNCE-style margin loss on `sem_proj`.
  - Definition modeling — `word: definition` CE loss.
  - Char-level aux head — predicts next byte from hidden state (256-class).
  - Cross-modal contrastive — image+caption InfoNCE (when COCO data lands).
- **Intrinsic exploration** — every `--idle-every` steps the model self-rolls a
  `Q:` continuation via its own LM head. Future versions feed the rollout into
  `sf.self_learn.ExperienceReplayBuffer` weighted by `sf.intrinsic.NoveltyDrive`.
- **Safety**:
  - z-loss + label smoothing for stable softmax.
  - Optimizer m/v restored on warm-start (no momentum cold-start).
  - Gradient clipping (norm 0.5) + linear warmup + cosine decay.
  - Per-layer PLIF spike-rate monitor flags dead/saturated neurons.

### Training-time tools (NOT inference)

- `auto_ckpt_backup.py` — periodically promotes best-val-ppl ckpts per run to
  `/workspace/best_ckpts/`, tar.gz uploads to GitHub release every 4h.
- `auto_rsync_ckpt.py` — glob-scans run dirs and pushes `.log/.json/.csv` to
  `Lihfdgjr/synapforge-runs` every 10 min.
- `github_push_retry.py` — handles intermittent github.com network filtering.
- `train_100m_kd.py` — text-only logit-KD trainer (used for v1.3 baseline).

All daemons read `GH_TOKEN` from env, **never hardcoded**.

### Distributed (NV12 NVLink, 2× A100-80GB)

```bash
torchrun --nproc_per_node=2 /workspace/synapforge/train_v15_full.py \
    --warmstart /workspace/runs/synapforge_full_modal/ckpt_step4000.pt \
    --steps 6000 --batch-size 16 --lr 1.5e-4 --warmup 300
```

`synapforge.distributed.wrap_model` adds plastic-buffer-aware DDP (1.78x scaling
already verified on 2× A800 with `PlasticBufferSync`).

### Known limitations

- WT103 alone caps ppl around ~150 even with a 100M-param model; **ppl < 50 needs
  more pretrain tokens** (FineWeb-Edu mixin partially addresses this).
- Triton fused kernel works in fresh subprocess but occasionally fails the first
  compile in cold rentals — script falls back to pure-PyTorch (slower but correct).
- Self-rolled Q: prompts are not yet fed back as training targets; that closes
  the loop in v1.6.
