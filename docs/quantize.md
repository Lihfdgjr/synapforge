# `sf.quantize` — BitNet b1.58 ternary QAT

Module: `synapforge.quantize`
References:
- BitNet (Wang et al. 2024) — https://arxiv.org/abs/2310.11453
- The Era of 1-bit LLMs (Ma et al. 2024, "BitNet b1.58") — https://arxiv.org/abs/2402.17764
- Inference kernel: bitnet.cpp — https://github.com/microsoft/BitNet

## Why

For a "deploy on commodity laptop" pitch, fp32 weights are dead weight. BitNet
b1.58 stores each weight as one of three values `{-1, 0, +1}` (= log2(3) ≈ 1.58
bits per weight). Crucially, ternary matmul reduces to integer add/subtract —
no multiplications — which on CPU gives BitNet's reported **5-10x speedup**
for batch=1 inference plus an order-of-magnitude energy reduction, with
quality matching fp32 after a short fine-tune.

This module implements the *training-time front-end*: keep the model in fp32
during training but constrain the forward pass to ternary weights so the model
adapts to the discretization. Backward uses the **straight-through estimator**
(STE) so gradients flow through `round()`. Conversion to packed `int2` /
`int8` weights at deployment is the job of bitnet.cpp or a future synapforge
runtime backend — *not* this file.

## API

```python
import synapforge as sf
from synapforge.quantize import (
    TernaryLinear,
    convert_model_to_ternary,
    freeze_gamma,
    quantize_ternary,
)

# Convert a pre-trained fp32 model in place. Embeddings and LM head stay fp32.
n = convert_model_to_ternary(model, exclude=("emb", "lm_head"))
print(f"replaced {n} nn.Linear -> TernaryLinear")

# Fine-tune for 1-5% of the original step budget (see "When to convert" below).
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for step in range(qat_steps):
    loss = ...
    loss.backward()
    optimizer.step()

# Optional: lock gamma before evaluation/export.
freeze_gamma(model)
```

### Building from scratch with ternary linears

```python
import torch.nn as nn
from synapforge.quantize import TernaryLinear

block = nn.Sequential(
    nn.LayerNorm(dim),
    TernaryLinear(dim, 4 * dim),
    nn.GELU(),
    TernaryLinear(4 * dim, dim),
)
```

## Methodology (how the QAT works)

1. **Forward**: weight `w` is normalized by the per-tensor scale `gamma` and
   rounded to `{-1, 0, +1}`, then re-multiplied by `gamma` (so the layer
   numerically still does fp32 GEMM during training).

   ```
   gamma = mean(|w|)               # per-tensor, EMA-tracked (see below)
   w_q   = round(w / gamma).clamp(-1, 1)
   y     = (w_q * gamma) @ x.T + b
   ```

2. **Backward (STE)**: the discretization step is non-differentiable, so we
   pass the upstream gradient straight through to `w` (and never propagate
   into `gamma`).

   ```
   d_loss / d_w = upstream
   d_loss / d_gamma = 0   (gamma is a buffer, not a parameter)
   ```

3. **Gamma EMA**: recomputing `mean(|w|)` per forward is wasteful and noisy.
   We track gamma via EMA (default momentum 0.99) for the first
   `gamma_warmup_steps` (default 1000) forwards, then freeze. After warmup
   the forward is fully deterministic, which makes the STE pass-through
   semantically clean.

4. **Exclusions**: by convention, three classes of weight are NOT quantized:
   - **Embeddings** (`emb`, `embedding`) — their LSB carries token identity;
     ternary embeddings collapse the vocab.
   - **LM heads** (`lm_head`, `head`) — output logits at the boundary need
     fine-grained scale.
   - **Plasticity / fast weights** (`hebb_*`, `stdp_*`, `fast_*`) — online
     updates rely on small gradient magnitudes that ternarization erases.
     These live as raw `nn.Parameter`/buffers (not `nn.Linear`), so the
     module-walk converter never touches them.

## When to convert

The cookbook (BitNet b1.58 and follow-ups) is:

1. **Pretrain** the model in **fp32** to convergence (or near-convergence).
2. **Convert** all `nn.Linear` (except emb / lm_head / plasticity) to
   `TernaryLinear`. Weights are copied over, so the model starts numerically
   equivalent (modulo discretization noise).
3. **Fine-tune** for **1-5% of the original training step budget** with a
   slightly reduced learning rate (1e-4 to 3e-4 typical). This recovers
   quality lost to the round step. The BitNet b1.58 paper reports parity
   with fp32 LLaMA-2 7B baselines after ~2-3% of the original token budget
   (~50B tokens for a 2T-token pretrain).
4. **(Optional) SFT** with the QAT model — proceed with full fine-tune as
   normal; ternary-aware optimizer doesn't need special handling beyond
   what AdamW already does.
5. **Export**: at deploy time, convert to packed `int2` / `int8` codes.
   This is what gives the 5-10x CPU speedup. See "Production path" below.

For synapforge mscfc-style models, plasticity stays fp32 throughout — only
the static linears get ternary.

## Production path: bridging to true CPU int8 inference

This module gets you a **trained** ternary model. The actual deploy-time
speedup requires running the model on an int8 / popcount kernel. Two options:

### Option A: bitnet.cpp (recommended)

[bitnet.cpp](https://github.com/microsoft/BitNet) is the official
inference engine. Pipeline:

1. After QAT, dump weights via `TernaryLinear.ternary_codes()` (returns
   `int8` tensor in `{-1, 0, +1}`) and `TernaryLinear.gamma` (per-tensor
   `fp32` scale).
2. Pack into bitnet.cpp's GGUF-like format (4 ternary codes per byte after
   a `+1` shift to `{0, 1, 2, 3}`, then 2 bits per weight; or run their
   `convert-helper.py`).
3. Run with `llama.cpp`-style CLI; CPU kernel uses `popcount` and table
   lookups. Reports show **5-7x** speedup vs fp16 on x86, **2-4x** vs
   int4 quantization, batch=1.

### Option B: custom synapforge backend

A future `synapforge.runtime.backends.bitnet_cpu` will:
- Pack ternary codes inline at compile time.
- Generate AVX-512 / NEON kernels via Triton-CPU or hand-written intrinsics.
- Expose the same `runtime.compile(model, backend="bitnet_cpu")` interface
  as the existing `gpu_dense` backend.

This is on the v0.3 roadmap (see `synapforge/__init__.py` docstring) and
will land alongside event-driven CPU inference.

## Bench numbers (this build, A100, bs=64, T=256, 6-layer 256-dim block)

| Metric                                      | Value          |
|--------------------------------------------|----------------|
| fp32 inference latency                      | ~7.7 ms/iter   |
| ternary inference latency (fp32 dequant)    | ~8.5 ms/iter   |
| **Ternarizable layer compression**          | **~20x**       |
| Whole-model deploy compression (incl emb)   | ~5.3x          |
| Quality after 100 QAT steps on toy LM       | loss decreases |

The 0.9x latency on GPU is **expected** — both modes hit cuBLAS fp32 GEMM,
ternary just adds a `round/clamp`. The 20x layer-only compression confirms
the storage win, and the 5-10x CPU inference win lands once weights are
exported to bitnet.cpp.

## See also

- `synapforge/quantize.py` — the implementation
- `synapforge/test_ternary.py` — 7 tests (buckets, STE, EMA, freeze,
  100-step QAT, state-dict, plasticity-untouched)
- `synapforge/bench_ternary.py` — fp32 vs ternary latency + on-disk size
