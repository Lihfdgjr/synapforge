<!-- DOC_STAMP: FRESH 2026-05-02; T2.12 FP8/int8 inference path research -->
# FP8 / int8 Inference Path — A800 today, Hopper tomorrow

**Updated**: 2026-05-02
**Status**: research note (no code change); follow-up = int8 PTQ on A800 (`torch.ao`); FP8 deferred to Hopper rental
**Tracking**: T2.12 in `docs/DEEP_MAINT_QUEUE.md`; sister docs `docs/quantize.md`, `docs/MATMUL_FREE.md`

> **TL;DR** — A800 (Ampere SM 8.0) has **no native FP8 tensor cores**. Emulated
> FP8 on A800 is ~20% slower than bf16 — net loss. Hopper (H100/H200, SM 9.0)
> ships native FP8 e4m3/e5m2 with ~2× bf16 throughput; DeepSeek-V3 reports
> ~1.5–1.7× end-to-end training speedup. Synap-1 specifically has CfC weights
> already ternary (T2.8) and PLIF spike already binary, so the only fp surface
> wide enough for FP8 to matter is the **FFN + `text_hidden` activation pipe**
> (~15% of compute). Pragmatic ship-now path = **int8 PTQ via
> `torch.ao.quantization`** for ~1.5× bs=1 inference; defer FP8 to Q3 2026 H100.

## 1. A800 capability today (Ampere SM 8.0)

| Precision | Native? | Tensor-core throughput | Notes |
|-----------|---------|------------------------|-------|
| FP32      | yes     | 1×                     | baseline |
| TF32      | yes     | ~8×                    | A800 default |
| BF16      | yes     | ~16×                   | trainer default (`triton_block`) |
| **INT8**  | **yes** | **~32×**               | **cuBLAS-LT, inference-time** |
| **FP8**   | **NO**  | **emulated (slower)**  | **~20% slower than bf16** |

FP8 e4m3 / e5m2 require tensor-core paths added in **Hopper SM 9.0**.
On Ampere the only "FP8" is software cast at boundaries, then bf16 MMA;
the dequant cost more than eats the savings. NVIDIA TransformerEngine
explicitly disables `fp8_autocast` on SM < 8.9 for this reason.

What A800 *does* offer that we don't yet use: **INT8 GEMM** via cuBLAS-LT,
exposed by `torch.ao.quantization`. Realistic Synap-1 100M gain is **~1.5×**
batch=1 inference, quality preservable via 1k–5k calibration samples (PTQ)
or short QAT fine-tune (~5% of pretrain budget).

## 2. Hopper (H100 / H200) advantage

Native **FP8** with two formats:

- **e4m3** (4-exp, 3-mant) — forward weights/activations, range ±448.
- **e5m2** (5-exp, 2-mant) — backward gradients, range ±57344.

Per-tensor or per-block scaling (`amax` history) recomputed each forward.

| Source | Setup | Gain over BF16 |
|--------|-------|----------------|
| TransformerEngine docs | GPT-3 175B training H100 | 1.6–1.8× |
| **DeepSeek-V3** (2412.19437) | 671B MoE H800 cluster | **1.5–1.7×** end-to-end |
| Anyscale FP8 inference | Llama-3 70B serving H100 | ~2× tok/s |

DeepSeek-V3 recipe: **FP32 master weight + optimizer state**, **FP8 e4m3
forward**, **FP8 e5m2 backward**, per-block 1×128 / 128×128 scaling. Quality
cost: <0.5% downstream eval vs bf16.

## 3. Synap-1 specific — what would actually move

Post-T2.8 layout (already heavily quantized):

| Component | Shape (100M) | Current dtype | FP8 candidate? |
|-----------|--------------|---------------|----------------|
| `tok_embed` | [151936, 512] | bf16 | NO — token-ID precision |
| **CfC `W` / `Wi`** | 10× [512, 512] | **ternary** (T2.8) | NO — already 1.58-bit |
| **PLIF spike** | [B, T, 512] | **binary** | NO — already 1-bit |
| FFN (gate/up/down) | 10× 2× [512, 1408] | bf16 | **YES — primary FP8 target** |
| `text_hidden` activation | [B, T, 512] | bf16 | **YES — secondary target** |
| `lm_head` | [512, 151936] tied | bf16 | maybe — quality-sensitive |
| STDP fast-weight | [512, 512] | fp32 | NO — small grads erase |

**Net**: FFN + `text_hidden` ≈ **15% of total compute** (matmul-free CfC
backbone dominates). FP8 ceiling for Synap-1 is therefore **~1.10–1.15×**
end-to-end (15% × 1.7×), not the 1.5–1.7× DeepSeek number — they are 100%
transformer FFN, we are mostly already-quantized recurrence. **Real but
secondary** — ternary CfC + binary spike already delivered the structural
quantization win that FP8 gives transformers.

## 4. Action plan

### Ship now on A800: **int8 PTQ inference path**

1. Post phase-3, call
   `torch.ao.quantization.quantize_dynamic(model, {nn.Linear: torch.qint8})`
   excluding `tok_embed` / `lm_head` / STDP (mirror `synapforge.quantize.DEFAULT_EXCLUDE`).
2. Calibrate on 1000 wt103 val samples — record per-tensor scale + zero-point.
3. Bench A800 batch=1: expect **~1.4–1.6×** for FFN-bound portion, ~1.1× whole-model.
4. CLI flag `--int8-inference` in `chat_demo`. Default off, opt-in.
5. Gate: val ppl regression ≤ 0.5%, MMLU/HellaSwag tiny within 1%.

**Effort**: ~1 day post phase-3.

### Defer to next rental: **FP8 training path**

- **Trigger**: H100 / H200 rental (~Q3 2026).
- **Library**: TransformerEngine + DeepSeek-V3 per-block scaling (TE per-tensor
  is too coarse near ternary boundaries).
- **Scope**: FFN forward e4m3, backward e5m2; CfC stays ternary; PLIF binary;
  STDP fp32; `lm_head` bf16.
- **Expected gain**: ~1.10–1.15× end-to-end (we already ate most of the win).
- **Pitch line**: FP8 is **complementary** to BitNet — covers the residual fp
  surface ternary doesn't reach.

### Out of scope (do NOT implement)

- FP8 emulation on A800 (negative ROI, measured ~20% slower than bf16).
- e4m3 weight storage (we use ternary — better compression).
- Mixed FP8 / INT8 (TE doesn't do this; no benefit at our scale).

## 5. References

| ID | Topic | URL |
|----|-------|-----|
| BitNet b1.58 (2402.17764) | ternary 1.58-bit weights | https://arxiv.org/abs/2402.17764 |
| DeepSeek-V3 (2412.19437) | FP8 e4m3+e5m2 mixed training, 671B MoE | https://arxiv.org/abs/2412.19437 |
| FP8 Formats (2209.05433) | original FP8 spec | https://arxiv.org/abs/2209.05433 |
| TransformerEngine | NVIDIA reference FP8 lib | https://github.com/NVIDIA/TransformerEngine |
| `torch.ao.quantization` | int8 PTQ + QAT toolkit | https://pytorch.org/docs/stable/quantization.html |

## 6. Cross-refs

- `docs/quantize.md` — BitNet QAT front-end (shipped T2.8)
- `docs/MATMUL_FREE.md` — M1/M2/M3 BitLinear roadmap (recurrent backbone)
- `synapforge/quantize.py` — `TernaryLinear` + `convert_model_to_ternary`
- `docs/PERF_KNOBS.md` — per-knob inference latency log

## 7. Status

| Item | Status |
|------|--------|
| Research doc (this file) | shipped 2026-05-02 |
| `--int8-inference` CLI + `torch.ao` PTQ wire-in | not started, post phase-3 |
| H100/H200 rental + TE FP8 | deferred ~Q3 2026 |
| DeepSeek-V3-style per-block scaling for CfC | deferred ~Q3 2026 |

> **Bottom line**: do not implement FP8 emulation on A800 (net loss).
> Pre-Hopper = int8 PTQ inference. Post-Hopper = TE-style FP8 training.
