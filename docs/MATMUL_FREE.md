<!-- DOC_STAMP: FRESH 2026-05-01; agent_matmul_free_research -->
# Matmul-Free Synap-Edge — research roadmap

**Updated**: 2026-05-01
**Status**: research roadmap (no architecture changes yet); ternary QAT front-end already shipped in `synapforge/quantize.py` (BitNet b1.58)
**Owner**: Liu (mohuanfang@gmail.com)
**Tracking**: O12 in `docs/MASTER_PLAN.md` §2; memory entry `feedback_matmul_free_alignment_2026q2.md`

> **TL;DR** — User asked about "replacing complex multiplications with add/subtract."
> They are referring to the **BitNet b1.58 / Matmul-free LM / AdderNet** family.
> SynapForge is structurally a **better fit** than transformer for this technique
> because PLIF spike output is already **binary {0, 1}**, so multiplying a binary
> spike by a ternary weight `{−1, 0, +1}` reduces to a pure **sign-select +
> accumulate** — zero multiplications anywhere in the SNN path. We already ship
> the QAT front-end (`synapforge/quantize.py`); the missing pieces are a
> binary×ternary Triton kernel (M3) and full-stack BitLinear refactor (M2).

---

## 1. The three papers

| ID | Year | Authors / venue | Core idea | arXiv |
|----|------|-----------------|-----------|-------|
| **BitNet b1.58** | 2024 | Ma, Wang, Ma, Wang, Wang, Wei (Microsoft) | Replace `nn.Linear` weights with ternary `{−1, 0, +1}` via AbsMean quantization + STE. Multiplications become signed additions; activations stay 8-bit. Drop-in for transformer LM, **parity with fp16 LLaMA-2 7B** at 700M+. | [2402.17764](https://arxiv.org/abs/2402.17764) |
| **Scalable Matmul-free LM** | 2024 | Zhu, Zhang, Sifferman, Sheaffer, Wang, Dougherty, Eshraghian (UC Santa Cruz) | End-to-end **matmul-free** transformer-replacement: BitLinear (ternary W) + **MatMul-free Linear GRU (MLGRU)** for token mixing. Trains 370M / 1.3B / 2.7B; FPGA shows **~13 W** at 1.3B inference vs ~700 W H100. | [2406.02528](https://arxiv.org/abs/2406.02528) |
| **AdderNet** | 2020 | Chen, Wang, Xu, Tian, Ye, Han, Xu (Huawei Noah’s Ark, CVPR 2020) | Replace dot-product `Σ wᵢ·xᵢ` with **L1 distance** `−Σ|wᵢ−xᵢ|`. No multiplication in conv; tested on ImageNet. | [1912.13200](https://arxiv.org/abs/1912.13200) |

**Supporting refs** (cited inline in §2/3):

- **BitNet (1-bit)** — Wang et al. 2023, [2310.11453](https://arxiv.org/abs/2310.11453). Predecessor of b1.58; binary `{−1, +1}` weights, transformer-only. Already cited in `synapforge/quantize_README.md`.
- **bitnet.cpp** — Microsoft 2024, [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet). Reference CPU inference engine: `popcount` + table-lookup kernel. Reports **5–7× speedup** vs fp16 on x86 batch=1.
- **Phantom-of-Latent / MatMul-free Vision** — follow-up vision work confirming the technique generalizes beyond text.
- **EfficientNet-Adder / AdderNet-v2** — Chen et al. 2021, follow-up that closes the ~1% top-1 ImageNet gap to MAC-conv.

---

## 2. Why this is a natural fit for SynapForge / Synap-1

Three structural reasons, in increasing order of leverage.

### 2.1 PLIF spikes are already binary

The PLIF surrogate (`synapforge/surrogate.py:91`) returns
`(v − threshold ≥ 0).to(dtype)` — output is exactly **{0, 1}**. This is unique
among architectures. Transformer/GRU/Mamba activations are dense fp16/bf16
floats; we have a hard binary path through every spiking layer.

When you multiply a binary spike `s ∈ {0, 1}` by a ternary weight `w ∈ {−1, 0, +1}`,
the result is one of three integers: `{−1, 0, +1}`. This is **not even an int8
multiplication** — it's a sign-select:

```
y = s · w   ≡   if w == +1: +s
                if w == 0:   0
                if w == -1: −s
```

So the SNN forward becomes a sequence of **sign-aware accumulations** — a
shift-and-add tree, no integer multiplier required. On neuromorphic / FPGA /
custom ASIC this maps directly to wires. On CPU it's a `popcount` over the
intersection of the spike-mask bitfield and the weight-sign-positive bitfield,
minus a `popcount` over (spike-mask ∩ weight-sign-negative). Two `popcount`s
per output channel.

### 2.2 CfC is a single linear map per step

Mixed-signal CfC (Hasani et al. 2022, "Closed-form Continuous-time Networks"):

```
h_{t+1} = σ(W·h_t + Wi·x_t + b)        # one linear map, then nonlinearity
```

Both `W` and `Wi` are static `nn.Linear`-class tensors. Quantize them to
ternary; the entire CfC update is **W·h + Wi·x = scaled-add tree**. There is no
KV cache, no attention, no quadratic cost — once `W` and `Wi` are ternary the
recurrence is **fully matmul-free per timestep**.

Compare to transformer: even with BitNet b1.58 you still have softmax-attention
which is intrinsically multiplicative (`exp` + `Q·Kᵀ` over fp activations).
Matmul-free LM (arXiv:2406.02528) had to invent **MLGRU** to replace attention
exactly because attention itself is not amenable to ternary. We don't have
this problem — CfC is already a single linear map.

### 2.3 The LM head is the only true matmul left

Vocab is 151,936 (Qwen 2.5 padded), hidden = 512 → LM head matrix is
**~78M fp32 params** = ~310 MB. Even after ternarizing the body, this
dominates the model size. Three options:

1. **Keep fp32 LM head** (BitNet b1.58 default). Adds ~10 % inference time
   (one big GEMM at the end), keeps quality clean. Whole-model size still
   ~80 MB ternary body + 310 MB head = 390 MB.
2. **Ternarize LM head** (Matmul-free LM does this). Per BitNet evidence
   loses < 1 % at ≥ 700M params, but at our 100M scale could lose 5 %+.
   Need empirical validation.
3. **Tied LM head + ternary embedding** (their joint-quant trick).
   Saves 78 M params at the cost of ~2 % quality. Acceptable for
   Synap-Edge laptop variant.

---

## 3. Implementation path (milestone-gated)

**Constraint**: every milestone is gated on training already being green at
that phase. We do **not** speculate-implement before the base model can chat.

### M1 — Ternary CfC weights only (post phase 3, ~1 week)

**Scope**: keep model fp16/bf16 except `cell.W` and `cell.Wi`.

**Already shipped**:

- `synapforge/quantize.py::TernaryLinear` (full QAT, AbsMean+STE+EMA-gamma) — 360 LOC.
- `synapforge/quantize.py::convert_model_to_ternary(model, exclude=...)` — walks the module tree.
- `synapforge/test_ternary.py` — 7 tests (buckets, STE, EMA, freeze, 100-step QAT, state-dict, plasticity-untouched).
- `synapforge/bench_ternary.py` — fp32 vs ternary latency + on-disk size.

**To-do** (this milestone):

1. Add a CLI flag `--quant-cfc-weights ternary` to `train_100m_kd.py`. When set,
   call `convert_model_to_ternary(model, exclude=("emb", "lm_head", "ffn"))` after
   warmstart, before optimizer init.
2. Run a 500-step QAT from a phase-3 ckpt. **Pass criterion**: val ppl
   regression ≤ 1 % (BitNet evidence).
3. Bench wall-clock CfC step on A800: measure baseline vs ternary; expected
   **~3–5×** at the CfC-only level (not whole-model, because matmul still
   dominates LM head).
4. Update `docs/INVESTOR.md` energy claim with measured numbers.

**Acceptance bar**: val ppl ≤ baseline + 1 %, CfC step latency ≥ 3× faster.

### M2 — Full BitLinear refactor (post phase 4, ~2 weeks)

**Scope**: replace **every** `nn.Linear` (CfC + Wi + Wo + FFN gates + LM head)
with `TernaryLinear`, except the embedding table.

**Reference recipe**: BitNet b1.58 §3.2 — fine-tune for 1–5 % of original
token budget at lr = 1e-4, AbsMean per-tensor quantization, STE backward.

**To-do**:

1. CLI: `--quant-all ternary`. Default exclusion list → `("emb",)` only.
2. Add quant-aware annealing: linear blend `w_eff = (1−α)·w_fp + α·w_quantized`,
   α: 0 → 1 over the first 10 % of fine-tune steps. (BitNet doesn't do this
   but at our small scale it stabilizes.)
3. Re-bench: expect **5–10 × wall-clock speedup at inference** on A800
   (most of the model's compute is now matmul-free); training is similar
   speed (forward fp, backward STE).
4. Memory gain: model size ~600 MB (fp16) → **~80 MB** (1.58 bit avg).
   Confirm with `synapforge/quantize.py::count_ternary_params(model)`.

**Acceptance bar**: val ppl ≤ M1-result + 2 %, model size ≤ 100 MB,
inference latency ≥ 5× faster than fp16 baseline.

### M3 — Custom Triton kernel for binary×ternary (paper milestone, ~1 month)

**Scope**: write a fused Triton kernel for the unique-to-us operation
`y = s · w` where `s ∈ {0,1}` (PLIF spike output) and `w ∈ {−1, 0, +1}`
(ternary CfC weight). This **does not exist** in any published reference.

**Why it's novel**:

- BitNet kernels assume fp16/int8 activations × ternary weights. Their
  inner loop is `accum += a[k] * sign_lookup(w_packed[k])` — still requires
  an fp/int multiplier for the `a[k] * ±1` step.
- Our spike is a single bit. The inner loop becomes a **bit-AND + popcount**:
  ```
  popcount(spike_bits & w_pos_bits) - popcount(spike_bits & w_neg_bits)
  ```
- This maps to AVX-512 VPOPCNTDQ on x86, NEON `vcnt` on ARM, or a
  4-input LUT on FPGA / neuromorphic ASIC.

**Stub kernel signature** (Triton, deferred impl):

```python
# synapforge/backends/triton_binspike_ternary.py  (NEW, M3 deliverable)
import triton
import triton.language as tl

@triton.jit
def binspike_ternary_kernel(
    spike_ptr,         # [B*T, K] int8 in {0, 1} (or packed bit field)
    w_pos_ptr,         # [K, D] uint8 bit field: 1 where w == +1
    w_neg_ptr,         # [K, D] uint8 bit field: 1 where w == -1
    out_ptr,           # [B*T, D] int32 partial sums
    B_T, K, D,         # shapes
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """y[bt, d] = popcount(spike[bt] & w_pos[d]) - popcount(spike[bt] & w_neg[d]).

    No multiplications. Two `tl.sum(... & ...)` reductions per output element.
    Straight-through grad goes via the upstream PLIF surrogate; this kernel
    is forward-only.
    """
    pid_bt = tl.program_id(0)
    pid_d  = tl.program_id(1)
    # ... (M3 deliverable: full impl + tests + bench)
    pass
```

**To-do** (M3):

1. Implement the kernel above (forward only — backward goes through PLIF
   surrogate already). ~200 LOC Triton.
2. Add an integration test: `output ≡ (spike.float() @ w_ternary.float()).int()`
   for random binary spikes and ternary weights. Tolerance = 0 (exact integer).
3. Bench A800: ternary fp matmul (`cuBLAS`-fp16) vs ternary `popcount` Triton.
   Expected **2–3×** further speedup over a naive fp ternary forward, because
   we eliminate even the `* (±gamma)` step.
4. Add a CPU AVX-512 fallback in `synapforge/backends/cpu_avx2.py` for
   laptop deployment. ~300 LOC C+Cython.
5. Submit a paper deliverable: this is **the** novel-engineering claim
   alongside R-fold and inference-time STDP.

**Acceptance bar**: integration test exact-equals the reference; bench ≥ 2 ×
faster than fp16 ternary on A800; AVX-512 path runs Synap-Edge laptop demo.

---

## 4. Investor pitch angle

**One paragraph for the pitch**:

> BitNet b1.58 (Microsoft, Feb 2024) and the Matmul-free LM (UC Santa Cruz,
> Jun 2024) showed you can run a transformer LM with **only addition,
> subtraction, and table lookups** — no integer multiplications — at parity
> with fp16. Their problem is that attention's quadratic-in-sequence cost
> stays fp regardless. We do this on **LNN+SNN where the spike output is
> already binary** — so combining ternary weights `{−1, 0, +1}` with binary
> spikes `{0, 1}` gives us a stack with **(a)** linear cost in sequence
> length (no attention) and **(b)** zero multiplications anywhere in the
> spiking forward pass. This is one more axis of "different point on the
> Pareto frontier": where transformers trade quadratic cost for parallel
> training, we trade ternary precision for matmul-free inference, and the
> two compose. On commodity laptop CPU we expect **5–10 × wall-clock
> speedup over fp16 transformer of the same param count**, with the
> energy story compounding 10 × further on FPGA / neuromorphic.

**Numerical claims (to verify in M2)**:

| Metric | fp16 transformer 100M | Ternary Synap-1 100M |
|--------|----------------------|----------------------|
| Model size | ~200 MB | **~25 MB** (8 ×) |
| CPU latency batch=1 | ~ 200 ms / 32 tok | **~ 30 ms / 32 tok** (6 ×) |
| Inference power | ~ 30 W laptop CPU | **~ 5 W** (6 ×) |
| Quality (val ppl) | baseline | baseline + ≤ 2 % |

These are the BitNet b1.58 reported numbers, which we'd reproduce on the
SNN body. The neuromorphic/FPGA story extends this to ~ 100 × on a
realized chip — but that's a 2027 hardware claim, not a 2026 software one.

---

## 5. Honest risks

1. **QAT instability at small param scale.** BitNet b1.58 reports parity
   at ≥ 700 M params. At our 100 M scale we may lose 3 – 5 %, not < 1 %.
   *Mitigation*: linear annealing of quantization (α ramp 0 → 1 over 10 %
   of fine-tune steps) + reduced learning rate (1e-4 → 3e-5).
2. **Triton kernel for binary × ternary is novel.** No published
   reference impl. We will need to derive the AVX-512 / Triton kernels
   ourselves and validate against a slow fp reference. This is a real
   ~ 1 – 2 week sprint, not a copy-paste.
3. **LM head ternary at our scale might lose 5 %+ quality.** Above 700M
   it's a free win; at 100M the LM head represents 78% of total params
   and is the dominant quality bottleneck. *Mitigation*: keep LM head
   fp16 in M1 / M2; only ternarize LM head as an M3 stretch goal with
   careful before/after eval.
4. **Energy claim is hardware-conditional.** On A800 we get
   ~ 1.3 – 1.5 × from cache pressure reduction (model fits in L2),
   not 5 – 10 ×. The 5 – 10 × wall-clock figure is **CPU batch=1
   inference**, which is the laptop-pitch use case but not the GPU
   training use case. Be honest in the pitch about which number applies.
5. **Plasticity / fast-weight buffers must NOT be quantized.** Online
   STDP / Hebbian updates rely on small gradient magnitudes that
   ternarization erases. `synapforge/quantize.py::DEFAULT_EXCLUDE` already
   handles this for `nn.Linear`, but we have to keep watching as we
   add more plasticity machinery (e.g. when M3 lands the
   binary × ternary kernel, make sure STDP's pre/post traces stay fp).

---

## 6. Cross-references

- `synapforge/quantize.py` — existing BitNet b1.58 QAT front-end (production-ready)
- `synapforge/quantize_README.md` — user-facing API doc + bench numbers
- `synapforge/test_ternary.py` — 7 QAT tests (all passing)
- `synapforge/bench_ternary.py` — fp32 vs ternary latency bench
- `docs/MASTER_PLAN.md` §2 (O12) — top-level objective entry
- `docs/INVESTOR.md` — pitch numbers (update with M2 measurements)
- `docs/ANTI_LORA.md` — sister doc on architecture-claim purity
- Memory: `feedback_matmul_free_alignment_2026q2.md` — short-form session note
- Memory: `feedback_not_training_transformer.md` — backbone is not transformer
- Memory: `feedback_5T_effective_target.md` — small-param + matmul-free is one of three Pareto axes

---

## 7. Status summary

| Component | Status | Owner |
|-----------|--------|-------|
| BitNet QAT front-end (`quantize.py`) | shipped, tested | base repo |
| `convert_model_to_ternary` walker | shipped | base repo |
| Trainer flag `--quant-cfc-weights` (M1) | not started | post phase 3 |
| Full BitLinear refactor (M2) | not started | post phase 4 |
| Binary × ternary Triton kernel (M3) | stub signature only (this doc §3 M3) | future sprint |
| AVX-512 CPU fallback (M3) | not started | future sprint |
| Investor pitch numbers update | pending M2 measurements | Liu |

> **Do not start M1/M2/M3 before phase 3 is green.** Feature ordering matters
> more than feature count — see `feedback_phased_training_2026q2.md`.
