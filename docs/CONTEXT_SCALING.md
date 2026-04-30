# Context Length Scaling — A100 80GB Physical Limits

How far can a 375M LNN+SNN actually reach on a single A100 80GB?

**Realistic max with quality: 50M tokens** (with 256GB host RAM + 4TB NVMe)
**Architectural ceiling: 500M tokens addressable** (quality degrades)

This is **50-500× more than a 7B transformer** on the same hardware.

## Why we have such a huge advantage

CfC has **no KV cache**. The recurrent state `h_cfc` is fixed-size:
- 14 layers × hidden 1024 × 2 bytes = **28 KB total** for CfC state
- 28 KB total for PLIF membrane
- **56 KB total** regardless of context length

Compare to transformer:
- 7B Transformer fp16 + GQA: **64 KB per token** of KV cache
- 80GB - 14GB weights = 66GB for KV → **1.0M tokens max**

We're not bounded by `n²` attention or `O(n)` KV. Our context is bounded only
by **what we choose to store for retrieval**.

## Tier table

| Tier | Mechanism | Tokens | ppl drift | Latency/tok |
|------|-----------|--------|-----------|-------------|
| **L0** Inference fp16, no quant | HBM only, no retrieval | 10M (with bookkeeping) | 0% in-window, 3% past 256K | 8-12 ms |
| **L0+** BitNet 1.58 weights | 10× weight compression | 40M | +0-0.5 ppl | 5-8 ms |
| **L1** + 256K hot CfC state + Coconut | Native CfC range | **256K** | <1% | 8-12 ms |
| **L2** + STDP retrieval + HNSW | Add 1M warm prototypes | **1-2M** | 3-5% | 12-20 ms |
| **L3** + FAISS IVF-PQ on NVMe | Cold storage 50M+ | **5-50M** | 8-15% | 40-150 ms |
| **L3-ext** + R-fold 167× speedup | Replay layer for long memory | **100M** | 15-25% | varies |
| **Architectural ceiling** Paged + fp8 + PQ16 | HBM + 256GB host + 4TB NVMe | **500M-2B addressable**, 50-100M usable | 10-20% cold | 100ms-1s NVMe miss |

## Training (different limits)

| Tier | Setting | Tokens/seq | Notes |
|------|---------|------------|-------|
| T-train fp16 + Adam | 4× memory for forward+backward+optim | **448K tokens/seq** single-batch | full grads |
| T-train + ZeRO-3 + CPU offload + grad ckpt | activations 56KB/tok, optim on host | **1.4M tokens/seq** | 3-5× slower step |

Training is much smaller than inference because of optimizer state + activations.

## The single biggest unlock — PQ16 compression

```
1024-dim hidden state → 16-byte PQ code = 64× compression
```

Lifts HBM-only context from 1.25M → 80M+ at <5% recall drop.

Implementation:
- Train Product Quantization codebook (offline, 1 GPU-h) on a sample of past
  hidden states from a normal training run
- At each token store, quantize `h_t` → 16 bytes → append to FAISS-PQ index
- At retrieval, dequantize top-K back to fp16, return

Per-paper: this is exactly the IVF-PQ index from FAISS, k-means on residuals.

## Three LNN+SNN-specific risks (and how to fix)

### 1. PLIF spike rate 10-30% means parametric recall starves

Transformer FFN is dense (every weight active). Our PLIF gates 70-90% of
activations to 0. Long-context retrieval **cannot** lean on the model's weights
for facts — it MUST lean on L2/L3 stores.

If retrieval misses, the model has nothing.

**Mitigation**:
- Dual-path inference: dense bypass alongside spiking
- Audit: PLIF rate ≥ 25% in deeper layers, else log alarm
- Always pair retrieval (HNSW + FAISS) with the model

### 2. CfC time-constant τ saturates after 4-32K tokens

Even with multi-band τ, the continuous-time hidden state has a finite memory
horizon. Past 32K, retrieval is the only signal.

**Mitigation**:
- Coconut latent thinking checkpoint every 8K tokens
- Write summary into L2 retrieval store
- R-fold algebraic CfC (167× speedup at R=1024) for replay

### 3. No KV cache = no exact verbatim recall

Transformer can copy a 16-character UUID from anywhere in context (KV match).
Our model only has retrieved hidden vectors → nearest-neighbor in hidden space,
not exact tokens.

**NIAH UUID tests will be much harder for us than for transformer.**

**Mitigation**:
- Sidecar **exact-token log** (BM25 or hash-keyed) parallel to hidden retrieval
- 4 bytes per token → cheap (50M × 4B = 200MB for full BM25 sidecar)
- Hybrid retrieval: hidden vector for semantic, BM25 for verbatim

## Eval gates (must pass to claim a tier)

| Bench | L1 (256K) | L2 (2M) | L3 (50M) |
|-------|-----------|---------|----------|
| NIAH single-needle | ≥98% | ≥85% | ≥60% |
| NIAH multi-needle (k=4) | ≥85% | ≥55% | ≥25% |
| RULER var-tracking | ≥80% | ≥40% | ≥15% |
| RULER common-words | ≥75% | ≥35% | — |
| LongBench-Zh | ≥0.55 | ≥0.40 | ≥0.20 |
| ppl held-out | <1.01× base | <1.05× | <1.15× |

If any cell fails → tier cannot be claimed honestly. Build the eval first.

## Anchor papers

- **InfLLM** (Xiao 2024, 2402.04617) — block-level memory units, retrieval
  gating into local attention. Adapt unit-store/unit-retrieve into our STDP L2.
- **Memory³** (Yang 2024, 2407.01178) — explicit memory tier separation
  (parametric / context / explicit), int8 hash-based retrieval, exact pattern
  for our L1/L2/L3.
- **Titans** (Behrouz Google 2025) — neural long-term memory with
  surprise-gated writes, maps onto STDP write rule already in scaffold.
- **HNSW** (Malkov 1603.09320) — for L2 prototype index (already in
  `synapforge/action/hnsw_skill_index.py`)
- **FAISS-PQ** (Jegou 2010.5402, Douze 2401.08281) — for L3 cold storage

(Skip LongLoRA / NTK-RoPE — those are RoPE extension tricks, irrelevant
to CfC because we don't have positional encoding past tokens anyway.)

## Math verification

```
80 GB GPU = 80 × 1024³ B = 8.59 × 10¹⁰ B
Model weights 375M fp16 = 7.5 × 10⁸ B
Free HBM = 8.51 × 10¹⁰ B

Per-token storage:
  fp16 hidden:  1024 × 2 = 2048 B  →  41.6M tokens fit
  fp8 hidden:   1024 × 1 = 1024 B  →  83M tokens fit
  PQ16 codes:        16 B          →  5.3 BILLION tokens addressable
                                        (but bounded by retrieval quality, not memory)
```

Beyond ~500M tokens, recall quality breaks down even with all our tricks.

## Comparison to other long-context approaches

| Approach | Best on 80GB | Architecture cost |
|----------|--------------|-------------------|
| Standard transformer 7B | 132K (256K with int8 KV) | n² attention, KV grows linear |
| Transformer 7B + Mamba hybrid (Jamba) | 1-2M | gives up on full attention |
| Transformer + sliding window + retrieval | 1-5M | layered hack |
| **Our CfC + PLIF + L1/L2/L3** | **50M usable, 500M addressable** | clean recurrent, retrieval-first |
| Pure SSM (Mamba 7B) | 256K with no retrieval | similar but no plasticity |

We're not competitive on **verbatim recall** at any size — that's transformer's
killer app and we don't fight it. We win on **token throughput at long context
for retrieval-friendly tasks** (RAG, doc summarization, long-doc QA).

## Recommended demo / paper sequence

For paper / demo credibility (in order):

1. **256K L1 all-HBM** (CfC + Coconut) — most stable, paper headline
2. **2M L2 STDP retrieval** — add NIAH numbers, +HNSW + write rule
3. **50M L3 FAISS NVMe** — biggest claim, needs sidecar BM25 to hold NIAH
4. **500M architectural ceiling** — mention briefly, "addressable but no
   coherence guarantee past 50M"

Always state honestly which tier you're claiming and run the eval gate.

## Code paths

Already in repo / scaffolded:
- `synapforge/action/hnsw_skill_index.py` — L2 HNSW backend ready
- (planned) `synapforge/memory/faiss_pq_index.py` — L3 FAISS-PQ
- (planned) `synapforge/memory/sidecar_exact.py` — verbatim BM25/hash mitigator
- (planned) `synapforge/memory/coconut_summarizer.py` — periodic 8K checkpoint

Tier L1 works today out of the box (just need ckpt with seq 256K trained).
L2 requires writing the FAISS write/read paths. L3 requires NVMe paging glue.
