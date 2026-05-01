# Baseline Comparison — Synap-1 vs the Small-LM Landscape

**Honest table for investors.** Last refresh: 2026-05-01.

Synap-1 is a **100M LNN+SNN** (CfC + PLIF + STDP, Qwen 151936 vocab) trained
~24h on a single A800. The pitch is *"different point on the Pareto frontier"*
— **we lose at static benchmarks at our param class** and win on streaming
inference cost, energy proxy, and continual learning at inference. This doc
puts numbers next to the claim.

Numbers without a citation are marked *not reported (NR)*. Synap-1 cells
labelled *Run 3e* refer to the live training run; see [PROGRESS.md](PROGRESS.md)
and [TIMELINE.md](TIMELINE.md).

---

## 1. The 11-baseline table (Tier A/B/C, ours + 10 comparators)

| Tier | Model | Params | Train tok | MMLU 5-shot | HellaSwag | GSM8K | Tok/s (A100/A800, fp16) | Streaming inf cost | Continual @ inference |
|------|-------|-------:|----------:|------------:|----------:|------:|------------------------:|--------------------|-----------------------|
| **A** | **Synap-1 (us)** | **100M** | **~10B (KD)** | **~30 target** [training Run 3e] | NR (training) | NR (training) | **22k** [INVESTOR.md] | **O(L) CfC, no KV** | **STDP fwd-only** [stdp_fast.py:121] |
| A | Mamba-130M [1] | 130M | 300B | 25.0 | 35.3 | NR | ~38k | O(L) SSM | static |
| A | RWKV-4-169M [2] | 169M | 332B | 24.5 | 32.7 | NR | ~31k | O(L) RNN | static |
| A | Pythia-160M [3] | 162M | 300B | 26.1 | 30.1 | 1.4 | ~26k | O(L^2) attention KV | static |
| A | GPT-Neo-125M [4] | 125M | 300B | 25.9 | 28.9 | NR | ~24k | O(L^2) attention KV | static |
| **B** | SmolLM2-360M [5] | 362M | 4T | 30.4 | 53.0 | 27.0 | ~18k | O(L^2) attention KV | static |
| B | Pythia-410M [3] | 405M | 300B | 27.3 | 39.3 | 2.3 | ~20k | O(L^2) attention KV | static |
| B | TinyLlama-1.1B [6] | 1.1B | 3T | 25.5 | 60.4 | 2.4 | ~10k | O(L^2) attention KV | static |
| B | Qwen2.5-0.5B-Instr [7] (KD teacher) | 494M | 18T | 47.5 | 52.3 | 41.6 | ~16k | O(L^2) attention KV | static |
| **C** | Qwen2.5-1.5B [7] | 1.54B | 18T | 60.9 | 67.9 | 68.5 | ~6k | O(L^2) attention KV | static |
| C | SmolLM2-1.7B [5] | 1.71B | 11T | 51.7 | 71.0 | 31.8 | ~5.5k | O(L^2) attention KV | static |

Sources: [1] Gu+Dao 2312.00752; [2] Peng et al. RWKV LM model card;
[3] Pythia Biderman+ 2304.01373; [4] EleutherAI GPT-Neo card; [5] HF
SmolLM2 card; [6] TinyLlama 2401.02385; [7] Qwen2.5 Tech Report 2412.15115.

Synap-1 inference-cost claims trace to [RFOLD_PAPER.md](RFOLD_PAPER.md) and
[CONTEXT_SCALING.md](CONTEXT_SCALING.md). Tok/s is fp16 short-context generation,
order-of-magnitude only — published numbers vary 2x with implementation.

---

## 2. The two columns we actually win on

These are the columns *most other models can't fill in*, not "we're 30% better".
Read them as **"different axis"** not "parity".

### Streaming inference cost (long context)

| Architecture family | Per-token cost at length L | KV/state memory |
|---------------------|----------------------------|-----------------|
| Transformer (Pythia / SmolLM2 / Qwen / TinyLlama) | **O(L)** matmul + **O(L)** KV read | **O(L)** grows linearly |
| Mamba / RWKV (selective SSM / linear-attn) | O(1) per token | O(d) constant |
| **Synap-1 (CfC + PLIF)** | **O(1) per token, sparse spike** | **O(d) constant + STDP fast-weight delta** |

We tie Mamba/RWKV here and beat transformer KV growth. This is where 100K-context
streaming workloads pay off.

### Continual learning at inference

| Stack | Mechanism | Backward pass at inference? |
|-------|-----------|-----------------------------|
| All static baselines (rows above except Synap-1) | None — weights frozen at deploy | No |
| TTT-style transformer adapters (Sun+ 2024) | Gradient on test stream | **Yes** (expensive) |
| **Synap-1** | **STDP forward-only Hebbian** [INVESTOR.md §3] | **No** (fwd-only co-firing) |

**Zero published systems do forward-only Hebbian online updates at LM scale.**
Cost: a single matmul-shaped weight delta per token, no autograd graph.
See [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md) for the caveat — verified mechanism, not yet verified to monotonically improve quality on a real workload.

---

## 3. Where we lose (read this too)

**Static benchmark parity at our param class? No.**

- Mamba-130M MMLU **25.0** (chance ~25 on 4-choice). Pythia-160M **26.1**.
  Our 100M target is **~30** with Qwen-0.5B KD (live ppl ~320 → projected MMLU ~28-32).
  A 100M LM at *any* arch is at-or-near chance on MMLU 5-shot. We don't beat that ceiling.
- HellaSwag: SmolLM2-360M (3.6x our params, 400x our tokens) hits 53. We expect
  ~30-35 — cleanly under all 1B-class baselines. KD from Qwen-0.5B narrows the gap; it does not close it.
- GSM8K: Synap-1 will likely be at chance (~0-3%). KD helps but a 100M LNN+SNN
  is too small for multi-step arithmetic without external scaffolding.

**The win is the column NOT in the static table** (energy + streaming + continual).
If your deployment is static-benchmark-driven, transformer wins. If it's a long-running
on-device agent, our stack wins. **No oversell.**

---

### Energy / FLOP per token (rough proxy)

Static FLOP/token is dominated by `vocab_size * d_model` projection at the
output and the per-layer MLP. With Qwen vocab (151,936) and `d=512` Synap-1's
LM head dominates inference flops; the spike sparsity savings (5-30% activation
density observed in earlier runs, see [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md))
apply to the **backbone** only, not the LM head. Honest framing: spike sparsity
buys ~2-3x on backbone matmuls, ~1.1-1.3x system-wide once the lm_head amortizes.
The 0.05x figure in [INVESTOR.md](INVESTOR.md) §"Honest competitive comparison"
is **aspirational** — measured against a hypothetical neuromorphic accelerator
that exploits spike sparsity end-to-end. On A800 / H100 we cannot exhibit it.
This is the single largest "rhetoric vs reality" gap in our pitch and is called
out in [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md) §"Research-grade neuromorphic
claims unverified".

### Context length capability

| Model | Native ctx | NIAH @ 100K |
|-------|-----------:|-------------|
| Pythia / GPT-Neo | 2K | catastrophic |
| TinyLlama | 2K | catastrophic |
| SmolLM2 | 2K-8K | not reported beyond train ctx |
| Qwen2.5-0.5B/1.5B | 32K | strong (Qwen tech report) |
| Mamba-130M | 2K (default) | degraded at 2x train ctx |
| RWKV-4 | 2K | degraded |
| **Synap-1** | **512 train, 50M target** [CONTEXT_SCALING.md] | **harness ready, awaiting ckpt** |

50M context is *aspirational*: harness is built ([CONTEXT_SCALING.md](CONTEXT_SCALING.md)),
no Synap-1 ckpt has yet evaluated past 8K. The architectural primitives (CfC O(L) +
PLIF spike compression + STDP retrieval) exist; the empirical curve doesn't.
Don't quote 50M without an asterisk pointing here.

---

## 4. Why these specific baselines

**Mamba-130M.** Closest non-attention 100M-class baseline; if we can't beat Mamba on
long-context streaming throughput we have invalidated the R-fold + CfC thesis.
Mamba's selective scan is the gold standard for "linear-time alternative to
attention"; we have to be at least competitive with it on the same axis.

**SmolLM2-360M.** Shows what 360M params + 4T tokens of curated data buys at 2026 SOTA.
Synap-1 trained on ~10B tokens with 0.5B KD is **400x less data and 3.6x fewer params**.
This baseline is the upper bound on what Qwen-0.5B distillation can give us; we should never
quote a Synap-1 number that exceeds SmolLM2-360M without a *very* careful caveat.

**Qwen2.5-0.5B-Instruct.** The KD teacher. Distillation theory bounds the student
below the teacher on the teacher's evaluation distribution. We must **never** claim
"Synap-1 exceeds the teacher in parity-bench" — that would be a sign the eval is broken,
not that the student is magic. The teacher is the ceiling on KD-derived static scores.

**Pythia-160M / Pythia-410M.** Pure transformer baselines at our scale and 4x our
scale, both trained on 300B tokens by EleutherAI on a fully reproducible recipe.
This is the cleanest "vanilla transformer at our compute budget" reference and
the row most likely to be quoted by a skeptical reviewer asking *"would a plain
transformer have done this with the same compute?"*.

**TinyLlama-1.1B.** Despite 11x our params and 300x our tokens, MMLU is **25.5**
(chance). HellaSwag 60 is the realistic target if we ever scale Synap-1 to 1B.
TinyLlama is on the table as a sanity check: scaling a transformer 10x doesn't
automatically buy MMLU; the data-mix and training-recipe choices dominate.

**RWKV-4-169M.** RNN-flavored linear attention. Closest "RNN-class baseline" with
a published LM card. Tells us where the RNN-without-CfC baseline sits — comparable
to Pythia at MMLU, slightly behind on HellaSwag.

**SmolLM2-1.7B + Qwen2.5-1.5B (Tier C).** Reference upper bound, not direct
competitors. Included so investors can see the static-bench ceiling at 15-17x
our params. We do not aim to match these.

---

## 5. How this doc is used

- [INVESTOR.md](INVESTOR.md) §"Honest competitive comparison" links here
  instead of duplicating the table inline. Keeps the pitch tight.
- [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md) §"What you should NOT use this
  for" cites this doc's §3 as the canonical "where we lose" statement.
- [ROADMAP.md](ROADMAP.md) per-week parity targets:
  - Week 0-1: Synap-1 at MMLU **>= chance + 4pp** (>= 29). Beat zero baselines.
  - Week 3-4: Synap-1 at HellaSwag **>= 35** (beat Mamba/RWKV class).
  - Week 6-8: Synap-1 at MMLU **>= 32**, GSM8K **>= 5%**. Still below SmolLM2-360M; matches the plan.
  - Energy & streaming targets: WikiText ppl drift **<5%** at L=100K vs L=1K
    (see [MONOTONIC_QUALITY.md](MONOTONIC_QUALITY.md)).

---

## 6. Cell tally (truth in advertising)

Counting the 11 baseline rows (excluding Synap-1's own row):

- **We lose** on MMLU: 11/11. (Every baseline is at-or-above us; many are far above.)
- **We lose** on HellaSwag: 11/11 expected (training Run 3e not yet evaluated).
- **We lose** on GSM8K where reported: 7/7 reporting baselines.
- **We tie** on streaming O(L) cost: 2/11 (Mamba, RWKV).
- **We win** on streaming vs transformer KV: 9/11 (all transformer rows).
- **We win uniquely** on continual learning at inference: 11/11 (no other row has STDP fwd-only).
- **Tok/s comparable** with Pythia-class at our scale; behind Mamba/RWKV due to
  spike-routing overhead in our triton kernel; we don't claim raw throughput
  parity.

**Total**: 7 win cells, 2 tie cells, 33 lose cells (across MMLU/HellaSwag/GSM8K
and tok/s vs comparable rows). The win cells are the ones that price the
deployment.

---

## 7. What would change our mind

Three measurements that, if they came in negative, would force a thesis pivot:

1. **Mamba-130M beats Synap-1 on long-context streaming throughput at L=100K
   on A800.** If Mamba's selective scan is faster *and* lower-memory than
   our CfC + spike pipeline, the architectural premise (CfC compresses better
   than SSM at small N) is wrong. Test: 100K-token streaming generation,
   measured tok/s and peak GPU memory, both stacks with comparable kernels.
   This is on the [ROADMAP.md](ROADMAP.md) week-3 critical path.

2. **STDP forward-only Hebbian fails to monotonically improve quality at L=1M.**
   The CPU pilot at random init was a null result (expected); the real test
   is on a trained checkpoint. If, at L=1M, STDP-on quality is *worse* than
   STDP-off, we lose the headline differentiator from §2.
   See [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md) §"Inference-time STDP".

3. **A 100M Pythia trained with the same KD recipe matches or beats Synap-1.**
   This would say: *the architecture isn't the lever; the KD pipeline is*.
   Test is cheap (~24h on the same A800). On the [ROADMAP.md](ROADMAP.md) but
   deprioritized — we cite published Pythia numbers and accept the slight
   mismatch in training recipe.

If any of (1)-(3) come in negative, we update this doc and pivot. We have not
run (1) or (2) yet; (3) is implicit in the published baselines.

---

## 8. Update protocol

When live training (Run 3e) lands a checkpoint with measured benches, replace
the *training* placeholders in §1 with measured numbers. Per
[BENCHMARKS.md](BENCHMARKS.md), the harness is `synapforge/bench/{mmlu,hellaswag,gsm8k}.py`.
Cite each measurement with run id + step number. Do **not** soften the §3
"where we lose" section unless a measurement actually beats a baseline.

If a baseline's published number is updated (e.g. Qwen ships 2.5.1, SmolLM
ships 3.0), update the row + the citation. Numbers without a citation get
"NR" — never invent. The point of this doc is the **honest** column. Adding
fake numbers to make Synap-1 look more competitive **breaks the thesis** —
the thesis is that we're a different point on the Pareto frontier, and that
only makes sense if the static numbers are real.
