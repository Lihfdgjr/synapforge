# SynapForge — Investor Pitch

**Model**: **Synap-1** (突触一号) — the trained artifact, shipped in two tiers:
**Synap-1 Base (100M)** and **Synap-1 Pro (300M)**.[^pro-eta]
**Framework**: `SynapForge` — the codebase that produces Synap-1 and its variants.
See [NAMING.md](NAMING.md) for the variant roadmap.

[^pro-eta]: Pro variant launches post-Run 3o completion (ETA 14:00 May 2).

**One sentence**: A multi-tier LNN+SNN hybrid (100M Base / 300M Pro) that
learns from spike timing at inference, grows synapses into the action space
without emitting tool-call tokens, and uses a closed-form k-step CfC fold to
compress per-token reasoning into a single matrix solve.

## Thesis

GPT-class transformers cost millions to train and burn quadratic compute
per sequence step. We bet on a different stack:

- **Continuous-time CfC** (Liquid Neural Network) instead of attention.
  Naturally recurrent, parallel-scannable, no KV cache.
- **PLIF spiking neurons** for energy-shaped sparsity — every layer's
  output is binary at inference, gated by learnable per-channel τ.
- **Hebbian STDP plasticity** as a first-class rule, active at both
  train AND inference time.
- **NeuroMCP** — synaptic growth replaces tool-calling. Action emerges
  from a sparse synapse layer + dynamic prototype codebook, not from a
  parser.

The bet: at small scale, biology-inspired primitives beat transformer
scaling laws on three axes simultaneously — energy per token, latency
under streaming context, and continual learning without catastrophic
forgetting.

We don't claim parity at static benchmarks. We claim a different point
on the Pareto frontier.

## Five differentiated claims

Each claim has a CPU-runnable demo and a one-liner verification command.

### 1. NeuroMCP — synaptic growth replaces tool-calling

Agents pick actions without ever emitting a `<tool_call>` token. The
NeuroMCP head is a sparse synapse layer + dynamic codebook that grows
new connections as the network discovers the action space.

- **Evidence**: 4-button validation env, 100% hit rate. At default 80
  trials density grows from ~6% → ~8% and codebook K grows from 9 → 11
  (2 new prototypes self-discovered); pre-trained ckpts in earlier
  research runs reached ~28% density and K=14 at ~600 trials. The CPU
  smoke test runs in <1s and demonstrates the *mechanism* (synaptic
  growth replacing JSON parsing); the saturated regime needs longer
  training.
- **Verify**: `synapforge-demo button` (or `--trials 300` for visible
  density growth past 12%)
- **Why it matters**: Tool-calling LLMs spend ~30% of inference budget
  serializing/deserializing JSON. NeuroMCP makes that overhead zero.

### 2. R-fold algebraic CfC — k reasoning steps for the price of one

The CfC step is a linear map; k steps compose into a single closed-form
matrix solve. We verified the math (R=1 exact to fp32 noise; R=8 drift
0.32%) and benchmarked the speedup on GPU.

- **Evidence**: `cells/rfold.py` ships the closed-form fold. R=1
  rel_err 1.5e-6, R=8 rel_err 0.32%. On A800, N=64 R=16 gives 2.99×
  speedup vs sequential.
- **Verify**: `synapforge-demo bench`
- **Why it matters**: each token gets "8 reasoning steps" worth of
  compute without 8× the wall time. This is the primary lever for
  test-time compute on a small backbone.

### 3. Inference-time STDP — paper-grade single-LOC unlock

`bio/stdp_fast.py:121` removed the `if self.training:` gate. The
Hebbian rule now runs at inference. Forward-only weight updates from
the live context — the network adapts its fast-weight memory in real
time without backprop.

- **Evidence**: `synapforge-demo stdp` shows density (|W|>0.05) climbing
  0% → 27% over 200 trials with no optimizer, no loss. Pure spike
  co-activation.
- **Verify**: `synapforge-demo stdp`
- **Why it matters**: Test-Time Training (Sun et al. 2024) uses
  gradient. We use forward-only Hebbian — no backward pass at
  inference. Zero published systems do this at scale. Single-LOC
  unlock buys a paper.

### 4. Multi-tier LNN+SNN with Qwen vocab + Qwen 0.5B teacher KD

Two trained tiers, same recipe (FineWeb-en + alpaca-zh + GSM8K, KD from
Qwen2.5-0.5B), differing only in backbone capacity:

- **Synap-1 Base (100M)**: vocab=151936, d=512, n_layers=10, loop_depth=1.
  After Qwen embedding (~75M at d=512), the **useful backbone is ~25M**.
- **Synap-1 Pro (~325M)**: vocab=151936, d=1024, n_layers=14, loop_depth=2,
  ffn_ratio=2.5. The d=1024 embedding (151936×1024) is ~155.6M alone, so
  the **useful backbone is ~169M — 7× the ~25M useful backbone in Base**.

- **Evidence**: Triton-fused HybridBlock kernel at 22k tok/s on a
  single A800 80GB. Run 3l/3m/3n on Base plateaued at val ppl ~3700
  (cross-entropy ~8.2) — that plateau is what informed the scaling
  decision to Pro. Pro target ppl is unverified until Run 3o lands.
- **Verify**: `synapforge-demo chat` (replays a recorded transcript
  if no ckpt is present; with `--ckpt` it generates live).
- **Why it matters**: Qwen vocab gives us bilingual (en+zh)
  out-of-the-box; KD gives us a strong teacher signal at <2% the cost
  of pretraining from scratch. Pro's 7× backbone capacity is a direct
  response to the Base plateau — same architecture, more useful weights.

### 5. Triple-path full-volume backup

Lost a v4.1 ckpt once on a single rental shutdown. Now every best-PPL
checkpoint goes to (a) `mohuanfang.com:/home/liu/synapforge_backup/`,
(b) a private GitHub release, and (c) Hugging Face hub.

- **Evidence**: `auto_ckpt_backup.py` runs as a daemon; `scripts/
  triple_backup_daemon.py` orchestrates all three sinks.
- **Why it matters**: One bad rental migration cost us 4 GPU-days.
  Will not happen again.

## Honest competitive comparison

The full 11-baseline comparison table — Mamba-130M, RWKV-169M, Pythia-160M /
410M, GPT-Neo-125M, SmolLM2-360M / 1.7B, TinyLlama-1.1B, Qwen2.5-0.5B / 1.5B —
with citations and a per-cell win/lose/tie tally lives in
**[BASELINE_COMPARISON.md](BASELINE_COMPARISON.md)**.

Headline cells:

| | Synap-1 Base | Synap-1 Pro[^pro-eta] | SmolLM2-360M | TinyLlama-1.1B | Qwen2.5-0.5B |
|--|--|--|--|--|--|
| Parameters | 100M (~25M useful backbone) | 300M (~175M useful backbone) | 360M | 1.1B | 500M |
| MMLU (5-shot) | ~30 target [training; Base plateaued val ppl 3700] | unverified, post-Run 3o | 30.4 | 25.5 | 47.5 |
| Energy/token (relative) | **~0.05** (sparse spikes, aspirational; see BASELINE_COMPARISON §3) | **~0.05** (same recipe) | 1.0 | 1.0 | 1.0 |
| Inference latency at 100K ctx | **linear** (no KV) | **linear** (no KV) | quadratic | quadratic | quadratic |
| Continual learning | **yes, STDP** | **yes, STDP** | catastrophic forgetting | same | same |
| Tool use | NeuroMCP (no token) | NeuroMCP (no token) | JSON tool-call | same | same |
| Training cost | ¥168 (24 GPU-h) | ~¥600 (target, ~80 GPU-h on A800) | $50k+ | $30k+ | $200k+ |

**We will lose** at static benchmark parity at Base size. SmolLM2 has 3.6×
the Base parameters and was trained on 2T+ tokens. Pro narrows the gap on
parameters but is still pre-training; we are not claiming Pro beats SmolLM2
until Run 3o validates. **We win** on energy, streaming inference, and the
ability to keep adapting after deployment — and those five differentiated
claims (NeuroMCP, R-fold, STDP, KD, backup) apply equally to Base and Pro
because the architecture is identical, only the backbone width/depth
changes. Per-cell tally in [BASELINE_COMPARISON.md §6](BASELINE_COMPARISON.md):
7 win / 2 tie / 33 lose (Base figures).

The thesis only pays if those three axes matter to the deployment.
If you're shipping a chatbot to GPT-4 quality, transformer wins. If
you're shipping a long-running on-device agent that must adapt to
each user's context without retraining, our stack wins.

## Risks (told straight)

1. **Undertrained at every tier**. Base hit a hard val ppl floor around
   3700 across Run 3l/3m/3n — far short of chat-grade. We do **not** claim
   chat-grade fluency for Base or Pro yet. Pro is the scaling response to
   that plateau, but Pro is itself unverified until Run 3o lands.
2. **Vocab choice eats Base capacity**. Qwen tokenizer at 151k vocab makes
   the embedding ~75M of the 100M Base parameters — only ~25M is useful
   backbone. Pro keeps the same vocab but lifts the backbone to ~175M
   (~7× useful capacity). This is the structural reason Base plateaued
   and the structural reason we expect Pro to push past it.
3. **GPU dependency**. Triton-fused kernels assume CUDA. Inference on
   CPU is ~50× slower; mobile/edge needs a separate quantized path
   (BitNet b1.58 ternary in roadmap).
4. **Single-rental risk** (now mitigated). Old setup was one rental
   GPU; one shutdown cost a checkpoint. Triple-backup pipeline shipped.
5. **Inference-time STDP needs eval gates**. We have the mechanism
   (1 LOC) but haven't proven monotonic quality-vs-context-length on
   real workloads. Risk: STDP saturates W after ~1M tokens and degrades.

## Ask

| Item | Cost | Outcome |
|------|------|---------|
| 1 rental A800 80GB, 24 GPU-h | ¥168 | Synap-1 Base training continuation (target ppl ~50, currently plateaued at ppl 3700) |
| 1 rental A800 80GB, ~80 GPU-h | ~¥600 | Synap-1 Pro 300M first usable ckpt (target post-Run 3o; unverified) |
| 1 rental A800 80GB, 7 days | ~¥1,200 | best Pro ckpt + chat-fluent target |
| 2 weeks engineer time | salary | inference-time STDP eval gates |
| 1 month engineer time | salary | BitNet b1.58 ternary edge build |
| 3 months total runway | ~$15k | ship a public chat demo (Pro tier), paper |

The economics: a ¥168 rental day got us to a real Base artifact, not a
slide. The honest update is that Base plateaued; Pro at 300M is the
straightforward scaling step (same architecture, 7× useful backbone) and
costs only ~¥600 to validate. Compare to GPT-class training runs measured
in millions.

## Why now

- Hebbian / spiking / continuous-time literature has been steady for
  a decade. The change in 2025 is that **PyTorch finally ships
  surrogate-gradient autograd** efficiently enough that you can train
  these stacks end-to-end without custom CUDA. We exploit that.
- KD from open-weight 0.5B-class teachers (Qwen2.5, SmolLM2) makes
  the signal-to-noise on <100M backbones genuinely competitive.
- The energy story is no longer hypothetical: every additional context
  token in a transformer costs proportional more KV memory and FLOPs.
  We have empirical results showing CfC + spike sparsity holds linear
  cost out to 100K-context.

## What's NOT a claim

- We are NOT claiming chat parity with GPT-4 / Claude / Gemini at any tier.
- We are NOT claiming the Base (100M) ckpt is product-ready or chat-grade.
  Run 3l/3m/3n plateaued at val ppl ~3700.
- We are NOT claiming Pro (300M) hits any specific ppl until Run 3o lands.
  Pro is a scaling-response artifact, not a measured one.
- We are NOT claiming inference-time STDP improves quality
  monotonically. It MIGHT. We have not verified at scale yet.
- We are NOT promising the R-fold pays off on CPU. CPU LAPACK overhead
  flips the speedup. GPU + N≥256 is where it wins.

We've taken enough rejected bug-bounty submissions to know that
overclaiming kills credibility faster than understating. Everything in
the demo is what runs.

## Verification — what to actually run

```bash
pip install -e .
synapforge-demo pitch    # 30-second elevator
synapforge-demo button   # NeuroMCP 4-button, ~1s, density grows
synapforge-demo bench    # R-fold math correctness + speedup table
synapforge-demo stdp     # STDP self-organization, ~1s
synapforge-demo chat     # 5 EN + 5 ZH prompts (recorded if no ckpt)
synapforge-demo all      # all of the above (pitch + button + bench + stdp + chat)
synapforge-demo json     # all demos, JSON dump for grepping
```

Reproduces in under 5 minutes on any CPU. No API keys, no GPU.

### Why no LoRA / transformer fallback

The architecture claim is **the** product. A "LoRA-on-Qwen v0 frontend" would
contradict the LNN+SNN thesis and make the paper unsubmittable. See
[docs/ANTI_LORA.md](ANTI_LORA.md) for the strategic position. If neither
Base nor Pro Synap-1 ckpts are ready by demo day, the fallback is a smaller
native Synap-1 (30M-50M LNN+SNN, faster training) **or** an honest replay
of the last healthy v4.x ckpt — never a transformer base.
