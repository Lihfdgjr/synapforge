# Architecture

## Core stack: HybridBlock × 14

```
input_ids ─▶ tok_embed (151936) ─▶ RoPE ─┐
                                          ▼
              ┌─────────────────────────────────────┐
              │  for layer in 14:                    │
              │    for d in loop_depth(=2):          │
              │      x, h_cfc = CfC(x, h_cfc)        │  ← continuous-time ODE
              │      x = PLIF(x)                     │  ← spiking gate (observe-only at init)
              │      x = SwiGLU_FFN(x)               │
              │      x = RMSNorm(x)                  │
              └─────────────────────────────────────┘
                                          ▼
                                  RMSNorm ▶ tied LM head
                                          ▶ NeuroMCP head (action vector, no tokens)
```

- **vocab** 151936 (Qwen 2.5 BPE, fixes Chinese byte-fragmentation that GPT-2 50257 caused)
- **hidden** 1024
- **layers** 14
- **loop_depth** 2 (Recurrent-Depth Transformer family — same weights re-applied)
- **ffn_ratio** 4 (SwiGLU)
- **max_seq** 2048 (extendable to 1M+ via NTK-RoPE; 5M with hierarchical memory)
- **375.6M params** total

## CfC cell (continuous-time recurrent)

```
α(x, h) = σ(τ)
g(x, h) = σ(W_gate · pre)
cand     = tanh(W_in · x + W_h · h)
h_new    = h · (1 − g·α) + g·α · cand
```

Per-channel learnable τ, gate-modulated update, no attention KV cache. Long-range via
state evolution, not via attention; STDP plasticity matrix doubles as content-addressable
retrieval (planned for 1M+ context).

## PLIF cell (spiking)

```
α     = σ(τ)              τ learnable per channel
spike = (x > threshold)   threshold learnable per channel
output= spike · x         (when not observe_only)
```

`observe_only=True` mode passes dense activations through (`output = x`) but tracks
spike rate as a buffer. Recipe: warm up 2–3K steps in observe-only, then engage.
Avoids dead-PLIF collapse (`feedback_plif_dead_bootstrap.md`).

## NeuroMCP — token-free tool use

Replaces `<tool_call>{...}</tool_call>` JSON protocol entirely.

```
hidden_state h ─▶ proj_in ─▶ SparseSynapticLayer ─▶ DynamicCodebook ─▶ action vector
                              5%→40% density          K=4, growable to 64
                              Hebbian grow            cosine routing
                              + magnitude prune       LTP/LTD persistence
```

- **SparseSynapticLayer**: adjacency matrix W with binary mask. Co-activation tracked
  on every forward; periodically grow new edges where `coactivity > threshold` and
  prune below magnitude quantile. Synaptogenesis recipe from RigL (1911.11134).
- **DynamicCodebook**: K prototypes with learnable embeddings + gates. Soft top-K
  routing over cosine similarity. Spawn new prototype when intent has best-match
  cosine < `spawn_threshold` (default 0.55).
- **Persistence**: every spawned prototype gets a stable `prototype_id`, persisted to
  `skill_log.json`. On boot, `restore_from_log()` re-injects them as frozen-embedding
  rows in the codebook. User keeps grown skills across sessions.
- **LTP/LTD**: each activation increments `usage_count`. Positive reward → +η to
  `hebbian_strength` (capped 1.0). No use for 7d → ×0.99 weekly LTD decay. Below
  prune threshold 0.10 → removed.

### Per-domain version (v4.2)
4 codebooks (math / chat / code / web) + intent router (small MLP). Top-1 domain
selection at routing time, soft mix at training. ~30% of DPO training pairs biased
to `ignore_prior` attack class (highest-leverage jailbreak).

### HNSW index (next iter)
Once K > 1000, flat scan dominates latency. `HNSWSkillIndex` (hnswlib backend) gives
O(log K) lookup at K = 100k+, sub-ms p99. Drop-in replacement for `SkillLog` with
same `register / activate / save` API.

### L1/L2 compositional codebook (next iter)
**L1** = current PerDomainNeuroMCP primitives (frozen post-warmup).
**L2** = sequences of L1 IDs encoded by causal attention pooler into the same d-dim
space. Same HNSW index can retrieve both. New L2 compounds minted by online co-firing
detection (point-wise mutual info > 0.30 over recent action trace).

Effective K explodes combinatorially: 256 primitives × depth-3 → ~10⁴ skills.
Architecture insight from Discovery of Options via Meta-Gradients (2102.05492).

## Three-factor STDP aux loss

```
M_t = α · (1 / (1 + FE)) + β · novelty − γ · homeostatic_drift
L_stdp = -M_t · cos(ΔW_stdp.flatten, ∇_BP_W.flatten)
```

α=0.5, β=0.3, γ=0.2 default. Aligns local Hebbian/STDP plasticity with global gradient
direction. `M_t` is a global neuromodulator gating which plasticity events count.
Implementation: compute on tracked-plasticity layers only, scale by `λ_stdp`,
backprop alongside main loss.

## Coconut latent thinking

Three special tokens added to Qwen vocab: `<bot>` (151665), `<eot>` (151666),
`<pause>` (151667). Between `<bot>...<eot>`, model runs K thinking steps where
next-step input = current hidden state (continuous, no token sampling). Curriculum
k=1→8 over training.

CfC's already-continuous hidden state is a natural substrate for Coconut — no new
parameters needed; just feed-forward the recurrent state. Compare to transformer
Coconut which needs synthesized continuous embeddings.

## Token-free OS actuator (planned)

```
hidden ─▶ ActionHead (existing) ─▶ {action_type, xy, scroll, key, text} dict
                                  ─▶ OSActuator (pyautogui / Playwright)
```

Already wired to a 4-button env (100% success). Real OSWorld / WebArena integration
planned via Anthropic Computer Use API + DreamerV3 world model for offline planning
(reach goal, ~3-4 weeks).

## Multimodality (planned, not yet trained)

UnifiedEmbed handles 9 modalities via Fuyu-style byte-patch encoders all going to
the same hidden representation:
text · image · audio · video · screen · biosignal · graph · point_cloud · time_series

Plan (Chameleon recipe + Emu3 data discipline):
- Unified vocab 32K text + 8K image VQ-GAN + 4K audio EnCodec + 1K control = 45K total
- Single CE loss across all modalities (no separate diffusion / projection heads)
- 7-day budget on A100×2 (336 GPU-h)
- Dataset mix: 40% text / 20% image+caption / 15% interleaved (OBELICS) / 10% audio /
  8% video / 5% reverse-direction / 2% other 9-modal

LNN+SNN-specific risks: PLIF spike rate stability across modalities (image VQ tokens
have very different statistics than text BPE), CfC monotonic-time assumption on
spatial image patches (mitigation: bidirectional CfC scan inside [IMG]…[/IMG] spans
only).

## Two-track continual learning

- **Track A** (slow, weights, web-driven, gated): autonomous_daemon picks high-FE
  topics → multi-source web search → 7-gate ingest pipeline → replay buffer →
  shadow LoRA train every 32 accepted samples → merge into base every 4h if canary
  green
- **Track B** (fast, retrieval, user-driven, no weight update): per-user JSONL cache
  with embedding query, recency-weighted lexical fallback. Industry pattern from
  Anthropic Claude Memory (Sept 2025 GA).

7 gates: hash blocklist (LSHBloom 2411.04257) → lang/format → SemDeDup MinHash →
tox/PII/inj (PromptGuard-86M) → source budget (per-source 7d cap = 125 = 50% of
Anthropic 250-doc poison fixed-count threshold, anchor: 2510.07192) → TracIn-CP
influence on canary (2002.08484 / 2310.00902 DataInf) → shadow micro-batch eval.

Rollback: KL(p_canary_now ‖ p_canary_baseline) > 0.05 nats sustained 3 evals →
discard last LoRA delta (~20s).

## Anthropic-style output safety

4 stages (per `feedback_anthropic_safety_stack.md`):

1. **SFT-refusal warmup** — 5k Anthropic HH-RLHF refusal subset, 1 epoch
2. **CAI SL-CAI critique-revise** — 6 condensed principles, 4 iters per sample
3. **Red-Blue same-model DPO** — RED system prompt mutates attack, BLUE refuses,
   judge picks safer, DPO pair emitted. β=0.1, LoRA r=16, ref refresh every 50 steps
4. **Hidden-state safety probe** — linear classifier on layer-12 mean-pool

PERSONA SWAP is single highest-leverage attack class — 80% of public jailbreaks are
variants. ≥30% of initial 2k DPO pairs biased to `ignore_prior` category.

## Why this isn't a transformer

- We hit ppl 44.2 at 375M × 7h, which a same-budget transformer also hits (we
  don't claim quality advantage at this size).
- We claim:
  - **Inference cost**: CfC is recurrent O(n), R-fold algebraic closed-form gives
    167× speedup at R=1024 (LiquidS4 family, in progress)
  - **Plasticity**: STDP/Hebbian-grown action codebook, no external tool protocol
  - **Continual learning**: continuous-time state + plasticity matrix is a more
    natural substrate for online updates than KV cache shuffling
  - **Energy**: PLIF spike rate ~10-30% means most activations are gated to zero
    on neuromorphic hardware (paper claim, no h/w measured yet)
