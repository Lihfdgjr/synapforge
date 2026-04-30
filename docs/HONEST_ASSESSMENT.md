# Honest Assessment

What works, what doesn't, what's untested. 2026-04-30.

## Works

**Trained and verified on real data:**

- v4.0 base 375M LNN+SNN trained from scratch in 7h on A100×2, ppl 84 at step 36k
- v4.1 NeuroMCP wire-in: best ppl 44.2 at step 46350, NeuroMCP K=9, synapse density grew 5%→40% as expected
- Qwen 151936 vocab fixed Chinese byte-fragmentation (was a real blocker on GPT-2 50257)
- Response-only loss masking (`ignore_index=-100`) breakthrough at v2.6 took ppl 451 → 187
- Autonomous learning daemon: cycle 1 ran end-to-end, +6 admit / -8 reject through 7-gate
- HNSW skill index: smoke-tested with hnswlib backend, K → 100k+ with sub-ms p99
- DPO pair generator (red/blue self-play): 5 pairs generated successfully via stub judge
- Provenance tracker with per-source trust EMA: 100M-entry blocklist via Bloom filter

## Doesn't work yet (or unproven)

**v4.2 trainer has 3 bugs (patches written but not deployed):**
- `Path("Qwen/...")` doesn't catch HF repo ids → KD silently disabled (kd=0.000)
- PerDomainNeuroMCP runs all 4 domain heads every step → 17× slowdown (1100 vs 19500 tok/s)
- response-only mask hides 95%+ of tokens → ppl 1.4 false reading; need parallel
  unmasked CE for monitoring

**Multimodal pipeline exists but only text trained:**
- All 9 modality byte-patch encoders compile and forward, none have seen real
  image/audio/video during training
- Anti-fakery test (zero out image embeds) not run

**NeuroMCP at scale untested:**
- 4-button env 100% success, but anything more complex unverified
- L1/L2 compositional codebook scaffold written, no Hebbian co-firing minting tested
- Real OS / web actuator not wired (still synthetic env only)

**3D understanding not started:**
- Plan written (DUSt3R + EGNN, 140 GPU-h), no code yet
- ScanQA / 3DSRBench numbers all hypothetical

**Anthropic safety stack written, not run:**
- All 4 modules (red_team_corpus, red_blue, constitutional, dpo, judge) compile
- Stub judge tested. API judge requires OpenAI-compatible key (DeepSeek tier).
- No actual SFT-refusal or CAI training run yet

**Continual learning gates not battle-tested:**
- TRAK influence gate (G6) is the most expensive one — not yet run
- Per-source 7d cap = 125 (50% of Anthropic 250 poison threshold) is aspirational;
  no actual poison injection test
- Shadow LoRA merge every 4h: code path exists, never triggered in real run

**R-fold algebraic CfC closed-form:**
- 167× speedup is from LiquidS4 paper, not yet measured on our impl
- Agent-investigated 3 approaches, none tried at training scale yet

**Research-grade "neuromorphic" claims unverified:**
- Energy advantage from PLIF spike rates: never measured on actual neuromorphic h/w
- Spike rates 5-30% reported but never compared to dense baseline FLOPs
- "Biology-inspired" is rhetoric; the math is closer to vanilla RNN with extra plumbing

## Things I claim but you should verify

These aren't lies but each has a "but actually..." caveat:

| Claim | Caveat |
|-------|--------|
| 375M params | True, includes embedding 156M + backbone 220M |
| ppl 44.2 | Best **single batch**, not averaged. Steady-state realistic: 60-90 |
| 7h training | Yes, 60k steps × bs=8 × seq=1024 = 500M tokens. Anyone could replicate. |
| 9 modalities | Encoders exist, only text trained. Don't believe demo until evals run. |
| Persistent skills "user gets back next session" | Code path exists, never tested across actual session restart |
| 7-gate poison defense | 4 of 7 gates implemented; 3 (TRAK / shadow merge / per-source 7d) are scaffold |
| Anthropic safety | Pipeline written; no DPO run completed yet |
| 17M LNN beats 17M GPT2 | True on certain seq lengths (long-range). On WikiText103 GPT-2 wins. |

## What you should NOT use this for

- **Production**: this is a research framework, not a deployed product. Use Claude / GPT-4 / Gemini for real work.
- **Safety-critical**: no formal guarantees. Output safety stack is written, not battle-tested.
- **Benchmarking against modern small LMs**: SmolLM2-360M reaches WikiText ppl ~20 with 2T tokens; we're at ppl 50 with 500M tokens. Ours has different optimization target (continual learning + plasticity), but at static eval we lose.

## What it's good for

- Research playground for non-transformer LM architectures at small scale
- Studying continual learning + persistent skill memory
- Studying neural action routing as alternative to JSON tool calling
- Cheap baseline for "is X really transformer-specific?" questions

## Honest recommendation

If you want to use SynapForge:
1. Wait for v4.2 to finish training and post real numbers (1-2 days)
2. Start with the 4-button NeuroMCP demo (works), not multimodal (untested)
3. Don't skip the 7-gate ingest pipeline if you turn on continual learning
4. Use Claude Memory pattern for chat (Track B retrieval-only) — don't update weights
   from user messages unless you've tested poison defenses

If you're considering using ideas from this work in your own paper:
- NeuroMCP composition (L1/L2) is the strongest novel angle
- Two-track continual learning has Anthropic Memory as industry precedent
- DUSt3R-as-data-engine for 3D LLM is publishable as a methods note
- Don't quote our energy claims — they're aspirational
