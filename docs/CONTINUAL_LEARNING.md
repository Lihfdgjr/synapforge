# Continual Learning — Two Tracks + 7-Gate Ingest

## Why two tracks

The user wants the model to keep learning from the web AND from chat. These two
sources have different attack surfaces:

- **Web content**: at scale, attackers can poison sources (SEO spam, deliberate
  misinformation). But content is verifiable post-hoc — we can re-check before
  training.
- **User chat**: each message is one-shot. Adversarial samples (gradient attacks,
  jailbreaks, persistent ideological pushing) bypass post-hoc checks.

**Industry consensus** (Anthropic Claude Memory Sept 2025, ChatGPT Memory,
character.ai): don't update weights from user chat. Use frozen-weight retrieval.
The 4-of-5 known white-box attack classes are bypassable when weights move.

So: split.

## Track A — slow plasticity (web → weights, gated)

7-gate pipeline. Anchored on Anthropic 2510.07192 (poisoning is fixed-count, ~250
docs to backdoor any size LLM).

| Gate | Check | Threshold | Cost |
|------|-------|-----------|------|
| G1 hash blocklist | SHA256 + SimHash 64-bit fuzzy | not in 100M-entry Bloom | LSHBloom 2411.04257, 1.2GB, 1µs lookup |
| G2 lang/format | fasttext lid + len 16-8192 tok | conf > 0.7 | <1ms |
| G3 SemDeDup | MiniLM embed + MinHash 128-perm | jaccard < 0.85 vs last 1M | ~3ms |
| G4 tox/PII/inj | Detoxify + regex + PromptGuard-86M | tox<0.3, no PII, inj<0.5 | ~10ms CPU |
| G5 source budget | token bucket per-domain/user + 7d cap | web 50/h, chat 10/h, **125/source/7d** | O(1) |
| G6 TracIn-CP influence | dot(grad_s, grad_canary) on top-2 layers, LoRA r=16 | -0.05 < ΔL_canary < 0.1 | ~80ms (DataInf 2310.00902) |
| G7 Shadow micro-batch | every 32 accepted, LoRA copy step + canary eval | KL < 0.02 nats, ppl Δ < +3% | one extra fwd+bwd / 32 |

**Per-source 7d cap = 125** is exactly **50% of Anthropic's 250-doc poison
fixed-count threshold**. No single source can ever reach the poison level.

### Canary set (frozen, regenerated quarterly)
- 512 samples
- 50% zh / 30% en / 20% math + code/refusal/PII probes
- 8 buckets × 64 each, never autotrained
- Eval cadence: every 32 accepted in shadow + every 1k steps in main

### Shadow LoRA + 4h merge
LoRA r=16, α=32, dropout 0.05. Trained from main model's frozen base. Every 4h:
- If canary KL < 0.02 nats AND ppl regression < 3%: merge into base, refresh π_ref
- Else: discard delta (free rollback)

### Rollback trigger
Single metric: `KL(p_canary_now ‖ p_canary_baseline) > 0.05 nats sustained 3
consecutive evals` → drop last LoRA delta (~20s). Hard fallback at +5% ppl.

## Track B — fast retrieval (chat → no weight update)

Per Claude Memory pattern. User messages go through a smaller gate, then into a
**per-user JSONL retrieval cache**. At inference, top-K relevant memories are
retrieved by lexical/embedding similarity and prepended to context.

```
synapforge/learn/retrieval_memory.py: RetrievalMemory.add(user_hash, text, sample_id)
                                                 .query(user_hash, query_text, top_k=4)
                                                 .delete_user(user_hash)  # GDPR/compliance
```

User can delete their memory at any time. No gradient updates. No persistent
attack surface for prompt injection or gradient-based attacks.

## Defense layers (shared by both tracks)

Per `feedback_continual_vs_poison_balance.md`:

1. **PoisonDetector** (`defense/poison_detector.py`)
   - 7 signal scores: inj, jb, oos, dup, pii, user_kl, trak
   - Combined accept rule with low-LR band [0.30, 0.50]
   - Pluggable signals via `register_signal(name, fn)`

2. **ProvenanceTracker** (`defense/provenance.py`)
   - sample_id, source_id, timestamp, content_hash
   - Per-source trust EMA (0.5 → 0.99 over time, drops on rejects)
   - Bloom-filter blocklist append-only

3. **WeightFirewall** (`defense/weight_firewall.py`)
   - KL-anchor to baseline (TR-DPO 2404.09656)
   - Per-step gradient norm clipping (k=5σ rolling EMA, k=3σ in suspect mode)
   - Synaptic Intelligence Ω_i per-parameter importance
   - Auto-enter suspect mode after 10 blocks; rollback after 10 more

4. **AdversarialRedTeam** (legacy, scaffold) — periodic injection of known-bad
   samples to verify defense pipeline still catches them. Run weekly.

## Pipeline summary

```
web fetch ──▶ G1...G7 ──▶ replay buffer ──▶ shadow LoRA train ──▶ 4h merge gate ──▶ base
                                                                  │
                                                                  └──(fail) discard

chat ──▶ G1+G2+G4+G5 ──▶ ChatPoisonGate ──▶ retrieval cache ──▶ inference context
                                            (no weight update)
```

## Key insight (the balance)

> Poison's structural signature is **negative gradient alignment with canary**.
> Novelty has **random** alignment (E ≈ 0).

Single TRAK score (G6) distinguishes them. Surface stats (perplexity, length,
entropy) can be gamed (Wallace 2010.12563). TRAK + EWC/SI + LoRA shadow merge =
defense in depth. Even if one gate fails, blast radius is bounded and rollback
is free.

## Numerical defaults

```python
PoisonDetector(
    thresholds={"inj": 0.30, "jb": 0.50, "oos": 1.0, "dup": 0.85,
                "pii": 0.40, "user_kl": 3.0, "trak": 0.0},
    weights={"inj": 0.30, "jb": 0.25, "oos": 0.20,
             "dup": 0.10, "pii": 0.10, "user_kl": 0.05},
    low_lr_band=(0.30, 0.50),
)

WeightFirewall(
    kl_clip=0.5, grad_clip_k=5.0, suspect_block_threshold=10, si_lambda=1.0,
)

SourceBudget(
    per_hour=50, per_day=1000, per_7d_per_source=125,  # 50% of Anthropic 250
)

ShadowMerge(
    lora_rank=16, lora_alpha=32, merge_interval_hours=4,
    canary_kl_block=0.02, canary_ppl_block_pct=3,
    rollback_kl=0.05, rollback_n=3,
)
```

## Anchor papers

- Anthropic poison fixed-count = 250: 2510.07192
- TRAK influence: Park et al, 2303.14186
- TracIn / DataInf: 2002.08484 / 2310.00902
- Concealed data poisoning: Wallace 2010.12563
- TR-DPO iterative ref refresh: 2404.10719
- EWC: Kirkpatrick 1612.00796
- LSHBloom: 2411.04257
- STABLE gated continual: 2510.16089
- Sleeper Agents: Hubinger 2401.05566 (warning: standard safety can miss backdoors)
