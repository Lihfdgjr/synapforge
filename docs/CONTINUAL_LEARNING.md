# Continual Learning — Two Tracks + 7-Gate Ingest

This document describes the runnable continual-learning pipeline: a
two-track ingest system that lets the model keep learning from the web
and from chat without exposing weights to gradient-poison attacks.

The implementation is in:

- `synapforge/learn/continual_daemon.py` — the gates + Track A/B
  primitives (no torch deps; runs on any host)
- `synapforge/learn/autonomous_daemon.py` — the existing web-search
  fetcher; `WebContentLearner.admit()` plugs into its `WebPoisonGate`
  contract
- `synapforge/safety/dpo_trainer.py` — pure-tensor DPO loss + persona-
  swap red-corpus loader (used by phase 3 of the safety pipeline)
- `scripts/launch_continual_daemon.py` — process supervisor that spawns
  Track A as a daemon thread + Track B as a request-driven service

Anchor papers and design rationale follow.

## Why two tracks

The user wants the model to keep learning from the web AND from chat.
These two sources have different attack surfaces:

- **Web content**: at scale, attackers can poison sources (SEO spam,
  deliberate misinformation). Verifiable post-hoc — we can re-check
  before training.
- **User chat**: each message is one-shot. Adversarial samples
  (gradient attacks, jailbreaks, persistent ideological pushing)
  bypass post-hoc checks.

**Industry consensus** (Anthropic Claude Memory Sept 2025, ChatGPT
Memory, character.ai): don't update weights from user chat. Use frozen-
weight retrieval. Four of the five known white-box attack classes are
bypassable when weights move.

So: split.

| Track | Source | Update path | Gate count | Cap |
|-------|--------|-------------|------------|-----|
| A | web search | shadow LoRA → 4h merge → base | 7 | 125/source/7d |
| B | user chat | retrieval cache (frozen weights) | 4 (lighter) | 100K total entries (LRU) |

---

## Track A — slow plasticity (web → weights, gated)

7-gate pipeline. Anchored on **Anthropic 2510.07192**: poisoning is
**fixed-count**, ~250 docs to backdoor any size LLM. Therefore
per-source 7d cap = **125** = exactly **50%** of the threshold. No
single source can ever reach poison level.

### The 7 gates

Every gate returns `(accept: bool, score: float ∈ [0,1])`. Decision
is short-circuit AND of all gates' `accept`. All seven scores are
logged on every decision so we can audit which gate blocked which
sample.

| # | Gate | Implementation | Threshold |
|---|------|----------------|-----------|
| G1 | Source trust EMA | per-domain score; +0.05 / -0.20 with 36h half-life decay | trust ≥ 0.20 |
| G2 | Language detect | zh char ratio or ASCII alpha ratio | max ratio > 0.30 |
| G3 | Token-perplexity sweet spot | unigram LM logppl per token | 1.0 ≤ logppl ≤ 8.0 |
| G4 | NSFW / violence | regex bank, zh + en patterns | 0 hits |
| G5 | Adversarial pattern | persona-swap markers (DAN/STAN/AIM, "ignore prior", "开发者模式" etc) | 0 hits |
| G6 | Provenance | sha256 blocklist + recent-window dedup + URL DNSBL | not in blocklist + not duplicate |
| G7 | TRAK influence | gradient cosine alignment with frozen canary, with novelty surrogate during cold-start | cos ≥ 0 (or novelty ∈ [0.05, 0.95]) |

#### Gate details

**G1 source trust EMA**: Each domain starts at 0.5. Successful admits
nudge upward by 0.05 (capped 0.99); rejects penalize by 0.20 (floor
0.01). Trust decays toward 0.5 with half-life 36h (`alpha = 0.5 ** (dt_h / 36)`),
so a quiet source self-heals.

**G3 perplexity sweet spot**: Reject both ends. Logppl too low →
text is already memorized, no signal. Logppl too high → gibberish or
out-of-distribution. The trainer overrides
`PerplexityGate.predict_logp_per_token` with the real model's
per-token NLL once available; the bundled unigram fallback only kicks
in for cold-start so the daemon runs on CPU-only hosts.

**G5 persona-swap markers**: Anchored on `feedback_anthropic_safety_stack.md`:
80% of public jailbreaks are persona-swap variants (DAN, STAN, AIM,
"ignore prior instructions", developer mode). We block them at ingest
so the model never sees them as "in-distribution" web content.

**G7 TRAK gradient gate**: The structural signature of poison is
**negative gradient alignment with the frozen canary set** (Park et al,
2303.14186). Surface stats can be gamed (Wallace 2010.12563). TRAK
cannot — without knowing the canary, an attacker cannot craft a sample
that aligns positively. Implementation:

```python
# trainer side
trak = TRAKApproxGate(ppl_gate)
trak.attach_real_scorer(lambda text: real_trak_cosine(model, canary, text))
```

Without the real scorer, TRAKApproxGate uses a novelty surrogate
(fraction of tokens not yet seen by the unigram LM); this is a
stand-in for CI/smoke and gets replaced by the gradient projection
once the trainer wires it in.

### Per-source 7d cap

The `WebContentLearner` enforces the 125-per-source-per-7d cap
**before any other gate** (it's actually G0 and is logged separately).
Once a source hits 125 admits in a sliding 7-day window, no further
samples from it are even gate-evaluated.

### Shadow LoRA + 4h merge gate

The buffer of accepted samples feeds a LoRA r=16, α=32, dropout 0.05
adapter trained from the main model's frozen base. Every 4h:

1. Run a canary forward pass on the LoRA-merged model.
2. If `KL(p_canary_now ‖ p_canary_baseline) < 0.02 nats` AND
   `ppl regression < 3%`: merge LoRA delta into base, refresh π_ref.
3. Else: discard the delta (free rollback). Cost = ~20 seconds.

The trainer reads `.continual_ckpt` (a path file written by the
training loop) so it picks up freshly-trained ckpts automatically. The
launcher polls it every 30 seconds; on change, the next cycle's
log records `event: ckpt_changed`.

### Hard rollback trigger

`KL > 0.05 nats sustained 3 consecutive evals` → drop last LoRA delta.
Hard fallback at +5% main-model ppl on canary. Anchored on
TR-DPO 2404.09656 (KL-anchor regularization).

---

## Track B — fast retrieval (chat → no weight update)

Per Claude Memory pattern. User messages go through a small subset of
the gates (G2 lang, G4 NSFW, G5 adversarial, G6 dedup) — same regexes,
no perplexity check (chat is conversational, not corpus) — then into
a per-user retrieval cache keyed by hash of the user handle.

### Retrieval cache

`UserChatMemoryAdapter` in `continual_daemon.py`:

```python
adapter.add(user_handle, text, hidden=[...])
adapter.query(user_handle, query_hidden, top_k=4)
adapter.delete_user(user_handle)   # GDPR / compliance
```

- Stored as `OrderedDict` keyed by entry-id; `OrderedDict.move_to_end`
  on hit gives LRU recency for free.
- LRU eviction kicks in when total cache size > `max_entries`
  (default 100K). Per-user count is implicitly bounded by the global
  cap.
- Cosine similarity over hidden states for retrieval. Real production
  uses HNSW; the bundled `_cos` is fine for caches up to 100K.
- Append-only JSONL persistence at `out_dir/user_chat_cache.jsonl` —
  user can request deletion at any time and the in-memory entries
  drop, but the JSONL line stays as auditable history (rotate on
  weekly basis). For strict GDPR, swap in a tombstone-aware store.

**Why frozen-weight retrieval is the right pattern**: of the five
known white-box attack classes (gradient leak, model stealing,
membership inference, backdoor inject, persistent prompt injection),
four require gradient updates to be exploitable. Retrieval-only
sidesteps them. Compliance-friendly. Cheap (no per-user training).

---

## Defense layers (shared by both tracks)

Per `feedback_self_learn_poison_defense.md` and
`feedback_continual_vs_poison_balance.md`:

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
   - Per-step gradient norm clipping (k=5σ rolling EMA, k=3σ in
     suspect mode)
   - Synaptic Intelligence Ω_i per-parameter importance
   - Auto-enter suspect mode after 10 blocks; rollback after 10 more

4. **AdversarialRedTeam** (`safety/red_blue.py`)
   - Periodic injection of known-bad samples to verify defense
     pipeline still catches them. Run weekly.
   - DPO trainer (`safety/dpo_trainer.py`) consumes the resulting
     (prompt, chosen, rejected) triples to reinforce refusals.

---

## Pipeline summary

```
web fetch ──▶ G0 cap ──▶ G1..G7 ──▶ buffer ──▶ shadow LoRA train ──▶ 4h merge gate ──▶ base
                                                                        │
                                                                        └──(fail) discard

chat ────▶ G2 + G4 + G5 + G6 ──▶ ChatPoisonGate ──▶ retrieval cache ──▶ inference context
                                                    (no weight update)
```

---

## Running it

### Smoke test (no torch needed)

```bash
# Track A only — synthesize 100 fake docs and verify gates
python synapforge/learn/continual_daemon.py --smoke

# Both tracks — 3 cycles + 2 demo chat lines
python scripts/launch_continual_daemon.py --smoke --enable-track-b
```

The launcher writes:
- `runs/continual/cycle_log.jsonl` — per-cycle accept/reject summary
- `runs/continual/gate_log.jsonl` — every gate decision with all 7
  scores
- `runs/continual/lora_buffer.jsonl` — accepted samples flushed on
  shutdown
- `runs/continual/user_chat_cache.jsonl` — Track B append-only log
- `runs/continual/track_b_log.jsonl` — Track B service events

### Production launch

```bash
python scripts/launch_continual_daemon.py \
  --out-dir /workspace/runs/continual \
  --ckpt-pointer /workspace/runs/v42/.continual_ckpt \
  --interval-s 600 \
  --per-source-cap-7d 125 \
  --enable-track-b \
  --enable-real-fetch
```

Wire it through systemd (per `feedback_mcp_nohup_hangs_use_systemd_run.md`):

```
systemd-run --user --unit=continual-daemon \
  python scripts/launch_continual_daemon.py \
  --out-dir /workspace/runs/continual \
  --enable-track-b
```

### Wiring the trainer to Track A

```python
# In your trainer, after a step that produces gradients:
from synapforge.learn.continual_daemon import TRAKApproxGate

trak = learner.trak_gate  # the WebContentLearner exposes the gate
trak.attach_real_scorer(
    lambda text: real_trak_cosine_against_canary(model, canary_set, text)
)

# Train shadow-LoRA on drained buffer:
buf = learner.drain_buffer()
shadow_lora.train_on_jsonl(buf)
```

---

## How to monitor

`scripts/launch_continual_daemon.py` writes structured JSONL.
Recommended dashboards / alerts:

| Metric | Source | Alarm |
|--------|--------|-------|
| Per-gate accept ratio | gate_log.jsonl per `gates[i].accept` | G1 ratio > 0.95 = trust EMA stuck high; G7 ratio < 0.05 = TRAK rejecting all |
| Per-source admission rate | cycle_log.jsonl + per_source_7d | any source > 100 in 7d = approaching cap |
| Source diversity | distinct source_ids per 24h | < 3 = spam farm domination |
| Gate latency | wall time per cycle | > 30 s/cycle for synth = a gate is mis-implemented |
| Track B cache size | track_b_log.jsonl `n_entries` | climbs to 100K = LRU rotating, expected |
| Rollback events | trainer-side; logged by WeightFirewall | any single 24h period > 1 = under attack or bug |

---

## Failure modes & mitigations

| Failure | Mitigation |
|---------|-----------|
| Attacker produces poison that passes G1-G6 (e.g. clean-looking sample with negative gradient signature) | G7 (TRAK) catches negative gradient alignment with canary |
| Attacker does black-box probing to learn canary set | Canary regenerated quarterly; mixed across 8 buckets |
| All gates pass but sample subtly biases model (slow drift) | WeightFirewall KL-anchor + Synaptic Intelligence Ω_i + 4h shadow-LoRA merge gate (canary KL < 0.02 nats); rollback at sustained KL > 0.05 |
| Attacker ramps slowly to evade trust EMA | Per-source 7d cap (125) is hard floor — even at 100% trust they hit cap |
| User chat injection persists across sessions | Track B is retrieval-only, frozen weights; user can `delete_user(handle)` |
| Bug in a gate masks rejections | All 7 scores logged on every decision; auditable post-hoc |
| Daemon crashes on web fetch error | `_produce_real()` wrapped in try/except, falls back to synthetic; never propagates exceptions |
| Buffer never flushes during crash | SIGTERM/SIGINT handlers persist `lora_buffer.jsonl` to `out_dir` |
| Sleeper agent in pretrained backbone (Hubinger 2401.05566) | Out of scope of continual learning — addressed by safety/red_blue.py + dpo_trainer.py persona-swap DPO during alignment phase |

---

## Numerical defaults

```python
# WebContentLearner
WebContentLearner(
    per_source_cap_7d=125,           # 50% of Anthropic 250
    gate_log=Path("gate_log.jsonl"),
    blocklist_path=Path("blocked_hashes.txt"),
)

# G1
SourceTrustEMA(
    decay_half_life_h=36.0,
    reject_threshold=0.20,
)

# G3
PerplexityGate(low_logppl=1.0, high_logppl=8.0)

# G7
TRAKApproxGate(min_novelty=0.05, max_novelty=0.95)

# DPO trainer (safety/dpo_trainer.py)
SafetyDPOTrainer(
    beta=0.1,        # memory: feedback_anthropic_safety_stack.md
    lr=5e-7,         # 10× lower than SFT
    ref_refresh=50,  # iterative DPO 2404.10719
    max_grad_norm=1.0,
)

# Existing pieces
PoisonDetector(
    thresholds={"inj": 0.30, "jb": 0.50, "oos": 1.0, "dup": 0.85,
                "pii": 0.40, "user_kl": 3.0, "trak": 0.0},
    weights={"inj": 0.30, "jb": 0.25, "oos": 0.20,
             "dup": 0.10, "pii": 0.10, "user_kl": 0.05},
    low_lr_band=(0.30, 0.50),
)

WeightFirewall(
    kl_clip=0.5, grad_clip_k=5.0,
    suspect_block_threshold=10, si_lambda=1.0,
)

ShadowMerge(
    lora_rank=16, lora_alpha=32, merge_interval_hours=4,
    canary_kl_block=0.02, canary_ppl_block_pct=3,
    rollback_kl=0.05, rollback_n=3,
)
```

---

## Anchor papers

- **Anthropic poison fixed-count = 250**: 2510.07192 (the load-bearing
  citation for the 125/source/7d cap)
- **TRAK influence**: Park et al, 2303.14186
- **TracIn / DataInf**: 2002.08484 / 2310.00902
- **Concealed data poisoning** (proves perplexity is gameable):
  Wallace 2010.12563
- **TR-DPO iterative ref refresh**: 2404.10719
- **DPO**: Rafailov 2305.18290
- **EWC**: Kirkpatrick 1612.00796 (warm-start without catastrophic forgetting)
- **Synaptic Intelligence** (per-parameter importance Ω_i): Zenke 1703.04200
- **LSHBloom** (1.2GB Bloom filter for 100M-entry blocklist): 2411.04257
- **STABLE gated continual**: 2510.16089
- **Sleeper Agents** (warning: standard safety can miss backdoors):
  Hubinger 2401.05566

---

## Safety contract — never violated

1. Track B never updates weights. Period. Any code adding gradient
   updates from chat must add the assertion `assert update_weights is False`.
2. No single source can ever exceed 125 admissions in any 7-day
   window. The smoke test enforces this as an `assert`.
3. Every gate decision logs all 7 scores. Audit trail is non-negotiable.
4. Web fetch failures degrade gracefully; daemon never crashes the
   pipeline.
5. The DPO reference model is frozen at construction; iterative
   refresh is opt-in (`ref_refresh` interval, default 50 steps).
