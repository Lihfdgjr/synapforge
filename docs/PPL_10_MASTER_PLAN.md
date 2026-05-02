<!-- DOC_STAMP: LIVE 2026-05-02 -->
# PPL 10 Master Plan — Synap-1 Ultra → val ppl ≤ 10

**Status**: planning. Run 5 (Synap-1 Ultra 535M, d=1280 / n_layers=16 /
loop_depth=2) currently sits at **val_ppl_holdout = 5456 @ step 10000**
(see `docs/PROGRESS.md` and `docs/PERF_AUDIT_2026-05-02.md`).
**Owner**: Liu. Cross-links: [SCALING_RATIONALE.md](SCALING_RATIONALE.md),
[SNN_FREQUENCY_LIMIT_NEURIPS25.md](SNN_FREQUENCY_LIMIT_NEURIPS25.md),
[RUN3L_DIAGNOSIS.md](RUN3L_DIAGNOSIS.md), [DEEP_MAINT_QUEUE.md](DEEP_MAINT_QUEUE.md),
[MASTER_PLAN.md](MASTER_PLAN.md), [PHASE_TRAINING.md](PHASE_TRAINING.md).
**This is the canonical roadmap for the ppl-10 ambition. All sister-agent
work (T9.4 SFT, T9.5 GRPO, T9.6 cross-domain eval, T9.7 self-distill)
plugs into the phase gates here.**

---

## §1 Why ppl 10 — target rationale

### The ambition number and its peers

| Model                       | Reported / inferred val ppl on similar holdout | Source                       |
|-----------------------------|------------------------------------------------|------------------------------|
| Qwen 2.5-0.5B (our teacher) | **~9–11 on WikiText-style holdout**            | huggingface model card + KD logits |
| GPT-2 medium (355M)         | ~12–15                                         | Radford et al. 2019          |
| SmolLM2-360M                | ~10                                            | smollm2 release notes        |
| Synap-1 Base @ Run 3n end   | 3697 (plateau)                                 | `docs/SCALING_RATIONALE.md`  |
| Synap-1 Ultra @ Run 5 step10000 | **5456**                                   | this doc                     |

**Reading**: `ppl ≤ 10` is the band where a model stops emitting word
salad, generates a mostly-grammatical sentence in EN+ZH, and starts
being coherent enough that downstream SFT (T9.4) and RL (T9.5) actually
have a base to work with. Below 10 we'd be teacher-class; we are not
claiming to beat Qwen 0.5B at static benchmarks (see [INVESTOR.md
"What's NOT a claim"](INVESTOR.md)). We **are** claiming we can stand
next to it on raw next-token prediction with `~535M / ~340M useful
backbone` and a **100% LNN+SNN** stack — which is the paper hook.

### Why not aim for ppl 60 (the original phase-3 trigger)?

`docs/MASTER_PLAN.md` §3 still lists `val ppl ≤ 60 → SFT` as the
chat-emergence threshold. That target was set against the **100M Base**
plateau of 3700; ppl 60 is "at most 60× better than the plateau". For
**Ultra at 535M**, hitting only ppl 60 leaves ~80% of capacity unused;
the per-parameter compute and rental spend is wasted. The Ultra-class
target must be 5-10× lower or the scaling decision was a mistake.

---

## §2 Math of where we are

### Log-perplexity gap

```
log(5456) = 8.604
log(  10) = 2.303
gap        = 6.301 log-units
```

For reference, **each log-unit corresponds to roughly one phase / one
order-of-magnitude data-or-quality jump** in our retrospective
(`docs/TRAINING_ISSUES_RETROSPECTIVE.md`): Run 1→3b was 1.0 log-units,
3b→3n another 0.4, 3n→Ultra step10000 another 0.3 going **the wrong
way** (Ultra started over from scratch). The remaining **6.3 log-units**
will not come from one knob.

### Chinchilla data-optimality lens

Using the Hoffmann et al. 2022 rule of thumb **`tokens ≈ 20 × params`**:

| Model size | Chinchilla-optimal tokens | Synap actual | Ratio        |
|------------|---------------------------|--------------|--------------|
| 535M (Ultra) | **10.7B**                 | ~55M (Run 5 to step10000)\* | **~190× sub-optimal** |
| 100M (Base)  | 2.0B                      | ~492M (Run 3n) | ~4× sub-optimal |

*Token estimate: Run 5 reports ~14k tok/s, step 10000, bs effective 48,
seq 256. `10000 × 48 × 256 ≈ 123M tokens of forward passes`; with
shuffle-buffer 10000 and a 491M-token corpus, the **unique-token
exposure is bounded above by 55–60M**. Chinchilla math is not a law,
but **3-4 log-units of our 6.3 gap are pure data-starvation** and
cannot be closed by architecture knobs alone.

### What the gap decomposes into (best-honest split)

| Source                                | log-units we expect to recover | confidence |
|---------------------------------------|-------------------------------|------------|
| LM continuation on bigger / cleaner data (Phase 1) | **~1.3** (5456 → ~1500) | high |
| SFT alpaca-zh response-only loss (Phase 2) | **~2.9** (~1500 → ~80) | medium |
| GRPO RL with sympy/AST verifier (Phase 3) | **~1.0–1.4** (~80 → 20–30) | low |
| Self-distill student-teacher rebound (Phase 4) | **~0.7–1.1** (20–30 → ≤10) | low |

**Sum**: 5.9-6.7 log-units — i.e. it is *just* possible if every phase
delivers the high end. There is **no phase to spare**; missing one
budget is the difference between ppl 10 and ppl 30.

---

## §3 4-phase plan

```
                 |  start ppl  |  end ppl  | wall-clock | $$  | task IDs |
Phase 1  LM cont.|  5456       |  ~1500    |  18-24 h   | $130-170 | this doc |
Phase 2  SFT     |  ~1500      |  ~80      |   4-6 h    | $30-45 | T9.4    |
Phase 3  GRPO RL |  ~80        |  20-30    |   2-3 h    | $15-25 | T9.5    |
Phase 4  self-distill | 20-30  |  ≤ 10     |   1-2 h    | $10-15 | T9.7    |
                       --------                            -------
                                                          25-35 GPU-h, $175-245
```

Cross-domain eval harness (T9.6) runs at **every phase boundary**, not at
the end. See §5.

### Phase 1 — LM continuation (Run 6, full stack on diverse 1B-token corpus)

- **Trigger**: Run 5 `step_010000.pt` exists and is backed up (mohuanfang
  + GitHub release).
- **Launch**: `scripts/launch_synap1_ultra_run6.sh` (already on disk).
  Warmstart from `step_010000.pt`, `--no-strip-optim` (Run 5 was healthy,
  not divergent — keep momentum).
- **Data**: **1B-token diverse corpus** (`docs/DATA_EXPANSION.md`):
  WikiText-103 + FineWeb-EDU 250M + synth_zh 200M + alpaca_zh raw text
  150M + GSM8K chains 50M + HumanEval/MBPP 30M. Tokenize once,
  shuffle-buffer 50000 (5× current), shuffle-seed-rotate per epoch.
- **Knobs unique to Run 6** (everything from `docs/PERF_AUDIT_2026-05-02.md`
  recommended combo, plus):
  - `--cuda-sync-every 10`, `--clip-grad-cache`, `--kd-async-teacher`,
    `--prefetch-factor 4`, `--pin-memory`.
  - `--byte-patch-pool max+avg` (T-snn-freq A1).
  - `--high-pass-residual-weight 0.1` (T-snn-freq A2 — addresses PLIF
    dead 16/16 via frequency, see `SNN_FREQUENCY_LIMIT_NEURIPS25.md` §4).
  - `--plif-tau-init tri-modal` (T-snn-freq A3).
  - `--shuffle-buffer 50000`, `--shuffle-seed 1109`.
  - `--lr 8e-5` (Ultra warmstart-continuation-safe; see
    `feedback_cosine_lr_warmstart_replateau.md`).
  - `--lr-schedule constant` for first 5000 steps, then cosine to 1e-5
    over remaining budget.
- **Target**: val_ppl_holdout `5456 → ≤ 1500` by step 60000 (Ultra
  budget extension to 18-24h).
- **Abort criteria** (per `feedback_run3c_divergence_threshold.md`):
  val ppl in 500 steps crosses 7000 OR 1000 steps ≥ 3× best → SIGTERM,
  diagnose, **don't wait**.
- **Best-ckpt symlink**: T5.4 `--best-ckpt-track` ON (mandatory per
  `feedback_best_ckpt_track_mandatory.md`).

### Phase 2 — SFT alpaca-zh (T9.4, sister agent owns)

- **Trigger**: Phase 1 final ckpt with val_ppl_holdout ≤ 1500
  **on the cross-domain harness** (T9.6 confirms not just WT-103).
- **Switch trainer**: `train_100m_sft.py` (sister agent shipping).
  Data: `alpaca_zh_qwen_tokenized.parquet` + `alpaca_en` + 5%
  GSM8K reasoning chains for breadth.
- **Knobs**: `--response-only-loss`, `--lr 1e-4` (per MASTER_PLAN §3
  phase-3 row), `--epochs 3`, KD weight retained at 0.4 from Phase 1.
- **Target**: val ppl `~1500 → ~80`, chat eval ≥ 35% pass on 5 EN + 5 ZH
  prompts (`docs/CHAT_SAMPLES.md`).
- **Risk**: SFT on a model with ppl 1500 produces fluent-sounding
  hallucination. The 1500 must be on cross-domain holdout; if it's only
  on the SFT distribution, Phase 2 numbers will lie. T9.6 is the gate.

### Phase 3 — GRPO RL (T9.5, sister agent owns)

- **Trigger**: Phase 2 ppl ≤ 80 AND chat eval ≥ 35%.
- **Trainer**: GRPO with sympy/AST verifier on GSM8K-train (`docs/MASTER_PLAN.md`
  §3 phase-4). Reward = `1[verified_correct] - β·N·max(p_solve, 1/K)`
  per `reference_alp_reward_2506.md` (length-aware, hard-negative
  cushioned).
- **Target**: val ppl `~80 → 20-30` AND GSM8K-eval pass-rate ≥ 25%.
- **Risk**: GRPO can collapse a generative head if reward sparsity is
  too high. Curriculum gate β=0 until p_solve > 0.3.

### Phase 4 — Self-distill (T9.7, sister agent owns)

- **Trigger**: Phase 3 ppl ≤ 30.
- **Method**: best Phase 3 ckpt becomes the **teacher**; train a fresh
  Ultra-shape student with 100% KD weight from this teacher on the
  Phase 1 corpus + alpaca + GSM8K chains. The student gets KD signal
  from a model that already knows EN+ZH+math, instead of from
  Qwen 0.5B which is broader but less in-domain.
- **Target**: val ppl `20-30 → ≤ 10`.
- **Risk**: self-distill amplifies the teacher's bias; gating MUST be
  cross-domain (T9.6) on a held-out distribution the teacher never saw.

---

## §4 Risks (told straight)

1. **Data scarcity is the dominant gap.** ~190× sub-Chinchilla is not
   closeable with a synth_zh + alpaca_zh combo. Phase 1 *requires* the
   1B-token diverse corpus per `docs/DATA_EXPANSION.md`. If that corpus
   is not assembled and pre-tokenized **before** Phase 1 starts, the
   plan's first row (1.3 log-units) does not happen. **This is the
   #1 risk.**
2. **Architecture cap may be real.** 100% LNN+SNN at 535M has never been
   trained to ppl 10 by anyone (literature search 2026-04). Pure
   transformer at this size hits ppl 10 routinely. We are betting that
   biology-inspired primitives don't have a higher floor than scaled
   attention; we have **no proof** of this above ppl ~3700 today.
   Mitigation: T9.6 cross-domain eval on each phase boundary. If Phase 1
   plateaus > 1500 *across all four eval domains*, this risk is
   confirmed and we drop the ambition to ppl ~80 (Phase 2 only) —
   i.e. honest-investor answer becomes "ppl 60-80 with the chat-emerge
   threshold met", not "ppl 10".
3. **PLIF dead 16/16 in Ultra.** Per `RUN3L_DIAGNOSIS.md` H2 + Run 3n
   evidence, the spike layer has been all-zero across the entire 100M
   lineage. Ultra inherits this. The "true SNN benefit" the pitch
   relies on (energy/sparsity/STDP) is **unrealized in training** and
   `synapforge.bio.stdp_fast` is technically running on noise.
   `--high-pass-residual-weight` (Run 6 knob) is the SEW-style fix per
   `SNN_FREQUENCY_LIMIT_NEURIPS25.md`. If spike rate is still 0/16 by
   Run 6 step 5000, kill and replan — we should not pay rental cost
   for a CfC-only run.
4. **Eval contamination.** Single-domain WT-103 ppl is not credible
   below ~100. Phase 1 self-shuffle includes WT-103-train; SFT data
   includes alpaca-eval-style splits; GRPO data includes GSM8K-train.
   Cross-domain holdout (T9.6) using **C4-en, C4-zh, and at least one
   never-seen general-ZH source** is the *only* way the ppl 10 number
   means something.
5. **Run-3c-class divergence** (`feedback_run3c_divergence_threshold.md`).
   Long Phase 1 budget (60000 steps) is exactly the regime where data-
   ordering drift bites at 2500 + 5000 step boundaries. The
   shuffle-buffer 50000 mitigation has never been tested at Ultra scale.
6. **Cross-phase composability.** SFT after KD-LM has worked at our
   scale; GRPO after SFT has not. Sister agents (T9.4/9.5) own the
   verification work; if their wiring breaks, this plan stalls.

---

## §5 Verification protocol — cross-domain holdout

The harness (T9.6, sister agent) reports val ppl on **four independent
holdout sets** at every phase boundary:

| Holdout      | Source                        | Why it matters                  |
|--------------|-------------------------------|---------------------------------|
| WT-103 val   | huggingface `wikitext-103-v1` | continuity with Run 3 history   |
| C4-en       | allenai/c4 `en` 5K-shard      | tests EN distribution shift     |
| C4-zh       | allenai/c4 `zh` 5K-shard      | tests ZH distribution shift     |
| general-ZH   | THUCNews + zhwiki sample      | tests open-domain ZH coherence  |

**Rule**: a phase is considered *passed* only when ppl ≤ target on
**all four**. WT-103-only pass = continue training, do not advance phase.

**Per-phase log emission** (one line in `docs/PROGRESS.md`):
```
Phase N step X: WT103=A.AA, C4en=B.BB, C4zh=C.CC, genZH=D.DD, GATE=PASS|FAIL
```

Dashboards / phase-manager hook on the GATE field.

---

## §6 ETA — 25-35 GPU-h, $175-245, 2-3 days wall-clock

| Item                    | GPU-h | Cost (¥7/h A800 spot) | Wall  |
|-------------------------|-------|------------------------|-------|
| Phase 1 (Run 6)         | 18-24 | ¥126-168 ($18-24)      | 1d   |
| Phase 2 SFT (T9.4)      | 4-6   | ¥28-42 ($4-6)          | 6h   |
| Phase 3 GRPO (T9.5)     | 2-3   | ¥14-21 ($2-3)          | 3h   |
| Phase 4 self-distill    | 1-2   | ¥7-14 ($1-2)           | 2h   |
| Cross-domain eval (T9.6) | 0.5  | ¥3.50 ($0.50)         | inline |
| Buffer (re-launches)     | ~3   | ¥21 ($3)              | -    |
| **Total**               | **25-35** | **$175-245** USD-equivalent at $25-35/GPU-h cloud-rate equivalent | **2-3 days** |

The "$175-245" headline assumes **rented A800 80GB at ~¥7/h spot
($1/h)** which is what we've been paying. At AWS p4 ($32.77/h on-demand)
the same campaign costs ~$820-1150 — still below a single fine-tune of
a 7B-class transformer.

---

## §7 Decision matrix — when to advance, when to abort

### When to **advance** a phase

| From → To | Numeric trigger                              | Eval gate           |
|-----------|----------------------------------------------|---------------------|
| 1 → 2     | Phase 1 val ≤ 1500 on **all 4** T9.6 holdouts | chat eval ≥ 0% (no regression) |
| 2 → 3     | Phase 2 val ≤ 80 on **all 4** holdouts        | chat eval ≥ 35% pass |
| 3 → 4     | Phase 3 val ≤ 30 + GSM8K-eval ≥ 25%           | chat eval ≥ 50% pass |

### When to **abort or replan**

| Symptom                                          | Action                                       |
|--------------------------------------------------|----------------------------------------------|
| Phase 1 plateaus > 1500 for ≥ 3000 steps × 2 phases (~6000 steps total) | Reassess data: did the 1B-token corpus actually land? Is shuffle healthy? Is freq-residual on? |
| Phase 1 plateaus > 1500 across all 4 holdouts    | **Architecture cap confirmed.** Drop ambition to ppl 80 (Phase 2 only). Tell investors honestly. |
| Run-3c-class divergence (val crosses 7000 in 500 steps OR ≥ 3× best in 1000) | SIGTERM, restart from best-ckpt symlink, drop LR by 0.5×, do **not** continue cosine. |
| Spike rate 0/16 still at Run 6 step 5000          | Kill — frequency knobs failed. Re-plan: `--plif-amplify` flag (next iter) or accept LNN-only and re-pitch. |
| Phase 2 ppl ≤ 80 BUT chat eval < 25%              | SFT data is over-fit. Replan SFT mix (lower α, add raw text). |
| Phase 4 self-distill increases ppl on cross-domain (only WT103 drops) | Reject phase 4. Ship Phase 3 ckpt as final. |

### Success criteria per phase (final answer)

- **Phase 1 success**: val_ppl_holdout (4-domain mean) ≤ 1500. PLIF
  spike rate ∈ [0.05, 0.20] on at least 8/16 layers. Loss curves
  monotonic over last 5000 steps (drift < 5%).
- **Phase 2 success**: val ≤ 80 (4-domain mean) AND chat eval ≥ 35%.
  Best ckpt at the rebound — SFT often overshoots.
- **Phase 3 success**: val ≤ 30 (4-domain mean) AND GSM8K ≥ 25% AND
  HumanEval ≥ 5%. Reward curve monotonic over last 1500 RL steps.
- **Phase 4 success**: val ≤ 10 (4-domain mean) AND chat eval ≥ 60%.
  No domain regression (no holdout > 30).

If §7's success criteria for **all four phases** are met, we ship
"Synap-1 Ultra v1.0" and update `INVESTOR.md` with the verified numbers.

If only Phase 1-2 succeed, we ship "Synap-1 Ultra Lite (ppl ~80)" and
honest-pitch the result. The deck does not change shape — claim 1-5
in INVESTOR.md hold at any of these endpoints because they are
mechanism claims, not ppl claims.

---

**Last updated**: 2026-05-02 (initial draft on `feature/ppl-10-plan`).
**Next refresh**: at every phase boundary OR when Run 6 status changes.
