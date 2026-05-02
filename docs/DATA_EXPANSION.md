# Data Expansion — Diverse 1B-Token Corpus

This document describes the diverse-corpus build pipeline that breaks
the val-ppl plateau the trainer currently sees on a single ZH parquet
shard. It complements:

- `scripts/build_diverse_corpus.py` — the new fetcher / mixer
- `tests/integration/test_diverse_corpus.py` — 4-test contract suite
- `docs/PRETRAIN_DATA.md` — sibling doc covering the existing 5-corpus
  WuDao+SkyPile+FineWeb mix
- `synapforge/data/__init__.py` — `ParquetTokenStream` consumer

The output schema (`text`, `corpus`) is identical to
`mix_pretrain_corpora.py` so the trainer ingests it with no changes.

---

## 1. Current data inventory

Two parquet shards are reaching the trainer in production today:

| Shard | Path | Rows | Tokens (est.) | Lang | Notes |
|---|---|---:|---:|---|---|
| ZH synth pretrain | `/workspace/data/synth_zh_phase1.parquet` | 50K | ~30M | zh | `synth_chinese_pretrain.py` templated; tiny vocab tail |
| Alpaca instruct | `/workspace/data/alpaca_zh_qwen_tokenized.parquet` | 48K | ~25M | zh+en | `prep_alpaca_qwen.py`; SFT-shaped only |

The trainer's `ParquetTokenStream` cycles these forever (P24 in
`docs/MASTER_PLAN.md` §6 — deterministic ordering already root-caused as
the source of three Run 3a/3b/3c divergences at step ~2500). With
`shuffle_buffer=10000` enabled in `train_100m_kd.py` the lexical
sequence is now decorrelated within an epoch, but the **vocab support**
is unchanged: ~55M tokens of templated Chinese cannot teach English LM,
math, or code at all.

That is why val ppl has plateaued at ~7000.

---

## 2. Gap analysis

### Domain coverage table

| Domain | Currently reaching trainer | Why we need it | Severity |
|---|---|---|---|
| English LM (web) | ~5M (alpaca-en cells inside zh shard) | Required for MMLU / HellaSwag / WikiText eval; teacher KD distillation logits land in Qwen vocab whose 60% mass is English | CRITICAL |
| Code | 0 | HumanEval/MBPP eval need code modelling; `feedback_pure_code_kills_english.md` says cap at 30% | HIGH |
| Math chain-of-thought | 0 | GSM8K eval; MATH-500 RL phase will be unreachable | HIGH |
| Open instruction (en + zh) | ~25M alpaca | More instruction diversity = better chat habit transfer | MEDIUM |
| Real Chinese web (vs synth) | 0 | Templated-only ZH has ~5K vocab tail; real WuDao/CCI3 has ~80K | HIGH |

### Anchor scaling laws (for the ppl-impact estimate)

- **Chinchilla** ([2203.15556](https://arxiv.org/abs/2203.15556)) —
  100M params want ~2B tokens to compute-optimum. We are at ~55M, i.e.
  **~36× under-trained on tokens**, which alone explains a plateau.
- **FineWeb-Edu** ([2406.17557](https://arxiv.org/abs/2406.17557)) —
  high-quality English web filter; documented to drop val ppl ~30% vs
  generic CC at the same param count.
- **D2Z scheduling** (community 2025) — no-LR-decay + 4-epoch on a
  curated mix beats LR-decay + 1-epoch on a larger mix at small scale
  — already absorbed into Run 3n's constant-LR schedule.
- **`feedback_5T_effective_target.md`** (memory) — target 5T
  *effective* tokens via teacher distillation; first stepping stone is
  a real 1B mix, not the current 55M.

---

## 3. The diverse-corpus pipeline

### Per-category default mix (1B target)

| Category | Tokens | % | Source(s) | Why |
|---|---:|---:|---|---|
| `en` | 500M | 50% | FineWeb-Edu 10BT sample (fallback FineWeb) | English LM backbone — the largest under-served domain |
| `zh` | 200M | 20% | BAAI/CCI3-HQ (fallback SkyPile-150B) + existing synth_zh | Real Chinese web instead of templated-only |
| `code` | 150M | 15% | bigcode/the-stack-v2-train-smol Python | HumanEval/MBPP coverage; cap honours code-30% rule |
| `math` | 100M | 10% | open-web-math + EleutherAI/proof-pile-2 | GSM8K / MATH-500 chain-of-thought |
| `instruct` | 50M | 5% | silk-road/alpaca-data-gpt4-chinese + tatsu-lab/alpaca | More chat habit than 25M today |

`--include` filters to a subset; ratios renormalise over what's kept.

### Override examples

```bash
# default 1B mix
python scripts/build_diverse_corpus.py \
  --target-tokens 1B \
  --include en,zh,code,math,instruct \
  --out /workspace/data/diverse_corpus.parquet

# English-only smoke (debug)
python scripts/build_diverse_corpus.py --smoke --include en \
  --out /tmp/en_smoke.parquet

# 500M test mix without code (when stack-v2 is offline)
python scripts/build_diverse_corpus.py \
  --target-tokens 500M \
  --include en,zh,math,instruct \
  --out /workspace/data/diverse_500m.parquet
```

### Implementation details

- **HF streaming** (`datasets.load_dataset(..., streaming=True)`) so
  100 GB rentals don't pre-download terabytes — we stop pulling rows
  the moment per-cat budget is hit.
- **Mirror walk** per category (2 candidates each): first source that
  loads wins, errors logged + we move to the next.
- **sha256(first 4096 chars) dedup** across sources (catches FineWeb's
  ~3% cross-shard duplicates, plus code/math overlap with web).
- **Smoke mode** short-circuits all HF calls with hand-written
  fixtures (5 docs/category) — required for CI / Windows-dev / no-net
  rentals. All four integration tests use `--smoke` exclusively.
- **Manifest sidecar** (`<out>.manifest.json`) records target tokens,
  estimated tokens, by-corpus row counts, dedup-drops, and timestamp.
- **Tokenizer fallback** — real mode uses Qwen 2.5 0.5B for accurate
  token counting; if the tokenizer load fails (offline rental) we fall
  back to the char/4 estimate that the smoke path uses.

---

## 4. Expected ppl impact

Estimate framework: at 100M params we are ~36× below Chinchilla-optimum
on tokens. Holding architecture fixed and going from ~55M → 1B
tokens (18× more, still ~2× under Chinchilla) is the single biggest
lever available without changing the trainer.

| Lever | Mechanism | Δ val ppl (rough) |
|---|---|---|
| Real EN web (FineWeb-Edu 500M) | Teaches English LM from scratch; current run has zero EN signal | **−5000 → −6000** (the dominant source of the plateau) |
| Real ZH (CCI3-HQ 200M) | Replaces templated synth with vocab tail of ~80K | **−400 → −800** |
| Code + math (250M combined) | Adds new domain modality; small absolute ppl drop, big eval lift | **−100 → −300** (val) but **+15-25%** HumanEval pass@1 |
| Cross-source dedup | Removes ~3% near-duplicate rows that overweight common stems | **−50 → −150** |

**Combined target: val ppl 7000 → ~500-1500** within the first 2-3
epochs of the new mix, before any further architecture change.

This is *not* a guarantee — these scaling-law estimates assume the
trainer-side loss is bounded by data, not by an architecture bug. The
companion lever is `feedback_phased_training_2026q2.md` (phase 0
LM-only KD → ppl<=250 phase 1) which the trainer already uses; the
diverse corpus drops phase 0's ppl ceiling by an order of magnitude.

If the plateau persists after the new mix lands, the next suspects in
priority order are:

1. teacher KD chunk OOM (already root-caused; see `feedback_training_root_causes_2026q2.md`)
2. PLIF dead-bootstrap during warmstart from a non-spectral-norm ckpt
3. cosine-LR re-plateau on top of warmstart (per `feedback_cosine_lr_warmstart_replateau.md`,
   use constant LR until val ppl is stable for ≥2k steps)

---

## 5. Operational steps

```bash
# 1. real fetch (rental, network, ~30 min for 1B at 5 MB/s):
python scripts/build_diverse_corpus.py \
  --target-tokens 1B \
  --include en,zh,code,math,instruct \
  --out /workspace/data/diverse_corpus.parquet

# 2. point trainer's data glob at the new shard
python scripts/train_100m_kd.py \
  --data-glob /workspace/data/diverse_corpus.parquet \
  --shuffle-buffer 10000 \
  --backend triton_block --bs 128 \
  ...

# 3. sanity-check the parquet without touching the trainer
python -c "
import pyarrow.parquet as pq
t = pq.read_table('/workspace/data/diverse_corpus.parquet')
print(t.num_rows, 'rows; columns:', t.column_names)
print('by_corpus:', {c: t.column('corpus').to_pylist().count(c)
                     for c in set(t.column('corpus').to_pylist())})
"
```

---

## 6. Smoke / CI behaviour

`scripts/build_diverse_corpus.py --smoke`:

- Hits no network — fully hermetic.
- Writes 25 rows (5 per category × 5 categories) regardless of
  `--target-tokens`.
- Records `smoke: true` in the manifest so it can never silently ship
  to production.
- Uses char-based token estimation (no `transformers` required).

`tests/integration/test_diverse_corpus.py` covers four contracts:

1. `test_smoke_writes_parquet` — schema + manifest contract.
2. `test_token_count_matches_target` — manifest target field +
   parser unit-tests for `B/M/K` suffixes.
3. `test_per_category_ratios` — `--include` filtering + ratio
   renormalisation.
4. `test_no_duplicates_across_sources` — deduper actually runs and a
   deliberately-injected cross-source duplicate is dropped exactly
   once.

All four tests run on CPU, complete in < 2 seconds, and require only
`pyarrow`.

---

## 7. License compliance pointer

This script's per-category sources inherit `docs/PRETRAIN_DATA.md` §7
license rules verbatim:

| Category | Default source | License |
|---|---|---|
| en | FineWeb-Edu | ODC-By 1.0 |
| zh | BAAI/CCI3-HQ | MIT-equivalent (BAAI README) |
| code | bigcode/the-stack-v2 | Permissive only (MIT/BSD/Apache filtered upstream) |
| math | open-web-math | research-only |
| instruct | tatsu-lab/alpaca + silk-road alpaca | CC-BY-NC 4.0 |

Production / commercial release of trained weights requires dropping
the alpaca instruction shard (CC-BY-NC) and re-licensing the open-web-math
math shard. The fallback sources (SkyPile, proof-pile-2) have similar
caveats — verify before commercial use.
