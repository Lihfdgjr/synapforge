# Pretrain Data — Multilingual Corpus Reference

This document describes how SynapForge sources, mixes, and stages pretrain data
for the multilingual chat / paper-bench / multilingual-balanced training tracks.
It is the companion of:

- `scripts/download_pretrain_multilingual.sh` — fetcher (3-mirror walk per corpus)
- `scripts/mix_pretrain_corpora.py`           — ratio-based mixer + dedup + filter
- `scripts/synth_chinese_pretrain.py`         — offline-only Chinese fallback
- `docs/DATA_SOURCES.md`                      — sibling doc covering SFT / multimodal / eval

All real-corpus downloads are **idempotent** (size + sha256 gate); re-running
with files in place skips network IO. Smoke mode (`--size smoke`) tolerates
total network failure and is the recommended first run on any new rental.

---

## 1. Why multilingual pretrain

The 100M LNN+SNN backbone trained on `WikiText-103` + 1×FineWeb-edu file is
**English-only**. Real chat traffic on the model is bilingual (Chinese + English)
plus code, so pretrain must reflect that mix or the model will degrade in
non-English domains.

Anchor papers and references:

- **Chinchilla** (Hoffmann et al., 2022, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) — token budget = 20 × params (so 100M params ⇒ 2B tokens).
- **FineWeb / FineWeb-edu** (Penedo et al., 2024, [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)) — high-quality English web filter pipeline.
- **D2Z scheduling** (community 2025) — no-LR-decay, 4 epoch on a curated mix beats LR-decay + 1 epoch on a larger mix at small scale.
- **WuDao** (BAAI 2021, technical report `WuDao 2.0` 2003.07000 series) — 200 GB Chinese web + literature.
- **SkyPile-150B** (Skywork 2023, [arXiv:2310.19341](https://arxiv.org/abs/2310.19341)) — 150 B token Chinese pretrain.
- **CCI3-HQ** (BAAI 2024) — Chinese Common Crawl filtered, MIT-equivalent license.
- **Cosmopedia v2** (HuggingFace TB 2024) — 25B token synthetic educational text.
- **The-Stack v2** (BigCode 2024, [arXiv:2402.19173](https://arxiv.org/abs/2402.19173)) — 67 TB de-duped permissively licensed source code.

The `feedback_pure_code_kills_english.md` lesson (adv12b_code, 2026-04-21):
pure code pretrain blows English ppl from 606 to 5257 in 3000 steps. Code
must be ≤ 30 % of pretrain mix; the 70/20/10 mix below is what we ship.

---

## 2. Per-corpus card

Sources, sizes, licenses, and recommended ratios. Sizes are subset sizes —
the full corpus rarely fits on a 100 GB rental disk.

| Corpus | Source (mirror) | Lang | Subset size | License | Ratio (chat) | Ratio (paper-bench) |
|---|---|---|---|---|---|---|
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` 10BT sample | en | 2 GB / 10 K docs | ODC-By 1.0 | 0.35 | 0.50 |
| `wudao` | `BAAI/WuDaoCorpora` p208p2002 mirror | zh | 2 GB / 10 K docs | Apache-2.0 | 0.20 | 0.10 |
| `skypile` | `Skywork/SkyPile-150B` | zh | 1 GB / 10 K docs | Apache-2.0 | 0.15 | 0.10 |
| `the_stack_v2_python` | `bigcode/the-stack-v2-train-smol` Python | code | 500 MB / 10 K files | Permissive only | 0.20 | 0.20 |
| `cosmopedia_v2` | `HuggingFaceTB/cosmopedia-v2` | en | 1 GB / 5 K docs | ODC-By 1.0 | 0.10 | 0.10 |
| `cci3_hq` | `BAAI/CCI3-HQ` | zh | 2 GB / 10 K docs | MIT-equivalent | optional | optional |

License notes:

- All HF mirrors are **research-only** for the model weights step. Commercial
  release of fine-tuned weights requires re-licensing (typically by sourcing
  ODC-By 1.0 / Apache-2.0 / MIT-only subsets).
- `the_stack_v2_python` filters to permissive licenses only (MIT, BSD, Apache);
  GPL / AGPL / LGPL are excluded by BigCode upstream.
- WuDao is officially Apache-2.0 in the public mirror, but the original BAAI
  release is research-only — verify the mirror you actually use.

---

## 3. Mix strategy table

Three ratios, parameterised by `--ratios` to `mix_pretrain_corpora.py`:

| Goal | EN web | ZH web | Code | Synth EN | Notes |
|---|---|---|---|---|---|
| **chat** (default) | 0.35 fineweb | 0.35 (wudao 0.20 + skypile 0.15) | 0.20 stack | 0.10 cosmopedia | balanced bilingual |
| **paper-bench** | 0.50 fineweb | 0.20 (wudao 0.10 + skypile 0.10) | 0.20 stack | 0.10 cosmopedia | English-heavy, MMLU/HellaSwag friendly |
| **multilingual** | 0.30 fineweb | 0.40 (wudao 0.20 + skypile 0.15 + cci3_hq 0.05) | 0.20 stack | 0.10 cosmopedia | Chinese-leaning |
| **code-heavy** | 0.30 fineweb | 0.20 (wudao 0.10 + skypile 0.10) | 0.40 stack | 0.10 cosmopedia | HumanEval / MBPP focused — risk of English regression |

Pass-through CLI:

```bash
# chat default
python scripts/mix_pretrain_corpora.py \
  --root /workspace/data/pretrain \
  --out  /workspace/data/pretrain/pretrain_mix.parquet \
  --target-rows 200000

# paper-bench override
python scripts/mix_pretrain_corpora.py \
  --ratios "fineweb_edu:0.50,wudao:0.10,skypile:0.10,the_stack_v2_python:0.20,cosmopedia_v2:0.10" \
  --out /workspace/data/pretrain/paperbench_mix.parquet

# multilingual override
python scripts/mix_pretrain_corpora.py \
  --ratios "fineweb_edu:0.30,wudao:0.20,skypile:0.15,cci3_hq:0.05,the_stack_v2_python:0.20,cosmopedia_v2:0.10" \
  --out /workspace/data/pretrain/multilingual_mix.parquet
```

Mixer features:

- **sha256-prefix dedup** on first 4096 chars of each text (catches identical
  doc, near-identical tail); FineWeb still ships ~3 % cross-shard duplicates.
- **min-chars / max-chars filter** (default 50 / 32 000). Drops empty,
  blank-only, or oversize docs (oversize blow up tokenizer).
- **language id** (CJK ratio heuristic) recorded as `lang` column so trainer
  can re-balance batches without re-shuffling source.
- **deterministic seed** for reproducible mixes between two rental rentals.

---

## 4. Disk budget per `--size`

| `--size` | Per-corpus quota | Total disk | Use case |
|---|---|---|---|
| `smoke`  | ~100 docs / corpus | < 100 MB | CI, offline rental, code-path test |
| `small`  | ~10 K docs / corpus | ~10 GB | Real subset, fits on cheap 100 GB rental |
| `medium` | ~100 K docs / corpus | ~30 GB | Targeted bench, fits on 100 GB rental |
| **full** | full upstream | 100 GB+ | **Will not fit**; use sharded streaming |

The 100 GB rental floor used by the project means **`full` mode is intentionally
not supported** by the bash script. To stream the full upstream you must use
HF `datasets.load_dataset("...", streaming=True)` from the trainer instead.

---

## 5. Synthetic Chinese fallback

When real ZH downloads fail (mirror outage / new rental with no internet),
`scripts/synth_chinese_pretrain.py` generates 50K plausible articles from
hand-written templates × topic substitutions:

- 100 title templates × 500 topic substitutions = 50K determined-by-seed articles
- Each 200-1500 chars: 标题 + 引言 + 3-5 段落 + 结论
- 10 topic domains (科技 / 历史 / 文学 / 数学 / 物理 / 经济 / 哲学 / 心理 / 生物 / 地理) ×
  15 subtopics each = 150 (topic, subtopic) anchors
- Output `train.parquet` with same schema as real corpora (`text`, `lang`)
- Deterministic: seed=42 → byte-identical output

```bash
python scripts/synth_chinese_pretrain.py \
  --out /workspace/data/pretrain/synth_zh/train.parquet \
  --n 50000 --seed 42

# smoke
python scripts/synth_chinese_pretrain.py --smoke
```

**Honest disclaimer:** synthetic Chinese pretrain is **demo-fallback only**.
Templated text has tiny vocab tail and trainable on it only teaches canonical
中文 article shape, not real lexical / topical coverage. Use:

- as 1-2 % admixture (trainer learns 中文 grammar skeleton even when network is dead)
- to validate the trainer code-path on completely offline rentals
- **never** as the only Chinese source for the production model

The real WuDao / SkyPile / CCI3-HQ corpora are required for production.

---

## 6. End-to-end recipe

```bash
# 1. fetch raw corpora (smoke mode tolerates network failure)
bash scripts/download_pretrain_multilingual.sh --size small

# 2. fall back synthetic ZH if real ZH all failed
if [[ ! -f /workspace/data/pretrain/wudao/train.parquet \
   && ! -f /workspace/data/pretrain/skypile/train.parquet ]]; then
  python scripts/synth_chinese_pretrain.py --n 50000
fi

# 3. mix into a single parquet
python scripts/mix_pretrain_corpora.py \
  --root /workspace/data/pretrain \
  --out  /workspace/data/pretrain/pretrain_mix.parquet \
  --target-rows 200000

# 4. trainer ingests pretrain_mix.parquet (no schema change required)
python train_100m.py --pretrain-parquet /workspace/data/pretrain/pretrain_mix.parquet
```

The trainer reads the `text` column. The `corpus` and `lang` columns are
optional metadata that downstream curriculum schedulers can use to weight
batches (e.g. ramp ZH from 0.10 → 0.40 over the first epoch).

---

## 7. License compliance one-pager

| If you ship... | You need to verify... |
|---|---|
| Pretrained weights (research only) | All sources are research-license-or-better. WuDao mirror license is verified. |
| Pretrained weights (commercial) | Drop CC-BY-NC sources entirely. Use FineWeb-edu (ODC-By) + The-Stack-v2 (permissive only) + Cosmopedia (ODC-By) + CCI3-HQ (MIT). Do **not** use the WuDao p208p2002 mirror — verify upstream Apache-2.0 licensure first. |
| SFT-only fine-tuning weights | Pretrain license carries through. Add SFT licenses on top (Alpaca = CC-BY-NC). |
| Just the trainer code | No data license issue. |

A separate `LICENSES.txt` per corpus is written by the downloader at
`/workspace/data/pretrain/<corpus>/manifest.json` containing `source` and
`updated` timestamps for audit.

---

## 8. Refresh cadence

- **Weekly**: re-run `download_pretrain_multilingual.sh --size small` to catch
  upstream mirror updates (HF re-shards roughly monthly).
- **Per-rental-rebuild**: always re-run with `--size smoke` first to validate
  network reachability, then upgrade to `small` / `medium`.
- **Pre-paper-submission**: re-run with `--size medium` so the published mix
  is the same one used for the bench numbers.

The downloader's idempotent gate means re-runs are cheap when files are fresh.

---

## 9. Known issues / open work

1. The bash downloader does **not** apply MinHash dedup across corpora — only
   within-corpus (handled by upstream HF). Cross-corpus near-duplicates
   between WuDao and SkyPile may persist; mixer's sha256 prefix catches
   identical heads but not paraphrased duplicates. Track a future MinHash pass.
2. `the_stack_v2_python` ships only a single Python shard — JS / TS / Rust /
   Go / Java are absent from the default subset. Add per-language shards to
   the bash script for real code coverage.
3. `cci3_hq` license is MIT-equivalent in the BAAI README, but not formally
   stated in the dataset card. Lawyers should verify before commercial release.
4. No upstream sha256 manifest for any of these corpora — the downloader's
   `*.sha256` file is computed locally, so it only protects against
   in-download corruption, not upstream tampering.
5. Synthetic Chinese is templated and will create vocab spike on a small set
   of nouns / verbs / adjectives. Don't train on it in isolation; ensure
   real ZH corpus precedes it in any mix.
