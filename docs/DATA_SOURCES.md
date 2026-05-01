# Data Sources

Single-page reference for every external dataset SynapForge depends on:
where it comes from, what it is for, license terms, and how to refresh.

This complements the prep scripts:

- `scripts/download_alpaca.sh`           — SFT instruction data (~60 MB)
- `scripts/download_multimodal_real.sh`  — image / audio / video / time-series / graph (~8 GB cap)
- `scripts/download_eval_data.sh`        — bench harness data (~100 MB)
- `scripts/prep_alpaca_qwen.py`          — JSON → parquet tokenizer
- `scripts/prep_multimodal_data.py`      — byte-patch parquet writer (synth fallback)
- `scripts/prep_3d_data.py`              — synthetic 3D
- `scripts/sync_to_mohuanfang.sh`        — defense-in-depth backup loop

All scripts are idempotent. Re-running with files already on disk skips the
network step (size + sha gate).

---

## 1. Mirror priority

The rental sits in mainland China, so direct `huggingface.co` is slow or
blocked. Order of preference for every URL:

| Tier | Host                                  | Why                                             |
|------|---------------------------------------|-------------------------------------------------|
| 1    | `hf-mirror.com`                       | Full HF mirror, China-friendly, no auth needed  |
| 2    | `modelscope.cn`                       | Alibaba mirror, often has Chinese variants      |
| 3    | `huggingface.co` (origin)             | Fallback if mirrors lag or 404                  |
| 4    | `raw.githubusercontent.com`           | Project-original copies (alpaca, lima, mbpp)    |
| 5    | `data.binance.vision`                 | OHLCV archives (free, public)                   |

Each fetcher in the bash scripts walks 3 URLs in this order. First one to
return >`MIN_BYTES` of valid data wins.

---

## 2. SFT data — `/workspace/data/sft/`

| Dataset       | File              | Source (mirror)                                                | Size  | License             | Purpose                          |
|---------------|-------------------|----------------------------------------------------------------|-------|---------------------|----------------------------------|
| Alpaca-EN     | `alpaca_en.json`  | `tatsu-lab/alpaca` / `yahma/alpaca-cleaned`                    | 25 MB | CC-BY-NC 4.0        | English SFT (52K)                |
| Alpaca-ZH     | `alpaca_zh.json`  | `silk-road/alpaca-data-gpt4-chinese` / `shibing624/alpaca-zh`  | 30 MB | CC-BY-NC 4.0        | Chinese SFT (51K)                |
| LIMA          | `lima.json`       | `GAIR/lima` / `64bits/lima_vicuna_format`                      |  3 MB | CC-BY-NC-SA 4.0     | High-quality SFT (1K curated)    |
| MetaMath / OWM| `math_qa.json`    | `meta-math/MetaMathQA` / `microsoft/orca-math-word-problems-200k` | 50 MB | MIT / CDLA-2.0    | Math QA bilingual (10K subset)   |

**License note:** all of these are research-only or non-commercial. Do not
ship trained weights commercially without re-licensing the SFT step.

Refresh:

```bash
bash scripts/download_alpaca.sh                # all 4
bash scripts/download_alpaca.sh --only lima    # one
```

Tokenize → parquet:

```bash
python scripts/prep_alpaca_qwen.py \
  --alpaca-en /workspace/data/sft/alpaca_en.json \
  --alpaca-zh /workspace/data/sft/alpaca_zh.json \
  --out /workspace/data/sft/alpaca.parquet \
  --tokenizer-path /workspace/teachers/qwen2.5-0.5b
```

---

## 3. Multimodal real data — `/workspace/data/multimodal/`

| Modality    | Subset                          | Source (mirror)                                          | Cap   | License           | Purpose                            |
|-------------|---------------------------------|----------------------------------------------------------|-------|-------------------|------------------------------------|
| image       | CC12M-LR 50K pairs              | `Lin-Chen/CC12M` / LAION-COCO / Conceptual Captions       | 5 GB  | research only     | Byte-patch image-caption training  |
| audio       | LibriSpeech-clean-100 (5h)      | rsync from `mohuanfang.com:/home/liu/synapforge_backup/` | 2 GB  | CC-BY 4.0         | Mel memmap for raw-audio path      |
| video       | WebVid-tiny 1K / HowTo100M-mini | `TempoFunk/webvid-10M`                                    | 1 GB  | research only     | Short video clips                  |
| time-series | ETHUSDT 1m OHLCV (1 month)      | `data.binance.vision`                                     | 50 MB | public domain     | Real OHLCV vs GBM-synth A/B        |
| graph       | ZINC-250K (5K subset)           | HF mirrors + snap.stanford.edu                            | 10 MB | research only     | Molecule-graph byte payload        |

**Synthetic fallback:** every modality in `prep_multimodal_data.py` has
a synthetic generator that produces byte-shape-compatible parquet rows. If
real data isn't available the trainer keeps working — just with synth
labels (e.g. "a red circle" instead of CC12M captions).

**License note:** CC12M, LAION, WebVid, and ZINC are research-only datasets.
The byte-patch encoder doesn't store the source images per se, but the
trained weights inherit the upstream license.

Refresh:

```bash
bash scripts/download_multimodal_real.sh                    # all
bash scripts/download_multimodal_real.sh --only audio       # rsync librispeech
bash scripts/download_multimodal_real.sh --skip image       # everything but images
```

Per-modality manifest at `/workspace/data/multimodal/manifest.json`. Each
modality records `{source, path, size_bytes, status, n_items, updated}`.

---

## 4. Eval data — `/workspace/data/eval/`

| Bench       | Script                              | Source (mirror)                                          | Size  | License        |
|-------------|-------------------------------------|----------------------------------------------------------|-------|----------------|
| HumanEval   | `synapforge.bench.humaneval`        | `openai_humaneval` (parquet) + GitHub jsonl.gz fallback   | 0.5 MB| MIT            |
| MBPP        | `synapforge.bench.mbpp`             | `google-research-datasets/mbpp` parquet + raw GitHub      | 2 MB  | CC-BY 4.0      |
| MMLU        | `synapforge.bench.mmlu`             | `cais/mmlu` parquet (test + dev splits)                   | 50 MB | MIT            |
| GSM8K       | `synapforge.bench.gsm8k`            | `openai/gsm8k` parquet + GitHub jsonl                     | 5 MB  | MIT            |
| HellaSwag   | `synapforge.bench.hellaswag`        | `Rowan/hellaswag` parquet + GitHub jsonl                  | 25 MB | MIT            |
| LAMBADA     | `synapforge.bench.lambada`          | `EleutherAI/lambada_openai` jsonl                         | 10 MB | research only  |

`bench/*` modules already prefer a local file passed via `--data PATH`
and fall back to `datasets.load_dataset()` when missing — so this download
step is purely a perf/cost optimization (no repeated HF API calls during
sweep runs).

Refresh:

```bash
bash scripts/download_eval_data.sh             # all 6
bash scripts/download_eval_data.sh --only mmlu # one only
```

Files land in `/workspace/data/eval/<name>/`. A summary lands at
`/workspace/data/eval/SUMMARY.txt`.

---

## 5. Pretraining data — `/workspace/data/fineweb/`

Fetched separately via the FineWeb-edu HF dataset. Currently 1 file ~2 GB
loaded by the active trainer. Not in the bash scripts above because it's
already on disk; if it gets evicted, refresh with:

```bash
huggingface-cli download HuggingFaceFW/fineweb-edu \
  --include 'sample/350BT/000_00000.parquet' \
  --local-dir /workspace/data/fineweb
```

License: ODC-By-1.0 (commercially usable with attribution).

---

## 6. Disk budget

The rental has 100 GB system disk; we want to keep room for ckpts (~40 GB)
and OS overhead.

| Bucket                         | Cap   |
|--------------------------------|-------|
| SFT (`alpaca_*`, `lima`, math) | 60 MB |
| Multimodal real (5 modalities) | 8 GB  |
| Eval (6 benches)               | 100 MB|
| FineWeb-edu (1 shard)          | 2 GB  |
| Run dir (ckpts, logs)          | 40 GB |
| **Total**                      | ~50 GB|

Safe margin: 50 GB free even at peak.

---

## 7. Offline / smoke mode

Every script has a synthetic or skip path so smoke runs work without
internet:

| Script                                | Offline behaviour                                                  |
|---------------------------------------|--------------------------------------------------------------------|
| `prep_multimodal_data.py --smoke`     | All 9 modalities → synthetic byte-patch parquet (~30 s)            |
| `prep_alpaca_qwen.py`                 | Skip mode = both `--alpaca-*` flags blank → exit early             |
| `prep_3d_data.py`                     | Pure synthetic                                                     |
| `download_alpaca.sh --only X`         | Each fetcher independently retries; failures recorded but skipped  |
| `download_multimodal_real.sh`         | Each modality wrapped in try/except; manifest records `failed`     |
| `download_eval_data.sh`               | Bench code already falls back to `datasets.load_dataset` at runtime|

Reproducibility:

- `prep_multimodal_data.py` is seeded by `--seed` (default 42), so synth
  runs are byte-deterministic on the same machine.
- `download_*.sh` scripts skip files already on disk above the size floor.

---

## 8. Backup paths

Defense-in-depth (any one of these surviving = no data loss):

| Path                                                          | Mechanism                            |
|---------------------------------------------------------------|--------------------------------------|
| `mohuanfang.com:/home/liu/synapforge_backup/<run>/`           | `scripts/sync_to_mohuanfang.sh` loop |
| `mohuanfang.com:/home/liu/synapforge_backup/<run>/` (redundant)| `triple_backup_daemon.py`            |
| `gh release Lihfdgjr/synapforge:auto-<run>-<date>`            | `triple_backup_daemon.py`            |
| `hf://datasets/Lihfdgjr/synapforge-ckpts/<run>/`              | `triple_backup_daemon.py`            |

`sync_to_mohuanfang.sh` is intentionally redundant with the python daemon —
simpler, fewer dependencies, runs even if python is wedged.

Schedule the simple loop via cron:

```cron
# every 10 min, push runs/v24h_qwen to mohuanfang
*/10 * * * * /workspace/synapforge/scripts/sync_to_mohuanfang.sh --once \
    --run v24h_qwen >> /workspace/runs/sync.log 2>&1
```

Or run it as a foreground loop under tmux/screen for live tail.

---

## 9. Refresh cadence

| Data            | When to refresh                                              |
|-----------------|--------------------------------------------------------------|
| SFT JSONs       | Once per rental rebuild (rare; data is static)               |
| Multimodal      | Once per rental rebuild + if `manifest.json` reports failed  |
| Eval data       | Once per rental rebuild (datasets are versioned releases)    |
| FineWeb shard   | Per-trainer config; one shard is enough for our run length   |
| Backups         | Continuous (every 10 min via cron)                           |

---

## 10. Commercial readiness

Short answer: **not yet**. SFT is CC-BY-NC, multimodal is research-only,
LAMBADA is research-only. To ship a commercial model we would need to:

1. Re-train SFT step on commercially licensed data (e.g. Wizardcoder-CC,
   OpenHermes commercial subset).
2. Replace CC12M with a licensed image dataset (e.g. CC + paid synthetic).
3. Drop LAMBADA from the bench harness.
4. Audit FineWeb-edu attribution chain.

Alpaca-style instructions, MetaMathQA, MBPP, GSM8K, MMLU, HumanEval are
the lowest-friction parts (most are MIT or CC-BY).
