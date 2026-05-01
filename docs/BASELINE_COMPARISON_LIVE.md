# BASELINE_COMPARISON_LIVE — measured (not just published)

> Skeleton placeholder. Auto-overwritten by
> `scripts/baseline_smollm2_compare.py` when the rental-side runner
> populates the live numbers. Companion to
> [BASELINE_COMPARISON.md](BASELINE_COMPARISON.md) — that doc has the full
> 11-baseline citation table; THIS doc has the **measured numbers**.

## 1. SmolLM2-360M vs Synap-1 — perplexity + tok/s + energy

Wikitext-103 test split. Skeleton — populate via the harness.

| Model | Params | ppl (wt-103) | tok/s | FLOPs/tok | Notes |
|-------|-------:|-------------:|-------:|----------:|-------|
| **Synap-1 (us)** | 100M | TBD | TBD | TBD | spike_rate~=0.10; STDP fwd-only |
| **SmolLM2-360M** (measured) | 362M | TBD | TBD | TBD | dense xfmr; KV grows O(L) |
| SmolLM2-360M (published) | 362M | NR | NR | — | MMLU 30.4 / HellaSwag 53.0 / GSM8K 27.0 (model card) |

**Energy proxy**: Synap-1 / SmolLM2 FLOPs-per-token ratio = TBD.

Caveat: the ratio counts FLOPs only. **It does not exhibit on dense GPU
hardware** — the spike sparsity savings are only realized on a neuromorphic
accelerator that exploits sparse activations end-to-end. See
[BASELINE_COMPARISON.md §3](BASELINE_COMPARISON.md) for the full caveat.

## 2. Reproduction

### SmolLM2 leg (any host with `transformers` + HF cache):
```bash
python scripts/baseline_smollm2_compare.py \
  --n-samples 1000 \
  --output docs/BASELINE_COMPARISON_LIVE.md
```

### Synap-1 leg (rental-side, requires GPU + ckpt):
```bash
ssh root@rental 'cd /workspace/synapforge_git && \
  python3 scripts/baseline_smollm2_compare.py \
    --synap-ckpt /workspace/runs/v24h_qwen3/best_<step>.pt \
    --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
    --n-samples 1000 \
    --output docs/BASELINE_COMPARISON_LIVE.md'
```

## 3. Status

- [ ] Synap-1 measured (awaits rental run with healthy ckpt)
- [ ] SmolLM2 measured (awaits HF download — `HF_HUB_CACHE` must be set)
- [ ] alpaca-zh-eval leg (T6.7 follow-up; pulls Chinese chat eval set)

Once both legs land, this row replaces the SmolLM2-360M row in
[BASELINE_COMPARISON.md §1](BASELINE_COMPARISON.md) with measured numbers
(citation: this run + commit hash).
