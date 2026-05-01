# Perf Knobs

Throughput knobs for `train_100m_kd.py` on A800-80GB. Last validated
2026-05-01 against the live `v24h_qwen3` run (vocab=151936, seq=256,
backend=`triton_block`, kd-every=4).

## Summary table

### Batch size sweep

| `--batch-size` | tok/s | GPU mem | OOM risk | Notes |
|----------------|-------|---------|----------|-------|
| 32             | ~12k  | ~52 GiB | none     | Legacy default; 1.7x headroom wasted. |
| 64             | ~21k  | ~72 GiB | low      | Prior cap (full-vocab z-loss = 9.5 GiB intermediate). |
| **80**         | ~26k  | ~76 GiB | low w/ sparse z-loss | **Recommended.** Sparse z-loss top-K 2048 shrinks the intermediate from 11.9 GiB to ~3 GiB, which is what unlocks this row. |
| 96             | OOM   | --      | high     | KD chunked softmax stays full-vocab; activation peak crosses 80 GiB mid-step. |

### KD frequency

| `--kd-every` | KD signal | Throughput cost | Notes |
|--------------|-----------|-----------------|-------|
| 1            | maximal   | 1.0x baseline   | Teacher forward dominates every step; ~0.5 KD-step / 0.35 LM-step ratio means ~2.4x slowdown vs no-KD. |
| 4            | strong    | ~3x faster than every-step | **Recommended.** KD on 25% of steps keeps teacher signal alive while LM gradient runs at ~LM-only speed on the other 75%. |
| 6            | moderate  | ~3.5x | Slight ppl regression on small-data runs; use when GPU is the bottleneck. |
| 8            | weak      | ~3.7x | Risk of teacher signal "fading" over long horizons; only for very long runs. |

### Sparse z-loss top-K

| `--z-loss-topk` | Memory (B*T,V,fp32) at bs=80 / V=151936 / seq=256 | Numerical accuracy | Notes |
|-----------------|---------------------------------------------------|--------------------|-------|
| 0 (full)        | 11.9 GiB | exact         | Disables sparse path; falls back to `torch.logsumexp(logits, dim=-1)` over the full vocab. |
| 8192            | 0.64 GiB | mean abs diff < 1e-6 nats | Overkill for V=151936; barely faster than 2048. |
| **2048**        | 0.16 GiB | mean abs diff < 1e-4 nats post-warmup; corr > 0.999 | **Recommended.** Captures >=99.99% of softmax mass at vocab=151936 once the model is past random init. |
| 512             | 0.04 GiB | mean abs diff ~5e-4 nats post-warmup | Aggressive; only safe when CE loss is < 3 nats (model is well-trained). May bias z-loss in early training. |

The z-loss term `(logsumexp(logits))**2` is a regularizer (PaLM/Gemma
style) that pulls the partition function toward 1. The top-K
approximation is a **strict lower bound** on the full logsumexp
(provable: `sum_{top-k} exp(x_i) <= sum_all exp(x_i)`), so the
underestimation is in the safe direction (less penalty pull).

The sparse approximation is mathematically rigorous post-warmup. If
you want the full-vocab path back for any reason (paranoia, paper
ablation), pass `--z-loss-topk 0`.

### KD async teacher stream

| `--kd-async-teacher` | Step time | Notes |
|----------------------|-----------|-------|
| off (default)        | 1.3s on KD steps  | Synchronous: student forward -> teacher forward -> KL -> backward. |
| on                   | 1.15-1.25s on KD steps | ~5-10% step win on bs=80. Pushes teacher forward onto a side CUDA stream so it overlaps with student backward (which still owns stream 0). Teacher is frozen + `eval()` so no autograd interaction. We sync before KL to guarantee correctness. |

### torch.compile

| `--torch-compile`     | Status         | Notes |
|-----------------------|----------------|-------|
| off (default)         | safe           | No compile overhead; the `triton_block` backend already fuses Liquid+PLIF. |
| reduce-overhead       | recommended for non-fused path | Wraps `model.forward` with `dynamic=True` so eval batches don't trigger recompile. Covers PLIFCell sequence loop + lm_head + FFN that triton_block doesn't touch. Best with `gpu_dense` backend. |
| max-autotune          | risky          | May NaN under bf16 on some GPU/driver combos. Trainer auto-rolls back to `off` on compile failure. |

CUDA graph mode is intentionally NOT exposed (would break with dynamic
seq_len in the eval-vs-train mismatch). `reduce-overhead` is the safer
pick.

## Recommended combo for A800-80GB

```bash
python3 -u train_100m_kd.py \
  --backend triton_block \
  --batch-size 80 \
  --z-loss-topk 2048 \
  --kd-every 4 \
  --torch-compile off \
  # grad_checkpoint stays off; bs=80 fits comfortably with sparse z-loss.
  # --kd-async-teacher  # opt-in once you've validated no NaN drift on your driver
  ...
```

Expected: **~26k tok/s** on the live trainer (up from the bs=64 baseline
of ~21k tok/s). Headroom for ~3 GiB more activation if you stack on
`--kd-async-teacher` (which uses ~0.5 GiB extra for the side stream's
teacher logits buffer during overlap).

## Why bs=96 OOMs

Activation peak at bs=96 is dominated by the KD chunked softmax path,
not the z-loss intermediate. `_kd_loss` chunks the (bs, seq, V) logits
and runs a `(chunk*seq, V, fp32)` softmax + KL per chunk; at chunk
auto-tune size 16 (the safe cap) and `seq=256, V=151936`, that's still
a ~2.5 GiB transient per chunk plus the resident model + optimizer
state. The KD activation floor sits around 12-14 GiB at bs=96, leaving
no room for the rest.

If you need bs=96+ in the future:
* Drop the teacher to a smaller vocab (GPT-2 50257) — KD chunked
  softmax becomes ~3.4x cheaper.
* Or enable `--grad-checkpoint` to trade compute for activation memory
  on the CfC sequence loop.

## See also

* `tests/integration/test_sparse_z_loss.py` -- numerical proof that
  top-2048 vs full logsumexp differs by <1e-4 nats at vocab 151k.
* `tests/integration/test_perf_knobs_compose.py` -- argparse smoke for
  the 4 new knobs.
* `tests/integration/test_kd_chunk_autotune.py` -- the existing KD
  chunk auto-tune (pre-bumped to use `mem_get_info` headroom budget).
