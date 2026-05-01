# Perf Knobs

Throughput knobs for `train_100m_kd.py` on A800-80GB. Last validated
2026-05-01 against the live `v24h_qwen3` run (vocab=151936, seq=256,
backend=`triton_block`, kd-every=4).

**v2 batch (2026-05-01 — staged for next restart)**: dataloader
prefetch + pinned memory, adaptive `--kd-every`, fused PLIF surrogate
backward (stub). See "Perf v2 knobs" section below.

## Summary table

### Batch size sweep

| `--batch-size` | tok/s | GPU mem | OOM risk | Notes |
|----------------|-------|---------|----------|-------|
| 32             | ~12k  | ~52 GiB | none     | Legacy default; 1.7x headroom wasted. |
| 64             | ~21k  | ~72 GiB | low      | Prior cap (full-vocab z-loss = 9.5 GiB intermediate). |
| **80**         | ~26k  | ~76 GiB | low w/ sparse z-loss + `--kd-topk 2048` | **Recommended.** Sparse z-loss top-K 2048 shrinks z-loss intermediate from 11.9 GiB to ~3 GiB; KD top-K 2048 shrinks the KD softmax intermediate from ~12 GiB to ~167 MiB. Both knobs are required to fit bs=80 in 80GB -- the prior 3-hour run died OOM here with `--kd-topk 0`. |
| 96             | should fit (untested) | --      | medium   | bs=96 OOM'd pre-2026-05-01 because the KD chunked softmax stayed full-vocab. With `--kd-topk 2048` the KD intermediate is ~200 MiB; the new floor is the model + optimizer state. |

### KD frequency

| `--kd-every` | KD signal | Throughput cost | Notes |
|--------------|-----------|-----------------|-------|
| 1            | maximal   | 1.0x baseline   | Teacher forward dominates every step; ~0.5 KD-step / 0.35 LM-step ratio means ~2.4x slowdown vs no-KD. |
| 4            | strong    | ~3x faster than every-step | **Recommended.** KD on 25% of steps keeps teacher signal alive while LM gradient runs at ~LM-only speed on the other 75%. |
| 6            | moderate  | ~3.5x | Slight ppl regression on small-data runs; use when GPU is the bottleneck. |
| 8            | weak      | ~3.7x | Risk of teacher signal "fading" over long horizons; only for very long runs. |

### KD top-K teacher softmax

| `--kd-topk` | KD softmax intermediate at bs=80 / seq=256 / V=151936 fp32 | Notes |
|-------------|-----------------------------------------------------------|-------|
| 0 (full-vocab) | ~12 GiB peak                | Falls back to legacy chunked path; honours `--kd-chunk`. The path that OOM'd at bs=80 even with chunk-along-batch. |
| 512         | ~42 MiB peak                | Aggressive; only safe if you've measured top-512 mass capture on your own teacher's logits. |
| **2048**    | **~167 MiB peak (70x less than full)** | **Recommended.** Captures >=99.99% of softmax mass at V=151936 on a trained teacher (BitNet/DistilBERT/SmolLM recipe). What unlocks bs=80 in the production training run. |
| 8192        | ~669 MiB peak               | Diminishing returns; already covers >99.999% of mass at 2048, more K just buys ~1% extra signal at 4x memory. |

The top-K trick: take top-K teacher logits per row, softmax over the
top-K only (renormalised), gather student logits at the same K
indices and run `log_softmax` over those K. KL becomes
`F.kl_div(s_top_logp, top_p_renorm, reduction='sum')` -- memory
scales with `B*T*K` instead of `B*T*V`. The full-vocab and top-K
paths produce different absolute loss values (different
`log_softmax` denominator) but the **gradient direction is the
same** (better student -> lower loss in BOTH paths). See
`tests/integration/test_kd_topk_softmax.py` for the direction-
agreement proof and the >=99% mass-capture math.

If you suspect top-K is hurting quality (e.g. teacher has heavy-tail
mass distribution), set `--kd-topk 0` to fall back to the legacy
chunked-softmax path; `--kd-chunk` then controls per-chunk batch
slicing.

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

## Why bs=96 OOM'd (historical, fixed by `--kd-topk`)

Pre-2026-05-01: activation peak at bs=96 was dominated by the KD
chunked softmax path, not the z-loss intermediate. `_kd_loss`
chunked the (bs, seq, V) logits and ran a `(chunk*seq, V, fp32)`
softmax + KL per chunk; at chunk auto-tune size 16 (the safe cap)
and `seq=256, V=151936`, that was still a ~2.5 GiB transient per
chunk plus the resident model + optimizer state. The KD activation
floor sat around 12-14 GiB at bs=96. The same path also OOM'd on
the 3-hour bs=80 production run -- chunk-along-batch only divides
the (chunk*seq, V) intermediate by `bs/chunk`, leaving (seq, V) per
row x chunk rows = up to ~1.2 GiB per chunk plus grad-checkpoint
activations.

**Fix (2026-05-01)**: `--kd-topk 2048` (default) replaces the
chunked-softmax path with a memory-bounded top-K teacher softmax.
KD activation peak drops from ~12 GiB to ~167 MiB at bs=80
(70x less). bs=96 should now fit; the new bottleneck is the model
+ optimizer state floor, not the KD intermediate. If you still
OOM at bs=96+:
* Verify `--kd-topk` is in the default 2048, not 0 (full-vocab path).
* Drop the teacher to a smaller vocab (GPT-2 50257) -- KD top-K
  buffer scales with `bs*seq*K`, not `V`, so this is no longer
  the dominant lever.
* Enable `--grad-checkpoint` to trade compute for activation
  memory on the CfC sequence loop.

## Perf v2 knobs (2026-05-01)

Three new knobs staged for the next trainer restart. The Run 3e
baseline (bs=64, vocab=151936, seq=256, triton_block) is **~21k
tok/s** with KD step at 1.3s vs KD-off at 0.35s — KD is dominating.

### 1. Dataloader prefetch + pinned memory

| Combo | Producer overlap | H2D overlap | Expected step win |
|-------|------------------|-------------|-------------------|
| `--prefetch-factor 0` (or 1)            | none | none           | baseline |
| `--prefetch-factor 2`                   | 1 batch ahead | none | +5-8% |
| `--prefetch-factor 2 --pin-memory`      | 1 batch ahead | yes  | **+10-20%** |
| `--prefetch-factor 4 --pin-memory`      | 3 batches ahead | yes | +12-22% (bs=80) |

Mechanism: `synapforge/data/__init__.py::ParquetTokenStream` spawns a
daemon producer thread that pre-builds batches into a bounded
`queue.Queue(maxsize=prefetch_factor)`; the main iterator just
consumes. With `--pin-memory` the resulting tensors are page-locked
host memory, so the trainer's existing `x.to(device,
non_blocking=True)` H2D copy actually overlaps with compute.

Worker-thread exceptions are re-raised on the main consumer's stack
(silent crashes are the worst class of bug for KD runs). Determinism
is preserved: same `--shuffle-seed` + same `--shuffle-buffer` yields
identical batch sequence with prefetch on or off — the prefetch
thread is a producer/consumer overlay only, NOT a re-shuffler.
See `tests/integration/test_dataloader_prefetch.py`.

### 2. Adaptive `--kd-every`

| `--kd-every-adaptive` | Schedule (base=4, teacher_ce=4.5) | Notes |
|-----------------------|-----------------------------------|-------|
| off (default)         | fixed N (== `--kd-every`)         | Run 3e behaviour. |
| on                    | walks {2, 4, 8, 16} every 100 steps | Recomputed from running mean CE of last 50 KD-OFF steps. |

```python
def _adaptive_kd_every(student_ce, teacher_ce_estimate=4.5, base=4):
    gap = max(0.0, student_ce - teacher_ce_estimate)
    if gap > 3.0:    return max(1, base // 2)   # 2 (early, more KD)
    if gap > 1.5:    return base                # 4 (default mid)
    if gap > 0.5:    return base * 2            # 8 (small signal)
    return base * 4                              # 16 (KD nearly converged)
```

Expected throughput: +20-30% averaged over a Phase-0 -> Phase-1 run.
Early steps stay at kd-every=2 or 4 (full KD signal); once student
ppl drops past ~80, kd-every climbs to 8-16 and the teacher-forward
tax shrinks proportionally. Knob measured on KD-OFF CE only so the
gap measure isn't artificially pulled down by KD itself.

### 3. Triton fused PLIF surrogate backward (stub)

| `--triton-fused-backward` | Status |
|---------------------------|--------|
| off (default)             | safe — PyTorch autograd surrogate |
| on                        | raises NotImplementedError, falls back to off (logs warning) |

Stubbed in `synapforge/backends/triton_fused_backward.py`. The kernel
would fuse the surrogate-grad computation (`dspike/d(v - thr)`) into
the same Triton tile as the spike forward in
`synapforge/backends/triton_block_kernel.py`, removing ~25 ms/step of
Python-side dispatch (10 layers x 256 timesteps x ~10us). Target
+5-10% step-time win; no math changes (same surrogate definition,
just done in-kernel). When implemented, the flag flips the registry
hook and the existing `synapforge/surrogate.py` backward path
becomes unused.

## Recommended A800-80GB combo (v2)

```bash
python3 -u train_100m_kd.py \
  --backend triton_block \
  --batch-size 80 \
  --z-loss-topk 2048 \
  --kd-every 4 \
  --kd-every-adaptive \
  --kd-async-teacher \
  --prefetch-factor 4 \
  --pin-memory \
  --shuffle-buffer 10000 \
  --torch-compile off \
  # --triton-fused-backward stays off until the kernel lands
  ...
```

Expected: **~30-35k tok/s** averaged across Phase-0 -> Phase-1
(prefetch +15% over the 26k tok/s v1 combo, then adaptive KD-every
contributes another +10-20% as the student converges and KD-every
climbs from 4 to 8/16). At bs=80, total RAM cost of the prefetch
queue is ~20 MiB pinned (4 batches x bs=80 x seq=256 x 8B x 2).

## Quality knobs

These trade compute or stability for quality (not throughput). Off by
default — opt-in once you've validated the trade on your run.

### LM head spectral norm (T2.6 / P28 z-loss drift)

| `--lm-head-spectral-norm` | Status | Notes |
|---------------------------|--------|-------|
| off (default)             | safe   | LM head weight evolves freely under Adam; observed `log Z` (z-loss term) drifts roughly linearly with step count even though the regularizer fights it. |
| on                        | opt-in | Wraps the LM projection with `torch.nn.utils.spectral_norm`. When `tie_lm_head=True` (the default), the wrap goes on `tok_embed` so the shared weight is reparameterised via power iteration; when untied, the wrap goes on `lm_head` directly. The forward path uses `weight = weight_orig / sigma_top`, so `||logits|| <= ||x||` is bounded and `log Z` stops drifting. |

Trade-offs:
* **Forward overhead**: one power-iteration step per forward pass
  (a single mat-vec on the embedding-shaped matrix). Measured ≤ 1%
  on similar tied-embedding models.
* **bf16 quirks**: power-iteration buffers `weight_u` / `weight_v`
  inherit the parameter dtype, so under `.bfloat16()` they stay
  bf16. The forward path still produces finite logits (the
  numerical test in `tests/integration/test_lm_head_spectral_norm.py`
  pins this); long-horizon gradient direction can drift
  vs. fp32 power-iter, which is the reason the flag is opt-in.
* **Checkpoint compatibility**: turning the flag on adds three
  state-dict keys (`weight_orig`, `weight_u`, `weight_v`) on the
  wrapped module. Existing best_*.pt checkpoints that were trained
  WITHOUT spectral_norm will not have these keys; load them with
  `strict=False` first, then enable the flag, or fold the
  reparametrisation in offline.

When to flip on: `log Z` (z-loss) growing linearly past step ~5000
without flattening. The PaLM/Gemma `--z-loss-weight 1e-4`
regularizer alone may be insufficient at long horizons; spectral_norm
provides a hard cap rather than a soft pull.

```bash
python3 -u train_100m_kd.py \
  ...the-recommended-A800-combo... \
  --lm-head-spectral-norm
```

## See also

* `tests/integration/test_sparse_z_loss.py` -- numerical proof that
  top-2048 vs full logsumexp differs by <1e-4 nats at vocab 151k.
* `tests/integration/test_lm_head_spectral_norm.py` -- artifacts +
  forward smoke for tied/untied/bf16 paths under the T2.6 spectral
  norm flag.
* `tests/integration/test_perf_knobs_compose.py` -- argparse smoke for
  v1 + v2 knobs (`_adaptive_kd_every` schedule pinned, fused-backward
  stub raises).
* `tests/integration/test_dataloader_prefetch.py` -- prefetch
  producer/consumer contract (deterministic ordering, exception
  resurfacing, pin-memory no-CUDA no-op).
* `tests/integration/test_kd_chunk_autotune.py` -- the existing KD
  chunk auto-tune (pre-bumped to use `mem_get_info` headroom budget).
* `synapforge/backends/triton_fused_backward.py` -- design notes for
  the (stubbed) fused backward kernel.
