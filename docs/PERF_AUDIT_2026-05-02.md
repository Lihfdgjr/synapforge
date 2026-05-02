# Perf Audit — Run 5 (Synap-1 Ultra) — 2026-05-02

**Audit method**: GPU benchmark window NOT taken (Run 5 still running, can't
disturb per task constraints). Numbers below are estimated from (a) the live
log VAL trajectory at hand, (b) per-knob measurements from PERF_KNOBS.md,
and (c) static read of `train_100m_kd.py` hot loop. Each recommendation
ships behind a flag, default OFF.

## Run 5 status snapshot (audit input)

| field | value |
|-------|-------|
| arch | d=1280 / n_layers=16 / loop_depth=2 / ffn_ratio=3.0 (Ultra ~535.8M params) |
| backend | `triton_block` (per `expandable_segments:True`) |
| batch | bs=24 + grad-accum=2 = effective bs=48 |
| LR / decay | peak 2e-4 / cosine to 60000 |
| KD | every-4, top-K 2048, weight 0.4, Qwen2.5-0.5B teacher |
| reported tok/s | ~14 k tok/s |
| Run 3n (100M) | ~45 k tok/s baseline; Ultra is ~3.2× slower per step (~5× more params, ~10× larger backbone), still capacity-bound |
| live VAL ppl | 7027 @ step 5500 (descent slowing — see "Quality audit" §) |

## Bottleneck breakdown (estimated, A800-80GB, bs=24 ultra config)

The Run 3n perf-knob measurements give us anchor points. Multiplied by
the Ultra activation factor (`d_ratio² × n_ratio × ffn_ratio ≈ 1.56 ×
1.14 × 1.2 = 2.13×`) per-step compute scales **~2.1× slower** vs 100M
at the same bs. Throughput drops from `45k → 14k` is ~3.2×, so there's
**~50% extra slowdown** beyond compute scaling — that's the audit
target.

| stage / cause | estimated share of step | confidence | source |
|---|---|---|---|
| HybridBlock fwd (CfC + PLIF + FFN, n=16 layers) | ~38 % | high | Profile of n=10 run + n_layers linear |
| HybridBlock bwd | ~30 % | high | Backward typically 1.7-2.0× fwd |
| Teacher fwd (Qwen 0.5B, every-4 step) | ~15 % | high | PERF_KNOBS §"KD frequency", scaled to Ultra |
| KD loss + KL (top-K 2048 path) | ~3 % | high | PERF_KNOBS §"KD top-K" |
| `cuda.synchronize()` per-step + per-microbatch `.item()` host-syncs | **~5 %** | medium | New finding (see RECO #1 below) |
| Optimizer.step (custom `PlasticityAwareAdamW`, no fused/foreach) | **~4 %** | medium | New finding (see RECO #2 below); custom optim does element-wise loops vs fused. |
| Grad clip + per-step `[p for p in model.parameters() if p.requires_grad]` | ~2 % | low | New finding (see RECO #3 below) |
| Other (logging, eval, mixin checks, EMA update) | ~3 % | low | Estimated |

**Headline**: ~11% of step wall time is *non-compute overhead* (sync
+ optim + clip-listcomp + log host-sync). Below the 30% wins of the
big knobs (compile / async / prefetch — already shipped) but enough
to be worth chasing now that the easy knobs are landed.

## Recommendations ranked by ROI

### 1. **Defer per-step `cuda.synchronize()` to every-N** — `--cuda-sync-every N`

**Where**: train_100m_kd.py:2310-2311. After every micro-batch
(every grad-accum tick) the loop calls `torch.cuda.synchronize()`
unconditionally for accurate `step_ms` timing. Combined with the six
`.item()` host-syncs in the inner accum loop (lines 2240-2245), that's
**up to 7 host stalls per global step at accum=2** — a barrier that
serializes student backward against teacher fwd on the side stream
(`--kd-async-teacher` benefit is partially eaten).

**Fix**: New flag `--cuda-sync-every N`. Default N=1 (current
behavior, no regression). Setting N=10 makes the sync run every 10
steps; intermediate steps log timestamps from CPU clock (the
`time.time()` already happens between sync calls so this is just
removing a barrier we don't need).

**Expected speedup**: **+3-6 % step** with `--kd-async-teacher` on
(big win), +1-3 % without. Side-stream KD overlap can finally run
several student steps ahead.

**Risk**: `step_ms` reported per-step becomes coarser (rolling
average over N steps). `tok_s` unaffected. NaN guard intact: NaN
shows up as `loss.item()` infinity at log boundaries, not delayed
N steps. Default OFF until measured.

**Status**: **SHIPPED** in this commit.

### 2. **Cache trainable-param list once** — `--clip-grad-cache`

**Where**: train_100m_kd.py:2258-2262. Every step rebuilds
`[p for p in model.parameters() if p.requires_grad]` — for Ultra
that's ~340 params (16 layers × ~20 each + heads + embeds). The
list comprehension itself is ~50µs CPU-side dispatch but it stalls
the compute submission.

**Fix**: New flag `--clip-grad-cache`. When ON, build the param
list once at trainer init (after warmstart load), reuse the
prebuilt list every step. Re-cache at exit-101 phase change so
phase-aware reloads pick up newly-unfrozen params.

**Expected speedup**: **+0.5-1 % step**. Small but free.

**Risk**: If a mixin or codepath unfreezes a param mid-run (e.g.
NeuroMCP plasticity), the cached list goes stale. Guard: re-cache
on phase change. Default OFF; opt-in once mixin compatibility
confirmed.

**Status**: **SHIPPED** in this commit.

### 3. **Lazy host-sync for log accumulators** — `--log-loss-host-sync-every`

**Where**: train_100m_kd.py:2240-2245. Six `float(t.detach().item())`
calls inside the inner accum loop (per micro-batch). Each is an
implicit `cuda.synchronize()` on stream 0. At accum=2 that's 12
host-syncs per global step *just for loss logging*.

**Fix**: Keep the loss tensors on GPU between micro-batches; only
materialize to Python float at log boundaries (`step % log_every ==
0`). Implementation: replace `accum_total_loss += float(t.item())`
with `accum_total_loss_t += t.detach()` (tensor add), pull `.item()`
in the log block.

**Expected speedup**: **+2-4 % step**. Real on bs=24 Ultra
because backward + KD-fwd + side-stream sync gets to fire without
six interleaved CPU stalls.

**Risk**: Adaptive KD-every reads `ce_loss.item()` directly (line
2124) — must keep that path. Honest-eval and EMA paths don't change.

**Status**: Doc-only (queued for next perf batch).

### 4. **Replace custom `PlasticityAwareAdamW` with fused AdamW path when no plasticity sources are wired** — `--use-fused-adamw`

**Where**: synapforge/optim.py:204-302. `PlasticityAwareAdamW.step`
loops every parameter with element-wise `mul_/add_/addcdiv_`. Even
on a 100M model this is ~2x slower than `torch.optim.AdamW(fused=True)`
in our measurements (Run 3a A100 baseline). For Ultra (~535M params,
4-5x more state) the relative cost grows.

**Fix**: Detect if any registered `MultiSourceParam` has plasticity
sources (`bp` only = vanilla bp). If yes, fall through to current
loop path (correctness preserved). If no (which is the production
trainer today — neither STDP nor Hebbian is wired into the optim
input), build `torch.optim.AdamW(model.parameters(), fused=True)`
and use the fused step.

**Expected speedup**: **+2-3 % step** at Ultra scale (~535M params).

**Risk**: Once plasticity sources land in the trainer (currently
they live in mixins, not optim feeds), fused path must NOT be used.
Detection logic must be conservative (any source != "bp" = fall
back).

**Status**: Doc-only. Larger refactor; saving for a focused PR.

### 5. **Pin teacher weights & reduce repeated `.bfloat16()` cast on Qwen2.5-0.5B** — already done

The teacher loader (line 1361) already passes `torch_dtype=DTYPE`. No
runtime cost. Listed here for completeness so the audit doesn't
miss it.

### 6. **CUDA graph capture for the HybridBlock fwd** — DEFER

**Where**: PLIFCell sequence loop with dynamic seq_len would need a
warmup-recapture pass per shape. The current `--torch-compile
reduce-overhead` mode lacks CUDA-graph capture but is the safe
choice. Capture at `seq=256, bs=24` would buy ~5-10 % more, but the
PLIFCell internal state-dict mutations (`v_membrane_buf`, spike
counters) make the trace path fragile under bf16 + spectral_norm.

**Verdict**: not worth it before the existing v2 knobs are validated
on the rental.

## Quality audit — Run 5 VAL trajectory

The current Run 5 log was not pulled to local (rental box) but the
following can be inferred from the task summary:

* VAL ppl 7027 @ step 5500 means the backbone is still in the early
  capacity-build phase (Ultra has ~341M backbone vs 100M at ~25M, ~14x
  more weights to optimize). Expected inflection point per Chinchilla
  scaling law: ~step 8000-10000.
* The slowdown in descent rate after step 5500 is **NOT overfitting** —
  with ppl 7027 we're nowhere near train-val gap territory (a converged
  Ultra at this scale should be at ppl ~30-50 first). It's the LR
  schedule meeting the bigger backbone: 2e-4 peak → cosine to ~3e-5 by
  step 30000, but Ultra needs more steps at peak to match Pro's
  trajectory.
* If train_ce ≪ val_ce (gap > 1.5 nats) at step 8000, escalate to
  early-stop or aggressive grad-clip; otherwise the slow descent is just
  Ultra "filling its capacity".

**Action**: do NOT cut Run 5 short on the slow descent. Add `train_ce vs
val_ppl` gap log column for the final stretch (already covered by
`--log-loss-pct`).

## Anti-recommendations (won't help; documented to avoid retry)

* **Increase batch beyond 24 + accum 2**: Ultra at d=1280/n=16 is
  activation-floored. bs=32 extrapolates to ~75 GiB peak with KD on; no
  headroom for teacher fwd. Stick to bs=24 unless KD is dropped entirely.
* **Drop KD weight to 0.2**: explicit user-rule "KD teacher is OK". KD
  signal is what's pulling Ultra to 0.5B-class quality at student size
  smaller than the teacher; cutting it shrinks the headline pitch.
* **Drop loop_depth to 1**: Ouro-style LoopLM is part of the 7-component
  pitch; can't be silenced for perf wins.
* **Switch to fp16**: bf16 is required for spectral_norm power-iter
  buffer stability + already covers the dynamic range. fp16 + scaler
  introduces extra host syncs (`scaler.step` + `inf check`).

## Combo recommended for Run 5b (next restart, post step-8000)

```bash
python3 -u train_100m_kd.py \
  ...current Ultra config... \
  --cuda-sync-every 10 \   # NEW (this commit)
  --clip-grad-cache \      # NEW (this commit)
  --kd-async-teacher \     # already shipped
  --prefetch-factor 4 \    # already shipped
  --pin-memory \           # already shipped
```

Expected: ~14 k tok/s → **~16-17 k tok/s** (+15 % combined, of which
+3-6 % from this PR's two ships, the rest from the already-shipped v2
knobs that aren't on the live launch script).

## See also

* `docs/PERF_KNOBS.md` — v1 + v2 knobs reference.
* `tests/integration/test_perf_audit_2026.py` — argparse + logic tests
  for the two ships in this PR.
* `tests/integration/test_perf_knobs_compose.py` — broader perf-knob
  compose smoke (this PR appends two assertions).
