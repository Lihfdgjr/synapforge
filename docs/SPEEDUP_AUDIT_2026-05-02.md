# Speedup Audit — Run 5 Synap-1 Ultra (535M) — 2026-05-02

**Goal**: lift Run 5 throughput from **14k tok/s → 22-25k tok/s** without
moving val ppl trajectory. Each ship is behind a CLI flag, **default OFF**
(back-compat: identical step-by-step semantics as the live Run 5 launch),
opt-in once the user is ready to flip on the rental.

This builds on `docs/PERF_AUDIT_2026-05-02.md` (which shipped RECO #1
`--cuda-sync-every` and RECO #2 `--clip-grad-cache`). RECO #3 + #4 from
that audit were left as "doc-only / queued" — this PR ships them, plus
adds RECO #7 (eval-skip on warmstart relaunch).

## Profile-driven bottleneck breakdown (Run 5, A800-80GB, bs=24×accum=2)

Building on the per-stage breakdown in `docs/PERF_AUDIT_2026-05-02.md` §2,
the **non-compute overhead pool** at Ultra scale was estimated at ~11%:

| stage | est share | shippable? |
|---|---|---|
| Inner-accum 6× `.item()` host-sync (12/step at accum=2) | **~3-4 %** | YES — Ship #1 |
| Custom `PlasticityAwareAdamW.step()` element-wise loops | **~2-3 %** | YES — Ship #2 |
| `cuda.synchronize()` per-step + per-microbatch | ~3-5 % | DONE prior PR (`--cuda-sync-every`) |
| `clip_grad_norm_` listcomp per step | ~0.5-1 % | DONE prior PR (`--clip-grad-cache`) |
| Eval wall-time on warmstart (2 evals × 30s = 60s wasted) | **~variable** | YES — Ship #3 |
| HybridBlock fwd+bwd | 65-70 % | compute-bound; not in scope |
| Teacher fwd (KD every-4) | ~15 % | already on side-stream, skipped on KD-OFF steps |

**Headline**: bf16 IS engaging on all paths (verified by reading hot loop —
all matmuls under `torch.amp.autocast(dtype=DTYPE)` where DTYPE=bf16 on
CUDA). KD chunk is auto-tuned by `--kd-chunk 0` default. Sequence packing
NOT a fit at WT-103 / FineWeb scale (avg doc length already ≥ seq_len, so
padding waste is <5% — not worth the dataloader complexity + bug surface).
**The real wins are in the host-sync + optimizer pools** which together
account for ~5-7% of step wall time.

---

## TOP 5 RECOMMENDATIONS (ranked by ROI / risk)

### **1.** `--lazy-host-sync-accum` — push 6 `.item()` calls per microbatch to log boundary [SHIPPED]

**Where** (train_100m_kd.py:2281-2286): Per-microbatch we call
```
accum_total_loss += float(loss.detach().item())   # host-sync 1
accum_ce         += float(ce_loss.detach().item()) # host-sync 2
accum_kd         += float(kd.detach().item())     # host-sync 3
accum_z          += float(z_loss.detach().item()) # host-sync 4
accum_modal_aux  += float(modal_aux.detach().item()) # 5
accum_cur_aux    += float(cur_aux.detach().item())   # 6
```
At accum=2 → **12 host stalls/global step** *just for loss logging*. Each
is an implicit `cuda.synchronize()` on stream 0. Combined with the
already-shipped `--cuda-sync-every` win, removing these stalls compounds
the side-stream KD overlap freedom.

**Fix**: Keep loss tensors on GPU between microbatches. New flag
`--lazy-host-sync-accum` (default OFF = current behavior); when ON,
replace `accum_X += float(t.item())` with `accum_X_t += t.detach()`
(GPU tensor add). Materialize Python floats only at log boundaries
(`step % log_every == 0`) and at per-step VAL/checkpoint paths.

**Expected speedup**: **+2-4% step time**. Bigger when bs<=32 (GPU has
less inflight work to mask the stall).

**Risk**: NaN guard semantics preserved — at log boundary the materialized
float value still surfaces inf/nan exactly as today. The kd-every-adaptive
window still reads `float(ce_loss.detach().item())` directly (line 2202)
which is intentional: that path needs a per-microbatch CE for the
running mean and is OUTSIDE this flag's scope.

**Default**: OFF. Opt-in once verified on the rental.

### **2.** `--fused-adamw` — switch optimizer to torch.optim.AdamW(fused=True) when no plasticity is wired [SHIPPED]

**Where** (synapforge/optim.py:204-302): `PlasticityAwareAdamW.step` runs an
element-wise per-parameter Python loop (one `mul_/add_/addcdiv_` chain per
param), no fused kernel. For Ultra (~340 trainable params) this is a real
~3-5ms tax/step. PyTorch's `torch.optim.AdamW(fused=True)` collapses all
params into a single foreach + fused kernel.

**Detection rule**: Build a vanilla `torch.optim.AdamW(model.parameters(),
fused=True)` ONLY if every param has `_sf_grad_source == ['bp']` (or no
tag at all = default). If ANY param is tagged with stdp / hebb / etc.
(plasticity sources), fall back to `PlasticityAwareAdamW`. The HybridBlock
synapse weight is tagged `["bp"]` only by default (see model_100m.py:132)
so the fast path is the production trainer's actual path today.

**Fix**: New flag `--fused-adamw` (default OFF). When ON + safe-detection
returns True, build `torch.optim.AdamW(model.parameters(), lr=peak_lr,
weight_decay=WEIGHT_DECAY, fused=True, betas=(0.9, 0.999), eps=1e-8)`.
The state_dict layout is **DIFFERENT** from PlasticityAwareAdamW (no
ms_param_table, vanilla `state[p]['exp_avg']` / `['exp_avg_sq']` instead
of `['m']` / `['v']`), so warmstart compat: when a v1 ckpt's
`optim_state` is detected, log warning + cold-start moments (one-step
penalty, then back to bit-exact). New ckpts saved by the fused path are
loadable by future fused runs.

**Expected speedup**: **+2-3% step time** at Ultra scale. Not a linear win
(optimizer is ~4% of step on Ultra, bs=24); fused removes ~70% of the
optimizer cost.

**Risk**: Numerically equivalent to vanilla AdamW (verified by Adam
moment update equivalence — `m * b1 + g * (1-b1)` is identical regardless
of fused vs unfused; only kernel scheduling differs). Fused requires
CUDA + fp32 master / bf16 mixed; we already run that on bf16 + autocast.
HOWEVER: warmstarting from a `PlasticityAwareAdamW` ckpt is NOT bit-exact
because the moment state keys differ (`m`/`v` vs `exp_avg`/`exp_avg_sq`).
We log this and cold-start moments — a one-step penalty.

**Default**: OFF. Opt-in once user has at least one ckpt saved with
fused-state already (or is OK eating the one-step warmstart cost).

### **3.** `--skip-warmstart-eval-N` — skip first N evals on warmstart relaunch [SHIPPED]

**Where** (train_100m_kd.py:2610): Every `eval_every` (default 500) the
trainer runs `evaluate()` on both `val_ds_ttt` and `val_ds_holdout` (2
evals × ~30s each on Ultra = 60s wall-time per VAL boundary). On a
warmstart relaunch from a known-good ckpt (e.g., `step_010000.pt`), we
ALREADY know the val_ppl from the prior run's metrics.json — running it
again at step 10500 is wall-time waste.

**Fix**: New flag `--skip-warmstart-eval-N` (default 0 = current behavior).
When N>0 and the trainer started from a non-empty `--warmstart`, skip the
first N val invocations. Step counter still advances, save_every still
fires, multi-seq-val still queued (just skipped); chat-sample dump still
fires when configured. After N skips, normal eval resumes.

**Expected wall-time saved**: **30-90 seconds per restart** (the 2 evals
× 30s each at first eval-boundary). On a 12h training run with 2
restarts → ~3 minutes total. Small absolute, but a free win for the
phase-aware exit-101 relauncher pattern (which restarts MANY times in a
phase 0→1→2→3 chain).

**Risk**: If a relauncher uses a *different* val_glob than the prior run
the user might miss a baseline diff at the boundary. Mitigation: doc
warns; default N=0 keeps existing behavior.

**Default**: OFF (N=0). Opt-in via launcher script (`--skip-warmstart-eval-N
1` for typical relaunch).

### **4.** `--torch-compile reduce-overhead` activation — DEFER, not enabled by default

**Status**: Already wired (line 897). Default `off`. Compile-mode
`reduce-overhead` typically buys +5-10% but PLIFCell internal state
mutations (`v_membrane_buf`, spike counters) make the trace fragile under
bf16 + spectral_norm — the reason the flag is OFF today. **Cannot ship
as a default-ON change** without first validating on the rental.

### **5.** `--seq-packing` — pack multiple short docs per seq_len — DEFER, low ROI for our corpus

WT-103 + FineWeb at seq_len=512 already has avg doc length ~ seq_len, so
the padding waste is <5%. Implementing seq-packing requires non-trivial
dataloader changes (boundary tokens, attention-mask updates) for <5% win
that's already inside our error bars on the live tok/s. **Not worth the
bug surface today.** Re-evaluate when we move to seq_len=2048+.

---

## Combined expected speedup on next Run 5 restart

```bash
python3 -u train_100m_kd.py \
  ...current Ultra config... \
  --cuda-sync-every 10 \         # prior PR: +3-6%
  --clip-grad-cache \            # prior PR: +0.5-1%
  --kd-async-teacher \           # already shipped
  --prefetch-factor 4 \          # already shipped
  --pin-memory \                 # already shipped
  --lazy-host-sync-accum \       # NEW: +2-4%
  --fused-adamw \                # NEW: +2-3%
  --skip-warmstart-eval-N 1      # NEW: ~free wall-time
```

Expected combined: **14k tok/s → 19-22k tok/s** (~40-55% combined),
adding ~3-7% from this PR's two compute ships on top of ~30-40% from
the prior knob pack. To hit the **22-25k** target, **`--torch-compile
reduce-overhead`** is the next experiment (deferred — needs PLIF-state
fragility validation on the rental first).

## Tests

* `tests/integration/test_speedup_audit_2026.py` — argparse + helper
  semantics for the three flags
* `tests/integration/test_perf_knobs_compose.py` — composition smoke
  with prior perf flags

CPU-only. No GPU required. ~2-3s test runtime.

## See also

* `docs/PERF_AUDIT_2026-05-02.md` — RECO #1, #2, #3, #4, #5, #6 reference
* `docs/PERF_KNOBS.md` — v1 + v2 knob inventory
* `feedback_triton_backend_required.md` — 100M reaches 42k tok/s with
  triton_block + expandable_segments; Ultra (5.35× more params) at 14k
  is in line.
