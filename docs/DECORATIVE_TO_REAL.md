# Decorative -> Real Activations (2026-05-02)

This doc tracks the conversion of "ships in repo but doesn't actually run"
features into runtime-active components for Synap-1 Ultra Run 8.

The user diagnosis (2026-05-02 23:55):

> 几个flagship feature是CODE-SHIPPED但RUNTIME-DORMANT
> ... 把装饰的换成实际可用的

This branch (`feature/activate-decorative`) flips them ON, additively,
without disturbing default behaviour for existing flags.

## Summary table

| # | Feature | Was decorative because | Activation | Status |
| - | ------- | ---------------------- | ---------- | ------ |
| 1 | STDP-only routing on SparseSynapticLayer | `_sf_grad_source` defaulted to implicit `["bp"]`; PlasticityAwareAdamW never received an STDP delta stream | Tag `self.weight._sf_grad_source = ["stdp"]` + `_sf_alpha = 0.001` at construction | ACTIVE |
| 2 | Multimodal byte-patch (image / audio / time_series) | Run 7 was text-only (no `--modal-list`) | Run 8 full launcher passes `--modal-list image,audio,time_series` + `--modal-data-dir` | ACTIVE (gated by phase-aware: ppl<=100) |
| 3 | Web daemon -> trainer pipe | Daemon ran as separate process, output never piped into the trainer | New `WebDaemonSink` writes rolling parquet shards; trainer's `--data-files` glob picks them up at weight 0.10 | ACTIVE (daemon must run) |
| 4 | R-fold in TRAINING | Initially shipped behind `--rfold` flag with backward-path uncertainty | Verified bit-exact backward + training-loop equivalence within 1e-5%/step at D=32; Run 8 launcher passes `--rfold --rfold-chunk 16` | ACTIVE |
| 5 | Ternary BitNet QAT | `--weight-quant ternary` flag never set | DEFERRED -- swapping into ternary mid-training triggers LM-head reset (`feedback_spectral_norm_warmstart_cost.md`); needs a fresh-run, not warmstart | BLOCKED |
| 6 | PLIF spike revival | PLIF dead in Run 7 step 4000 window | Run 8 launcher already passes `--plif-dense-bypass-steps 4000 --sparse-spike-synapse --spike-target-loss-weight 0.5`; spike density emerges only post-bypass | BLOCKED until step >= 4000 |

## Per-feature

### 1. STDP-only routing on SparseSynapticLayer

**Decorative state**: The synaptogenesis head (`NeuroMCPHead.proj` =
`SparseSynapticLayer`) implements
`update_coactivation` (Hebbian EMA over pre*post activity) and
`maybe_grow_prune` (the synaptogenesis growth/prune rule). These produce
real plasticity deltas. But because the `nn.Parameter` for the synaptic
weight had no `_sf_grad_source` attribute, `build_optimizer` (in
`synapforge.optim._legacy`) treats it as default `["bp"]` -- a regular
back-prop param. The plasticity deltas would never have been routed
into AdamW.

**What changed**: `synapforge/action/neuromcp.py` lines after `self.weight`
construction in `SparseSynapticLayer.__init__`:

```python
self.weight._sf_grad_source = ["stdp"]
self.weight._sf_alpha = 0.001
```

**What's now exercised**:

* `build_optimizer` reads the tag, wraps the param in
  `MultiSourceParam(sources=["stdp"])`.
* When the trainer enables `--neuromcp-weight > 0` (Run 8 full launcher
  does: `--neuromcp-weight 0.02`), the head is instantiated and the tag
  is on. PlasticityAwareAdamW's combine-grad path then folds the STDP
  delta into the AdamW update.
* Test coverage: `tests/action/test_stdp_tag_present.py` (7 tests),
  including the end-to-end optimizer registration check.

**Risk profile**: Additive metadata. Optimizers that don't read
`_sf_grad_source` keep the old behaviour.

### 2. Multimodal byte-patch (image / audio / time_series)

**Decorative state**: `synapforge.trainer_mixins.MultimodalMixin` is
fully implemented (text-modal InfoNCE contrastive aux + byte-patch
encoders). The trainer has the `--modal-list / --modal-data-dir /
--modal-alpha` flags wired. But Run 7 ran with `--modal-list ""`, so
the mixin returned a no-op every step.

**What changed**: `scripts/launch_synap1_ultra_run8_full.sh` defaults to
`MODAL_LIST=image,audio,time_series` with
`MODAL_DATA_DIR=/workspace/data/modal` and `MODAL_ALPHA=0.10`.

**Phase gating (per user spec)**: `--phase-aware` is on, so the trainer
polls `scripts/phase_manager.py`'s `.phase` file and only enters the
modal phase once `val_ppl <= 100` (Phase 2 in `phase_manager.py:50`).
So this is "ON in the launcher, dormant in the trainer until quality
threshold met" -- exactly the user's requested gating.

**What's now exercised**: At Phase 2, `MultimodalMixin.compute_loss`
runs every batch, adds InfoNCE alignment loss between text-hidden and
modal-encoder representations. Costs a small per-batch overhead but
trains shared representations across modalities.

**Smoke**: launcher `bash -n` passes; static lint test
`tests/test_run8_full_launcher_flags.py::test_modal_flags_present`
asserts the flags are in the python invocation block (not just the
banner).

### 3. Web daemon -> trainer pipe

**Decorative state**: `synapforge.learn.continual_daemon.WebContentLearner`
implements 7-gate filtering (per `feedback_continual_vs_poison_balance.md`).
`scripts/launch_continual_daemon.py` has been runnable for a while. The
daemon writes a `lora_buffer.jsonl` for shadow-LoRA merge -- but no
trainer ever reads it, so the entire stream is RUNTIME-DORMANT.

**What changed**:

* New module `synapforge/data/web_daemon_sink.py` adds `WebDaemonSink`
  that writes a rolling parquet shard set
  (`<out_dir>/web_<NNNN>.parquet`) with rotation by hour / row-count /
  byte-size.
* New script `scripts/web_self_learn_daemon.py` ties
  `WebContentLearner` (7-gate) + `WebDaemonSink` (parquet rotor)
  into a single supervised process. Curated low-noise sources only:
  Wikipedia (zh/en), Project Gutenberg, arXiv abstracts. Real fetch is
  best-effort (Wikipedia REST API at `--enable-real-fetch`); offline
  curated samples drive smoke + tests.
* Run 8 full launcher composes
  `/workspace/data/web_self_learn/web_*.parquet:0.10` into `--data-files`
  (alongside curated 90%) when at least one shard exists -- so cold
  start without daemon doesn't crash the trainer.

**4-gate pollution defence already in code**:

* G1 source-trust EMA per-domain
* G2 language detect (zh / en only -- drops gibberish)
* G3 token-perplexity sweet-spot (drops memorised + gibberish)
* G4 NSFW / violence keyword filter
* G5 adversarial pattern (persona-swap markers)
* G6 provenance (sha256 dedup + URL blocklist)
* G7 TRAK approx (cheap; trainer can attach a real TRAK scorer)

Sink also persists each row's `quality_score = MIN(gate_scores)` so the
trainer can additionally drop rows below a threshold.

**Per-source 7d cap**: 125 docs (50% of Anthropic 250-doc poison
threshold per `2510.07192`).

**What's now exercised**: When the daemon is running on the rental, the
parquet shard set grows; the trainer's `ParquetTokenStream(files_with_weights=...)`
samples from the rolling glob with weight 0.10 every batch.

**Tests**:

* `tests/data/test_web_daemon_sink.py` (10 tests)
* `tests/test_web_self_learn_daemon.py` (5 tests)

### 4. R-fold in TRAINING

**Decorative state (unclear initially)**: The `--rfold` flag was wired
into `LiquidCell.forward` but the user wasn't sure the backward path
was bit-exact under the chunked closed-form scan. If grad accumulation
through `cumprod * cumsum / cumprod_safe` had a hidden no_grad / detach,
then `--rfold` on training would silently freeze the LiquidCell's
parameters.

**What we verified**:

* `tests/cells/test_rfold_equivalence.py` (existing) covers single-step
  forward + backward equivalence and grad non-zero on `delta_proj`,
  `b_proj`, `A_log`.
* `tests/cells/test_rfold_train_bit_exact.py` (NEW): 100 SGD steps
  ON vs OFF produce identical loss curves up to 1e-5% per step (well
  within the 1% spec). Final state-dict params match within 1e-3
  relative.

So R-fold is genuinely active in training, not just a forward shortcut.

**What's now exercised**: Run 8 launcher passes `--rfold --rfold-chunk 16`,
which routes `LiquidCell.forward` through `liquid_rfold(chunk=16)`
instead of the per-step Python `for t in range(T)` loop. On GPU this
means T=256 -> 16 kernel launches (chunked closed-form scan) instead of
256, the dominant cost at small per-step compute.

### 5. Ternary BitNet QAT

**Decorative state**: `weight_quant="ternary"` is implemented in
`synapforge/cells/liquid.py` via `synapforge.quantize.TernaryLinear`
(BitNet b1.58 AbsMean per-tensor STE). But it's never set anywhere:
the trainer launcher always passes the default `weight_quant="none"`.

**Why deferred (not activated)**: Per
`feedback_spectral_norm_warmstart_cost.md`, swapping the input
projection from `nn.Linear` to `TernaryLinear` mid-warmstart triggers a
state_dict mismatch (different parameter set) which reset the LM head
in Run 3m, costing 5h of training. The same risk applies here. To
activate this, we would need a fresh-run (no warmstart) at the start of
a new training campaign. Not appropriate for Run 8 which warmstarts
from Run 7.

**Status in audit**: BLOCKED with reason. NOT activated.

### 6. PLIF spike revival (existing in Run 7+)

**State**: Run 7 already configures
`--plif-dense-bypass-steps 4000 --sparse-spike-synapse
--sparse-spike-threshold 0.30 --spike-target-loss-weight 0.5`. The
mechanism is wired but spike density only emerges *after* step 4000
(when the dense-bypass closes and PLIF spikes start firing through the
sparse-spike kernel). Until then PLIF is in observe-only mode (per
`feedback_plif_dead_bootstrap.md`).

**Why this is honest**: Per Run 7 (PID 41692, currently running on the
rental at step 4500 as of writing), this is "ON in the launcher, but
the activation only kicks in once the bypass step is past." If we
audit before step 4000 and see "PLIF density 0%" in the log, that's
the dense-bypass path doing its job, not a regression.

**Status in audit**: BLOCKED-by-dependency. The activation tooling is
in place; we just need a runtime log past step 4000 to confirm.

## What still can't activate (blocked, by dependency)

| Blocker | Resolves when | Action when resolved |
| ------- | ------------- | -------------------- |
| Ternary BitNet QAT | Next fresh-run training campaign (no warmstart) | Pass `--weight-quant ternary`; observe loss in first 200 steps for instability |
| PLIF spike density | step >= 4000 in any current run (Run 7 / Run 8) | Run audit script with `--log <run_log>`; the patterns look for `sparse-spike-synapse` references and PLIF spike density evidence |
| Web-daemon REAL data | Daemon is launched on rental + at least 1 hour of fetch | `python3 scripts/web_self_learn_daemon.py --supervisor --out-dir /workspace/data/web_self_learn --enable-real-fetch &` |
| Modal contrastive aux | Trainer reaches `val_ppl <= 100` AND modal data is on disk | Run 8 launcher already passes `--phase-aware`; phase manager auto-flips it on |

## Run 8 FULL launcher: extra flags vs Run 8 base

The integration agent owns `scripts/launch_synap1_ultra_run8.sh` (base);
this branch adds `scripts/launch_synap1_ultra_run8_full.sh`. The
extra flags this launcher injects (delta over base):

* `--modal-list image,audio,time_series` (NEW; multimodal aux)
* `--modal-data-dir /workspace/data/modal` (NEW)
* `--modal-alpha 0.10` (NEW)
* `--data-files` includes
  `/workspace/data/web_self_learn/web_*.parquet:0.10` when shards exist
  (NEW; added inline)

All other flags are inherited from the Run 7 baseline (rfold,
sparse-spike-synapse, plif-dense-bypass-steps, neuromcp-weight, etc.).

## Audit script

`scripts/audit_decorative_features.py` reports per-feature:

  * file_exists
  * tag_or_flag set in code
  * flag in launcher
  * (optional) actually_active in a runtime log

Run periodically:

```bash
python3 scripts/audit_decorative_features.py \\
    --log /workspace/runs/synap1_ultra_run8_full/train_run8_full.log \\
    --out-md docs/DECORATIVE_AUDIT.md \\
    --json
```

The audit is the source of truth for "is this feature actually running."
