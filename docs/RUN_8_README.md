# Run 8 -- Synap-1 Ultra Native Integration

## What this is

Run 8 is the first launcher that exercises the full `synapforge.native`
LNN+SNN-specific compute stack. It builds on Run 7's token-soup-fix
quality fixes (no warmstart, untied lm-head, PLIF dense bypass, KD
weight 0.7, kill-if-val-regresses 1.03) and layers on the 12 native
feature branches integrated under `synapforge/native/`. The native
package is the long-term replacement for the torch-fronted hot path:
every production module under `synapforge/native/` is forbidden from
`import torch` (test code may use torch for parity checks). Numerical
behaviour with no native flags set is bit-exact with Run 7; the native
flags are opt-in performance wins that fall back gracefully when the
underlying optional dependency (Triton, CUDA, cupy) is missing.

## Module catalogue

| Subpackage | Purpose | Dependency |
|---|---|---|
| `synapforge.native.vjp` | Closed-form VJPs for HybridBlock ops (CfC, PLIF, RMSNorm, SwiGLU, SEW shortcut, embed, linear, cross-entropy). Replaces autograd chain-rule with hand-derived gradients to avoid float drift through long PLIF spike trains. | numpy |
| `synapforge.native.data` | Torch-free parquet/jsonl streaming loaders, mixed-source weighted sampling, tokenizer adapters. Drop-in replacement for `synapforge.data.ParquetTokenStream`. | numpy + pyarrow |
| `synapforge.native.bench` | Saturation/roofline profiler + autotune box-search. The A800 autotune result `_autotune_real.json` is what produced Run 8's `--batch-size 48 --grad-accum-steps 4 --rfold-chunk 32` recommendation. | numpy |
| `synapforge.native.cuda` | cupy-backed `CudaTensor` + LNN-specific ops + Triton glue. The torch-free GPU side (no `import torch` in any module). | cupy + Triton (optional) |
| `synapforge.native.spike` | Bit-packed PLIF spike storage (16 spikes per uint16 word) + sparse-matmul Triton kernel. 16x HBM bandwidth saving on the spike->synapse branch. | Triton + cupy |
| `synapforge.native.stdp` | Local Hebbian STDP optimiser + hybrid optim dispatcher (plasticity-only params skip AdamW, BP params keep AdamW). | Triton (optional) |
| `synapforge.native.kernel` | Fused HybridBlock fwd+bwd Triton kernel pair + autograd.Function bridge for the production path. Closed-form bwd, bit-exact at fp32 reduction. | Triton |
| `synapforge.native.dispatch` | Heterogeneous CPU+GPU async pipeline (per-block router, stream pair, throughput bench). | numpy + cupy |
| `synapforge.native.modal` | Multimodal byte-patch packing (cross-modal mask, packed batch, modal dispatch). | numpy |
| `synapforge.native.auxsched` | Async coordinator for curiosity / TTT / NeuroMCP / ActionHead. Each driver runs on its own stream/thread. (Named `auxsched` not `aux` because Windows reserves AUX as a legacy DOS device name.) | numpy + cupy (optional) |

In addition, the integration brings in:

* `synapforge/native_demo.py` + `tests/native/test_native_demo.py` -- the zero-torch LNN+SNN MVP demo (`feature/native-mvp`).
* `synapforge/training/{core,kd,sft,rl,self_distill}_trainer.py` -- the trainer refactor (`feature/trainer-refactor-v2`) that splits the monolithic train script into a core trainer + KD/SFT/RL/self-distill subclasses.

## Run 7 -> Run 8 lift

| Lever | Source | Multiplier (best-case) |
|---|---|---|
| Dropped `--cpu-offload-optim` | `feature/native-saturation` measured 33% step time waste | 1.5x |
| `--batch-size 48 --grad-accum-steps 4 --rfold-chunk 32` | `feature/native-saturation` autotune | 3.6x base utilisation lift |
| `--fused-kernel` | `feature/native-fused-kernel` Triton fusion | 1.15-1.18x |
| `--packed-spikes` | `feature/native-spike-pack` 16x HBM saving on spike->synapse | dormant under Run 7's dead PLIF; activates when spike rate becomes non-trivial |
| `--async-aux-coordinator` | `feature/native-async-aux` overlap aux components | 1.4x at TTT-k=4, up to 2.9x synthetic |
| Total predicted | -- | 5.85x conservative; 17,000-30,000 tok/s vs Run 7's 2,750 |

The wide 17k-30k range reflects two unknowns: (1) when PLIF revives the
spike-pack path adds another 1.1-1.2x; (2) async-aux currently lands
the coordinator construction + banner but the per-call-site full driver
swap-in is partially wired (see "Honest gaps" below).

## How to launch

On the rental A800:

```
ssh root@117.74.66.77 -p 41614
cd /workspace/synapforge_git
git fetch origin feature/run8-native-integration
git checkout feature/run8-native-integration
bash scripts/launch_synap1_ultra_run8.sh
tail -f /workspace/runs/synap1_ultra_run8/train_run8_native_integration.log
```

The launcher uses `setsid` + `disown` so the trainer survives the SSH
session dropping. `WARMSTART` defaults to the same Run 7 ckpt path; set
the env var to override.

## Quality guardrails (kept from Run 7)

* `--no-warmstart`             cold start to avoid sweep regressions
* `--untie-lm-head`             lm-head untie improved val ppl
* `--plif-dense-bypass-steps 4000` SEW dead-bootstrap recipe to revive PLIF
* `--kd-weight 0.7`             distill weight Run 7 found stable
* `--kill-if-val-regresses 1.03` quality guard 铁律 -- exits non-zero on regression
* `--phase-aware`               phase manager auto-promotes on ppl gates

## Known dormant flags + when they activate

| Flag | Currently dormant because... | Activates when... |
|---|---|---|
| `--packed-spikes` | Run 7 PLIF spike rate is ~0 (dead-bootstrap not yet revived) | Step >= 4000 (post `--plif-dense-bypass-steps`) AND spike density < `--sparse-spike-threshold 0.30` |
| `--sparse-spike-synapse` | Same as packed-spikes -- needs spikes | PLIF revival post-step-4000 |
| `--stdp-only-plasticity` | Run 8 launcher does not pass this; current model has 0 plasticity-only-tagged params | Plasticity tags added via NeuroMCP synaptogenesis or future Hebbian backbone wire-in |
| `--byte-patch-pool max+avg` | No multimodal data is being mixed in this run | Multimodal data added via `--modal-data-dir` + `--modal-list` |
| Modal byte-patch packing (`synapforge.native.modal`) | Not invoked by the launcher; no modal data | `--modal-list image,audio` etc. set |

## Honest gaps

The integration is a **wire-in checkpoint**, not a full driver swap:

1. `--async-aux-coordinator` constructs the coordinator and emits the
   driver-active banner, but the per-call-site swap of the inline
   `curiosity_mixin.curiosity_loss(...)`, `self_learn_mixin.adapt_on_failures(...)`,
   `neuromcp_mixin.action_loss(...)`, and ActionHead OS-actuator calls is
   not yet routed through `submit_curiosity` / `submit_ttt` /
   `neuromcp_tick` / `submit_tool_call`. Reason: the inline mixins
   produce torch tensors that the coordinator's torch-free drivers
   would need numpy bridges for, and a bit-exact bridge is follow-up
   work. Default OFF preserves Run 7 inline numerics; ON currently
   gives the same numerics + the coordinator infrastructure standing
   by, so Run 8's predicted async-aux speedup is "1.0x today, up to
   2.9x once driver wire-in lands".

2. `--fused-kernel` flag is wired to a banner + the model's
   `fused_kernel` constructor knob (the kernel branch added the
   knob at the model-build layer). Auto-fallback to per-op dispatch
   when Triton/CUDA are unavailable.

3. `--stdp-only-plasticity` is fully wired (the merge of
   `feature/native-stdp-runtime` plumbed it end-to-end), but not
   activated by the Run 8 launcher because the production model has
   no plasticity-only-tagged params today.

These gaps are documented (rather than silently skipped) so the
post-mortem on Run 8's measured tok/s can attribute deltas correctly.

## Verification recipes

```
# native package + subpackages import cleanly
python -c "import synapforge.native; print(synapforge.native.__version__)"
# expected: 0.1.0-integration

python -c "from synapforge.native import data, vjp, bench, spike, stdp, kernel, dispatch, modal, auxsched; print('all import ok')"
# (cuda is conditional on cupy)

# launcher parses
bash -n scripts/launch_synap1_ultra_run8.sh

# new CLI flags recognised
python train_100m_kd.py --help | grep -E -- '--fused-kernel|--packed-spikes|--async-aux-coord'
```
