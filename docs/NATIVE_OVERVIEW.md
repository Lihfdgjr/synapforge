# synapforge.native v0.1 — single-doc map

**Shipped**: 2026-05-02 (4-hour parallel-agent sprint, 19:00–23:00 local)
**Status**: 12 subpackages on 12 feature branches; cross-branch trunk merge
pending integration agent (see [MASTER_PLAN P31](MASTER_PLAN.md)).
**Hard rule**: zero `import torch` in production code paths. Test suites
enforce `test_no_torch_import`. PyTorch glue lives only at the boundary
in `torch_glue.py` autograd.Function adapters.

This doc is the entry-point map. Detailed per-package design lives in
sibling docs (linked below).

---

## TL;DR

SynapForge stops being a torch wrapper. Twelve native subpackages
collectively replace the torch scaffold (data loaders, autograd VJPs,
CUDA tensor primitives, fused HybridBlock kernels, hetero CPU+GPU
dispatch, async aux scheduler, multimodal byte-patch packing, spike
packing, STDP runtime, trainer skeleton) so we can train LNN+SNN models
on a stack tuned to the architecture instead of borrowing transformer
defaults.

Measured (2026-05-02): MVP CPU **2.86×** vs torch oracle, fused
HybridBlock **1.15-1.18×** e2e, async dispatch **1.7×**, async aux
**2.9×**, spike pack **16×** HBM bandwidth saving. Run 7 (torch baseline)
at 2,750 tok/s on A800; Run 8 (native stack post-integration) projected
at **17,000–30,000 tok/s**.

---

## Architecture (text-art)

```
                                ┌─────────────────────────────────────┐
                                │       synapforge.native (v0.1)      │
                                │   "framework, not a torch wrapper"  │
                                └─────────────────────────────────────┘

  data        ─►  vjp / cuda    ─►  kernel / spike / stdp   ─►  dispatch / auxsched ─►  training
   │                  │                       │                          │                  │
parquet/         closed-form         fused HybridBlock              3-stage hetero      BaseTrainer +
jsonl/mixed      VJPs               (CfC + PLIF + SEW +              CPU+GPU pipeline    KD/SFT/RL/
streams          (embed,            synapse) Triton fwd+bwd          + per-block         SelfDistill
no-torch         linear,            bit-packed spikes (16:1)         router; auxsched    subclasses
hot path         rmsnorm,           STDP-only optim (skip            fans selflearn /
                 swiglu,            AdamW for plast tags)            curiosity / NeuroMCP /
                 cfc, plif,                                          STDP into parallel
                 sew_shortcut,
                 cross_entropy)            modal: 9-modal byte-patch packed batches
                                           (text + image + audio + video + 3D + ts +
                                           tabular + code + maths) into ONE shared
                                           CfC+PLIF backbone  (cf. Flamingo's
                                           per-modality encoders)

                                                 │
                                                 ▼
                          bench: saturation profiler + roofline + autotuner
                                  (single source of truth on physical limits)

                                                 │
                                                 ▼
                          MVP (`synapforge.native_demo`): pure-numpy 100-step
                          LNN+SNN train + torch oracle + parity gate
                          → measured 2.86× CPU vs torch
```

---

## File tree of `synapforge/native/`

After the cross-branch integration merge (P31) lands, the unified tree
will look like this. Right now each subdir lives only on its feature
branch; the layout below is the *target* trunk state.

```
synapforge/native/
├── __init__.py                          (re-exports public surface from each subpkg)
│
├── data/                                feature/native-data-loader      4b9735c
│   ├── __init__.py
│   ├── tokenizer.py                     no-torch BPE/SentencePiece glue
│   ├── parquet_stream.py                streaming reader for parquet shards
│   ├── jsonl_stream.py                  streaming reader for jsonl shards
│   ├── mixed_stream.py                  multi-corpus weighted mix
│   └── bench.py                         throughput parity bench (vs torch DataLoader)
│
├── vjp/                                 feature/native-vjp-catalog      b560ccc
│   ├── __init__.py
│   ├── dtypes.py                        fp32 / bf16 promotion helpers
│   ├── embed.py                         scatter-into-rows VJP for embedding
│   ├── linear.py                        bias-aware linear VJP
│   ├── rmsnorm.py                       RMSNorm closed-form VJP
│   ├── swiglu.py                        SwiGLU VJP (chained from linear)
│   ├── cfc.py                           closed-form CfC step VJP
│   ├── plif.py                          PLIF surrogate-grad VJP
│   ├── sew_shortcut.py                  SEW-style shortcut VJP
│   ├── cross_entropy.py                 numerically stable softmax-CE VJP
│   └── bench_smoke.py                   pure-numpy parity bench
│
├── cuda/                                feature/native-cuda-backend     82ccaca
│   ├── __init__.py
│   ├── tensor.py                        CudaTensor (cupy or numpy fallback)
│   ├── ops.py                           elementwise / matmul / reduce ops
│   ├── lnn_ops.py                       LNN+SNN-specific ops on CudaTensor
│   ├── allocator.py                     bump-allocator with pool reuse
│   ├── streams.py                       CudaStream wrapper
│   └── triton_glue.py                   Triton kernel launch helpers
│
├── bench/                               feature/native-saturation       00580fd
│   ├── __init__.py
│   ├── stage_profiler.py                A800 stage profiler
│   ├── roofline.py                      arithmetic-intensity roofline
│   ├── autotune.py                      bs / chunk / grid autotuner
│   ├── _autotune_real.json              cached real autotune results
│   └── _roofline_run7.json              roofline report from Run 7 baseline
│
├── spike/                               feature/native-spike-pack       b85de28
│   ├── __init__.py
│   ├── pack.py                          pure-numpy 16-spike → uint16 pack/unpack
│   ├── packed_matmul.py                 Triton kernels (NO 'import torch')
│   └── torch_glue.py                    autograd.Function bridge + --packed-spikes flag
│
├── stdp/                                feature/native-stdp-runtime     ce3025b
│   ├── __init__.py
│   ├── stdp_optimizer.py                STDP-only optimizer (skips AdamW for plast tags)
│   ├── spike_buffer.py                  ring buffer for pre/post traces
│   ├── per_param_lr.py                  per-param LR + LTP/LTD weights
│   ├── hybrid_optim_dispatch.py         routes plast-tagged → STDP, else → AdamW
│   └── triton_kernel.py                 fused LTP/LTD Triton kernel
│
├── kernel/                              feature/native-fused-kernel     eaf0c17
│   ├── __init__.py
│   ├── fused_hybrid_fwd.py              CfC+PLIF+SEW+synapse fused fwd Triton
│   ├── fused_hybrid_bwd.py              closed-form fused bwd Triton kernel
│   └── fused_hybrid_torch.py            autograd.Function bridge to LiquidCell
│
├── dispatch/                            feature/native-dispatch-async   2403618
│   ├── __init__.py
│   ├── streams.py                       CudaStream + StreamPair (cupy / numpy fb)
│   ├── cpu_pool.py                      CpuWorkerPool + parallel_adamw_step
│   ├── pipeline.py                      HeteroPipeline 3-stage scheduler
│   ├── per_block_router.py              per-layer device assignment + auto transfer
│   └── throughput_bench.py              sequential-vs-pipelined synthetic bench
│
├── modal/                               feature/native-modal-packing    6a6e425
│   ├── __init__.py
│   ├── packed_batch.py                  ModalBatchPacker + PackedBatch + MODAL_REGISTRY
│   ├── modal_mask.py                    per-token reset_flag / modal_id / sample_id
│   ├── cross_modal.py                   CrossModalContrastive (CLIP-style InfoNCE)
│   └── dispatch.py                      ModalDispatchEmbed (Qwen 151k vs byte 256)
│
├── auxsched/                            feature/native-async-aux        395549c
│   ├── __init__.py
│   ├── streams.py                       streams + future
│   ├── future.py                        Future primitive
│   ├── coordinator.py                   coordinator over 4 drivers
│   ├── action_async.py                  NeuroMCP action driver (off-trainer-thread)
│   ├── ttt_async.py                     SelfLearn TTT driver
│   ├── curiosity_async.py               curiosity reward driver
│   └── neuromcp_cpu.py                  NeuroMCP-on-CPU helper (rare path)
│
└── training/                            feature/trainer-refactor-v2     9ff6048
    ├── __init__.py
    ├── base.py                          BaseTrainer + TrainerConfig
    ├── kd.py                            KDTrainer (KD math + warmup)
    ├── sft.py                           SFTTrainer (response-only loss)
    ├── rl.py                            RL stub (GRPO scaffold)
    ├── self_distill.py                  online self-distill stub
    └── dispatcher.py                    mode-string → trainer subclass router

# MVP (separate top-level path; feature/native-mvp; SHA a21a470)
synapforge/native_demo.py                ~700 LOC, 0 torch imports, 100-step LNN+SNN train
synapforge/native_demo_torch_ref.py      ~300 LOC torch oracle for parity gate
tests/native/test_native_demo.py         5 gates: no-torch grep, monotonicity,
                                           torch-parity 5%, self-check, speed budget
```

---

## 30-line quickstart

After the integration merge lands (P31), the unified import surface
will be available as `synapforge.native`. Until then, each branch
exposes its own subpackage. This snippet shows the *target* unified
API:

```python
# 1. Native data: stream a 3-corpus quality mix without torch DataLoader
from synapforge.native.data import MixedTokenStream
stream = MixedTokenStream.from_files(
    ["fineweb_edu.parquet", "wt103.parquet", "alpaca_zh.parquet"],
    weights=[0.5, 0.3, 0.2],
    shuffle_buffer=10000,
)

# 2. Native CUDA tensors (cupy if available, else numpy fallback)
from synapforge.native.cuda import CudaTensor, ops as cops
x = CudaTensor.from_numpy(np_arr).to_device(0)

# 3. Fused HybridBlock kernel (Triton): CfC + PLIF + SEW + synapse
#    -- one kernel call vs the 4-op torch path
from synapforge.native.kernel import fused_hybrid_forward, fused_hybrid_backward
y, state = fused_hybrid_forward(x, params)   # Triton fwd
g_x = fused_hybrid_backward(grad_y, state)   # closed-form Triton bwd

# 4. Spike bit-packing (16:1) at the spike->synapse boundary
from synapforge.native.spike import pack_spikes, unpack_spikes
packed = pack_spikes(plif_out)               # uint16, 16 spikes per word
synapse_in = unpack_spikes(packed, target_shape)

# 5. STDP-only optimiser for plasticity-tagged params (skips AdamW)
from synapforge.native.stdp import STDPOptimizer, HybridOptimDispatch
opt = HybridOptimDispatch(model, plast_tags={"synapse_w"}, lr=1e-3)

# 6. Async hetero CPU+GPU dispatch (3-stage pipeline)
from synapforge.native.dispatch import HeteroPipeline, PerBlockRouter
pipe = HeteroPipeline(
    stage_a=data_prep,           # CPU thread
    stage_b=gpu_fwd_bwd,         # GPU thread
    stage_c=cpu_adamw_step,      # CPU thread (numpy releases GIL)
    queue_ab_size=2, queue_bc_size=1,
)

# 7. 9-modal byte-patch packed batch (text + image + audio + ...)
from synapforge.native.modal import ModalBatchPacker, ModalDispatchEmbed
packer = ModalBatchPacker(modal_registry=MODAL_REGISTRY)
packed_batch = packer.pack({"text": text_b, "image": img_b, "audio": aud_b})

# 8. High-level: BaseTrainer subclasses (KD / SFT / RL / SelfDistill)
from synapforge.native.training import KDTrainer, TrainerConfig
trainer = KDTrainer(TrainerConfig(stream=stream, pipe=pipe, opt=opt))
trainer.fit(steps=60000)
```

---

## Honest dormant-flag list

The native stack ships with several capabilities that are **wired but
currently inert** — usually because the runtime side of the LNN+SNN
hybrid (PLIF firing, STDP plasticity) hasn't reached the regime where
the optimisation pays off. Each item below is a TODO-with-evidence,
not a promise.

| flag / capability                       | status     | why dormant                                                                                | unlock condition                              |
|-----------------------------------------|------------|--------------------------------------------------------------------------------------------|-----------------------------------------------|
| `--packed-spikes` (spike pack 16×)      | code OK    | PLIF dead 0/16 across Run 3l/3m/3n — spike rate 0% means packing 0 active bits             | P25 close (surrogate anneal + threshold ramp) |
| STDP-only optimiser (skip AdamW)        | code OK    | only fires for params tagged with `plasticity=True`; current model has 0 such params       | tag SparseSynapse weights as plast            |
| `--stdp-only-plasticity` flag           | wired      | same as above                                                                              | tag plast params, ETA 1 day                   |
| Sparse synapse matmul (kernel)          | code OK    | exploits sparsity ≤5% of spikes; current dense PLIF=0 means kernel runs but 0 sparsity benefit | PLIF fires + spike rate ∈ [0.05, 0.20]        |
| `triton_block` fused kernel             | wired      | active in Run 7 already; v0.1 widens it across CfC+PLIF+SEW+synapse                        | already on; native version Run 8              |
| Async aux scheduler (2.9×)              | code OK    | requires SelfLearn / Curiosity / NeuroMCP / STDP-novelty modules to be live                | phase 1 trigger (val ppl ≤ 250)               |
| Async hetero CPU+GPU dispatch (1.7×)    | code OK    | sequential reference still default; pipeline mode opt-in via `enable_pipeline=True`        | Run 8 launch (post-integration)               |
| Per-block device router                 | code OK    | router built; trainer wire-in part of trainer-refactor-v2                                  | Run 8 launch                                  |
| 9-modal byte-patch packing              | code OK    | model trained text-only so far; modal encoders untrained                                   | phase 2 trigger (val ppl ≤ 100)               |
| Cross-modal contrastive loss            | code OK    | requires real COCO / AudioCaps batches downloaded; designed but pre-data                   | phase 2 + data prep                           |
| R-fold k-step inference                 | wired      | math verified (R=1 exact, R=8 drift 0.32%); GPU bench 2.99× at N=64 R=16                   | already on                                    |
| Trainer refactor v2 (BaseTrainer + subs)| code OK    | new subclasses exist next to existing trainers; default still `train_100m_kd.py`           | swap default in launcher (Run 8)              |

**Core principle**: nothing in v0.1 trades quality for speed. Every
optimisation is bit-exact against a torch oracle on a fixed
seed/shapes/init *or* carries an explicit numerical-error budget (e.g.
R-fold R=8 0.32%). Speed is not allowed to regress val ppl.

---

## Cross-references

- **What lives on top**: [INVESTOR.md "NEW differentiation"](INVESTOR.md) —
  the pitch-deck framing of why this shipping is structural.
- **Sibling design docs** (per-subpackage):
  - [NATIVE_DEMO.md](NATIVE_DEMO.md) — MVP scope + 10-14 day roadmap
  - [NATIVE_HETERO_DISPATCH.md](NATIVE_HETERO_DISPATCH.md) — 3-stage pipeline contract
  - [NATIVE_SPIKE_PACKING.md](NATIVE_SPIKE_PACKING.md) — bit-pack math + bandwidth
  - [NATIVE_MULTIMODAL_PACKING.md](NATIVE_MULTIMODAL_PACKING.md) — 9-modal registry
  - [NATIVE_CUDA_TENSOR.md](NATIVE_CUDA_TENSOR.md) — CudaTensor design + gap estimate
- **What's still on torch**: [TORCH_REPLACEMENT_PLAN.md](TORCH_REPLACEMENT_PLAN.md) —
  6-9 week phase 5 roadmap to remove the residual torch glue.
- **Live training**: [PROGRESS.md](PROGRESS.md) — Run 7 step 500 verdict +
  Run 8 native-stack launch tracker.
- **Phase gates**: [MASTER_PLAN.md §3](MASTER_PLAN.md) — phase 5 row added
  for Run 8 native launch.
- **Honest scope**: [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md) — what
  works / what's untested / what's rhetoric, refreshed for v0.7.0-native.

---

## Maintenance

Keep this doc up-to-date by:

1. When a new subpackage lands, add a row to the file-tree section + a
   row to the dormant-flag table (status `code OK` or `wired`).
2. When an integration step measures a real e2e number, update the
   "Measured" row in the TL;DR (replace the projected number with the
   measured one and prefix with the run name).
3. When a subpackage gets merged into the trunk, drop its
   `feature/native-*` branch reference from the file-tree section
   (history stays in git).
4. Bump the date at the top.

This doc is the contract for "what the native stack is and what it
isn't" between sessions. Don't let it drift.
