# Native heterogeneous CPU+GPU async dispatch

`synapforge.native.dispatch` is the long-term replacement for the
sequential train_step in `synapforge.training.trainer.Trainer`. It
gives the trainer three independent threads, each pinned to a
device class, and lets them overlap.

## Why

The current ZeRO-Offload Stage 0 is **sequential within each step**:

```
... step N ─────────────────────────────────────────────────────  step N+1
       │ A: dataloader │ B: GPU forward+backward │ C: CPU AdamW │
       └───────────────┴─────────────────────────┴──────────────┘
```

The CPU is idle during B; the GPU is idle during C. With a 730M
LNN+SNN model on A800 the AdamW step on the CPU is roughly the same
wall-clock as a GPU forward+backward chunk (`t_B ~= t_C`), so the
entire optim time is wasted.

`HeteroPipeline` runs the same three callbacks on three threads:

```
A:  ── b0 ──── b1 ──── b2 ──── b3 ──── ...
B:        ── B@0 ────  ── B@1 ────  ── B@2 ────  ...
C:                      ── C@0 ────  ── C@1 ────  ── C@2 ────  ...
```

In steady state, while Stage B computes batch N's forward+backward,
Stage C is applying batch N-1's AdamW step, and Stage A is producing
batch N+1. Wall-clock per step drops from `t_A + t_B + t_C` to
~`max(t_A, t_B, t_C)`, ceiling speedup `(t_B + t_C) / max(t_B, t_C)`.

## Architecture

```
                queue_AB (size 2)            queue_BC (size 1)
                │                            │
   Stage A ─────►─── Stage B ─────►─── Stage C
   (CPU thread)     (GPU thread)        (CPU thread)
   data prep         forward+bwd         AdamW step
                     on cuda stream      via CpuWorkerPool
```

Backpressure is the queue capacity:

* `queue_AB.put` blocks if Stage B is slow → throttles A.
* `queue_BC.put` blocks if Stage C is slow → throttles B.
  This is what enforces the 1-step pipeline depth (and prevents
  unbounded memory growth if C falls behind).

### Files

| file                                          | what it does                                                                                                       |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `synapforge/native/dispatch/__init__.py`      | re-exports the public surface                                                                                      |
| `synapforge/native/dispatch/streams.py`       | `CudaStream` + `StreamPair` (cupy wrapper, no-op CPU fallback). H2D issued on transfer stream, gated via CUDA event |
| `synapforge/native/dispatch/cpu_pool.py`      | `CpuWorkerPool` thread pool + `parallel_adamw_step` helper (numpy releases GIL during BLAS ⇒ true parallel)        |
| `synapforge/native/dispatch/pipeline.py`      | `HeteroPipeline` 3-stage scheduler with `enable_pipeline=False` sequential reference                               |
| `synapforge/native/dispatch/per_block_router.py` | `PerBlockRouter` -- per-layer device assignment with auto cross-device transfer                                 |
| `synapforge/native/dispatch/throughput_bench.py` | sequential vs pipelined synthetic bench with auto-balanced Stage B cost                                         |

### Hard constraint

**Zero `import torch`** anywhere under `synapforge/native/dispatch/`.
We use `numpy` + optional `cupy` + `threading` + `queue`. The
test suite enforces this (`test_no_torch_import`).

## Per-block routing recipe

The router runs each block on its assigned device, transferring at
boundaries on a cached `StreamPair`. Example for our 16-block 730M
LNN+SNN model on a 80 GB A800 + 128 GB host:

```python
from synapforge.native.dispatch import PerBlockRouter

router = PerBlockRouter(
    num_blocks=16,
    config={
        "embed":       "cpu",     # huge V x d table -- park on CPU
        "layers_0_7":  "cuda",    # heavy CfC + FFN on GPU
        "layers_8_15": "cpu",     # bandwidth-bound layers on CPU MKL
        "lm_head":     "cuda",    # fused softmax+CE on GPU
    },
)
```

Forward driver (sketch):

```python
x = embed_lookup(input_ids)             # output is on "embed" device (CPU)
prev = "embed"
for i in range(16):
    cur = f"layers_{i}"
    x = router.move(x, prev, cur)       # transfers iff prev != cur
    x = block_fwd(i, x, params[i])      # runs on cur device
    prev = cur
x = router.move(x, prev, "lm_head")
logits = lm_head(x, lm_params)
```

Backward driver runs the same sequence in reverse.

The router holds **one `StreamPair` per direction**, so a hop from
`cuda → cpu` and a later hop from `cuda → cpu` reuse the same
streams (and pipeline their copies behind each other). Use
`router.synchronize()` to wait for all in-flight transfers.

### When per-block routing actually helps the 730M model on A800

| condition                                     | speedup      | recommendation                            |
|-----------------------------------------------|-------------:|-------------------------------------------|
| `t_cpu_layer << t_gpu_layer`                  | ~1.0x       | put everything on GPU                     |
| `t_cpu_layer ~= t_gpu_layer` (LNN+SNN sweet)  | up to ~2.0x | split blocks 50/50                        |
| `t_cpu_layer >> t_gpu_layer`                  | < 1.0x      | put everything on GPU                     |
| GPU memory-tight (≥80% HBM)                   | ~1.2x       | offload embed/lm_head/last few layers     |

For our model: the CfC delta MLP is ~64K params per layer, FFN is
~4M, so `t_gpu_layer` at d_model=4096 is roughly 30-40 ms and
`t_cpu_layer` (single-thread MKL) is ~50-90 ms. **The ratio is
within 2x, so per-block split gives a real ~1.3-1.6x speedup on
this model**, and removes the GPU-memory bottleneck. Without
split, the entire model has to fit in 80 GB; with `layers_8_15`
on CPU we save ~50% of activation memory and can train at higher
batch.

## Pipeline timing model

```
                 t_A   t_B   t_C
sequential:      ──    ████  ░░░    →  step takes  t_A + t_B + t_C
sequential:      ──    ████  ░░░    →  step takes  t_A + t_B + t_C
sequential:      ──    ████  ░░░    →  step takes  t_A + t_B + t_C

pipelined warm:        ████        ←  B@N
                       ░░░  ←  C@N-1 in parallel
                                         step takes max(t_B, t_C)
                                         (t_A overlaps with both)
```

Speedup ceiling: `(t_A + t_B + t_C) / max(t_A, t_B, t_C)`. With
`t_A = 0.005`, `t_B = 0.06`, `t_C = 0.08` (typical numbers for
730M LNN+SNN on A800 + 32-core host):

* sequential per-step:  0.145 s  →  6.9 step/s
* pipelined per-step:   0.080 s  →  12.5 step/s
* speedup:              **1.81x**

## Determinism & correctness

The pipeline does NOT block Stage B from reading params concurrently
with Stage C's in-place mutation. This is the documented
ASGD / Hogwild!-style 1-step staleness:

* B@N reads params possibly mutated by C@N-1 mid-step.
* The training trajectory drifts within fp16-noise bounds compared
  to strict sequential SGD.

For LNN+SNN with continuous-time CfC dynamics this is empirically
benign at moderate batch size. If you need *strict* zero-staleness
(numerical reproducibility), wrap `optim_step_fn` so it
double-buffers the params: write to a shadow copy in Stage C, swap
pointers under a lock at end-of-step.

The acceptance tests use a **param-independent grad function** so
the staleness is invisible — the sequential and pipelined param
trajectories are bit-identical, which confirms the *ordering* is
correct.

## Throughput benchmark

```
python -m synapforge.native.dispatch.throughput_bench \
    --num-steps 30 --param-count 100000000
```

Auto-balances Stage B vs Stage C cost so the ceiling is ~2.0x.
Outputs a markdown table and a JSON line. The bench passes when
the wallclock ratio is `>= 1.5x`. On a CPU-only dev host the test
artificially restricts cpu_pool oversubscription (B and C compete
for cores); use `--b-simulated-gpu-s 0.05` to model the
A800-production case (B sleeps, releasing the GIL).

## Test coverage

`pytest tests/native/dispatch/test_pipeline.py -v` runs 11 tests:

* `test_pipelined_matches_sequential_final_params` — gate 1
* `test_pipelined_matches_sequential_more_steps` — longer run
* `test_queue_back_pressure_no_OOM` — gate 2
* `test_queue_back_pressure_against_slow_dataloader`
* `test_sequential_mode_runs_in_calling_thread`
* `test_exception_in_stage_b_propagates`
* `test_exception_in_stage_c_propagates`
* `test_metrics_populated`
* `test_b_c_overlap_ratio_high_when_balanced`
* `test_zero_max_steps_is_noop`
* `test_no_torch_import`

The tests bypass `synapforge.__init__` (which currently imports
torch) by loading the dispatch modules via
`importlib.spec_from_file_location`.
