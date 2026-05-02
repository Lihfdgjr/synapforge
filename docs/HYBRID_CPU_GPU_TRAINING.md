# Hybrid CPU+GPU Training (ZeRO-Offload Stage 0)

**Date authored:** 2026-05-02
**Branch of origin:** `feature/hybrid-cpu-gpu-training` off `feature/torch-replacement-phase2`
**Owner:** Run 6 throughput uplift workstream
**Triggered by:** user request 2026-05-02 18:55 — "把 cpu 用起来吗,上混合训练,把内存也用起来"

## TL;DR

The A800 80GB rental has ~256 GB DRAM that sits at <5% utilisation while
the 80 GB HBM is the bottleneck. This doc describes three opt-in CLI
flags on `train_100m_kd.py` that move increasingly heavy state from
HBM → pinned CPU RAM, freeing HBM headroom that we reinvest in larger
micro-batches.

| Flag | What moves to CPU | HBM freed | Step-time delta | Status |
| ---- | ----------------- | --------- | --------------- | ------ |
| `--cpu-offload-optim` | AdamW m/v + master fp32 param | ~2*N*4 bytes (~0.8 GB @ 100M, ~4 GB @ Ultra) | +3-5 ms (CPU AdamW + H2D/D2H, pipelined) | shipped |
| `--teacher-cpu-int8` | KD teacher (Qwen 0.5B class) | ~2 GB | +200 ms (CPU forward, partially overlapped) | shipped |
| `--activation-cpu-offload` | Per-layer activations during forward | up to 10-15 GB at bs=24 d=1280 | +H2D for each saved tensor; net positive | stretch (TODO) |

All three are independent; layering all three on Run 6 should permit
**bs=64 at d=1280 vs the current bs=24**, lifting throughput from
**5240 tok/s** estimated to **8-12 k tok/s** depending on PCIe headroom
and KD-every cadence.

## 1 — Where state lives (per-step)

```
                 ┌─────────────────────────────────────┐
                 │            A800 80GB HBM            │
                 │  (target: maximise micro-batch B)   │
                 ├─────────────────────────────────────┤
                 │ student params (fp32)              │  ─── always
                 │ student grads (fp32)               │  ─── always
                 │ activations (B*T*d at fp32/bf16)   │  ─── unless --activation-cpu-offload
                 │ KD teacher (fp16)                  │  ─── unless --teacher-cpu-int8
                 │ AdamW m, v, master                 │  ─── unless --cpu-offload-optim
                 │ KD logits buffer                   │  ─── always
                 │ workspace                          │  ─── triton scratch + dropout masks
                 └─────────────────────────────────────┘

                 ┌─────────────────────────────────────┐
                 │       Xeon 16-core + 256 GB DRAM    │
                 │       (target: load idle resources) │
                 ├─────────────────────────────────────┤
                 │ AdamW m, v (pinned, fp32)          │  ─── --cpu-offload-optim
                 │ AdamW master fp32 param            │  ─── --cpu-offload-optim
                 │ pinned grad staging buffer         │  ─── --cpu-offload-optim
                 │ teacher (INT8 via bnb, or fp32)    │  ─── --teacher-cpu-int8
                 │ activation pinned ring buffer      │  ─── --activation-cpu-offload (stretch)
                 └─────────────────────────────────────┘
```

## 2 — Data flow per step (`--cpu-offload-optim` + `--teacher-cpu-int8`)

```
GPU default stream                 CPU (16 cores, OpenMP)              GPU side stream (kd-async)
──────────────────                 ────────────────────                 ──────────────────────────
│ student fwd                                                          
│ student fwd                       teacher fwd (CPU INT8)              ← input ids H2D (non_blocking)
│ student bwd                       teacher fwd                          
│ student bwd                       teacher fwd → CPU logits            
│   p.grad ready                                                        
│   ─── kick D2H grad ─────────►   recv grads (pinned)                 → logits H2D back to GPU
│   ─── synchronize() ─────────►                                        
│                                  ┌── AdamW step on master fp32 ──┐   
│                                  │  m_t = β₁ m + (1−β₁) g         │   
│                                  │  v_t = β₂ v + (1−β₂) g²        │   
│                                  │  master ← master − step        │   
│                                  └────────────────────────────────┘   
│   ─── recv H2D updated p ◄────                                        
│ student fwd (next step)                                               
│ student fwd (next step)
```

The critical-path cost of the offload is the per-step CPU AdamW math.
With `addcmul_` / `sqrt` / `addcdiv_` dispatched to ATen vectorised
CPU paths, 100M params land at ~3-5 ms on a 16-core Xeon. The H2D
grad copy + D2H param copy together are bound by PCIe gen-4 x16
(~32 GB/s effective). At 100M params * 4 bytes = 400 MB per direction,
that's ~13 ms uncovered, which the pipelining with the next forward
hides ~50-80% of.

## 3 — Memory budget at bs=24 vs bs=64 (d=1280, n_layers=16, seq=256)

Approximate, fp32 values scaled to bf16 where activations live in bf16.

### bs=24 (Run 6 baseline)

| Component | Bytes | Comment |
| --------- | ----- | ------- |
| Student params (fp32) | 400 MB | 100M * 4 |
| Student grads (fp32) | 400 MB | 100M * 4 |
| Adam m + v (fp32) | 800 MB | 2 * 100M * 4 |
| Adam master (in-place, no copy) | 0 | (no master in default path) |
| KD teacher (Qwen 0.5B fp16) | 1.0 GB | |
| Activations (bs * seq * d, bf16, ~30 saved per block) | ~14 GB | 24*256*1280*2*16*~7 |
| KD logits (top-k 2048 path, fp32) | ~80 MB | |
| Workspace + tritons + masks | ~3 GB | |
| **HBM peak** | **~20 GB** | well within 80GB |

### bs=64 (target with all offload flags ON)

| Component | Bytes | Comment |
| --------- | ----- | ------- |
| Student params (fp32) | 400 MB | unchanged |
| Student grads (fp32) | 400 MB | unchanged |
| Adam m + v | 0 (on CPU) | `--cpu-offload-optim` |
| KD teacher | 0 (on CPU) | `--teacher-cpu-int8` |
| Activations (bs * seq * d, bf16) | ~37 GB | 64*256*1280*2*16*~7 |
| KD logits (top-k 2048 path) | ~210 MB | |
| Workspace + tritons + masks | ~3 GB | |
| **HBM peak** | **~41 GB** | leaves ~40 GB headroom |

The headroom is intentional: PyTorch's caching allocator fragments
heavily on 100M-class model with KD chunked softmax; we want enough
slack to keep `cudaMalloc` pressure low, otherwise bs=64 becomes
fragile.

### bs=80+ (with `--activation-cpu-offload` STRETCH)

If we also offload activations, the bs * seq * d * n_layers term moves
to CPU at the cost of an extra H2D per saved tensor during backward.
At PCIe gen-4 + 30 saved tensors per block * 16 blocks, we cap at
bs ≈ 80-96 before the activation-replay cost dominates.

## 4 — Expected tok/s estimate

**Baseline (Run 6):** 5240 tok/s @ bs=24

### `--cpu-offload-optim` alone

* HBM freed: ~0.8 GB (100M model). Allows bs=24 → bs=28 (~17% lift).
* Per-step overhead: +3-5 ms CPU AdamW, mostly hidden by H2D copy of next batch.
* **Estimate: 5500-6300 tok/s** (5-20% lift; tighter on 100M, wider on Ultra where 4 GB freed = bs=24 → bs=32-36 = +50%).

### `--cpu-offload-optim` + `--teacher-cpu-int8`

* HBM freed: ~0.8 GB + ~2 GB = ~2.8 GB. Allows bs=24 → bs=32-40 (~33-67% lift).
* Per-step overhead: +3-5 ms (optim) + 200-500 ms teacher every `--kd-every` steps. With kd-every=4 the per-step amortised overhead is +50-125 ms, which the larger batch must outweigh.
* **Estimate: 6500-8000 tok/s** if the GPU forward is bandwidth-bound at the new bs; less if the teacher forward saturates.

### All three (stretch — adds `--activation-cpu-offload`)

* HBM freed: 10-15 GB additional. Allows bs=64-80 (~3x baseline).
* Per-step overhead: significant additional H2D for saved tensors during backward; pipelining helps but the activation-replay cost is real.
* **Estimate: 8000-12000 tok/s**, very PCIe-dependent. On A800 (gen-4 x16 ~32 GB/s effective) the 30 saved tensors per block * 16 blocks * bs=80 * seq=256 * d=1280 * 2 bytes ≈ 31 GB per backward, ~1 second @ 32 GB/s. We need this hidden by the GPU compute on the 30 forward tensors before it. On Hopper/Blackwell with PCIe gen-5 x16 (~60 GB/s) the math closes; on A800 it's tight.

**Honest range for Run 6 with deliverables 1+2 ON, deliverable 3 OFF:** **6500-8500 tok/s**, 25-60% lift over baseline. 8000+ is contingent on KD-every>=4 and the teacher overlap working as advertised; on KD-every=1 the CPU teacher is the bottleneck.

## 5 — `cpu_avx2` backend interaction (future work)

`synapforge/backends/cpu_avx2.py` already exists in the repo and ships
AVX2-vectorised CfC + PLIF kernels for CPU-only inference. **It is
NOT activated by any of the flags in this doc.** Per-layer hybrid
execution (some blocks on GPU, some on CPU via `cpu_avx2`) is a
separate design with its own routing logic — Stage 4 of the
torch-replacement plan, gated on cpu_avx2 reaching parity with the
GPU dense backend on bf16 arithmetic. We note its existence here so
the next person knows to look there before reinventing CPU forward
infra.

## 6 — What's next (out of scope for this PR)

* **ZeRO-Offload Stage 3 (param + grad sharding)**: required for
  >1B-param student. Out of scope for this run; the deliverable 1
  module is intentionally Stage 0 only.
* **NVMe offload** of optimizer state: cheaper RAM at the cost of
  ~10 GB/s NVMe BW. Worth it on >7B models; overkill for 100M-500M.
* **Async optimizer step** (CPU AdamW runs concurrently with the next
  GPU forward, not just the H2D copy): saves another ~5 ms per step.
  Requires a CPU thread pool and double-buffering of master fp32 param.
  Tracked in `docs/DEEP_MAINT_QUEUE.md` as a Q3 item.
* **Per-layer hybrid (cpu_avx2)**: see §5.

## 7 — How to use

```bash
# Drop-in to Run 6 launcher (scripts/launch_run6.sh):
python train_100m_kd.py \
    --kd-weight 0.7 --kd-every 4 --kd-async-teacher \
    --teacher Qwen/Qwen2.5-0.5B \
    --teacher-cpu-int8 \
    --cpu-offload-optim \
    --batch-size 40 \
    ...
```

Each flag is independently opt-in; default behaviour is unchanged so
the math-simplification agent's `--rfold` / `--grad-ckpt` flags
compose cleanly.

## 8 — Test coverage

* `tests/optim/test_cpu_offload_adamw.py` — 5 tests, all passing on
  CPU CI. The 50-step bit-exact contract test (`rel_err < 1e-5` vs
  `synapforge.optim.AdamW`) is the load-bearing one.
* `tests/teachers/test_cpu_int8_qwen.py` — 4 tests; 3 always run
  (signature smoke, device check, shape/dtype contract on a tiny
  stand-in LM), 1 gated heavyweight (real Qwen 0.5B load + forward,
  enabled via `RUN_HEAVY_TEACHER_TESTS=1`).
* CI runs the full `tests/optim/ tests/teachers/` battery on every
  push to this branch.
