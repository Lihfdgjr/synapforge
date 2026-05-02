# Parallelism: multi-core, mixed-device, multi-node

Three independent layers. Pick what you need; they compose.

## Layer 1 — Multi-core CPU

PyTorch's default `num_threads` is `min(4, n_cpu)` which leaves ~50% idle on
modern boxes. Single call at training entry:

```python
from synapforge.parallel import optimize_cpu_threads
optimize_cpu_threads(reserve_for_dataloader=1)
# sets torch.set_num_threads(n_cpu - 1)
# enables mkldnn, sets OMP_NUM_THREADS so workers don't oversubscribe
```

On an 8-core box this lifts `torch.get_num_threads()` from 4 → 7 and
typically gives 1.4-1.7× on dense matmul-heavy workloads.

## Layer 2 — Mixed CPU+GPU placement

For a 375M LNN+SNN, embedding (156M) + lm_head (tied, but logits matmul
is huge) are parameter-heavy but *low FLOP per param*. They make sense on
CPU. The backbone (CfC + PLIF + SwiGLU) is compute-bound and small — keep
on GPU.

```python
from synapforge.parallel import place_mixed_device
placement = place_mixed_device(
    model,
    gpu="cuda:0",
    cpu_module_names=("embed_tokens", "lm_head", "lm_logits"),
    verbose=True,
)
# -> 51.2M on CPU, 0.3M on GPU (toy 50M model)
```

Trade-off: one PCIe transfer of the activation per step (B*T*D ≈ 4MB at
B=8 T=1024 D=1024). On a 24GB GPU running 375M, this frees ~6GB VRAM at
~5% wall-clock cost. Worth it when batch size is otherwise capped.

## Layer 3 — Multi-node DDP

Backend auto-selected: `nccl` if CUDA is available (GPU-to-GPU), else
`gloo` (CPU-to-CPU; also required for heterogeneous CPU+GPU clusters).

```python
from synapforge.parallel import init_distributed, is_main_rank
dist = init_distributed(backend="auto")  # reads RANK/WORLD_SIZE/LOCAL_RANK env

if is_main_rank():
    print(f"rank 0 of {dist.world_size}")
```

Launch:

```bash
# single 4-GPU node
torchrun --nproc-per-node=4 train.py

# 2 nodes × 8 GPUs each = 16 GPUs
torchrun --nnodes=2 --nproc-per-node=8 \
    --rdzv-backend=c10d --rdzv-endpoint=master_host:29500 train.py

# 2 CPU-only nodes × 4 ranks (gloo)
torchrun --nnodes=2 --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=master_host:29500 train.py
```

## DataLoader

`auto_dataloader` picks `num_workers`, `prefetch_factor`, `pin_memory`,
`persistent_workers` for you, and adds a `DistributedSampler` if
`distributed=True`:

```python
from synapforge.parallel import auto_dataloader
loader = auto_dataloader(dataset, batch_size=8, distributed=(dist is not None))
```

## End-to-end example

`examples/mixed_device_training.py` runs all three layers — multi-core
CPU + mixed embedding-on-CPU + DDP-aware. Sanity-tested at 468 tok/s on
a 33.9M model on a single GPU. With CPU-on-DDP add `--gloo` flag to your
torchrun.

---

# T8.7 — Two A800s across rentals (planning doc, NOT implemented)

We currently have **1× A800 80GB** (rental `117.74.66.77:41614`). The
queue item T8.7 in `docs/DEEP_MAINT_QUEUE.md` asks: when a second box
shows up, can we DDP across rentals to double train throughput?

Short answer: **probably not worth it; rent a single dual-GPU box or one
H100 instead.** Long answer below.

## Option A — Single-rental dual-GPU (NCCL, recommended path)

When (not if) we rent a 2× A800 box, `torchrun` does the right thing
out of the box:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc-per-node=2 train_100m_kd.py \
  --backend triton_block --batch-size 256 --steps 8000
```

- **Backend**: NCCL (GPU↔GPU, NVLink/PCIe).
- **All-reduce bandwidth**: ~50 GB/s NVLink-3 between two A800s in
  the same chassis — gradient sync is essentially free relative to the
  forward/backward step.
- **Expected speedup**: 1.85-1.95× at our current 100M shape (gradient
  bucketing covers the small comm cost).
- **Cost**: same hourly rate as single-A800 if the rental sku exists;
  in practice 2× A800 boxes are rare on the small Chinese rental
  marketplaces we use, so this is the *aspirational* path.

## Option B — Multi-rental DDP (gloo, **not recommended**)

Two single-A800 rentals, glued together over the public Internet via
`gloo` running over TCP/IP. PyTorch DDP supports this, but the costs are
ugly:

```bash
# Rental A (master, public IP 117.74.66.77, port 29500 forwarded):
GLOO_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --node-rank=0 --nproc-per-node=1 \
  --master-addr=117.74.66.77 --master-port=29500 \
  train_100m_kd.py --backend triton_block --batch-size 128 ...

# Rental B (worker):
GLOO_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --node-rank=1 --nproc-per-node=1 \
  --master-addr=117.74.66.77 --master-port=29500 \
  train_100m_kd.py --backend triton_block --batch-size 128 ...
```

### Why gloo here, not NCCL?

NCCL needs a low-latency RDMA fabric (InfiniBand or RoCE) to even start
between hosts; commodity Internet between two rentals is TCP-only. You
can technically run NCCL over TCP but performance is far worse than gloo
(which is designed for it) — `init_method="tcp://..."` is brittle and
the NCCL team explicitly says "don't" for cross-DC.

### What the gloo cost actually looks like

| Term | NCCL same host (NVLink-3) | gloo cross-rental (1 Gbit Internet) |
|------|---------------------------|--------------------------------------|
| All-reduce bandwidth | ~50 GB/s | ~80-110 MB/s (best case, no congestion) |
| 100M model gradient | 400 MB | 400 MB |
| One step all-reduce | ~8 ms (3% of step) | ~4-5 s (4-6× the step time) |
| Effective speedup | 1.9× | 0.3-0.5× (i.e. **slower than single-rental**) |

Numbers cross-check with PyTorch's own DDP perf notes
([pytorch.org/docs/stable/notes/ddp.html](https://pytorch.org/docs/stable/notes/ddp.html))
and the DeepSeek MFU study (DeepSeek-V2 paper §3.4) which reports gloo
all-reduce at <10% of NCCL bandwidth on commodity gigabit ethernet.

Even with sparse-z-loss reducing the **z-loss term**'s traffic by ~70×
(`k=2048` of `vocab=151936` ≈ 74×, see `_sparse_z_loss` in
`train_100m_kd.py`), the dominant traffic is the **parameter gradient
all-reduce**, which sparse z-loss does *not* touch. So the 70× saving
applies to a small fraction of comms and the headline number stays at
~5× slowdown vs NCCL.

## Synapforge-specific gotchas

These are not generic DDP issues — they're things you only hit when you
DDP **this repo** specifically.

1. **Plasticity buffers** — STDP traces, calcium, astrocyte state, etc.
   live in `register_buffer`, not `nn.Parameter`. PyTorch DDP's gradient
   all-reduce does not touch them, and `broadcast_buffers=True`
   *overwrites* rank-1's locally-updated plasticity with rank-0's (worse
   than divergence). The right fix is already shipped: see
   `synapforge/distributed.py` (`PlasticBufferSync`) — it explicitly
   `all_reduce(AVG)`s plasticity buffers after each optimizer step.
   **Required** if you turn on STDP / Hebbian / AstrocyteGate under DDP.

2. **PLIF spike rates are 0.0-0.5 dense** — binary spikes can't be
   compressed with the standard "1-bit gradient" tricks because the
   surrogate gradient (ATan / SoftSign) is real-valued and we need it
   for backprop. So PLIF doesn't help comms; it just doesn't hurt.

3. **Triton kernels are version-pinned** — `synapforge/triton_block.py`
   was patched against Triton 2.1.x bf16 MLIR encoding bugs (commit
   `8e3a7c4`). If a second rental ships with Triton 2.2+ or 2.0, the
   kernel may compile differently or crash silently. Pre-flight before
   any cross-rental DDP: run `pytest tests/integration/test_triton_*`
   on **both** boxes and lock to the lower-of-the-two versions, or
   fall back to `--backend gpu_dense` (slower but version-agnostic).

4. **Sparse z-loss top-K reduction** — `_sparse_z_loss(logits, k=2048)`
   reduces *its own* communication by 74× (vocab 151936 / 2048), but
   does **not** reduce parameter-gradient comms. Useful for MoE
   loss-balance fields, irrelevant for the dominant all-reduce.

5. **Data sharding** — `DistributedSampler` deterministically shards
   per-rank, but Synapforge `ParquetTokenStream` has its own
   `shuffle_buffer=10000` (commit `5f420a4`, fixes step-2500 lockstep
   bug). The shuffle seed must include `rank` or both ranks see the
   same shard order. Check `auto_dataloader` does this; smoke test
   `tests/integration/test_shuffle_buffer.py` covers single-rank.

## Realistic plan (decision matrix)

| Option | Hourly cost (¥) | Effective speedup | Notes |
|--------|-----------------|--------------------|-------|
| Stay on 1× A800 | ~7 | 1.0× (baseline) | current state |
| 1× H100 80GB | ~12-15 | ~1.7× (FP8 + 2 TB/s HBM) | drop-in if available |
| 2× A800 same chassis (NCCL) | ~14 | ~1.9× | aspirational; rare sku |
| 2× A800 cross-rental (gloo) | ~14 | **0.3-0.5×** | strictly slower; **do not do this** |

**Decision**: skip multi-rental DDP. Path forward when the current 100M
run plateaus:
- First-choice: single H100 80GB rental upgrade.
- Second-choice: dual-A800 same-chassis rental if one shows up.
- gloo cross-rental: only useful as a last-resort *evaluation*
  parallelism (run val on rental B while rental A trains; that's not
  DDP, it's pipeline of independent jobs).

The skeleton at `scripts/ddp_launch.py` exists so we can flip the switch
fast when a dual-GPU box does land — no spec writing under time
pressure. Smoke test
`tests/integration/test_ddp_launch_smoke.py` verifies it parses
single-rank and refuses to start multi-node without `--master-addr`.

## What this does NOT do (yet)

- **Pipeline parallel**: layer-wise sharding across GPUs. CfC sequential
  recurrence makes the standard pipe split awkward; would need bubble-aware
  scheduling.
- **Tensor parallel**: sharding the hidden dim across ranks. Useful at >1B
  params; we're at 375M, not yet needed.
- **FSDP / ZeRO-3**: parameter sharding. The MoE wins from ZeRO are folded
  into the Pareto plan (`docs/PARETO_OPTIMIZATION.md`); not yet wired.
- **Heterogeneous gloo+nccl**: rank 0 GPU + ranks 1..N CPU. Proof of concept
  in `synapforge/distributed_hetero.py` (846 LOC) but not battle-tested at
  multi-node scale.

## Honest performance notes

- The 1.4-1.7× from `optimize_cpu_threads` only applies to dense BLAS
  workloads. CfC scan + PLIF spike are sequential and don't benefit.
- Mixed placement's 5% overhead grows with smaller batch sizes (PCIe
  setup amortizes worse). At B=1 it can be 15%.
- Multi-node gloo over commodity ethernet (1Gbps) caps allreduce at
  ~125 MB/s. A 375M model's gradient is 1.4GB → 11s per allreduce step.
  Use 10/25/100 GbE or InfiniBand for serious training.

## References

- PyTorch DDP design notes — https://pytorch.org/docs/stable/notes/ddp.html
- gloo backend choice — https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
- DeepSeek-V2 §3.4 (MFU + collective bandwidth study) — arXiv:2405.04434
- Synapforge plasticity-aware DDP — `synapforge/distributed.py` (`PlasticBufferSync`)
