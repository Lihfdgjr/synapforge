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
