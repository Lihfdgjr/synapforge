"""Multi-core / mixed-device / multi-node training helpers.

Three layers, each independently usable:

  Layer 1 — CPU thread tuning
    optimize_cpu_threads() picks the right intra/inter-op thread split,
    enables MKL-DNN if available, sets OMP_NUM_THREADS so child workers
    don't oversubscribe.

  Layer 2 — Mixed CPU+GPU placement
    place_mixed_device(model) pushes embedding + lm_head to CPU (they're
    huge, parameter-heavy, low-FLOP) and backbone CfC/PLIF blocks to GPU
    (compute-bound, small parameters). For a 375M model with 156M of
    embedding/head, this moves ~40% of params off the GPU at the cost of
    one PCIe transfer per step. Net win when GPU memory is tight.

  Layer 3 — Multi-node DDP
    init_distributed() picks gloo (CPU) / nccl (GPU) automatically and
    discovers peers from torchrun env vars. Drop-in for `torch.distributed`.
    Supports the heterogeneous case where rank 0 is GPU + ranks 1..N are
    CPU (gloo + reduce-only-on-CPU pattern).

The functions are deliberately small and free-standing — no class
hierarchy, no global state besides what `torch.distributed` already keeps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Layer 1: CPU thread tuning
# ----------------------------------------------------------------------------


@dataclass
class ThreadConfig:
    intra_op: int
    inter_op: int
    omp_num_threads: int
    mkl_enabled: bool
    mkldnn_enabled: bool


def optimize_cpu_threads(reserve_for_dataloader: int = 1) -> ThreadConfig:
    """Pick a sensible thread split for CPU-heavy training.

    PyTorch's default is min(4, num_cpus) which leaves 50%+ idle on modern
    8/16-core boxes. We reserve `reserve_for_dataloader` cores per worker
    process and give the rest to BLAS/MKL via intra-op threads. inter-op
    threads stays at 1 (CfC scan + PLIF spike are sequential anyway).

    Side effects: sets OMP_NUM_THREADS so child dataloader workers don't
    each launch their own MKL pool and oversubscribe.
    """
    n_cpu = os.cpu_count() or 1
    intra_op = max(1, n_cpu - reserve_for_dataloader)
    inter_op = 1

    torch.set_num_threads(intra_op)
    try:
        torch.set_num_interop_threads(inter_op)
    except RuntimeError:
        # interop threads can only be set once before any parallel work starts;
        # idempotent re-entry is fine, just keep the existing value.
        inter_op = torch.get_num_interop_threads()
    os.environ.setdefault("OMP_NUM_THREADS", str(intra_op))
    os.environ.setdefault("MKL_NUM_THREADS", str(intra_op))

    mkl_enabled = bool(getattr(torch.backends.mkl, "is_available", lambda: False)())
    mkldnn_enabled = bool(getattr(torch.backends.mkldnn, "is_available", lambda: False)())
    if mkldnn_enabled:
        torch.backends.mkldnn.enabled = True

    return ThreadConfig(intra_op, inter_op, intra_op, mkl_enabled, mkldnn_enabled)


# ----------------------------------------------------------------------------
# Layer 2: Mixed CPU+GPU placement
# ----------------------------------------------------------------------------


@dataclass
class MixedPlacement:
    cpu_modules: list[str]
    gpu_modules: list[str]
    cpu_param_count: int
    gpu_param_count: int


def place_mixed_device(
    model: nn.Module,
    gpu: torch.device | str = "cuda:0",
    cpu_module_names: Iterable[str] = ("embed_tokens", "lm_head", "lm_logits"),
    verbose: bool = False,
) -> MixedPlacement:
    """Push named modules to CPU, the rest to GPU.

    For a 375M LNN+SNN with embed=156M + head=tied + backbone=219M, this
    moves ~40% of params off the GPU. The CPU module's forward incurs one
    H2D copy of the activation (B*T*D ≈ 4MB at B=8 T=1024 D=1024) per step
    — small relative to the freed VRAM.

    Pattern matching is exact-name on the top-level module attribute path
    (e.g., `model.embed_tokens`, `model.backbone.embed_tokens`). For
    NeuroMCP head + skill index, leave them on GPU (hot path).
    """
    gpu = torch.device(gpu)
    cpu = torch.device("cpu")
    cpu_set = set(cpu_module_names)

    cpu_mods, gpu_mods = [], []
    cpu_params, gpu_params = 0, 0

    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in cpu_set:
            module.to(cpu)
            n = sum(p.numel() for p in module.parameters(recurse=False))
            cpu_params += n
            cpu_mods.append(name)

    # Everything else to GPU
    for name, p in model.named_parameters():
        if not any(name.startswith(n + ".") or name == n + ".weight" for n in cpu_mods):
            p.data = p.data.to(gpu)
            gpu_params += p.numel()

    if verbose:
        print(f"  mixed: {cpu_params/1e6:.1f}M on CPU, {gpu_params/1e6:.1f}M on GPU")

    return MixedPlacement(cpu_mods, gpu_mods, cpu_params, gpu_params)


# ----------------------------------------------------------------------------
# Layer 3: Multi-node DDP
# ----------------------------------------------------------------------------


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device


def init_distributed(backend: str = "auto") -> DistInfo | None:
    """Initialize torch.distributed using torchrun env vars.

    backend="auto" -> nccl if CUDA available, gloo otherwise. For
    heterogeneous CPU+GPU clusters, gloo is required (NCCL is GPU-only).

    Returns None if env vars are missing (single-process training).

    Launch:
        torchrun --nnodes=1 --nproc-per-node=4 train.py        # 4-GPU node
        torchrun --nnodes=2 --nproc-per-node=8 \\               # 16-GPU multi-node
                 --rdzv-backend=c10d --rdzv-endpoint=HOST:29500 train.py
        torchrun --nnodes=2 --nproc-per-node=4 train.py        # 8-CPU multi-node (gloo)
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None

    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return DistInfo(rank, world_size, local_rank, backend, device)


def is_main_rank() -> bool:
    """True for rank 0 or single-process. Use for logging / ckpt saves."""
    return int(os.environ.get("RANK", "0")) == 0


# ----------------------------------------------------------------------------
# Layer 2.5: optimal DataLoader knobs
# ----------------------------------------------------------------------------


def auto_dataloader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    num_workers: int | None = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    pin_memory: bool | None = None,
    distributed: bool = False,
):
    """DataLoader with sensible defaults for our workloads.

    - num_workers=None -> min(4, n_cpu // 2) for CPU box, n_cpu for GPU box
    - persistent_workers=True (avoids per-epoch worker fork cost)
    - prefetch_factor=4 (GPU stays fed at >0.5 step throughput)
    - pin_memory auto-on when CUDA available
    - distributed=True wraps with DistributedSampler (no manual sampler)
    """
    from torch.utils.data import DataLoader

    if num_workers is None:
        n_cpu = os.cpu_count() or 4
        on_gpu = torch.cuda.is_available()
        num_workers = n_cpu if on_gpu else max(2, n_cpu // 2)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # sampler handles it

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
    )


# ----------------------------------------------------------------------------
# Convenience: print full setup at training entry
# ----------------------------------------------------------------------------


def print_setup() -> None:
    """One-line summary of current parallel configuration. Idempotent."""
    cfg = optimize_cpu_threads()
    dist = init_distributed(backend="auto")

    cuda = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count() if cuda else 0
    print(
        f"  parallel: cpu_intra={cfg.intra_op} cpu_inter={cfg.inter_op} "
        f"mkldnn={cfg.mkldnn_enabled} mkl={cfg.mkl_enabled} "
        f"cuda={cuda} n_gpu={n_gpu}"
    )
    if dist is not None:
        print(
            f"            distributed: rank={dist.rank}/{dist.world_size} "
            f"backend={dist.backend} device={dist.device}"
        )


if __name__ == "__main__":
    # `python -m synapforge.parallel` smoke-test: print thread + device setup.
    print("=== synapforge.parallel ===")
    print_setup()
