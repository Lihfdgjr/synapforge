"""sf.distributed - multi-GPU DDP with plasticity-buffer-sync.

Solves the buffer-sync problem PyTorch DDP doesn't:
  STDP / Hebbian / AstrocyteGate buffers must average across ranks
  after each step, otherwise each rank's plasticity drifts.

Why vanilla DDP is not enough
-----------------------------
DDP all-reduces *gradients* of `nn.Parameter`s, but plasticity state in
synapforge lives in `register_buffer` (W_fast, calcium, stdp_trace,
astro_state, fast_mem, hebb, elig, coact_ema...). PyTorch's
``broadcast_buffers=True`` only broadcasts rank-0 buffers to the rest -
this overwrites rank-1's locally-updated plasticity with rank-0's,
which is *worse* than divergence (it discards half the data).

The right fix: ``broadcast_buffers=False`` (so DDP doesn't touch them)
plus an explicit ``all_reduce(AVG)`` of plasticity buffers after the
optimizer step. This is what `PlasticBufferSync` does.

Public API
----------
    >>> import synapforge as sf
    >>> import os
    >>> sf.distributed.init_dist()                         # torchrun env
    >>> rank = int(os.environ["LOCAL_RANK"])
    >>> model = MyHybrid(D=512, L=12)
    >>> ddp_model = sf.distributed.wrap_model(model, rank)
    >>> sync = sf.distributed.PlasticBufferSync(ddp_model)
    >>> for x, y in loader:
    ...     loss = ddp_model(x, y)
    ...     loss.backward()
    ...     opt.step(); opt.zero_grad()
    ...     sync.sync()                                    # <- the magic line

Launch
------
    torchrun --nproc-per-node=2 my_train.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Plasticity-buffer pattern matching
# ---------------------------------------------------------------------------
# These substring patterns identify buffers that participate in plasticity
# updates and therefore *diverge* across ranks (each rank sees a different
# data shard so each rank's STDP trace, calcium concentration, etc. drifts).
# Vanilla DDP's gradient all-reduce does not touch these because they are
# `register_buffer`, not `nn.Parameter`. We must average them ourselves.
#
# The list is intentionally a substring match so submodule prefixes
# (`block_3.plif.last_spike_rate`, `triton_block.elig`) all match.
_PLASTIC_PATTERNS: tuple[str, ...] = (
    "W_fast",          # plasticity.STDP / plasticity.Oja fast weights
    "calcium",         # neuromodulation calcium dynamics
    "burst_count",     # burst-firing neuron history
    "stdp_trace",      # spike-timing eligibility traces
    "astro_state",     # astrocyte gate hidden state
    "fast_mem",        # fast-weight episodic memory
    "hebb",            # Hebbian buffers / co-activation
    "elig",            # triton_block_kernel eligibility accumulator
    "coact_ema",       # co-activation exponential moving average
    "last_spike_rate", # PLIF online spike-rate monitor
)


def is_plastic_buffer(name: str) -> bool:
    """Return True if `name` matches any synapforge plasticity-buffer pattern."""
    return any(p in name for p in _PLASTIC_PATTERNS)


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------
def init_dist(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize torch.distributed from torchrun-set env vars.

    Returns
    -------
    (rank, world_size, local_rank)
    """
    if not dist.is_initialized():
        # torchrun guarantees these are set; fail loudly otherwise.
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "init_dist() requires torchrun-set env vars "
                "(RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT). "
                "Launch with: torchrun --nproc-per-node=N your_script.py"
            )
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    return rank, world_size, local_rank


def cleanup_dist() -> None:
    """Tear down distributed group; safe to call from any rank."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------
def wrap_model(
    model: torch.nn.Module,
    local_rank: int,
    find_unused_parameters: bool = True,
    static_graph: bool = False,
) -> DDP:
    """Move model to `cuda:local_rank` and wrap with DDP.

    `broadcast_buffers=False` is **critical**: with it on, PyTorch would
    broadcast rank-0's plasticity buffers to all other ranks at every
    forward, *overwriting* their locally-updated state. We do the
    averaging ourselves via `PlasticBufferSync`.

    `find_unused_parameters=True` because synapforge's HybridBlock has
    branches that may be skipped (PLIF on a non-spiking forward, gated
    plasticity heads, etc.). Set False for ~5-10% speedup if your model
    uses every parameter every step.
    """
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,             # <- THE critical flag
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        gradient_as_bucket_view=True,        # cheap memory win
    )


# ---------------------------------------------------------------------------
# Buffer enumeration
# ---------------------------------------------------------------------------
def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP wrapper if present."""
    return model.module if hasattr(model, "module") else model


def get_plastic_buffers(model: torch.nn.Module) -> list[torch.Tensor]:
    """Enumerate all floating-point plasticity buffers in `model`.

    Filters:
      - name matches `_PLASTIC_PATTERNS`
      - tensor is floating-point (skip int spike counts which would need
        a different reduction; users who want them averaged can wrap them
        in fp tensors themselves)
      - tensor.numel() > 0 (skip placeholder zero-d tensors that some
        backends register before dim is known)
    """
    raw = _unwrap(model)
    out: list[torch.Tensor] = []
    for name, buf in raw.named_buffers():
        if not is_plastic_buffer(name):
            continue
        if not buf.is_floating_point():
            continue
        if buf.numel() == 0:
            continue
        out.append(buf)
    return out


def get_plastic_buffer_names(model: torch.nn.Module) -> list[str]:
    """Same as `get_plastic_buffers` but returns names. Debug helper."""
    raw = _unwrap(model)
    return [
        n for n, b in raw.named_buffers()
        if is_plastic_buffer(n) and b.is_floating_point() and b.numel() > 0
    ]


# ---------------------------------------------------------------------------
# Plasticity buffer synchronization
# ---------------------------------------------------------------------------
class PlasticBufferSync:
    """Averages plasticity buffers across DDP ranks.

    Call `sync()` after `optimizer.step()` (so each rank's local update
    has already happened). One flat all-reduce is faster than per-buffer
    all-reduces on NCCL because of fixed launch overhead per collective.

    Notes
    -----
    * Uses `dist.ReduceOp.AVG` (NCCL >=2.10, PyTorch >=1.10). If your
      stack lacks AVG, falls back to SUM-then-divide.
    * The all-reduce is on a *flat* buffer assembled from all plasticity
      tensors. We use `torch._utils._flatten_dense_tensors` /
      `_unflatten_dense_tensors` (same primitives DDP uses internally)
      so heterogeneous shapes/dtypes within one bucket are handled.
    * Re-enumerates buffers lazily once at init. If you mutate the
      module structure (add/remove submodules) at runtime, call
      `refresh()`.

    Example
    -------
    >>> ddp_model = sf.distributed.wrap_model(model, local_rank)
    >>> sync = sf.distributed.PlasticBufferSync(ddp_model)
    >>> for batch in loader:
    ...     loss = ddp_model(batch).loss
    ...     loss.backward()
    ...     optim.step(); optim.zero_grad()
    ...     sync.sync()
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.bufs: list[torch.Tensor] = get_plastic_buffers(model)
        self._has_avg_op = self._probe_avg_op()

    @staticmethod
    def _probe_avg_op() -> bool:
        """Detect whether dist.ReduceOp.AVG is available."""
        try:
            _ = dist.ReduceOp.AVG  # type: ignore[attr-defined]
            return True
        except (AttributeError, RuntimeError):
            return False

    def refresh(self) -> None:
        """Re-scan the model for plasticity buffers (use after structural changes)."""
        self.bufs = get_plastic_buffers(self.model)

    def num_buffers(self) -> int:
        return len(self.bufs)

    def numel(self) -> int:
        return sum(b.numel() for b in self.bufs)

    @torch.no_grad()
    def sync(self) -> None:
        """All-reduce(AVG) the plasticity buffers across ranks. No-op single-GPU."""
        if not dist.is_initialized():
            return
        if not self.bufs:
            return

        world_size = dist.get_world_size()
        if world_size == 1:
            return

        # Group by dtype (NCCL all-reduce wants a single dtype per call).
        from collections import defaultdict
        by_dtype: dict[torch.dtype, list[torch.Tensor]] = defaultdict(list)
        for b in self.bufs:
            by_dtype[b.dtype].append(b)

        for dtype, group in by_dtype.items():
            data = [b.data for b in group]
            flat = torch._utils._flatten_dense_tensors(data)
            if self._has_avg_op:
                dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            else:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(world_size)
            for b, t in zip(
                group,
                torch._utils._unflatten_dense_tensors(flat, data),
            ):
                b.data.copy_(t)


# ---------------------------------------------------------------------------
# Convenience: barrier with timeout warning
# ---------------------------------------------------------------------------
def barrier(name: str = "sync") -> None:
    """Distributed barrier with a friendly tag for debugging hangs."""
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Diagnostic: divergence probe (call from tests)
# ---------------------------------------------------------------------------
@torch.no_grad()
def buffer_l2_per_rank(model: torch.nn.Module) -> dict[str, float]:
    """Return {buffer_name: this_rank's L2 norm}.

    Use as a divergence probe: gather across ranks and compare. After
    `PlasticBufferSync.sync()` all ranks must show identical L2.
    """
    raw = _unwrap(model)
    out: dict[str, float] = {}
    for n, b in raw.named_buffers():
        if is_plastic_buffer(n) and b.is_floating_point() and b.numel() > 0:
            out[n] = b.detach().float().norm().item()
    return out


__all__ = [
    "init_dist",
    "cleanup_dist",
    "wrap_model",
    "get_plastic_buffers",
    "get_plastic_buffer_names",
    "is_plastic_buffer",
    "PlasticBufferSync",
    "barrier",
    "buffer_l2_per_rank",
]
