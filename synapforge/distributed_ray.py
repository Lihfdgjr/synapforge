"""sf.distributed_ray — async multi-machine distributed training via Ray.

v1.0 milestone M9. Complementary to ``sf.distributed`` (DDP):

  sf.distributed     synchronous all-reduce, one-box, NCCL/Gloo
                     ⇒ best for single 8-GPU node (low launch overhead)
  sf.distributed_ray asynchronous parameter-server, multi-box, Ray
                     ⇒ best for multi-machine (cross-node bandwidth bound)

Architecture
------------
  +-----------+      grad      +-----------+
  | Worker 0  |  ----------->  |           |
  | (replica) |  <----- W ---  |           |
  +-----------+                |    Coord  |  <-- AsyncTrainer
  +-----------+                |  (PS-like)|
  | Worker 1  |  ----------->  |           |
  | (replica) |  <----- W ---  |           |
  +-----------+                +-----------+
  ...

Each ``RayWorker`` (Ray actor) holds one model replica. Workflow per step:
  1. coordinator broadcasts current weights to all workers
  2. each worker computes gradient on its data slice (in parallel)
  3. coordinator gathers gradients, averages them, applies optim.step
  4. plasticity buffers are also averaged (same pattern as DDP path)

Falls back gracefully if Ray is missing — ``AsyncTrainer.run`` then walks
the same training loop in a single process so existing code keeps working.

Ray async vs DDP — when does each win?
--------------------------------------
* DDP: synchronous, NCCL all-reduce, ~1-3 ms/step overhead on a single
  8-GPU box. Wins when GPUs share NVLink + you want bit-exact reproducibility.
* Ray async: process-isolated, gRPC + plasma object store, ~5-15 ms/step
  overhead per round-trip on localhost. Wins when:
    - workers live on DIFFERENT machines (NCCL needs network bring-up,
      Ray brings-its-own object store)
    - workers have heterogeneous compute (Ray queues stragglers)
    - elastic scaling: add/remove a worker mid-training without restarting
* On a single-host 1-2 GPU box: DDP almost always faster.
* On a 4+ node cluster: Ray async can be 2-5x faster wall-clock because
  it overlaps gradient compute with weight pull (DDP's all-reduce blocks).

This implementation does *synchronous* gradient averaging across N workers
(closer to "Ray-distributed DDP" than true async parameter-server). Pure
async is easier to layer on top later via `RayWorker.compute_grad_async`.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Optional Ray import — graceful fallback if the user hasn't pip-installed.
# ---------------------------------------------------------------------------

try:
    import ray  # type: ignore
    _HAS_RAY = True
except ImportError:  # pragma: no cover - tested via env without ray
    ray = None  # type: ignore
    _HAS_RAY = False


def is_available() -> bool:
    """Return True iff Ray was importable at module load."""
    return _HAS_RAY


# ---------------------------------------------------------------------------
# Plasticity-buffer pattern matching (re-export from sf.distributed for
# consistency — both transports must agree on which buffers to average).
# ---------------------------------------------------------------------------

from .distributed import _PLASTIC_PATTERNS, is_plastic_buffer  # noqa: E402


# ---------------------------------------------------------------------------
# Worker (Ray actor when Ray is available, otherwise an in-proc shim)
# ---------------------------------------------------------------------------


class _WorkerImpl:
    """Plain-Python worker logic, used both by RayWorker and by the
    single-process fallback. Decoupled from ``ray.remote`` so the same
    code path tests in both regimes.
    """

    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        worker_id: int,
        device: str = "cpu",
    ) -> None:
        self.worker_id = worker_id
        self.device = torch.device(device)
        self.model = model_factory().to(self.device)
        # Cached parameter-name list — used to flatten/unflatten gradients
        # for transport. Fixed at init (Ray actors don't see structural
        # mutations on the coordinator).
        self._param_names: List[str] = [
            n for n, _ in self.model.named_parameters()
        ]
        self._buf_names: List[str] = [
            n for n, b in self.model.named_buffers()
            if is_plastic_buffer(n) and b.is_floating_point() and b.numel() > 0
        ]

    # --------------------------------------------------------------- weights

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return a copy of all parameter tensors (CPU, for transport)."""
        return {
            n: p.detach().cpu().clone()
            for n, p in self.model.named_parameters()
        }

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """In-place copy of weights into local model."""
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in weights:
                    p.copy_(weights[n].to(p.device))

    def get_plastic_buffers(self) -> Dict[str, torch.Tensor]:
        """Return CPU copies of plasticity buffers, for averaging."""
        named = dict(self.model.named_buffers())
        return {
            n: named[n].detach().cpu().clone()
            for n in self._buf_names
            if n in named
        }

    def set_plastic_buffers(self, bufs: Dict[str, torch.Tensor]) -> None:
        """In-place copy of averaged buffers back into local model."""
        named = dict(self.model.named_buffers())
        with torch.no_grad():
            for n, b in bufs.items():
                if n in named:
                    named[n].copy_(b.to(named[n].device))

    # --------------------------------------------------------------- compute

    def compute_grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Forward + backward on this worker's data slice.

        Returns (flat dict of grads keyed by param name, loss value).
        Caller is responsible for averaging across workers.
        """
        self.model.zero_grad(set_to_none=False)
        x_d = x.to(self.device)
        y_d = y.to(self.device)
        out = self.model(x_d)
        loss = loss_fn(out, y_d)
        loss.backward()
        grads: Dict[str, torch.Tensor] = {}
        for n, p in self.model.named_parameters():
            if p.grad is None:
                grads[n] = torch.zeros_like(p, device="cpu")
            else:
                grads[n] = p.grad.detach().cpu().clone()
        return grads, float(loss.detach().item())


if _HAS_RAY:

    @ray.remote(num_cpus=1)  # type: ignore[misc]
    class RayWorker(_WorkerImpl):
        """Ray actor wrapping :class:`_WorkerImpl`.

        Subclassing rather than composition so all methods of _WorkerImpl
        are exposed automatically as Ray remote methods, and the actor
        can be pickled to a remote node identically to local invocation.
        """

else:

    class RayWorker(_WorkerImpl):  # type: ignore[no-redef]
        """Stub for environments without Ray. Behaves like _WorkerImpl,
        just lets ``isinstance(w, RayWorker)`` checks work both ways.
        """


# ---------------------------------------------------------------------------
# Async trainer
# ---------------------------------------------------------------------------


@dataclass
class AsyncTrainerConfig:
    """User-tunable knobs for AsyncTrainer."""

    n_workers: int = 4
    lr: float = 1e-2
    sync_buffers_every: int = 1  # how often to all-reduce plasticity buffers
    use_ray: bool = True          # True -> Ray actors; False -> single-proc fallback
    log_every: int = 10


class AsyncTrainer:
    """Coordinates N RayWorkers via gradient-averaging parameter-server.

    Usage
    -----
        def make_model():
            return MyHybrid(D=64)

        def loss_fn(out, y):
            return ((out - y) ** 2).mean()

        trainer = AsyncTrainer(model_factory=make_model,
                               cfg=AsyncTrainerConfig(n_workers=4))
        trainer.start()
        for step, (xs_per_worker, ys_per_worker) in enumerate(loader):
            stats = trainer.step(xs_per_worker, ys_per_worker, loss_fn)
        trainer.stop()
    """

    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        cfg: Optional[AsyncTrainerConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or AsyncTrainerConfig()
        self.model_factory = model_factory
        self.device = device
        self.workers: List = []
        self.coord_model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._step_count = 0
        self._using_ray = bool(self.cfg.use_ray and _HAS_RAY)
        if self.cfg.use_ray and not _HAS_RAY:
            warnings.warn(
                "Ray not installed — AsyncTrainer falling back to single-process. "
                "`pip install ray==2.10.0` for true distributed.",
                RuntimeWarning,
                stacklevel=2,
            )

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Spawn workers + build coordinator-side reference model+optim."""
        # Coordinator holds a model copy whose parameters are the canonical
        # (averaged) weights. Workers pull from it before each step.
        self.coord_model = self.model_factory().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.coord_model.parameters(), lr=self.cfg.lr,
        )

        if self._using_ray:
            if not ray.is_initialized():  # type: ignore[union-attr]
                ray.init(  # type: ignore[union-attr]
                    ignore_reinit_error=True,
                    log_to_driver=False,
                    include_dashboard=False,
                )
            self.workers = [
                RayWorker.remote(  # type: ignore[attr-defined]
                    self.model_factory, i, "cpu",
                )
                for i in range(self.cfg.n_workers)
            ]
        else:
            self.workers = [
                _WorkerImpl(self.model_factory, i, "cpu")
                for i in range(self.cfg.n_workers)
            ]

        # Initialize all workers with the coordinator's weights.
        weights = {n: p.detach().cpu().clone()
                   for n, p in self.coord_model.named_parameters()}
        self._broadcast_weights(weights)

    def stop(self) -> None:
        """Tear down Ray actors. Safe to call multiple times."""
        if self._using_ray and self.workers:
            for w in self.workers:
                try:
                    ray.kill(w)  # type: ignore[union-attr]
                except Exception:
                    pass
            self.workers = []

    # --------------------------------------------------------------- transport

    def _broadcast_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Push the same set of weights into every worker."""
        if self._using_ray:
            handles = [w.set_weights.remote(weights) for w in self.workers]  # type: ignore[attr-defined]
            ray.get(handles)  # type: ignore[union-attr]
        else:
            for w in self.workers:
                w.set_weights(weights)

    def _broadcast_buffers(self, bufs: Dict[str, torch.Tensor]) -> None:
        if self._using_ray:
            handles = [w.set_plastic_buffers.remote(bufs) for w in self.workers]  # type: ignore[attr-defined]
            ray.get(handles)  # type: ignore[union-attr]
        else:
            for w in self.workers:
                w.set_plastic_buffers(bufs)

    def _gather_grads(
        self,
        xs: List[torch.Tensor],
        ys: List[torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
        """Fan out compute_grad to N workers, gather grads + losses."""
        if self._using_ray:
            handles = [
                self.workers[i].compute_grad.remote(xs[i], ys[i], loss_fn)  # type: ignore[attr-defined]
                for i in range(self.cfg.n_workers)
            ]
            results = ray.get(handles)  # type: ignore[union-attr]
        else:
            results = [
                self.workers[i].compute_grad(xs[i], ys[i], loss_fn)
                for i in range(self.cfg.n_workers)
            ]
        grads = [r[0] for r in results]
        losses = [r[1] for r in results]
        return grads, losses

    def _gather_buffers(self) -> List[Dict[str, torch.Tensor]]:
        if self._using_ray:
            handles = [w.get_plastic_buffers.remote() for w in self.workers]  # type: ignore[attr-defined]
            return ray.get(handles)  # type: ignore[union-attr]
        return [w.get_plastic_buffers() for w in self.workers]

    # --------------------------------------------------------------- step

    def step(
        self,
        xs_per_worker: List[torch.Tensor],
        ys_per_worker: List[torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Dict[str, float]:
        """One gradient step:

        broadcast W -> compute grads -> avg -> optim.step -> avg buffers
        """
        assert self.coord_model is not None and self.optimizer is not None, (
            "AsyncTrainer.start() must be called before step()"
        )
        if len(xs_per_worker) != self.cfg.n_workers:
            raise ValueError(
                f"got {len(xs_per_worker)} input slices for {self.cfg.n_workers} workers"
            )

        t0 = time.perf_counter()
        # 1) gather grads
        grads_list, losses = self._gather_grads(xs_per_worker, ys_per_worker, loss_fn)
        t1 = time.perf_counter()

        # 2) average grads, write into coord_model.grad
        avg_grads: Dict[str, torch.Tensor] = {}
        for k in grads_list[0]:
            stacked = torch.stack([g[k] for g in grads_list], dim=0)
            avg_grads[k] = stacked.mean(dim=0)

        for n, p in self.coord_model.named_parameters():
            if n in avg_grads:
                p.grad = avg_grads[n].to(p.device)

        # 3) optim.step
        self.optimizer.step()

        # 4) broadcast new weights
        new_weights = {
            n: p.detach().cpu().clone()
            for n, p in self.coord_model.named_parameters()
        }
        self._broadcast_weights(new_weights)
        t2 = time.perf_counter()

        # 5) average plasticity buffers across workers periodically.
        if self.cfg.sync_buffers_every and (
            self._step_count % self.cfg.sync_buffers_every == 0
        ):
            bufs_list = self._gather_buffers()
            if bufs_list and bufs_list[0]:
                avg_bufs: Dict[str, torch.Tensor] = {}
                for k in bufs_list[0]:
                    stacked = torch.stack([b[k] for b in bufs_list], dim=0)
                    avg_bufs[k] = stacked.mean(dim=0)
                self._broadcast_buffers(avg_bufs)

        t3 = time.perf_counter()
        self._step_count += 1

        return {
            "step": self._step_count,
            "mean_loss": sum(losses) / len(losses),
            "loss_min": min(losses),
            "loss_max": max(losses),
            "ms_compute": (t1 - t0) * 1000,
            "ms_optim_bcast": (t2 - t1) * 1000,
            "ms_buf_sync": (t3 - t2) * 1000,
            "ms_total": (t3 - t0) * 1000,
        }


__all__ = [
    "RayWorker",
    "AsyncTrainer",
    "AsyncTrainerConfig",
    "is_available",
]
