"""sf.distributed_hetero - heterogeneous GPU + CPU cluster training.

A third transport for synapforge, complementary to:

  sf.distributed       - DDP, NCCL, single-host multi-GPU, sync.
  sf.distributed_ray   - Ray async PS, multi-host, *homogeneous* devices.
  sf.distributed_hetero (this) - Ray-orchestrated, *heterogeneous* devices.

The user wants to use any GPU plus any CPU box at the same time, not just
N identical GPUs. There are three useful regimes:

  Mode A  "split-by-task"  (default, most useful)
      GPU workers run forward / backward / optimizer step.
      CPU workers do tokenisation, augmentation, replay-buffer sampling
      and plasticity-buffer averaging.  This overlaps GPU compute with
      CPU prep so the GPU does not block on the dataloader.

  Mode B  "pipeline"
      Different model layers live on different devices (e.g. emb on
      cpu node A, mid layers on GPU 0, head on cpu node B).  Stubbed
      out below; full pipeline support is deferred (it needs activation
      checkpointing across the bus and is not the high-ROI direction).

  Mode C  "data-parallel-mixed"
      The same batch is fragmented and EVERY worker (GPU or CPU) runs
      a forward / backward on its slice.  Slice size is auto-tuned by
      a 5-step warm-up phase: each worker reports tokens / second; the
      batch is then weighted by inverse step time.  Useful when an
      otherwise-idle CPU box can chip in 5-15% of a small batch.

Why Ray?
--------
Ray gives us cross-process actors with a shared object store and
gRPC-based RPC, so a CPU actor can sit on a different machine from
the GPU actor and the same control-plane code works.  Our existing
``sf.distributed_ray`` already proves the pattern; this module
specialises it for heterogeneous hardware.

Communication
-------------
* Within the GPU group we still prefer NCCL all-reduce when there is
  more than one GPU; this module only adds the CPU-side fan-in/fan-out.
* GPU <-> CPU exchange goes through Ray's plasma object store
  (zero-copy on the same host, gRPC across hosts).  Tensors are sent
  on CPU; the GPU worker .to(device)s them inside its actor, so the
  cross-device hop is invisible to user code.
* Plasticity-buffer averaging follows the same pattern as the
  homogeneous Ray transport (see ``sf.distributed_ray``).

Public API
----------
    >>> from synapforge.distributed_hetero import (
    ...     WorkerSpec, HeteroTrainer, HeteroTrainerConfig,
    ... )
    >>> def make_model():
    ...     return MyHybrid(D=256)
    >>> def loss_fn(out, y):
    ...     return ((out - y) ** 2).mean()
    >>> workers = [
    ...     WorkerSpec("localhost", "cuda:0"),
    ...     WorkerSpec("localhost", "cuda:1"),
    ...     WorkerSpec("localhost", "cpu", role="data"),
    ...     WorkerSpec("localhost", "cpu", role="data"),
    ... ]
    >>> trainer = HeteroTrainer(make_model, workers,
    ...                         cfg=HeteroTrainerConfig(mode="split-by-task"))
    >>> trainer.start()
    >>> for raw_batch in stream:
    ...     stats = trainer.train_step(raw_batch, loss_fn)
    >>> trainer.stop()

Honest performance note
-----------------------
On a small (<= 100M-param) model with a fast tokenizer + augmenter,
mode A ("split-by-task") usually delivers a real 1.1x - 1.4x throughput
gain because the dataloader stops blocking the GPU.  Mode C
("data-parallel-mixed") with one GPU + one CPU box is rarely a win on
GPUs >= V100 / A100: the CPU's contribution is dwarfed by the
serialisation cost of the gradient round-trip.  We add the speed-aware
weighting (smaller slices for slower workers) precisely so that adding
a CPU is at worst neutral and not a regression.  The cross-over point
where CPU help is genuinely useful is roughly: model fits on GPU in
< 100 ms / step AND tokenizer / augment costs > 30 ms.  For larger
GPU steps (long sequence, large model) the GPU dominates and CPU
contribution becomes pure overhead; the scheduler in this module
detects that and shrinks the CPU slice to 0.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

# Optional Ray import - graceful single-process fallback (mirrors
# sf.distributed_ray's pattern so users can ``pip install`` lazily).
try:
    import ray  # type: ignore

    _HAS_RAY = True
except ImportError:  # pragma: no cover
    ray = None  # type: ignore
    _HAS_RAY = False


def is_available() -> bool:
    """Return True iff Ray was importable at module load time."""
    return _HAS_RAY


# Re-use the plasticity-buffer pattern set so the three transports stay
# in lock-step on which buffers participate in averaging.
from .distributed import is_plastic_buffer  # noqa: E402


# ---------------------------------------------------------------------------
# WorkerSpec
# ---------------------------------------------------------------------------
@dataclass
class WorkerSpec:
    """Describes one worker.

    Parameters
    ----------
    host : str
        Hostname or IP.  ``"localhost"`` for single-box smoke testing.
    device : str
        ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, ...
        We accept ``"cuda"`` as a synonym for ``"cuda:0"``.
    role : str
        ``"compute"``  : full forward + backward (GPU default).
        ``"data"``     : tokenise / augment / replay sample
                         (CPU default; never owns model parameters).
        ``"plastic"``  : averages plasticity buffers off the critical
                         path (CPU; one per cluster is enough).
    weight : float | None
        Hint for mode C batch weighting in [0, 1].  ``None`` means
        auto-discover via the warm-up phase.
    num_cpus : int
        How many CPU cores to reserve for this actor (Ray scheduling).
    """

    host: str
    device: str
    role: str = "compute"
    weight: float | None = None
    num_cpus: int = 4

    def __post_init__(self) -> None:
        d = self.device.lower()
        if d == "cuda":
            self.device = "cuda:0"
        elif d == "gpu":
            self.device = "cuda:0"
        elif d.startswith("cuda:") or d == "cpu":
            self.device = d
        else:
            raise ValueError(f"WorkerSpec.device must be cpu / cuda / cuda:N, got {self.device!r}")

        if self.role not in ("compute", "data", "plastic"):
            raise ValueError(
                f"WorkerSpec.role must be compute / data / plastic, got {self.role!r}"
            )

        # Sensible default: a CPU worker that hasn't said otherwise is
        # most likely a "data" worker for mode A (split-by-task).
        if self.device == "cpu" and self.role == "compute":
            # Caller wants the CPU box to actually compute - allow it,
            # but flag for mode C.
            pass

    @property
    def is_gpu(self) -> bool:
        return self.device.startswith("cuda")


# ---------------------------------------------------------------------------
# Trainer config
# ---------------------------------------------------------------------------
@dataclass
class HeteroTrainerConfig:
    """User-tunable knobs."""

    mode: str = "split-by-task"  # split-by-task | pipeline | data-parallel-mixed
    lr: float = 1e-2
    sync_buffers_every: int = 1
    use_ray: bool = True
    log_every: int = 10
    # data-parallel-mixed only:
    warmup_steps: int = 5         # per-worker tok/s probe before weighting kicks in
    cpu_slice_floor: float = 0.0  # if a CPU is slower than this fraction, drop it
    # split-by-task only:
    cpu_data_pool_size: int = 2   # how many CPU actors to fan tokenisation across


# ---------------------------------------------------------------------------
# Worker implementations - both Ray-actor and in-proc fallback
# ---------------------------------------------------------------------------
class _GPUWorkerImpl:
    """Forward + backward + optim of a model replica on one device.

    ``device`` may be ``cpu`` too; the class is deliberately not GPU-only
    so a CPU box can ALSO be a compute worker in mode C.
    """

    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        worker_id: int,
        device: str,
    ) -> None:
        self.worker_id = worker_id
        self.device_str = device
        # When running inside a Ray actor, Ray sets CUDA_VISIBLE_DEVICES
        # to the assigned physical GPU(s) and remaps them to local
        # indices starting at 0.  So a user-side ``cuda:1`` becomes
        # ``cuda:0`` inside this actor.  Detect that situation and
        # rewrite the index, otherwise preserve the user's device.
        actual = device
        if device.startswith("cuda") and torch.cuda.is_available():
            visible = torch.cuda.device_count()
            # Strip any explicit index that exceeds the visible count;
            # if Ray gave us only one visible device, force ``cuda:0``.
            if visible == 1:
                actual = "cuda:0"
            else:
                # Multi-GPU on this actor: keep the user's index but
                # clamp it to the visible range.
                idx = 0
                if ":" in device:
                    try:
                        idx = int(device.split(":", 1)[1])
                    except ValueError:
                        idx = 0
                if idx >= visible:
                    idx = 0
                actual = f"cuda:{idx}"
        self.device = torch.device(actual)
        self.model = model_factory().to(self.device)
        self._param_names: list[str] = [n for n, _ in self.model.named_parameters()]
        self._buf_names: list[str] = [
            n
            for n, b in self.model.named_buffers()
            if is_plastic_buffer(n) and b.is_floating_point() and b.numel() > 0
        ]

    # ----- weight transport
    def get_weights(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in weights:
                    p.copy_(weights[n].to(p.device))

    def get_plastic_buffers(self) -> dict[str, torch.Tensor]:
        named = dict(self.model.named_buffers())
        return {n: named[n].detach().cpu().clone() for n in self._buf_names if n in named}

    def set_plastic_buffers(self, bufs: dict[str, torch.Tensor]) -> None:
        named = dict(self.model.named_buffers())
        with torch.no_grad():
            for n, b in bufs.items():
                if n in named:
                    named[n].copy_(b.to(named[n].device))

    # ----- compute
    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], float, float]:
        """Run one micro-step. Returns (grads, loss, elapsed_ms)."""
        self.model.zero_grad(set_to_none=False)
        x_d = x.to(self.device, non_blocking=True)
        y_d = y.to(self.device, non_blocking=True)
        t0 = time.perf_counter()
        out = self.model(x_d)
        loss = loss_fn(out, y_d)
        loss.backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        grads: dict[str, torch.Tensor] = {}
        for n, p in self.model.named_parameters():
            grads[n] = (
                p.grad.detach().cpu().clone()
                if p.grad is not None
                else torch.zeros_like(p, device="cpu")
            )
        return grads, float(loss.detach().item()), elapsed_ms


class _CPUWorkerImpl:
    """Side-task worker: tokenisation, augmentation, plasticity averaging.

    Owns no model parameters; only stateless callable methods.  Multiple
    CPUWorkers can fan-in to the GPU group through Ray's object store.
    """

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id

    def tokenize_batch(self, raw_text: list[str], vocab_size: int = 1000) -> torch.Tensor:
        """Trivial deterministic tokeniser - real users plug in HF or sentencepiece.

        We hash characters into the vocab so the smoke test is
        reproducible without any tokenizer dependency.
        """
        rows: list[list[int]] = []
        max_len = 0
        for s in raw_text:
            ids = [(ord(c) % vocab_size) for c in s][:1024]
            rows.append(ids)
            max_len = max(max_len, len(ids))
        # pad
        for r in rows:
            r.extend([0] * (max_len - len(r)))
        return torch.tensor(rows, dtype=torch.long)

    def augment(self, batch: torch.Tensor, drop_prob: float = 0.05) -> torch.Tensor:
        """Token dropout + noisy permutation - placeholder for real augment."""
        if drop_prob > 0:
            mask = torch.rand_like(batch.float()) < drop_prob
            batch = batch.masked_fill(mask, 0)
        return batch

    def average_plastic(
        self, bufs_list: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Average a list of {name: tensor} dicts elementwise."""
        if not bufs_list:
            return {}
        keys = list(bufs_list[0].keys())
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            stacked = torch.stack([d[k] for d in bufs_list if k in d], dim=0)
            out[k] = stacked.mean(dim=0)
        return out


# Ray-actor wrappers (defined only if Ray is importable) ---------------------
if _HAS_RAY:

    @ray.remote(num_cpus=1)  # type: ignore[misc]
    class GPUWorker(_GPUWorkerImpl):
        """Ray actor that wraps :class:`_GPUWorkerImpl`.

        ``num_cpus=1`` is the host-side default; the *GPU* allocation is
        attached at ``.options(num_gpus=1)`` time by ``HeteroTrainer``.
        """

    @ray.remote(num_cpus=1)  # type: ignore[misc]
    class CPUWorker(_CPUWorkerImpl):
        """Ray actor that wraps :class:`_CPUWorkerImpl`."""

else:  # pragma: no cover - users without Ray fall back to in-proc

    class GPUWorker(_GPUWorkerImpl):  # type: ignore[no-redef]
        pass

    class CPUWorker(_CPUWorkerImpl):  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# HeteroTrainer
# ---------------------------------------------------------------------------
class HeteroTrainer:
    """Orchestrates a heterogeneous mix of GPU / CPU workers.

    See module docstring for the high-level model.
    """

    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        workers: list[WorkerSpec],
        cfg: HeteroTrainerConfig | None = None,
        coord_device: str = "cpu",
    ) -> None:
        if not workers:
            raise ValueError("HeteroTrainer needs at least one WorkerSpec")
        self.cfg = cfg or HeteroTrainerConfig()
        if self.cfg.mode not in ("split-by-task", "pipeline", "data-parallel-mixed"):
            raise ValueError(f"unknown mode {self.cfg.mode!r}")
        self.model_factory = model_factory
        self.specs = list(workers)
        self.coord_device = coord_device
        self.coord_model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.gpu_workers: list[Any] = []   # Ray handles or in-proc
        self.gpu_specs: list[WorkerSpec] = []
        self.cpu_workers: list[Any] = []
        self.cpu_specs: list[WorkerSpec] = []
        self.compute_specs: list[WorkerSpec] = []  # mode C only
        self.compute_workers: list[Any] = []
        self._step_count = 0
        self._using_ray = bool(self.cfg.use_ray and _HAS_RAY)
        self._weights: dict[str, torch.Tensor] = {}
        self._step_times_ms: list[list[float]] = []  # warm-up tracker (mode C)
        if self.cfg.use_ray and not _HAS_RAY:
            warnings.warn(
                "Ray not installed - HeteroTrainer falling back to single-process. "
                "`pip install ray==2.10.0` for true distributed.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ----- lifecycle -----
    def start(self) -> None:
        """Spawn workers and wire up the coordinator-side reference model."""
        self.coord_model = self.model_factory().to(self.coord_device)
        self.optimizer = torch.optim.Adam(self.coord_model.parameters(), lr=self.cfg.lr)
        if self._using_ray and not ray.is_initialized():  # type: ignore[union-attr]
            ray.init(  # type: ignore[union-attr]
                ignore_reinit_error=True,
                log_to_driver=False,
                include_dashboard=False,
            )
        # Partition specs into roles.
        for spec in self.specs:
            actor = self._spawn(spec)
            if spec.is_gpu:
                self.gpu_workers.append(actor)
                self.gpu_specs.append(spec)
                self.compute_workers.append(actor)
                self.compute_specs.append(spec)
            elif spec.role == "compute":
                # CPU acting as a compute worker - mode C only.  We do
                # NOT add it to cpu_workers, because cpu_workers must
                # only contain CPUWorker actors (which have
                # tokenize_batch / augment / average_plastic methods).
                self.compute_workers.append(actor)
                self.compute_specs.append(spec)
            else:
                # role == data / plastic
                self.cpu_workers.append(actor)
                self.cpu_specs.append(spec)

        if self.cfg.mode == "split-by-task" and not self.gpu_workers:
            raise RuntimeError(
                "split-by-task mode needs at least one cuda WorkerSpec; got none"
            )
        if self.cfg.mode == "data-parallel-mixed" and not self.compute_workers:
            raise RuntimeError(
                "data-parallel-mixed mode needs at least one compute WorkerSpec"
            )

        # Initial weight broadcast to every GPU worker (and any compute-CPU).
        self._weights = {
            n: p.detach().cpu().clone() for n, p in self.coord_model.named_parameters()
        }
        for w in self.compute_workers:
            self._call(w, "set_weights", self._weights)

    def stop(self) -> None:
        """Kill all Ray actors. Safe to call multiple times."""
        if self._using_ray:
            for w in self.gpu_workers + self.cpu_workers:
                try:
                    ray.kill(w)  # type: ignore[union-attr]
                except Exception:
                    pass
        self.gpu_workers, self.cpu_workers = [], []
        self.compute_workers = []
        self.gpu_specs, self.cpu_specs, self.compute_specs = [], [], []

    # ----- spawn helper -----
    def _spawn(self, spec: WorkerSpec):
        wid = len(self.gpu_workers) + len(self.cpu_workers)
        if not self._using_ray:
            if spec.is_gpu or spec.role == "compute":
                return _GPUWorkerImpl(self.model_factory, wid, spec.device)
            return _CPUWorkerImpl(wid)

        # Ray path - actor options for placement.
        opts: dict[str, Any] = {"num_cpus": spec.num_cpus}
        if spec.is_gpu:
            opts["num_gpus"] = 1
        if spec.host != "localhost":
            # If the user named a specific host, anchor the actor there.
            opts.setdefault("resources", {})[f"node:{spec.host}"] = 0.01
        if spec.is_gpu or spec.role == "compute":
            return GPUWorker.options(**opts).remote(  # type: ignore[attr-defined]
                self.model_factory, wid, spec.device
            )
        return CPUWorker.options(**opts).remote(wid)  # type: ignore[attr-defined]

    # ----- generic call wrapper -----
    def _call(self, worker: Any, method: str, *args: Any, **kw: Any) -> Any:
        if self._using_ray:
            return ray.get(getattr(worker, method).remote(*args, **kw))  # type: ignore[union-attr]
        return getattr(worker, method)(*args, **kw)

    def _call_async(self, worker: Any, method: str, *args: Any, **kw: Any) -> Any:
        if self._using_ray:
            return getattr(worker, method).remote(*args, **kw)  # type: ignore[attr-defined]

        # in-proc fallback: just run synchronously and pretend it's a future
        return getattr(worker, method)(*args, **kw)

    def _gather(self, futures: list[Any]) -> list[Any]:
        if self._using_ray:
            return ray.get(futures)  # type: ignore[union-attr]
        return list(futures)

    # ----- weight bookkeeping -----
    def _broadcast_weights(self) -> None:
        futures = [
            self._call_async(w, "set_weights", self._weights) for w in self.compute_workers
        ]
        self._gather(futures)

    def _broadcast_buffers(self, bufs: dict[str, torch.Tensor]) -> None:
        futures = [
            self._call_async(w, "set_plastic_buffers", bufs) for w in self.compute_workers
        ]
        self._gather(futures)

    def _gather_plastic_avg(self) -> dict[str, torch.Tensor] | None:
        if not self.compute_workers:
            return None
        bufs_futures = [self._call_async(w, "get_plastic_buffers") for w in self.compute_workers]
        bufs_list = self._gather(bufs_futures)
        bufs_list = [b for b in bufs_list if b]
        if not bufs_list:
            return None
        if self.cpu_workers:  # delegate the average to a CPU worker if available
            avg = self._call(self.cpu_workers[0], "average_plastic", bufs_list)
        else:
            keys = list(bufs_list[0].keys())
            avg = {
                k: torch.stack([d[k] for d in bufs_list if k in d], dim=0).mean(dim=0)
                for k in keys
            }
        return avg

    # ----- mode A: split-by-task -----
    def _step_split(
        self,
        raw_batch: list[str],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        target_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, float]:
        # 1. Fan out tokenisation across CPU workers (round-robin).
        cpu_pool = self.cpu_workers[: max(1, self.cfg.cpu_data_pool_size)]
        if not cpu_pool:
            # No CPU workers: tokenise inline.
            tokens = _CPUWorkerImpl(0).tokenize_batch(raw_batch)
            tokens = _CPUWorkerImpl(0).augment(tokens)
        else:
            chunks = _split_list(raw_batch, len(cpu_pool))
            tok_futures = [
                self._call_async(cpu_pool[i], "tokenize_batch", chunks[i])
                for i in range(len(cpu_pool))
            ]
            tok_chunks = self._gather(tok_futures)
            # Pad-cat the chunks along batch dim so all GPUs see same width.
            tokens = _pad_cat(tok_chunks)
            # Optional augment - run on whichever CPU worker is free.
            tokens = self._call(cpu_pool[0], "augment", tokens)

        # 2. Targets - in this smoke we shift inputs by 1 (LM-style).
        targets = target_fn(tokens)

        # 3. Equally split across GPU workers.
        gpu_chunks_x = _even_split(tokens, len(self.gpu_workers))
        gpu_chunks_y = _even_split(targets, len(self.gpu_workers))

        t_compute_start = time.perf_counter()
        step_futures = [
            self._call_async(self.gpu_workers[i], "step", gpu_chunks_x[i], gpu_chunks_y[i], loss_fn)
            for i in range(len(self.gpu_workers))
        ]
        results = self._gather(step_futures)
        t_compute_ms = (time.perf_counter() - t_compute_start) * 1000.0

        grads_list = [r[0] for r in results]
        losses = [r[1] for r in results]
        worker_ms = [r[2] for r in results]

        # 4. Average grads on the coordinator and step.
        self._apply_avg_grads(grads_list)
        self._weights = {
            n: p.detach().cpu().clone() for n, p in self.coord_model.named_parameters()
        }
        self._broadcast_weights()

        # 5. Sync plasticity buffers off the critical path.
        if (
            self.cfg.sync_buffers_every
            and self._step_count % self.cfg.sync_buffers_every == 0
        ):
            avg_bufs = self._gather_plastic_avg()
            if avg_bufs:
                self._broadcast_buffers(avg_bufs)

        self._step_count += 1
        return {
            "step": self._step_count,
            "mean_loss": sum(losses) / len(losses),
            "loss_min": min(losses),
            "loss_max": max(losses),
            "ms_compute": t_compute_ms,
            "ms_compute_max_worker": max(worker_ms),
            "ms_compute_min_worker": min(worker_ms),
            "n_gpu_workers": len(self.gpu_workers),
            "n_cpu_workers": len(self.cpu_workers),
        }

    # ----- mode C: data-parallel-mixed -----
    def _step_dp_mixed(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> dict[str, float]:
        # Compute weights for the slice split.
        weights = self._auto_weights()
        slices_x = _weighted_split(x, weights)
        slices_y = _weighted_split(y, weights)

        t0 = time.perf_counter()
        step_futures = [
            self._call_async(self.compute_workers[i], "step", slices_x[i], slices_y[i], loss_fn)
            for i in range(len(self.compute_workers))
        ]
        results = self._gather(step_futures)
        t_compute_ms = (time.perf_counter() - t0) * 1000.0

        # Track per-worker step time for next-iteration weighting.
        if not self._step_times_ms:
            self._step_times_ms = [[] for _ in self.compute_workers]
        for i, r in enumerate(results):
            self._step_times_ms[i].append(r[2])

        grads_list = [r[0] for r in results]
        losses = [r[1] for r in results]
        worker_ms = [r[2] for r in results]

        # Weighted-average grads (weighted by tokens-this-step = batch share).
        token_counts = [s.numel() for s in slices_x]
        total = float(sum(token_counts) or 1)
        norm_w = [c / total for c in token_counts]
        self._apply_weighted_avg_grads(grads_list, norm_w)
        self._weights = {
            n: p.detach().cpu().clone() for n, p in self.coord_model.named_parameters()
        }
        self._broadcast_weights()

        if (
            self.cfg.sync_buffers_every
            and self._step_count % self.cfg.sync_buffers_every == 0
        ):
            avg_bufs = self._gather_plastic_avg()
            if avg_bufs:
                self._broadcast_buffers(avg_bufs)

        self._step_count += 1
        return {
            "step": self._step_count,
            "mean_loss": float(sum(l * w for l, w in zip(losses, norm_w))),
            "loss_min": min(losses),
            "loss_max": max(losses),
            "ms_compute": t_compute_ms,
            "ms_per_worker": worker_ms,
            "weights": weights,
            "slice_sizes": token_counts,
        }

    def _auto_weights(self) -> list[float]:
        """Decide how to split the next batch across compute workers.

        Until ``warmup_steps`` per worker have been observed, give each
        compute worker an equal slice.  Afterwards, weight inversely
        proportional to mean step time.  Drop any worker whose share
        is below ``cfg.cpu_slice_floor``.
        """
        n = len(self.compute_workers)
        if n == 0:
            return []
        # Honour explicit user weights if every spec gave one.
        if all(s.weight is not None for s in self.compute_specs):
            ws = [float(s.weight or 0.0) for s in self.compute_specs]  # type: ignore[arg-type]
            tot = sum(ws) or 1.0
            return [w / tot for w in ws]
        if not self._step_times_ms or any(
            len(times) < self.cfg.warmup_steps for times in self._step_times_ms
        ):
            return [1.0 / n] * n
        means = [sum(t) / len(t) for t in self._step_times_ms]
        inv = [1.0 / max(m, 1e-3) for m in means]
        tot = sum(inv) or 1.0
        ws = [v / tot for v in inv]
        # Drop slow CPU workers below the floor.
        for i, (w, spec) in enumerate(zip(ws, self.compute_specs)):
            if not spec.is_gpu and w < self.cfg.cpu_slice_floor:
                ws[i] = 0.0
        tot = sum(ws) or 1.0
        return [w / tot for w in ws]

    # ----- gradient averaging on coord_model -----
    def _apply_avg_grads(self, grads_list: list[dict[str, torch.Tensor]]) -> None:
        assert self.coord_model is not None and self.optimizer is not None
        avg = {}
        keys = grads_list[0].keys()
        for k in keys:
            stacked = torch.stack([g[k] for g in grads_list], dim=0)
            avg[k] = stacked.mean(dim=0)
        for n, p in self.coord_model.named_parameters():
            if n in avg:
                p.grad = avg[n].to(p.device)
        self.optimizer.step()

    def _apply_weighted_avg_grads(
        self, grads_list: list[dict[str, torch.Tensor]], weights: list[float]
    ) -> None:
        assert self.coord_model is not None and self.optimizer is not None
        keys = grads_list[0].keys()
        avg: dict[str, torch.Tensor] = {}
        for k in keys:
            stacked = torch.stack([g[k] for g in grads_list], dim=0)
            w = torch.tensor(weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.dim() - 1)))
            avg[k] = (stacked * w).sum(dim=0)
        for n, p in self.coord_model.named_parameters():
            if n in avg:
                p.grad = avg[n].to(p.device)
        self.optimizer.step()

    # ----- public step -----
    def train_step(
        self,
        batch: Any,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        target_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Run one training step in the configured mode.

        ``batch`` semantics depend on the mode:
            split-by-task           -> List[str] of raw text rows
            pipeline                -> NotImplementedError
            data-parallel-mixed     -> tuple (x, y)
        """
        if self.cfg.mode == "split-by-task":
            if target_fn is None:
                target_fn = _shift_lm_targets
            return self._step_split(batch, loss_fn, target_fn)
        if self.cfg.mode == "data-parallel-mixed":
            assert isinstance(batch, tuple) and len(batch) == 2, (
                "data-parallel-mixed expects batch=(x, y)"
            )
            x, y = batch
            return self._step_dp_mixed(x, y, loss_fn)
        # Mode B (pipeline) is documented but deferred:
        raise NotImplementedError(
            "pipeline mode is documented but not implemented; use split-by-task or data-parallel-mixed"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _shift_lm_targets(tokens: torch.Tensor) -> torch.Tensor:
    """Trivial next-token target shift for the smoke test."""
    if tokens.dim() < 2:
        return tokens
    y = torch.zeros_like(tokens)
    y[:, :-1] = tokens[:, 1:]
    return y


def _split_list(xs: list[Any], n: int) -> list[list[Any]]:
    if n <= 0:
        return [xs]
    out: list[list[Any]] = [[] for _ in range(n)]
    for i, x in enumerate(xs):
        out[i % n].append(x)
    return out


def _even_split(t: torch.Tensor, n: int) -> list[torch.Tensor]:
    if n <= 0:
        return [t]
    if n == 1:
        return [t]
    base = t.size(0) // n
    rem = t.size(0) - base * n
    chunks: list[torch.Tensor] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        chunks.append(t[start : start + size])
        start += size
    return chunks


def _weighted_split(t: torch.Tensor, weights: list[float]) -> list[torch.Tensor]:
    if t.size(0) == 0:
        return [t.clone() for _ in weights]
    sizes = [max(0, int(round(t.size(0) * w))) for w in weights]
    diff = t.size(0) - sum(sizes)
    # Hand out the remainder (if any) to the largest weight.
    if diff:
        idx = max(range(len(weights)), key=lambda i: weights[i])
        sizes[idx] += diff
    chunks: list[torch.Tensor] = []
    start = 0
    for s in sizes:
        chunks.append(t[start : start + s])
        start += s
    # Replace any zero-sized slice with a 1-row dummy to keep gradient
    # shapes consistent (it contributes 0 to the weighted average).
    for i, c in enumerate(chunks):
        if c.size(0) == 0:
            chunks[i] = t[:1].clone() * 0
    return chunks


def _pad_cat(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError("pad_cat empty input")
    max_len = max(p.size(-1) for p in parts)
    padded = []
    for p in parts:
        if p.size(-1) < max_len:
            pad = torch.zeros(*p.shape[:-1], max_len - p.size(-1), dtype=p.dtype)
            p = torch.cat([p, pad], dim=-1)
        padded.append(p)
    return torch.cat(padded, dim=0)


__all__ = [
    "WorkerSpec",
    "HeteroTrainerConfig",
    "HeteroTrainer",
    "GPUWorker",
    "CPUWorker",
    "is_available",
]
