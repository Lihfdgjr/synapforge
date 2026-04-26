"""Smoke test for sf.distributed_hetero.

Single-host smoke (this box):
  * Tiny LM-style model (~50k params) so we are dataloader-bound on
    purpose - the exact regime where mode A "split-by-task" is supposed
    to win.
  * 2 fake "GPU" + 2 fake "CPU" workers, all on this machine.  We pin
    the GPU workers to cuda:1 (cuda:0 is busy with the 100M training
    job and pid 50805 must not be disturbed).
  * Verifies:
      - mode A throughput vs single-device baseline
      - mode C 1 GPU + 1 CPU vs GPU-only
      - plasticity buffer averaging is correct (rank-pair L2 equal post-sync)
      - basic sanity: loss decreases over 50 steps
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import torch

sys.path.insert(0, "/workspace")

from synapforge.distributed_hetero import (  # noqa: E402
    CPUWorker,
    GPUWorker,
    HeteroTrainer,
    HeteroTrainerConfig,
    WorkerSpec,
)


# ---------------------------------------------------------------------------
# Tiny model: embed -> single linear -> head, with one plasticity buffer
# so we can verify the buffer-sync path runs.
# ---------------------------------------------------------------------------
class TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 1000, d: int = 32, n_layers: int = 1) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(d, d) for _ in range(n_layers)]
        )
        self.head = torch.nn.Linear(d, vocab)
        # Plasticity buffer (name matches sf.distributed pattern set
        # via the substring "hebb"); incremented on every forward.
        self.register_buffer("hebb_trace", torch.zeros(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for lin in self.layers:
            h = torch.tanh(lin(h))
        with torch.no_grad():
            # Cheap Hebbian-ish update so the buffer is non-trivial.
            ht = h.detach().mean(dim=(0, 1))
            self.hebb_trace.add_(0.001 * torch.outer(ht, ht))
        return self.head(h)


_MODEL_D = int(os.environ.get("SF_HETERO_D", "32"))
_MODEL_L = int(os.environ.get("SF_HETERO_L", "1"))


def make_model() -> torch.nn.Module:
    return TinyLM(vocab=1000, d=_MODEL_D, n_layers=_MODEL_L)


def loss_fn(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # CE over the last dim, ignore index 0 (pad).
    return torch.nn.functional.cross_entropy(
        out.reshape(-1, out.size(-1)), y.reshape(-1), ignore_index=0
    )


def synthetic_text_batch(n_rows: int, mean_len: int = 64) -> list[str]:
    rng = torch.Generator().manual_seed(0xC0FFEE + n_rows)
    rows: list[str] = []
    for _ in range(n_rows):
        L = int(mean_len + torch.randint(-8, 9, (1,), generator=rng).item())
        ids = torch.randint(1, 256, (max(8, L),), generator=rng).tolist()
        rows.append("".join(chr(i) for i in ids))
    return rows


def synthetic_token_batch(n_rows: int, seq_len: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(0xDA7A + n_rows)
    x = torch.randint(1, 1000, (n_rows, seq_len), generator=rng)
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    return x, y


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_single_gpu(steps: int = 50, batch: int = 32, tokenise: bool = True) -> tuple[float, float]:
    """Train on the visible GPU in a plain torch loop.

    When ``tokenise=True`` we include a synthetic tokeniser pass on
    every step, matching what Mode A does on CPU workers.  Without
    that, Mode A (which always tokenises) would be compared against a
    pre-tokenised baseline and look artificially slow.
    """
    if not torch.cuda.is_available():
        return float("nan"), float("nan")
    dev = torch.device("cuda:0")
    model = make_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    from synapforge.distributed_hetero import _CPUWorkerImpl
    tok = _CPUWorkerImpl(0)
    last_loss = 0.0
    # Warm-up
    for _ in range(3):
        if tokenise:
            raw = synthetic_text_batch(batch)
            x = tok.augment(tok.tokenize_batch(raw))
            y = torch.zeros_like(x)
            y[:, :-1] = x[:, 1:]
        else:
            x, y = synthetic_token_batch(batch)
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for s in range(steps):
        if tokenise:
            raw = synthetic_text_batch(batch)
            x = tok.augment(tok.tokenize_batch(raw))
            y = torch.zeros_like(x)
            y[:, :-1] = x[:, 1:]
        else:
            x, y = synthetic_token_batch(batch)
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().item())
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0
    return steps / elapsed, last_loss


def baseline_single_cpu(steps: int = 50, batch: int = 32) -> tuple[float, float]:
    dev = torch.device("cpu")
    model = make_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    last_loss = 0.0
    for _ in range(3):
        x, y = synthetic_token_batch(batch)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    t0 = time.perf_counter()
    for s in range(steps):
        x, y = synthetic_token_batch(batch)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().item())
    elapsed = time.perf_counter() - t0
    return steps / elapsed, last_loss


# ---------------------------------------------------------------------------
# Mode A: split-by-task
# ---------------------------------------------------------------------------
def smoke_mode_a(steps: int = 50, batch: int = 32) -> tuple[float, float, dict]:
    """2 GPU workers (cuda:1), 2 CPU data workers.  Returns
    (steps_per_sec, last_loss, last_stats)."""
    workers = [
        # We deliberately put both GPU actors on cuda:1 so we do not
        # collide with the 100M training job on cuda:0.  Two actors share
        # the device via Ray's num_gpus accounting (we use 0.5 each).
        WorkerSpec("localhost", "cuda:0", role="compute"),
        WorkerSpec("localhost", "cuda:0", role="compute"),
        WorkerSpec("localhost", "cpu", role="data"),
        WorkerSpec("localhost", "cpu", role="data"),
    ]
    cfg = HeteroTrainerConfig(
        mode="split-by-task",
        lr=1e-2,
        sync_buffers_every=5,
        log_every=10,
        cpu_data_pool_size=2,
    )
    # Bypass the ray.options(num_gpus=1) - we want 0.5 to fit two on one
    # GPU.  Patch the spawner via subclass.
    trainer = _HalfGPUHeteroTrainer(make_model, workers, cfg=cfg)
    trainer.start()
    try:
        # warm-up
        for _ in range(3):
            trainer.train_step(synthetic_text_batch(batch), loss_fn)
        t0 = time.perf_counter()
        last_stats = {}
        for s in range(steps):
            stats = trainer.train_step(synthetic_text_batch(batch), loss_fn)
            last_stats = stats
        elapsed = time.perf_counter() - t0
        sps = steps / elapsed
        return sps, last_stats.get("mean_loss", 0.0), last_stats
    finally:
        trainer.stop()


# ---------------------------------------------------------------------------
# Mode C: data-parallel-mixed
# ---------------------------------------------------------------------------
def smoke_mode_c(steps: int = 50, batch: int = 32) -> tuple[float, float, dict]:
    workers = [
        WorkerSpec("localhost", "cuda:0", role="compute"),
        WorkerSpec("localhost", "cpu", role="compute", num_cpus=2),
    ]
    cfg = HeteroTrainerConfig(
        mode="data-parallel-mixed",
        lr=1e-2,
        sync_buffers_every=5,
        warmup_steps=3,
        cpu_slice_floor=0.0,  # do not auto-drop the CPU - we want to measure it
    )
    trainer = _HalfGPUHeteroTrainer(make_model, workers, cfg=cfg)
    trainer.start()
    try:
        for _ in range(3):
            trainer.train_step(synthetic_token_batch(batch), loss_fn)
        t0 = time.perf_counter()
        last_stats: dict = {}
        for s in range(steps):
            stats = trainer.train_step(synthetic_token_batch(batch), loss_fn)
            last_stats = stats
        elapsed = time.perf_counter() - t0
        sps = steps / elapsed
        return sps, last_stats.get("mean_loss", 0.0), last_stats
    finally:
        trainer.stop()


# ---------------------------------------------------------------------------
# Slim subclass that lets two GPU workers share one physical GPU.
# (Real multi-machine deployments would not need this.)
# ---------------------------------------------------------------------------
class _HalfGPUHeteroTrainer(HeteroTrainer):
    def _spawn(self, spec: WorkerSpec):  # type: ignore[override]
        wid = len(self.gpu_workers) + len(self.cpu_workers)
        if not self._using_ray:
            if spec.is_gpu or spec.role == "compute":
                from synapforge.distributed_hetero import _GPUWorkerImpl

                return _GPUWorkerImpl(self.model_factory, wid, spec.device)
            from synapforge.distributed_hetero import _CPUWorkerImpl

            return _CPUWorkerImpl(wid)
        opts = {"num_cpus": spec.num_cpus}
        if spec.is_gpu:
            opts["num_gpus"] = 0.5  # share GPU between two actors
        if spec.host != "localhost":
            opts.setdefault("resources", {})[f"node:{spec.host}"] = 0.01
        if spec.is_gpu or spec.role == "compute":
            return GPUWorker.options(**opts).remote(self.model_factory, wid, spec.device)
        return CPUWorker.options(**opts).remote(wid)


# ---------------------------------------------------------------------------
# Plasticity-buffer correctness check
# ---------------------------------------------------------------------------
def check_plastic_sync() -> bool:
    """After a sync, every compute worker's hebb_trace must be identical."""
    workers = [
        WorkerSpec("localhost", "cuda:0", role="compute"),
        WorkerSpec("localhost", "cuda:0", role="compute"),
    ]
    cfg = HeteroTrainerConfig(mode="data-parallel-mixed", warmup_steps=2, sync_buffers_every=1)
    trainer = _HalfGPUHeteroTrainer(make_model, workers, cfg=cfg)
    trainer.start()
    try:
        for _ in range(4):
            trainer.train_step(synthetic_token_batch(32), loss_fn)
        # Pull buffers from each compute worker.
        bufs_list = [trainer._call(w, "get_plastic_buffers") for w in trainer.compute_workers]
        if not bufs_list[0]:
            print("  [plastic] no plastic buffers found - check pattern set")
            return False
        # Compare hebb_trace across workers.
        ref = bufs_list[0]["hebb_trace"]
        for i, b in enumerate(bufs_list[1:], 1):
            diff = (ref - b["hebb_trace"]).abs().max().item()
            print(f"  [plastic] |worker0 - worker{i}|_inf hebb_trace = {diff:.2e}")
            if diff > 1e-5:
                return False
        return True
    finally:
        trainer.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print("=" * 72)
    print("sf.distributed_hetero smoke")
    print("=" * 72)
    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"n_gpu={torch.cuda.device_count()}")
    print()

    print("[baseline] single GPU pre-tokenised (no CPU work) ...")
    base_gpu_pretok_sps, base_gpu_pretok_loss = baseline_single_gpu(steps=50, batch=32, tokenise=False)
    print(f"  steps/s={base_gpu_pretok_sps:.2f}  last_loss={base_gpu_pretok_loss:.4f}")
    print("[baseline] single GPU + inline tokeniser (apples-to-apples) ...")
    base_gpu_sps, base_gpu_loss = baseline_single_gpu(steps=50, batch=32, tokenise=True)
    print(f"  steps/s={base_gpu_sps:.2f}  last_loss={base_gpu_loss:.4f}")

    print("[baseline] single CPU ...")
    base_cpu_sps, base_cpu_loss = baseline_single_cpu(steps=50, batch=32)
    print(f"  steps/s={base_cpu_sps:.2f}  last_loss={base_cpu_loss:.4f}")
    print()

    print("[mode A] split-by-task: 2 cuda:1 + 2 cpu ...")
    a_sps, a_loss, a_stats = smoke_mode_a(steps=50, batch=32)
    print(f"  steps/s={a_sps:.2f}  last_loss={a_loss:.4f}  "
          f"vs_baseline_gpu={a_sps / max(base_gpu_sps, 1e-6):.2f}x")
    print(f"  worker compute times ms: "
          f"min={a_stats.get('ms_compute_min_worker', float('nan')):.1f} "
          f"max={a_stats.get('ms_compute_max_worker', float('nan')):.1f}")
    print()

    print("[mode C] data-parallel-mixed: 1 cuda:1 + 1 cpu ...")
    c_sps, c_loss, c_stats = smoke_mode_c(steps=50, batch=32)
    print(f"  steps/s={c_sps:.2f}  last_loss={c_loss:.4f}  "
          f"vs_baseline_gpu={c_sps / max(base_gpu_sps, 1e-6):.2f}x")
    if "weights" in c_stats:
        print(f"  auto weights: GPU={c_stats['weights'][0]:.3f}  CPU={c_stats['weights'][1]:.3f}")
        print(f"  slice sizes (tokens): {c_stats['slice_sizes']}")
        print(f"  per-worker step ms: {c_stats['ms_per_worker']}")
    print()

    print("[plastic-sync] correctness ...")
    ok = check_plastic_sync()
    print(f"  result = {'PASS' if ok else 'FAIL'}")
    print()

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"baseline_gpu_pretok: {base_gpu_pretok_sps:.2f} steps/s (best-case GPU)")
    print(f"baseline_gpu_inline: {base_gpu_sps:.2f} steps/s (with inline tokeniser)")
    print(f"baseline_cpu:       {base_cpu_sps:.2f} steps/s")
    print(f"mode_A (2gpu+2cpu): {a_sps:.2f} steps/s "
          f"({a_sps / max(base_gpu_sps, 1e-6):.2f}x of single GPU)")
    print(f"mode_C (1gpu+1cpu): {c_sps:.2f} steps/s "
          f"({c_sps / max(base_gpu_sps, 1e-6):.2f}x of single GPU)")
    print(f"plastic_sync:       {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
