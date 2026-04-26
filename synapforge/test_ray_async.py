"""test_ray_async — verify M9 (Ray async distributed).

Three checks:

  1. Ray-async path: 4 workers, 100 steps, loss decreases.
  2. Single-process fallback: same model, same steps, also converges.
  3. Convergence parity: Ray async final loss within reasonable band of
     the single-proc baseline (not bit-equal — averaging order differs).

Throughput: report ms/step for 1-worker and 4-worker. On a single host,
4-worker should NOT beat 1-worker by 4x — Ray IPC dominates. The test
documents this honestly.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/workspace")

from synapforge.distributed_ray import (
    AsyncTrainer,
    AsyncTrainerConfig,
)
from synapforge.distributed_ray import (
    is_available as ray_available,
)


# ---------------------------------------------------------------------------
# Tiny model with a plasticity-like buffer so we exercise the buffer-sync path.
# ---------------------------------------------------------------------------
class TinyModel(nn.Module):
    def __init__(self, d_in: int = 32, d_hidden: int = 64, d_out: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        # Plasticity-named buffer to test buffer-sync. Pattern matches
        # `coact_ema` in synapforge.distributed._PLASTIC_PATTERNS.
        self.register_buffer("coact_ema", torch.zeros(d_hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        # Mutate the plasticity buffer so cross-worker averaging is observable.
        with torch.no_grad():
            self.coact_ema.mul_(0.9).add_(h.detach().mean(dim=0) * 0.1)
        return self.fc2(h)


def make_model() -> nn.Module:
    torch.manual_seed(42)
    return TinyModel()


def loss_fn(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((out - y) ** 2).mean()


def make_data(n_workers: int, batch: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    xs = [torch.randn(batch, 32) for _ in range(n_workers)]
    ys = [torch.randn(batch, 8) for _ in range(n_workers)]
    return xs, ys


# ---------------------------------------------------------------------------
# Run one trainer for `n_steps` steps; return (losses, mean_step_ms).
# ---------------------------------------------------------------------------
def train_loop(trainer: AsyncTrainer, n_steps: int, n_workers: int):
    losses = []
    times = []
    for step in range(n_steps):
        xs, ys = make_data(n_workers, batch=8, seed=step)
        stats = trainer.step(xs, ys, loss_fn)
        losses.append(stats["mean_loss"])
        times.append(stats["ms_total"])
        if step % 20 == 0 or step == n_steps - 1:
            print(f"  step {step:3d}  loss={stats['mean_loss']:.4f}  "
                  f"ms/step={stats['ms_total']:.2f}")
    return losses, sum(times) / len(times)


def main() -> int:
    print(f"[ray installed] {ray_available()}")

    # --------- Test 1: 1-worker baseline (Ray async, n=1) ---------
    print("\n=== Test 1: 1-worker Ray-async baseline ===")
    cfg1 = AsyncTrainerConfig(n_workers=1, lr=1e-2, use_ray=True)
    t1 = AsyncTrainer(model_factory=make_model, cfg=cfg1)
    t1.start()
    try:
        losses_1, mean_ms_1 = train_loop(t1, n_steps=100, n_workers=1)
    finally:
        t1.stop()
    final_1 = losses_1[-1]

    # --------- Test 2: 4-worker Ray-async ---------
    print("\n=== Test 2: 4-worker Ray-async ===")
    cfg4 = AsyncTrainerConfig(n_workers=4, lr=1e-2, use_ray=True)
    t4 = AsyncTrainer(model_factory=make_model, cfg=cfg4)
    t4.start()
    try:
        losses_4, mean_ms_4 = train_loop(t4, n_steps=100, n_workers=4)
    finally:
        t4.stop()
    final_4 = losses_4[-1]

    # --------- Test 3: Single-process fallback (Ray-off) ---------
    print("\n=== Test 3: Single-process fallback (use_ray=False) ===")
    cfg_sp = AsyncTrainerConfig(n_workers=4, lr=1e-2, use_ray=False)
    t_sp = AsyncTrainer(model_factory=make_model, cfg=cfg_sp)
    t_sp.start()
    try:
        losses_sp, mean_ms_sp = train_loop(t_sp, n_steps=100, n_workers=4)
    finally:
        t_sp.stop()
    final_sp = losses_sp[-1]

    # --------- Convergence + throughput report ---------
    print("\n=== Results ===")
    print(f"  Ray  1-worker : init_loss={losses_1[0]:.4f}  "
          f"final={final_1:.4f}  ms/step={mean_ms_1:.2f}")
    print(f"  Ray  4-worker : init_loss={losses_4[0]:.4f}  "
          f"final={final_4:.4f}  ms/step={mean_ms_4:.2f}")
    print(f"  Single-proc 4w: init_loss={losses_sp[0]:.4f}  "
          f"final={final_sp:.4f}  ms/step={mean_ms_sp:.2f}")

    # All three runs must converge. Random batches each step ⇒ use a
    # smoothed window (mean of first 10 vs mean of last 10) instead of
    # comparing single noisy step-end values.
    def conv(losses):
        return sum(losses[-10:]) / 10 < sum(losses[:10]) / 10
    assert conv(losses_1), (
        f"Ray 1-worker did not converge: "
        f"first10={sum(losses_1[:10])/10:.3f} last10={sum(losses_1[-10:])/10:.3f}"
    )
    assert conv(losses_4), (
        f"Ray 4-worker did not converge: "
        f"first10={sum(losses_4[:10])/10:.3f} last10={sum(losses_4[-10:])/10:.3f}"
    )
    assert conv(losses_sp), (
        f"Single-proc fallback did not converge: "
        f"first10={sum(losses_sp[:10])/10:.3f} last10={sum(losses_sp[-10:])/10:.3f}"
    )

    # Ray 4-worker and single-proc should land in same loss neighborhood
    # since gradient averaging is mathematically identical (only the
    # transport differs).
    rel_diff = abs(final_4 - final_sp) / max(abs(final_sp), 1e-6)
    print(f"\n  Ray vs single-proc final-loss rel.diff: {rel_diff:.4f}")
    assert rel_diff < 0.30, (
        f"Ray and single-proc diverged too much "
        f"(final_4={final_4:.4f}, final_sp={final_sp:.4f})"
    )

    # Throughput honesty
    speedup_ray = mean_ms_1 / mean_ms_4 if mean_ms_4 > 0 else 0.0
    print(f"\n[throughput] Ray 1w vs 4w speedup = {speedup_ray:.2f}x")
    print("  (single-host: Ray IPC overhead caps speedup. On multi-node, "
          "Ray async beats DDP because it overlaps weight pull with compute.)")

    print("\n=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
