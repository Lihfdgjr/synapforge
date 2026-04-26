"""bench_surrogate — compare ATan / sigmoid / triangle / fast_sigmoid / SLAYER
on a tiny LIF training task.

Task: temporal XOR with spikes. Two binary inputs are presented as Poisson
spike trains over T steps. The network — a 2->H spiking hidden + linear
readout from spike rates — must output the XOR.

Reports per surrogate:
    * Final accuracy on held-out batch.
    * Steps to reach 90% accuracy (or "n/a" if it never gets there).
    * Mean grad-norm on the readout over training.
    * Wall-clock time.

Run: /opt/conda/bin/python /workspace/synapforge/bench_surrogate.py
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from synapforge.surrogate import PLIFCell


def make_xor_batch(B: int, T: int, rate_hi: float = 0.5, rate_lo: float = 0.05,
                   device: str = "cpu", generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """B samples of (input spikes [B,T,2], xor label [B])."""
    g = generator
    a = torch.randint(0, 2, (B,), device=device, generator=g)
    b = torch.randint(0, 2, (B,), device=device, generator=g)
    rates = torch.stack([
        torch.where(a == 1, torch.full_like(a, 0).float() + rate_hi,
                    torch.full_like(a, 0).float() + rate_lo),
        torch.where(b == 1, torch.full_like(b, 0).float() + rate_hi,
                    torch.full_like(b, 0).float() + rate_lo),
    ], dim=1)  # (B, 2)
    rates = rates.unsqueeze(1).expand(B, T, 2)
    spikes = (torch.rand(B, T, 2, device=device, generator=g) < rates).float()
    y = (a ^ b).long()
    return spikes, y


class SpikeXOR(nn.Module):
    def __init__(self, hidden: int = 32, surrogate: str = "atan", alpha: float = 2.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(2, hidden)
        self.cell = PLIFCell(hidden, tau_init=8.0, threshold_init=0.5,
                             surrogate=surrogate, alpha=alpha)
        self.readout = nn.Linear(hidden, 2)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # (B, T, 2) -> (B, T, hidden)
        cur = self.in_proj(x_seq)
        s_seq, _ = self.cell.forward_seq(cur)
        rate = s_seq.mean(dim=1)  # (B, hidden) — population rate
        return self.readout(rate)


def train_one(name: str, *, alpha: float = 2.0, steps: int = 600, B: int = 64, T: int = 25,
              hidden: int = 32, lr: float = 1e-2, device: str = "cpu", seed: int = 0) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = SpikeXOR(hidden=hidden, surrogate=name, alpha=alpha).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    grad_norms: list[float] = []
    steps_to_90: int | None = None
    losses: list[float] = []
    t0 = time.time()
    for step in range(steps):
        x, y = make_xor_batch(B, T, device=device, generator=gen)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        # Track readout grad norm (a proxy for surrogate signal strength).
        gn = model.readout.weight.grad.norm().item()
        grad_norms.append(gn)
        opt.step()
        losses.append(loss.item())

        # Eval acc every 25 steps with a fresh batch
        if step % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                xv, yv = make_xor_batch(256, T, device=device, generator=gen)
                acc = (model(xv).argmax(1) == yv).float().mean().item()
            if steps_to_90 is None and acc >= 0.9:
                steps_to_90 = step
    elapsed = time.time() - t0

    # Final acc, big batch
    with torch.no_grad():
        xv, yv = make_xor_batch(1024, T, device=device, generator=gen)
        final_acc = (model(xv).argmax(1) == yv).float().mean().item()

    return {
        "name": name,
        "final_acc": final_acc,
        "steps_to_90": steps_to_90,
        "mean_grad_norm": sum(grad_norms) / len(grad_norms),
        "final_loss": losses[-1],
        "elapsed_s": elapsed,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bench] device={device} torch={torch.__version__}")
    results = []
    for surrogate in ["atan", "sigmoid", "triangle", "fast_sigmoid", "slayer"]:
        r = train_one(surrogate, device=device, seed=42)
        results.append(r)
        s90 = r["steps_to_90"] if r["steps_to_90"] is not None else "n/a"
        print(
            f"  {surrogate:14s} acc={r['final_acc']*100:5.1f}%  steps>=90%={str(s90):>5s}  "
            f"meanGN={r['mean_grad_norm']:.4f}  loss={r['final_loss']:.4f}  "
            f"t={r['elapsed_s']:.1f}s"
        )
    converged = [r for r in results if r["steps_to_90"] is not None]
    if converged:
        winner = min(converged, key=lambda r: r["steps_to_90"])
        print(f"\n[bench] fastest convergence to 90%: {winner['name']} @ step {winner['steps_to_90']}")
    else:
        winner = max(results, key=lambda r: r["final_acc"])
        print(f"\n[bench] none reached 90%; best final acc: {winner['name']} @ {winner['final_acc']*100:.1f}%")


if __name__ == "__main__":
    main()
