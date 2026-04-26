"""03_plasticity.py — Hebbian + STDP rules with PlasticityEngine.

Demonstrates the modern, autograd-safe plasticity API: rules OBSERVE in
forward, deltas are computed AFTER the forward step, and PlasticityEngine
applies them atomically to weight tensors.

Run: python examples/03_plasticity.py
"""
import torch
import torch.nn as nn
import synapforge as sf


class PlasticBlock(sf.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.cfc = sf.LiquidCell(dim, dim)
        # Plastic weight: small random init so post != 0 from the start.
        self.register_buffer("W_plastic", 0.05 * torch.randn(dim, dim))
        self.dim = dim
        self.heb = sf.Hebbian(lr=1e-2)

    def forward(self, x):
        h = self.cfc(x)
        post = torch.relu(h @ self.W_plastic.t())
        # Observe traces from this forward.
        self.heb.observe(pre=h.flatten(0, 1), post=post.flatten(0, 1))
        return post


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = PlasticBlock(dim=32).to(device)

    engine = sf.PlasticityEngine(
        rules={"W_plastic": model.heb},
        schedule="every:1",
    )

    initial_norm = model.W_plastic.norm().item()
    print(f"device      : {device}")
    print(f"||W||_F     : {initial_norm:.4f}  (initial)")

    for step in range(50):
        x = torch.randn(4, 16, 32, device=device)
        _ = model(x)
        deltas = engine.step(t=step, weight_dict={"W_plastic": model.W_plastic})
        engine.apply(deltas, weight_dict={"W_plastic": model.W_plastic})

    final_norm = model.W_plastic.norm().item()
    print(f"||W||_F     : {final_norm:.4f}  (after 50 plastic steps)")
    assert final_norm > initial_norm, (
        f"Hebbian did not strengthen co-firing connections: "
        f"{initial_norm:.4f} -> {final_norm:.4f}"
    )
    print(f"Hebbian co-firing strengthened W by {final_norm/initial_norm:.2f}x")
    print("OK")


if __name__ == "__main__":
    main()
