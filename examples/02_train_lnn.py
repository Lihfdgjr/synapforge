"""02_train_lnn.py — train a single LiquidCell on toy regression.

Learns a sine target from its delta-encoded input. Single-GPU or CPU.
Runs in ~5 seconds on CPU.

Run: python examples/02_train_lnn.py
"""
import torch
import torch.nn as nn
import synapforge as sf


class TinyLNN(sf.Module):
    def __init__(self, in_dim: int = 1, hidden: int = 16, out_dim: int = 1):
        super().__init__()
        self.cell = sf.LiquidCell(in_dim, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):                  # x: (B, T, in_dim)
        return self.out(self.cell(x))      # (B, T, out_dim)


def make_batch(B: int = 8, T: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0.0, 6.28, T).unsqueeze(0).repeat(B, 1)
    phase = torch.rand(B, 1) * 6.28
    target = torch.sin(t + phase)                              # bounded in [-1, 1]
    inp = torch.cat([torch.zeros(B, 1), target.diff(dim=1)], dim=1)
    inp = inp.unsqueeze(-1) + 0.02 * torch.randn(B, T, 1)
    return inp, target.unsqueeze(-1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = TinyLNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    print(f"device : {device}")
    print(f"params : {sum(p.numel() for p in model.parameters()):,}")

    losses: list[float] = []
    for step in range(150):
        x, y = make_batch()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(loss.item())
        if step % 25 == 0 or step == 149:
            print(f"step {step:4d}  loss = {loss.item():.5f}")

    assert losses[-1] < losses[0], (
        f"training did not converge: start={losses[0]:.4f}  end={losses[-1]:.4f}"
    )
    print(f"converged: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("OK")


if __name__ == "__main__":
    main()
