"""01_hello.py — minimal sf.Module end-to-end.

Run: python examples/01_hello.py
"""
import torch
import synapforge as sf


class Hello(sf.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.cell = sf.LiquidCell(in_dim, hidden)

    def forward(self, x):
        return self.cell(x)


def main() -> None:
    model = Hello(in_dim=8, hidden=16)
    x = torch.randn(2, 32, 8)
    y = model(x)
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(y.shape)}")
    print(f"params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"version: synapforge {sf.__version__}")
    assert y.shape == (2, 32, 16)
    print("OK")


if __name__ == "__main__":
    main()
