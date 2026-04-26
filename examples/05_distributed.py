"""05_distributed.py — 2-GPU DDP smoke with PlasticBufferSync.

Run with torchrun:
    torchrun --nproc-per-node 2 examples/05_distributed.py

On a single-GPU box the script falls back to a single-rank simulation.
"""
import os
import torch
import torch.nn as nn
import synapforge as sf


class HybridBlock(sf.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.cfc = sf.LiquidCell(d, d)
        self.head = nn.Linear(d, d)

    def forward(self, x):
        return self.head(self.cfc(x))


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        rank, world_size, local_rank = sf.init_dist()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank = 0, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridBlock().to(device)

    if world_size > 1 and torch.cuda.is_available():
        model = sf.wrap_model(model, local_rank=local_rank, find_unused_parameters=False)
        sync = sf.PlasticBufferSync(model)
        print(f"rank{rank}  PlasticBufferSync watching "
              f"{sync.num_buffers()} buffers ({sync.numel():,} elements)")
    else:
        sync = None

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(20):
        x = torch.randn(4, 16, 32, device=device)
        out = model(x)
        loss = out.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sync is not None:
            sync.sync()
        if rank == 0 and step % 5 == 0:
            print(f"rank{rank}  step {step:3d}  loss = {loss.item():.5f}")

    if rank == 0:
        print("OK")


if __name__ == "__main__":
    main()
