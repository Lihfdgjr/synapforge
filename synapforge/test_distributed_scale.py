"""Scale-up throughput test: bigger D so DDP's 2-GPU win shows up."""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from test_distributed_smoke import TinyHybrid

from synapforge import distributed as sfd


def main():
    rank, world, lr_ = sfd.init_dist()
    torch.cuda.set_device(lr_)
    device = torch.device(f"cuda:{lr_}")
    torch.manual_seed(42 + rank)
    D, L, B, T = 512, 4, 32, 64
    model = TinyHybrid(d=D, layers=L)
    model = sfd.wrap_model(model, lr_, find_unused_parameters=False)
    sync = sfd.PlasticBufferSync(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    target = torch.randn(B, T, D, device=device)
    # warmup
    for _ in range(5):
        x = torch.randn(B, T, D, device=device)
        F.mse_loss(model(x), target).backward(); opt.step(); opt.zero_grad(); sync.sync()
    torch.cuda.synchronize(device)
    t0 = time.time()
    for _ in range(30):
        x = torch.randn(B, T, D, device=device)
        F.mse_loss(model(x), target).backward(); opt.step(); opt.zero_grad(); sync.sync()
    torch.cuda.synchronize(device)
    t1 = time.time()
    if rank == 0:
        print(f"[scale] D={D} L={L} B={B} T={T}  2-GPU: {(t1-t0)*1000/30:.2f} ms/step", flush=True)
    sfd.cleanup_dist()

if __name__ == "__main__":
    main()
