"""Apples-to-apples throughput: 1-GPU at global B=64 vs 2-GPU at B=32/rank (=64 global)."""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from test_distributed_smoke import TinyHybrid

from synapforge import distributed as sfd

D, L, T = 512, 4, 64
GLOBAL_B = 64

if "RANK" in os.environ:  # torchrun mode
    rank, world, lr_ = sfd.init_dist()
    torch.cuda.set_device(lr_)
    device = torch.device(f"cuda:{lr_}")
    torch.manual_seed(42 + rank)
    B = GLOBAL_B // world  # split global batch across ranks
    model = TinyHybrid(d=D, layers=L)
    model = sfd.wrap_model(model, lr_, find_unused_parameters=False)
    sync = sfd.PlasticBufferSync(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    target = torch.randn(B, T, D, device=device)
    for _ in range(5):
        x = torch.randn(B, T, D, device=device)
        F.mse_loss(model(x), target).backward(); opt.step(); opt.zero_grad(); sync.sync()
    torch.cuda.synchronize(device); t0 = time.time()
    for _ in range(30):
        x = torch.randn(B, T, D, device=device)
        F.mse_loss(model(x), target).backward(); opt.step(); opt.zero_grad(); sync.sync()
    torch.cuda.synchronize(device); t1 = time.time()
    if rank == 0:
        ms = (t1 - t0) * 1000 / 30
        sps = GLOBAL_B / (ms / 1000)
        print(f"[2GPU] global_B={GLOBAL_B} per_rank_B={B}  ms/step={ms:.2f}  samples/s={sps:.1f}", flush=True)
    sfd.cleanup_dist()
else:  # single-GPU mode, full global batch on one GPU
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    torch.manual_seed(42)
    B = GLOBAL_B
    model = TinyHybrid(d=D, layers=L).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    target = torch.randn(B, T, D, device=device)
    for _ in range(5):
        x = torch.randn(B, T, D, device=device); F.mse_loss(model(x), target).backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(device); t0 = time.time()
    for _ in range(30):
        x = torch.randn(B, T, D, device=device); F.mse_loss(model(x), target).backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(device); t1 = time.time()
    ms = (t1 - t0) * 1000 / 30; sps = B / (ms / 1000)
    print(f"[1GPU] global_B={GLOBAL_B}  ms/step={ms:.2f}  samples/s={sps:.1f}", flush=True)
