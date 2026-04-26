"""1-GPU baseline at the bigger scale."""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from test_distributed_smoke import TinyHybrid

torch.manual_seed(42)
device = torch.device("cuda:0"); torch.cuda.set_device(device)
D, L, B, T = 512, 4, 32, 64
model = TinyHybrid(d=D, layers=L).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.001)
target = torch.randn(B, T, D, device=device)
for _ in range(5):
    x = torch.randn(B, T, D, device=device); F.mse_loss(model(x), target).backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize(device); t0 = time.time()
for _ in range(30):
    x = torch.randn(B, T, D, device=device); F.mse_loss(model(x), target).backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize(device); t1 = time.time()
print(f"[scale_baseline] D={D} L={L} B={B} T={T}  1-GPU: {(t1-t0)*1000/30:.2f} ms/step", flush=True)
