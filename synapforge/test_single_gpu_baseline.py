"""1-GPU baseline for throughput comparison with DDP smoke."""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from test_distributed_smoke import TinyHybrid

torch.manual_seed(42)
device = torch.device("cuda:0")
torch.cuda.set_device(device)

D, L, B, T = 64, 2, 8, 16
model = TinyHybrid(d=D, layers=L).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
target = torch.randn(B, T, D, device=device)

# warmup
for _ in range(5):
    x = torch.randn(B, T, D, device=device)
    y = model(x); F.mse_loss(y, target).backward(); opt.step(); opt.zero_grad()

torch.cuda.synchronize(device)
t0 = time.time()
for _ in range(50):
    x = torch.randn(B, T, D, device=device)
    y = model(x); loss = F.mse_loss(y, target); loss.backward()
    opt.step(); opt.zero_grad()
torch.cuda.synchronize(device)
t1 = time.time()
print(f"[baseline] 1-GPU: 50 steps in {t1-t0:.3f}s = {(t1-t0)*1000/50:.2f} ms/step", flush=True)
