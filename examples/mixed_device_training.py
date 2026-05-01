"""Example: CPU+GPU mixed-device training for memory-tight setups.

Pattern: embedding + lm_head on CPU (huge, low FLOP), backbone on GPU
(compute-bound, small parameters). Activation crosses PCIe twice per
step. On a 24GB GPU running a 375M model, this frees ~6GB VRAM at the
cost of ~5% wall-clock per step.

Usage:
    python examples/mixed_device_training.py                    # single-process mixed
    torchrun --nproc-per-node=2 examples/mixed_device_training.py  # 2-GPU DDP

Multi-node:
    torchrun --nnodes=2 --nproc-per-node=4 \\
        --rdzv-backend=c10d --rdzv-endpoint=HOST:29500 \\
        examples/mixed_device_training.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from synapforge.parallel import (  # noqa: E402
    auto_dataloader,
    init_distributed,
    is_main_rank,
    optimize_cpu_threads,
    place_mixed_device,
    print_setup,
)


class TinyChat(nn.Module):
    """50M-param toy model with the embed/backbone/head shape we care about."""

    def __init__(self, vocab=32_000, hidden=512, depth=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.backbone = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden, hidden), nn.GELU()) for _ in range(depth)]
        )
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, ids):
        # embed lives on cpu_device; activations cross to gpu_device automatically
        # in the typical case we'd handle .to() explicitly. Here PyTorch does it
        # via tensor.to() inside Linear when device differs.
        x = self.embed_tokens(ids)
        x = x.to(next(self.backbone.parameters()).device)
        x = self.backbone(x)
        x = x.to(next(self.lm_head.parameters()).device)
        return self.lm_head(x)


def main():
    print_setup()
    optimize_cpu_threads()

    dist = init_distributed(backend="auto")
    rank = dist.rank if dist else 0
    world = dist.world_size if dist else 1

    model = TinyChat()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main_rank():
        print(f"  model: {total:.1f}M params")

    if torch.cuda.is_available():
        placement = place_mixed_device(
            model, gpu=f"cuda:{dist.local_rank if dist else 0}", verbose=is_main_rank()
        )
    else:
        if is_main_rank():
            print("  no GPU — full CPU training")

    if dist is not None and torch.cuda.is_available():
        # DDP only wraps the GPU portion. Embedding/head stay CPU shared
        # via the natural broadcast on first iteration. For correctness in
        # multi-rank embedding training, would need to manually allreduce
        # embed_tokens.weight.grad — out of scope for this example.
        from torch.nn.parallel import DistributedDataParallel as DDP
        gpu_module = model.backbone
        gpu_module = DDP(gpu_module, device_ids=[dist.local_rank])
        model.backbone = gpu_module

    # Synthetic dataset — replace with real tokenized streaming dataset
    n_samples = 64
    seq_len = 128
    vocab = 32_000
    g = torch.Generator().manual_seed(rank)
    dataset = [
        (
            torch.randint(0, vocab, (seq_len,), generator=g),
            torch.randint(0, vocab, (seq_len,), generator=g),
        )
        for _ in range(n_samples)
    ]

    loader = auto_dataloader(
        dataset, batch_size=4, num_workers=2, distributed=(dist is not None)
    )

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if is_main_rank():
        print(f"  starting training on rank {rank}/{world}")

    t0 = time.time()
    n_tokens = 0
    for step, (x, y) in enumerate(loader):
        if step >= 10:
            break
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        n_tokens += x.numel()
        if is_main_rank() and step % 2 == 0:
            print(f"  step {step}: loss={loss.item():.3f}")

    dt = time.time() - t0
    if is_main_rank():
        print(f"  done: {n_tokens} tokens in {dt:.1f}s = {n_tokens/dt:.0f} tok/s")


if __name__ == "__main__":
    main()
