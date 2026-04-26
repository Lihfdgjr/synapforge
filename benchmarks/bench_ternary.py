"""Bench: full-precision (fp32) vs ternary-QAT inference for synapforge.

What this measures
------------------
* Forward latency at bs=64, T=256 over a small transformer-shaped block.
* Wall-clock speedup, fp32 mode (de-quantized forward; ternary still goes
  through fp32 GEMM via TernaryQuantizer).
* "On-disk" size reduction at deploy time: ternary codes are int8 (or 1.58
  bits packed), vs fp32 weights at 32 bits each.

What this does NOT measure
--------------------------
The actual deploy-time speedup. In QAT, both modes still call cuBLAS fp32
GEMM, so latency is ~equal (sometimes ternary is slightly slower because
of the extra round/clamp). The real 5-10x speedup comes at *deployment*,
when ternary codes are dispatched to bitnet.cpp's int8/popcount kernels.

The point of this bench is to confirm two things:
  1. QAT does not blow up training-time inference latency.
  2. Once exported, weights are 16-32x smaller on disk.
"""

from __future__ import annotations

import io
import os
import sys
import time

sys.path.insert(0, "/workspace")

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapforge.quantize import (
    TernaryLinear,
    convert_model_to_ternary,
    count_ternary_params,
    freeze_gamma,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Benchmark model: emb -> N x (linear-up -> GELU -> linear-down) -> head.
# ---------------------------------------------------------------------------


class BenchBlock(nn.Module):
    def __init__(self, vocab: int, dim: int, n_layers: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_ratio * dim),
                    nn.GELU(),
                    nn.Linear(mlp_ratio * dim, dim),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for blk in self.layers:
            x = x + blk(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------


def bench_inference(model: nn.Module, ids: torch.Tensor, warmup: int = 10, iters: int = 50) -> float:
    """Return mean forward latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(ids)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        for _ in range(iters):
            _ = model(ids)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        elapsed = time.time() - t0
    return elapsed / iters * 1000.0  # ms/iter


def fp32_disk_size(model: nn.Module) -> int:
    """Serialize the full state_dict and report bytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def packed_ternary_disk_size(model: nn.Module) -> int:
    """Estimate deploy-time on-disk size.

    For each TernaryLinear:
        * weight stored as ceil(numel * log2(3) / 8) bytes (1.58 bits / weight)
        * gamma: 4 bytes (fp32 scale)
        * bias: numel * 4 bytes (we keep bias in fp32; small overhead)
    All other parameters (emb, lm_head, LayerNorm, ...) stored as fp32
    (4 bytes / element). This mirrors the bitnet.cpp deployment format.
    """
    total = 0
    seen_param_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            n = int(m.weight.numel())
            # log2(3) ~= 1.5849625, packed.
            total += (int(n * 1.5849625 / 8) + 1)  # +1 for ceil
            total += 4  # gamma fp32
            seen_param_ids.add(id(m.weight))
            if m.bias is not None:
                total += int(m.bias.numel()) * 4
                seen_param_ids.add(id(m.bias))
    # Anything else (emb/lm_head/LN) stays fp32.
    for p in model.parameters():
        if id(p) in seen_param_ids:
            continue
        total += int(p.numel()) * 4
    return total


# ---------------------------------------------------------------------------


def main() -> int:
    vocab = 1024
    dim = 256
    n_layers = 6
    bs = 64
    T = 256

    print(f"device: {DEVICE}")
    print(f"config: vocab={vocab} dim={dim} n_layers={n_layers} bs={bs} T={T}")
    print("-" * 72)

    # FP32 baseline.
    torch.manual_seed(7)
    fp32 = BenchBlock(vocab, dim, n_layers).to(DEVICE)
    fp32.eval()

    # Ternary version: clone weights, then convert.
    torch.manual_seed(7)
    tern = BenchBlock(vocab, dim, n_layers).to(DEVICE)
    n_replaced = convert_model_to_ternary(tern, exclude=("emb", "lm_head"))
    # Ensure gamma is initialized (do one forward), then freeze.
    with torch.no_grad():
        _ = tern(torch.randint(0, vocab, (4, 8), device=DEVICE))
    n_frozen = freeze_gamma(tern)
    n_tern, n_total = count_ternary_params(tern)
    print(f"ternary: replaced {n_replaced} layers, frozen {n_frozen}; "
          f"{n_tern / 1e3:.0f}K of {n_total / 1e3:.0f}K params ternarizable "
          f"({n_tern / n_total:.0%})")
    print("-" * 72)

    ids = torch.randint(0, vocab, (bs, T), device=DEVICE)

    fp32_ms = bench_inference(fp32, ids, warmup=20, iters=100)
    tern_ms = bench_inference(tern, ids, warmup=20, iters=100)
    speedup_fp32_path = fp32_ms / tern_ms

    print(f"fp32 inference  : {fp32_ms:7.2f} ms/iter")
    print(f"ternary (fp32)  : {tern_ms:7.2f} ms/iter")
    print(f"  speedup (fp32 path, expected ~1x): {speedup_fp32_path:.2f}x")
    print()

    # On-disk size.
    fp32_bytes = fp32_disk_size(fp32)
    fp32_compressed_bytes = fp32_disk_size(tern)  # naive serialization (fp32 weights)
    packed_bytes = packed_ternary_disk_size(tern)
    fp32_only_size = fp32_bytes - sum(
        int(m.weight.numel()) * 4 for m in tern.modules() if isinstance(m, TernaryLinear)
    )  # not used for ratio, just diagnostic

    print(f"fp32 state_dict on disk : {fp32_bytes / 1e6:7.2f} MB")
    print(f"ternary fp32-stored     : {fp32_compressed_bytes / 1e6:7.2f} MB "
          f"(QAT checkpoint -- not the deploy format)")
    print(f"packed ternary deploy   : {packed_bytes / 1e6:7.2f} MB "
          f"(int1.58 + fp32 emb/head)")
    print(f"  size reduction vs fp32: {fp32_bytes / packed_bytes:.2f}x "
          f"({(1 - packed_bytes / fp32_bytes) * 100:.1f}% smaller)")

    # Per-layer ternary-only ratio.
    fp32_layer_bytes = sum(
        int(m.weight.numel()) * 4 for m in tern.modules() if isinstance(m, TernaryLinear)
    )
    packed_layer_bytes = sum(
        int(int(m.weight.numel() * 1.5849625 / 8) + 1)
        for m in tern.modules()
        if isinstance(m, TernaryLinear)
    )
    print(f"  layer-only fp32: {fp32_layer_bytes / 1e6:6.2f} MB -> "
          f"{packed_layer_bytes / 1e6:6.2f} MB "
          f"({fp32_layer_bytes / max(packed_layer_bytes, 1):.1f}x compression on "
          f"ternarized weights only -- target ~20x for pure {{-1,0,+1}} pack)")

    print("-" * 72)
    print("note: deploy-time CPU speedup (5-10x) is realized in bitnet.cpp,")
    print("      not in this benchmark. Here we only confirm QAT path is")
    print("      not pathologically slow and that on-disk weights shrink.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
