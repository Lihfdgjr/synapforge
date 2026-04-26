"""04_triton_speedup.py — bench gpu_dense vs triton_block forward.

Reports forward-pass latency of a single LiquidCell with two backends.
On 2x A800 + torch 2.1 + triton 2.1 we measured ~29x speedup at
B=8, T=128, D=256.

Triton is optional. If unavailable, the script reports the dense
baseline and exits cleanly.

Run: python examples/04_triton_speedup.py
"""
import time
import torch
import synapforge as sf


def bench(fn, warmup: int = 10, iters: int = 50) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main() -> None:
    if not torch.cuda.is_available():
        print("[skip] CUDA not available; Triton bench requires GPU")
        return

    device = "cuda"
    B, T, D = 8, 128, 256
    cell = sf.LiquidCell(D, D).to(device)
    x = torch.randn(B, T, D, device=device)

    rt_dense = sf.compile(cell, backend="gpu_dense")
    print(f"backend gpu_dense  : {bench(lambda: rt_dense(x)):.3f} ms / fwd")

    try:
        rt_triton = sf.compile(cell, backend="triton_block")
    except Exception as exc:
        print(f"[skip] triton backend unavailable: {exc}")
        return
    print(f"backend triton_block: {bench(lambda: rt_triton(x)):.3f} ms / fwd")
    print("OK")


if __name__ == "__main__":
    main()
