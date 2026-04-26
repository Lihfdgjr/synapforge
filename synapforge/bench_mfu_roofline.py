"""bench_mfu_roofline.py — roofline analysis for synapforge_100m hot ops.

For every matmul in the model, compute:
    FLOPs = 2 * M * N * K
    Bytes = 2 * (M*K + K*N + M*N)         # bf16 (2 B / element)
    AI = FLOPs / Bytes  (FLOPs/byte)

Roofline ceiling on A100:
    if AI >= AI_ridge=152.6 (peak/HBM), compute-bound at 312 TFLOPS
    else memory-bound at AI * 2.04e12 = AI * 2040 GFLOPS

Outputs a table to bench_mfu_roofline.txt
"""

from __future__ import annotations

A100_PEAK = 312e12
A100_HBM = 2.04e12  # 2.04 TB/s
RIDGE = A100_PEAK / A100_HBM  # ~153 FLOPs/byte


def matmul_metrics(name: str, M: int, K: int, N: int, batch: int = 1, dtype_bytes: int = 2):
    flops = 2 * M * N * K * batch
    bytes_io = dtype_bytes * (M * K + K * N + M * N) * batch
    ai = flops / bytes_io
    if ai >= RIDGE:
        ceiling_flops = A100_PEAK
        regime = "compute-bound"
    else:
        ceiling_flops = ai * A100_HBM
        regime = "memory-bound"
    return {
        "op": name,
        "M": M, "K": K, "N": N, "B": batch,
        "GFLOPs": flops / 1e9,
        "MB_moved": bytes_io / 1e6,
        "AI": ai,
        "regime": regime,
        "ceiling_TFLOPS": ceiling_flops / 1e12,
    }


def main() -> None:
    rows = []
    B = 32
    T = 256
    d = 512
    h = int(d * 8)  # SwiGLU hidden = 4096
    V = 50257
    n_layers = 10
    loop_depth = 4

    # LM head (forward): (B*T, d) @ (d, V) -> (B*T, V)
    rows.append(matmul_metrics("lm_head_fwd", M=B * T, K=d, N=V))

    # LM head (backward, also a GEMM)
    rows.append(matmul_metrics("lm_head_bwd_dW", M=d, K=B * T, N=V))

    # SwiGLU per-layer (one of three projections)
    rows.append(matmul_metrics("swiglu_w_gate", M=B * T, K=d, N=h))
    rows.append(matmul_metrics("swiglu_w_up",   M=B * T, K=d, N=h))
    rows.append(matmul_metrics("swiglu_w_down", M=B * T, K=h, N=d))

    # SparseSynapse (linear with masked weight): (B*T, d) @ (d, d)
    rows.append(matmul_metrics("synapse_dense", M=B * T, K=d, N=d))

    # LiquidCell: delta_proj + b_proj  d->d each
    rows.append(matmul_metrics("liquid_proj", M=B * T, K=d, N=d))

    # Total per fwd
    print(f"{'op':<24} {'shape (MxKxN)':<22} {'GFLOPs':>10} {'MB':>10} {'AI':>8} {'regime':<14} {'ceil TF':>10}")
    print("-" * 105)
    out_lines = []
    for r in rows:
        shape = f"{r['M']}x{r['K']}x{r['N']}"
        line = f"{r['op']:<24} {shape:<22} {r['GFLOPs']:>10.2f} {r['MB_moved']:>10.1f} {r['AI']:>8.1f} {r['regime']:<14} {r['ceiling_TFLOPS']:>10.1f}"
        print(line)
        out_lines.append(line)

    # Aggregate per-step (forward only, ignoring small ops)
    fwd_layers = sum(r['GFLOPs'] for r in rows if r['op'].startswith(('swiglu', 'synapse', 'liquid'))) * n_layers * loop_depth
    fwd_head = rows[0]['GFLOPs']
    print(f"\n[per-fwd-step] hybrid_block GEMMs ({n_layers} layers x {loop_depth} loops): {fwd_layers:.0f} GFLOPs")
    print(f"[per-fwd-step] LM head fwd:                                    {fwd_head:.0f} GFLOPs")
    print(f"[per-fwd-step] sum:                                            {fwd_layers + fwd_head:.0f} GFLOPs")

    # Compare to FlopCounter total / 3 (rough fwd-only) ~ 5230 GFLOPs
    head_share = fwd_head / (fwd_layers + fwd_head)
    print(f"\n[insight] LM head is {head_share*100:.1f}% of fwd GEMM FLOPs.")
    print(f"[insight] LM head AI = {rows[0]['AI']:.1f} (RIDGE={RIDGE:.0f}); regime: {rows[0]['regime']}.")
    print(f"[insight] Synapse AI = {rows[5]['AI']:.1f}; regime: {rows[5]['regime']}.")
    print(f"[insight] SwiGLU w_gate AI = {rows[2]['AI']:.1f}; regime: {rows[2]['regime']}.")

    # Save table
    out_path = "/workspace/synapforge/bench_mfu_roofline.txt"
    with open(out_path, "w") as f:
        f.write(f"# A100 SXM 80GB roofline. Peak = {A100_PEAK/1e12:.0f} TFLOPS bf16, HBM = {A100_HBM/1e9:.2f} TB/s\n")
        f.write(f"# Ridge AI = {RIDGE:.1f} FLOPs/byte (above = compute-bound).\n\n")
        f.write(f"{'op':<24} {'shape (MxKxN)':<22} {'GFLOPs':>10} {'MB':>10} {'AI':>8} {'regime':<14} {'ceil TF':>10}\n")
        f.write("-" * 105 + "\n")
        f.write("\n".join(out_lines) + "\n\n")
        f.write("[summary]\n")
        f.write(f"  LM head GEMM share of fwd FLOPs: {head_share*100:.1f}%\n")
        f.write(f"  LM head regime:    {rows[0]['regime']} (AI={rows[0]['AI']:.1f}, ceil {rows[0]['ceiling_TFLOPS']:.0f} TF)\n")
        f.write(f"  SwiGLU regime:     {rows[2]['regime']} (AI={rows[2]['AI']:.1f}, ceil {rows[2]['ceiling_TFLOPS']:.0f} TF)\n")
        f.write(f"  Synapse(dense)reg: {rows[5]['regime']} (AI={rows[5]['AI']:.1f}, ceil {rows[5]['ceiling_TFLOPS']:.0f} TF)\n")
        f.write(f"  Liquid proj reg:   {rows[6]['regime']} (AI={rows[6]['AI']:.1f}, ceil {rows[6]['ceiling_TFLOPS']:.0f} TF)\n")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
