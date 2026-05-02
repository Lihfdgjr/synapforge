# Triton Block Backward — Per-Op Audit

**Date:** 2026-05-02
**Scope:** the chain of ops in `synapforge.model_100m.HybridBlock.forward`
(`synapforge/model_100m.py:202-231`) and which of them currently have a
Triton backward vs a Python-fallback (torch.autograd) backward.

## Forward chain (single HybridBlock)

```
x_in ─► RMSNorm(ln1) ─► LiquidCell(CfC scan) ─► PLIF(Heaviside) ─► SEW(s+h)
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        ▼
        SparseSynapse(spike_input) * sigmoid(Linear gate(spike_input))
                                     │
        ┌────────────────────────────┘
        ▼
        residual: x = x_in + drop(gated)
        ▼
        x = x + drop(SwiGLU(RMSNorm(ln2(x))))
        ▼
        + optional high-pass residual (off in production)
```

## Per-op fwd / bwd implementation status

| # | Op | Forward | Backward | Closed-form Jacobian formula | Status |
|---|----|---------|----------|------------------------------|--------|
| 1 | RMSNorm `ln1` | `x*rms*w`, `rms = (x.pow(2).mean(-1) + eps).rsqrt()` | torch autograd (default) | Standard, see Zhang/Sennrich 2019 §A.1 | KEEP TORCH (low priority — fast already) |
| 2 | CfC liquid step `h_t = A_t·h_{t-1} + b_t` | Triton fused (`fused_lnn_snn_block_kernel`) | Triton fused (`fused_lnn_snn_block_bwd_kernel`) — reverse-time scan | Hasani 2022 §3.2: `dh_t/dh_{t-1} = A_t`, `dh_t/dA_t = h_{t-1}`, `dh_t/db_t = 1` | DONE in Triton |
| 3 | PLIF spike (ATan surrogate) `s_t = (h_t > thr)` | Triton fused (Heaviside in same kernel as #2) | Triton fused (ATan surrogate `ds/dm = α/(2(1+(π/2·α·m)²))`) | Fang 2021 ICCV — already in `fused_lnn_snn_block_bwd_kernel` lines 207-208 | DONE in Triton |
| 4 | Subtract reset `h_post = h_pre·(1-s)` | Triton fused (#2 same kernel) | Triton fused (#2 same kernel — `dh_post/dh_pre = (1-s) - h_pre·ds/dm` line 212) | Algebraic; chain rule gives the `-h_pre·ds/dm` term via dependence of `s` on `h_pre` | DONE in Triton |
| 5 | LiquidCell projections `delta = softplus(W_d·x)`, `b = delta·(W_b·x)`, `A_t = exp(-delta·A)` | torch (Linear+softplus+exp) | torch autograd | Standard linear/softplus/exp; not on the bottleneck | KEEP TORCH (Linear is cuBLAS, fast) |
| 6 | SEW shortcut `spike_input = s + h` | torch | torch autograd (additive add node) | Identity: `dspike_input/ds = 1`, `dspike_input/dh = 1` | TARGET (kernel #1 in new file) — additive but the `+` node is the most-launched op in the block |
| 7 | SparseSynapse `synapse(spike_input)` | torch.autograd.Function (`_MaskedLinear`) | torch.autograd.Function (`_MaskedLinear`) — custom dW with mask | Standard; `dY/dx = (W·M)`, `dY/dW = M·(grad·x.T)` | KEEP — already custom, mask-aware |
| 8 | Linear `gate(spike_input)` | nn.Linear | torch autograd | Standard | KEEP TORCH (cuBLAS) |
| 9 | Sigmoid `sigmoid(gate_pre)` | torch | torch autograd | `dsigmoid(z)/dz = sigmoid(z)·(1-sigmoid(z))` | TARGET (kernel #2 in new file) — fuse with #10 |
| 10 | Multiplication `synapse_out * sigmoid_out` | torch | torch autograd | `df/da = b`, `df/db = a` for `f = a·b` | TARGET (kernel #2, fused with #9) |
| 11 | Dropout `drop(gated)` | torch | torch autograd (mask saved) | Identity·mask | KEEP TORCH (off in production: `dropout=0.0`) |
| 12 | Residual add `x_in + drop(gated)` | torch | torch autograd (additive) | Identity | KEEP TORCH (one add) |
| 13 | RMSNorm `ln2` | torch | torch autograd | Standard, same as #1 | KEEP TORCH |
| 14 | SwiGLU FFN `w_d(silu(w_g·x) * w_u·x)` | torch (3× Linear, silu, mul) | torch autograd | Standard SwiGLU | KEEP TORCH (cuBLAS-bound, not Python-bound) |
| 15 | High-pass residual (optional) | torch | torch autograd | Standard Conv1d | KEEP TORCH (off in production, `lambda=0.0`) |

## What's actually on the critical path of Python overhead?

For a `n_layers=10` × `loop_depth=4` model running at bs=24, T=256:

* **Already Triton-fused (1 launch / layer-loop iter):** ops #2, #3, #4 — CfC + PLIF + reset
* **Per-layer-loop torch dispatches still firing:** ops #1, #5 (3 Linear), #6 (1 add), #7 (1 _MaskedLinear), #8 (1 Linear), #9 (1 sigmoid), #10 (1 mul), #11 (drop), #12 (add), #13 (RMSNorm), #14 (3 Linear + silu + mul), #15 (off)

Of those, the highest-leverage Python overheads (most launches, smallest payload) are:
* #6 (additive SEW `s + h`) — 1 launch with ZERO compute
* #9 (sigmoid) — 1 launch, elementwise
* #10 (mul) — 1 launch, elementwise
* These three combined are 3 launches for a fused operation that should take 1 launch.

#7, #8, #14 are cuBLAS-bound (matmul); fusing them via Triton wouldn't beat cuBLAS, so they
stay torch.

## Closed-form Jacobian summary for the new kernel

The new `triton_block_kernel_bwd.py` adds backward for the **synapse-gate fusion**:

```
forward:
    spike_input = s + h                       # SEW
    syn_out     = SparseSynapse(spike_input)  # already custom autograd
    gate_pre    = Linear_gate(spike_input)    # standard Linear
    gate        = sigmoid(gate_pre)
    gated       = syn_out * gate              # OUTPUT

backward (given grad_gated):
    grad_syn_out = grad_gated * gate
    grad_gate    = grad_gated * syn_out
    grad_gate_pre = grad_gate * gate * (1 - gate)              # sigmoid'
    # then chain through Linear_gate (cuBLAS) and synapse (custom)
    # SEW: grad_s = grad_spike_input,  grad_h = grad_spike_input
    # where grad_spike_input flows from {syn_out path, gate_pre path}
```

The SEW backward is just two identity copies; we expose it via a single
fused kernel that also produces the sigmoid+mul backward in one launch
(one launch instead of three).

## Out of scope for this commit

* RMSNorm / LayerNorm Triton bwd — kept on torch. Already cuBLAS-fast for
  these sizes; not the Python-overhead bottleneck.
* SwiGLU Triton bwd — same reasoning. cuBLAS dominates.
* High-pass residual — `lambda=0.0` in production.
* Dropout — `0.0` in production.
* Allocation pooling / `synapforge.cuda.MemPool` — Phase 4 risk register §2,
  separate work item.

## File map

* `synapforge/backends/triton_block_kernel.py` — pre-existing CfC+PLIF+reset
  fwd+bwd (op #2, #3, #4). No changes in this commit.
* `synapforge/backends/triton_block_kernel_bwd.py` — NEW. SEW+sigmoid+mul
  fused fwd+bwd (ops #6, #9, #10). Also exposes a thin
  `torch.autograd.Function` wrapper.
* `tests/backends/test_triton_block_bwd.py` — NEW. Numerics gate
  (`< 1e-4` rel err vs torch.autograd.grad on a fixture).
* `scripts/bench_hybrid_block.py` — NEW. Throughput benchmark vs the
  pre-existing torch path.
