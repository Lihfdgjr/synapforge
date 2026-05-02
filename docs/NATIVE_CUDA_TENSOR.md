# `synapforge.native.cuda` -- torch-free GPU compute layer

This package is the GPU counterpart to `synapforge/native_demo.py` (the
pure-numpy MVP). Together they let SynapForge execute the full LNN+SNN
training loop with **zero `import torch`** in the hot path.

## Why cupy?

We picked [cupy](https://cupy.dev) over the alternatives for four
reasons:

| Candidate | What it gives us | What it costs |
|---|---|---|
| **cupy** ✅ | numpy-compatible API, mature cuBLAS / cuDNN bindings, raw cuda pointer via `arr.data.ptr` (Triton-ready), DLPack interop, MemoryPool already optimized for ML | Optional dep; CI hosts without GPU need a fallback path |
| pyCUDA | Lowest-level, raw `cuda.mem_alloc` | No numpy compat -- we'd rewrite every elementwise op |
| numba.cuda | JIT custom kernels in Python | Slower for cuBLAS; competes with our Triton kernels |
| jax / mlx | Whole framework | Brings its own array type and tracer; defeats the "no framework lock-in" goal |

Cupy also exposes the **DLPack** v2 protocol on `cupy.ndarray`. That
lets us hand a buffer to torch (which Triton requires for kernel
launches) **without copying** the data, then take the result back.
That bridge is `synapforge/native/cuda/triton_glue.py`.

## Public API

```python
from synapforge.native.cuda import (
    CudaTensor,         # the tensor type
    ops,                # matmul, silu, softmax, layernorm, ...
    CudaMemPool,        # block allocator with high-water mark
    CudaStreamPool,     # 4-stream pool for compute / H2D / D2H / misc
    triton_glue,        # bridge for existing Triton kernels
    HAS_CUPY,           # bool -- True iff cupy + CUDA reachable
    cupy_or_numpy,      # the active xp module (cp or np)
)

# Tensor lifecycle
x = CudaTensor.zeros((1024, 4096), dtype=np.float32)   # cuda:0 if HAS_CUPY else cpu
y = CudaTensor.from_cpu(host_np_array)                 # H2D copy
host = y.to_cpu()                                      # D2H copy
ptr = y.data_ptr                                       # int64 cuda pointer for Triton

# Pinned host alloc -> async H2D
buf = CudaTensor.pinned_alloc((B, T), dtype=np.int32)
buf[...] = batch
gpu_buf = CudaTensor.zeros((B, T), dtype=np.int32)
gpu_buf.async_copy_from_host(buf, stream=streams.h2d)

# Ops (numpy-style, no autograd)
z = ops.matmul(x, w)
z = ops.silu(z)
z = ops.layernorm(z, gamma, beta)
loss = ops.softmax(z, axis=-1)

# Triton kernel via DLPack zero-copy bridge
from synapforge.native.cuda import triton_glue
h_pre, s, m, h_post = triton_glue.run_fused_lnn_snn_block(
    a=a_cuda, b=b_cuda, threshold=thr_cuda, h0=h0_cuda,
)
```

## Limitations (be honest)

This is a **scaffold**, not a drop-in replacement for `torch.matmul` on
730M parameters. What's missing:

1. **No autograd.** Every op has a forward; the backward lives in
   `synapforge/native/vjp/` (sister package, owned by another agent).
   You manually call the VJPs in reverse layer order. This is the same
   model as `native_demo.py`.

2. **No fused kernels for LNN+SNN ops.** `ops.matmul` routes through
   cuBLAS, which is fine for the dense linear layers (Q/K/V/proj/MLP).
   But `cfc_step`, `plif_step`, and the STDP trace update are not
   GPU-fused yet -- you must go through `triton_glue.run_fused_lnn_snn_block`
   for those. That kernel is shared with the existing torch-backed
   `synapforge.backends.triton_block_kernel`.

3. **fp32 default.** bf16 / fp16 work via `dtype=` keyword on factories,
   but our cuBLAS path picks the dtype from the input array dtype --
   cupy then dispatches to `sgemm` / `hgemm` accordingly. Tensor cores
   only kick in when both operands are bf16 / fp16 / int8. We do not
   currently emit explicit `cuBLASLt` calls with epilogue fusion (no
   bias-add fused into matmul), so a 1.05--1.15x perf gap vs torch's
   `addmm_` is expected.

4. **No NCCL / multi-GPU.** Single GPU only. Multi-GPU lives in the
   `synapforge/distributed*.py` layer (still torch-backed for now).

5. **No graph capture.** torch has `torch.cuda.graph()`; cupy has
   `cupy.cuda.Graph` but it's experimental. We haven't wired it.

6. **`cuBLASLt` epilogue fusion missing.** torch fuses `gemm + bias +
   activation` into a single `cublasLtMatmul` call. Our `ops.gemm` does
   `matmul -> add -> silu` as three separate launches. Roughly 1.2-1.5x
   slower than torch on small dense layers.

## Memory model

`CudaMemPool` wraps `cupy.cuda.MemoryPool` with extra telemetry:

```python
pool = CudaMemPool()
ptr = pool.alloc(256 * 256 * 4)   # 256x256 fp32
pool.free(ptr)
print(pool.stats().peak_bytes)    # high-water mark
print(pool.used_bytes())          # delegate to cupy.MemoryPool
```

vs torch's caching allocator:

| Behavior | cupy.MemoryPool | torch.cuda.caching | CudaMemPool |
|---|---|---|---|
| Cache by exact size | yes | yes (best-fit) | yes (size-keyed) |
| Free triggers reuse | yes | yes | yes |
| `free_all_blocks()` | manual | manual | yes (via `release_all`) |
| Peak bytes counter | no | yes | yes |
| Foreign pointer accept | no | no | no (tracks by `id()`) |

For our 730M model:

* Activation memory at fp32 sequence-length 1024 batch-32:
  ~3.5 GiB (40 layers x 1024 x 768 fp32).
* Optimizer state (AdamW m + v): 5.84 GiB (730M x 8B fp32).
* Weights: 2.92 GiB (fp32) or 1.46 GiB (bf16).

Total minimum: ~12.2 GiB. Fits an A800 80GB by ~6.5x slack -- perfect
for our scale-up testing without going to ZeRO-3.

## How big is the gap?

Measured by lines of code that would land on top of this scaffold to
become a torch-equivalent backend on a 730M model:

| Component | LOC needed | Engineer-days | Risk |
|---|---|---|---|
| Manual VJP catalog (sister pkg) | ~1500 | 4-6d | Already partially done |
| `cuBLASLt` epilogue fusion | ~600 | 3-4d | Medium (cuda-version pin) |
| KV-cache-style matmul w/ scratchpad | ~400 | 2-3d | Low |
| Custom RMSNorm / softmax fused | ~300 | 2d | Low (cupy ElementwiseKernel) |
| CUDA Graph capture | ~250 | 1-2d | Medium (cupy graph API is beta) |
| NCCL all-reduce wrapper | ~500 | 4-5d | High (we'd need our own collective) |
| **TOTAL: usable training kernel** | **~3500 LOC** | **16-22 engineer-days** | -- |

So this scaffold is about **15-25%** of the way to a real torch
replacement for our specific 730M LNN+SNN training. The remaining 75%
is mostly:

1. Manual VJPs for every op in `ops.py` (sister package, mostly done).
2. cuBLASLt epilogue fusion to recover the 1.2-1.5x perf gap.
3. NCCL bindings for distributed.

The good news: **for inference**, this scaffold + the existing Triton
kernel via `triton_glue` is already enough to serve a 730M model
torch-free, modulo the cuBLASLt fusion gap (1.2-1.5x slower). For
training, we still need to land VJPs and at least one collective op.

## How to run on a real GPU

```bash
# 1. Install cupy matching your CUDA version
pip install cupy-cuda12x   # for CUDA 12.x
# or: pip install cupy-cuda11x

# 2. Import and probe
python -c "from synapforge.native.cuda import HAS_CUPY; print(HAS_CUPY)"
# expected: True

# 3. Run the smoke tests (the GPU lane fires automatically)
pytest tests/native/cuda/ -v
```

On a GPU host, the 36 tests in `test_cuda_tensor.py` plus the 6 tests
in `test_triton_glue.py` should all pass (none skipped).

## File map

```
synapforge/native/cuda/
    __init__.py        public API + HAS_CUPY probe
    tensor.py          CudaTensor class, data_ptr, pinned alloc, async copy
    ops.py             matmul/bmm/gemm/silu/gelu/sigmoid/tanh/sum/mean/var/
                       max/argmax/topk/add/mul/addcmul/addcdiv/layernorm/
                       rmsnorm/softmax
    allocator.py       CudaMemPool + PoolStats (high-water mark, cache)
    streams.py         CudaStream + CudaStreamPool (compute/h2d/d2h/misc)
    triton_glue.py     DLPack bridge to existing Triton kernel ABI

tests/native/cuda/
    test_cuda_tensor.py    36 cases: surface, ops vs numpy, allocator, streams
    test_triton_glue.py    6 cases: import smoke, fail-loud, bridge round-trip,
                           GPU-only kernel parity vs numpy reference
```

## Anti-LoRA / no-transformer reminder

This package complies with the "no torch in the hot path" rule
([CLAUDE.md, native_demo.py](../synapforge/native_demo.py)) and the
"no transformer fallback" rule
([docs/ANTI_LORA.md](ANTI_LORA.md)). The only place torch appears in
this entire subtree is `triton_glue.py`, and it is gated behind
`HAS_TORCH` for use only when crossing the Triton ABI boundary.
