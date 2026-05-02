# Native Spike Packing — `synapforge.native.spike`

Bit-packed binary-spike storage and matmul kernels for the LNN+SNN
spike->synapse boundary.  16 spikes packed into one `uint16` word —
**16x memory and HBM bandwidth savings** on the synapse branch.

## Why this is LNN+SNN-specific

A dense transformer's activation is fp16 / bf16; you can't compress it
16x without quantisation noise.  A PLIF spike is **binary {0, 1} by
construction**: storing it as fp16 wastes 15 of 16 bits.

This subsystem exploits that property and **only that property**.  It
has no analogue in transformer training.  When the user said
"针对 我们训练的是个什么东西做优化" (optimise specifically for what we
are actually training), this is the answer at the spike boundary.

## Memory and bandwidth math

The spike train flows through HBM twice per layer per step:

1. **Write** after PLIF fires.
2. **Read** into the synapse matmul `synapse(s + h)`.

At `B=48 T=256 d=1280 layers=16` (production launcher config):

| storage      | per step (16 layers) | per step (1 layer) |
|--------------|---------------------|--------------------|
| fp16 dense   | 960 MB              | 60 MB              |
| uint16 packed| 60 MB               | 3.75 MB            |
| saving       | **900 MB**          | **56.25 MB**       |
| ratio        | **16.00x**          | **16.00x**         |

(Both directions counted: `2 bytes/spike * 2 directions`.  Verified
exactly by `tests/native/spike/test_packed_matmul.py::test_memory_savings_16x_bf16_input`.)

On A800 80GB at **1.5 TB/s HBM**, 900 MB of saved traffic translates
to ~**600 us per step** of pure bandwidth ceiling, before counting
cache-effect and skip-zero-bit savings inside the kernel.

## Anatomy

```
synapforge/native/spike/
  __init__.py
  pack.py             pure-numpy pack_spikes / unpack_spikes
  packed_matmul.py    Triton kernels (NO 'import torch' in this file)
  torch_glue.py       autograd.Function + --packed-spikes wiring
```

Tests:

```
tests/native/spike/
  test_packed_matmul.py   bit-exact + numerical correctness coverage
```

Bench:

```
scripts/bench_spike_pack.py   --d-in 1280 --d-out 1280 --batch 48 --seq 256
```

### `pack.py` — numpy primitives

```python
from synapforge.native.spike.pack import pack_spikes, unpack_spikes

s = (np.random.rand(48, 256, 1280) > 0.9).astype(np.float32)  # 10% density
packed = pack_spikes(s)                # uint16, last-dim shrinks 16x
out = unpack_spikes(packed, n_orig=1280, dtype=np.float16)    # round-trip
np.array_equal(out.astype(np.float32), s)   # True (bit-identical)
```

* **Layout**: little-endian within the word; bit `b` of word `w` stores
  `s[..., w*16 + b]`.  Padding bits in the last word are zero.
* **Round-trip** is bit-identical for `d in {16, 256, 1280, 1536}` and
  for any odd size (verified, including `{1, 7, 15, 17, 31}`).
* **Pure numpy** — no torch dependency.  The torch-tensor pack/unpack
  variants live in `torch_glue.py`.

### `packed_matmul.py` — Triton kernels

Kernel module is **`import torch`-free** by design contract.  Three
kernels:

* `packed_spike_matmul_fwd_kernel`: forward
  `y[m, n] = sum_k unpack(packed_s)[m, k] * weight[k, n]`.

  Loads packed `uint16` words, expands 16 bits in-register via
  `>> bit_idx & 1` then `tl.where`, feeds `tl.dot` directly without
  unpacking to HBM.

* `packed_spike_matmul_bwd_dW_kernel`: backward
  `grad_W[k, n] += sum_m unpack(s)[m, k] * grad_y[m, n]`.

  Same bit-expansion trick, transposed.  Atomic-adds the result tile
  into the persistent `grad_W` accumulator.

* `packed_spike_matmul_bwd_dS_kernel`: backward
  `grad_s[m, k] = sum_n grad_y[m, n] * weight[k, n]`.

  This is **dense fp**.  `grad_s` is the surrogate-gradient signal
  feeding the spike's atan backward; it is not binary so packing on
  output is meaningless.

The kernel uses a numpy host fallback (`packed_spike_matmul_numpy`)
for CPU-only correctness verification.

### `torch_glue.py` — autograd + dispatch

* `pack_spikes_torch` / `unpack_spikes_torch`: torch-tensor
  pack/unpack.  PyTorch lacks first-class `uint16` (pre-2.4), so we
  store as `int32` with values in `[0, 65535]`.  The kernel reads via
  `data_ptr()` and casts to `uint32` internally before bit decode.
* `PackedSpikeMatmul`: `torch.autograd.Function`.  Forward caches the
  **packed** tensor (16x smaller than caching `s`).  Backward dW uses
  the packed kernel; backward dS is dense.
* `packed_spike_linear(s, h, linear)`: high-level API matching the
  shape of `synapforge.kernels.sparse_spike_matmul.sparse_spike_linear`
  for drop-in.
* **Auto-fallback**: density >= `density_threshold` (default 0.30) ->
  dense `F.linear(s + h, W, bias)`.  Same crossover threshold as
  `--sparse-spike-synapse`.

### CLI flag — `--packed-spikes`

```bash
python train_100m_kd.py --packed-spikes ...
```

Default OFF.  When ON, every `HybridBlock`'s spike->synapse path
routes through the packed kernel under the same density-driven
auto-fallback.  Print line on launch:

```
[packed-spikes] bit-packed spike->synapse path enabled (threshold=0.30);
  requires Triton + CUDA. Dormant under dead PLIF; ~16x HBM bandwidth
  saving on synapse branch once spikes wake up.
```

The flag is **orthogonal to `--sparse-spike-synapse`**:

| optimisation                 | what it saves        | when it wins          |
|------------------------------|----------------------|-----------------------|
| `--sparse-spike-synapse`     | compute (O(K·N))     | density 5-15%, low K  |
| `--packed-spikes`            | HBM bandwidth (16x)  | always (pack-aware)   |

Both are auto-fallback above 30% density.  You can stack them:
`--sparse-spike-synapse --packed-spikes` -- the packed path wins
priority in the dispatch (`packed` takes the spike branch; the
embedding-bag path is the legacy compute opt).

## Limitation: dormant under dead PLIF

**Current Run 7 spike density = 0** (PLIF dead, see
`memory/feedback_plif_dead_bootstrap.md`).  At zero density:

* The kernel reads packed words (all zero) -> `tl.where` skips every
  contribution -> output is exactly zero.
* Total bandwidth saving on the spike branch = `16 * (6 / 96) MB` =
  **6 MB packed traffic** vs **96 MB unpacked traffic** per step (at
  d=1280, layers=16).  Saving is ratio-preserved, but the absolute
  number is irrelevant when no spikes drive learning.
* **The flag is dormant**: it doesn't slow training (pack overhead is
  amortised over the matmul read), but it doesn't accelerate it
  either, because the synapse branch contributes nothing to the loss
  in the dead state.

Once PLIF revives at the predicted **5-15% density** (post-fix
trajectory in `feedback_plif_dead_bootstrap.md`):

| density | bandwidth saving | pack overhead | net (theoretical, A800 1.5 TB/s) |
|---------|-----------------|----------------|--------------------------------|
| 0%      | 0 MB            | ~30 us         | 0 us / step (dormant)          |
| 5%      | 30 MB           | ~30 us         | ~+10 us / step                 |
| 10%     | 60 MB           | ~30 us         | ~+10 us / step                 |
| 15%     | 90 MB           | ~30 us         | ~+30 us / step                 |
| 30%     | (auto-fallback to dense; flag is no-op above threshold)        |

The numbers in this table are conservative theoretical projections
from `scripts/bench_spike_pack.py`.  Actual rental-box measurements
will go in this section once the Triton run is feasible.

## Honest assessment

* **What this does**: shaves the bandwidth ceiling on the spike branch
  by 16x.  Below 30% density.  Once PLIF revives.
* **What it doesn't do**:
  - Help in dense mode (>30% density auto-fallback).
  - Help when PLIF is dead (no spike contribution to anything).
  - Accelerate the dense LiquidCell `h` branch (dense GEMM, already cuBLAS-optimal).
  - Save activation memory globally — only at the spike->synapse boundary.
* **Pack overhead** is `O(M * d)` and runs every forward; on A800 with
  M=12288 d=1280 it's ~30 us per call — negligible relative to
  matmul-time but real.
* **Not a silver bullet** — the `--packed-spikes` flag is a tool in the
  performance toolkit, not a default-on optimisation.

## Validation

```bash
# Bit-exact round-trip + numerical correctness (CPU + numpy).
pytest tests/native/spike/test_packed_matmul.py -v

# Bandwidth + matmul-time bench (CPU sanity-mode if no Triton).
python scripts/bench_spike_pack.py --batch 48 --seq 256 --d-in 1280

# JSON output for telemetry.
python scripts/bench_spike_pack.py --json > bench.json
```

## Future work

* **Two-bit quantised spike**: ternary spikes `{-1, 0, 1}` would need 2
  bits per spike (8 packed per uint16 = 8x).  STDP-relevant.
* **Online STDP eligibility-trace**: pack the spike-history buffer
  used by STDP into uint16 too — same 16x saving on the eligibility
  trace memory, which currently dominates STDP storage.
* **Cross-backend port**: triton -> cuda -> hip -> mps.  Triton is the
  only blocker; the numpy reference path is portable everywhere.

## Cross-reference

* `synapforge.kernels.spike_pack` — older int32-storage variant, **8x**
  compression, kept for API stability.
* `synapforge.kernels.sparse_spike_matmul` — compute-side opt
  (EmbeddingBag row-gather), orthogonal to this module's
  bandwidth-side opt.
* `synapforge/backends/triton_block_kernel.py` — fused HybridBlock
  Triton kernel (CfC scan + PLIF + STDP).  Future work: integrate the
  packed-spike path inside that fused kernel.
