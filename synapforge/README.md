# synapforge — purpose-built LNN+SNN framework

**v0.1** (2026-04-26): API surface + GPU-dense backend + numerical-equivalent
LiquidCell / PLIF / SparseSynapse / Hebbian / STDP. Wraps PyTorch under the
hood; speedups land in v0.2 (Triton fusion) and v0.3 (event-driven CPU).

## Why a new framework?

PyTorch's nn.Module is built for static, dense, sync compute graphs. The
LNN+SNN bet of MSCFC (continuous-time recurrence + spike events + plasticity)
is none of those. synapforge is the long-term home for:

- **First-class plasticity ops** (Hebbian / STDP / synaptogenesis)
  treated as registered rules, not buffer hacks pinned to a model.
- **Event-graph IR** so a downstream backend can schedule sparse spike
  events without materializing dense (B, T, D) tensors per step.
- **Pluggable backends**: GPU dense (PyTorch today, Triton-fused tomorrow),
  CPU event-driven (numba), and Lava export (neuromorphic).

v0.1 only delivers the API surface and dense backend. The IR exists but
isn't on the execution path yet.

## Layout

```
synapforge/
├── __init__.py        # public API: sf.LiquidCell / sf.PLIF / sf.compile / ...
├── module.py          # sf.Module — nn.Module + plasticity hooks
├── cells/
│   ├── liquid.py      # sf.LiquidCell (Heinsen parallel scan, hasani init)
│   ├── plif.py        # sf.PLIF (learnable threshold + ATan surrogate)
│   └── synapse.py     # sf.SparseSynapse (mask + grow / prune)
├── plasticity.py      # sf.HebbianPlasticity / sf.STDP
├── ir/
│   ├── graph.py       # IRNode / IRGraph data structures
│   └── compiler.py    # walk sf.Module -> IRGraph
├── backends/
│   ├── base.py        # Backend ABC + get_backend()
│   ├── gpu_dense.py   # v0.1: delegates to PyTorch forward
│   ├── cpu_event.py   # STUB: passes through to PyTorch CPU
│   └── lava_export.py # STUB: NotImplementedError
├── runtime.py         # sf.compile(model, backend) -> Runtime
├── bench/
│   └── compare_pytorch.py  # synapforge vs mscfc on identical workloads
└── tests/
    └── test_correctness.py # rel_err vs mscfc + backward sanity + grow/prune
```

## 10-line example

```python
import synapforge as sf
import torch

class HybridBlock(sf.Module):
    def __init__(self, d):
        super().__init__()
        self.cfc  = sf.LiquidCell(d, d, init="hasani")
        self.plif = sf.PLIF(d, threshold=0.3, reset_by_subtract=True)
    def forward(self, x):
        h = self.cfc(x)
        spk, mem = self.plif(h)
        return spk, mem, h

model = HybridBlock(256).cuda()
rt = sf.compile(model, backend="gpu_dense")
spk, mem, h = rt(torch.randn(64, 256, 256, device="cuda"))
```

## Numerical correctness vs mscfc (target rel_err < 1e-3)

| Test                            | CPU      | CUDA     |
|---------------------------------|----------|----------|
| LiquidCell vs mscfc.LiquidS4    | 0.00e+00 | 0.00e+00 |
| compile() vs direct forward     | 0.00e+00 | 0.00e+00 |
| backward — all grads finite     | OK       | OK       |
| PLIF spike rate (random drive)  | 0.67     | 0.64     |

Bit-perfect on LiquidCell because we share the exact Heinsen parallel-scan
math from mscfc.liquid_s4. (NaN at long T comes from the closed-form scan's
`exp(-cumsum(logA))` overflow — same in both libs.)

## Bench (A100 GPU 1, B=64 D=256, see bench_results.txt)

| Workload          | mscfc    | synapforge | overhead |
|-------------------|----------|------------|----------|
| LiquidCell T=256  | 1.198 ms | 1.208 ms   | +0.84%   |
| LiquidCell T=1024 | 4.163 ms | 4.091 ms   | -1.73%   |
| HybridBlock direct| —        | 1.359 ms   | —        |
| HybridBlock Runtime|—        | 1.352 ms   | (no overhead) |

v0.1 just wraps PyTorch — overhead is within noise of the underlying op.

## What is NOT in v0.1 (deferred)

1. Triton-fused Liquid+PLIF kernel (v0.2 — separate task by another agent)
2. Event-driven CPU scheduler with numba (v0.3) — `cpu_event.py` is a stub
3. Lava neuromorphic export — `lava_export.py` raises `NotImplementedError`
4. Autograd-aware plasticity (rules are buffer-only updates today)
5. Distributed training (FSDP / DDP wrappers)
6. Mixed-precision policy (everything is fp32 inside the scan)
7. Real IR-driven dispatch (today the dense backend just calls forward)
8. Spike-aware sparse storage (we use dense tensors)
9. Quantization (bf16 / int8 / ternary inference)
10. Training script — that lives in mscfc/ for now and migrates in v0.5

## Running

```bash
# Correctness (CPU + CUDA if available, ~10s)
/opt/conda/bin/python /workspace/synapforge/tests/test_correctness.py

# Bench — uses GPU 1 only (GPU 0 is reserved for baseline training)
CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python -u \
    /workspace/synapforge/bench/compare_pytorch.py 2>&1 \
    | tee /workspace/synapforge/bench_results.txt
```

## Roadmap (12-month, M1-M3 of the broader plan)

- **M1 (now → +3mo)** — synapforge v0.1 ships (this commit).
  v0.2 adds a single fused Triton kernel for `LiquidCell.forward_seq + PLIF`
  (target: 2–4x A100 speedup on B=64 T=256 D=256).
- **M2 (+3 → +6mo)** — synapforge v0.3: event-driven CPU backend (numba)
  proves the spike-sparse code path. mscfc training moves to import
  synapforge instead of raw PyTorch.
- **M3 (+6 → +12mo)** — synapforge v0.4 (Lava export) + v1.0
  (autograd-through-plasticity). Mythos-class small models train end-to-end
  on the synapforge stack only; mscfc/ becomes legacy.

## Compatibility notes

- `synapforge.LiquidCell` is bit-equivalent to `mscfc.liquid_s4.LiquidS4Cell`
  when params are copied (verified by `tests/test_correctness.py`).
- `synapforge.PLIF` is functionally equivalent to `mscfc.model.PLIFNeuron`
  but exposes per-channel learnable threshold by default (mscfc default is
  scalar). Set `learnable_threshold=False` for the legacy behavior.
- `sf.Module.__call__` triggers `plasticity_step()` automatically in
  `.train()` mode. Switch to `.eval()` to suppress.
