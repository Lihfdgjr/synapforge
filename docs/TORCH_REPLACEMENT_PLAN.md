# Torch Replacement — Phased Plan

**Date:** 2026-05-02
**Owner:** lead-architect (synapforge)
**Companion docs:** `docs/TORCH_REPLACEMENT_PHASE0_AUDIT.md` (audit numbers,
per-category decisions), `docs/ANTI_LORA.md` (project ethos: 100% LNN+SNN,
no transformer fallback).

## Why now

The `synapforge` framework still depends heavily on PyTorch (`torch.Tensor`,
`torch.autograd`, `torch.nn.Module`, `torch.optim`, `torch.compile`). The
project is supposed to be a purpose-built LNN+SNN framework, not a torch
wrapper, so we are committing to a phased migration.

## Non-negotiables

* Architecture stays **100 % LNN+SNN** (CfC + PLIF + SwiGLU + RMSNorm —
  no transformer / attention).
* Training quality must not regress vs. the current torch-based path.
* Existing checkpoints must keep loading (or have a documented one-time
  conversion script).
* **Honest naming.** Don't call it "torch-free" if it still imports torch.
  The deliverable per phase is precise about which torch APIs are
  replaced, and a `synapforge.diagnostics.list_torch_imports()`
  helper (Phase 5) prints any residual torch usage on import.
* The teacher (Qwen-2.5 0.5B, frozen) stays on torch. KD-only — only
  emits logits. Justified by the project ethos.

## Phases

### Phase 1 — `synapforge.optim.AdamW` (1 week, **SHIPPED 2026-05-02**)

**Status:** done in this PR. See `synapforge/optim/adamw.py`.

* Pure-python AdamW with bias correction, eps, decoupled weight decay.
* Operates on iterables of `torch.Tensor`. Drop-in for the trainer.
* Numerics match `torch.optim.AdamW(fused=True)` to rel-err < 1e-5
  over 50-step regression (`tests/optim/test_synapforge_adamw.py`).
* `state_dict` / `load_state_dict` use the canonical
  `exp_avg` / `exp_avg_sq` / `step` keys — round-trips with a
  ckpt saved by `torch.optim.AdamW`.
* Wired as opt-in `--synapforge-adamw` flag in `train_100m_kd.py`. Default
  remains `PlasticityAwareAdamW` (no behaviour change for current runs).
* `synapforge.optim` was promoted from a single-file module to a sub-package
  to host this addition — no public API change, all
  `from synapforge.optim import X` continues to work.

**Risk:** ~2-3 % step-time vs. `--fused-adamw` (no fused CUDA kernel).
Acceptable trade — the adopting flag is opt-in and the perf gap closes
in Phase 4.

### Phase 2 — `synapforge.module.Module` finish migration (1 week, **SHIPPED 2026-05-02**)

**Status:** done in this PR (commit on `feature/torch-replacement-phase2`).
See `synapforge/module.py` for the implementation; tests under
`tests/module/`.

**Delivered:**

* `synapforge.module.Module` is the documented public base class for
  every synapforge block. Phase 2 surfaces the API contract
  explicitly: `register_parameter` / `register_module`, `parameters` /
  `named_parameters`, `state_dict` / `load_state_dict`, `.to` /
  `.cuda` / `.cpu`, `.eval` / `.train`, `.zero_grad`, plus the
  preserved plasticity hooks (`register_plasticity` /
  `plasticity_step`) and `compile_to_ir` IR entry point. Internally
  still inherits from `torch.nn.Module` — this is the deliberate
  gradual-migration anchor; Phase 3 swaps the parent class once
  `synapforge.tensor` lands.
* `synapforge.module.Parameter` is a thin subclass of
  `torch.nn.Parameter` with `requires_grad=True` defaulted on. Phase
  2 contract: state-dict-byte-equivalent to `nn.Parameter`, so
  existing `torch.save(model.state_dict())` ckpts under `runs/` round-
  trip without conversion. Verified by
  `tests/module/test_state_dict_compat.py`.
* `synapforge/model_100m.py::HybridBlock` and `SynapForge100M`
  inherit from `synapforge.module.Module` (already true pre-Phase-2);
  the inner blocks `_RMSNorm`, `_SwiGLU` flipped from `nn.Module` to
  `synapforge.module.Module` in this PR. The two direct-`nn.Parameter`
  callsites in this file (`pos_embed`, `hp_lambda`) flipped to
  `synapforge.module.Parameter`. `nn.Linear` / `nn.Embedding` /
  `nn.LayerNorm` / `nn.Dropout` stay on torch — drop-in replacements
  arrive in Phase 3.
* Bit-exact equivalence with the torch baseline:
  `tests/module/test_module_torch_compat.py` proves that a
  `synapforge.module.Module`-based block produces forward outputs
  identical (within rel-err 1e-5; in practice == 0 on CPU fp32) to
  the `nn.Module` reference for the same weights.
* State-dict round-trip: `tests/module/test_state_dict_compat.py`
  loads an existing torch-format ckpt, saves it back via the
  Phase-2 model, loads again, verifies byte-equality.

**Skipped vs. brief (deferred to Phase 3):**

* Drop-in `synapforge.module.Linear` / `RMSNorm` / `Embedding` /
  `Dropout`. These are not blockers for Phase 2 — the existing
  `nn.Linear` etc. continue to compose with our `Module` base class
  because `Module` extends `nn.Module`. The drop-in replacements
  ship alongside the `synapforge.tensor` storage migration in Phase
  3, where they'll have meaningful work to do (route their forwards
  through Triton kernels with `synapforge.tensor` I/O). Until then,
  shipping empty wrappers around `nn.Linear` would be churn.
* `PlasticityAwareAdamW` migration to a `synapforge.optim.Optimizer`
  base. Currently inherits `torch.optim.Optimizer`; the
  registration-plumbing migration is fully gated on Phase 3 (when
  we replace param-group machinery and tensor storage).

**Risk register:** no expected ckpt-format change; `compile_to_ir`
and plasticity hooks fire unchanged (preserved verbatim from v0.1
`Module`); no perf delta because we still go through `nn.Module`'s
parameter tracking. Verified across 87 unit tests in `tests/`
(`tests/test_correctness.py`, `tests/test_plasticity.py`, etc).

### Phase 3 — `synapforge.tensor` thin wrapper (1-2 weeks)

**Goal:** replace forward tensor ops in `HybridBlock` with a thin
`synapforge.tensor` wrapper that:
* holds a raw CUDA pointer + shape + dtype (still backed by `cudaMalloc`,
  not torch's caching allocator — see Risk Register below).
* dispatches hot ops (`matmul`, `layernorm`, `softmax`, residual `add`)
  to Triton kernels.
* falls back to `torch.Tensor` ops for cold ops we haven't ported yet.
  Marker: each unported op gets a `_SF_TORCH_FALLBACK[<op>] += 1`
  counter so we can see what's left in `synapforge.diagnostics`.
* same `__add__` / `__matmul__` / `.shape` / `.dtype` API as `torch.Tensor`
  for in-place ops, so callsites are largely textual renames.

**Risk:** Triton kernels currently take `torch.Tensor` as I/O. Phase 3
introduces a `from_torch` / `to_torch` adapter at the kernel entry/exit
so this is a no-op at first; we collapse it once `synapforge.tensor`
is the canonical type.

### Phase 4 — `synapforge.autograd` (3-4 weeks)

**Goal:** replace `torch.autograd.Function` with `synapforge.autograd.Function`.

Strategy:
* `synapforge.autograd.Function` initially **inherits from**
  `torch.autograd.Function` (transitional adapter — same `forward`/`backward`
  contract). Existing call sites keep working.
* Each kernel migrates to closed-form VJPs already living in
  `synapforge/backends/triton_fused_backward.py` (already exists for the
  hottest paths). The remaining hand-derived gradients (PLIF surrogate,
  CfC) get lifted from `surrogate.py:8` and `cells/liquid.py`.
* Once **every** Function-using site is on `synapforge.autograd.Function`,
  we cut the `torch.autograd.Function` parent class and the migration is
  complete.
* The 29 `torch.autograd.Function` call sites (most concentrated in
  `surrogate.py`) port one file at a time over ~3-4 weeks.

**Risk:** see Risk Register §3 (autograd graph + multi-step backward
ordering). Mitigations: keep `torch.no_grad()` semantics by default in
`synapforge.autograd`, mirror the existing `inputs.requires_grad`
predicate logic.

### Phase 4 — partial shipment (2026-05-02)

**Status:** SEW + sigmoid-gate fused fwd+bwd shipped on branch
`feature/triton-block-bwd-closed-form`. See
`synapforge/backends/triton_block_kernel_bwd.py` and the audit at
`docs/TRITON_BLOCK_BACKWARD_AUDIT.md`.

What landed:
* Closed-form fused Triton fwd+bwd for the elementwise tail of
  `HybridBlock`: `s + h` (SEW) + `sigmoid(gate_pre)` + `syn_out * gate`.
  Three torch kernel launches collapsed into ONE Triton launch fwd
  and ONE Triton launch bwd.
* Plugged in via `torch.autograd.Function` (`SEWSigmoidGateFn`) — the
  rest of the autograd graph (Linear, SparseSynapse, RMSNorm) is
  unchanged, so this is a strict perf-only swap.
* Numerical correctness verified to **rel-err < 1e-7** on cpu fallback
  path (exact through SEW-add and `grad_syn = gg * gate`; 8.87e-7
  noise on `grad_gate_pre` due to fp32 round-trip).
* Benchmark `scripts/bench_hybrid_block.py` with shape (B=24, T=256,
  D=1280) defers cuda-side numbers to the rental (local Windows is
  CPU-only torch).
* CfC + PLIF + reset bwd was already Triton-fused
  (`fused_lnn_snn_block_bwd_kernel` in
  `synapforge/backends/triton_block_kernel.py`) — this PR documents
  that and reuses it. No change to that kernel.

What's left for full Phase 4:
* RMSNorm Triton bwd (LOW priority — `RMSNorm.weight` grad and
  `dY/dx` already cuBLAS-bound for d=1280).
* SwiGLU Triton bwd (LOW priority — three cuBLAS matmuls dominate;
  fusing the silu+mul into Triton beats nothing if matmuls stay torch).
* `synapforge.autograd.Function` — replace
  `torch.autograd.Function` parent class. Currently `SEWSigmoidGateFn`
  inherits from `torch.autograd.Function`; the migration is a 1-line
  parent-class swap once Phase 4 generic infra lands.
* Allocation pooling (`synapforge.cuda.MemPool`) — separate work item,
  see Risk Register §2.

Estimated remaining for full Phase 4: 2 weeks (`synapforge.autograd`
infra + RMSNorm/SwiGLU Triton bwd).

### Phase 5 — drop `import torch` from student path (final, ~1 week)

Once Phases 1-4 land:

* `import torch` survives only in `synapforge.kd_teacher` (frozen
  Qwen-2.5 teacher) and `synapforge.interop_torch` (HF interop bridge).
* `synapforge.diagnostics.list_torch_imports()` confirms there are no
  other `import torch` statements in `synapforge/`.
* CI gate: `tests/test_no_residual_torch_imports.py` greps the
  package and fails the build if a new `import torch` is added outside
  the allowlist.
* `train_100m_kd.py` becomes the single torch-touching file in the
  trainer surface and only because it imports the teacher.

## Risk Register

### 1. `torch.compile` is hard to replace

**Severity:** N/A.
**Mitigation:** none needed. Audit (§Phase 0 audit, category f) shows
`torch.compile` is **already off in production** — the `--torch-compile`
flag in `train_100m_kd.py` is `default=False` and conflicts with our
Triton kernels. Phase 5 simply removes the dead flag. If we ever need
graph-level optimization later, we already write Triton manually for
hot paths and that's the long-term replacement.

### 2. `torch.cuda` allocator caching (cuMemPool replacement)

**Severity:** HIGH — this is the only torch-API surface where naively
porting loses perf.

**Background:** PyTorch's caching allocator pools `cudaMalloc`d blocks
and re-uses them within a process. Without it, every `torch.empty(...)`
becomes a `cudaMalloc` syscall — measured in our profile data as ~30 µs
per allocation. At 100k allocations / training step that's 3 seconds /
step, untenable.

**Mitigation plan:**
* Phase 3: keep the underlying `torch.Tensor` storage class (and thus
  the caching allocator) while we migrate the *ops* to
  `synapforge.tensor`. The wrapper holds the same caching-allocated
  storage.
* Phase 4: introduce `synapforge.cuda.MemPool` — thin wrapper over
  `cuMemPool_*` (CUDA 11.2+) and `cudaMallocAsync` for stream-ordered
  allocs. This is bookwork (~1k lines) but well-trodden — `cudf` and
  `cupy` both ship one.
* Phase 5 cutover: replace `torch.empty` / `torch.zeros_like` calls
  in `synapforge.tensor` with `synapforge.cuda.MemPool.alloc()`.
* **Validation:** add `tests/perf/test_alloc_throughput.py` that
  measures alloc/free of 1M tensors and asserts < 5 µs/op (matching
  torch's caching allocator).

If `synapforge.cuda.MemPool` lands but is more than 20 % slower than
torch, we **revert** that step and keep torch storage. The plan
remains "torch-free everywhere except the allocator backend" until we
close that gap.

### 3. `torch._foreach_*` for fused optimizer step

**Severity:** MEDIUM — the perf gap is real (~2-3 % step time).

**Background:** `torch.optim.AdamW(fused=True)` uses
`torch._foreach_mul_`, `_foreach_addcmul_` etc. to coalesce per-param
ops into single CUDA dispatches. Our pure-python AdamW (Phase 1) does
one Python loop iteration per parameter — fine for ~340 trainable
params on a 100M model, but a 5-10× perf drop on a 1B+ model.

**Mitigation:** Phase 4 adds a Triton kernel `adamw_step.triton.py`
that does the equivalent fused update. Until then, the perf gap is the
"price of getting out of `torch.optim`" and is opt-in via
`--synapforge-adamw` (default OFF). Production runs continue to use
`PlasticityAwareAdamW` or `--fused-adamw`.

### 4. `torch.serialization` (ckpt save/load)

**Severity:** MEDIUM — high blast radius (warmstart of all production
runs depends on this).

**Background:** All ckpts in `runs/` were saved with `torch.save`,
which uses Python pickle internally. We want safetensors for safer
serialization (no pickle RCE risk) and to fully drop torch.

**Mitigation:**
* Insert a dedicated Phase 2.5 (post-Phase 2, pre-Phase 3) for
  ckpt-format migration.
* Step 1: `synapforge.io.save_ckpt(path, model, optim, ...)` defaults
  to safetensors but can read either format on load.
* Step 2: `scripts/convert_torch_ckpt_to_st.py` — one-time conversion
  with a `--verify` flag that loads both, runs an inference batch, and
  asserts logit-equality.
* Step 3: deprecation period of 2 weeks before flipping load-default
  to safetensors-only.
* **Backwards-compat guarantee:** existing torch ckpts continue to
  load until Phase 5. Documented in `CHANGELOG.md`.

### 5. Distributed (DDP)

**Severity:** LOW — currently dead code.

**Background:** `synapforge/distributed.py` and `parallel.py` use
`torch.distributed.*`. Production today is A800-single-card; multi-GPU
runs were last seen on the rented A100×2 (now offline).

**Mitigation:** **DROP** these files when we get to Phase 5. If we ever
need DDP again:
* `pynccl` provides the same NCCL primitives without `torch.distributed`.
* The `cb-mpc` patterns we saw in earlier audits suggest a custom
  `synapforge.distributed.AllReduce` is ~2 weeks of work.
* Until that demand materializes, the DDP path is gated behind
  `if torch.cuda.device_count() > 1` and never fires.

### 6. AMP (autocast / GradScaler)

**Severity:** LOW — not blocking the migration, but eats a perf win
if we drop it before we have a replacement.

**Mitigation:** keep AMP inside the `torch.Tensor` envelope through
Phase 3. Phase 4's `synapforge.tensor` adds a `dtype` field; we then
implement `synapforge.amp.autocast` as a context manager that flips
the default dtype + a `synapforge.amp.GradScaler` that mirrors torch's.
Loss-scale heuristics are well-documented.

### 7. Hidden torch usage in HuggingFace adapters

**Severity:** LOW — but worth flagging.

`synapforge/huggingface_adapter.py` and `hf_trainer.py` are by-design
torch-coupled. They wrap `transformers` which IS torch. We should
NOT attempt to make these torch-free; they exist as the bridge for
external KD teachers. The Phase 0 audit explicitly excludes them
from the migration scope.

### 8. Existing call sites we missed

**Severity:** UNKNOWN — the audit relies on regex over `.py` files
and may miss edge cases.

**Mitigation:** Phase 5 introduces a CI check
(`tests/test_no_residual_torch_imports.py`) that fails the build if
any new `import torch` lands outside the allowlist
(`synapforge.interop_torch`, `synapforge.kd_teacher`,
`synapforge.huggingface_adapter`, `synapforge.hf_trainer`).
This catches drift after the migration completes.

## Timeline summary

| Phase | Description                              | Calendar | Status   |
|-------|------------------------------------------|----------|----------|
| 1     | `synapforge.optim.AdamW`                 | 1 wk     | **SHIPPED 2026-05-02** |
| 2     | `synapforge.module.Module` finish        | 1 wk     | **SHIPPED 2026-05-02** |
| 2.5   | safetensors ckpt migration               | 0.5 wk   | planned  |
| 3     | `synapforge.tensor` thin wrapper         | 1-2 wks  | planned  |
| 4     | `synapforge.autograd`                    | 3-4 wks  | partial 2026-05-02 (SEW+gate fused bwd shipped) |
| 5     | drop `import torch` from student path    | 1 wk     | planned  |

Total: ~7-9 calendar weeks for a one-engineer migration.

## How to verify Phase 1 (anyone can repro)

```pwsh
# 1. Run the unit tests
.venv\Scripts\python.exe -m pytest tests/optim/test_synapforge_adamw.py -v

# 2. Train a smoke run with --synapforge-adamw
.venv\Scripts\python.exe train_100m_kd.py --synapforge-adamw --max-steps 100 --save-every 50 [...other flags]
# Expected: numerics indistinguishable from --fused-adamw on the same seed
```

## Open questions

* Phase 4: do we want to ship `synapforge.autograd` as a `torch.autograd`
  superset (allow torch ops alongside) or strict subset (refuse torch
  tensors at the autograd boundary)? Decision deferred — likely strict
  subset once Phase 3 is done, lenient before that to ease migration.
* Phase 5 timing: should we do it before or after the next investor demo?
  Argues for **before** (frees us to claim "torch-free student path") but
  **after** (avoids a potentially-buggy Phase 4-5 derail blocking the demo).
  Defer to project lead.
