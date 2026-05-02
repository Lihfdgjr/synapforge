# Torch Replacement — Phase 0 Audit

**Date:** 2026-05-02
**Scope:** entire `synapforge/` package + top-level trainers
  (`train_100m_kd.py`, `train_100m_sft.py`, `train_100m_self_distill.py`,
  `train_100m_rl.py`).
**Audit method:** ripgrep over all `.py` files for `torch.*` symbol use.
**Goal:** scope and categorize every distinct PyTorch API surface so
Phase 1-5 of the replacement plan (see `TORCH_REPLACEMENT_PLAN.md`)
can be sized accurately.

## TL;DR

| Category               | Total call sites | Files | Hot-path | Decision               |
|------------------------|-----------------:|------:|---------|------------------------|
| (a) tensor ops         |          ~1,400+ |  ~120 | YES     | REPLACE (Phases 3-4)   |
| (b) autograd           |             ~283 |  ~130 | YES     | REPLACE (Phase 4)      |
| (c) optim              |              ~30 |   ~25 | YES     | REPLACE (Phase 1, **shipped**) |
| (d) modules / params   |             ~790 |  ~127 | YES     | REPLACE (Phase 2)      |
| (e) cuda / device      |             ~274 |  ~100 | mixed   | KEEP via thin shim     |
| (f) compile / jit      |              ~47 |   ~16 | NO      | DROP (no replacement needed) |
| (g) IO / serialization |             ~101 |   ~53 | NO      | REPLACE (safetensors)  |
| (h) autocast / amp     |              ~36 |   ~15 | NO      | KEEP for now           |
| (i) distributed        |              ~12 |    ~8 | NO      | DROP post-A800-single  |

Numbers are approximate (ripgrep counts include comments, docstrings,
and tests). The action item per category is the third column; counts
exist primarily to size the work, not to be tracked precisely.

## Per-category breakdown

### (a) Tensor ops (~1,400 sites, 120 files)

`torch.cat`, `torch.stack`, `torch.zeros_like`, `torch.matmul`,
`torch.softmax`, `torch.tanh`, `torch.sigmoid`, `torch.no_grad`,
arithmetic methods (`mul_`, `add_`, `addcmul_`, `addcdiv_`, `sqrt`)
on `torch.Tensor`.

Top hot-path locations:
* `synapforge/cells/liquid.py` — CfC forward (`torch.tanh`,
  `torch.sigmoid`, `torch.matmul`)
* `synapforge/cells/plif.py` — PLIF surrogate forward
* `synapforge/backends/triton_block_kernel.py` — Triton kernels DO
  bypass torch tensor ops by going to raw CUDA pointers, but the
  driver code in `triton_block.py` still uses torch tensors as I/O
* `synapforge/world_model.py`, `synapforge/wave_mixer.py`,
  `synapforge/latent_thinking.py` — large surface here
* `synapforge/surrogate.py` — atan / fast surrogate spike functions

Decision: **REPLACE in Phase 3** with a `synapforge.tensor` thin wrapper
that holds a raw CUDA pointer + shape/dtype metadata, dispatches hot
ops (matmul, layernorm, softmax) to Triton kernels, and falls back to
torch ops for cold ops while we port them. Phase 4 then replaces the
storage backing too.

### (b) Autograd (~283 sites, 130 files)

`backward()`, `requires_grad`, `detach`, `torch.no_grad`,
`torch.autograd.Function` (only ~29 sites).

`torch.autograd.Function` call sites (full list — these are the only
hard places):
* `synapforge/backends/cpu_avx2.py` — 4 (cpu inference path; LOW priority)
* `synapforge/backends/gpu_dense.py` — 1
* `synapforge/backends/triton_fused_backward.py` — 2
* `synapforge/backends/triton_block_kernel.py` — 1
* `synapforge/cells/synapse.py` — 1
* `synapforge/cells/plif.py` — 1
* `synapforge/runtime.py` — 2
* `synapforge/quantize.py` — 1
* `synapforge/surrogate.py` — 8 (largest — surrogate gradients for SNN)
* `synapforge/adversarial.py` — 2

Decision: **REPLACE in Phase 4** with `synapforge.autograd.Function`.
Triton kernels already have closed-form gradients in
`triton_fused_backward.py`; the rest (PLIF surrogate, CfC) have
hand-derived gradients we can lift directly. The transitional adapter
layer in Phase 4 keeps `torch.autograd.Function` as the parent class
of `synapforge.autograd.Function` so existing code keeps working
while we migrate one kernel at a time.

### (c) Optimizers (~30 sites, 25 files) — Phase 1 SHIPPED 2026-05-02

`torch.optim.AdamW`, `torch.optim.SGD`, `torch.optim.Optimizer`.

Production training calls (the only ones that matter):
* `train_100m_kd.py:1931` — `torch.optim.AdamW(fused=True)` (--fused-adamw path)
* `train_100m_kd.py` (now also) — `synapforge.optim.AdamW` (--synapforge-adamw, **NEW**)
* `train_100m_kd.py:1943` — `synapforge.optim.PlasticityAwareAdamW`
* `train_100m_sft.py`, `train_100m_self_distill.py`,
  `train_100m_rl.py` — all use `synapforge.optim.build_optimizer`

Cold-path / examples / tests (~25 sites): trivial, can mass-migrate
in a follow-up commit.

Decision: **REPLACED (opt-in) in Phase 1**. See
`synapforge/optim/adamw.py` and `tests/optim/test_synapforge_adamw.py`.
Numerics match `torch.optim.AdamW(fused=True)` to rel-err < 1e-5
(verified). Wired as opt-in `--synapforge-adamw` flag.
`PlasticityAwareAdamW` still inherits from `torch.optim.Optimizer`
because it lives on the plasticity path which depends on torch's
state_dict machinery; that gets cleaned up in Phase 2.

### (d) Modules / Parameters (~790 sites, 127 files)

`torch.nn.Module`, `torch.nn.Parameter`, `torch.nn.Linear`,
`torch.nn.LayerNorm`, `torch.nn.Embedding`, `torch.nn.Dropout`,
`torch.nn.functional.*`.

Decision: **REPLACE in Phase 2**. `synapforge.module.Module` already
exists (inherits from `nn.Module`, adds plasticity hooks). Phase 2
finishes the migration so all of `synapforge/cells/`, `routers/`,
`moe/`, `modal/` inherit from `synapforge.module.Module` directly,
and we replace `nn.Linear` / `nn.LayerNorm` with `synapforge`
equivalents. The parent `nn.Module` continues to back parameter
registration in Phase 2 — full removal happens in Phase 3.

### (e) CUDA / device (~274 sites, 100 files)

`torch.cuda.is_available()`, `torch.cuda.synchronize`,
`torch.cuda.Stream`, `.cuda()`, `.to(device)`, `.device`.

Decision: **KEEP** behind a thin shim `synapforge.device`. CUDA
runtime calls are stable across torch versions and reimplementing
them via `cuda-python` adds a brittle dependency for zero perf gain.
Shim should expose: `current_device()`, `synchronize()`,
`is_available()`, `Stream`, `Event`, `memory_allocated()`. ~10 hours
work, mostly mechanical.

### (f) compile / jit (~47 sites, 16 files)

`torch.compile`, `torch.jit.*`.

Top users:
* `train_100m_kd.py:9 sites` — already gated behind `--torch-compile`
  flag; default OFF in production due to instability with our custom
  Triton kernels
* `synapforge/backends/gpu_dense.py:1` — single-tensor compile path
* `synapforge/interop_torch.py:4` — interop adapter
* `scripts/bench_torch_compile.py:13` — bench script (not training)

Decision: **DROP**. We never ship with `torch.compile` enabled (it
fights our Triton kernels and adds ~5 % spurious overhead even when
nominally enabled). All hot kernels are already manually written in
Triton. Replacement = remove the flag + the `compile` calls. Easy,
zero risk. `torch.jit` calls are all in dead/legacy paths.

### (g) IO / serialization (~101 sites, 53 files)

`torch.save`, `torch.load`.

Top users (all production):
* `train_100m_kd.py:5` — ckpt save/load (most critical)
* `train_100m_sft.py:3`, `train_100m_self_distill.py:3`,
  `train_100m_rl.py:2` — same pattern
* `synapforge/training/ema.py:3` — EMA shadow snapshots
* `legacy/*.py` — many; ignore (not in production path)

Decision: **REPLACE with safetensors** in a dedicated phase between
Phase 2 and Phase 3. Already have a hard requirement (no torch ckpt
loading post Phase 5) and safetensors is well-tested. Provide a
one-time conversion script `scripts/convert_torch_ckpt_to_st.py`.
Estimated 2-3 days for the trainer wiring + tests.

### (h) Autocast / AMP (~36 sites, 15 files)

`torch.cuda.amp.GradScaler`, `torch.autocast`.

Top users: `train_100m_kd.py:3` (the live path),
`train_100m_*.py` (other trainers), and a couple of legacy files.

Decision: **KEEP for now**. AMP is a meaningful perf win we can't
easily replicate without a torch-equivalent dtype dispatch. Revisit
in Phase 4 once we have `synapforge.tensor` dtype machinery; until
then, AMP stays inside the `torch.Tensor` envelope.

### (i) Distributed (~12 sites, 8 files)

`torch.distributed.*`.

Used only in `synapforge/distributed.py`, `synapforge/parallel.py`, a
couple of multi-GPU test files, and benchmarks. Production training
on A800 is single-card — DDP is dead code today.

Decision: **DROP** post-A800-single-card phase. If we ever need
multi-card again, NCCL has a Python binding (`pynccl`) that gives us
the same primitives without `torch.distributed`. Until then, leaving
the imports gated behind `if torch.cuda.device_count() > 1` is fine.

## Top 3 surprises during audit

1. **`torch.optim` is a tiny surface** (~30 call sites, of which
   maybe 6 actually run in production). Phase 1 (the
   `synapforge.optim.AdamW`) is genuinely the easiest win — it
   removes a torch-API surface we'd otherwise have to drag along
   forever, and the work fits in <300 lines of pure-python.

2. **`torch.autograd.Function` is concentrated in `surrogate.py`**.
   Of the 29 total `torch.autograd.Function` definitions, 8 are in
   `surrogate.py` (SNN spike surrogates). That's a single file we can
   port in one sitting — and once it's gone the entire
   `synapforge/cells/` directory has only 2 remaining
   `Function` subclasses. Phase 4 is much smaller than the headline
   "replace autograd" implies.

3. **`torch.compile` has been off in production the entire time**.
   The flag exists in `train_100m_kd.py` and is heavily documented,
   but it's `default=False` and we've never shipped with it on
   because it conflicts with our custom Triton kernels. The whole
   "replacing `torch.compile`" risk in the brief is moot — there's
   nothing to replace. Phase 5 just deletes the dead flag.

## Files NOT in scope (by intent)

* `synapforge/interop_torch.py` — by definition a torch interop
  surface; should keep `import torch`.
* `synapforge/huggingface_adapter.py`, `synapforge/hf_trainer.py` —
  HuggingFace integration, must keep torch.
* `synapforge/distill.py` — KD path imports the teacher (Qwen-2.5
  0.5B in production). Teacher is allowed to keep torch
  per the project ethos (frozen, only emits logits).
* All `legacy/*.py` files (16 of them) — dead code, will be deleted
  before Phase 5.

## Gratuitous torch usage (flagged, not fixed in Phase 1)

While auditing I noticed several places where `torch.tensor([1.0])`
or `torch.zeros(())` is used where a plain Python scalar would do.
Examples:

* Multiple `torch.tensor([0.0])` initializations of running statistics
  in `synapforge/cells/plif.py` and `bio/*.py`
* `torch.full_like(x, 0.0)` followed immediately by `.zero_()` in a
  couple of places (one of these would do)

These don't move the torch-removal needle but they're cleanup work
we should do alongside Phase 2. Not fixing in this commit.

## Cross-references

* Plan + risk register: `docs/TORCH_REPLACEMENT_PLAN.md`
* Phase 1 deliverable: `synapforge/optim/adamw.py`,
  `tests/optim/test_synapforge_adamw.py`
* Wired-up flag: `--synapforge-adamw` in `train_100m_kd.py`
