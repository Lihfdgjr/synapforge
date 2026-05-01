# Profiling SynapForge training

When `tok/s` drops or you want to know where the budget goes, run the
per-stage profiler before reaching for guesswork. This page covers:

1. How to run `scripts/profile_train_step.py`.
2. How to read the output and bucket the run into dataloader-bound,
   compute-bound, or KD-teacher-bound.
3. Top 10 known bottlenecks with mitigations (drawn from the actual
   training runs of the 100M+Qwen-vocab+triton+KD pipeline on A800).
4. Cross-references to `synapforge/parallel.py` for CPU thread tuning and
   `feedback_triton_backend_required.md` for backend selection.

---

## 1. Running the profiler

```bash
# Smoke run on CPU/laptop. Default smoke settings; tiny model + tiny vocab.
python scripts/profile_train_step.py \
    --warmup 100 --steps 100 \
    --output runs/profile/step.json --visualize ascii

# Full A800 profile that mirrors the real KD trainer.
python scripts/profile_train_step.py \
    --warmup 100 --steps 100 \
    --batch-size 64 --seq-len 256 --hidden 512 --vocab 151643 \
    --device cuda \
    --output runs/profile/a800.json
```

`--steps 100` is enough to drive jitter < 5%. The `--warmup` phase fills the
caching allocator and triggers torch.compile/triton autotune so the measured
window is steady-state.

Under the hood, every measured iteration runs through six tagged stages:

| Stage                | What it covers                                       |
| -------------------- | ----------------------------------------------------- |
| `dataloader`         | `next(loader)` + H2D copy                            |
| `forward_student`    | student model forward, autocast wrapped              |
| `forward_teacher`    | teacher (frozen) forward — only on `kd_every` steps  |
| `kd_loss`            | CE + KL + z-loss                                     |
| `backward`           | `loss.backward()` (autograd graph traversal)         |
| `optimizer_step`     | AdamW step + grad clip + zero_grad                   |

`torch.profiler` records each stage's CPU + CUDA self-time and the per-stage
peak GPU memory (via `reset_peak_memory_stats` between stages), and emits a
`chrome://tracing` JSON at `--output`. Open it in
[perfetto.dev](https://ui.perfetto.dev/) for an interactive flamegraph.

A second file `step.summary.json` is written with the per-stage statistics
suitable for CI dashboards. Schema:

```json
{
  "device": "cuda",
  "tok_per_s": 28934,
  "stage": {
    "forward_student": {"mean_ms": 7.21, "std_ms": 0.4, "p95_ms": 7.9, "pct": 23.1, "mem_mb": 4310},
    ...
  },
  "stage_raw_ms": { "forward_student": [...], ...}
}
```

---

## 2. Reading the output

The CLI prints a per-stage table sorted in canonical order:

```
=== per-stage timing (mean ± std over measured steps) ===
stage                   mean_ms   std_ms   p95_ms    pct     mem_MB
------------------------------------------------------------------------
dataloader                 4.21     0.20     4.50    14%       128   ###---
forward_student            7.34     0.32     8.12    24%      4310   ######-
forward_teacher           11.82     0.59    12.77    38%      6011   ##########
kd_loss                    0.81     0.07     0.95     3%      4310   #------
backward                   5.20     0.41     5.91    17%      4520   ####---
optimizer_step             1.20     0.04     1.27     4%      4520   #------
```

Use this rule of thumb:

- `dataloader pct > 20%` → **dataloader-bound**. Check `num_workers`, parquet
  shard sizes, or whether the tokenizer call is on the hot path. See
  `synapforge/parallel.py::optimize_cpu_threads` to size MKL/OMP threads to
  leave headroom for the loader.
- `forward_teacher pct > 30%` → **KD-bound**. Either (a) raise `--kd-every`
  so the teacher fires less often, (b) move the teacher to bf16 (it's
  frozen — no precision concern), or (c) use a smaller teacher (Qwen-0.5B
  vs the 1.5B variant).
- `forward_student + backward pct > 60%` and `tok/s` low → **compute-bound**.
  Switch backend to `triton_block` (see `feedback_triton_backend_required.md`:
  `gpu_dense` 6.7k → `triton_block` 42k tok/s on A800) and ensure
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported.
- `optimizer_step pct > 10%` is unusual. Adam's foreach kernels are usually
  <2 ms on a 100M model. If it's elevated, suspect `set_to_none=False` or a
  fused kernel that's failing back to the per-tensor path; rerun with
  `torch.optim.AdamW(..., fused=True)` if available.

The follow-up table lists the top-30 ops by CUDA self-time (or CPU on
CPU-only runs) — that's the fast path to spotting that a custom kernel is
silently falling back to `aten::mm`.

### Chrome / Perfetto trace tips

Open `runs/profile/a800.json` in [perfetto.dev](https://ui.perfetto.dev/)
and look at the row labelled `python_function` (CPU side) layered over the
`gpu0` track. Common diagnoses:

- **Long python_function gaps with idle GPU**: dataloader stall.
- **Solid GPU activity but CPU spinning at sync points**: kernel launch
  overhead from many small kernels — fuse via triton_block.
- **GPU underused at start of step**: prefetcher not pinning memory; pass
  `pin_memory=True` and `non_blocking=True` to `.to(device)`.

---

## 3. Top-10 known bottlenecks (with mitigations)

| # | Symptom in profile                                  | Cause                                                                 | Mitigation                                                            |
| - | --------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 1 | `forward_student` mean 30ms+, GPU at 30% util       | Backend = `gpu_dense` (PyTorch passthrough)                          | `--backend triton_block` — 6.3× speedup measured                      |
| 2 | `dataloader` pct > 25%                              | Single-process loader, parquet shards too large                       | DataLoader `num_workers=4-8`, shard parquet to ~256MB each            |
| 3 | `forward_teacher` pct > 40%                         | Teacher running every step in fp32                                    | `--kd-every 2 or 4`; cast teacher with `torch.autocast(bf16)`         |
| 4 | top-op = `aten::mm` for tens of thousands of calls  | Block kernel autotune fell back to dense matmul                       | Pin Triton 2.1.0+, set `TRITON_CACHE_DIR` to writable, redo warmup    |
| 5 | step time spikes every 8 steps, std > 30% of mean   | CUDA caching allocator fragmentation                                  | Export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`             |
| 6 | `backward` mean 2× forward_student                  | Per-layer activation checkpointing reactivated                        | Disable `--grad-checkpoint` once batch fits comfortably                |
| 7 | `optimizer_step` > 5ms                              | Optimizer in non-fused path                                           | `torch.optim.AdamW(..., fused=True)`; or our `PlasticityAwareAdamW`    |
| 8 | tok/s < 5000 even on small model on CPU             | OMP threads = 1, MKL threads default                                  | Call `optimize_cpu_threads()` from `synapforge.parallel`               |
| 9 | top-op `aten::index` heavy in forward               | Embedding lookups on long-tail vocab tokens                            | Use `nn.Embedding(..., sparse=True)` for the LM head adapter          |
| 10| chrome trace shows single-stream GPU activity       | All ops on default stream, no overlap                                 | Wrap teacher forward in a side stream, sync before kd_loss             |

The numbers in column 1 cross-reference to anchor measurements:

- Row 1: `feedback_triton_backend_required.md` (gpu_dense ~6.7k vs
  triton_block ~42k tok/s, A800 80GB, batch=128, seq=256, vocab=151643).
- Row 5: `feedback_triton_backend_required.md` also documents the
  `expandable_segments` flag pairing.
- Row 8: `synapforge/parallel.py::optimize_cpu_threads` sets OMP/MKL based
  on `os.cpu_count()` and reserves threads for the dataloader.

---

## 4. CPU thread tuning

`synapforge/parallel.py` exposes `optimize_cpu_threads(reserve_for_dataloader=1)`
which sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and PyTorch intra/inter-op
threads to a sane default. Call it once at the top of your trainer:

```python
from synapforge.parallel import optimize_cpu_threads, print_setup
optimize_cpu_threads(reserve_for_dataloader=2)
print_setup()
```

When the profiler shows `dataloader pct > 25%`, raise
`reserve_for_dataloader` to 2-4 so PyTorch's intra-op pool doesn't crowd
out the loader.

---

## 5. Continuous regression with `perf_regression_test.py`

Pair `profile_train_step.py` with `scripts/perf_regression_test.py` for
CI-grade gating. The regression test:

- Runs 100 measured steps on a tiny model (CPU smoke-friendly).
- Persists per-stage means in `tests/perf_baseline.json`.
- Fails on `tok/s < floor` OR any stage exceeding 1.20× its baseline.

Wire it into a GitHub Action / pre-merge hook:

```yaml
- name: Perf regression
  run: python scripts/perf_regression_test.py --device cpu --baseline-tok-per-s 5000
```

On the A800 box, run with `--device cuda --baseline-tok-per-s 25000` so a
20% drop on the actual hot path also fails CI.

---

## 6. Workflow checklist

1. Suspected slowdown? Run `profile_train_step.py` first — never guess.
2. Ratio above expectations on `forward_teacher`? Drop `--kd-every`.
3. Big std in `dataloader`? Increase loader workers; check parquet shards.
4. Top-op a fallback like `aten::mm`? Triton kernel didn't compile.
5. Net tok/s still low after fixes? Re-run with `--no-trace` to remove
   profiler overhead and confirm the baseline.
6. Save the chrome trace into `runs/profile/<date>.json` and link from the
   PR description so reviewers can replay.

For the LNN/SNN-specific kernel paths (CfC parallel scan, PLIF surrogate,
SparseSynapse), the profiler labels each op with the registered kernel
name. If you see `aten::*` instead of `synapforge::*` in the top-30 table,
the custom kernel is not registered and the run is on the slow path —
that's a one-line fix in the backend init.
