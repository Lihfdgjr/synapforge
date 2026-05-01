# Test coverage report (T7.4)

Generated: 2026-05-02 02:09 UTC | run hash: pre-T7.4 commit `f668332`
Tool: `pytest --cov=synapforge --cov-report=term --cov-report=html --timeout=30`

## Run summary

| metric | value |
|--------|------:|
| total tests collected | 287 |
| passed | 269 |
| failed | 7 |
| skipped (CUDA / transformers / fp16 backward) | 10 |
| deselected (`-m "not slow"`) | 1 |
| ignored (long-context, > 30s timeout) | 2 files |
| wall-time | 91.9 s |

The 7 failures are pre-existing (synth_image / synth_zh / grad_accum CLI assertion); they
are **not** caused by this task. They run on CPU but assert on environment-specific
artifacts (PyTorch RNG drift, missing CLI flag in trainer). Tracking them is out of scope
for T7.4 — file as a follow-up Tier-7 maintenance item.

Long-context tests (`test_long_context_50m.py`, `test_long_context_monotonic_quality.py`)
are excluded from this run because each forward pass on the 100M model exceeds the 30s
per-test budget on CPU. They are GPU-targeted and run nightly on the rental box.

## Top-level coverage

| scope | percent | covered / total |
|-------|--------:|----------------:|
| **whole package** (incl. bench/example/in-pkg test scripts) | **18.4%** | 4018 / 21804 |
| **filtered** (excluding `bench_*.py` / `example_*.py` / `synapforge/test_*.py`) | **22.0%** | 4018 / 18253 |

The "filtered" number is the more honest one: in-package `test_*.py` files are smoke
scripts (declared in `conftest.py:collect_ignore_glob`) and `bench_*.py` / `example_*.py`
are runnable harnesses — neither is a unit-test target.

## Top 10 highest-covered modules (>= 30 statements)

| coverage | stmts | module |
|---------:|------:|--------|
| 93.1% | 58 | `synapforge/cells/plif.py` |
| 89.1% | 137 | `synapforge/action/web_actuator.py` |
| 88.5% | 52 | `synapforge/cells/liquid.py` |
| 87.6% | 137 | `synapforge/optim.py` |
| 86.8% | 174 | `synapforge/surrogate.py` |
| 86.2% | 152 | `synapforge/model_100m.py` |
| 84.5% | 97 | `synapforge/cells/synapse.py` |
| 83.0% | 47 | `synapforge/ir/compiler.py` |
| 80.4% | 204 | `synapforge/data/remote_warehouse.py` |
| 80.4% | 56 | `synapforge/__init__.py` |

Cells / surrogate / quantize / model_100m all sit above 75% — exactly the modules that
have been the focus of recent T2.x / T3.x / T5.x ships. The framework's "load-bearing"
math is well-tested.

## Top 10 lowest-covered modules (>= 30 statements, < 50%)

The full list contains **105** modules below 50%. The biggest ones (most missed
lines) are:

| coverage | stmts | missed | module |
|---------:|------:|-------:|--------|
| 0.0% | 398 | 398 | `synapforge/distributed_hetero.py` |
| 0.0% | 377 | 377 | `synapforge/learn/continual_daemon.py` |
| 0.0% | 335 | 335 | `synapforge/backends/triton_block_kernel.py` |
| 0.0% | 309 | 309 | `synapforge/trainer_mixins.py` |
| 0.0% | 281 | 281 | `synapforge/backends/cpu_avx2.py` |
| 0.0% | 214 | 214 | `synapforge/train_100m.py` |
| 0.0% | 196 | 196 | `synapforge/runtime_cuda_graph.py` |
| 0.0% | 193 | 193 | `synapforge/learn/autonomous_daemon.py` |
| 0.0% | 191 | 191 | `synapforge/safety/dpo_trainer.py` |
| 0.0% | 186 | 186 | `synapforge/eval/compare_chat.py` |

Pattern: most of the 0%-coverage giants are **GPU-only kernels** (`triton_*`, `cpu_avx2`,
`runtime_cuda_graph`), **long-running daemons** (`continual_daemon`, `autonomous_daemon`),
or **CLI training entry points** (`train_100m`, `dpo_trainer`, `compare_chat`). They
don't lend themselves to fast unit-test coverage and should be smoke-tested by the
nightly rental run, not by the CPU CI suite.

The more interesting weak spots are mid-size library modules with non-trivial
existing coverage — those are listed below as action items.

## Action items: 3 modules where 1-2 tests would lift > 5pp coverage

For these, a single import + one happy-path call would cover roughly 60-100 statements
each. The expected coverage delta below assumes ~80 lines covered per added smoke test
against the *filtered* 18253-statement denominator.

### A1 — `synapforge/infinite.py` (current 16.6%, 656 stmts, 547 missed)

This is the 100M-context FAISS / IVF_PQ / HNSW retrieval module — the "infinite context"
core named in `feedback_self_learn_and_100M_context`. Currently only the import path is
exercised. **Suggested test**: `tests/integration/test_infinite_smoke.py` —

1. Build a small `InfiniteContext` instance (vocab=50, dim=32, capacity=256).
2. Feed 50 random hidden states, query 5 of them back, assert top-1 recall > 0.5.
3. Round-trip `save()` / `load()` → another query returns the same indices.

Estimated lift: **+0.4 to +0.6pp overall**, takes the module from 16.6% to ~30%.

### A2 — `synapforge/tools.py` (current 19.6%, 372 stmts, 299 missed)

This is the 9-component tool-use subsystem (`Registry`/`WebSearch`/`Fetch`/`Scrape`/...,
see `feedback_tool_use_web_access`). All 9 components are imported but their happy-path
methods are uncalled. **Suggested test**: `tests/integration/test_tools_registry.py` —

1. Instantiate `ToolRegistry`, register one stub tool, list, dispatch by name.
2. Patch `httpx.get` (or whatever HTTP client is used) and run `Fetch.run("http://x")`
   end-to-end; assert payload routed to learning loop.
3. Run `Toolformer` self-supervision dry-run on a 100-token snippet; assert no exception
   and that at least 1 tool-token was injected.

Estimated lift: **+0.3 to +0.5pp overall**, takes module from 19.6% to ~35%.

### A3 — `synapforge/self_learn.py` (current 14.0%, 307 stmts, 264 missed)

This is the TTT / ExperienceReplay / SelfPlay / MAML self-learning pipeline. Used by the
`--self-learn-ttt` flag in trainer. **Suggested test**:
`tests/integration/test_self_learn_smoke.py` —

1. Build a tiny model (hidden=32, n_layers=2) and a `SelfLearnEngine`.
2. Run 1 TTT step on a dummy batch; assert `delta_W` non-zero.
3. Run 1 ExperienceReplay sample-and-update; assert no NaN in params.

Estimated lift: **+0.2 to +0.4pp overall**, plus this module is on the critical path for
the phase-1 self-learn gate (`MASTER_PLAN §4`), so testing it has compounding value.

**Combined estimated lift**: ~+1.0 to +1.5pp overall coverage from these 3 tests alone —
but the per-module lift is what matters: each module crosses from "untested" to "smoke
tested" which catches import / signature regressions on every CI run.

## Notes / caveats

- `htmlcov/` is generated alongside this report and is **gitignored** (entry already
  present at `.gitignore:13`). Do not commit it.
- 2 source files cannot be parsed by `coverage.py` due to embedded null bytes:
  `synapforge/chat/streaming.py` and `synapforge/memory/bm25_sidecar.py`. They're
  excluded from the totals — repair candidates for a separate Tier-7 task.
- Coverage was measured on Python 3.10, torch 2.0.1+cpu. `transformers` is intentionally
  not installed, which is why 4 tests skip with `could not import 'transformers'` —
  that's expected and matches the constraints in T7.4.
