# Integration Tests

This document describes the cross-module integration suite under
`tests/integration/`. The suite is **distinct** from the unit tests
under `tests/test_*.py`: unit tests verify a single module in
isolation; integration tests verify the *wiring between modules*.

The June 2026 audit found that ~10 modules had passing standalone smoke
tests but never exercised their cross-module connections. The integration
suite is the answer.

## Why this exists

Concrete failures the suite catches that unit tests do not:

- `phase_manager.write_phase_signal` writes a `.phase` JSON whose
  key names match what `phase_signal.consume_phase` expects, AND the
  flag list mirrors the canonical `PHASES` table.
- `chat_eval_gate` actually separates word-salad from coherent output
  (i.e. the heuristic gate is not a no-op).
- `skill_log_v2` recovers byte-for-byte from a corrupt-mid-write file
  using its rotation siblings.
- `triple_backup_daemon` fires its loud-WARN on the 5th empty cycle
  (the systemd-points-at-wrong-path footgun).
- `auto_eval_daemon` detects a fresh ckpt drop AND deduplicates on
  the second pass (sha256-based).
- `prep_alpaca_qwen` and `chat_repl` use the same instruction template
  byte-for-byte (drift here causes the v2.5 wikitext-leak class of bug).
- `universal_codebook.mint_from_co_firing` actually mints when given a
  repeating pattern with the correct trigger_seq metadata.
- `rfold_bench` and `verify_rfold` produce numerically consistent
  correctness numbers (both <1e-3 at R=1, <10% at R=8).
- `synapforge.demo.cli` dispatches subcommands without crashing.

## Test inventory

| # | Test                                                | Sub-claim                                               | Heavy deps    |
|---|-----------------------------------------------------|---------------------------------------------------------|---------------|
| 1 | `test_phase_signal_write_consume_cycle`             | write -> read -> consume -> idempotent                  | none          |
| 2 | `test_chat_eval_gate_scores_good_and_bad`           | gate separates word-salad from coherent                 | none          |
| 3 | `test_skill_log_v2_save_kill_recover`               | mid-write kill, rotation-sibling recovery, idempotent   | torch         |
| 4 | `test_triple_backup_daemon_empty_dir_warning`       | WARNING on 5th empty cycle, recovery msg on first file  | none          |
| 5 | `test_auto_eval_daemon_detects_fresh_ckpt`          | fresh ckpt detection + sha256 dedup on second pass      | torch         |
| 6 | `test_alpaca_to_sft_pipeline`                       | prep -> SFT loss -> chat template byte-equal            | torch         |
| 7 | `test_universal_codebook_l1_to_l2_co_firing_mint`   | repeated [0,1,2] mints L2 with that trigger_seq         | torch         |
| 8 | `test_rfold_bench_matches_verify_rfold`             | R=1 < 1e-3, R=8 < 0.10, 5 documented shapes covered     | torch         |
| 9 | `test_synapforge_demo_all_smoke`                    | `synapforge-demo bench` dispatches through cli.main     | torch         |

Tests gated on `torch` skip cleanly via `pytest.mark.skipif` when torch
is not installed. Tests that need `playwright` use
`pytest.mark.needs_playwright` and are skipped via the conftest hook.

## Running on a clean clone

```bash
git clone https://github.com/Lihfdgjr/synapforge.git
cd synapforge
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e ".[dev]"   # also installs pytest, pyarrow

# Single-line entry: runs full integration suite + prints summary
bash tests/integration/run_smokes.sh

# Or directly:
pytest tests/integration -v

# Run one test:
pytest tests/integration/test_end_to_end.py::test_phase_signal_write_consume_cycle -v
```

Expected runtime on CPU (no GPU, no rental access):

```
9 passed in ~5 seconds
```

If you don't have torch installed locally, the 6 torch-dependent tests
skip; the remaining 3 run and report PASS in <1 second total. CI sees
the skips via `pytest -ra` and does not treat them as failures.

## What CI should run pre-merge

A reasonable GitHub Actions job:

```yaml
- name: Integration smoke
  run: bash tests/integration/run_smokes.sh
```

The suite is hermetic (no internet, no GPU, no rental access required)
and finishes in <30 seconds, so it can run on every PR. We recommend:

1. **Pre-merge gate (every PR)**: `bash tests/integration/run_smokes.sh`
2. **Pre-release gate (tag push)**: above + `pytest tests/ -v` (full
   unit tests) + `synapforge-demo all` (CPU end-to-end demo).
3. **Nightly gate (cron)**: above + the rental-side `auto_eval_daemon`
   on the latest `best_*.pt`.

## What's NOT covered

By design, the suite intentionally does NOT exercise:

- Real training (live KD/SFT/RL loops on >1k tokens). Use the rental
  trainer scripts directly.
- Real LLM inference (real Qwen/GPT-2 weights). Use `chat_repl.py`
  with a real ckpt.
- Real network calls (HuggingFace download, GitHub API, mohuanfang scp,
  bilibili daemon).
- Real GPU / Triton / CUDA-graph code paths. Run `pytest tests/ -v`
  with CUDA available.
- Playwright real-browser action emission. The `WebBrowserEnv` mock is
  the test-time stand-in.

## Known flaky / triage

These are the tests most likely to surface non-test-code bugs first
and what to look at:

- `test_skill_log_v2_save_kill_recover` flakes when `os.replace` on
  Windows fails because the destination is open in another process
  (e.g., a hung daemon). Triage: `tasklist | findstr python`,
  kill stragglers.
- `test_universal_codebook_l1_to_l2_co_firing_mint` is sensitive to
  the codebook's `co_fire_window` and `co_fire_min_repeats` defaults.
  If the test was passing then breaks after a `universal_codebook.py`
  edit, check whether the defaults moved.
- `test_rfold_bench_matches_verify_rfold` runs 5 (N, R) shapes,
  including N=512. On under-resourced CI machines the wall-clock
  exceeds the 5s envelope (correctness still passes; just slow).
  Add `-k 'not rfold'` to skip if necessary.
- `test_chat_eval_gate_scores_good_and_bad` depends on the canned
  `_smoke_generator` returning category-conditional answers; if you
  edit `chat_eval_gate.py::_smoke_generator` and break the contract,
  this test catches it.

## Adding a new integration test

1. Pick ONE cross-module flow that isn't already covered.
2. Make it hermetic: use `tmp_out_dir` for filesystem state, mock heavy
   deps with `monkeypatch` or `_load_module_isolated`.
3. Single, clear assertion + a useful error message that names BOTH
   the symptom AND likely cause.
4. Add to the inventory table above.
5. If you need a new fixture, add it to `tests/integration/conftest.py`.

Anti-patterns to avoid:

- DO NOT touch the network. Mock with `monkeypatch.setattr(...)`.
- DO NOT load real LLM weights. Use the `fake_ckpt` and
  `fake_tokenizer` fixtures.
- DO NOT shell out to subprocesses unless you can guarantee the
  binary exists in PATH on Linux + macOS + Windows.
- DO NOT assert on wall-clock numbers. They are not reproducible
  across CI runners.

## File layout

```
tests/integration/
├── __init__.py
├── conftest.py             # pytest fixtures + collection hooks
├── run_smokes.sh           # one-line CI entrypoint
└── test_end_to_end.py      # the 9 integration tests
```

Total surface: ~660 LOC across the test files + ~150 LOC of docs
(this file). Per-test runtime budget: <5 seconds CPU.
