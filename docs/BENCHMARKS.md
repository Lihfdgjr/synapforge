<!-- DOC_STAMP: STALE since 2026-05-01; check refs to synapforge/bench/NEWBENCH.py -->
# Paper-Grade Benchmarks

This harness lives at `synapforge/bench/` and ships six static benchmarks, each
callable as a CLI module and as a Python function. They share one `--ckpt`
(SynapForge100M) and one tokenizer (`Qwen/Qwen2.5-0.5B` by default).

## Honest disclaimer

A 100M-parameter LNN+SNN model trained on ~1B tokens **will lose** every static
benchmark on this list to a 1.1B-parameter Transformer trained on ~3T tokens.
We ship this harness so the regression is **measured and reported**, not
discovered later by a confused reader of our paper.

The point of this repo is the architecture (recurrent depth, R-fold, NeuroMCP,
PLIF biological dynamics), not the static-benchmark headline. We expect to see:

- **Static benchmarks**: -10 to -30 percentage points vs TinyLlama-class.
- **Long-context recall (NIAH)**: parity or better at 100K+ tokens.
- **Inference cost (R-fold)**: 100×+ less compute per recurrent step.

When the benchmark deltas are negative, that is data, not failure.

---

## The six benches

| Bench       | Metric    | Dataset                       | Examples | Default `max_new` |
|-------------|-----------|-------------------------------|---------:|------------------:|
| HumanEval   | pass@1    | `openai_humaneval`            |     164  | 384               |
| MBPP        | pass@1    | `mbpp` (test split)           |     974  | 384               |
| MMLU        | acc       | `cais/mmlu` (57 subjects)     |   14042  | 1 (logits only)   |
| GSM8K       | acc       | `gsm8k` (main, test)          |    1319  | 256               |
| HellaSwag   | acc       | `hellaswag` (validation)      |   10042  | logits only       |
| LAMBADA     | acc       | `lambada` (test)              |    5153  | 8                 |

Total runtime on a single H800 with bs=1 generation: ~3-6 hours for the full
set on a 100M model. Smoke runs (`--n 50` per bench) take ~5 minutes.

---

## Public baselines we compare against

These numbers come from each model's published card or the lm-eval-harness
leaderboard. Keep them in mind as the floor we need to beat (or, where we lose,
explain in the paper).

| Model           | HumanEval | MBPP   | MMLU   | GSM8K  | HellaSwag | LAMBADA |
|-----------------|----------:|-------:|-------:|-------:|----------:|--------:|
| TinyLlama-1.1B  |     4.9 % |  8.5 % | 25.3 % |  2.4 % |    59.5 % |  58.0 % |
| SmolLM2-135M    |     1.2 % |  2.0 % | 29.3 % |  1.3 % |    41.8 % |  33.8 % |
| Qwen2.5-0.5B    |    30.5 % | 39.5 % | 47.5 % | 41.8 % |    52.3 % |  49.8 % |
| Mythos (target) |    40.0 % | 50.0 % | 55.0 % | 60.0 % |    65.0 % |  62.0 % |

The SynapForge100M targets in our paper (after Phase-3 SFT and Phase-4 RL) are
TinyLlama-class on HellaSwag/LAMBADA and best-effort on the code/math benches.

---

## Running individual benches

Each bench is standalone:

```bash
# HumanEval, full 164 problems
python -m synapforge.bench.humaneval \
    --ckpt runs/sf_100m/best.pt \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-new 384 \
    --out runs/bench/humaneval.json

# MBPP, smoke (50 problems)
python -m synapforge.bench.mbpp \
    --ckpt runs/sf_100m/best.pt --n 50 --out runs/bench/mbpp_smoke.json

# MMLU, all 57 subjects
python -m synapforge.bench.mmlu \
    --ckpt runs/sf_100m/best.pt --out runs/bench/mmlu.json

# MMLU, just the easy / fast subjects
python -m synapforge.bench.mmlu \
    --ckpt runs/sf_100m/best.pt \
    --subjects elementary_mathematics,high_school_geography,marketing \
    --out runs/bench/mmlu_smoke.json

# GSM8K, smoke
python -m synapforge.bench.gsm8k --ckpt runs/sf_100m/best.pt --n 100

# HellaSwag, smoke (10042 takes ~30 min on a small model)
python -m synapforge.bench.hellaswag --ckpt runs/sf_100m/best.pt --n 200

# LAMBADA, smoke
python -m synapforge.bench.lambada --ckpt runs/sf_100m/best.pt --n 200
```

Each module also has a Python entry point — `run_bench(...)` — that accepts a
preloaded `model=...` and `tok=...` so the orchestrator does not pay a load
penalty per bench.

---

## Running the full harness

```bash
python scripts/run_all_bench.py \
    --ckpt runs/sf_100m/best.pt \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --benches humaneval,mbpp,mmlu,gsm8k,hellaswag,lambada \
    --out runs/bench/full.json
```

Useful smoke flag: `--n 50` caps each bench at 50 examples (per subject for
MMLU, total for everything else). Wall-time drops from hours to minutes.

The orchestrator prints a comparison table at the end:

```
bench       ours    tinyllama-1.1b  smollm2-135m  qwen2.5-0.5b  mythos-target
----------  ------  --------------  ------------  ------------  -------------
humaneval   0.018   0.049           0.012         0.305         0.400
mmlu        0.241   0.253           0.293         0.475         0.550
gsm8k       0.005   0.024           0.013         0.418         0.600
...
delta vs tinyllama-1.1b:
  humaneval   -0.031
  mmlu        -0.012
  gsm8k       -0.019
```

---

## Sandbox notes (HumanEval / MBPP)

We do **not** use Docker, gVisor, or any heavyweight sandbox. The runner is:

```python
subprocess.run([sys.executable, "-c", code], timeout=5, capture_output=True)
```

This is enough for our threat model: we only ever feed our own model's code
generations through it, and a 100M-param LM that scores <5 % on HumanEval is
not going to compose a syscall-level escape. If you re-purpose this for a
larger model or untrusted code, swap the runner out for a jail (e.g., Bubblewrap
or Docker with `--read-only` and `--cap-drop=ALL`).

The 5-second timeout is enforced two ways:
1. `subprocess.timeout` kills the whole interpreter.
2. `signal.alarm(5)` inside the harness catches loop-bound code that hides from
   the parent.

Outputs are scored as **pass** iff the subprocess exits 0 **and** prints the
sentinel `__OK__`. Any other state (timeout, exception, assertion failure,
import error, segfault) is a fail.

---

## Adding a new bench

1. Create `synapforge/bench/NEWBENCH.py` exporting `run_bench(...)`.
2. Add an entry to `BENCH_REGISTRY` in `synapforge/bench/__init__.py`.
3. (Optional) Add a baseline column to `BASELINES` in `scripts/run_all_bench.py`.

Required `run_bench(...)` signature (kwargs-only, all optional):

```python
def run_bench(
    ckpt: Optional[str] = None,        # path to ckpt; load if model=None
    tokenizer: Optional[str] = None,   # tokenizer name; load if tok=None
    model: Any = None,                 # preloaded model (orchestrator hot path)
    tok: Any = None,                   # preloaded tokenizer
    n: Optional[int] = None,           # smoke cap
    out: Optional[str] = None,         # JSON output path
    **_kw,                             # accept and ignore unknown kwargs
) -> Dict[str, Any]:
    ...
```

Return at minimum `{"name": ..., "<metric>": float, "n_total": int, "wall_s": float}`.

---

## What this harness does NOT cover

- **WikiText-103 PPL** — already lives in the ad-hoc training eval; not part of
  this orchestrator (it is a perplexity number, not a benchmark accuracy).
- **LiveCodeBench / Aider / SWE-bench** — heavy infra (Docker, real PR replay,
  cost). Planned as a Phase-2 addition. See `feedback_all_benchmarks_parity` in
  memory for the master list.
- **Long-context (NIAH)** — already in `synapforge/eval/niah.py`. That bench is
  about *our actual edge*, not parity.
- **Multimodal benches** — once the 9-modal byte-patch trainer ships.

When those land, they will follow the same `run_bench(...)` contract so the
orchestrator can dispatch them too.
