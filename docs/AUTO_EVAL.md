# Auto-Eval Pipeline

Per-checkpoint automated eval. The pipeline runs in the background while
the trainer pumps out new ckpts; for every fresh checkpoint it dumps a chat
sample, runs a fast benchmark subset, and (on `best*.pt`) optionally runs the
heavy generation-bound suite.

The point is to make "training-curve hallucination" structurally impossible:
no number lands in our paper that hasn't been re-measured by the daemon at
the corresponding ckpt step. See [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md)
for context on why we measure this aggressively.

---

## What gets produced

For a watched directory `runs/sf_100m/`, the daemon writes:

```
runs/sf_100m/
  step_001000.pt
  step_001500.pt
  best_step_001500.pt
  ...
  auto_eval/
    .dedup.json                 # ckpt-name -> sha256[:16] (don't re-eval same bytes)
    index.json                  # bucket -> {step, verdict, bench: {...}}
    001000/
      chat.json                 # 5 EN + 5 ZH greedy generations
      bench.json                # mmlu / hellaswag / lambada (fast)
    001500/
      chat.json
      bench.json
      bench_heavy.json          # humaneval / mbpp / gsm8k (best ckpts only)
    ...
```

`index.json` is the one file that downstream tools (the plot script, paper
figures, the README badge) read.

---

## Running it

The daemon is a long-lived process; spawn it once and forget about it.

```bash
# typical launch alongside training
python scripts/auto_eval_daemon.py \
    --watch /workspace/runs/synapforge_100m \
    --interval 60 \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --bench-light mmlu,hellaswag,lambada \
    --bench-heavy humaneval,mbpp,gsm8k \
    --heavy-only-best \
    --device auto \
    --n-light 50 --n-heavy 20
```

Useful one-shot smoke variant — runs a single discovery pass and exits:

```bash
python scripts/auto_eval_daemon.py --watch /tmp/sf_smoke --interval 1 --once
```

With no checkpoints in the watch dir, the daemon idles and logs:

```
[auto-eval HH:MM:SS] cycle 1: waiting (no new ckpts; pending=0)
```

That's the expected initial state. The daemon will wake up the first time
a `step_*.pt` lands.

---

## CLI flags worth knowing

| Flag | Default | What it does |
|------|---------|--------------|
| `--watch DIR` | required | The trainer's `--out` dir (where it drops `step_*.pt`). |
| `--interval N` | 60 | Seconds between discovery cycles. |
| `--tokenizer PATH` | `Qwen/Qwen2.5-0.5B` | Passed to bench harness. |
| `--bench-light A,B,C` | `mmlu,hellaswag,lambada` | Logits-only, fast (~1-3 min each at `--n-light 50`). Run on every fresh ckpt. |
| `--bench-heavy A,B,C` | `humaneval,mbpp,gsm8k` | Generation-bound, slow (~10-30 min each). Best-only by default. |
| `--n-light N` | 50 | Cap per light bench. `0` for full set. |
| `--n-heavy N` | 20 | Cap per heavy bench. `0` for full set. |
| `--heavy-only-best` | on | Heavy benches only run on `best*.pt`. Negate with `--no-heavy-only-best`. |
| `--device auto\|cuda\|cpu` | `auto` | `auto` checks `nvidia-smi`: GPU busy -> CPU; else CUDA if available. |
| `--once` | off | Single pass + exit. Good for smoke. |

---

## How GPU sharing works

Training holds the GPU. Evals running concurrently would either OOM or
slow training to a crawl. The daemon's defaults are safe by construction:

1. `--device auto` shells out to `nvidia-smi --query-compute-apps=pid` once
   per ckpt. If any other process is holding the GPU, eval falls back to CPU.
2. Workers serialize. The internal `ThreadPoolExecutor` has `max_workers=1`,
   so even if 3 ckpts land in one cycle they queue up and run one at a time.
3. The daemon never bypasses the trainer. If the trainer is dead, the GPU
   becomes free and subsequent evals naturally upgrade to CUDA.

If you specifically want eval on GPU during training (rare; only useful when
the GPU is under-utilized), pass `--device cuda` and accept the contention.

---

## Reading the index

`index.json` schema:

```json
{
  "001500": {
    "step": 1500,
    "verdict": "GRAMMAR_OK",
    "chat_n": 10,
    "bench": {
      "mmlu": 0.245,
      "hellaswag": 0.301,
      "lambada": 0.18
    },
    "ppl": 380.0
  },
  "best_1746091200": { ... },
  ...
}
```

Bucket keys are zero-padded step numbers when extractable, else
`best_<mtime>` for `best.pt` files without an embedded step.

`verdict` mirrors the heuristic in `scripts/honest_eval_hook.py`:

- `WORD_SALAD` — words but no punctuation, output is incoherent
- `TOKEN_SOUP` — same word repeated >5x in a generation (early collapse)
- `TOO_SHORT` — average <3 tokens (model learned to stop early)
- `GRAMMAR_OK` — at least half of samples have sentence punctuation
- `COHERENT` — passes both grammar and length checks

The verdict is a *fast* heuristic, not ground truth. Always read the
actual `chat.json` samples before celebrating.

---

## Plotting

`scripts/plot_eval_curves.py` reads the index and emits ASCII line plots.
No matplotlib required; investor demo runs anywhere.

```bash
python scripts/plot_eval_curves.py --index runs/sf_100m/auto_eval/index.json
```

Sample output:

```
--- ce ---
  y in [3.95, 5.94]   x in [1000, 8500]
    5.94 |
         :
         :      *
         :
         :                 *
         :                            *
   3.951 |                                                      *
         +---------------------------------------------
         1000                                       8500
  N=5  first=(1000, 5.94)  last=(8500, 3.95)
```

Optional PNG output if matplotlib is installed:

```bash
python scripts/plot_eval_curves.py \
    --index runs/sf_100m/auto_eval/index.json \
    --png runs/sf_100m/auto_eval/figs/
```

The script handles missing matplotlib gracefully (prints a stderr note,
still emits ASCII).

---

## What to look for in eval curves

The curves are **noisy** by nature — a 100M LNN+SNN evaluating 50 MMLU
questions per subject has a per-bucket sampling SE around ±1-2 percentage
points. Read trends across 5-10 ckpts, not single deltas.

### Healthy patterns

* **CE / PPL monotonically declining**, with bumps at phase transitions
  (we re-init the optimizer state when phase manager flips a flag —
  expect a 50-200-step wiggle, then resumed descent).
* **Light benches (MMLU / HellaSwag / LAMBADA) tracking ce loss.** When ce
  drops 1.0, expect ~+1-3 percentage points on each. If they don't move,
  the model is overfitting to the training distribution.
* **Heavy benches (HumanEval / MBPP / GSM8K) tracking only after Phase 3 SFT.**
  Phase 0-1 ckpts will score near zero on these. That's expected.
* **Verdict transitions: TOKEN_SOUP → WORD_SALAD → GRAMMAR_OK → COHERENT.**
  This usually happens at ce ≈ 7 → 4 → 3 → 2.

### Unhealthy patterns

* **CE plateau > 200 steps with verdict stuck at WORD_SALAD.** This is the
  signal the user calls "training-curve hallucination" — the loss stops
  moving but the *quality* didn't improve. Stop training, look at chat.json,
  consider phase transition.
* **CE down, light benches flat.** The model is memorizing surface n-grams
  but not learning anything that generalizes. Often a sign the loss mask
  is hiding too many tokens (response-only mask gone overzealous, see
  the v2.5/v2.6 history in HONEST_ASSESSMENT).
* **Light benches up, heavy benches *down*.** The model is starting to
  blow up on long generations. Usually means the inference STDP weights
  (`stdp_fast.py`) are drifting; consider an STDP reset or lowering
  the inference-time learning rate.
* **CE up suddenly.** Either the optimizer state was reset incorrectly
  on warmstart (check `.phase` was honored), or the data loader is
  serving corrupted samples. Look at the daemon's chat.json for the
  same step.

### Plateau detection

The honest_eval hook exposes `check_plateau(window=5, eps=0.5)` which
returns True if the last 5 evals' ppl spread is below `eps`. The daemon
doesn't act on this directly (that would be over-aggressive), but you
can wire it into the phase manager to auto-advance phases.

---

## Cross-references

- [BENCHMARKS.md](BENCHMARKS.md) — the six benches and their public baselines.
- [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md) — what we already know works
  and where the daemon's output should converge.
- [PHASE_TRAINING.md](PHASE_TRAINING.md) — phase signals + ppl gates.
- [BACKUP.md](BACKUP.md) — daemon ships nicely with the triple-backup
  daemon; both watch the same out-dir.
- [INDEX.md](INDEX.md) — full doc map.
