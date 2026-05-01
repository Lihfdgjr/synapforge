# RELEASE_FLOW — pretrain → SFT → chat eval → investor release in 24h

This is the "press one button, get an investor-shippable artifact" pipeline.
The trainer that's already running on the rental does NOT need to be touched;
the orchestrator watches its log, gates on `ppl ≤ 60`, then drives the rest.

## Diagram

```
                    [pretrain_to_sft.jsonl]   <- audit log every stage
                              ^
                              |
+-------------+   ppl ≤ 60   +-----------+   best.pt   +----------+
|  pretrain   |  ───────►    | SFT iter  |   ───────►  | chat eval|
|  (running)  |              | (1..3)    |             |  gate    |
| FineWeb-edu |              | alpaca    |             | 50 prompt|
| ~30k tok/s  |              | response- |             | thresh   |
| → ppl<60    |              | only loss |             | ≥ 0.6    |
+-------------+              +-----------+             +----------+
                                   ↑                         |
                                   |     pass_rate < 0.6     |
                                   └─── retry (next strat) ──┘
                                                             |
                                                             v
                                                  +--------------------+
                                                  | release dump       |
                                                  | best.pt + metrics  |
                                                  | + 5 screenshots    |
                                                  | + run_demo.sh      |
                                                  | + SHA256 manifest  |
                                                  +--------------------+
                                                             |
                                                             v
                                                  v0.1.0.tar.gz
                                          (and split parts if > 100 MB)
                                                             |
                                                             v
                                            investor: bash run_demo.sh
                                            chat REPL in ~30 sec
```

## Stage detail

### Stage 0: WATCH (orchestrator polls the running pretrain trainer)

`scripts/auto_pretrain_to_sft.py` opens `<pretrain-out>/train.log` every 60 s
and parses `VAL step N: ppl=X.YY` lines (matches the log format emitted by
`train_100m_kd.py`). When the latest VAL `ppl ≤ --ppl-target` (default 60.0),
advance to stage 1.

Honest note: pretrain stays running unmodified through this stage. We only
read the log file; no signal is sent yet.

### Stage 1: TRANSITION (graceful shutdown of pretrain)

The orchestrator reads `<pretrain-out>/<pidfile>`, sends `SIGTERM`, and waits
up to `--save-grace` (default 90 s) for the trainer to flush its final ckpt
and exit cleanly. Then it scans `<pretrain-out>` for the highest-step
`best_step_*.pt` (falling back to `step_*.pt`, then `final.pt`).

If the pretrain trainer doesn't exit within the grace window, the orchestrator
records the failure in the audit log and aborts (rc=12). Manual intervention:
the partial ckpt that was on disk before shutdown is still usable for SFT —
just rerun the orchestrator with `--smoke` skipped and `state.json` already
populated past the watch stage; it will resume from the SFT stage.

### Stage 2: SFT (response-only Alpaca SFT, up to 3 iterations)

The orchestrator spawns `train_100m_sft.py` with the located best ckpt as
`--warmstart`, the configured `--sft-cmd` (which carries the parquet path,
backend, batch size, etc.). Each iteration writes to `<sft-out>/iter<N>/`.

Iteration retry strategies (`scripts/auto_pretrain_to_sft.py::iter_strategies`):

| iter | strategy                          |
|------|-----------------------------------|
| 0    | original cmd as configured        |
| 1    | `--steps 6000 --lr 3e-5` (longer + warmer) |
| 2    | `--steps 8000 --lr 5e-6` (long + colder)   |

After each iteration the SFT trainer's exit code is captured but not used to
decide success/failure. The chat-eval gate (stage 3) is the actual decision.

### Stage 3: GATE (50-prompt heuristic chat tripwire)

`scripts/chat_eval_gate.py` runs 50 fixed prompts (25 EN + 25 ZH, 5 categories
× 10 prompts each: factual, instruction, conversational, reasoning, boundary)
through the SFT ckpt. Each generation is scored by three heuristics:

| heuristic   | weight | criterion                                                |
|-------------|--------|----------------------------------------------------------|
| h1 not_empty| 0.30   | non-empty AND no token > 50% repeat AND len ≥ 2 tokens   |
| h2 keywords | 0.30   | response contains at least 1 expected keyword            |
| h3 format   | 0.40   | per-category: factual/instr/conv/reasoning/refusal       |

Per-prompt: `score = 0.30·h1 + 0.30·h2 + 0.40·h3`, `passed if score ≥ 0.5`.

Aggregate: `pass_rate = passed/50`, GATE: `pass_rate ≥ 0.6` (default).

### What "pass rate ≥ 0.6" means (be honest)

This is **not a learned judge**. There is no Claude/GPT-4 in the loop. It is a
**heuristic tripwire** designed for one job: separate WORD_SALAD ("the the
the") from minimally COHERENT output. False-pass rate is non-zero — a
sufficiently fluent but factually wrong model can still pass. We use this as
an automated gate so the orchestrator can iterate on SFT hyperparams without
human attention.

The actual final gate is the **investor demo** itself: a human types prompts
into the REPL and judges quality. The release tarball ships those 50 generated
samples + 5 screenshots so the investor can spot-check the gate's decisions
against real outputs.

Validation we ran on the gate during development:

| input                | pass_rate | passed |
|----------------------|-----------|--------|
| canned coherent      | 1.00      | yes    |
| word salad           | 0.00      | no     |
| empty string         | 0.20      | no     |
| mid-quality answers  | 0.78      | yes    |

So the gate clearly fails for catastrophically broken models, and clearly
passes for coherent ones; the boundary is fuzzy in between.

### Stage 4: RELEASE (bundle + ship)

On gate pass, the orchestrator copies to `<release-dir>` (default
`~/.synapforge/release/v0.1.0/`):

* the SFT best.pt
* `chat_eval_gate.json` (50-prompt detailed scores)
* `release.json` (git head + pretrain cmd + sft cmd + checksums)
* `run_demo.sh` (one-line entry: load ckpt + start chat REPL)

Then `scripts/release_packager.py --from <release-dir> --out ~/.synapforge/release`:

1. tarballs the release dir as `v0.1.0.tar.gz`
2. if any ckpt is > 100 MB, splits it into ~90 MB parts under `v0.1.0.parts/`
   with a `rejoin_*.sh` to reassemble (so ckpt fits the GH Release file cap)
3. stages the `synapforge/` package + `scripts/` + `requirements.txt` in the
   tarball so a fresh-machine `pip install -r requirements.txt && bash run_demo.sh`
   gets the investor a working chat REPL in ~30 seconds
4. auto-generates 5 demo screenshots (PNG via Pillow if available, .txt
   fallback) showing best chat samples per category
5. writes `MANIFEST.json` (sha256 of every file) and `v0.1.0.sha256`
6. optionally `gh release create / upload` if `--upload-gh` is passed

## Manual intervention points

Each stage records a JSONL line to `<out>/auto_pretrain_to_sft.jsonl` and
saves cumulative state to `<out>/auto_pretrain_to_sft.state.json`. To pause
or override:

| action                          | how                                                           |
|---------------------------------|---------------------------------------------------------------|
| stop the orchestrator           | SIGINT / SIGTERM — state saves automatically before exit      |
| skip the watch stage            | edit state.json to `{"stage": "transition"}` then rerun       |
| force a specific best ckpt      | edit `pretrain_best_ckpt` in state.json                       |
| skip SFT iterations             | edit `sft_iter` in state.json (set to N to start at iter N)   |
| force-pass the chat-eval gate   | not supported — investor demo IS the final gate, see honesty  |
| change retry strategies         | edit `iter_strategies()` in `auto_pretrain_to_sft.py`         |
| inspect partial-state release   | `release_packager.py --from <partial-release-dir>`            |

## Smoke-test the entire pipeline locally (no GPU, no real ckpt)

```bash
# 1) Heuristic gate works against soup/empty/canned generators
python scripts/chat_eval_gate.py --smoke --out /tmp/gate

# 2) Orchestrator state machine cycles all 4 stages on fakes
python scripts/auto_pretrain_to_sft.py --smoke \
    --pretrain-out /tmp/p_pre --sft-out /tmp/p_sft \
    --release-dir /tmp/p_rel  --out /tmp/p_audit

# 3) Packager builds a tarball from a fake release dir
python scripts/release_packager.py --smoke
```

All three should exit 0; smoke is wired into CI in
`tests/test_release_pipeline.py` (or add it).

## Investor demo flow (the customer experience)

1. Hand the investor `v0.1.0.tar.gz` (or a download link to the GH Release)
2. They run on a fresh laptop:
   ```bash
   tar -xzf v0.1.0.tar.gz
   cd v0.1.0
   pip install -r requirements.txt
   bash run_demo.sh                         # starts chat REPL
   ```
3. ~30 sec to first prompt: the REPL loads ckpt + Qwen tokenizer + waits at `>`.
4. They type any prompt (EN or ZH); the REPL prints the model's response.
5. The 5 demo screenshots in `demo_screenshots/` show what's expected.
6. The 50-prompt gate report at `chat_eval_gate.json` shows the heuristic
   pass rate the build was approved on.

## Production launch (rental, ~24h budget)

Assuming pretrain has been running for ~6h on FineWeb-edu and just crossed
ppl ≤ 60 on a fresh val set:

```bash
# already running on rental: train_100m_kd.py (pretrain). Has trainer.pid.
# Now launch the orchestrator alongside (it just watches the log):

nohup python scripts/auto_pretrain_to_sft.py \
    --pretrain-out /workspace/runs/v24h_qwen \
    --pretrain-pidfile /workspace/runs/v24h_qwen/trainer.pid \
    --pretrain-cmd-record "$(cat /workspace/runs/v24h_qwen/launch.cmd)" \
    --sft-cmd "python train_100m_sft.py --backend triton_block --batch-size 16 \
               --steps 4000 --lr 1e-5 \
               --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
               --sft-parquet /workspace/data/sft/alpaca_combined.parquet \
               --grad-checkpoint" \
    --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
    --sft-out /workspace/runs/v24h_sft \
    --release-dir ~/.synapforge/release/v0.1.0 \
    --ppl-target 60 \
    --max-sft-iters 3 \
    --out /workspace/runs/auto_p2sft \
    > /workspace/runs/auto_p2sft.log 2>&1 &
```

When the gate passes:

```bash
python scripts/release_packager.py \
    --from ~/.synapforge/release/v0.1.0 \
    --out ~/.synapforge/release \
    --upload-gh --gh-tag v0.1.0 --gh-repo Lihfdgjr/synapforge
```

Total wall-clock from `ppl<60` crossing to investor-shippable tarball:

| stage           | time      |
|-----------------|-----------|
| transition      | ~ 1-2 min |
| SFT iter 0      | ~ 4 hr    |
| chat eval gate  | ~ 5-10 min|
| release packing | ~ 1-2 min |
| **total**       | **~ 4-5 hr** |

If iter 0 fails the gate, expect +4-6 hr per retry (max +12-18 hr if all 3
iters needed). Budget for 24h end-to-end conservatively.
