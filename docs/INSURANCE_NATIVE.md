# Native-Synap Insurance Paths — Replacement Runbook

**Updated**: 2026-05-01 (initial revision; replaces deleted Plan C / Qwen-LoRA route).

## Why this doc exists

`docs/ANTI_LORA.md` (2026-05-01) deleted the Plan C LoRA-on-Qwen-0.5B
fallback for two strategic reasons:

1. **Architecture-claim violation** — a transformer-base + LoRA-adapter
   "insurance demo" makes the Synap-1 LNN+SNN claim non-falsifiable in
   practice (a reviewer can't distinguish "Synap-1 chats" from "Qwen+LoRA
   chats").
2. **Paper unsubmittability** — NeurIPS / ICLR / ICML reviewers reject any
   paper whose claimed architecture has a transformer fallback in the
   demo.

But the underlying risk Plan C was hedging didn't go away: **what if
Synap-1 (100M LNN+SNN) hasn't reached chat-grade by demo day?** This doc
enumerates the **three native-Synap-only fallbacks** that satisfy
ANTI_LORA, gives concrete decision criteria for each, and points at the
code that implements them.

All three are 100% LNN+SNN. Use them in priority order: A first (if time
budget allows), B second (always works as last resort), C third (when
chat is broken AND we'd rather not show recorded outputs).

---

## Option A — Synap-Mini (~30M LNN+SNN, fast train)

The "smaller model trained faster" path. Same architecture as Synap-1,
shrunken so we can squeeze a chat-grade ckpt into a single rental window.

### Architecture

| Knob          | Synap-1 (flagship)  | Synap-Mini (insurance) | Why               |
|---------------|---------------------|------------------------|-------------------|
| `vocab`       | 151936              | 151936                 | same Qwen vocab — must match teacher KD path |
| `d`           | 512                 | 256                    | 1/2 width — quadratic on FFN params |
| `n_layers`    | 10                  | 6                      | 3/5 depth — linear on most layers |
| `loop_depth`  | 1                   | 1                      | LoopLM/Ouro recursion at 1 keeps step time fast |
| `ffn_ratio`   | 8.0                 | 4.0                    | smaller MLP — biggest single param saver |
| `sparsity`    | 0.95                | 0.95                   | STDP plasticity-aware sparsity unchanged |

Param breakdown at vocab=151936 / d=256 / n_layers=6 / ffn_ratio=4:

```
tok_embed + lm_head (tied):       151936 *  256 ~  39M  (~80% of total!)
per-layer body (CfC + PLIF + FFN at d=256 ffn=1024):
                                  ~  1M each * 6 ~  6M
total                                            ~ 45-50M
```

So "30M" is aspirational — the Qwen 151k vocab dominates everything.
With `tie_lm_head=True` (default in Synap-1), the embedding-table mass
is paid once. To shave further would need a smaller vocab (BPE retrain,
which is a multi-day side project), so we accept ~45M as the practical
floor for "Synap-Mini with full Qwen tokenizer."

### Training recipe

Same KD recipe as Synap-1 (frozen Qwen2.5-0.5B teacher), but smaller
arch + bigger batch:

```
backend     triton_block            (mandatory; gpu_dense wastes 90% of A800)
batch       128                     (up from 80 — smaller body fits more samples)
lr          2e-4                    (gentler than Synap-1's 1e-4 won't apply
                                     here; smaller models tolerate higher LR)
steps       30_000
KD          --teacher Qwen/Qwen2.5-0.5B --kd-every 4 --kd-weight 0.7
budget      6-12h on A800-80GB at ~30k tok/s effective
save-every  1000
eval-every  500
```

Memory: `(B*T, V, fp32)` = 128 × 256 × 151936 × 4 ~= 19 GB even with sparse
z-loss; chunked KD splits it across forwards so peak < 50 GB. The smaller
body frees enough for bs=128 to fit comfortably (Synap-1 capped at bs=80
because of the bigger encoder + teacher overhead).

### Acceptance criterion

**val ppl ≤ 80 on alpaca-zh-eval** = "weakly chat-coherent" (it can
complete short EN+ZH prompts, basic Q/A works, but it's not GPT-grade
fluency). This is the honest bar for 30-50M LNN+SNN — if Synap-Mini hits
ppl 80 we have a viable demo; if it stalls at ppl 200+ we fall through
to Option B.

### Decision criterion to actually launch

Synap-Mini is a *parallel-track* insurance, not a Synap-1 replacement.
Launch if **all three** hold:

1. Synap-1 (100M) hasn't tripped phase 1 (val ppl ≤ 250) within **8h**
   of training (the conservative phase-0 budget — see
   `docs/PHASE_TRAINING.md`).
2. Rental still has ≥ **6 GPU-h** on the clock (Synap-Mini needs the
   full window).
3. We have **at least 3 days** before the investor demo (Synap-Mini's
   own training takes 6-12h, then ckpt-load + chat eval takes ~1h).

If Synap-1 tripped phase 1 already, do NOT launch Synap-Mini — the
flagship is on track and we'd waste GPU-h. If rental has < 6 GPU-h,
fall through to Option B (recorded replay always ships).

### Concrete launch (rental)

```bash
# On 117.74.66.77:41614 / fresh shell, after auth:
setsid bash scripts/launch_synap_mini.sh \
  > /workspace/runs/synap_mini/train.log 2>&1 &
disown
# Returns to caller in <1s (per feedback_mcp_remote_ssh_quirks.md;
# nohup hangs the MCP shell, setsid+disown is the proven escape).
```

Watch progress:

```bash
ssh -p 41614 root@117.74.66.77 'tail -50 /workspace/runs/synap_mini/train.log'
```

The script wraps `train_100m_kd.py` and sets every flag explicitly. It's
~80 LOC of pure bash with comments — see `scripts/launch_synap_mini.sh`.

### Required trainer flags

`train_100m_kd.py` already accepts everything Synap-Mini needs (verified
2026-05-01 against `_parse_args`):

- `--vocab` (line 179) — `MODEL_VOCAB` default = 151936
- `--d` (line 181) — `MODEL_D` default = 512
- `--n-layers` (line 183) — `MODEL_N_LAYERS` default = 10
- `--loop-depth` (line 185) — `MODEL_LOOP_DEPTH` default = 1
- `--ffn-ratio` (line 187) — `MODEL_FFN_RATIO` default = 8.0
- `--batch-size`, `--lr`, `--steps`, `--out`, `--no-warmstart` — all present

`_build_config_dict()` at trainer line 113 already pulls `vocab`/`d`/
`n_layers`/`loop_depth`/`ffn_ratio` via `getattr(args, ...)`, so the
config dict persisted into every Synap-Mini ckpt is honest about its
shape — `chat_demo.py::_try_load_live` will reconstruct correctly.

**No follow-up trainer patch needed**.

---

## Option B — Recorded replay (`chat_recorded.json`)

The "always works as last resort" path. We already have a healthy v4.1
ckpt's chat output baked into the demo package; we just need to be honest
about it.

### Source data

`synapforge/demo/chat_recorded.json` — 5 EN + 5 ZH prompt/response pairs
captured during v4.1 training (val ppl 44, 100M LNN+SNN). Same
architecture as Synap-1, just an older ckpt vintage.

### Mechanism

`chat_demo.py::run_demo` already falls back to `_load_recorded()` when
`_try_load_live(ckpt, tokenizer_path)` returns None (line 47:
`_default_recorded()`, line 207: the recorded branch). So if the live
ckpt is missing or unloadable, the demo path automatically replays.

### Honest disclosure (NEW 2026-05-01)

`synapforge/demo/disclose.py::disclose_replay()` returns the canonical
disclosure banner:

```
*** Honest disclosure: showing recorded v4.1 outputs from
April 28, 2026 ckpt (val ppl 44, 100M LNN+SNN). The v5 / Run 3c trainer
is still converging at the time of this demo. Same architecture,
different ckpt vintage. ***
```

This prints at the **top** of the chat block when the recorded path is
taken. Wired into `chat_demo.run_demo` recorded-mode branch — see
`chat_demo.py:208-218` for the integration point.

The disclosure must be visible to the audience BEFORE they see the
recorded output. Showing recorded outputs as if they were live is the
worst possible outcome — investors lose trust the moment they ask for
a different prompt and get nothing.

### Acceptance

**Always works.** The recorded JSON ships with the repo
(`MANIFEST.in` includes `synapforge/demo/chat_recorded.json`), so any
fresh clone can run `synapforge-demo chat` and get the disclosure +
replay path.

### When to use

- Synap-1 ckpt unloadable (architecture mismatch, corrupted file, etc.).
- Synap-Mini didn't train in time AND we still want a live-ish chat block.
- Demo-day network outage prevents loading any rental ckpt.

### When NOT to use

- If the **live** Synap-1 ckpt loads and runs, ALWAYS prefer live (the
  recorded path is fallback-only).
- If we want to run *new* prompts in front of investors. The recorded
  set is fixed at 10 prompts; novel prompts during a live demo would
  expose the replay.

---

## Option C — Mechanism-pivot demo (skip chat entirely)

The "chat is the LAST item, not the first" path. The Synap pitch already
leads with mechanism-level differentiators (NeuroMCP, R-fold, STDP);
chat is just the language-model demo on top. If chat is broken on demo
day, ship the mechanism demos by themselves.

### Reframing

`docs/INVESTOR.md` already lists 5 differentiated claims:

1. NeuroMCP synaptic growth replaces tool calling.
2. R-fold algebraic CfC — k reasoning steps per matrix solve.
3. Inference-time STDP — single-LOC unlock.
4. 100M LNN+SNN with Qwen 151k vocab.
5. Triple-path full-volume backup.

**Claims 1-3 + 5 are mechanism-level** and run on CPU in seconds. They
require no chat ckpt at all. Claim 4 needs a working ckpt — that's
where Options A/B kick in.

### Code change (NEW 2026-05-01)

`synapforge/demo/cli.py::cmd_all` reorders the `synapforge-demo all`
output so mechanism demos go FIRST, chat goes LAST. Plus a new
`--mechanism-only` flag that **skips the chat block entirely**:

```bash
synapforge-demo all --mechanism-only
# pitch + button (NeuroMCP) + bench (R-fold) + stdp + DONE.
# No chat block, no recorded transcript -- pure mechanism pitch.
```

The flag is on `_add_all_args` so both `all` and `json` accept it. See
`cli.py:90-115` for the reorder logic and the `--mechanism-only` early
return.

### When to use

- Live Synap-1 ckpt unavailable AND we don't want to show recorded
  outputs (e.g. an investor specifically asked for a *novel* prompt
  and we'd rather decline the chat block than fake it).
- Tight 5-minute pitch where the chat block adds time but doesn't add
  signal — the headline differentiators are already covered by 1+2+3.
- Paper supplement video — claims 1-3 are reproducible, chat is
  vintage-dependent.

### Acceptance

Always works. NeuroMCP / R-fold / STDP demos run on CPU in 1-5s each
(per `synapforge-demo all` runtime ~10-15s end-to-end without chat).

---

## Decision tree (demo-day flowchart)

```
Synap-1 ckpt loadable?
  YES -> live chat path (highest signal, no fallback needed)
  NO  -> Synap-Mini ckpt loadable?
           YES -> live chat path with Synap-Mini ckpt
                   (lower bar: ppl 80, but still LNN+SNN live)
           NO  -> investor wants live chat?
                    YES -> Option B (recorded replay + disclosure)
                    NO  -> Option C (--mechanism-only)
```

In practice, Options B and C are independent — you can do `--mechanism-only`
and skip both chat paths if the demo day is short on time, even with a
working live ckpt.

---

## Trainer flag verification (no follow-up patch needed)

Verified against `train_100m_kd.py` head as of 2026-05-01:

| Flag             | Line  | Default                  | Used by Synap-Mini? |
|------------------|-------|--------------------------|---------------------|
| `--vocab`        | 179   | `MODEL_VOCAB` (151936)   | yes (kept at default) |
| `--d`            | 181   | `MODEL_D` (512)          | yes -> 256            |
| `--n-layers`     | 183   | `MODEL_N_LAYERS` (10)    | yes -> 6              |
| `--loop-depth`   | 185   | `MODEL_LOOP_DEPTH` (1)   | yes (kept at default) |
| `--ffn-ratio`    | 187   | `MODEL_FFN_RATIO` (8.0)  | yes -> 4              |
| `--batch-size`   | 156   | 80                       | yes -> 128            |
| `--lr`           | 191   | 1e-4                     | yes -> 2e-4           |
| `--steps`        | 161   | 1000                     | yes -> 30000          |
| `--out`          | 150   | `OUT_DIR_DEFAULT`        | yes -> `synap_mini`   |
| `--no-warmstart` | 154   | n/a (action=store_const) | yes (cold start)      |

`_build_config_dict()` at line 113 already pulls `vocab` / `d` /
`n_layers` / `loop_depth` / `ffn_ratio` via `getattr(args, ..., FALLBACK)`,
so even older callers (P9-smoke tests etc.) keep working. Synap-Mini
ckpts persist their **own** shape into the config dict;
`chat_demo.py::_try_load_live` reads it and rebuilds the right shape
without falling back to the 100M defaults.

---

## Files added / changed for this insurance plan

| Path                                         | Kind       | Purpose                                |
|----------------------------------------------|------------|----------------------------------------|
| `docs/INSURANCE_NATIVE.md`                   | NEW doc    | this runbook                           |
| `scripts/launch_synap_mini.sh`               | NEW script | Option A launcher (~80 LOC bash)       |
| `synapforge/demo/disclose.py`                | NEW module | Option B disclosure banner             |
| `synapforge/demo/chat_demo.py` (edit)        | EDIT       | wire `disclose_replay()` into recorded path |
| `synapforge/demo/cli.py` (edit)              | EDIT       | reorder + `--mechanism-only` flag (Option C) |
| `docs/MASTER_PLAN.md` (edit, P5 + P23)       | EDIT       | point at this doc, mark P23 RESOLVED   |

---

## Cross-references

- `docs/ANTI_LORA.md` — strategic argument (why Plan C had to die)
- `docs/PHASE_TRAINING.md` — phase-0 budget + ppl gates (when Option A is "needed")
- `docs/INVESTOR.md` — pitch reduction (the 5 claims; mechanism vs chat split)
- `docs/MASTER_PLAN.md` §6 P5 / P23 — open-problems status
- `feedback_mcp_remote_ssh_quirks.md` — `setsid + disown` (NOT `nohup`) on rental
- `feedback_phased_training_2026q2.md` — ppl-gated activation (don't enable phase 1 too early)

— 2026-05-01
