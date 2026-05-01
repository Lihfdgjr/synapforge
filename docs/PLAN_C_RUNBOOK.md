# Plan C Runbook — LoRA Qwen 0.5B insurance demo

**Goal**: 60%+ chat eval pass rate on rental in <= 1 hour. Plan C is the
investor demo's safety net — if the native 100M LNN+SNN training fails to
hit phase 3 (`docs/MASTER_PLAN.md` §3), this is the artifact we ship.

**Status**: code done, smoke verified locally, real run pending on rental.
Tracking item: `docs/MASTER_PLAN.md` §6 P5.

> Honest framing: Plan C is the **v0 demo frontend**, not the architecture
> claim. The architecture claim is the SynapForge 100M LNN+SNN. Plan C
> only ships if the native model isn't chat-fluent in time.

---

## Prerequisites

- Rental SSH up: `ssh -p 41614 root@117.74.66.77` (creds in
  `reference_rental_a100x2_ssh.md`).
- `/workspace/teachers/qwen2.5-0.5b/` exists (Qwen2.5-0.5B-Instruct base
  model + tokenizer, downloaded once during rental setup).
- `data/sft/alpaca_combined.parquet` — produced by
  `scripts/prep_alpaca_qwen.py` (alpaca-en + alpaca-zh, ~100K rows with
  `input_ids` and `loss_mask` columns; loss-mask is response-only).
- `peft` installed (`pip install peft==0.11.1`). If unavailable, the
  trainer falls back to the inline LoRA path automatically — same math,
  no manual switch.

---

## Step 1: print-only sanity (5 min, no GPU needed)

```bash
python scripts/train_qwen_lora.py --smoke --print-only
```

**What it checks**: argparse resolves, all flag aliases recognised,
final.pt path looks sane, framework choice (`peft` vs `inline`) matches
what `pip` reports. Zero disk/network/GPU I/O.

Pass criteria: prints a "resolved plan" block ending with
`next_step: python scripts/chat_eval_gate.py --ckpt .../final.pt --threshold 0.6`.

---

## Step 2: smoke real (10 min)

```bash
python scripts/train_qwen_lora.py --smoke
```

**What it checks**: optimizer step actually lands, inline LoRA fallback
works, `final.pt` writes to `~/.synapforge/release/qwen_lora_v0/final.pt`,
config.json records `framework`/`vocab`/`n_trainable`. 5 mock steps on a
2-layer fake-Qwen with 64 random sequences (vocab=256). ZERO net access.

Pass criteria: log ends with `[done] {'step': 5, ...}`. `final.pt`
loads back via `torch.load` and contains `{model, config, framework,
lora, vocab}`.

---

## Step 3: 200-step real LoRA train (30 min on A800 80GB)

```bash
python scripts/train_qwen_lora.py \
    --steps 200 \
    --lora-r 16 --lora-alpha 32 \
    --output-dir /workspace/runs/plan_c_lora \
    --base-path /workspace/teachers/qwen2.5-0.5b \
    --data data/sft/alpaca_combined.parquet \
    --bs 8 --lr 2e-4 --max-seq 1024 \
    --warmup 50 --log-every 10 --sample-every 50 --save-every 100
```

**What it checks**: real Qwen base + peft LoRA on q/k/v/o_proj attention
projections, 200 SFT steps on alpaca_combined. Loss should drop from ~3.0
to ~1.5 (Qwen Instruct is already chat-tuned; LoRA only nudges it).

Pass criteria: train.log shows monotonic loss decrease, samples after
step 100 are coherent EN+ZH, `final.pt` written.

---

## Step 4: chat eval gate (5 min)

```bash
python scripts/chat_eval_gate.py \
    --ckpt /workspace/runs/plan_c_lora/final.pt \
    --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
    --threshold 0.6 \
    --out /workspace/runs/plan_c_lora/chat_eval
```

`chat_eval_gate.py` auto-detects Plan C ckpts via the `framework` key
in `final.pt` and routes to `qwen_lora_chat_repl.load_qwen_lora` (Qwen
base + LoRA). 50 fixed prompts (25 EN + 25 ZH), 5 categories.

Pass criteria: `pass_rate >= 0.6` AND `passed: true` in
`chat_eval_gate.json`.

---

## Decision tree

- **Step 4 passes (`pass_rate >= 0.6`)** → Plan C is ready as fallback
  demo. Keep `final.pt` plus `tokenizer/` and `merged.pt` (if inline)
  in the rental's backup pipeline (`scripts/triple_backup_daemon.py`
  picks it up automatically). Done.

- **Step 4 fails** — DO NOT abandon. Identify which knob is wrong:
  1. **Loss never drops in Step 3**: LoRA target modules may be wrong.
     Check `print_trainable_parameters()` output in train.log; expect
     ~0.3% trainable for Qwen 0.5B (≈1.5M / 494M). If trainable < 0.1%
     or > 5%, target_modules string is mismatched against the model's
     actual layer names.
  2. **Loss drops but eval fails on EN only / ZH only**: data imbalance
     in `alpaca_combined.parquet`. Re-run `scripts/prep_alpaca_qwen.py`
     with explicit ratios.
  3. **Eval fails on `boundary` category** (refusals): Qwen-Instruct
     already refuses — this should be near 1.0. If <0.5, the LoRA is
     overwriting the safety alignment. Drop LR to 1e-4 and re-run.
  4. **Eval fails on `reasoning` category**: 200 steps is short. Bump
     to `--steps 500` (60-70 min) and re-eval.

Fix and re-run from Step 3. Do not skip ahead.

---

## What "ready" means

Plan C is "ready as fallback demo" when:

1. `final.pt` exists at `/workspace/runs/plan_c_lora/final.pt`.
2. `chat_eval_gate.json` shows `passed: true`.
3. Triple-backup daemon has copied `final.pt` to mohuanfang +
   GitHub release + HF Hub (`scripts/triple_backup_daemon.py` log
   shows the sha256 in its dedup cache).
4. `scripts/qwen_lora_chat_repl.py --adapter /workspace/runs/plan_c_lora`
   answers 5 EN + 5 ZH canned prompts coherently in <30 s.

When all four hold, MASTER_PLAN.md §6 P5 can be marked RESOLVED.
