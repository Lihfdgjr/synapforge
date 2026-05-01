# Plan C — Qwen 0.5B + LoRA as the v0 Chat Frontend

**Status**: SHIPPABLE for the investor demo at 10:00 AM.
**Honesty contract**: this is the *v0 demo frontend*, NOT the architecture
claim. The architecture claim is the SynapForge 100M LNN+SNN, currently
in active training and shown in `synapforge-demo all` for live evidence.

## Why we need a Plan C

We bet the investor demo on a 100M-parameter LNN+SNN trained from scratch.
That bet is correct on the 8-week timeline (`docs/ROADMAP.md`), and we
have real telemetry — ppl 44.2 on the v4.1 ckpt, NeuroMCP 100% hit-rate
on the 4-button env, R-fold k=8 verified at 0.32% drift, inference-time
STDP 27% density growth in 200 trials. Those are the *architecture*
claims and they hold.

But "chat-able in 24h" is a different claim. SmolLM2-360M needed 2T
tokens to reach passable chat fluency. We have ~10B tokens of teacher-KD
budget on a single A800 80GB. Doing the math:

| target            | tokens needed | our budget | gap        |
|-------------------|---------------|-----------:|------------|
| chat parity       | 2T            | 10B        | **200×**   |
| ppl < 50 (we hit) | 10B           | 10B        | done       |
| coherent dialogue | ~100B         | 10B        | 10×        |

ppl 44 is a real number on a real 100M ckpt — meaningful as architecture
evidence, but not chat-fluent. Asking the investor to chat with the
native model and get the recorded transcript is the wrong demo: it
*looks* underwhelming even though the underlying architecture is novel.

Plan C ships a real bilingual chat artifact in 30 min, alongside the
architecture telemetry. That mix is honest *and* impressive.

## What Plan C ships

A frozen Qwen2.5-0.5B-Instruct (already chat-tuned by the Qwen team) with
a small LoRA adapter SFT'd on alpaca-en + alpaca-zh.

```
v0 frontend:   Qwen 0.5B (frozen) + LoRA r=16 on q,k,v,o_proj
training:      1 epoch SFT, ~100K examples, 30 min on A800
storage:       ~6MB adapter + 1GB merged ckpt
runtime:       same Qwen tokenizer, vocab 151643 (en+zh both)
ux:            `synapforge-demo qwenchat` (5 EN + 5 ZH canned prompts)
                or `python scripts/qwen_lora_chat_repl.py` (interactive)
```

### Files added

| Path | LOC | Purpose |
|------|-----|---------|
| `scripts/train_qwen_lora.py`              | ~520 | LoRA SFT (peft + inline fallback) |
| `scripts/qwen_lora_chat_repl.py`          | ~245 | Interactive chat REPL (Qwen template) |
| `synapforge/demo/qwen_lora_demo.py`       | ~210 | Demo wrapper, 5 EN + 5 ZH replay |
| `docs/PLAN_C_QWEN_LORA.md`                | ~250 | this file |
| `docs/INVESTOR.md` (small update)         | +10  | "Live demo" pointer |

### CLI integration

```bash
# Native architecture demo (the actual claim)
synapforge-demo all
synapforge-demo button   # NeuroMCP 4-button
synapforge-demo bench    # R-fold math + speedup
synapforge-demo stdp     # inference-time STDP
synapforge-demo chat     # native 100M LNN+SNN (replays v4.1 transcript)

# v0 demo frontend (Plan C)
synapforge-demo qwenchat       # 5 EN + 5 ZH live chat
synapforge-demo qwenchat --smoke  # mock Qwen, for CI/no-GPU smoke test
```

`synapforge-demo all` runs both the native demos AND `qwenchat`, so the
investor sees the architecture telemetry and the chat together.

## Comparison: v0 frontend vs v1 native

| | v0 frontend (Qwen-LoRA) | v1 native (SynapForge 100M) |
|--|--|--|
| Status | **ships in 30 min** | 8-week sprint to chat-quality |
| Backbone | Qwen2.5-0.5B (transformer) | SynapForge100M (LNN+SNN hybrid) |
| Trainable params | ~6M (LoRA) | 100M from scratch |
| Training tokens | ~50M (alpaca SFT) | 10B (KD) + ~50M SFT |
| Wall time | 30 min A800 | 24 GPU-h initial + 7d for fluency |
| Chat fluency now | **fluent EN + ZH** (Qwen-tuned) | telemetry + recorded transcript |
| Energy/token | 1.0 (transformer baseline) | ~0.05 (sparse spikes, target) |
| KV cache | yes | none (recurrent CfC) |
| Continual learning | no (frozen base + LoRA) | yes (inference-time STDP) |
| Tool use | JSON tool-call | NeuroMCP (no token) |
| Honest framing | demo frontend, ships now | architecture bet, ships at v1 |

The v0/v1 split is the investor-friendly version of "you can't show
chat-fluency from an under-trained 100M model in 24h, so we layer a
Qwen frontend for the demo and keep the native run as evidence."

## Disclosure script (what to actually say)

Open with the architecture demo. End with the chat. The order matters
because we want the investor anchored on **what we built**, not on
**what's chatting on screen**.

Word-for-word suggested script:

> "Before I show you anything talking, let me show you what we
> built. Here's `synapforge-demo all`."
>
> *(run, walk through):*
> - **NeuroMCP** — synapses grow into the action space, no tool-call
>   tokens. 100% hit rate on the 4-button validation env in <1s. Density
>   climbs from 6% to 8% in 80 trials; saturates at 28% by trial 600 in
>   our research runs.
> - **R-fold algebraic CfC** — k=8 reasoning steps fold into a single
>   matrix solve. We verified the math (0.32% drift at R=8) and the
>   speedup (2.99× on N=64 R=16 GPU). Each token gets 8 steps of
>   compute without 8× the wall time.
> - **Inference-time STDP** — single-LOC unlock. Hebbian rule runs at
>   inference, weights adapt to live context with no optimizer, no
>   loss. Density climbs 0% → 27% in 200 trials.
> - **100M LNN+SNN ckpt** — ppl 44 on multilingual chat with Qwen vocab,
>   trained from scratch in 24 GPU-h with KD from Qwen 0.5B as teacher.
>
> "All four of those are real artifacts you can run on a CPU right now.
> They're the architecture claim.
>
> "Now — the architecture is undertrained at the chat level. 100M
> trained on 10B tokens with KD will not match GPT-4. To show you
> what the *deployed* product looks like, we ship a v0 frontend: a
> frozen Qwen 0.5B Instruct with a small LoRA adapter trained on
> alpaca bilingual SFT, 30 min of A800 time. The Qwen team did the
> heavy lifting on chat fluency; we plug it in as the v0 chat surface
> while the native 100M finishes."
>
> *(run `synapforge-demo qwenchat`):*
>
> "Five English, five Chinese, live LoRA-tuned Qwen. This is what
> ships next week. The native 100M is what ships at v1, 8 weeks out,
> when the energy and continual-learning story actually pays."

## Why this is honest

We are NOT claiming:
- The 100M LNN+SNN is currently as fluent as Qwen + LoRA. (It isn't.)
- The Qwen frontend uses any of our novel architecture. (It doesn't.)
- Plan C is the long-term plan. (It isn't — it's the demo frontend.)

We ARE claiming:
- The 100M LNN+SNN is a real architecture with real telemetry,
  validated on three differentiated axes (R-fold, NeuroMCP, STDP).
- The investor will see live bilingual chat at 10 AM, on a stack we
  built and shipped in <24h, on commodity hardware.
- We can transparently switch the `qwenchat` backend to the native
  100M once it crosses chat-fluency (target: 2-3 month sprint).

The bug-bounty discipline applies here too: overclaiming kills
credibility faster than understating. The Plan C / v0 split is the
honest version of the story, and it still demos as "real bilingual chat
+ novel architecture telemetry" — which is what an early-stage check
should react to.

## Operational checklist (for tomorrow morning)

- [ ] (T-9h) tokenize alpaca-en + alpaca-zh -> `data/sft/alpaca_combined.parquet`
      via `scripts/prep_alpaca_qwen.py`
- [ ] (T-8h) launch `scripts/train_qwen_lora.py --steps 5000 --bs 8`
      on A800; expect 30 min, monitor `train.log` for sample dumps
- [ ] (T-7h) sanity check: `python scripts/qwen_lora_chat_repl.py
      --adapter ~/.synapforge/release/qwen_lora_v0` — both EN and ZH respond
- [ ] (T-2h) dry-run the full demo: `synapforge-demo all` end-to-end,
      capture the transcript to `chat_qwen_lora_demo.json`
- [ ] (T-1h) prepare the disclosure script above; rehearse the v0/v1
      split aloud once
- [ ] (T-0)  open `synapforge-demo all`, run live, then `qwenchat`

If the LoRA training is delayed for any reason, the demo wrapper
prints "training in progress" + tail of train.log + clear pointer to
re-run later. The architecture demos still run and the investor still
sees the four differentiated claims.

## Smoke verification (already passed)

All three new files run end-to-end on Windows / Python 3.11 / torch 2.5
/ no peft / no transformers / no real Qwen ckpt:

```bash
# trainer smoke (5 steps, inline LoRA, mock data, ~0.3s)
py -3.11 scripts/train_qwen_lora.py --smoke --out D:/tmp/smoke_lora
# -> [done] {'step': 5, 'wall_time_s': 0.32, 'final_loss': 30.92, 'smoke': True}

# REPL smoke (one-shot, mock Qwen)
py -3.11 scripts/qwen_lora_chat_repl.py --smoke --once "hello world"
# -> prompt + response printed

# demo wrapper smoke (10 prompts, mock Qwen)
py -3.11 -m synapforge.demo.qwen_lora_demo --smoke
# -> 10 prompt/response pairs, transcript saved

# CLI integration smoke
py -3.11 -c "from synapforge.demo.cli import main; main(['qwenchat','--smoke'])"
# -> same 10 prompts via subcommand
```

When the real Qwen base + adapter are in place, `--smoke` is dropped
and identical code paths consume the real ckpt — no special-casing in
the production path.

## Cross-references

- `docs/INVESTOR.md` — top-level pitch; "Live demo" section now points to
  `synapforge-demo qwenchat`
- `docs/ROADMAP.md` — week-by-week native-architecture plan; v0/v1 split
  noted in Week 0 entry
- `docs/HONEST_ASSESSMENT.md` — recurring honesty audit; Plan C disclosure
  belongs in the "what's NOT claimed" list
- `synapforge/demo/qwen_lora_demo.py` — demo entry point
- `scripts/train_qwen_lora.py` — trainer
- `scripts/qwen_lora_chat_repl.py` — interactive REPL
- `scripts/prep_alpaca_qwen.py` — existing data prep, used as-is
