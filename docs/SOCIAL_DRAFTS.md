# SOCIAL_DRAFTS.md — milestone tweet drafts (DO NOT POST)

> Drafts for X/Twitter announcements when each milestone fires. Tone rules
> (per `docs/INVESTOR.md` "What's NOT a claim"): honest, no GPT-4 parity
> claim, lead with mechanism, engineer audience. No emojis. Cite real
> numbers (ppl, density %) verbatim from `docs/INVESTOR.md`.
>
> Each tweet ≤ 280 chars. Each links to GitHub repo or a concrete doc.
> Repo: https://github.com/Lihfdgjr/synapforge

---

## 1. STDP demo — mechanism tweet

**Trigger**: any time after T1.4 records density ≥ 20% at trial 200
(currently 27% at 200, 59.8% at 1000 in the latest run). Post on a low-
traffic morning slot to seed reach before the bigger drops below.

**Suggested tag**: none. Mention `@PyTorch` only if engagement looks dead
after 24h.

**Tweet body** (270 chars):

```
Inference-time STDP: synapse density 0% to 27% in 200 trials.
No optimizer, no loss, no backward pass. Pure spike co-activation
on a 100M LNN+SNN backbone.

Hebbian rule at inference, forward-only weight updates.

github.com/Lihfdgjr/synapforge
Run: synapforge-demo stdp
```

---

## 2. R-fold algebraic — paper teaser

**Trigger**: when `docs/RFOLD_PAPER.md` is public on GitHub (already
landed) AND T1.6 records the A800 N=64 R=16 number (done: 2.28x). Post
within 24h of the paper PDF being browsable on the repo.

**Suggested tag**: none. Mention `@arxiv` if linking to a preprint
later.

**Tweet body** (263 chars):

```
The CfC step is a linear map. k steps compose into one closed-form
matrix solve.

R-fold: k reasoning steps for the price of 1 matmul.
R=1 rel_err 1.5e-6, R=8 rel_err 0.32%. A800 N=64 R=16: 2.99x speedup.

Math + bench in the paper.
github.com/Lihfdgjr/synapforge
```

---

## 3. NeuroMCP — radical claim

**Trigger**: when T1.5 NeuroMCP 6000-trial run records K >= 14 AND
density >= 20%. This is the "agents without tool tokens" headline drop.

**Suggested tag**: none on first post. If it lands, follow up tagging
`@huggingface` with the demo command.

**Tweet body** (273 chars):

```
NeuroMCP: agents pick tools without emitting a tool_call token.

A sparse synapse layer + dynamic prototype codebook grows new
connections as the net discovers the action space.

4-button env, 100% hit. Synaptic growth replaces JSON parsing.

github.com/Lihfdgjr/synapforge
```

---

## 4. Inference-time STDP unlock — technical bait

**Trigger**: post when T8.1 (inference STDP weight diff) lands real
||delta-W|| numbers from a 1K-token chat run. This is the "single-LOC
unlock buys a paper" drop, aimed at neuromorphic + TTT folks.

**Suggested tag**: none on the first post. If it gets engagement, reply
with the line number and a screenshot of the diff.

**Tweet body** (278 chars):

```
bio/stdp_fast.py:121 had `if self.training:` gating the Hebbian rule.

Removed it. STDP now runs at inference. Forward-only weight updates
from live context, no backward pass.

TTT (Sun et al. 2024) uses gradient. We don't.

One line. Paper-grade.
github.com/Lihfdgjr/synapforge
```

---

## 5. Phase 1 reached — milestone trigger

**Trigger**: H5 phase autopilot fires when val ppl <= 250 AND
phase 1 self-learn flags activate (`--self-learn-ttt --self-learn-k 8
--curiosity-weight 0.05 --phase-aware`). Post within 1 hour of the
relaunch landing.

**Suggested tag**: none. This is internal-engineering-flavored.

**Tweet body** (243 chars):

```
Synap-1 hit val ppl 250. Phase 1 unlocked: TTT self-learn k=8 +
STDP-driven curiosity (w=0.05) now active in training.

100M LNN+SNN, Qwen 0.5B teacher KD, no transformer in the student.

Phase 0 -> 1 in N hours.
github.com/Lihfdgjr/synapforge
```

---

## 6. Phase 2 reached — milestone

**Trigger**: H5 fires when val ppl <= 100 AND
`--modal-list image,audio,time_series` flags activate. Phase 2 = byte-
patch multimodal turning on.

**Suggested tag**: none. If reach is dead, reply with the byte-patch
reference (Pillow GIF -> imageio mel patches in the synth pipeline).

**Tweet body** (269 chars):

```
Synap-1 val ppl 100. Phase 2: byte-patch multimodal active.

image, audio, time_series all enter the same token stream as bytes.
No frozen vision encoder, no projection adapter. One backbone,
many modalities, native tokens.

100M LNN+SNN.
github.com/Lihfdgjr/synapforge
```

---

## 7. First chat-grade ckpt — public release announce

**Trigger**: val ppl <= 60 AND chat_eval >= 0.6 (Phase 4 RL gate from
H5 table). T4.1 GH Release `synap-1-chat-v1` uploaded. This is the
first checkpoint that can hold a coherent multi-turn chat.

**Suggested tag**: `@huggingface` (HF Hub release alongside GH).

**Tweet body** (274 chars):

```
First chat-grade Synap-1: val ppl X, chat_eval Y on the holdout set.

100M LNN+SNN, trained on a single A800 80GB. Qwen 0.5B teacher KD,
no transformer code path in the student.

Weights + 5 EN + 5 ZH transcripts on the release page.

github.com/Lihfdgjr/synapforge/releases
```

**Fill X (val ppl) and Y (eval) verbatim from `docs/CHAT_SAMPLES.md`
row at trigger time.**

---

## 8. Open-source frame — community

**Trigger**: post 24-48h after #7 (chat-grade release). Pin to profile.
This is the "if you read this far, here is the frame" drop, aimed at
contributors not investors.

**Suggested tag**: none on first post. Reply with `@github` if it
lands.

**Tweet body** (275 chars):

```
synapforge is MIT, ~5K LOC, single-author. Replaces transformers
with continuous-time CfC + PLIF spiking neurons + Hebbian STDP.

If you've wanted to grep a small LNN+SNN trainer end to end,
this is one.

PRs welcome: kernels, eval, bio rules.

github.com/Lihfdgjr/synapforge
```

---

## Posting rules (for the human, NOT for cron)

1. Drafts only. Do NOT auto-post. Verify each tweet's claim is still
   true at the moment of posting.
2. If a number changes (val ppl drifted, density saturated lower),
   edit the tweet to match `docs/INVESTOR.md` and `docs/PROGRESS.md`.
3. Never claim parity with GPT-4, Claude, Gemini. Never claim the
   100M ckpt is product-ready unless `chat_eval >= 0.7` is in
   CHAT_SAMPLES.md.
4. Tweet 7 has placeholders X and Y. Fill them from the actual run
   when posting.
5. Keep emojis off. Engineer audience.
