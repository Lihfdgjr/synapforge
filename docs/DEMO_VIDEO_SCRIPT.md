# SynapForge — 3-Minute Investor Demo Video Script

**Target runtime**: 180 seconds (±5 s).
**Format**: screencast — terminal foreground, occasional slide cut-in for the
elevator pitch and the close. No music. Voiceover only.
**Tone**: honest, no hype. `INVESTOR.md` "What's NOT a claim" is gospel.
**Recording surface**: 1920×1080, dark terminal, 14 pt mono font, prompt
prefix `~/synapforge $ ` so commands read clearly.
**Pre-flight**: `pip install -e .` and a single dry run of
`synapforge-demo all --mechanism-only` to warm Python caches; then record.

> Numbers below pulled from `docs/INVESTOR.md`, `docs/TIMELINE.md`,
> `docs/RFOLD_PAPER.md`, and the verified entries in `DEEP_MAINT_QUEUE.md`
> (T1.4 STDP 1000-trial, T1.6 R-fold A800 bench). DO NOT promise chat
> parity with GPT-4. DO NOT claim ppl < 50 — current best holdout VAL is
> ~320 (Run 3e step 1000).

---

## Scene 1 — Elevator pitch (0:00–0:30, 30 s)

**Camera**: full-screen slide. Single line of body text on dark background:
"Synap-1 — 100M LNN+SNN. No transformer. No KV cache."
At 0:15 cut to a 3-axis label slide:
"Energy / Streaming / Continual Learning."

**Bash**: none in this scene. (The pitch text is rendered by
`synapforge-demo pitch`, but for the video we use a slide — terminal text
at 14 pt is hard to read at YouTube 1080p compression. Reference command
shown lower-third for honesty: `synapforge-demo pitch`.)

**Voiceover** (~75 words, 30 s at 150 wpm):

> GPT-class transformers cost millions to train and burn quadratic compute
> on every long sequence. We took a different bet. Synap-1 is a hundred-
> million-parameter hybrid: continuous-time CfC instead of attention,
> spiking PLIF neurons for sparsity, Hebbian STDP for plasticity at
> inference. Three axes we care about — energy per token, latency under
> streaming context, and continual learning without catastrophic
> forgetting. We will lose at static benchmarks. We win on those three.

---

## Scene 2 — NeuroMCP button demo (0:30–0:45, 15 s)

**Camera**: terminal full-screen. Cursor visible.

**Bash**:

```bash
synapforge-demo button --trials 200
```

**Expected output** (truncated to 2 key lines for the video frame):

```
  initial synapse density: 5.0%   initial codebook K: 9
  trial 200: density 8.1%   K=11   hit-rate 100%
```

**Camera direction**: zoom on the two density lines after the run; underline
"5.0% -> 8.1%" and "K 9 -> 11" with a 1-frame text overlay.

**Voiceover** (~38 words, 15 s):

> NeuroMCP. Synapses grow into the action space — no JSON tool calls, no
> tool-call tokens. Two hundred trials of the four-button env: density
> climbs from five to roughly eight percent and the prototype codebook
> grows two new entries. Mechanism, on CPU, in one second.

---

## Scene 3 — R-fold bench (0:45–1:05, 20 s)

**Camera**: terminal full-screen. Slow-scroll the (N, R) table as it
prints.

**Bash**:

```bash
synapforge-demo bench
```

**Expected output** (the two lines we underline):

```
  correctness: R=1 rel_err=1.5e-06   R=8 rel_err=3.2e-03
     N    R   sequential       r-fold   speedup
    64   16        7.42         2.48      2.99x   (A800, recorded)
```

**Camera direction**: at 0:55 cut a small inset from `docs/RFOLD_PAPER.md`
Appendix B header for one second to show the numbers are recorded, not
generated for the pitch. Then back to terminal.

**Voiceover** (~50 words, 20 s):

> R-fold algebraic CfC. K reasoning steps fold into a single closed-form
> matrix solve. R equals one — exact to floating-point noise. R equals
> eight — drift zero point three percent. On A800, sixty-four batch
> times sixteen folds, two point nine nine times speedup over sequential.
> Eight reasoning steps, one wall-clock step.

---

## Scene 4 — STDP demo (1:05–1:30, 25 s)

**Camera**: terminal full-screen. Watch the ASCII heatmap fill in across
trials 0, 50, 200.

**Bash**:

```bash
synapforge-demo stdp --trials 200 --hidden 64 --seed 11
```

**Expected output** (pull the trial-200 frame for the close-up):

```
  trial   0  density 0.0%   mean|W| 0.000
  trial  50  density 11.3%  mean|W| 0.041
  trial 200  density 27.4%  mean|W| 0.118
              .:=*#*=:.
              :=*#@#*=:
              ...   (8x8 ASCII heatmap, fills in as W grows)
```

**Camera direction**: at 1:20 freeze on the trial-200 heatmap. Bottom-right
text overlay: "no optimizer. no loss. forward-only Hebbian." Hold for
2 s before cutting.

**Voiceover** (~62 words, 25 s):

> STDP at inference. We removed one line — the `if self.training` gate —
> from `bio/stdp_fast.py` line one twenty-one. The Hebbian rule now
> updates fast weights forward-only, from spike co-activation, no
> backprop. Two hundred trials: density climbs zero to twenty-seven
> percent. No optimizer. No loss. Test-Time Training uses gradients.
> We don't. Single-line unlock, paper-grade.

---

## Scene 5 — Chat demo (1:30–2:30, 60 s)

**Camera**: terminal full-screen, font scaled up one notch since each
prompt+response pair gets ~5 s of screen time. If `--ckpt` is unset
the demo replays the last recorded transcript and labels it that way.

**Bash**:

```bash
synapforge-demo chat --max-new 80 --temperature 0.7
```

(Optional, if a healthy ckpt is available:
`synapforge-demo chat --ckpt /workspace/runs/v24h_qwen3/best_*.pt`.
On demo day check `docs/TIMELINE.md` first; if no ckpt has hit val
ppl ≤ 100 yet, **drop this scene to 30 seconds and replay only the
mechanism prompts** — see "Replay fallback" below.)

**Expected output** — 5 EN + 5 ZH, each rendered as
"prompt -> response" pair. Pace: ~6 s per pair. Examples (final wording
matches whatever the live ckpt or recorded transcript prints; do NOT
hand-edit the output for the video):

```
EN-1  "What is a Liquid Neural Network?"
      -> A Liquid Neural Network is a continuous-time recurrent...

EN-3  "Explain the difference between SNN and ANN."
      -> Spiking Neural Networks emit binary spikes; ANNs emit...

ZH-2  "什么是 Hebbian 学习?"
      -> Hebbian 学习是一种基于神经元同时激活...

(... 7 more pairs, paced 6 s each ...)
```

**Camera direction**: cut a small banner across the bottom for the full
60 s reading "RECORDED TRANSCRIPT — last healthy checkpoint" if `--ckpt`
was not passed. If live, banner reads "LIVE — ckpt step N, val ppl X".
Honesty rule: **never hide which mode we're in**.

**Voiceover** (~150 words, 60 s — pace it slow, let the text breathe):

> Chat — five English, five Chinese. The model is a hundred-million-
> parameter Synap-1. Vocabulary is Qwen two-point-five, fifty-one
> thousand tokens. Training is knowledge distillation from Qwen
> two-point-five-zero-point-five-B as teacher, FineWeb plus Alpaca-ZH
> plus GSM8K. Honest framing — this is not GPT-four. This is what a
> twenty-four-GPU-hour rental bought us, on one A800. The prompts you
> see are the standard ten-prompt evaluation set we run on every
> checkpoint. If the on-screen banner says "recorded," the live
> checkpoint hasn't crossed our quality gate yet, and you are watching
> the last passing transcript. We will not fake this number. The
> training timeline is in `docs/TIMELINE.md`, updated every turn —
> phase three SFT, ppl sixty, lands roughly forty-eight hours from
> the date stamp on this video.

**Replay fallback** (if no recent live ckpt):
- Use `synapforge-demo all --mechanism-only`, skip Scene 5 entirely.
- Pad Scenes 2–4 by 10 s each (slow zoom on numbers).
- Add a slide at 1:30 reading "Chat checkpoint pending — see
  `docs/INSURANCE_NATIVE.md` Option C." Hold 3 s. Cut to Scene 6.

---

## Scene 6 — Close (2:30–3:00, 30 s)

**Camera**: split slide. Left half: three bullets. Right half: a static
terminal block showing the verify commands.

**Slide left (text only, no animation)**:

```
ETA to chat-grade ckpt:  ~48h  (TIMELINE.md, best case)
GitHub:                   github.com/<org>/synapforge
What you saw runs today:  pip install -e . && synapforge-demo all
```

**Slide right (terminal block, monospace)**:

```bash
pip install -e .
synapforge-demo pitch     # 30s elevator
synapforge-demo button    # NeuroMCP, ~1s
synapforge-demo bench     # R-fold, ~3s
synapforge-demo stdp      # STDP, ~2s
synapforge-demo chat      # 5 EN + 5 ZH, recorded if no ckpt
synapforge-demo all       # everything in <60s
```

**Camera direction**: hold the split for the full 30 s. At 2:55 fade-out
to a single-line end-card: "Synap-1 — built in twenty-four GPU-hours,
two hundred dollars."

**Voiceover** (~75 words, 30 s):

> What you just saw runs today, on any laptop, in under a minute. The
> live chat checkpoint is forty-eight hours out — best case, per the
> timeline doc. Three months of runway gets you a public chat demo
> and the paper. Three GPU-rental dollars per training day. The repo
> is open. The verify commands are on screen. We're not promising
> GPT-four parity. We're promising a different point on the Pareto
> frontier — energy, streaming, continual learning. The artifact is
> real.

---

## Production checklist

- [ ] Record 1920x1080, 30 fps, 48 kHz mono audio.
- [ ] Pre-warm Python by running `synapforge-demo all` once and
      discarding the recording. First run pays the import cost.
- [ ] Morning of recording, re-check `docs/TIMELINE.md` "current
      coordinate". If best val ppl > 250, use the Scene 5 replay
      fallback.
- [ ] Disable terminal bell, mouse-click sound, OS notifications.
- [ ] Final pass: confirm every on-screen number matches a doc.
      Scene 2 -> `INVESTOR.md` claim 1.
      Scene 3 -> `RFOLD_PAPER.md` Appendix B.
      Scene 4 -> `INVESTOR.md` claim 3 + T1.4 entry.
      Scene 5 ckpt step / val ppl -> `docs/TIMELINE.md`.
      Scene 6 ETA -> `docs/TIMELINE.md` "Phase 3 SFT" row.
- [ ] If a number drifts, edit this script first, commit, then record.

## Hard rules

1. Do NOT promise chat parity with GPT-4 / Claude / Gemini.
2. Do NOT claim ppl < 50 unless live ckpt has crossed.
3. Do NOT hide live-vs-recorded — Scene 5 banner required.
4. Do NOT improvise voiceover — every sentence is calibrated.
5. The `--mechanism-only` fallback is FINE. Honest beats inflated.
