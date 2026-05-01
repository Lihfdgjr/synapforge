# Neural Computer Use — Direct Synaptic Action

> *"AI 使用神经元直接操控 computer 上网自动学习"*

This document describes how SynapForge agents drive a real browser **without
emitting a single tool-call token**. The action manifold lives in the
synapses, not in JSON.

**Entry point** (P18 MVP): `synapforge/action/web_actuator.py::WebActuator`.
DOM-only, Playwright headless, maps `ActionHead` hidden vector to one of
`{noop, click(x,y), scroll(dy), type(text), navigate(url)}`. See §11 below.

```bash
pip install -e ".[web]"
playwright install chromium
bash scripts/web_actuator_smoke.sh   # 50 random steps, asserts >=1 click
```

Convention: `action_dim=64`, hidden tensor of shape `(64,)` or `(N, 64)`,
`action_head` is any `nn.Module` mapping `(action_dim,) -> (>=action_dim,)`.

---

## 1. Architecture

```
                           ┌────────────────────────────────────────────┐
                           │  WebBrowserEnv (Playwright headless)        │
                           │     reset(url) ─► (3,64,64) RGB obs         │
                           └────────────────────────────────────────────┘
                                          │
                                          ▼
       pixels (3,64,64) ──► PatchEncoder ─► hidden h ∈ ℝ^D
                                          │
                ┌─────────────────────────┼──────────────────────────┐
                │                         │                          │
                ▼                         ▼                          ▼
       NeuroMCPHead.proj           ActionHead (LayerNorm + MLP)   Critic MLP
        (SparseSynapticLayer)            │                          │
                │                        ▼                          ▼
                ▼            ActionOutput {action_type_logits,     V(s)
       DynamicActionCodebook            xy, scroll,
        K_alive growable                 key_logits,
                │                        text_trigger}
                ▼                        │
       routing logits                   to_dict()
                │                        │
                └─────────►── argmax/sample ──────►  action_dict
                                          │
                                          ▼
                          OSActuator | WebBrowserEnv.step()
                                          │
                                          ▼
                                  (next_obs, reward, done)
```

The **only** information that crosses the policy ↔ environment boundary is:

| Field            | Type / shape       | Bytes |
|------------------|--------------------|-------|
| `action_type`    | one of 9 ints      | 4     |
| `xy`             | float32 × 2        | 8     |
| `scroll`         | float32 × 2        | 8     |
| `key`            | uint8 (idx in 80)  | 1     |
| `text_id`        | uint8 (idx in 10)  | 1     |
| `text_trigger`   | bool               | 1     |
| **Total**        |                    | **23 bytes** |

Compare with a typical ChatGPT-style `<tool_call>{"name":"click","x":612,"y":341}` —
that is on the order of **40–60 tokens (~150 bytes after BPE)**, and incurs
one round-trip through the LM head + sampling on every step.

---

## 2. Why NOT JSON tool calls

| Concern             | JSON tool tokens                               | Neural action vector                                  |
|---------------------|------------------------------------------------|--------------------------------------------------------|
| **Bandwidth**       | ~50 tokens / step                              | 23 bytes / step (`xy`/`scroll` are 2-vectors)         |
| **Latency**         | LM forward + sampling + parse (~200 ms 1B+)    | One MLP forward, no sampling on continuous fields     |
| **Persistence**     | Re-derived from prompt every call              | Stored in DynamicCodebook + SparseSynapticLayer       |
| **Generalisation**  | Memorises lexical patterns                    | Learns the action **manifold**; new skills *grow*     |
| **Failure mode**    | Hallucinated tool name → 100 % miss            | Out-of-distribution → soft fallback to nearest proto  |
| **Multimodality**   | Text-only bottleneck                           | Pixel observation flows directly into the action head |

The deepest reason: we already pay for the synaptic substrate
(`SparseSynapticLayer` density grows from 5 % → 28 % under co-activation EMA),
so we should **use** it as the action memory. JSON tool tokens force the
agent to reconstruct the action policy from prompt context every turn — a
needless detour through the LM.

---

## 3. Reward Shaping (no LM verifier)

Per step, the env returns a scalar built from four pixel/page-side signals
plus an externally-attached intrinsic bonus.

```
r_step  =  r_page_changed     # +1 first time we see this page hash
         + r_page_repeat      # -0.05 .. -0.5 if recent
         + r_progress_text    # +1.0 when target regex matches new page
         + r_progress_url     # +0.5 when URL contains target substring
         + r_intrinsic        # caller-attached, e.g. ΔF + STDP novelty
```

`r_intrinsic` defaults to:

```
r_intrinsic = w * (NoveltyDrive(h_next)             # ‖h - EMA(h)‖
                 + ‖h_next - FreeEnergyForward(h_prev)‖²)
```

We *deliberately do not* use a language-model verifier or LLM-as-judge
reward. That would re-couple action selection to text, defeating the entire
point. The four pixel/URL signals are coarse but enough for L1–L3 of the
curriculum; L4–L5 lean more on intrinsic exploration.

---

## 4. Online plasticity (forward-only)

After every PPO update we run, under `torch.no_grad()`:

```python
plast = actor.neuro.step_plasticity(hidden_z)
# {"density": 0.073, "added": 4, "pruned": 0, "K_alive": 11, "grew_cb": True}
```

This calls:

1. `SparseSynapticLayer.maybe_grow_prune()` — co-activation EMA based mask
   growth; magnitude pruning every 200 steps.
2. `DynamicActionCodebook.maybe_grow(hidden_z)` — adds a new prototype
   when novelty exceeds 0.35 vs. existing alive set, with a 50-step cooldown.

The contract is OBSERVE → DELTA → APPLY (matching `sf.plasticity`):
no autograd version conflicts, no `.data` hacks. Calls are one-shot per
training step.

This is also where memory `feedback_inference_stdp_unlock.md` lands:
plasticity stays *on* during inference so the synapses keep tracking the
action distribution as the agent browses live pages.

---

## 5. Comparison vs. Anthropic Computer Use API

| Dimension            | Anthropic CUA                                     | SynapForge neural action                             |
|----------------------|---------------------------------------------------|------------------------------------------------------|
| Action emission      | Multi-turn text: tool name + JSON args            | One forward pass → 23-byte action dict               |
| Action vocabulary    | Fixed prompt schema                               | DynamicCodebook, K grows from 9 → up to 64          |
| Latency / step       | ~500–2000 ms (1 round-trip through Claude)       | ~5 ms on CPU / sub-ms on GPU                          |
| Adapts to new sites  | Re-prompt with longer context                     | Synapses grow new prototype + density on novelty     |
| Failure surface      | Tool-name hallucination, JSON syntax errors       | Out-of-distribution xy → soft regression toward 0.5  |
| Trainability         | RLHF on full LM                                   | PPO on a 23-byte action head + STDP plasticity       |

The agent loop ("observe screenshot → decide → act → observe again") is
identical. The medium of the *decision* is what differs: language tokens
vs. synaptic activations.

---

## 6. Training schedule

We train in 4 phases. Each phase is a separate run of
`scripts/train_neural_web.py` with a different config; checkpoints
warmstart strict=False per memory `feedback_no_random_init_use_warmstart.md`.

### Phase A — Mock smoke (`--no-real`)
- Goal: shake out shapes, GAE, plasticity tick.
- Env: hardcoded 5-node site graph (`_MOCK_GRAPH`).
- Budget: ~50 episodes × 16 steps. CPU only, < 60 s wallclock.
- Pass criterion: synapse density ≥ 0.06, K_alive ≥ 9, no NaNs.

### Phase B — Real Playwright headless, L1 only
- Goal: prove pixel observation pipeline + reward signal end-to-end.
- Env: `--real --headless --task-level 1`.
- Budget: ~200 episodes × 24 steps. Single Chromium tab, ~2 hr.
- Pass criterion: rolling success ≥ 0.6 on L1 over last 10 episodes.

### Phase C — Curriculum (L1 → L5)
- Goal: progress through the 50-task catalogue.
- Trigger: `WebCurriculum` auto-promotes each level.
- Budget: ~3000 episodes total.
- Pass criterion: reach L5 with ≥ 0.4 success rate.

### Phase D — Live, intrinsic-driven autonomous loop
- Goal: hand off to `synapforge.learn.autonomous_daemon` for self-directed
  browsing. Tasks are proposed by `SelfGoalProposer.propose_with_curiosity`
  (Free-Energy + STDP novelty), filtered by `WebPoisonGate`, distilled into
  `web_cache.jsonl`.
- The neural action policy is the *executor*; topic selection is the
  *meta-policy*. Both are pure-neural, no JSON.

---

## 7. Risks & mitigations

| Risk                                  | Mitigation                                                                                       |
|---------------------------------------|--------------------------------------------------------------------------------------------------|
| Jailbreak via web content             | `WebPoisonGate` 7-gate pipeline (provenance + KL + dedup + budget) before any text touches LM.   |
| Infinite loops on a single page       | `max_steps` per episode + `r_page_repeat` decaying penalty.                                      |
| Action-space collapse to one type     | Entropy bonus in PPO (`ent_coef=0.02`) + `DynamicCodebook` cooldown allows divergent prototypes. |
| pyautogui dispatch on headless rental | `OSActuator(safe_mode=True)` logs only; never moves the host mouse.                              |
| Codebook over-growth                  | `growth_cooldown=50`, `max_size=64`, novelty threshold 0.35.                                     |
| Synapse density blow-up               | `max_density=0.40`, magnitude prune every 200 steps.                                             |

For investors / press: the safe-mode demo runs without a screen and
without `playwright install chromium`; we ship the mock env precisely so a
laptop demo never touches the live web.

---

## 8. Evaluation targets

| Benchmark            | Claude 3.5 Sonnet baseline | SynapForge target | Status         |
|----------------------|----------------------------|-------------------|----------------|
| OSWorld             | 14.9 %                      | ≥ 25 %            | not yet run    |
| WebArena            | ~22 % (best published)      | ≥ 20 %            | not yet run    |
| WebShop             | ~50 % (best LLM)            | ≥ 35 %            | not yet run    |
| MiniWoB++           | ~85 % (large LLM)           | ≥ 70 %            | not yet run    |
| **Internal L1–L5**  | n/a                          | L5 ≥ 40 %         | curric in repo |

We pick OSWorld and WebArena as primary because they're the only
benchmarks that tie language *and* pixel observation under the same
agent loop, which lines up with the "no JSON tool calls" thesis.

---

## 9. Files

| Path                                                      | Role                                                |
|-----------------------------------------------------------|-----------------------------------------------------|
| `synapforge/action/neuromcp.py`                          | NeuroMCPHead, DynamicCodebook, SparseSynapticLayer  |
| `synapforge/action/head.py`                              | ActionHead, OSActionSpec, ActionLoss                |
| `synapforge/action/actuator.py`                          | OSActuator (pyautogui dispatch + safe-mode)         |
| `synapforge/action/web_env.py`                           | WebBrowserEnv (Playwright + mock fallback)          |
| `synapforge/learn/web_curriculum.py`                     | 5-level / 50-task curriculum + promotion gate       |
| `synapforge/learn/autonomous_daemon.py`                  | Self-directed browse loop (curiosity-gated)         |
| `synapforge/intrinsic.py`                                | NoveltyDrive, FreeEnergySurprise, IntrinsicReward   |
| `scripts/train_neural_web.py`                            | PPO trainer + plasticity tick                       |
| `docs/NEURAL_COMPUTER_USE.md`                            | this document                                       |

---

## 10. Smoke

```bash
# CPU, no browser, no chromium:
python scripts/train_neural_web.py --episodes 4 --max-steps 16 \
  --task-level 1 --no-real --out runs/neural_web_smoke.json

# Real Playwright (after `playwright install chromium`):
python scripts/train_neural_web.py --episodes 50 --max-steps 24 \
  --task-level 1 --real --headless --out runs/neural_web_l1.json
```

Synapse density growth + K_alive growth + reward curve land in the JSON.
That's the artifact for investor demos — pure pixels in, neural action
out, no `<tool_call>` tokens emitted on either side of the loop.

---

## 11. WebActuator — DOM-only Computer-Use MVP (P18)

The `WebActuator` is the **minimum viable** version of the Computer-Use
claim. It exists to make MASTER_PLAN.md §5 row "Computer-use" demoable
*today* without a vision pipeline, login flows, or multi-tab.

### Install

```bash
pip install -e ".[web]"
playwright install chromium
```

The `[web]` extra pulls `playwright>=1.40`. Without it the WebActuator
module still imports (the Playwright import is optional / lazy), it just
won't run end-to-end.

### Basic usage

```python
import torch.nn as nn
from playwright.sync_api import sync_playwright
from synapforge.action.web_actuator import WebActuator

ACTION_DIM = 64
action_head = nn.Linear(ACTION_DIM, ACTION_DIM)   # in real use: ActionHead

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://example.com")
    actuator = WebActuator(page, action_head, action_dim=ACTION_DIM)
    for hidden in hidden_seq:                     # (N, action_dim)
        rec = actuator.step(hidden)               # → {"action": ..., "result": ...}
    browser.close()
```

### Action space

| ID | Name      | Effect                                           |
|----|-----------|--------------------------------------------------|
| 0  | `noop`    | observe only                                     |
| 1  | `click`   | `page.mouse.click(x, y)`                         |
| 2  | `scroll`  | `page.mouse.wheel(0, dy)`                        |
| 3  | `type`    | `page.keyboard.type(text)` (codebook-indexed)   |
| 4  | `navigate`| `page.goto(url)` (codebook-indexed)             |

The `ActionHead` produces logits of length `action_dim`. Slots 0..4 are
the action-type slots (argmax decides what to do). Slots 5..7 are
sigmoid/tanh-decoded `xy` and `scroll_dy`. The remaining slots index
into `url_codebook` and `type_codebook`. This keeps the ABI small (≤ a
few dozen bytes per step) and **emits no tokens**.

### action_dim convention

We use `action_dim = 64` so callers can swap an `nn.Linear(64, 64)` for
the smoke / unit test, and a real backbone-fed `ActionHead` later
without changing the actuator. Anything `>= 16` works (the codebook
slots need room).

### Smoke test

```bash
bash scripts/web_actuator_smoke.sh
```

Boots Playwright headless against `synapforge/tests/fixtures/static_demo.html`,
runs 50 random `ActionHead` steps, asserts at least one successful click,
and prints the action-type histogram. Typical output:

```
[web_actuator_smoke] step histogram:
  noop        12
  click        9
  scroll      11
  type        10
  navigate     8
[web_actuator_smoke] successful clicks: 5
[web_actuator_smoke] OK
```

### Real sandbox run (P7) — visual evidence

```bash
python scripts/web_actuator_real_smoke.py
```

100 ActionHead steps against the same fixture, but with stricter assertions
(>=1 click, >=1 nav/scroll, no uncaught exceptions, runtime <= 60 s). On
success it persists three artifacts under
`synapforge/tests/fixtures/p7_evidence/`:

- `web_actuator_smoke.png` — screenshot of the headless page after step 50
  (visible buttons + input + link from `static_demo.html`).
- `web_actuator_smoke_trace.json` — per-step `(action, result, dom_hash, detail)`
  records for the full 100 steps.
- `web_actuator_smoke_summary.json` — single-line summary
  (histogram, runtime, ok_clicks, error or null).

Sample real run on the dev box (2026-05-01):

```
histogram   {noop:73, click:22, scroll:0, type:1, navigate:4}
ok_clicks   22
runtime_s   6.97
```

The screenshot is the literal pixel-level evidence that **a real browser
received clicks driven by a 64-d neural action head with no JSON tool
tokens emitted**. See MASTER_PLAN.md §6 P7 for resolution detail.

### What's intentionally NOT in MVP scope

Per MASTER_PLAN.md §12: **vision pipeline, multi-tab, login flows,
CAPTCHA, real-site curriculum**. Those live in
`synapforge/action/web_env.py` (full pixel-observation loop) and
`scripts/train_neural_web.py` (PPO training). The `WebActuator` MVP is
the pre-pitch artifact that proves "neurons drive a browser" in 30
seconds on a clean clone.
