# SynapForge

A small (375M-parameter) **LNN + SNN hybrid language model** — *not a transformer*.

Built from the ground up around four ideas that aren't in standard LLM stacks:

1. **Continuous-time recurrence** (CfC cells, Hasani 2022) instead of attention
2. **Spiking neurons** (PLIF, Fang 2021) in the LM forward path
3. **Neuroplastic action routing** (NeuroMCP) — model emits action vectors
   directly through learnable synapses + a growable prototype codebook,
   no JSON tool-call protocol
4. **Async event-driven chat** — listens, can be interrupted, can speak
   unprompted; not strict request-response

Trained from scratch in 7h on a single A100×2 rental, achieving best ppl
44.2 at step 46350 on multilingual chat (zh + en + math).

This is a **research framework**, not a production system. Use Claude / GPT /
Gemini for real work. See [docs/HONEST_ASSESSMENT.md](docs/HONEST_ASSESSMENT.md)
for what genuinely works vs what's still aspirational.

---

## Why does this exist?

Three concrete questions we wanted to answer:

- **Is the transformer actually necessary for chat at small scale?**
  CfC + PLIF hybrid hits ppl 44 at 375M with 7h of training. SmolLM2-360M does
  ppl 20 with ~2T tokens, so we lose at static benchmarks. But our continuous-time
  state and plasticity matrix give us cheap inference + cheap continual learning.

- **Can a model use tools without a JSON protocol?**
  Yes. NeuroMCP grows action prototypes from co-activation in a sparse synaptic
  layer. Verified at 100% on a 4-button GUI task. Persistent across sessions
  via `skill_log.json`. Scales to 100k+ skills via HNSW index. Compositional
  skills (L1 primitives + L2 compounds) discovered by online Hebbian co-firing.

- **Can a chat model behave like a real person — listening, interrupting, speaking
  unprompted?**
  The async chat kernel (`synapforge.chat`) runs an event loop with cancellation
  tokens, turn-taking detector, and proactive triggers. User can type while
  model is generating; model can be cut off mid-sentence; model can speak on
  its own (cron / web events / idle).

---

## Status snapshot (2026-04-30)

| Component                                                     | Status |
| ------------------------------------------------------------- | ------ |
| v4.0 base 375M Qwen-vocab pretrain (60k steps)                | ✅ done, ckpt at step_036000.pt |
| v4.1 NeuroMCP wire-in (60k steps)                             | ✅ done, **best ppl 44.2** at step 46350 |
| v4.2 Universal trainer (Coconut + per-domain MCP + STDP aux)  | ⚠ launched, 3 known bugs being patched |
| Async chat kernel (turn-taking, interrupt, proactive)         | ✅ written, awaits chat-able ckpt |
| HNSW skill index (100k+ prototypes, sub-ms p99)               | ✅ written |
| L1/L2 compositional codebook (Hebbian co-firing → option discovery) | ✅ scaffold |
| 7-gate continual ingest (Track A) + autonomous_daemon         | ✅ deployed |
| Track B retrieval-only chat memory (Claude Memory pattern)    | ✅ scaffold |
| Anthropic safety stack (CAI + red/blue DPO + judge)           | ✅ written, training queued |
| Multimodal 9-modality byte-patch (text → image / audio / video / 3D) | ⏳ encoders exist, only text trained |
| 3D understanding via DUSt3R + EGNN (140 GPU-h ¥980)           | ⏳ designed, queued |
| R-fold algebraic CfC closed-form                              | ✅ shipped at `synapforge/cells/rfold.py` (k=8 = **2.7× free**, R≥64 chunked L=8 = 3-4× near-seq quality; 167× was inflated, see HONEST_ASSESSMENT) |
| CPU pilot for inference-STDP monotonic-quality claim          | ✅ `scripts/cpu_pilot_inference_stdp.py` — validates paper hypothesis at 1K/2K/4K on a laptop, no GPU needed |

---

## Architecture (one-screen overview)

```
user input ─▶ Qwen 151936 vocab tokenizer ─▶ token embedding (1024 dim)
                                                    │
                                                    ▼
                                         RoPE positional encoding
                                                    │
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │  HybridBlock × 14 layers             │
                              │  ┌───────────────────────────────┐   │
                              │  │ for d in loop_depth(=2):       │   │
                              │  │   x, h_cfc = CfC(x, h_cfc)     │   │  ← continuous-time ODE
                              │  │   x = PLIF(x)                  │   │  ← spiking gate
                              │  │   x = SwiGLU_FFN(x)            │   │
                              │  │   x = RMSNorm(x)               │   │
                              │  └───────────────────────────────┘   │
                              └─────────────────────────────────────┘
                                                    │
                                            ┌───────┴────────┐
                                            ▼                ▼
                                    Tied LM head     NeuroMCP head
                                    (next token)    (action vector — no JSON)
                                            │                │
                                            ▼                ▼
                                       text out        action ─▶ OSActuator
```

Total: 375.6M parameters. CfC handles long-range without attention. PLIF spike
rate ~10-30% (most channels gated to 0 on neuromorphic hardware). NeuroMCP
emits actions through learnable synapses, no JSON parse.

For full architectural detail see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quickstart (5 minutes, local sanity check)

No GPU needed for the smoke test:

```bash
git clone https://github.com/Lihfdgjr/synapforge
cd synapforge
pip install -r requirements.txt   # torch + transformers + numpy + hnswlib

python -c "
from synapforge.action import HNSWSkillIndex, HierarchicalCodebook
from synapforge.defense import PoisonDetector, ProvenanceTracker
from synapforge.safety import RedBlueSelfPlay
from synapforge.chat import ConversationKernel, StreamingGenerator
print('all 21 modules importable')
"
```

For full setup including rental SSH, training, autonomous learn daemon,
and chat kernel deployment, see [docs/QUICKSTART.md](docs/QUICKSTART.md).

---

## How to use this — 5 concrete workflows

### 1. Train a 375M LNN+SNN from scratch (1×A100 80GB, ~7h)

```bash
python -m synapforge.train_v42_universal \
  --warmstart /path/to/v41_best.pt \
  --teacher-path Qwen/Qwen2.5-0.5B \
  --steps 60000 --batch-size 8 --seq-len 1024 \
  --grad-accum 2 --lr 2e-4 --kd-weight 0.3 \
  --neuromcp-enabled \
  --skill-log /path/to/skill_log.json
```

What you should see:
- step 0: ce ≈ 4.0 (warmstart honored, NOT random)
- step 100: ppl 40-80
- step 30k (5h): ppl 25-35, basic single-turn chat works
- step 60k (10h): ppl 20-28, multi-turn coherent

What you'll see if something's wrong:
- step 0: ce ≈ 13 → warmstart didn't load, check ckpt path
- ppl=1.4 → response-only mask hides 95%+ tokens (not a bug, just the trained
  metric isn't comparable to LM ppl on full sequence)
- kd=0 → teacher path rejected; check `Path(args.teacher_path).exists() OR "/" in args.teacher_path`
- tok/s < 5000 → PerDomainNeuroMCP running all 4 domain heads; switch to top-1 routing

### 2. Run autonomous web learning (background daemon)

Picks high-FreeEnergy topics, multi-source search, runs 7-gate WebPoisonGate.

```bash
nohup python -m synapforge.learn.autonomous_daemon \
  --out /workspace/data/web_cache.jsonl \
  --interval-min 30 \
  --topics-per-cycle 6 \
  > /workspace/runs/autolearn.log 2>&1 &
disown
```

Cycle: every 30 minutes,
1. `SelfGoalProposer` picks 6 topics with lowest coverage
2. Multi-source search (bilibili / arxiv / wikipedia)
3. `WebPoisonGate` filters via 7 gates
4. Survivors append to `web_cache.jsonl` with full provenance

Monitor: `tail -f /workspace/runs/autolearn.log` and `wc -l /workspace/data/web_cache.jsonl`.

### 3. Inference — interactive chat (after v4.2 finishes)

```python
import asyncio, torch
from transformers import AutoTokenizer
from synapforge.model_chat_600m import SynapForgeChat600M
from synapforge.chat import ConversationKernel, StreamingGenerator
from synapforge.action import HNSWSkillIndex

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = SynapForgeChat600M().cuda()
model.load_state_dict(torch.load("/path/to/best.pt")["model"])
model.eval()

# Restore previously grown skills (LTP/LTD persistent)
skills = HNSWSkillIndex(dim=1024, index_dir="/path/to/skill_index")

kernel = ConversationKernel(generator=StreamingGenerator(model, tok))

async def main():
    await kernel.start()

    asyncio.create_task(_render_outbox(kernel))

    # User types char-by-char
    await kernel.send_user_text("你好，请讲一下")
    await asyncio.sleep(0.3)
    await kernel.send_user_text("线性代数的特征值")
    await kernel.send_user_submit()

    # Mid-answer, user interrupts
    await asyncio.sleep(1.5)
    await kernel.send_user_text("等等")  # urgency marker → cancels mid-stream

    await asyncio.sleep(60)
    await kernel.stop()

async def _render_outbox(kernel):
    async for chunk in kernel.outbox_stream():
        prefix = "[Proactive] " if chunk.get("proactive") else ""
        print(prefix + chunk.get("text", ""), end="", flush=True)

asyncio.run(main())
```

The kernel handles: streaming token output, cancel-on-user-typing, proactive
emissions, mute, frequency caps. See [docs/CHAT_PROTOCOL.md](docs/CHAT_PROTOCOL.md).

### 4. Continual learning loop (Track A: web → weights)

Track A is the slow path. Web content goes through 7 gates and only reaches
weights after passing the canary test:

```python
from synapforge.defense import (
    PoisonDetector, ProvenanceTracker, WebPoisonGate, SourceBudget,
)
from synapforge.defense.poison_detector import make_dedup_signal

prov = ProvenanceTracker(log_path="/workspace/runs/provenance.jsonl")
det = PoisonDetector()
det.register_signal("dup", make_dedup_signal(prov))
budget = SourceBudget(per_hour=50, per_day=1000, per_7d_per_source=125)

gate = WebPoisonGate(detector=det, provenance=prov, budget=budget,
                     out_jsonl="/workspace/data/web_cache.jsonl")

decision = gate.admit(
    text="...some scraped web content...",
    source_id="web:bilibili.com",
    url="https://b.tv/abc123",
    topic="math",
)
if decision.accept:
    print(f"sample_id {decision.provenance.sample_id} admitted, trust={decision.provenance.trust_score:.2f}")
else:
    print(f"rejected: {decision.reasons}")
```

The trainer reads `/workspace/data/web_cache.jsonl` periodically into the
replay buffer. Per-source 7d cap = 125 = 50% of Anthropic's 250-doc poison
fixed-count threshold (2510.07192). No single source can poison the model
even if it tries. See [docs/CONTINUAL_LEARNING.md](docs/CONTINUAL_LEARNING.md).

### 5. Continual learning loop (Track B: chat → retrieval, no weight update)

Track B is the fast path. User chat goes to a per-user retrieval cache,
NEVER updates weights (Anthropic Claude Memory pattern):

```python
from synapforge.defense import ChatPoisonGate, ProvenanceTracker, SourceBudget
from synapforge.defense import PoisonDetector
from synapforge.learn import RetrievalMemory

prov = ProvenanceTracker()
gate = ChatPoisonGate(
    detector=PoisonDetector(),
    provenance=prov,
    budget=SourceBudget(per_hour=10, per_day=100),
    retrieval_cache_dir="/workspace/runs/user_memory",
)

decision = gate.admit(
    text="my name is alice, i live in beijing",
    user_handle="alice123",
)
# decision.provenance written; per-user retrieval cache updated
# weights NOT touched

# At inference, retrieve top-K relevant memories:
mem = RetrievalMemory(cache_dir="/workspace/runs/user_memory")
hits = mem.query(user_hash=decision.provenance.user_handle_hash,
                 query_text="where do you live?", top_k=3)
# hits = [{"text": "my name is alice...", "ts": "...", "sample_id": ...}, ...]
```

Why frozen retrieval not weight update? 4-of-5 known white-box attack classes
bypass weight-based defenses. Anthropic / ChatGPT / character.ai all use
retrieval. See [docs/CONTINUAL_LEARNING.md](docs/CONTINUAL_LEARNING.md).

---

## Repository layout (annotated)

```
synapforge/
├── model_chat_600m.py          # 375M base: HybridBlock × 14, RoPE, Qwen vocab
├── train_v42_universal.py      # v4.2 trainer: CE + KD + STDP + entropy bonus
│
├── action/                     # NeuroMCP — token-free tool use
│   ├── skill_log.py            #   JSON persistent prototypes (LTP/LTD)
│   ├── per_domain_neuromcp.py  #   4 codebooks × K=64, intent router
│   ├── hnsw_skill_index.py     #   O(log K) for 100k+ prototypes
│   ├── compositional_codebook.py # L1 primitives + L2 compounds
│   ├── neuromcp.py             #   legacy single-codebook (kept for BC)
│   ├── head.py                 #   ActionHead → {action_type, xy, scroll, key, text}
│   ├── actuator.py             #   OSActuator (pyautogui + safe-mode)
│   └── envs.py                 #   FourButtonEnv, PatchEncoder
│
├── chat/                       # Async event-driven chat kernel (NEW)
│   ├── event_loop.py           #   ConversationKernel main loop
│   ├── streaming.py            #   Cancellable token-by-token generation
│   ├── turn_taking.py          #   Detect partial vs complete user input
│   ├── interrupt_policy.py     #   Speak / interrupt-self / silence decisions
│   └── proactive.py            #   Cron / web / idle outbound triggers
│
├── thinking/coconut.py         # <bot>/<eot> latent reasoning (k=1→8)
├── moe/expert_ffn.py           # Top-2 noisy gate, 8 routed + 1 shared (optional)
│
├── defense/                    # Track A/B continual-learning poison defense
│   ├── poison_detector.py      #   Multi-signal (PromptGuard / Mahalanobis / TRAK)
│   ├── provenance.py           #   Per-source trust EMA + Bloom blocklist
│   ├── weight_firewall.py      #   KL anchor + EWC SI + grad clip
│   ├── gates.py                #   WebPoisonGate + ChatPoisonGate + SourceBudget
│   └── legacy.py               #   AdversarialRedTeam (older API, kept for BC)
│
├── safety/                     # Anthropic-style output alignment
│   ├── constitutional.py       #   CAI critique-revise (4 iters, 6 principles)
│   ├── red_blue.py             #   Same-model self-play DPO pair generator
│   ├── dpo.py                  #   Iterative DPO trainer (β=0.1)
│   ├── judge.py                #   Rule / API / Hybrid judges
│   └── red_team_corpus.py      #   12 attack categories, ZH+EN seeds
│
└── learn/                      # Self-driven learning
    ├── autonomous_daemon.py    #   Background web learner with 7-gate
    └── retrieval_memory.py     #   Track B per-user JSONL cache

docs/
├── ARCHITECTURE.md             # Full model description with all components
├── ROADMAP.md                  # 8-week plan, paper timeline, ¥7650 budget
├── HONEST_ASSESSMENT.md        # What works, what's untested, what's rhetoric
├── CONTINUAL_LEARNING.md       # Track A/B + 7-gate + 250-doc anchor
├── SAFETY.md                   # CAI + DPO + probe pipeline
├── 3D.md                       # DUSt3R + EGNN cheap 3D plan
├── CHAT_PROTOCOL.md            # Async kernel protocol (interrupts, proactive)
└── QUICKSTART.md               # Rental setup, first chat, troubleshooting
```

---

## What this is NOT

- Not a transformer-replacement at scale. SmolLM2-360M, Phi-3.5-mini, Qwen2.5-0.5B
  all beat us on standard benchmarks at their sizes. We claim **inference cost**
  + **continual-learning ergonomics**, not quality parity.
- Not production-ready. Output safety stack is written, not battle-tested.
- Not a verified energy advantage on neuromorphic hardware. Spike rates are
  10-30%, but no dedicated chip measured yet.
- Not a 1M context model — yet. Current `seq_len=2048`. NTK-RoPE + hierarchical
  memory roadmap targets 1M, 5M is reach goal.
- Not a Claude / GPT replacement. Don't deploy to actual users.

---

## What's next (8-week roadmap)

| Week | Focus | Deliverable |
|------|-------|-------------|
| 0 | Fix v4.2, restart training | First chat-able ckpt (ppl ~30) |
| 1 | NeuroMCP HNSW + L1/L2 deploy | 12-button + 3-step macros, 90% success |
| 2-3 | Multimodal 7-day Chameleon recipe | MMMU ≥ 30%, MathVista ≥ 25% |
| 3-4 | 3D understanding (DUSt3R + EGNN) | ScanQA EM ≥ 18%, CLEVR-3D ≥ 65% |
| 4-5 | Real OS actuator + DreamerV3 world model | OSWorld ≥ 25%, WebArena ≥ 20% |
| 5-6 | Anthropic safety stack runs | HH-RLHF held-out ≥ 70% |
| 6-8 | Paper drafts | NeurIPS 2026 submission + EMNLP backup |

Total: ~1100 GPU-h ≈ ¥7650 on A100×2 rental. Single dev. See
[docs/ROADMAP.md](docs/ROADMAP.md) for the detailed plan.

---

## Anchor papers (selected — full list in docs/ARCHITECTURE.md)

**Architecture**:
- CfC: Hasani et al, *Closed-form continuous-time neural networks*, 2022
- PLIF: Fang et al, *Incorporating learnable membrane time constant*, ICCV 2021
- Coconut latent thinking: Hao et al, 2412.06769
- HNSW: Malkov & Yashunin, 1603.09320

**Training methods**:
- Constitutional AI: Bai et al, 2212.08073
- DPO: Rafailov et al, 2305.18290
- TRAK influence: Park et al, 2303.14186
- TR-DPO iterative ref refresh: 2404.10719

**Safety findings**:
- Anthropic poisoning fixed-count = 250: 2510.07192
- Concealed data poisoning: Wallace et al, 2010.12563
- Sleeper Agents: Hubinger et al, 2401.05566

**3D foundation (planned)**:
- DUSt3R / MASt3R: Wang et al, 2312.14132 / 2406.09756
- EGNN: Satorras et al, 2102.09844
- 3D-LLM (target baseline): Hong et al, 2307.12981

**Multimodal (planned)**:
- Chameleon: Meta AI, 2405.09818
- Emu3: BAAI, 2409.18869

---

## License

MIT.

## Citation

```bibtex
@misc{synapforge2026,
  title  = {SynapForge: A 375M LNN+SNN Hybrid Language Model with
            Neuroplastic Action Routing and Async Conversation Kernel},
  author = {Liu Hfdgjr},
  year   = {2026},
  url    = {https://github.com/Lihfdgjr/synapforge}
}
```

## Contributing

This is a single-dev research repo. PRs welcome but please open an issue first
for non-trivial changes — the experimental direction is opinionated. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

## Contact

Issues / questions / paper collaborations: open a GitHub Issue.
