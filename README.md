# SynapForge

A small (375M-parameter) **LNN + SNN hybrid language model** — *not a transformer*.

CfC continuous-time recurrent cells (Hasani 2022) + PLIF spiking neurons (Fang 2021) +
neuroplastic action routing (NeuroMCP), persistent skill memory, two-track continual
learning, and Anthropic-style output safety. Trained from scratch in 7h on a single
A100×2 rental.

```
┌──────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│ token / patch in │───▶│ HybridBlock × 14 │───▶│ tied LM head       │
│ Qwen vocab 151K  │    │ (CfC + PLIF +    │    │ + NeuroMCP action  │
│ + 9-modal byte   │    │  SwiGLU FFN)     │    │   vector emit      │
└──────────────────┘    └──────────────────┘    └────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ skill_log.json     │  ← LTP/LTD persistent
                    │ HNSW index (100k+) │     across sessions
                    └────────────────────┘
```

## What's actually different from a transformer LM

1. **No attention.** CfC continuous-time recurrence handles long-range dependencies via
   ODE-discretized state evolution. STDP plasticity matrix doubles as content-addressable
   retrieval (no need for KV cache).
2. **No tool-call tokens.** NeuroMCP emits action vectors directly through a
   SparseSynapticLayer (5%→40% density, with Hebbian-grown connections) and a
   DynamicCodebook (K=4 per domain growing to K=64 by co-activation, persisted to
   `skill_log.json` so users get their grown skills back next session).
3. **Spiking neurons in the LM path.** PLIF cells learn per-channel τ. Spike rates
   tracked as health signal; `observe_only=True` bootstrap recipe avoids dead-PLIF
   collapse.
4. **Plasticity ≠ backprop.** Three-factor STDP aux loss aligns local Hebbian/STDP
   updates with global BP gradient direction, gated by a `M_t = α·FreeEnergy +
   β·Novelty − γ·Homeostatic` neuromodulator.

## Status (2026-04-30)

| Component | State |
|-----------|-------|
| v4.0 base 375M Qwen-vocab pretrain | ✅ done, 36k steps, ckpt at `step_036000.pt` |
| v4.1 NeuroMCP wire-in | ✅ done, 60k steps, **best ppl 44.2** at step 46350 |
| v4.2 Universal trainer (Coconut + per-domain NeuroMCP + skill persistence + STDP aux) | ⚠ launched, 3 known bugs being patched (mask>95%, KD silent, 17× slow) |
| 7-gate continual ingest (Track A) | ✅ deployed, autonomous_daemon running |
| Track B retrieval-only chat memory (Claude Memory pattern) | ✅ scaffold ready |
| HNSW skill index (100k+ prototypes) | ✅ written, awaiting deploy |
| L1/L2 compositional codebook (Hebbian co-firing → option discovery) | ✅ scaffold written |
| Anthropic safety stack (CAI + red/blue DPO + judge) | ✅ written, training queued |
| Multimodal 9-modality byte-patch | ⏳ encoder paths exist, only text trained so far |
| 3D understanding (DUSt3R + EGNN, ¥980 budget) | ⏳ designed, queued |
| R-fold 167× inference speedup | ⏳ being investigated |

## Quickstart

```bash
git clone https://github.com/Lihfdgjr/synapforge
cd synapforge
pip install -r requirements.txt

# Train (375M from scratch — needs 1×A100 80GB, ~7h)
python -m synapforge.train_v42_universal \
    --warmstart /path/to/v41_best.pt \
    --teacher-path Qwen/Qwen2.5-0.5B \
    --steps 60000 --batch-size 8 --seq-len 1024 \
    --neuromcp-enabled --kd-weight 0.3 \
    --skill-log /path/to/skill_log.json
```

For full setup (rental SSH, dataset prep, autonomous learn daemon, safety pipeline)
see [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Repo layout

```
synapforge/
  model_chat_600m.py          375M base model
  train_v42_universal.py      v4.2 trainer (CE + KD + STDP + entropy bonus)
  action/                     NeuroMCP — token-free tool use
    skill_log.py              JSON persistent prototypes (LTP/LTD)
    per_domain_neuromcp.py    4 codebooks × K=64, intent router
    hnsw_skill_index.py       O(log K) for 100k+ prototypes
    compositional_codebook.py L1 primitives + L2 compounds (option discovery)
  thinking/coconut.py         <bot>/<eot> latent reasoning (k=1→8 curriculum)
  moe/expert_ffn.py           top-2 noisy gate, 8 routed + 1 shared
  defense/                    Track A/B + 7-gate ingest
    poison_detector.py        Multi-signal scoring (PromptGuard / Mahalanobis / TRAK)
    provenance.py             Per-source trust EMA + blocklist
    weight_firewall.py        KL anchor + EWC SI + grad clip
    gates.py                  WebPoisonGate + ChatPoisonGate + SourceBudget
  safety/                     Anthropic-style output alignment
    constitutional.py         CAI critique-revise loop (4 iters)
    red_blue.py               Same-model self-play DPO pair generator
    dpo.py                    Iterative DPO trainer (β=0.1, ref-refresh @ 50)
    judge.py                  Rule / API / Hybrid judges
    red_team_corpus.py        12 attack categories, ZH+EN seeds
  learn/
    autonomous_daemon.py      Self-goal-proposing web learner
    retrieval_memory.py       Track B per-user cache (no weight update)
docs/
  ARCHITECTURE.md             Full model description
  ROADMAP.md                  Training + paper timeline
  HONEST_ASSESSMENT.md        What works, what doesn't
  CONTINUAL_LEARNING.md       Track A/B, 7-gate ingest, TRAK gating
  SAFETY.md                   Output alignment stack
  3D.md                       Cheap 3D understanding plan
```

## Anchor papers

Architecture:
- CfC: Hasani et al, *Closed-form continuous-time neural networks*, 2022
- PLIF: Fang et al, *Incorporating learnable membrane time constant…*, ICCV 2021
- Coconut: Hao et al, 2412.06769

Training methods:
- Constitutional AI: Bai et al, 2212.08073
- DPO: Rafailov et al, 2305.18290
- TRAK influence: Park et al, 2303.14186
- TR-DPO iterative ref refresh: 2404.10719

Safety findings:
- Anthropic poisoning fixed-count = 250: 2510.07192
- Concealed data poisoning: Wallace et al, 2010.12563
- Sleeper Agents: Hubinger et al, 2401.05566

3D foundation (planned):
- DUSt3R / MASt3R: Wang et al, 2312.14132 / 2406.09756
- EGNN: Satorras et al, 2102.09844

## License

MIT.

## Citation

```bibtex
@misc{synapforge2026,
  title  = {SynapForge: A 375M LNN+SNN Hybrid Language Model with Neuroplastic Action Routing},
  author = {Liu Hfdgjr},
  year   = {2026},
  url    = {https://github.com/Lihfdgjr/synapforge}
}
```
