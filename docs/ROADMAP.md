# Roadmap

8-week plan to ship a working chat model + 1 paper draft + 1 backup paper queued.

## Week 0 (now): fix v4.2, restart training

| Day | Task | Outcome |
|-----|------|---------|
| 0 | **CPU pilot** for inference-STDP (`scripts/cpu_pilot_inference_stdp.py`) | Validate monotonic-quality claim before burning 12.5h on A100 |
| 0 | **R-fold k=8** (`synapforge/cells/rfold.py`) wired into Coconut latent loop | 2.7× free, no quality loss, no extra params |
| 0 | Push 3 v4.2 patches: HF-repo path check, top-1 NeuroMCP routing, encode/lm_logits split | Trainer launches without errors |
| 0 | Restart v4.2 with `--bs 8 --seq-len 1024 --grad-accum 2` | tok/s ~7000 (6× recovery from 1100) |
| 1-2 | v4.2 run to step 30k | First chat-able ckpt, ppl ~35 |
| 3 | One round of SFT (LIMA-30 + Alpaca-zh + GSM8K) | demo level chat, pp ~25 |

**SSH-down contingency** (per agent synthesis 2026-05-01): if rental sshd is
saturated, the highest-leverage move is the 算力牛 web console (Jupyter/noVNC
runs independent of sshd). If that fails too, force-reboot the instance (~90s).
Vast.ai spot A100 80GB ¥48/12.5h is the next fallback; full retrain from v4.0
ckpt would cost ¥77 + lose v4.1 (best ppl 44.2). Always run CPU pilot in
parallel — it de-risks the GPU spend regardless.

## Week 1: NeuroMCP enhancements + multimodal data prep

| Day | Task | Notes |
|-----|------|-------|
| 4 | Deploy HNSW skill index | Unblocks K → 100k. Code already written. |
| 5-7 | L1/L2 compositional codebook (already scaffolded) | Hebbian co-firing → L2 compound minting. Test on 4-button env. Target: 12-button + 3-step macros 90% success. |
| 5-7 | Download multimodal data: CC12M + LAION-COCO 5M + OBELICS 2M + LibriSpeech + AudioCaps | Use hf-mirror, ~80GB |
| 7 | Train VQ-GAN tokenizer (8K codebook) on CC12M | 12h on 1×A100 |

## Week 2-3: multimodal training (Chameleon recipe)

7-day end-to-end multimodal training, 336 GPU-h on A100×2.

- Phase 0 (12h): VQ-GAN train
- Phase 1-2 (5d): warmstart from v4.2, vocab 32K→45K, freeze old text rows 2K steps
- Phase 3 (8h SFT): LLaVA-Instruct-150K + M3IT 50K + AudioCaps QA
- Phase 4 (eval): MMMU / MathVista / AudioBench / VideoMME

Eval gates (must hit, else rollback):
- WikiText ppl ≤ 1.10× v4.2 baseline (forgetting check)
- MMMU ≥ 30%, MathVista ≥ 25%, AudioBench-MMAU ≥ 35%, VideoMME ≥ 30%
- **Anti-fakery**: zero out image embeddings → caption quality must collapse <50%

## Week 3-4: 3D understanding (DUSt3R + EGNN, 140 GPU-h ¥980)

Cheapest viable native 3D path. See [docs/3D.md](3D.md).

- 80GB free data: Habitat 20K trajectories + Objaverse 50K rendered + DUSt3R-pseudo-labeled ImageNet 200K + CLEVR-3D 50K
- +25M params: Plücker ray embedding (2M) + EGNN equivariant adapter (8M) + Q-K-norm fixes (15M)
- Loss: 0.4 LM + 0.3 pointmap_recon + 0.2 view_consistency + 0.1 QA
- Phase 0 (10h): freeze base, train EGNN adapter
- Phase 1 (60h): unfreeze last 4 CfC blocks
- Phase 2 (50h): full unfreeze + CLEVR-3D + ScanQA mix
- Phase 3 (20h): SFT on ScanQA train + 3DSRBench

Eval gates:
- Pointmap MSE < 0.15 on Habitat val (DUSt3R teacher achieves 0.08)
- CLEVR-3D spatial accuracy ≥ 65%
- ScanQA val EM ≥ 18% (3D-LLM baseline: 20.4% at 3B)
- WikiText ppl regression < 8% (forgetting check)

## Week 4-5: real OS actuator + DreamerV3 world model

For OSWorld / WebArena empirical numbers (paper headline).

- ~400 GPU-h, 3-4 weeks elapsed
- Anthropic Computer Use API integration with our `ActionHead`
- Playwright for web actuation
- DreamerV3 (2301.04104) latent world model on (screenshot, action, next_screenshot) tuples
- Offline planning rollouts before committing real actions

Eval gates:
- OSWorld ≥ 25% (Claude 3.5 baseline 14.9%, CoAct-1 60.76% with full LLM scaffold)
- WebArena ≥ 20%
- World-model planning lift ≥ +5pp over reactive policy

## Week 5-6: Anthropic safety stack runs

- SFT refusal (HH-RLHF subset, 2h)
- CAI SL-CAI critique-revise 4-iter (4h)
- Red-Blue DPO self-play 3k pairs (6h)
- Hidden-state safety probe (30min)

Eval: HH-RLHF held-out preference ≥ 70%, MMLU regression ≤ 2pt.

## Week 6-8: paper drafts

### Primary paper: NeuroMCP (NeurIPS 2026, May deadline)
**Title**: *NeuroMCP: Token-Free Tool Use via Sparse Synaptic Routing and Growing Prototype Codebooks*

Core contribution: replace JSON tool-calling with neural action emission via
density-adaptive synaptic layer + co-activation-grown prototype codebook +
persistent cross-session skill memory.

Closest prior: Toolformer (2302.04761), Voyager (2305.16291), Gato (2205.06175),
Neural Turing Machines (1410.5401). No direct competitor combines sparse
synaptic growth + dynamic codebook + LTP/LTD persistent memory for tool use.

Required experiments:
- 4-button (have it, 100%)
- 12-button + 3-step macros (need L1/L2 compositional)
- **OSWorld** (need real actuator + 400 GPU-h)
- **Mind2Web** or AgentBench
- Ablation: Sparse vs Dense vs Codebook-only vs Full
- Skill retention curve over 10 sessions with LTD decay
- Token-budget vs latency vs success-rate vs ReAct baseline

Risks: "4-button is toy, OSWorld unverified" — must run OSWorld before submit.

### Backup paper: SynapForge-375M LNN+SNN benchmark (EMNLP 2026)
**Title**: *SynapForge-375M: A Liquid-Spiking Hybrid Beats Transformer Baselines at Equal Parameter Count*

First public benchmark of CfC + PLIF hybrid trained at 375M scale on multilingual
chat. Closest prior: SpikeGPT (2302.13939, encoder-only), SpikingSSMs (2401.16808),
LiquidS4 (2209.12951), Liquid Foundation Models (no public LM bench).

Required experiments:
- GPT-2 / Pythia / SmolLM2 baselines at 375M
- WikiText103, Lambada, HellaSwag perplexity
- Energy / FLOP comparison
- Ablate CfC-only vs PLIF-only vs hybrid

Risk: SmolLM2-360M reaches WikiText ppl ~20; we currently sit at ~50. Either
train longer (200 GPU-h) or lead with energy/inference advantages instead of
quality parity.

## Reasoning length penalty (ALP, arXiv 2506.05256)

After v4.2 has stable chat ckpt, add **Adaptive Length Penalty** RL on top of
Coconut latent thinking. Reward formula:

```
r(y, q) = 1[answer(y) = y*] − β · N · max(p_solve(q), 1/K)
```

- `N` = num Coconut thinking steps + 0.5 × num NeuroMCP tool calls
- `p_solve(q)` = empirical solve rate over K=8 GRPO rollouts
- Easy prompts → steep length penalty (force shortcut)
- Hard prompts → near-zero penalty (let it think)
- **GRPO-λ guard** (2505.18086): only penalize when group accuracy ≥ 0.4

Curriculum: `β = 0` until base `p_solve > 0.3` on held-out set; then ramp.
Premature length pressure collapses small models (warning from 2602.09591).

Reach goal: model learns "use as few thinking steps as possible while staying
correct." Effective on Coconut because k=1→8 is exactly the cost axis being
penalized.

Estimated cost: 60 GPU-h on top of base v4.2.

## Async chat kernel deployment

The `synapforge.chat` kernel (event_loop + interrupt_policy + streaming +
turn_taking + proactive) replaces synchronous request-response. Production
chat needs:

- WebSocket / SSE bridge from web UI to `kernel.send_user_*` / `outbox_stream`
- Per-keystroke partial-message support (already in kernel)
- Mute button → `kernel.send_mute(seconds)`
- Cron schedule UI → `kernel.proactive.cron.schedule(fire_at_ts, message)`

Estimated cost: 20 GPU-h for end-to-end testing with real users.

## Total resource estimate

| Item | GPU-h | Cost (¥7/h A100×2) |
|------|-------|-------------------|
| v4.2 fix + restart | 25 | 175 |
| L1/L2 codebook ablation | 80 | 560 |
| Multimodal 7-day | 336 | 2350 |
| 3D DUSt3R+EGNN | 140 | 980 |
| OSWorld actuator + DreamerV3 | 400 | 2800 |
| Anthropic safety | 12 | 85 |
| ALP reasoning RL (2506.05256) | 60 | 420 |
| Async chat kernel test runs | 20 | 140 |
| Misc eval / re-runs | 100 | 700 |
| **Total** | **1180** | **¥8210** |

8 weeks elapsed. Single dev. Ship NeurIPS submission + EMNLP backup.

## What's deprioritized

- BitNet b1.58 ternary QAT — saves memory, doesn't move the paper needle. Keep as
  optional flag for inference-time demo.
- Coconut k=4 deep latent thinking — added k=1 as flag, deeper untested. Bench gate:
  GSM8K +5pp vs k=1 baseline. If yes, ramp; if no, drop.
- WaveFormer1D wave-PDE — interesting but not novel enough vs Hyena/FNet to make
  a paper section. Cut.
- Sleeper Agents defenses — too research-y for our budget. Add probe but don't claim.
