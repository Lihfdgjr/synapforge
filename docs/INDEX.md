# SynapForge Documentation Index

This is the single source of truth for navigating the `docs/` tree. Each entry
below has a one-line summary, the most useful cross-references, and the file's
last-modified date so you can see at a glance when it was touched.

If you only read one thing first, read **[QUICKSTART](QUICKSTART.md)**.
If you want the warts-and-all picture, read **[HONEST_ASSESSMENT](HONEST_ASSESSMENT.md)**.

---

## Just want to use it

| Doc | Summary | Updated |
|-----|---------|---------|
| [QUICKSTART.md](QUICKSTART.md) | Three paths in 30s / 5min / 1h. CPU smoke, then full setup, then rental + training. | 2026-05-01 |
| [INVESTOR.md](INVESTOR.md) | The 30-second pitch + reproducible artifacts (`synapforge-demo all` runs in <2s on CPU). Cross-refs ARCHITECTURE, RFOLD_PAPER, NEUROMCP_UNIVERSAL. | 2026-05-01 |

## Want to train it

| Doc | Summary | Updated |
|-----|---------|---------|
| [ROADMAP.md](ROADMAP.md) | 8-week paper roadmap. Per-week deliverables, GPU-h budget (¥7650 total). | 2026-05-01 |
| [PHASE_TRAINING.md](PHASE_TRAINING.md) | Phased trainer: Phase 0 LM-only KD -> 1 intrinsic+self_learn+STDP -> 2 modal byte-patch -> 3 SFT alpaca -> 4 RL GRPO. ppl gates between phases. Cross-refs PARALLELISM. | 2026-05-01 |
| [MULTIMODAL_TRAINING.md](MULTIMODAL_TRAINING.md) | 9-modality byte-patch trainer recipe. Encoders + losses for image/audio/video/3D/text. | 2026-05-01 |
| [NEURAL_COMPUTER_USE.md](NEURAL_COMPUTER_USE.md) | ActionHead -> OSActuator pipeline. Neurons emit action vectors, no JSON tool calls. GUI-act dataset + STDP self-organization. | 2026-05-01 |
| [PHASE_RELAUNCH.md](PHASE_RELAUNCH.md) | (planned) Per-phase relaunch playbook for after `phase_signal` flips. Until then see PHASE_TRAINING + RENTAL_RECOVERY. | not yet written |

## Want to verify our claims

| Doc | Summary | Updated |
|-----|---------|---------|
| [BENCHMARKS.md](BENCHMARKS.md) | Six paper-grade benches (HumanEval, MBPP, MMLU, GSM8K, HellaSwag, LAMBADA). Public baselines + sandbox + how to add a new one. | 2026-05-01 |
| [AUTO_EVAL.md](AUTO_EVAL.md) | Per-checkpoint auto-eval daemon. What gets dumped, how to read curves, fast/heavy split. Cross-refs BENCHMARKS + HONEST_ASSESSMENT. | 2026-05-01 |
| [RFOLD_PAPER.md](RFOLD_PAPER.md) | R-fold algebraic CfC closed-form. Math derivation + honest GPU peak 2.99x at N=64 R=16. CPU correctness table. Appendix B retracts an earlier 167x extrapolation. | 2026-05-01 |
| [HONEST_ASSESSMENT.md](HONEST_ASSESSMENT.md) | What works (verified), what doesn't yet, what's still rhetoric. Updated each major version. | 2026-05-01 |
| [REVIEW_DEMO.md](REVIEW_DEMO.md) | Code review of `synapforge.demo` + R-fold bench + parallel helpers. Bug list + investor-flow walkthrough + top polish items. | 2026-05-01 |
| [REVIEW_MULTIMODAL.md](REVIEW_MULTIMODAL.md) | (planned) Code review of `synapforge.modal`. See MULTIMODAL_TRAINING + the encoders directly until written. | not yet written |

## Reliability + ops

| Doc | Summary | Updated |
|-----|---------|---------|
| [RELIABILITY.md](RELIABILITY.md) | Failure modes + on-call playbook (training hangs, ckpt corruption, SSH death, plateau detection). | 2026-05-01 |
| [BACKUP.md](BACKUP.md) | Triple-path backup (`triple_backup_daemon.py`): mohuanfang rsync + GitHub release + HF dataset. Recovery commands. | 2026-05-01 |
| [RENTAL_RECOVERY.md](RENTAL_RECOVERY.md) | Recovering after a rental SSH dies. Concrete steps: pull from GH, restore ckpt, restart phase manager. | 2026-05-01 |

## Architecture deep-dive

| Doc | Summary | Updated |
|-----|---------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full block diagram. HybridBlock (CfC + PLIF + SwiGLU + RMSNorm) x 14 layers. RoPE, tied LM head, NeuroMCP head. | 2026-04-30 |
| [CHAT_PROTOCOL.md](CHAT_PROTOCOL.md) | Async event-driven chat kernel. Turn-taking, interrupts, proactive emissions. State machine. | 2026-04-30 |
| [CONTEXT_SCALING.md](CONTEXT_SCALING.md) | 1K -> 1M -> ideally 100M context. NTK-RoPE + InfLLM + Memory^3 + Titans. A100 80GB physical limit analysis. | 2026-04-30 |
| [MONOTONIC_QUALITY.md](MONOTONIC_QUALITY.md) | Why long-context quality should rise (not drop). STDP retrieval + multi-band tau + BM25 sidecar. | 2026-05-01 |
| [PARETO_OPTIMIZATION.md](PARETO_OPTIMIZATION.md) | Joint accuracy^ + memory v target. PQ16 hidden-state compression, sparse activations. | 2026-05-01 |
| [PARALLELISM.md](PARALLELISM.md) | 3-layer parallel stack: CPU thread tune + CPU/GPU mixed placement + auto gloo/nccl DDP. | 2026-05-01 |
| [NEUROMCP_UNIVERSAL.md](NEUROMCP_UNIVERSAL.md) | NeuroMCP universal action protocol. Synapse growth + dynamic codebook + HNSW skill index. Replaces JSON MCP. | 2026-05-01 |
| [3D.md](3D.md) | 3D understanding via DUSt3R + EGNN. Cheap recipe. ScanQA / 3DSRBench targets. | 2026-04-30 |
| [3D_PLAN.md](3D_PLAN.md) | The actual deployment plan with GPU-h budget (140 GPU-h, ~Y980). Prerequisites + risks. | 2026-05-01 |

## Code reviews

| Doc | Summary | Updated |
|-----|---------|---------|
| [REVIEW_TRAINER.md](REVIEW_TRAINER.md) | Code review of `train_v42_universal.py` + `train_100m_kd.py` + mixins. Bug-classes, perf hot paths. | 2026-05-01 |
| [REVIEW_NEUROMCP.md](REVIEW_NEUROMCP.md) | Code review of `synapforge/action/`. Skill log, codebooks, ActionHead, actuator. | 2026-05-01 |

## Continual learning + safety

| Doc | Summary | Updated |
|-----|---------|---------|
| [CONTINUAL_LEARNING.md](CONTINUAL_LEARNING.md) | Track A (web -> weights via 7 gates) + Track B (chat -> retrieval cache). 250-doc Anthropic poison anchor. | 2026-04-30 |
| [SAFETY.md](SAFETY.md) | Output safety overview. CAI + DPO + judge + probe. | 2026-04-30 |
| [SAFETY_PLAN.md](SAFETY_PLAN.md) | The 4-stage Anthropic-style safety stack with timelines + failure cases. | 2026-05-01 |

## Misc

| Doc | Summary | Updated |
|-----|---------|---------|
| [quantize.md](quantize.md) | int8 / fp4 / ternary quantization recipes. BitNet b1.58 reference. | 2026-04-26 |

---

## Reading orders by audience

**New collaborator, 30 minutes**: QUICKSTART -> ARCHITECTURE -> HONEST_ASSESSMENT -> ROADMAP.

**Reviewer of the paper draft**: RFOLD_PAPER -> BENCHMARKS -> HONEST_ASSESSMENT -> NEUROMCP_UNIVERSAL.

**Investor / non-technical reader**: INVESTOR -> QUICKSTART (path 1 only) -> HONEST_ASSESSMENT (`Doesn't work yet` section).

**On-call when training breaks**: RELIABILITY -> RENTAL_RECOVERY -> BACKUP.

**Implementing a new phase**: PHASE_TRAINING -> REVIEW_TRAINER -> AUTO_EVAL.

**Hardening safety**: SAFETY_PLAN -> SAFETY -> CONTINUAL_LEARNING.

---

## Cross-refs (where to find which thing)

The file you want is rarely the only place a topic is discussed. This map shows
the *primary* doc for each concept; secondary mentions are usually in the
papers / READMEs themselves.

| Concept | Primary doc | Secondary |
|---------|-------------|-----------|
| 7-round discussion | (synap-ide-vscode CLAUDE.md, not synapforge) | n/a |
| CfC + PLIF block | ARCHITECTURE | RFOLD_PAPER, REVIEW_TRAINER |
| Coconut latent thinking | ARCHITECTURE (final section) | PHASE_TRAINING (phase 1 unlocks k>1) |
| NeuroMCP / action vectors | NEUROMCP_UNIVERSAL | NEURAL_COMPUTER_USE, REVIEW_NEUROMCP, INVESTOR |
| R-fold | RFOLD_PAPER | INVESTOR (the 2.99x peak speedup), HONEST_ASSESSMENT (full CPU/GPU table) |
| 7-gate Track A | CONTINUAL_LEARNING | SAFETY_PLAN |
| Chat memory (Track B) | CONTINUAL_LEARNING | CHAT_PROTOCOL |
| Triple-backup daemon | BACKUP | RELIABILITY (recovery flow) |
| Auto-eval daemon | AUTO_EVAL | BENCHMARKS, HONEST_ASSESSMENT |
| Phase signal | PHASE_TRAINING | REVIEW_TRAINER |
| 1M+ context | CONTEXT_SCALING | MONOTONIC_QUALITY, PARETO_OPTIMIZATION |
| Multi-modality | MULTIMODAL_TRAINING | ARCHITECTURE (modality heads) |
| 3D understanding | 3D | 3D_PLAN |
| Distributed / heterogeneous | PARALLELISM | REVIEW_TRAINER |
| BitNet b1.58 ternary | quantize | PHASE_TRAINING (phase ?) |

---

## Maintaining this index

This file is hand-written, but the `Updated` column is read from the disk mtime
at the time of the last edit. When you ship a docs change, please:

1. Update the relevant row's date column.
2. If you add a new doc, add a row in the right section.
3. If you delete or rename a doc, update both the row and any cross-refs.

A regression test that walks `docs/*.md` and warns about un-indexed files would
be an obvious next step but isn't shipped yet.
