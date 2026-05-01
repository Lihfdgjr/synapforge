# Deep-maintenance task queue (deterministic)

> Cron agents work top → bottom. NO RANDOM. When a task is `[x]`, skip it. When `[ ]`, do it.
> Each cron fire MUST advance ≥ 3 tasks AND spawn ≥ 1 Agent call. No "all clear, nothing to do" exits.

## How agents update this file

After completing a task (or proving it's blocked), edit the row:
- `[ ]` → `[x] (HH:MM, agent-summary)` for done
- `[ ]` → `[/] (HH:MM, blocked: <reason>)` for blocked
- `[ ]` → `[~] (HH:MM, deferred to <phase/condition>)` for deferred

When a task spawns an Agent worktree commit, append the commit hash. When it just runs an SSH command, log to `docs/PROGRESS.md` §3.

## Always-on health (every fire — does NOT count toward task budget)

- `H1` ssh check trainer alive + GPU util + last 3 step lines + last 3 VAL ppl
- `H2` disk Use% on rental — auto-prune step ckpts if ≥ 80% (keep last 5 + step_002000 + best_*)
- `H3` mohuanfang ckpt count vs rental count — manual rsync if drift > 2
- `H4` if trainer dead → relaunch from latest healthy step_*.pt with shuffle + kd-topk + grad-clip 0.5
- `H5` phase trigger: val ≤ 250 → phase 1 / val ≤ 100 → phase 2 / val ≤ 60 → phase 3 / chat ≥ 60% → phase 4

## Queue (advance top-down, ≥ 3 per fire, ≥ 1 Agent spawn per fire)

### Tier 1 — Quality validation (do immediately when ckpt available)

- [ ] **T1.1** Real chat sample on latest ckpt (5 EN + 5 ZH). Save verbatim output to `docs/CHAT_SAMPLES.md` with timestamp. NEVER fake.
- [ ] **T1.2** Light bench: mmlu_full + hellaswag_full + lambada via auto_eval_daemon. Pull score → `docs/PROGRESS.md` table.
- [ ] **T1.3** Adversarial probe: 10 edge prompts (refusal / 5K-token long / multi-lang switch / repetition trap). Detect crashes / NaN / loops.
- [ ] **T1.4** STDP inference-time real demo: `synapforge-demo stdp --trials 1000 --hidden 256`. Compare to recorded 27% density baseline.
- [ ] **T1.5** NeuroMCP 6000-trial extended: pytest `test_neuromcp_long_horizon.py --run-slow -k 6000`. Record (density, K, hit_rate) → INVESTOR.md.
- [ ] **T1.6** R-fold real GPU bench: `synapforge-demo bench` on rental. Compare to recorded 2.99× peak. → `docs/RFOLD_PAPER.md` Appendix.
- [ ] **T1.7** 50M context harness: `tests/integration/test_long_context_50m.py --run-slow` for 1K + 10K (skip ≥100K to keep CPU fire under 10 min).
- [ ] **T1.8** Quality-monotonic A/B: `test_long_context_monotonic_quality.py --run-slow`. STDP on/off at 1K/10K/100K.
- [ ] **T1.9** Inference latency profile: first-token / TTFT / per-step on A800. → `docs/PERF_KNOBS.md`.
- [ ] **T1.10** VRAM usage timeline within one step: `torch.profiler` + memory_summary. → docs.

### Tier 2 — Architecture research + ship (Agent-spawn each)

- [ ] **T2.1** ArXiv scan WebFetch → `docs/ARCHIVE_NEW_PAPERS.md` (keywords: ternary / matmul-free / spiking / liquid / STDP / SNN-LM / continual-learning). Last 7 days.
- [ ] **T2.2** Triton fused PLIF backward kernel — Agent: implement stub at `synapforge/backends/triton_fused_backward.py` (already stubbed commit `5a3ecef`). Add forward+backward path.
- [ ] **T2.3** Surrogate gradient annealing — Agent: anneal width 10 → 1 over 5000 steps. CLI flag `--surrogate-anneal-steps`. Address P25 PLIF dead.
- [ ] **T2.4** Frozen vocab tail mask — Agent: pad rows 151643-151935 of `tok_embed.weight` + `lm_head.weight` to require_grad=False. Address P26.
- [ ] **T2.5** Spike-rate-target loss term — Agent: penalty if rate < 0.05 or > 0.20. CLI flag `--spike-target-loss-weight`. Address P25.
- [ ] **T2.6** LM head spectral norm — Agent: `nn.utils.spectral_norm` on final projection. Address P28 z-loss drift.
- [ ] **T2.7** Adaptive grad accum — Agent: bs_eff=128 via 2× accum at bs=64. Address P29.
- [ ] **T2.8** Ternary CfC weights M1 — Agent: enable `synapforge/quantize.py::AbsMeanTernary` on CfC `W` only, fp activations. Behind `--quant-cfc-weights ternary` flag.
- [ ] **T2.9** Coconut latent thinking activate test — Agent: enable `--latent-k 8` and verify forward path doesn't NaN.
- [ ] **T2.10** MoE chain-of-experts (adv28 #1) — verify code path live, not orphan.
- [ ] **T2.11** torch.compile reduce-overhead real timing — Agent: A/B with/without on rental, log tok/s.
- [ ] **T2.12** FP8 / int8 inference path — Agent: research only (A800 has TF32 not FP8). Document in `docs/quantize.md`.

### Tier 3 — Data pipeline (Agent-spawn for synthesizers)

- [ ] **T3.1** Synth ZH pretrain to 500K rows — Agent: extend `synth_chinese_pretrain.py` to 500K, run on rental CPU cores 4-7.
- [ ] **T3.2** Image synthetic data generator (8x8 BPE patches) — Agent: write `scripts/synth_image_pretrain.py` outputting parquet of (token_ids, image_patches).
- [ ] **T3.3** Audio synthetic data generator (mel spectrogram patches) — Agent: write `scripts/synth_audio_pretrain.py`.
- [ ] **T3.4** Time-series synth — Agent: stock + sensor + bio signals as token sequences.
- [ ] **T3.5** GSM8K math chains tokenized — Agent: download GSM8K subset, tokenize for phase 4 RL.
- [ ] **T3.6** Mohuanfang warehouse activate — `bash scripts/setup_mohuanfang_warehouse.sh` real run; verify symlink + first lazy fetch.
- [ ] **T3.7** Pre-tokenize wikitext-103 (rerun, fix earlier "wt103 files: 0" bug — check actual data path).
- [ ] **T3.8** HumanEval / MBPP code data — Agent: download + tokenize for code-ability training mix.
- [ ] **T3.9** ARC-Easy / ARC-Challenge — Agent: download + tokenize for reasoning eval.
- [ ] **T3.10** SWE-bench mini subset — Agent: tokenize 50 issues for code-fix evaluation.

### Tier 4 — Backup + storage (mostly H3 territory; periodic deep)

- [ ] **T4.1** GH Release 备份 — manually upload step_002000.pt as anchor (`gh release create synap-1-anchor` + upload).
- [ ] **T4.2** HF Hub backup setup — research auth + upload script (`huggingface_hub` CLI). Document in `docs/BACKUP.md`.
- [ ] **T4.3** Mohuanfang warehouse first-cycle test — fetch a shard via `RemoteDataWarehouse.get_shard()` to verify rsync works.
- [ ] **T4.4** Disk usage analytics — daily du -sh per dir, plot growth, predict overflow date.

### Tier 5 — Training observability

- [ ] **T5.1** Loss component breakdown — log CE / KD / z-loss per-step contribution. Append to PROGRESS.md.
- [ ] **T5.2** Spike rate per layer (10 layers) — currently aggregated to mean. Split per-layer into log.
- [ ] **T5.3** Gradient norm histogram per-layer.
- [ ] **T5.4** Best ckpt selector by val_ppl_holdout — auto rename to `best_step_*.pt`. Hot-spare for warmstart.
- [ ] **T5.5** Train/val curves matplotlib — daily plot to `docs/CURVES_<date>.png`.
- [ ] **T5.6** Throughput timeline — tok/s every 100 steps, detect dips.

### Tier 6 — Investor / paper artifacts

- [ ] **T6.1** Demo video script — `docs/DEMO_VIDEO_SCRIPT.md` 3-min screencast outline.
- [ ] **T6.2** Paper draft section 4 (results) — auto-fill train curves + real benchmarks once available.
- [ ] **T6.3** README badges — CI status, last-commit, version, coverage.
- [ ] **T6.4** GitHub Issue auto-open per discovered bug — Agent: gh issue create per OOM/divergence root-cause, link to commit.
- [ ] **T6.5** CHANGELOG auto-append per phase transition.
- [ ] **T6.6** Tweet draft per milestone (chat-grade reached / 50M context working / matmul-free shipped).
- [ ] **T6.7** Compare to SmolLM2-360M real numbers via HuggingFace — pull official model + run on our val set.
- [ ] **T6.8** Chinese chat quality external rate — Claude API to rate 10 Synap-1 outputs vs Qwen 0.5B baseline.

### Tier 7 — Self-improvement

- [ ] **T7.1** Doc stamp refresh — `python scripts/check_doc_stamps.py --update-stamps`.
- [ ] **T7.2** Memory entries for durable session-discoveries — to `feedback_*.md` + index.
- [ ] **T7.3** Advance 1 P# per fire — read MASTER_PLAN §6, pick top pending, ship fix or mark deferred.
- [ ] **T7.4** Test coverage report — `pytest --cov` to see untested code paths.
- [ ] **T7.5** Python deps audit — `pip list --outdated` log key library versions.
- [ ] **T7.6** Network ping audit — github.com / huggingface.co / mohuanfang.com latency. If github > 5s timeout, prefer mohuanfang for git-relay.

### Tier 8 — Advanced features (post-phase-1+)

- [ ] **T8.1** Inference STDP weight diff — measure ‖ΔW_STDP‖ during 1K-token chat. Verify monotonic increase.
- [ ] **T8.2** Continual learning real test — feed a novel domain (e.g., 5K tokens medical), then test medical Q&A.
- [ ] **T8.3** Curriculum learning — easy → hard sort training data by perplexity.
- [ ] **T8.4** EMA weights at inference — exponential moving average for stable generation.
- [ ] **T8.5** Long sequence inference — 16K context, real generation, measure quality.
- [ ] **T8.6** Self-distillation — Synap-1-Pro 300M as teacher to Synap-1 100M v2.
- [ ] **T8.7** Multi-rental DDP — 2× A800 sync, document setup.
- [ ] **T8.8** Triton kernel autotune — block size / warps sweep.

## Rules

1. **Top-down order**. Skip `[x]` and `[/]`, do top `[ ]`.
2. **Each fire ≥ 3 advances + ≥ 1 Agent.spawn**. Even if all blocked, mark them `[/]` with reason — that counts.
3. **No randomness**. Reproducible across fires.
4. **Commit + push** at end. Message: `auto: tier-N done T1.1+T1.4+T2.3 (next: T2.5)`.
5. **Spawn agents** for any task that says "Agent:" — they're free per user opt-in. Use `general-purpose` + `isolation: worktree`.
6. **STOP** when chat eval ≥ 0.6 OR phase 4 RL running. CronDelete both crons + ScheduleWakeup user.
