# Changelog

All notable changes to **synapforge** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The 2026-04-29 → 2026-05-01 burst landed roughly ten releases' worth of code
in three days; this file is the canonical chronology of what shipped and why.
Entries are grouped by module under each date so cross-cutting changes (e.g.
"the trainer learned about phase signals") are visible at a glance.


---

## [0.7.0-native] — 2026-05-02

**The `synapforge.native` framework lands.** Twelve parallel-agent
worktrees ship a zero-torch LNN+SNN training stack between 19:00 and
23:00 local time. This is a structural step, not a tuning step:
synapforge stops being a torch wrapper and starts being its own
framework. See [README.md §synapforge.native v0.1](README.md) and
[docs/NATIVE_OVERVIEW.md](docs/NATIVE_OVERVIEW.md).

### Added — synapforge.native (12 subpackages)

| package                          | branch                          | head SHA  | what it ships                                                                                       |
|----------------------------------|---------------------------------|-----------|-----------------------------------------------------------------------------------------------------|
| `synapforge.native_demo` (MVP)   | `feature/native-mvp`            | `a21a470` | Pure-numpy 100-step LNN+SNN train, manual VJPs for embed/linear/rmsnorm/cfc/plif/swiglu/CE + AdamW. |
| `synapforge.native.data`         | `feature/native-data-loader`    | `4b9735c` | parquet/jsonl/mixed token streams, no torch in hot path. 17 unit tests, parity bench, swap-in guide. |
| `synapforge.native.vjp`          | `feature/native-vjp-catalog`    | `b560ccc` | Closed-form VJPs for embed/linear/rmsnorm/swiglu/cfc/plif/sew/cross_entropy + dtypes.py + edge-case tests. |
| `synapforge.native.cuda`         | `feature/native-cuda-backend`   | `82ccaca` | `CudaTensor` + ops + allocator + streams + Triton glue + LNN+SNN-specific ops on CudaTensor.        |
| `synapforge.native.bench`        | `feature/native-saturation`     | `00580fd` | Saturation profiler + roofline + autotuner. Real autotune results doc + roofline report.           |
| `synapforge.native.spike`        | `feature/native-spike-pack`     | `b85de28` | uint16 16-spike packing, packed-matmul Triton kernels, autograd.Function bridge. 16× bandwidth.    |
| `synapforge.native.stdp`         | `feature/native-stdp-runtime`   | `ce3025b` | STDP-only optimizer, ring buffer, hybrid dispatch, Triton kernel for fused LTP/LTD. 23 tests.      |
| `synapforge.native.kernel`       | `feature/native-fused-kernel`   | `eaf0c17` | Fused fwd Triton kernel + scan kernel + closed-form fused bwd Triton kernel + autograd bridge.      |
| `synapforge.native.dispatch`     | `feature/native-dispatch-async` | `2403618` | HeteroPipeline 3-stage scheduler, per-block device router, throughput bench, simplified backpressure. |
| `synapforge.native.modal`        | `feature/native-modal-packing`  | `6a6e425` | ModalBatchPacker + ModalMaskBuilder + CrossModalContrastive + ModalDispatchEmbed. 61 unit tests.   |
| `synapforge.native.auxsched`     | `feature/native-async-aux`      | `395549c` | Streams + Future + 4 driver scaffolds + 19 tests. 2.9× speedup vs sequential on Run-7 timings.     |
| `synapforge.native.training`     | `feature/trainer-refactor-v2`   | `9ff6048` | BaseTrainer + TrainerConfig + KDTrainer + SFTTrainer + RL/SelfDistill stubs + mode dispatcher.       |

### Performance — measured on 2026-05-02

- **MVP CPU 2.86× vs torch oracle** — `synapforge.native_demo` 1.00s
  vs `synapforge.native_demo_torch_ref` 2.87s on identical 100-step
  LNN+SNN train (seed 1234, d=64, 2 layers, vocab 256, seq 16, batch 4,
  156k params). Both monotonic, final losses 5.04 (native) vs 4.93
  (torch) — within parity gate.
- **Fused HybridBlock 1.15-1.18× e2e** — over the unfused 4-op torch
  path (CfC + PLIF + SEW + synapse), bit-exact at d=256.
- **Async hetero dispatch 1.7×** — `throughput_bench.py` synthetic
  3-stage pipeline (data | GPU fwd+bwd | CPU AdamW) at steady state
  vs sequential reference.
- **Async aux 2.9×** — `auxsched/bench` synthetic Run 7 timings
  (selflearn + curiosity + NeuroMCP + STDP) parallelised vs sequential.
- **Spike pack 16× HBM bandwidth** — bit-packed `uint16` storage at
  spike→synapse boundary; ~600µs/step ceiling on A800 80GB.

### Run 7 vs Run 8 — torch baseline vs native projection

| run     | stack                          | tok/s @ A800 80GB        | status                  |
|---------|--------------------------------|--------------------------|-------------------------|
| Run 7   | torch + triton_block + bs=64   | **2,750** baseline       | live, PID 41692 rental  |
| Run 8   | native stack (post integration)| **17,000-30,000** projected | pending merge          |

Conservative projection 17k is MVP-CPU 2.86× + kernel 1.15× +
dispatch 1.7× compounded with 60% integration-friction efficiency loss.
Headroom 30k is "all levers stack cleanly". **Both are projections
until Run 8 actually runs.** Even 12-15k beats baseline 4-5×.

### Honest scope notes — what is NOT yet integrated

- The 12 subpackages are merged INTO their respective feature branches
  but NOT yet merged into a single trunk. Integration agent owns the
  cross-branch merge → `feature/run8-native-integration` (in progress).
  This commit only updates docs.
- PLIF currently dead at runtime (Run 3l/3m/3n logged 0/16 spike rate).
  Spike-pack 16× HBM saving is the bandwidth ceiling assuming PLIF
  fires; with PLIF dead, saving is currently 0%. P25 in MASTER_PLAN.md
  open. The kernel work and the spike-pack work are correct in
  isolation, but the runtime benefit is dormant until P25 closes.
- Phase 5 (full torch removal, including `torch_glue.py`) is NOT in
  this v0.1. ETA 6-9 weeks per `docs/TORCH_REPLACEMENT_PLAN.md`. v0.1
  is *additive* — torch path still works, native path is opt-in via
  per-block device router.

### Documentation

- **`README.md`** — added top-level "synapforge.native v0.1" section.
- **`docs/NATIVE_OVERVIEW.md`** (NEW) — single-doc map of the 12
  subpackages, file tree, 30-line quickstart, dormant-flag list.
- **`docs/CHANGELOG.md`** — this entry.
- Pre-existing native docs (carried forward from feature branches):
  `docs/NATIVE_DEMO.md`, `docs/NATIVE_HETERO_DISPATCH.md`,
  `docs/NATIVE_SPIKE_PACKING.md`, `docs/NATIVE_MULTIMODAL_PACKING.md`,
  `docs/NATIVE_CUDA_TENSOR.md`.


---

## [0.6.5-fix] — 2026-05-02

TOKEN_SOUP root-cause diagnosis + fix landed before the native sprint.

### Fixed — Run 7 TOKEN_SOUP word salad

★ **Token-soup root cause** (`2765f46`, `e564e72`, `3df3bce`,
  `3c092e6`) — Run 6 (PID 24075, killed 14:00) emitted word salad at
  every checkpoint despite ce dropping. Three superimposed causes:
  1. **Tied LM head** (lm_head.weight = embed.weight) made the model
     learn embedding geometry, not output distribution; 151k×1024 tied
     parameters cannot do double duty cleanly at 100M backbone scale.
  2. **`--lm-head-spectral-norm`** wrapped a *new* lm_head in
     `nn.utils.spectral_norm` *after* tying, registering 71 missing
     state-dict keys and effectively cold-starting the LM head every
     warmstart attempt.
  3. **kd-weight 0.40** with token-soup data → kd loss masks the CE
     ascent.

  Fix landed as a 5-step diagnostic commit chain:
  - `--untie-lm-head` (default ON) — separate `lm_head.weight`
    parameter, embed and head no longer share storage.
  - `--no-warmstart` (default OFF, but ON for Run 7 cold-start) —
    bypass the spectral_norm key-mismatch landmine entirely.
  - Drop `--lm-head-spectral-norm` (P28 patch retracted; spectral_norm
    must be first-run or bridge-ckpt only, never bolted on later — see
    `feedback_spectral_norm_warmstart_cost.md`).
  - `--kd-weight 0.7` (up from 0.40) — let the teacher signal dominate
    while the student LM head re-learns from cold start.
  - `--data-files` literal-newline bug fixed (commit `2765f46`) —
    bash `\n` in the launcher line wasn't being interpreted by the
    parquet stream loader, so Run 6 had been training on a single
    parquet for 30k steps instead of the 3-corpus quality mix
    (fineweb_edu + wt103 + alpaca_zh).

### Verification

- Run 7 launched 2026-05-02 16:40, PID 41692 on rental
  `117.74.66.77:41614`. Step 500 verdict pending (~30 min after
  launch). Diagnosis is correct iff val ppl at step 500 lands
  ≤ 600 (down from Run 6's plateau at ppl ~3700).


---

## [Unreleased] — 2026-05-01 → ongoing

Investor demo packaging + integration test rollup.

### Added — Tests / CI

- **`tests/integration/`** — first end-to-end suite for the cross-module
  wiring that "smoke tests pass individually" did not cover:
  - `test_phase_signal_write_consume_cycle` — phase_manager.write_phase_signal
    -> phase_signal.read_phase -> phase_signal.consume_phase, including
    idempotency.
  - `test_chat_eval_gate_scores_good_and_bad` — proves the heuristic gate
    actually separates word-salad from coherent output.
  - `test_skill_log_v2_save_kill_recover` — kill mid-write, recover via
    rotation sibling, confirm idempotent reload.
  - `test_triple_backup_daemon_empty_dir_warning` — fires on the 5th
    consecutive empty cycle (the `systemd-points-at-wrong-path` footgun).
  - `test_auto_eval_daemon_detects_fresh_ckpt` — drop a `step_*.pt`,
    expect detection + sha-dedup on second pass.
  - `test_alpaca_to_sft_pipeline` — prep_alpaca_qwen tokenization
    convention + chat_repl template formatting must match byte-for-byte.
  - `test_universal_codebook_l1_to_l2_co_firing_mint` — repeat a 3-action
    pattern, confirm L2 prototype minted with the canonical trigger_seq.
  - `test_rfold_bench_matches_verify_rfold` — `synapforge.demo.rfold_bench`
    must agree with the standalone `scripts/verify_rfold.py` on R=1 (<1e-3)
    and R=8 (<10%).
  - `test_synapforge_demo_all_smoke` — `synapforge-demo bench` dispatches
    through `synapforge.demo.cli.main` without crashing.

  Heavy deps (torch, pyarrow, playwright) are stubbed via `importlib.util`
  isolated module loads + `_torch_available()` skipif guards, so the suite
  runs on a torch-less box and skips cleanly rather than failing.
- **`tests/integration/run_smokes.sh`** — one-line entrypoint
  (`bash tests/integration/run_smokes.sh`) that prints a friendly
  optional-deps summary and runs the suite.
- **`docs/INTEGRATION_TESTS.md`** — test inventory + how to run on a clean
  clone + CI guidance + known-flaky triage.

### Notes

- Real training is **not** exercised by these tests; that lives on the
  rental and is gated by `chat_eval_gate.py` + `auto_eval_daemon.py`.
- Total integration-suite runtime on CPU: ~5 seconds (each test < 1s
  except `rfold_bench` which is bounded by the (N=512, R=8) shape).


---

## [4.2.x] — 2026-05-01 (final-day-1 burst)

The "make the investor demo defensible" sprint. Six review-agent passes,
~50 bug fixes, full-volume backup, phase relauncher, integration tests.

### Added — Trainer (`train_v42_universal.py`, `train_100m_*.py`)

- **`scripts/relaunch_loop.sh`** (commit `80d8c3b`) — outer wrapper that
  catches trainer exit code 101 (phase change), parses the consumed
  `.phase` JSON, and respawns the trainer with the new phase's flags.
  Without this, `--phase-aware` is dead code: the trainer dropped
  `.phase.consumed.<ts>` and exited but nothing re-launched it.
  Includes anti-thrash guard (same-phase-twice-in-a-row → bail) and a
  25h wall-clock budget cap.
- **`scripts/phase_manager.py`** (commit `a177050`) — watches `train.log`
  for ppl thresholds; writes `.phase` JSON when crossings happen.
  Five phases:
  - 0 lm_kd (any ppl)         — pretrain CE+KD only
  - 1 intrinsic (ppl ≤ 250)   — + curiosity + self-learn TTT + STDP novelty
  - 2 multimodal (ppl ≤ 100)  — + byte-patch image/audio encoders
  - 3 chat_sft (ppl ≤ 60)     — Alpaca-zh SFT, response-only loss, lr 1e-5
  - 4 rl_grpo (chat-eval gated) — GRPO verifier-RL with sympy
- **`synapforge/phase_signal.py`** (atomic `.phase` plumbing) — `os.replace`
  rename, malformed-JSON quarantine to `.phase.malformed.<ts>`,
  idempotent consume.
- **`scripts/auto_pretrain_to_sft.py`** (commit `7a3ff09`) — auto-pivot
  pretrain → SFT when ppl < 250 + chat_eval pass rate < 0.6 (gate hit
  but quality gap remains). 17 bugs reported by 4-agent review fixed in
  the same commit.

### Added — Backup / reliability

- **`scripts/triple_backup_daemon.py`** (commit `f8194a0` + `3899d61`) —
  watch a runs dir, every `--interval` seconds rsync the entire volume to
  3 destinations in parallel:
  1. mohuanfang.com:/home/liu/synapforge_backup/<run>/  (rsync, primary)
  2. GitHub Releases  Lihfdgjr/synapforge:auto-<run>-<date>  (top-K
     `best*.pt` only; >100MB chunked to 50MB + manifest)
  3. HuggingFace dataset  Lihfdgjr/synapforge-ckpts  (full folder)
  At-least-one-of-three success per cycle = OK. Sha256[:16] dedup avoids
  re-uploading 600MB ckpts that haven't moved. WARNING fires on 5
  consecutive empty cycles (the systemd-mis-path footgun).
- **`scripts/triple_backup_daemon.py::_chunked_split_fallback`** —
  GitHub Releases silently truncate >100MB assets; we shred 600MB ckpts
  to 50MB chunks + a manifest with per-chunk sha256 + a recovery one-liner.
- **`scripts/honest_eval_hook.py`** (commit `2465dfb`) — every N steps,
  generate 5 EN + 5 ZH samples and append to a structured JSON; prevents
  the "ppl=80 but output is word salad" hallucination.
- **`scripts/auto_eval_daemon.py`** (commit `80d8c3b`) — watches the run
  dir for new `step_*.pt`/`best*.pt`, evals each on a light bench
  subset (mmlu/hellaswag/lambada) + heavy bench on `best*` only.
  Sha-deduped; rebuilds a `<run>/auto_eval/index.json` for plotting.
- **`scripts/install_watchdog_cron.sh`** + **`scripts/rental_watchdog.sh`**
  + **`scripts/rental_health.py`** — SSH/process watchdog + GPU-busy
  probe so eval defers to CPU when training holds CUDA.
- **`scripts/sync_to_mohuanfang.sh`** + **`scripts/backup_v41.sh`** —
  manual fallbacks when GH push protection / HF rate-limit fires.

### Added — Investor demo (`synapforge/demo/`)

- **`synapforge.demo.cli`** with subcommands
  `pitch | button | bench | chat | stdp | all | json` (commit `5e6130b`).
  `synapforge-demo all` runs all five demos in <60s, CPU-only, no
  network. Console UTF-8 reconfigure so Chinese / em-dashes render on
  Windows cp936 default + Linux + pipes.
- **`synapforge/demo/four_button.py`** — NeuroMCP 4-button live demo:
  agent learns to click the right colored square WITHOUT emitting any
  tool-call tokens; synapse density grows + codebook K grows.
- **`synapforge/demo/rfold_bench.py`** — closed-form R-fold vs sequential
  CfC, honest per-shape table at 5 (N, R) shapes.
- **`synapforge/demo/stdp_demo.py`** — inference-time STDP self-organisation;
  density (|W|>0.05) climbs from 0% to ~25-30% over 200 trials with no
  optimizer + no loss.
- **`synapforge/demo/chat_demo.py`** — 5 EN + 5 ZH prompts; loads a v24h
  ckpt if available, else replays `chat_recorded.json`.
- **`scripts/release_packager.py`** — gather `chat_demo.json`,
  `bench_results.txt`, the demo CLI output and zip with a timestamp.

### Added — Multimodal (`synapforge/modal/`)

- **`train_multimodal.py`** + **`train_3d.py`** + **`train_full_modal.py`**
  (commit `9442f19` + `cec04ed`) — joint native byte-patch training across
  9 modalities (text + image + audio + video + 3D + time-series + tabular
  + code + maths). Cross-modal contrastive aux + per-modality byte-patch
  encoders gated on Phase 2.
- **`scripts/prep_3d_data.py`** + **`scripts/download_multimodal_real.sh`**
  + **`scripts/prep_multimodal_data.py`** — data preparation pipelines.
- **`synapforge/modal/byte_patch.py`** — universal byte-patch tokenizer
  (replaces per-modality conv encoders flagged in audit `170`).

### Added — NeuroMCP / universal codebook (`synapforge/action/`)

- **`synapforge/action/universal_codebook.py`** (commit `6017c86`) —
  open-ended, lifelong, three-layer prototype space:
  - L1 (9 atomic primitives, frozen after warmup)
  - L2 (compounds minted from Hebbian co-firing detection)
  - L3 (macros minted from user text or successful traces)
  Idempotent reload (loading twice yields identical state; verified by
  the embedded `_smoke()` and the new integration test).
  HNSW backend (cosine, M=16, ef=200) with linear-fallback when
  hnswlib is missing.
- **`synapforge/action/skill_log_v2.py`** — atomic write + rotation
  (last 5 snapshots) + append-only history.jsonl. Mid-write crash
  recovers via the newest cleanly-parseable rotation sibling.
- **`synapforge/action/per_domain_neuromcp.py`** + **`compositional_codebook.py`** +
  **`hnsw_skill_index.py`** + **`skill_synthesizer.py`** — domain-conditioned
  routing, K=100k+ retrieval, AI-driven mint heuristics.
- **`scripts/skill_demo.py`** — end-to-end proof of the universal codebook:
  9 L1 → mint 3 user L3 → drive 50 episodes → save → reload from disk →
  identical state. Runs on CPU in <2s.

### Added — Continual learning / self-learn (`synapforge/learn/`)

- **`synapforge/learn/continual_daemon.py`** + **`autonomous_daemon.py`**
  (commit `4a6b2c0` + `80d8c3b`) — Track A: web content with 7-gate
  filter (TRAK + EWC + LoRA shadow merge) feeds the trainer; Track B:
  user chat into a retrieval cache (does not touch weights). Anchor:
  Anthropic 2510.07192 fixed-count poison.
- **`synapforge/learn/web_curriculum.py`** — auto-driven curriculum,
  replaces the bilibili hardcode.
- **`synapforge/learn/retrieval_memory.py`** — content-addressable cache
  for user chat that bypasses weight updates.
- **`scripts/launch_continual_daemon.py`** + **`scripts/train_neural_web.py`**
  — orchestrators.

### Added — Safety stack (`synapforge/safety/`)

- **`synapforge/safety/constitutional.py`** (commit `032c52d`) — CAI SL-CAI
  4-round self-critique pipeline (Anthropic 2212.08073).
- **`synapforge/safety/red_blue.py`** + **`persona_swap_corpus.py`**
  + **`persona_swap_red.jsonl`** — red/blue same-model self-play with DPO
  (β=0.1).  Persona-swap is the single highest-leverage axis (~80% of
  public jailbreaks are variants).
- **`synapforge/safety/dpo.py`** + **`dpo_trainer.py`** — Direct
  Preference Optimization (Anthropic 2305.18290).
- **`synapforge/safety/eval_harness.py`** + **`judge.py`** — automated
  red-team grader.
- **`scripts/run_safety_pipeline.py`** — single-shot orchestrator.

### Added — Bench harness (`synapforge/bench/`)

- **`synapforge/bench/{humaneval, mbpp, gsm8k, mmlu, hellaswag, lambada}.py`**
  (commit `9442f19`) — 6 benches sharing a `BENCH_REGISTRY` + `run_bench`
  contract. Auto-eval daemon dispatches a light subset
  (mmlu/hellaswag/lambada, ≤50 samples) on every step ckpt and a heavy
  subset (humaneval/mbpp/gsm8k, ≤20 samples) only on `best*.pt`.
- **`synapforge/bench/compare_pytorch.py`** — head-to-head fairness check
  vs a transformer baseline on shared (N, R) shapes.
- **`scripts/run_all_bench.py`** + **`run_all_paper_validations.sh`**
  + **`run_scaling_law.py`** — paper-prep harnesses.

### Added — Documentation

- **`docs/RELEASE_FLOW.md`** — pre-merge checklist + auto-release on tag.
- **`docs/PHASE_TRAINING.md`** — phase manager + relauncher contract.
- **`docs/RELIABILITY.md`** — three silent killers (KD-skip lr drop,
  empty-dir backup, EOT injection) and the daemons that catch them.
- **`docs/MONOTONIC_QUALITY.md`** + **`docs/CONTEXT_SCALING.md`** +
  **`docs/HONEST_ASSESSMENT.md`** — long-context honesty + A100 80GB
  physical limits.
- **`docs/CONTINUAL_LEARNING.md`** + **`docs/SAFETY.md`** +
  **`docs/SAFETY_PLAN.md`** — Track-A/B + Anthropic-stack design.
- **`docs/REVIEW_*.md`** — three review reports (REVIEW_DEMO,
  REVIEW_NEUROMCP, REVIEW_TRAINER) summarising the 25+ bug fixes.
- **`docs/RFOLD_PAPER.md`** + **`docs/INVESTOR.md`** — investor + paper
  positioning, including the honest-claim retractions below.

### Fixed — HIGH-severity bugs caught by review agents

★ **KD-skip 0.3× LR drop** (`train_100m_kd.py`) — when KD batch was
  skipped (teacher cache miss), the optimizer stepped with full lr but
  loss dropped 70%, effectively a 0.3× lr cliff at every miss. Now the
  optimizer step is gated by an explicit "did we have a real teacher
  signal" boolean.

★ **GPT-2 EOT injected into Qwen** (`train_100m_kd.py`) — the GPT-2
  teacher tokenizer's EOT id (50256) was being concatenated into Qwen
  student input ids (vocab 151643), corrupting the student's pad/eos
  alignment. Fixed by mapping EOT → Qwen's `<|endoftext|>` (151643-1) in
  the rep-level KD path; logit KD avoids the issue entirely (vocab-size
  mismatch is handled by the projection MSE).

★ **`_log` undef in `triple_backup_daemon`** — early `_log(...)` call in
  `_chunked_split_fallback` referenced the module-level `_log` before
  the function was defined under one branch. Fixed by hoisting `_log`
  to the top of the file.

★ **Chinese cp936 mojibake** (`synapforge/demo/cli.py`,
  `synapforge/demo/stdp_demo.py`) — Windows console + Unicode block
  glyphs (U+2580..U+258F) render as `??` even after `stdout.reconfigure`.
  Fixed by (1) explicit UTF-8 stdout reconfigure with `errors="replace"`,
  (2) ASCII-only heatmap glyph ladder ` .:-=+*#@`. Cross-platform
  identical output verified.

★ **L1 silent reset on reload** (`synapforge/action/universal_codebook.py
  ::load_dict`) — L1 primitives were re-initialised with a fresh random
  vector on every load while metadata was patched, so trained L1 slots
  silently drifted ~0.05 max-abs per reload. Fix: when an L1 entry has a
  saved `embedding`, restore it byte-for-byte AND rebuild the HNSW entry.

★ **Sandbox shell-injection** (`scripts/auto_eval_daemon.py`,
  `triple_backup_daemon.py`) — earlier draft passed the user-supplied
  `--watch` argv into a bash one-liner. Fixed by passing every command
  as `subprocess.run([...])` lists with no shell expansion; rsync /
  ssh paths quoted via `shlex.quote` where shell=True is unavoidable
  (the rsync `-e "ssh -o ..."` argument).

### Fixed — Other bugs (selected from the 25+ in `29bb1d4`)

- `train_100m.py` `--lr-decay` was parsed but never wired (lr stayed at
  3e-4 the whole run). Now `lr_at(step, peak, warmup, total, kind)` is
  called per step with proper warmup + cosine decay.
- `auto_rsync_ckpt.py` had a hard-coded 3-dir SRCS list that missed every
  run after v1.1. Now glob-scans `/workspace/runs/synapforge_*` for
  `.log/.json/.csv/.md`.
- `sys.path` shadowing in `train_100m.py` — the script directory was
  prepended ahead of `/workspace`, so the nested `synapforge/synapforge/`
  package won over the patched outer `synapforge/`. Now strips script
  dir from `sys.path` at startup.
- Triton 2.1 bf16 fused HybridBlock kernel: upcast bf16/fp16 → fp32
  inside `_triton_block_forward`/`backward`; force fp32 elig buffer
  (`atomic_add` doesn't support bf16); always pass `ENABLE_STDP=False`
  to the kernel and do full STDP in the PyTorch wrapper.
- PLIFCell DA-LIF `tau_init` accepts `{float, "bimodal", ("bimodal",
  fast, slow), "log_uniform"}`; bimodal default starts half channels
  fast (tau~3) and half slow (tau~30) to break the Fang-2021 PLIF
  symmetry.
- z-loss + label smoothing in `train_100m.py`: `loss = ce + α·logsumexp²`,
  PaLM/Gemma standard; `--z-loss-weight` (default 1e-4),
  `--label-smoothing` (default 0.0).
- Per-layer PLIF spike-rate monitor every 50 train steps prints
  `spike: mean=… range=… dead=N/M sat=N/M` from `last_spike_rate`
  buffers. Detects dead (<0.005) and saturated (>0.5) channels.
- Optimizer state save/restore in checkpoints — Adam m/v momentum now
  carries across warmstart runs.
- `train_100m.py --grad-checkpoint` flag wraps each `blk(x)` in
  `torch.utils.checkpoint(use_reentrant=False)` for ~50% activation
  memory at B≥128.
- `chat_repl.py` re-encoded the prompt every iteration to compute the
  slice offset (O(N²) decode). Cached `prompt_len` once at start.
- `chat_demo.py` template-mismatch with prep_alpaca_qwen tokens — now
  byte-equal (verified by the new integration test).

### Honest claim retractions

The agent-drop pipeline initially overclaimed several headline numbers.
We've retracted them in source, README, paper draft, and pitch deck.
What's published now is what we actually measured.

- **R-fold "167× speedup"** → **2.99× at R=8 chunked, GPU only.**
  167× was the abstract scaling at R=1024 with NO re-anchoring; gate
  drift at R≥64 makes that mathematically valid but practically
  unusable. Honest measured numbers (`scripts/verify_rfold.py`):
  CPU + N=128 R=8: fold is 0.5× (slower); GPU peak 2.99× at N=64 R=16;
  fold loses past N≥256 even on consumer GPU. Math: R=1 exact (1.5e-6),
  R=8 drift 0.3%, chunked L=2 shrinks R=8 error 10×.
  Public docs (`docs/RFOLD_PAPER.md`, `docs/INVESTOR.md`, README) and
  the paper abstract were all rewritten 2026-05-01 to match (commits
  `2673783`, `e1e7f5d`, `9225621`).
- **NeuroMCP "density 5%→28%"** → **5.7%→7.7% over 80 trials.**
  The 28% number was the inference-time STDP density (`stdp_demo.py`
  on the fast-weight matrix), NOT the NeuroMCP-head sparse synapse
  density. We were quoting two different metrics interchangeably.
  Now `four_button.py` reports the head's density truthfully and
  `stdp_demo.py` reports the fast-weight density separately.
- **"4-button 100% hit rate"** held only post-warmup with seed=7 +
  exactly the documented hyperparameters. Variance across seeds is
  60-95%; the demo intentionally pins the seed but we say so.
- **Mythos parity benchmarks** — the README's "all-Mythos-benchmark
  parity" claim is a TARGET, not a measured result. Current measured
  parity: WikiText val ppl band, LAMBADA tiny, HellaSwag tiny.
  HumanEval / MBPP / LiveCodeBench / Aider / SWE-bench: not yet run,
  not yet shipping.
- **375M flagship "ppl 44.2 multilingual chat"** — this is the v4.1
  ckpt's measured number on en+zh+math SFT eval, NOT the v4.2 universal
  ckpt currently retraining. We say so explicitly in the demo CLI's
  pitch text.

### Removed

- All hardcoded GitHub tokens from source (commit `183`-class fixes) —
  read `GH_TOKEN` from env. Repo CI / local dev unchanged; affected
  only the rental backup scripts.
- Hard-coded bilibili curriculum URL (replaced by `web_curriculum.py`).


---

## [4.1.0] — 2026-04-30

Long-context drift fix + STDP unlock + curiosity formula. The
"inference-time STDP gated by `self.training`" discovery alone earned
its own NeurIPS-targeted paper.

### Added

- **`synapforge/curiosity.py`** (commit `f4d6946`) — STDP-driven
  curiosity formula: `C = 0.40·ΔF + 0.25·‖ΔW_STDP‖ + 0.15·G_HNSW + 0.10·H[spike] + 0.05·N - 0.05·V_noise`.
  ‖ΔW_STDP‖ is mathematically immune to the noise-TV problem that kills
  ICM/RND.
- **`synapforge/infinite.py`** — five-tier memory hierarchy with BM25
  sidecar (single largest -6pp drift contributor on L3 50M ctx).
  MultiBandTau wired through PLIF, trained PQ codebook, dual-path 32K
  → dense fallback. L3 drift target: <5% (was 8-15%).
- **NIAH eval harness** + 12.5h scaling-law runner — automatic ablation
  + recovery on rental.
- **Async chat kernel** (commit `8fce009`) — listen / interrupt /
  proactive-message; replaces blocking REPL.
- **ALP reasoning reward** (`reference_alp_reward_2506.md`,
  arxiv 2506.05256) — `r = 1[correct] - β·N·max(p_solve, 1/K)`,
  curriculum-gated (β=0 until p_solve > 0.3).
- **`synapforge/learn/`** — autonomous web-driven curriculum +
  4-layer 污染防护 (PoisonDetector + ProvenanceTracker + WeightFirewall +
  AdversarialRedTeam).

### Fixed

- **`stdp_fast.py:121`** (commit `bc80041`) — `if self.training:` gated
  STDP at inference, defeating the entire forward-only-Hebbian story.
  Removed; STDP now always active. Single line; quality monotonic on
  context length.
- **PLIF dead at bootstrap** (`feedback_plif_dead_bootstrap.md`) —
  initial spike rate = 0 makes ATan surrogate × 0 = 0 autograd dead.
  Two-stage bootstrap per SEW recipe: phase 0 PLIF observe-only no_grad,
  dense CfC trains; PLIF lights up automatically as CfC representations
  mature.
- **PyTorch buffer in-place mutation** (`feedback_torch_buffer_inplace.md`)
  — fast-weight in-place updates were tripping autograd version mismatch.
  Forward emits a detached clone for downstream consumers, then mutates
  in place.

### Notes

- Ckpts saved to GitHub Release `Lihfdgjr/synapforge:auto-v41-*` (chunked).
- Mohuanfang.com private backup at `/home/liu/synapforge_backup/` (4.3GB,
  v4.1 step 10000 + v28 adv).
- Rental at `111.51.90.14:44938` (SSH dead 2026-05-01); v4.0/4.1/4.2
  ckpts are recoverable from GH Release if mohuanfang is partial.


---

## [1.5.0-dev] — 2026-04-26 (in progress on rental)

Web-augmented self-learning training. **Goal: WT103 val ppl < 50**
(still tracking; 24h run).

### Added

- **`train_v15_full.py`** — joint full-9-modality + KD + intrinsic +
  semantic trainer.
  - Stronger teacher: **Qwen2.5-0.5B** rep-level KD (vocab-agnostic;
    mean-pool teacher hidden → Linear projection → MSE; bypasses
    GPT-2 50257 vs Qwen 151643 mismatch).
  - Web-augmented data: WT103 + agent-generated Q/A + FineWeb-Edu chunks
    (downloaded one-shot at training prep, **not** runtime tool-calling
    — preserves NeuroMCP rule that the model itself never schema-calls).
  - Semantic understanding: triplet (anchor/synonym/antonym), definition
    modeling, char-level aux head, cross-modal contrastive (image+caption).
  - Intrinsic exploration: every N steps the model self-rolls a `Q:`
    continuation via its own LM; will feed back into NoveltyDrive replay
    in v1.6.
- **`train_100m_kd.py`** — text-only KD trainer with frozen GPT-2 teacher
  (logit KL distillation, batch-mean reduction with `T*T` scaling).
- **`train_full_modal.py`** — joint native training across all 9
  modalities + action + NeuroMCP. Used as warm-start basis for v1.5.
- **PLIFCell DA-LIF tau init** — `tau_init={float, "bimodal", ("bimodal",
  fast, slow), "log_uniform"}`. Bimodal default: 32 channels @ tau=3,
  32 @ tau=30 (verified for hidden=64).
- **z-loss + label smoothing** in `train_100m.py`.
- **Per-layer PLIF spike-rate monitor** every 50 steps.
- **Optimizer state save/restore** in checkpoints.
- **Gradient checkpointing flag** (`--grad-checkpoint`).
- **Auto checkpoint backup daemon** (`auto_ckpt_backup.py`) +
  **Auto metrics rsync** (`auto_rsync_ckpt.py`) +
  **GitHub push retry daemon** (`github_push_retry.py`).

### Fixed

- Triton 2.1 bf16 MLIR encoding bug in fused HybridBlock kernel
  (three patches, see 4.2.x section above for the full text).
  Result: standalone B=64 T=256 D=512 bf16 fwd+bwd peak 0.51 GB
  (vs 4.28 GB on the unpatched PyTorch fallback). v1.3 KD trains at
  **24,000 tok/s on A100** (vs v1.2 b80's silently-fallback
  4,712 tok/s, a 6.1× speedup).
- `train_100m.py --lr-decay` was dead (parsed but never wired). Now
  `lr_at(...)` is called per step.
- GitHub PUSH PROTECTION triggered by hardcoded GitHub PAT in early
  commit — sanitised, switched to env-var read.
- `sys.path` shadowing in `train_100m.py` — strips script dir at startup.

### Removed

- All hardcoded GitHub tokens from source — read `GH_TOKEN` from env.

### Pending

- v1.5 full launch with both A100s in DDP (NV12 NVLink topology
  confirmed; expected 1.8-1.9× scaling with
  `synapforge.distributed.wrap_model`).
- Cross-modal contrastive integration (data downloaded; loss wired but
  pending real COCO/AudioCaps batches).
- TestTimeTraining inference-time adaptation
  (`sf.self_learn.TestTimeTraining`).


---

## [0.5.0] — 2026-04-26

First public release. Code base ~7,100 LOC. Eleven of twelve milestones
(M1-M11) landed; v1.0 gates on Loihi-2 hardware verification.

### Added

- **Core API** — `sf.Module`, `sf.LiquidCell`, `sf.PLIF`, `sf.SparseSynapse`.
- **Plasticity** — `Hebbian`, `STDP`, `BCM`, `SynaptogenesisGrowPrune`,
  `PlasticityEngine`. All rules are local and gradient-free.
- **Surrogate-gradient registry** — six built-ins (atan, sigmoid,
  super-spike, fast-sigmoid, triangular, multi-Gaussian) plus
  `register()` for custom rules.
- **Backends** — `gpu_dense` (PyTorch eager, numerical-equivalent to
  `mscfc.LiquidS4Cell`), `triton_block` (fused parallel scan, ~29×
  forward), `cpu_event` (numba CSR raster for sparse <5% inference),
  `lava_export` (Loihi-2; code path complete, hardware verification
  pending).
- **Optimizer** — `PlasticityAwareAdamW` with `MultiSourceParam` to
  merge gradient and plastic-delta streams.
- **Distributed** — `sf.wrap_model` DDP wrapper with `PlasticBufferSync`
  (1.78× throughput on 2x A800).
- **Quantization** — Ternary BitNet 1.58 post-training quantization
  (~20× weight compression, <2pp accuracy loss on tested workloads).
- **CUDA-graph runtime** (`runtime_cuda_graph.py`) — ~2× throughput on
  small recurrent models.
- **Hugging Face adapter** (`huggingface_adapter.py`) — drop-in tokenizer
  + dataset wrapper for plug-and-play training.
- **Examples** — five runnable scripts in `examples/` covering
  hello-world, LNN training, plasticity, Triton speedup, and 2-GPU DDP.
- **Tests** — full pytest suite under `tests/` with CPU-friendly skips
  for GPU/Triton tests.
- **CI** — GitHub Actions matrix (Linux/macOS/Windows × Python 3.10/3.11)
  on push/PR; PyPI auto-publish on `v*.*.*` tags.

### Pending for v1.0

- Loihi-2 numerical equivalence verification (M11 hardware path).
- Full neuromorphic + analog co-deployment story (M12).
- Plasticity-aware autograd (currently plasticity is gradient-free; v1.0
  will thread plastic deltas through autograd for end-to-end
  differentiable rules).

### Notes

- Core install is `torch + numpy` only. `triton`, `lava`, `pyarrow`, and
  `transformers` are opt-in extras.
- Tested on torch 2.0/2.1/2.2, Python 3.10/3.11/3.12.
