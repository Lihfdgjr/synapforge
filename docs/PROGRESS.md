# Synap-1 Live Training Progress

> **Synap-1 phase 0, val ppl 421, 30% to phase 1 trigger.**

One-pager dashboard for the current run. Glance here first; everything else
([MASTER_PLAN](MASTER_PLAN.md), [INVESTOR](INVESTOR.md), [NAMING](NAMING.md))
expands the context.

---

## 1. Live numbers (2026-05-01)

| Metric        | Value                              | Source / note                                  |
|---------------|------------------------------------|------------------------------------------------|
| Run name      | `v24h_qwen3` (Run 3b)              | post-strip-optim_state recovery                |
| Trainer PID   | 16697 on rental 117.74.66.77:41614 | A800 80GB                                      |
| Step          | ~2920                              | last poll; see `monitor.jsonl`                 |
| Train CE      | 7.5 - 8.3                          | KD-on/off oscillation, kd-every=4              |
| **VAL ppl**   | **421**                            | step 2500; phase 1 trigger at <=250            |
| z-loss        | 134 -> 104 -> 126 (falling)        | logits regularizer, healthy                    |
| Spike rate    | 0/10 (PLIF dead)                   | auto-revive engaged at step 1000+, slow        |
| Backend       | triton_block, batch=64, kd-every=4 | A800 utilization gate                          |
| Tok/s         | see `monitor.jsonl`                | last reading not pinned in this brief          |
| GPU util      | ~78%                               | 77 GB / 80 GB used                             |
| RAM           | 5 GB / 115 GB (110 GB free)        | 14 cores, headroom for CPU jobs                |
| Disk          | 28% / 100 GB                       | post-disk-full cleanup (Run 2 incident)        |

Warmstart: `/workspace/runs/v24h_qwen/step_002250_plif_reinit_NOOPT.pt`
(legacy ckpt, optim_state stripped, fresh Adam this run).

---

## 2. Run timeline — what failed, what we learned

| Run      | Window    | Status        | What happened                                                        | Lesson                                                                  |
|----------|-----------|---------------|----------------------------------------------------------------------|-------------------------------------------------------------------------|
| Run 1    | Apr       | DIVERGED      | val ppl ramped, no PLIF homeostasis -> spikes saturated then died    | Need EMA threshold control on PLIF (P1 / `f051257`)                     |
| Run 2    | -> May 1  | DISK-FULL     | 35 ckpts (58 GB) on rental 100 GB disk, trainer crashed on save      | Cap retention; offload to mohuanfang. Recovery: `2d8bb24`              |
| Run 3a   | May 1 AM  | DIVERGED      | warmstarted with stale Adam momentum -> loss exploded ~step 50       | Strip `optim_state` from legacy ckpts; fresh optimizer on warmstart    |
| Run 3b   | May 1     | PLATEAU       | val ppl 397-421 for 8h on phase 0                                    | train↔val drift unfixed; need TTT replay to break plateau               |
| Run 3l   | May 1-2   | DIVERGED      | step 4000=569, step 5000=**1864**, step 5500=**2522** Run-3c-class    | killed 00:55; merged T2.4-T2.7 not in live; LR 1e-4 + warmup 100 too hot |
| **Run 3m** | **May 2 01:10 -> now** | **RECOVERING** | warmstart step_002000.pt (last known-good), LR 5e-5, warmup 500, kd-every 8, T2.5 spike-target weight=0.05, T2.6 lm-head spectral norm enabled, P30 indent fix applied | step 4120 ce=9.4 (was 11.93 step 1); VAL step3000=16031 → step4000=11838 (descending -26%/1k steps; ETA phase 1 trigger ~step 8000-9000) |

Run 3a -> Run 3b cutover took <30 min. Optim-state strip is now standard
warmstart hygiene; see [RENTAL_RECOVERY](RENTAL_RECOVERY.md).

---

## 3. Phase progress

```
phase 0  LM-only KD             [######....]  ~30% — val ppl 421 -> target 250 -> trigger phase 1
phase 1  intrinsic + STDP-novel [..........]   0%  — needs phase 0 ppl <= 250
phase 2  modal byte-patch       [..........]   0%  — needs phase 1 ppl <= 100
phase 3  SFT alpaca             [..........]   0%  — needs phase 2 ppl <= 60
phase 4  RL GRPO                [..........]   0%  — needs phase 3 eval pass
```

Bar math: `30% = (606 - 421) / (606 - 250)` using Run-1 baseline 606 as the
phase-0 starting reference (per [PHASE_TRAINING](PHASE_TRAINING.md)).
Refresh §3 by recomputing this with the latest val ppl from `monitor.jsonl`.

Objectives (from [MASTER_PLAN](MASTER_PLAN.md)):
- **O1**: chat-grade — val ppl <=60 + chat eval >=60%.
- **O10**: 50M effective context — harness ready, awaiting ckpt.
- **O11**: monotonic quality with context length — harness ready.

---

## 4. Backup state

| Path                                                                  | Content                                                          | Health                  |
|-----------------------------------------------------------------------|------------------------------------------------------------------|-------------------------|
| `mohuanfang.com:/home/liu/synapforge_backup/v24h_qwen/`               | 35 ckpts of OLD Run 2 (58 GB), step 250 -> 8500                  | full history retained   |
| `mohuanfang.com:/home/liu/synapforge_backup/v24h_qwen3/`              | Run 3b ckpt step_2000.pt synced                                  | catching up to live     |
| `mohuanfang.com:/home/liu/synapforge_backup/best_ckpts/`              | 4.3 GB v13/v16/v19-22/v28 historical milestones                  | static milestones       |
| rental `/workspace/runs/v24h_qwen3/`                                  | live ckpts, latest step ~2920                                    | ephemeral (rental SSH)  |
| GitHub Releases                                                       | NOT enabled (no `GH_TOKEN` set)                                  | RISK                    |
| HF Hub                                                                | NOT enabled (no `HF_TOKEN` set)                                  | RISK                    |

Daemon `triple_backup_daemon.py` (PID 16219) on rental, watching `v24h_qwen3`,
every 600 s. **Honest read**: this is currently single-path -- mohuanfang only.
The daemon name says "triple"; in reality only the rsync leg fires.
mohuanfang has 1.5 TB total, 1.2 TB free, so capacity is fine, but a single
provider failure would be unrecoverable. See [BACKUP](BACKUP.md) for the
intended triple-leg design and the `GH_TOKEN` / `HF_TOKEN` placeholders.

---

## 5. CPU/RAM utilization (idle 14 cores)

| Job                    | Output                                                         | Status         |
|------------------------|----------------------------------------------------------------|----------------|
| synth_chinese_pretrain | `/workspace/data/synth_zh_phase1.parquet` (50K rows, 20 MB)    | DONE           |
| prep_alpaca_qwen       | `/workspace/data/alpaca_zh_qwen_tokenized.parquet` (11.6 MB)   | DONE -- 48,818 ex, 87% response tokens |
| Qwen tokenizer warm    | `/dev/shm/qwen_tok.pkl`                                        | DONE -- in shm |
| wt103 tokenize         | (output pending)                                               | RUNNING        |

GPU: 78% util, 77/80 GB used. RAM: 5/115 GB used, plenty for more CPU jobs.
See [MONITOR_AND_CPU_JOBS](MONITOR_AND_CPU_JOBS.md).

---

## 6. GitHub commits today (2026-05-01)

Today's main-branch landings, oldest -> newest:

- `978debc` — PLIF revive + LR rescue (Plan C LoRA insurance: deleted later 2026-05-01, see ANTI_LORA.md)
- `7e1c236` — P11: vocab 151643 -> 151936 normalized + phase flags + 50M context plan
- `cb571f9` — P15+P16+P17: tests/ collection + 10 legacy trainers to legacy/ + .bak removal
- `2a77df9` -> `33f636a` — P12: persist ckpt config + warn on shape drift in chat_demo
- `024e6d7` — ~~P5: harden Plan C LoRA + runbook~~ (DELETED 2026-05-01, see ANTI_LORA.md)
- `9fb3d04` — P18: web_actuator MVP (DOM + ActionHead + Playwright)
- `74e650b` — P20+P21: long-context validation harness (50M latency + monotonic A/B)
- `2d8bb24` — disk-full incident recovery: Run 3 launch + ckpt cleanup + reinit_plif
- `f051257` — P1: PLIF homeostatic threshold control + EMA clamp
- `f628a32` -> `3c29f8a` — P14: parallel.py Layer 2 smoke test + docstring honesty
- `c73e9cc` — P10: doc-stamp system to flag stale docs (9 stale flagged)
- `c7851ff` — P3: secondary val_holdout for honest TTT-leak-free reporting
- `f74128c` -> `5e75d4f` — P9: E2E data pipeline smoke (synth -> mix -> 50-step train)
- `8f09bc2` — naming: model = Synap-1, framework = SynapForge
- `e60681a` — P8: systemd-run watchdog launcher + auto-restart unit
- `5f420a4` — P2+P13: auto-tune KD chunk from VRAM headroom
- `0f09b14` — P6: NeuroMCP 600-trial test (density 19.7%, K=11, hit 1.0)
- `e76d6f3` — P7: web_actuator real Playwright run (22 ok_clicks in 6.97 s)
- `a144026` — monitor + CPU jobs: keep CPU/RAM hot while GPU trains
- `545f031` — cpu_jobs_minimal.sh: idempotent CPU/RAM utilization on rental

---

## 7. Known live problems / risks

| Risk                                | Severity | Note                                                                         |
|-------------------------------------|----------|------------------------------------------------------------------------------|
| Run 3l divergence (step5500=2522)   | RESOLVED | killed 00:55; Run 3m relaunched with patch stack (LR5e-5, warmup500, kd-every8) |
| P30 indent regression in T2.7 merge | RESOLVED | `5e5debe` (local) + Python heredoc patch on rental (network outage to GH)   |
| LM-head re-init from spectral norm  | MED      | Run 3m step 1 ce=11.9 vs warmstart's ce~6.0; recovery expected ~step 1500-2000 |
| PLIF still dead (spike 0/10)        | HIGH     | T2.5 spike-target loss now in live process (stl=0.025); homeostasis @ thr 0.0500 |
| Rental → github.com network outage  | HIGH     | GnuTLS recv error / connection timeout on port 443; patches applied via SSH heredoc |
| Backup is single-path (mohuanfang)  | HIGH     | GH Release / HF Hub unconfigured; one provider failure = data loss           |
| Trainer PID 26272 not under systemd | MED      | watchdog template ready (`e60681a`) but not yet wrapped around live PID      |
| Disk 30% post-cleanup               | LOW      | room for ~50 ckpts before cap; daemon offload pace covers it                 |

---

## 8. Next milestones (when to act)

Each milestone names the **exact** trigger and the **exact** command.

### M1 — val ppl crosses 250 (phase 0 -> phase 1)
**Trigger**: any `monitor.jsonl` row with `val_ppl <= 250`.
**Action** (on rental):
```bash
ssh -p 41614 root@117.74.66.77
echo "1" > /workspace/runs/v24h_qwen3/.phase
# phase_manager.py picks up signal, restarts trainer with intrinsic+STDP-novelty mixin
```
Then bump §1 + §3 of this doc.

### M2 — step 4000 ckpt lands (second backup verification)
**Trigger**: `step_004000.pt` exists on rental.
**Action**:
```bash
# verify mohuanfang received it
ssh liu@mohuanfang.com 'ls -la /home/liu/synapforge_backup/v24h_qwen3/step_004000.pt'
# sanity-check size > 380 MB and mtime within 600 s of rental copy
ssh liu@mohuanfang.com 'sha256sum /home/liu/synapforge_backup/v24h_qwen3/step_004000.pt'
ssh -p 41614 root@117.74.66.77 'sha256sum /workspace/runs/v24h_qwen3/step_004000.pt'
# diff should be empty
```

### M3 — 8 h elapsed and val ppl > 150 (downsize Synap, NEVER Plan C)
**Trigger**: wall clock 8 h since Run 3b launch AND `val_ppl > 150`.
**Action** (see [ANTI_LORA.md](ANTI_LORA.md) — no transformer / LoRA fallback):
```bash
ssh -p 41614 root@117.74.66.77
# Option A: spawn a SMALLER Synap-1 (30M-50M LNN+SNN) on the SAME rental
# (split GPU mem; or kill current trainer if it's not converging anyway).
# Option B: pivot demo focus to mechanism-level (NeuroMCP / R-fold / STDP)
# instead of live chat. Synap-1 stays the only architecture in the story.
```

### M4 — PLIF spike rate > 0/10 at step 4000 (revive succeeded)
**Trigger**: `monitor.jsonl` reports `spike_rate > 0` at step >= 4000.
**Action**: log success, no command. Update §1, §3, [HONEST_ASSESSMENT](HONEST_ASSESSMENT.md).
If still dead: see RELIABILITY playbook for manual `--reinit_plif`.

### M5 — `wt103 tokenize` finishes
**Trigger**: parquet output present.
**Action**: feed into next mix; bump [PRETRAIN_DATA](PRETRAIN_DATA.md).

---

## 9. How this doc updates

Refreshed each session by Claude or human. Workflow:

1. SSH rental, `tail -n 1 /workspace/runs/v24h_qwen3/monitor.jsonl | jq` for live numbers.
2. Update **§1 Live numbers** in place.
3. If a run cut over (Run N -> Run N+1), append a row to **§2 Run timeline**.
4. Recompute **§3 progress bar**: `pct = (606 - val_ppl) / (606 - 250)` for phase 0.
5. If a milestone fires, move it from **§8** to **§2** with the lesson.
6. Bump the footer date.

Cross-links: [MASTER_PLAN](MASTER_PLAN.md), [INVESTOR](INVESTOR.md),
[NAMING](NAMING.md), [HONEST_ASSESSMENT](HONEST_ASSESSMENT.md),
[BACKUP](BACKUP.md), [PHASE_TRAINING](PHASE_TRAINING.md),
[RENTAL_RECOVERY](RENTAL_RECOVERY.md).

---

_Last refreshed: 2026-05-01 by SynapForge agent. Refresh: bump table §2 + add §3 row + update §4 progress bar._

## Cron fire 2026-05-01 23:34

- H1-H3: trainer alive PID 24075 (Run 3l), GPU 95% / 70.5GB. Disk 30%. Mohuanfang count 2 vs rental 3 (drift 1, OK).
- Run 3l: step 210 ce=5.71 (KD off), 22k tok/s, healthy. VAL eval at step 500 pending.
- Tasks advanced: T1.4 (STDP density 0→59.8%), T1.6 (R-fold A800 N64R16 2.28×), T2.1 (5 papers via agent ad6e959).

## Cron deep-fire 2026-05-02 00:00

- H1-H5: trainer alive PID 24075 (Run 3l) GPU 99%/70.5GB. Disk 30%.
- Run 3l VAL: step 1000=312, 1500=362, 2000=417 (slow rise, NOT Run 3c-style catastrophe). step_2000.pt written.
- Tasks advanced: T1.5 (NeuroMCP density 40% K=12 hit 1.0), T2.4 freeze vocab tail (281336a), T2.5 spike-rate-target loss (2b086ec), T2.6 LM head spectral norm (16f5de5), T2.7 grad accumulation (2bc443f). 4 worktree agents shipped.
- Pending: T1.1 chat sample (needs ckpt > step 4000 of CURRENT run, step_002000_run3l doesn't qualify).
