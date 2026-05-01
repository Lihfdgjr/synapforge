# Plan C — CPU Insurance Run Notes

Companion to [`PLAN_C_RUNBOOK.md`](PLAN_C_RUNBOOK.md). The runbook covers the
*GPU* path (30 min on A800). This file covers the **CPU-only** path that
runs in parallel to a busy GPU. Tracking: `docs/MASTER_PLAN.md` §6 P5.

> **Honest framing**: Plan C is the v0 demo frontend (Qwen 0.5B + LoRA), not
> the architecture claim. The architecture claim is Synap-1 (100M LNN+SNN).
> Plan C only ships if Synap-1 isn't chat-fluent in time.

---

## Why CPU?

The rental A800 is 100% occupied by Synap-1 training (`train_100m_kd.py`,
PID tracked in MASTER_PLAN §7). The CPU is mostly idle:

| Resource    | Used    | Free       | Plan C target           |
|-------------|---------|------------|-------------------------|
| 14 Xeon cores | ~2 cores (GPU dataloader) | ~12 cores | cores 8-13 (6 cores) |
| 110 GB RAM    | ~5 GB | ~105 GB    | ~3-4 GB resident       |
| Disk I/O     | low     | high       | nice -n 19, ionice idle |

Cores 0-7 stay reserved for GPU dataloader workers, NCCL, Python overhead.
Plan C never competes for them.

---

## ETA at CPU — honest numbers

The trainer has **no `--device` flag**; it auto-detects via
`torch.cuda.is_available()`. Forcing CPU is done with
`CUDA_VISIBLE_DEVICES=""`.

Estimated step time at `bs=4 max-seq=512` on 6 Xeon cores (no AVX-512 BF16):

- Forward: Qwen 0.5B is ~494M params. fp32 matmul on 6 cores ≈ ~80 GFLOPS
  effective. One forward at bs=4 seq=512 is ~1.0 TFLOPs → ~12-15 s.
- Backward (LoRA-only, ~0.3% trainable ≈ 1.5M params): ~1.5x forward → ~20 s.
- Optim step (AdamW on 1.5M params): negligible (<0.5 s).
- Data collation + tokenizer-pad: <0.5 s per batch.

**Per-step total: ~30-40 s.** Sample dump every 50 steps adds ~30 s.
Save every 100 steps adds ~5 s.

**200 steps wall-clock**: 200 × 35 s + 4 × 30 s sample + 2 × 5 s save
≈ 7,130 s ≈ **2 hours**.

If the rental Xeon doesn't have AVX-512 (most rentals don't), torch falls
back to AVX2 and you can roughly double that → **~4 hours per 200 LoRA
steps**. The original directive's "~12 h" estimate is the safe upper bound
for older Xeon E5-class (Haswell-era). On modern Skylake-X / Ice Lake the
realistic range is **2-6 hours**.

> **Pragmatic answer**: budget **4-6 hours per 200 steps** and be pleasantly
> surprised if it's faster. Loss should drop from ~3.0 to ~1.8 (Qwen
> Instruct is already chat-tuned; LoRA only nudges it).

---

## Decision tree

### When to **launch** Plan C CPU
- GPU trainer is in phase 0/1, ppl > 250, no sign of phase 1 trip soon, AND
- ≥4 hours of rental time budget remain, AND
- chat-eval gate at threshold 0.6 is the demo's hard requirement.

### When to **abandon** Plan C CPU mid-run
- GPU trainer hits phase 3 (`val ppl ≤ 60` per MASTER_PLAN §3): the native
  Synap-1 ckpt is the demo. Kill Plan C: `pkill -f train_qwen_lora.py`.
  Keep the partial `train.log` for honesty / paper appendix.
- Rental wall-clock approaches the rental-end deadline and Plan C hasn't
  reached step 100 (loss won't have dropped enough). Save what's there
  (`final.pt` is written every 100 steps via `save_every`), kill, and
  pivot to triple-backup before teardown.

### When to **escalate** Plan C CPU to GPU
- GPU trainer crashes and Synap-1 cannot recover (e.g. disk-full crash like
  Run 2's 99.9%-full event). Edit `launch_plan_c_cpu.sh` to drop the
  `CUDA_VISIBLE_DEVICES=""` line and re-run with `--bs 8 --max-seq 1024`.
  ETA collapses from hours to **30 min** on A800 (matches PLAN_C_RUNBOOK.md
  Step 3).

### When Plan C is **junk-binned**
- Synap-1 hits chat-eval pass rate ≥ 60% on the holdout (`docs/MASTER_PLAN.md`
  §3 phase 3 + §6 P5 acceptance criteria). Plan C output stays archived
  but is not promoted to `~/.synapforge/release/`.

---

## Acceptance criteria (Plan C CPU run "succeeds" if)

1. `train.log` shows monotonic loss decrease over 200 steps (final loss
   < 2.0 is a healthy signal).
2. `final.pt` exists at `$RUN_DIR/final.pt` with `framework`, `lora`, and
   `config` keys (verified by `chat_eval_gate.py`'s auto-detect).
3. `chat_eval_gate.py --ckpt $RUN_DIR/final.pt --threshold 0.6` returns
   `passed: true`.
4. Triple-backup daemon picks up the new `final.pt` (sha256 enters dedup
   cache).

When all four hold, MASTER_PLAN.md §6 P5 → RESOLVED.

---

## Quick reference

| Task | Command |
|------|---------|
| Launch | `bash /workspace/synapforge_git/scripts/launch_plan_c_cpu.sh` |
| Watch  | `tail -f /workspace/runs/plan_c_cpu/train.log` |
| Stop   | `pkill -f train_qwen_lora.py` |
| Eval   | `python3 /workspace/synapforge_git/scripts/chat_eval_gate.py --ckpt /workspace/runs/plan_c_cpu/final.pt --tokenizer-path /workspace/teachers/qwen2.5-0.5b --threshold 0.6` |
| Chat   | `python3 /workspace/synapforge_git/scripts/qwen_lora_chat_repl.py --adapter /workspace/runs/plan_c_cpu` |

---

## Cross-refs

- `docs/PLAN_C_RUNBOOK.md` — GPU-path runbook (30 min on A800).
- `docs/MASTER_PLAN.md` §6 P5 — tracking item.
- `scripts/launch_plan_c_cpu.sh` — the launcher this file documents.
- `scripts/train_qwen_lora.py` — trainer (do not modify; only orchestrate).
- `scripts/chat_eval_gate.py` — pass/fail gate.
- Memory: `feedback_mcp_remote_ssh_quirks.md`, `feedback_mcp_nohup_hangs_use_systemd_run.md`.
