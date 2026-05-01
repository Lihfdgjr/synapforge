# Monitor + CPU Jobs Runbook

Two ops tools that pair with the live trainer:

1. `scripts/monitor_run.py` — runs **on the operator's laptop**, polls the
   rental every N minutes via SSH, dumps structured JSONL + a one-line
   summary, and raises four alerts.
2. `scripts/cpu_utilization_jobs.sh` — runs **on the rental**, fills the
   ~half-idle CPU + RAM with useful pre-processing + eval work while the
   GPU trainer is busy.

Cross-refs: [RENTAL_OPS.md](RENTAL_OPS.md) (P8 nohup/systemd-run rules),
[AUTO_EVAL.md](AUTO_EVAL.md) (the bench daemon we re-use), [BACKUP.md](BACKUP.md).

---

## 1. Launch the monitor (laptop)

```bash
# Optional: paramiko for cleaner SSH; works without it via openssh CLI.
pip install paramiko

# Optional: password auth fallback (key auth preferred).
export RENTAL_SSH_PASSWORD='FM6CSP@xjdkdi#I'

python scripts/monitor_run.py \
    --host 117.74.66.77 --port 41614 --user root \
    --run-dir /workspace/runs/v24h_qwen3 \
    --interval 300 \
    --out monitor.jsonl
```

stdout prints one line per cycle, e.g.:

```
[19:40] step=1520 ce=5.78 ppl=461 spike=0.0 dead=10/10 gpu=78% disk=28% backup=#3-syncing
```

`monitor.jsonl` accumulates one JSON object per cycle. Sample row:

```json
{"ts":"2026-05-01T11:40:00+00:00","trainer_pid":16697,"trainer_alive":true,
 "step_lines":[{"step":1500,"ce":5.81,"kd":0.00,"z":0.001,"lr":6e-05,"tok_s":200,"mem_gb":4.7},
               {"step":1510,"ce":5.79,"kd":0.00,"z":0.001,"lr":6e-05,"tok_s":201,"mem_gb":4.7},
               {"step":1520,"ce":5.78,"kd":0.00,"z":0.001,"lr":6e-05,"tok_s":200,"mem_gb":4.7}],
 "spike_lines":[{"mean":0.000,"range_lo":0.0,"range_hi":0.0,"dead":10,"denom":10,"sat":0}],
 "val_ppl":461.34,
 "gpu":{"util_pct":78,"mem_used_mib":77345,"mem_total_mib":81920},
 "cpu_loadavg_1m":2.4, "ram":{"total_gb":196.6,"used_gb":31.2,"pct":15},
 "backup":{"cycle":3,"status":"syncing","ts":"2026-05-01T19:30:01"},
 "ckpt_count":4,"latest_ckpt":"/workspace/runs/v24h_qwen3/step_001500.pt",
 "latest_ckpt_step":1500,"disk_pct":28,
 "alerts":["PLIF_STUCK"]}
```

### Grep alerts

```bash
# All cycles where any alert fired
grep -E '"alerts":\["[A-Z]' monitor.jsonl | head

# Just the timestamps of CE rises
python -c "
import json,sys
for line in open('monitor.jsonl'):
    r = json.loads(line)
    if 'CE_RISING' in r.get('alerts',[]):
        print(r['ts'], r['step_lines'][-1] if r.get('step_lines') else '')"
```

---

## 2. Launch CPU jobs (rental)

```bash
ssh -p 41614 root@117.74.66.77

cd /workspace/synapforge_git
bash scripts/cpu_utilization_jobs.sh \
    --run-dir /workspace/runs/v24h_qwen3 \
    --data-dir /workspace/data
```

What it does (skipping any output that already exists):

| Tag | Job | Output | Approx CPU-h |
|-----|-----|--------|--------------|
| (a) | synth zh + mix phase 1 | `data/synth_zh_phase1.parquet` + `data/mix_phase1.parquet` | ~1h |
| (b) | tokenize alpaca-zh | `data/alpaca_zh_qwen_tokenized.parquet` | ~30 min |
| (c) | heavy bench loop (every 30 min on the latest ckpt) | `auto_eval/<step>/bench_heavy.json` | continuous |
| (d) | pre-warm Qwen tokenizer | `/dev/shm/qwen_tok.pkl` | ~10 s |

All jobs:

* run as `nice -n 19 ionice -c 3 taskset -c 8-31 ...` so they yield to
  the trainer's dataloader (pinned to CPUs 0-7) — see `--pin-cpus` to override.
* are launched as `nohup setsid bash -c '...' &` then `disown`'d, so the
  SSH/MCP channel can close (RENTAL_OPS.md P8) without killing them.
* log to `$RUN_DIR/cpu_jobs.log` (coordinator) and `$RUN_DIR/cpu_job_<name>.log` (per-job).

Re-running the script is safe: existing outputs are skipped, and the bench
loop is single-instance (guarded by `$RUN_DIR/.cpu_bench_loop_started`).

If `cpulimit` is not installed the script falls back to `nice -n 19 ionice -c 3`.

---

## 3. Alerts — first action

| Alert | Meaning | First action |
|-------|---------|--------------|
| `CE_RISING` | CE went up over the last 3 step samples in a row | Tail `train.log`; if it persists >10 cycles, check LR scheduler + recent ckpt size; consider rolling back to the previous best |
| `PLIF_STUCK` | All PLIF cells dead for >2000 training steps | Run `scripts/reinit_plif.py` on the latest ckpt and warmstart; see [RELIABILITY.md](RELIABILITY.md) §PLIF |
| `BACKUP_FAILED` | The backup daemon's last cycle wrote `status=failed` | `tail $RUN_DIR/backup.log`; check mohuanfang reachability with `scripts/rental_health.py status` |
| `DISK_FULL_RISK` | `df -h /` shows >=90% used on the system disk | Run `bash scripts/ckpt_cleanup.sh` to keep last 5 step + all best + all phase_change ckpts; cf. `launch_v24h_qwen3.sh` header for the 2026-05-01 incident |

---

## 4. Why this exists

The A800 trainer pegs the GPU at 75-90% but only uses ~25 GB / ~200 GB RAM
and 4-8 of 16-32 cores. Without (cpu_utilization_jobs.sh) we'd later
serialize phase-1 zh synth + alpaca tokenization + heavy bench AFTER the
training run, wasting wall-clock on the next phase.

`monitor_run.py` was added because the operator's laptop is the only place
that can see all four signals at once: trainer log + nvidia-smi + backup
daemon + disk. The watchdog in `rental_watchdog.sh` answers "is the box
reachable?"; this monitor answers "is training healthy?".
