# Rental Ops Runbook — Background Job Survival

P8 (MASTER_PLAN.md §6) RESOLVED 2026-05-01.

The MCP `proc_exec` shell on our rental box sometimes kills nohup'd background
children when the SSH session expires. This runbook tells you which detach
primitive to use for which scenario, and how to inspect/clean up after.

## TL;DR decision tree

| Scenario | Primitive | Auto-restart? | Tool |
|----------|-----------|---------------|------|
| One-shot smoke / 5-30 min job | `setsid + disown` | no | `launch_train_systemd.sh` falls back to this |
| Day-long run, you'll babysit | `systemd-run --user --unit=X` | no | `launch_train_systemd.sh` (default) |
| Overnight / unattended | `synapforge-trainer@.service` template | yes (`Restart=on-failure`) | install once, then `systemctl --user enable --now …` |

Memory cross-refs:
- `feedback_mcp_remote_ssh_quirks.md` — fs_read/fs_write hang via MCP; use proc_exec
- `feedback_mcp_nohup_hangs_use_systemd_run.md` — MCP channel doesn't release nohup
- `feedback_no_polling_loops_even_bg.md` — no `until ! check; do sleep` loops

---

## 1. `setsid + disown` (one-shot, good enough)

Use when the job is short enough that you don't care about restart semantics
and the box doesn't have user-systemd available.

```bash
setsid bash -c 'exec python3 -u my_script.py > /tmp/run.log 2>&1' </dev/null &
disown $!
```

Why both? `setsid` creates a new session so SIGHUP from the SSH/MCP shell
can't reach the child. `disown` removes the PID from the shell's job table
so the shell doesn't try to kill it on exit either.

Status: `ps -p <pid> -o pid,cmd` and `tail -f /tmp/run.log`.
Stop: `kill <pid>`; `kill -9 <pid>` if stuck.

The `setsid+disown` path is the fallback inside
`scripts/launch_train_systemd.sh` when systemd-run isn't reachable.

---

## 2. `systemd-run --user` (proper service status, no auto-restart)

Use when you want `systemctl status / stop / journalctl -f` and the rental
has user systemd. This is the **default path** in
`scripts/launch_train_systemd.sh`:

```bash
bash scripts/launch_train_systemd.sh \
  --name v24h_qwen3 \
  --warmstart /workspace/runs/v24h_qwen/step_002250_plif_reinit.pt \
  --out /workspace/runs/v24h_qwen3 \
  --steps 30000 \
  -- \
  --teacher Qwen/Qwen2.5-0.5B \
  --backend triton_block \
  --batch-size 64 \
  --kd-every 4 \
  --phase-aware
```

The launcher prints PID, log path, and the four management commands. The
unit is **transient** (no .service file persists), so a reboot or daemon
crash kills it permanently. Use the template unit (§3) for anything you
expect to survive a host reboot.

If `systemd-run --user` fails with `Failed to connect to bus`, the rental
is missing user-systemd. Run `loginctl enable-linger $USER` once and log
back in, or fall through to §1.

---

## 3. Template unit (auto-restart, survive reboots)

Use for overnight or unattended runs. The template is at
`scripts/synapforge-trainer.service.template` and is fully commented.

Install (one-time):

```bash
mkdir -p ~/.config/systemd/user
cp scripts/synapforge-trainer.service.template \
   ~/.config/systemd/user/synapforge-trainer@.service
systemctl --user daemon-reload
```

Drop the per-instance args (one file per run name):

```bash
mkdir -p ~/.config/synapforge-trainer
cat > ~/.config/synapforge-trainer/v24h_qwen3.env <<'EOF'
TRAINER_ARGS=--warmstart /workspace/runs/v24h_qwen/step_002250_plif_reinit.pt --teacher Qwen/Qwen2.5-0.5B --backend triton_block --batch-size 64 --kd-every 4 --steps 30000 --out /workspace/runs/v24h_qwen3 --phase-aware
EOF
```

Enable + start:

```bash
systemctl --user enable --now synapforge-trainer@v24h_qwen3.service
```

`Restart=on-failure RestartSec=30` means the trainer auto-restarts 30s
after any non-zero exit. `StartLimitBurst=5/StartLimitIntervalSec=600`
prevents an infinite crash-loop if the trainer is fundamentally broken
(5 fails in 10 minutes → systemd refuses to restart).

---

## 4. Daily inspection commands

```bash
# Which trainers are alive?
systemctl --user list-units --type=service | grep -i synap

# Status + last 20 log lines:
systemctl --user status synapforge-trainer@v24h_qwen3.service

# Follow the log live (Ctrl-C exits):
journalctl --user -u synapforge-trainer@v24h_qwen3.service -f

# Crash log only:
journalctl --user -u synapforge-trainer@v24h_qwen3.service -p err
```

---

## 5. Cleanup

After a run finishes (clean or crashed), tidy up so the next session
starts clean:

```bash
# Stop a transient or template unit:
systemctl --user stop synapforge-trainer@v24h_qwen3.service

# If `Restart=on-failure` was racing your stop:
systemctl --user disable synapforge-trainer@v24h_qwen3.service

# After 5 burst-failures the unit is in `failed` state — clear it:
systemctl --user reset-failed synapforge-trainer@v24h_qwen3.service

# Verify:
systemctl --user list-units --type=service --state=failed
```

For the `systemd-run` transient path, the unit auto-removes on stop, but
its journal entries persist. `journalctl --user --vacuum-time=7d` prunes
old logs if disk pressure becomes an issue (see also the
`v24h_qwen` Run 2 disk-full incident in MASTER_PLAN.md §7).

---

## 6. When to use what (one-line summary)

- 5-minute smoke: just `setsid+disown` (or run the launcher; it picks).
- Babysit-able run: `bash scripts/launch_train_systemd.sh --name X ...`.
- Overnight/unattended: install the template once, then enable per-run.

If in doubt, run the launcher — it picks the best primitive available on
the host and tells you which one it picked.

---

## 7. Auto-relauncher (phase-flip restart)

P22 (MASTER_PLAN.md §6) RESOLVED 2026-05-01.

The trainer's `--phase-aware` flag makes it write `<run_dir>/.phase` JSON
on threshold crossings (e.g. val ppl ≤ 250 → phase 1). Two ways to act
on that file:

| Path | Trigger | When to use |
|------|---------|-------------|
| `scripts/relaunch_loop.sh` | wraps the trainer; trainer exits 101 on transition; outer loop respawns | you start the trainer through this wrapper from the start |
| `scripts/phase_auto_relauncher.sh` | runs **alongside** the trainer; polls `.phase`; SIGTERMs+restarts | you already started the trainer some other way and want a watchdog |

**Use one or the other**, not both — they would double-spawn.

### `phase_auto_relauncher.sh` usage (P22)

```bash
bash scripts/phase_auto_relauncher.sh \
  --run-dir /workspace/runs/v24h_qwen3 \
  --interval 60
```

What it does on a phase change (N → N+1):
1. Read `<run_dir>/.phase` (jq, fall back to python3).
2. Pick the latest `phase_change_step_*.pt` (preferred) or newest `step_*.pt`.
3. Strip optim_state inline (stale Adam momentum cripples post-vocab-change
   warmstart — see `feedback_no_random_init_use_warmstart.md`). Output:
   `<ckpt>_no_optim.pt`.
4. SIGTERM current trainer (10 s grace, then SIGKILL).
5. Re-launch with `setsid + disown` (per
   `feedback_mcp_remote_ssh_quirks.md`), preserving the existing argv but:
   - dropping `--no-warmstart`
   - replacing `--warmstart`'s value with the stripped ckpt
   - appending the new phase's flags (mirror of `phase_manager.PHASES`)
6. Log every restart to `<run_dir>/phase_restart.log` with timestamp +
   old phase + new phase + new flags + new PID.

Safety guards:
- 5-minute anti-thrash debounce (`PAR_THRASH_SEC`, default 300).
- If trainer is dead (no PID), log + skip; do not auto-spawn.
- Drop `--no-warmstart` so the new instance loads the just-saved ckpt.

Phase 1 flag string (matches `scripts/phase_manager.py PHASES[1]`):
```
--self-learn-ttt --self-learn-k 8 --curiosity-weight 0.05 --phase-aware
```

Tests: `tests/integration/test_phase_auto_relauncher.py` covers `bash -n`
syntax, dry-run argv composition, the no-PID safety guard, the no-`.phase`
poll, and a drift-guard against `phase_manager.PHASES`.

Combine with `scripts/launch_train_systemd.sh` so the watchdog itself
runs as a systemd-run unit (you'll have two units: trainer + relauncher).
