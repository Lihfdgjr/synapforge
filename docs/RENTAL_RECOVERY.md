# Rental Recovery Runbook

After the 2026-04-30 v4.1 disaster (rental sshd saturated, 7h training
lost, ckpt unrecoverable for hours), every rental we use is now under
external watch from mohuanfang.com. This document explains why, and what
to do when it fires.

The current rental as of 2026-05-01 is `117.74.66.77:41614` (root). The
backup site is `liu@mohuanfang.com:/home/liu/synapforge_backup`. Both
sets of credentials live in the operator's memory file
`reference_rental_a100x2_ssh.md`, never in this repo.

---

## Architecture

```
   +-----------------+        cron */5         +------------------+
   |  mohuanfang.com | -------- ssh ---------> |  rental box      |
   |   (1.5T disk,   |    rental_watchdog.sh    |  (A800 80GB,     |
   |   real Linux)   | <----- best*.pt -------- |   training run)  |
   +-----------------+      emergency scp        +------------------+
            ^
            |   ssh
            |
   +-----------------+
   |  operator laptop |  rental_health.py status / --watch
   +-----------------+
```

**Why the watchdog cannot live on the rental.** When the rental's sshd
saturates the box stops accepting connections but the GPU may still be
running. A self-watchdog cannot signal anything we can read. Mohuanfang
is an external, persistent host with its own disk and outbound SSH —
exactly the role the v4.1 disaster taught us we need.

**Why not directly from the operator laptop.** Laptops sleep, change
networks, and the operator may be offline for 12+ hours during a
training run. Mohuanfang is up 24/7, so it sees every 5-minute tick.

**MCP-side reminder.** Per `feedback_mcp_shell_kills_bg_jobs.md`, MCP
`proc_exec` on the rental kills our backgrounded jobs. The watchdog
must run on mohuanfang via real `cron`, not via any MCP-launched
background.

---

## Files

| Path | Where it runs | Purpose |
|---|---|---|
| `scripts/rental_watchdog.sh` | mohuanfang | cron tick, ssh ping, escalation |
| `scripts/install_watchdog_cron.sh` | mohuanfang (one-shot) | bootstrap installer |
| `scripts/rental_health.py` | operator laptop | status check + manual recovery |
| `/tmp/synapforge_rental_health.json` | mohuanfang | watchdog state, JSON |
| `/home/liu/synapforge_backup/.healthy` | mohuanfang | last-success heartbeat |
| `/tmp/.rental_dead_ts` | mohuanfang | written when rental declared dead |
| `/var/log/synapforge_watchdog.log` | mohuanfang | append-only log, all ticks |

---

## Failure modes & detection

| Mode | Detection signal | Auto mitigation | Manual escalation |
|---|---|---|---|
| sshd saturated (v4.1 cause) | ssh ping timeout | n=3 emergency scp; n=5 dead flag | reboot via 算力牛 console |
| GPU hung but ssh alive | (not detected by watchdog) | none | log into rental, `nvidia-smi`, restart trainer |
| Network down (rental side) | ssh ping timeout, identical to sshd | n=3 emergency scp (will fail); n=5 dead flag | wait or reboot |
| Network down (mohuanfang side) | watchdog logs `ssh: connection refused` repeatedly | none | check mohuanfang's own uptime; operator probes from laptop |
| Disk full on rental | ckpts stop growing (need separate alarm; out of scope) | none | ssh in, find/clean |
| Power loss / instance terminated | ssh permanently dies | n=3 evacuation likely fails; n=5 dead flag | spin new instance, restore from latest mohuanfang mirror |
| Contract expired | ssh dies precisely at expiry | same as power loss | renew contract, or migrate |

The watchdog only resolves the first/third/sixth/seventh rows
automatically; the others need a human seeing the log or status output.

---

## Escalation cadence

```
fail # 1   log only
fail # 2   log only (counter increments)
fail # 3   notify WARN  + emergency scp best*.pt   (best-effort)
fail # 4   log only
fail # 5   notify DEAD  + write /tmp/.rental_dead_ts   (manual takeover)
fail # 6+  log only — already declared dead
```

Recovery (any successful ssh ping) resets the counter to 0 and removes
the dead flag.

---

## Setup on mohuanfang

```bash
# one-shot, on mohuanfang
git clone https://github.com/Lihfdgjr/synapforge && cd synapforge
sudo bash scripts/install_watchdog_cron.sh

# then edit credentials/glob if defaults don't match the current rental
$EDITOR /home/liu/synapforge_backup/watchdog/watchdog.env

# verify
crontab -l                                       # confirm */5 line
tail -f /var/log/synapforge_watchdog.log         # watch first tick
```

Credentials never go in code. The installer writes a template
`watchdog.env` (mode 0600) for the operator to fill in. It expects the
same keys `rental_watchdog.sh` reads:

```bash
WATCHDOG_RENTAL_HOST=117.74.66.77
WATCHDOG_RENTAL_PORT=41614
WATCHDOG_RENTAL_USER=root
WATCHDOG_BACKUP_DIR=/home/liu/synapforge_backup
WATCHDOG_REMOTE_GLOB=/workspace/runs/v24h_qwen/best*.pt
# WATCHDOG_SLACK_URL=https://hooks.slack.com/services/...
# WATCHDOG_NOTIFY_EMAIL=demery_guernsey032@mail.com
```

The watchdog uses `BatchMode=yes` SSH options; the `liu@mohuanfang`
account must hold an `id_rsa` registered on the rental's
`/root/.ssh/authorized_keys`. (This was set up during the recovery from
the v4.1 disaster; see `reference_mohuanfang_backup.md`.)

---

## From the operator laptop

```bash
# quick check
python3 scripts/rental_health.py status

# live dashboard
python3 scripts/rental_health.py status --watch

# rental looks dead — manually pull every best*.pt to local laptop
python3 scripts/rental_health.py emergency-recover --to D:\ai_tool\rental_evac
```

`status` exit codes are scriptable:

| Exit | Meaning |
|---|---|
| 0 | OK — recent heartbeat, 0 fails |
| 1 | WARN — heartbeat stale or mid-escalation |
| 2 | DEAD — `/tmp/.rental_dead_ts` is set; runbook below |
| 3 | NO CONTROL PLANE — cannot ssh to mohuanfang itself |

---

## Disaster runbook: rental flagged DEAD > 1h

If `rental_health.py status` reports DEAD and the dead flag is older
than ~1 hour, the rental is not coming back without intervention.
Execute these in order:

1. **Snapshot what we already have.**
   ```bash
   ssh liu@mohuanfang.com 'ls -la /home/liu/synapforge_backup/emergency_*'
   ```
   The watchdog's fail #3 emergency scp may have left a directory; if
   so, those `best*.pt` are your starting point.

2. **Check 算力牛 web console.** URL is the operator's account
   dashboard at https://www.suanlibao.com/console (login required).
   Look at the rental instance status. Three common states:
   - **Running, 0 traffic** → sshd is wedged. Issue a hard reboot from
     the console. Wait 5 min, retry `rental_health.py status`.
   - **Stopped / Suspended** → contract may have expired or been
     terminated by the provider. Open a support ticket.
   - **Errored / Migrating** → 算力牛 is moving the instance (notice
     was given for the 2026-05-01 datacenter move). Wait per their
     announcement; do not provision a replacement until they confirm.

3. **If SSH unreachable > 24h**, file a refund ticket with 算力牛
   support. Per their TOS, partial credit is granted for instances
   unreachable for more than 24 contiguous hours. Provide the
   `synapforge_watchdog.log` excerpt (timestamps + fail counts) as
   evidence — that file is exactly the audit trail support asks for.
   Support contact lives on the suanlibao console under "工单".

4. **Spin up replacement instance** (only after refund ticket is filed
   so it doesn't get rolled into a single billing event). Pull the
   latest known-good `best*.pt` from mohuanfang and warmstart per
   `docs/BACKUP.md` recovery section. Per
   `feedback_no_random_init_use_warmstart.md`, NEVER train from random
   init — recover the most recent ckpt, even if its ppl is suboptimal.

5. **Re-point the watchdog.** Once the new rental's IP/port are known,
   edit `watchdog.env` on mohuanfang and reset the counter:
   ```bash
   ssh liu@mohuanfang.com '
     rm -f /tmp/.rental_dead_ts /tmp/synapforge_rental_health.json
     vi /home/liu/synapforge_backup/watchdog/watchdog.env   # update host/port
   '
   ```
   The next 5-minute tick will start a fresh observation cycle.

---

## What this watchdog does NOT do

- It does not detect a wedged GPU when sshd is still up. The trainer
  itself logs to mohuanfang every 10 min via `triple_backup_daemon.py`;
  if those backups stop arriving, that is your GPU-hang signal.
- It does not detect quiet data corruption on the rental's disk. The
  triple backup uses content hashing (`docs/BACKUP.md`); compare hashes
  if you suspect bit-rot.
- It does not pay your bill. Contract expiry looks identical to a hard
  failure; the runbook step 2 covers this.
- It does not warm-start a replacement automatically. The replacement
  rental has its own IP/port and the operator must edit `watchdog.env`.

---

## See also

- `docs/BACKUP.md` — the inbound-from-rental triple backup daemon
- `auto_ckpt_backup.py` — periodic best-ppl uploader (rental-side)
- memory `reference_mohuanfang_backup.md` — mohuanfang account, disk
  layout, credentials
- memory `reference_rental_a100x2_ssh.md` — current rental's
  credentials and the dead OLD rental we need not to lose access to
  again
- memory `feedback_mcp_shell_kills_bg_jobs.md` — why the watchdog must
  use real cron, not MCP-launched background jobs
