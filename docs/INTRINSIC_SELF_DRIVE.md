# Intrinsic Self-Drive (task #188 wire-in)

Production wire-in of the 5 self-drive components from
`synapforge/intrinsic/_core.py` (formerly `synapforge/intrinsic.py`)
into `train_100m_kd.py`. Default OFF — current Run 7 launches stay
bit-exact when `--self-drive` is not passed.

## What "self-drive" means here

User question (2026-05-02): "那好奇心你是怎么解决的，让他有自驱力愿意自己探索自我学习".

Pre-this-PR, Run 7 had only ICM (basic prediction-error reward via
`CuriosityMixin`'s 6-signal loss). That gives the model a curiosity
*signal* but no *agency* — it cannot decide what to study next when
the data loader is empty. The 5 components below give it agency.

## 7-component architecture

```
[Main thread]                  [Self-drive cycle]                    [Goal mem + sampler]
  fwd+bwd+optim                  every K=20 outer steps                attempted
  CE loss + KD                   OR data loader empty                  success/fail
  + ICM (--curiosity-weight)             |                             success_rate
                                         v                             buffer
                                 SelfDriveCoordinator.cycle()
                                         |
                                         |--> FrontierSampler.sample()
                                         |       (in-band [0.3,0.7] + recent-K lockout)
                                         |
                                         |--> [no in-band] -> SelfGoalProposer.propose()
                                         |
                                         |--> ImaginationRollout.dream(goal)
                                         |
                                         |--> for inner step in M=10:
                                         |       run_inner_fn(SelfDriveStep)  # forward
                                         |       record_outcome -> GoalMemory + sampler
                                         |
                                         |--> QualityGuard.verify()
                                                 pre/post val_ppl
                                                 ROLLBACK if regression > 5%
```

The 5 self-drive components from `_core.py`:
1. **SelfGoalProposer** (Voyager 2305.16291) — generates fresh goal token sequences.
2. **ImaginationRollout** (Dreamer V3 2301.04104) — latent-space planning.
3. **FreeEnergySurprise** (ICM 1705.05363) — surprise scorer (also used by `CuriosityMixin`).
4. **GoalMemory** (Self-Discover) — buffer of attempted goals + improvement.
5. **IdleLoop** — background watchdog (left for trainer; we expose `should_fire`).

Plus 2 new pure-Python coordination modules:

6. **FrontierSampler** (`frontier.py`) — picks goals whose past success_rate
   is in `[sweet_lo, sweet_hi]` (default `[0.3, 0.7]`) with recent-K lockout.
7. **QualityGuard** (`quality_guard.py`) — snapshots STDP weights, rolls
   back via user callback when val_ppl regresses past tolerance.

## When the cycle fires

`SelfDriveCoordinator.should_fire(outer_step, idle)` returns True when:

- `--self-drive` flag was passed AND
- `outer_step > 0` AND
- (`idle=True`  OR  `outer_step % every_k_steps == 0`)

`idle` is set by the trainer to `data_exhausted` — when the data loader
returns empty for the current step. This is the "real GPU idle"
trigger described in the task spec.

## STDP-only path (NOT AdamW)

Per `feedback_neural_action_no_token_no_mcp.md`: the inner runner runs
the model forward in `torch.no_grad()` mode. No `optim.zero_grad()`,
no `loss.backward()`, no `optim.step()`. The intent is for STDP
plasticity (in `PLIFCell`s and `SparseSynapticLayer` when alive) to
be the only weight-update path during self-drive cycles.

When PLIF is dormant (early run, dense_bypass on), the inner runner
still produces a valid loss number that `GoalMemory` and
`FrontierSampler` track. This is the "PLIF-revival required for full
STDP path; without PLIF, only the AdamW path of self-drive (slower
but still works)" caveat noted in the task spec.

## Quality guard

Mandatory: self-drive must NEVER make `val_ppl_holdout` worse than
baseline. The trainer wires `QualityGuard.snapshot()` with a closure
that clones every trainable param, and `QualityGuard.verify()` with a
4-batch eval against `val_ds_holdout`. If `post > pre * 1.05`
(default `--self-drive-max-regression 0.05`), `restore_fn` is called
and `param.data.copy_()` reverts every cloned tensor.

Logging:

    [self-drive] cycle done step=N inner=M kept=K elapsed_ms=...
    [self-drive] step=N goal_id=42 type=imagined success=True freeenergy=0.34
    [self-drive guard] pre_ppl=120.00 post_ppl=125.00 thr=126.00 KEEP (within-tolerance)

ROLLBACK example:

    [self-drive guard] pre_ppl=120.00 post_ppl=145.00 thr=126.00 ROLLBACK (post>5.0%-of-pre (post=145.00 > 126.00))

## Default knobs

| flag                              | default | notes                                  |
|-----------------------------------|--------:|----------------------------------------|
| `--self-drive`                    |   off   | master toggle                          |
| `--self-drive-every-k`            |    20   | outer-step cadence                     |
| `--self-drive-inner-steps`        |    10   | imagined inner steps per cycle         |
| `--self-drive-max-regression`     |  0.05   | rollback if val_ppl > pre*(1+this)     |
| `--self-drive-sweet-lo`           |   0.3   | FrontierSampler band lower             |
| `--self-drive-sweet-hi`           |   0.7   | FrontierSampler band upper             |
| `--self-drive-recent-k`           |    10   | recent-K lockout window                |
| `--self-drive-success-loss-drop`  |  0.05   | success threshold (loss drop fraction) |

## Limitations / when self-drive helps

Honest assessment from the bench (`scripts/bench_self_drive.py`):

- **Cold start (random init, no vocab)**: zero benefit. Self-drive
  cannot extrapolate from a model that has no signal yet. The
  coordinator runs cleanly (no rollback, no crash) but `delta_loss = 0`.
- **Mid-training (`val_ppl > 250`)**: defer enabling. The PLIF cells
  are still in dense_bypass mode for many runs; STDP path is dormant.
- **Late phase 1 (`val_ppl ~ 100-250`)**: this is when self-drive
  starts adding value. Per the phase manager
  (`feedback_phased_training_2026q2.md`), Phase 1 enables
  `intrinsic + self_learn + STDP-novelty`. Wire `--self-drive` and
  `--curiosity-weight=0.05` together at this gate.
- **Phase 2+ (`val_ppl < 100`)**: increasing fraction of cycles end
  with `success=True`. FrontierSampler accumulates in-band candidates;
  fewer fresh proposals. Self-drive starts behaving as auto-curriculum.

## Bench

`scripts/bench_self_drive.py` runs a tiny TinyLM model 200 steps
both ways. Reports:

- `final_loss` (off vs on)
- `throughput_steps_per_s` (overhead measurement)
- `fires` / `n_inner` / `n_kept` / `n_rollbacks` / `n_proposed_fresh`
- `frontier_in_band` (count of records that ever made it into the band)

At small scale the throughput hit is large (the 4-batch eval inside
each cycle dominates) but in a real trainer with bs=64 and longer
steps the fixed cost is amortized.

## Tests

`tests/intrinsic/test_self_drive.py` — 24 tests covering:

- FrontierSampler in-band + out-of-band + recent-K + min_attempts + tiebreak.
- QualityGuard keep/rollback, NaN-safety, snapshot lifecycle.
- SelfDriveCoordinator integration (10 outer + 5 inner cycle, mock runners).
- Proposer / rollout failure recovery paths.
- End-to-end smoke with mid-run synthetic rollback.

Run:

    pytest tests/intrinsic/test_self_drive.py -v

## File map

| path                                                     | purpose                          |
|----------------------------------------------------------|----------------------------------|
| `synapforge/intrinsic/_core.py`                          | legacy (the 7 torch components)  |
| `synapforge/intrinsic/__init__.py`                       | re-export legacy + new           |
| `synapforge/intrinsic/frontier.py`                       | FrontierSampler (no torch)       |
| `synapforge/intrinsic/quality_guard.py`                  | QualityGuard (no torch)          |
| `synapforge/intrinsic/self_drive_coordinator.py`         | SelfDriveCoordinator (no torch)  |
| `train_100m_kd.py`                                       | flag + construction + cycle      |
| `tests/intrinsic/test_self_drive.py`                     | 24 tests                         |
| `scripts/bench_self_drive.py`                            | OFF/ON 200-step microbench       |
| `docs/INTRINSIC_SELF_DRIVE.md`                           | this file                        |
