# Trainer Refactor v2 — migration guide

The four trainer entry points (`train_100m_kd.py`, `train_100m_sft.py`,
`train_100m_rl.py`, `train_100m_self_distill.py`) duplicate ~80% of their
plumbing: argparse, ckpt save/load, val loop, log format, optimizer build,
warmstart resume, EMA, gradient clipping, phase-aware monitoring. Adding a
feature requires editing 4 files.

This refactor extracts the shared core into `synapforge/training/` and
makes each mode a thin subclass of `BaseTrainer`. The legacy scripts are
**left UNTOUCHED** (the rental Run 7 is currently running
`train_100m_kd.py` and we don't disturb it).

## What's new

```
synapforge/training/
  core_trainer.py          BaseTrainer + TrainerConfig + utilities
  kd_math.py               kd_loss / kd_topk_loss / kd_chunk_size
                           (lifted bit-exact from train_100m_kd._kd_loss)
  kd_trainer.py            KDTrainer(BaseTrainer) + KDTrainerConfig
  sft_trainer.py           SFTTrainer(BaseTrainer) + SFTTrainerConfig
  rl_trainer.py            RLTrainer (STUB — see migration path)
  self_distill_trainer.py  SelfDistillTrainer (STUB — see migration path)
  __main__.py              python -m synapforge.training --mode {kd,sft,rl,...}

synapforge/finetune.py     High-level run_sft / run_kd API
tests/training/            Unit tests + bit-exactness check
```

## Bit-exactness contract

`KDTrainer` produces the SAME loss as `train_100m_kd.py` given the same
`(student_logits, teacher_logits, x, y)` tuple. The contract is verified
by `tests/training/test_core_trainer.py::test_kd_math_bitexact` which
compares `synapforge.training.kd_math.kd_loss` against
`train_100m_kd._kd_loss` on synthetic logits at 1e-5 tol, exercising
both the top-K (`topk=128`) and chunked-full-vocab (`topk=0`) paths.

Math:

```
base_loss = CE(logits, y, label_smoothing) + z_w * z_loss
if step % kd_every == 0 AND teacher AND kd_weight > 0:
    kd_term = kd_loss(student_logits, teacher_logits, T, chunk, topk)
    loss    = (1 - kd_weight) * base_loss + kd_weight * kd_term
else:
    loss    = base_loss
```

## How to use the new dispatcher

```bash
python -m synapforge.training --mode kd  --steps 1000 --out runs/kd
python -m synapforge.training --mode sft --steps 5000 --out runs/sft \
    --warmstart runs/kd/best_step_001000.pt
```

The dispatcher builds the appropriate `*TrainerConfig` from argparse and
validates with `--dry-run`. Production users construct the trainer
programmatically:

```python
from synapforge.training.kd_trainer import KDTrainer, KDTrainerConfig
cfg = KDTrainerConfig(out_dir="runs/kd", steps=1000, kd_weight=0.4)
trainer = KDTrainer(cfg, model, optim, train_stream, val_stream,
                    teacher=teacher_model)
trainer.run()
```

Or use the high-level finetune API for SFT:

```python
import synapforge.finetune
synapforge.finetune.run_sft(
    ckpt="runs/kd/best_step_010000.pt",
    data="data/alpaca_zh.parquet",
    out="runs/sft_v1",
    steps=5000,
)
```

## When to use each entry point

| Entry point | When to use |
|-|-|
| `python -m synapforge.training --mode kd` | New KD runs (notebook driving, ablation sweeps) |
| `python -m synapforge.training --mode sft` | New SFT runs |
| `python train_100m_kd.py` (legacy) | Production Run 7 + everything that depends on it |
| `python train_100m_sft.py` (legacy) | Production phase-2 runs that rely on `--cross-val` etc. |
| `python train_100m_rl.py` (legacy) | RL runs (BaseTrainer doesn't yet support rollouts) |
| `python train_100m_self_distill.py` (legacy) | Self-distill (BaseTrainer needs an EMA-shadow forward hook) |

## Future work — full migration

To replace the four legacy scripts with the new dispatcher, the following
need to land:

1. **Rollout hook on BaseTrainer** — `train_100m_rl.py` does N rollouts
   per "step" before computing a loss. `BaseTrainer` would need a
   `rollout(batch) -> trajectories` hook called before `compute_loss`,
   plus a reference-policy snapshot mechanism for the KL constraint.

2. **EMA-shadow forward hook on BaseTrainer** — `train_100m_self_distill.py`
   forwards the same batch through both the live model and an EMA-decayed
   shadow. `compute_loss` would need access to both forwards. Easiest
   path: trainer holds an `ema_model` reference (lazy-built from
   `ema_state` when warm enough), exposes a `model_ema_forward(x)` helper.

3. **Mixin support on `compute_loss`** — `train_100m_kd.py` uses three
   optional contributions (modal contrastive, ICM curiosity, neuromcp
   action loss) gated by argparse flags. The new `KDTrainer.compute_loss`
   doesn't expose them. Either:
   - port the mixin protocol straight (each is `nn.Module` with a single
     extra-loss method), OR
   - keep mixins as a `train_100m_kd.py` feature and only support the
     core CE+KD+z_loss in `KDTrainer`.

4. **--data-files multi-glob support** — production launcher uses
   `--data-files "fineweb_edu*.parquet|wt103_raw/*.parquet|..."` (the
   2026-04-30 quality-corpus pipeline). The dispatcher doesn't yet wire
   this up; the high-level API only takes a single glob.

5. **Honest-eval hook integration** — `train_100m_kd.py` mounts
   `scripts/honest_eval_hook.py` to run real BLEU/ROUGE on val. Not yet
   wired into BaseTrainer.

Estimated effort: ~6h to land 1 + 2; ~2h for 3 + 4; ~1h for 5. Once all
land, the legacy `train_100m_*.py` files can be turned into thin shims
(or removed entirely after a deprecation window).

## Test results (2026-05-02)

```
$ pytest tests/training/test_core_trainer.py -v
test_trainer_construct PASSED
test_train_step_deterministic PASSED
test_ckpt_save_load_roundtrip PASSED
test_kd_math_bitexact PASSED              <-- bit-exact vs legacy
test_mode_dispatch_dry_run PASSED         <-- all 4 modes
test_kd_trainer_no_teacher_is_pure_ce PASSED
=== 6 passed in 14.6s ===
```
