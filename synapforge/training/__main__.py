"""synapforge.training.__main__ -- mode dispatcher entry point.

Usage::

    python -m synapforge.training --mode kd  --steps 1000 --out runs/kd
    python -m synapforge.training --mode sft --steps 5000 --out runs/sft
    python -m synapforge.training --mode rl  --steps 1000 --out runs/rl

This is the OPT-IN new entry point. The legacy ``train_100m_kd.py``
remains the production trainer (still invoked by the rental Run 7);
this dispatcher is only used when the caller explicitly chooses the
new core-trainer code path. See ``docs/TRAINER_REFACTOR.md`` for the
migration timeline.

Modes
-----
* ``kd``           -- Phase 0 KD distillation (KDTrainer).
* ``sft``          -- Phase 2 instruction-tune SFT (SFTTrainer).
* ``rl``           -- Phase 4 GRPO RL (currently stubbed; see
                      rl_trainer.py docstring).
* ``self_distill`` -- Phase 1.5 self-distill (currently stubbed; see
                      self_distill_trainer.py docstring).

The dispatcher intentionally builds the simplest model+optim+stream
triple sufficient to demonstrate the BaseTrainer plumbing. Production
users with custom data layouts (e.g. KD with pre-generated teacher
continuations) should construct the trainer programmatically::

    from synapforge.training.kd_trainer import KDTrainer, KDTrainerConfig
    cfg = KDTrainerConfig(...)
    trainer = KDTrainer(cfg, model, optim, train_stream, val_stream,
                        teacher=teacher)
    trainer.run()
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m synapforge.training",
        description="Synapforge multi-mode trainer dispatcher",
    )
    p.add_argument(
        "--mode",
        choices=["kd", "sft", "rl", "self_distill"],
        required=True,
        help="trainer mode to run",
    )

    # ---- shared knobs (mirror TrainerConfig) ----
    p.add_argument("--out", type=str, default="runs/synapforge",
                   help="output directory (TrainerConfig.out_dir)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--warmstart", type=str, default="",
                   help="path to a step_*.pt ckpt to warmstart from")

    # ---- model architecture ----
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--loop-depth", type=int, default=1)
    p.add_argument("--ffn-ratio", type=float, default=8.0)
    p.add_argument("--sparsity", type=float, default=0.95)

    # ---- KD-specific (only used when --mode=kd) ----
    p.add_argument("--kd-weight", type=float, default=0.4)
    p.add_argument("--kd-temperature", type=float, default=4.0)
    p.add_argument("--kd-chunk", type=int, default=0)
    p.add_argument("--kd-topk", type=int, default=2048)
    p.add_argument("--kd-every", type=int, default=1)

    # ---- SFT-specific (only used when --mode=sft) ----
    p.add_argument("--no-response-only-loss", dest="response_only_loss",
                   action="store_false", default=True,
                   help="ablation: full CE on all non-pad positions")

    # ---- data globs ----
    p.add_argument("--data-glob", type=str, default="",
                   help="train data parquet glob")
    p.add_argument("--val-glob", type=str, default="",
                   help="val data parquet glob")

    # ---- z-loss ----
    p.add_argument("--z-loss-weight", type=float, default=0.0)
    p.add_argument("--z-loss-topk", type=int, default=0)

    # ---- best ckpt ----
    p.add_argument("--best-ckpt-track", action="store_true", default=True)

    # ---- dry run ----
    p.add_argument("--dry-run", action="store_true",
                   help="parse args + build config; print + exit (no train)")

    return p


def _build_kd_config(args):
    from .kd_trainer import KDTrainerConfig
    return KDTrainerConfig(
        out_dir=args.out,
        steps=args.steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        accum_steps=args.accum_steps,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema_decay,
        warmstart_ckpt=args.warmstart,
        vocab=args.vocab,
        d=args.d,
        n_layers=args.n_layers,
        loop_depth=args.loop_depth,
        ffn_ratio=args.ffn_ratio,
        sparsity=args.sparsity,
        z_loss_weight=args.z_loss_weight,
        z_loss_topk=args.z_loss_topk,
        best_ckpt_track=args.best_ckpt_track,
        kd_weight=args.kd_weight,
        kd_temperature=args.kd_temperature,
        kd_chunk=args.kd_chunk,
        kd_topk=args.kd_topk,
        kd_every=args.kd_every,
    )


def _build_sft_config(args):
    from .sft_trainer import SFTTrainerConfig
    cfg = SFTTrainerConfig(
        out_dir=args.out,
        steps=args.steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        accum_steps=args.accum_steps,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema_decay,
        warmstart_ckpt=args.warmstart,
        vocab=args.vocab,
        d=args.d,
        n_layers=args.n_layers,
        loop_depth=args.loop_depth,
        ffn_ratio=args.ffn_ratio,
        sparsity=args.sparsity,
        best_ckpt_track=args.best_ckpt_track,
        response_only_loss=args.response_only_loss,
    )
    return cfg


def _build_rl_config(args):
    from .rl_trainer import RLTrainerConfig
    return RLTrainerConfig(
        out_dir=args.out,
        steps=args.steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        warmstart_ckpt=args.warmstart,
        best_ckpt_track=args.best_ckpt_track,
    )


def _build_self_distill_config(args):
    from .self_distill_trainer import SelfDistillTrainerConfig
    return SelfDistillTrainerConfig(
        out_dir=args.out,
        steps=args.steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        warmstart_ckpt=args.warmstart,
        best_ckpt_track=args.best_ckpt_track,
        ema_decay=max(args.ema_decay, 0.999),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    print(f"[dispatch] mode={args.mode} out={args.out} steps={args.steps}",
          flush=True)

    # Build the config FIRST so dry-run works without torch / data deps.
    if args.mode == "kd":
        cfg = _build_kd_config(args)
    elif args.mode == "sft":
        cfg = _build_sft_config(args)
    elif args.mode == "rl":
        cfg = _build_rl_config(args)
    elif args.mode == "self_distill":
        cfg = _build_self_distill_config(args)
    else:
        raise ValueError(f"unknown mode {args.mode!r}")

    if args.dry_run:
        print(f"[dispatch] dry-run: cfg={cfg}", flush=True)
        return 0

    # The actual model + stream construction is left to a higher-level
    # API (synapforge.finetune or the legacy train_100m_*.py scripts).
    # The dispatcher's job is just to validate the config; production
    # users should build streams + model themselves and invoke the
    # subclass programmatically. Document this clearly:
    print(
        "[dispatch] config built. To launch training, construct your "
        "model + train_stream + val_stream and invoke the subclass "
        "directly:\n"
        "    from synapforge.training.kd_trainer import KDTrainer\n"
        "    trainer = KDTrainer(cfg, model, optim, train_stream, "
        "val_stream, teacher=teacher)\n"
        "    trainer.run()\n"
        "Or use synapforge.finetune.run_sft(...) for the high-level "
        "API.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
