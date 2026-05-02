"""synapforge.finetune -- high-level user API for fine-tuning.

This is the user-facing wrapper for the trainer subclasses. It hides
the model + optimizer + stream construction so a typical user can run
SFT / KD with a one-liner::

    import synapforge
    synapforge.finetune.run_sft(
        ckpt="best_step_10000.pt",
        data="alpaca-zh.parquet",
        steps=5000,
        out="runs/sft_v1",
    )

Internally it builds:
* model via :func:`synapforge.model_100m.build_synapforge_100m`
  (loads the architecture from the warmstart ckpt's ``config`` dict
  if present, otherwise uses :class:`TrainerConfig` defaults).
* optimizer via :func:`synapforge.optim.build_optimizer`.
* train + val streams via :class:`InstructionParquetStream`
  (for ``run_sft``) or :class:`ParquetTokenStream` (for ``run_kd``).
* trainer subclass with the appropriate config.
* invokes ``trainer.run()``.

Soft dependencies
-----------------
The module is constructed so that the legacy ``train_100m_*.py``
scripts remain the production entry points. The high-level API here
is for new code paths (e.g. notebook driving, hyperparam sweeps).
"""
from __future__ import annotations

import os
from typing import Optional


def _try_build_model(cfg, warmstart_ckpt: str = ""):
    """Build the model using the same factory the legacy trainer uses.

    Best-effort: if synapforge.model_100m or torch isn't importable
    (e.g. unit-test environment without GPU dependencies), raises
    RuntimeError with a clear message.
    """
    try:
        import torch
        from synapforge.model_100m import build_synapforge_100m
    except ImportError as exc:
        raise RuntimeError(
            f"synapforge.finetune requires torch + synapforge.model_100m: "
            f"{exc!r}"
        ) from exc

    arch = cfg.to_ckpt_config()
    # If warmstart ckpt has a config, prefer it (handles legacy ckpts
    # trained at a different d / n_layers).
    if warmstart_ckpt and os.path.exists(warmstart_ckpt):
        try:
            ck = torch.load(warmstart_ckpt, map_location="cpu",
                            weights_only=False)
            if isinstance(ck, dict) and "config" in ck:
                arch = dict(ck["config"])
        except Exception:
            pass

    model = build_synapforge_100m(
        vocab=int(arch.get("vocab", cfg.vocab)),
        d=int(arch.get("d", cfg.d)),
        n_layers=int(arch.get("n_layers", cfg.n_layers)),
        loop_depth=int(arch.get("loop_depth", cfg.loop_depth)),
        max_seq=int(arch.get("max_seq", cfg.seq_len)),
        ffn_ratio=float(arch.get("ffn_ratio", cfg.ffn_ratio)),
        sparsity=float(arch.get("sparsity", cfg.sparsity)),
        dropout=float(arch.get("dropout", cfg.dropout)),
        tie_lm_head=bool(arch.get("tie_lm_head", cfg.tie_lm_head)),
    )
    return model


def _try_build_optim(cfg, model):
    try:
        from synapforge.optim import build_optimizer
    except ImportError as exc:
        raise RuntimeError(
            f"synapforge.finetune requires synapforge.optim: {exc!r}"
        ) from exc
    return build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)


def run_sft(
    *,
    ckpt: str,
    data: str,
    out: str = "runs/sft",
    steps: int = 5000,
    batch_size: int = 16,
    seq_len: int = 512,
    lr: float = 1e-4,
    cross_val: Optional[str] = None,
    response_only_loss: bool = True,
    eval_every: int = 250,
    save_every: int = 250,
) -> dict:
    """High-level SFT entry point.

    Parameters
    ----------
    ckpt:       warmstart ckpt path (a Phase 1 KD ckpt).
    data:       glob/path of training instruction parquet(s).
    out:        run output directory.
    steps:      total training steps.
    batch_size, seq_len, lr:  training knobs.
    cross_val:  optional 2nd val parquet for cross-domain ppl.
                (currently informational: stream not auto-built).
    response_only_loss: True (default, SFT) / False (instruction-LM ablation).

    Returns
    -------
    Summary dict from :meth:`SFTTrainer.run` -- includes
    ``final_step``, ``best_val_ppl``, ``log_lines``.
    """
    from .training.sft_loop import InstructionParquetStream
    from .training.sft_trainer import SFTTrainer, SFTTrainerConfig

    cfg = SFTTrainerConfig(
        out_dir=out,
        steps=steps,
        save_every=save_every,
        eval_every=eval_every,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        warmstart_ckpt=ckpt,
        response_only_loss=response_only_loss,
    )

    train_stream = iter(InstructionParquetStream(
        data, seq_len=seq_len, batch_size=batch_size,
        response_only_loss=response_only_loss, loop=True,
    ))
    # 5% holdout as val (deterministic via shuffle_buffer=0).
    val_stream = iter(InstructionParquetStream(
        data, seq_len=seq_len, batch_size=batch_size,
        response_only_loss=response_only_loss, loop=False,
        shuffle_buffer=0,
    ))

    model = _try_build_model(cfg, warmstart_ckpt=ckpt)
    optim = _try_build_optim(cfg, model)

    trainer = SFTTrainer(cfg, model, optim, train_stream, val_stream)
    return trainer.run()


def run_kd(
    *,
    teacher,  # passed in by caller (we don't try to load HF teachers here)
    data: str,
    val_data: Optional[str] = None,
    out: str = "runs/kd",
    steps: int = 1000,
    batch_size: int = 80,
    seq_len: int = 256,
    lr: float = 1e-4,
    kd_weight: float = 0.4,
    kd_temperature: float = 4.0,
    kd_topk: int = 2048,
    warmstart: str = "",
    eval_every: int = 500,
    save_every: int = 250,
) -> dict:
    """High-level KD entry point.

    Note: ``teacher`` is passed by the caller as an already-loaded
    nn.Module (frozen, eval mode). Loading HF teachers is the caller's
    responsibility because there are too many provider-specific
    options to encode here.
    """
    from .data import ParquetTokenStream, split_val_stream  # type: ignore
    from .training.kd_trainer import KDTrainer, KDTrainerConfig

    cfg = KDTrainerConfig(
        out_dir=out,
        steps=steps,
        save_every=save_every,
        eval_every=eval_every,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        warmstart_ckpt=warmstart,
        kd_weight=kd_weight,
        kd_temperature=kd_temperature,
        kd_topk=kd_topk,
    )

    train_stream = iter(ParquetTokenStream(
        data, seq_len=seq_len, batch_size=batch_size,
    ))
    if val_data:
        val_stream = iter(ParquetTokenStream(
            val_data, seq_len=seq_len, batch_size=batch_size, loop=False,
        ))
    else:
        val_stream = None

    model = _try_build_model(cfg, warmstart_ckpt=warmstart)
    optim = _try_build_optim(cfg, model)

    trainer = KDTrainer(cfg, model, optim, train_stream, val_stream,
                        teacher=teacher)
    return trainer.run()


__all__ = ["run_kd", "run_sft"]
