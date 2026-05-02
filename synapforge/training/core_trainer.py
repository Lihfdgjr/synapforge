"""synapforge.training.core_trainer -- shared BaseTrainer for KD/SFT/RL/SelfDistill.

Goal
----
The four trainer entry points (``train_100m_kd.py`` / ``_sft.py`` /
``_rl.py`` / ``_self_distill.py``) duplicate ~80% of their plumbing:
argparse, ckpt save/load, val loop, log format, optimizer build,
warmstart resume, EMA, gradient clipping, phase-aware monitoring.
Adding a feature requires editing 4 files. This module extracts the
shared core into :class:`BaseTrainer`; each mode is then a thin
subclass that overrides :meth:`compute_loss` (and optionally
:meth:`prepare_batch`).

Non-destructive
---------------
The legacy ``train_100m_*.py`` files are NOT touched -- they remain the
"ground truth" production trainers. The new core_trainer is opt-in via
the ``synapforge.training.__main__`` dispatcher (``python -m
synapforge.training --mode kd ...``). This is required so the running
rental Run 7 (``train_100m_kd.py``) is not disturbed.

Bit-exactness contract
----------------------
The :class:`KDTrainer` subclass MUST produce a loss curve that matches
``train_100m_kd.py`` to within 1e-5 on a 100-step fixture (the test
verifies this). It does so by reusing the exact same
``_kd_loss`` math, lifted into ``synapforge/training/kd_math.py``.

Public surface
--------------
* :class:`TrainerConfig` -- dataclass aggregating all shared knobs.
* :class:`BaseTrainer` -- main abstraction. Subclasses override
  :meth:`compute_loss`. Standard usage::

      cfg = TrainerConfig(
          out_dir="/tmp/run",
          steps=100,
          batch_size=4,
          seq_len=64,
      )
      trainer = KDTrainer(cfg, model, optim, train_stream, val_stream)
      trainer.run()

* Utility helpers (:func:`log_metrics`, :func:`gradient_clip`,
  :func:`warmstart_resume`, :func:`save_ckpt`, :func:`load_ckpt`) are
  module-level so they can be shared without subclassing.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Shared knobs for all trainer modes.

    Each subclass MAY add its own dataclass-extension for mode-specific
    options (e.g. ``KDTrainerConfig`` adds ``kd_weight``,
    ``kd_temperature``); but every mode reuses the fields below.

    Fields are grouped by concern:
        * Output / logging
        * Loop cadence (steps / save / eval / log)
        * Data
        * Optimization
        * Model architecture (for ckpt config-dict)
        * Checkpoint / warmstart
        * Device / dtype
    """

    # ---- output / logging ----
    out_dir: str = "runs/synapforge"
    log_every: int = 10

    # ---- loop cadence ----
    steps: int = 1000
    save_every: int = 250
    eval_every: int = 500
    val_n_batches: int = 16

    # ---- data ----
    batch_size: int = 8
    seq_len: int = 256

    # ---- optimization ----
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_steps: int = 0
    accum_steps: int = 1
    label_smoothing: float = 0.0

    # ---- z-loss (sparse-z) ----
    z_loss_weight: float = 0.0
    z_loss_topk: int = 0  # 0 disables; 2048 = sparse z-loss

    # ---- EMA (default OFF) ----
    ema_decay: float = 0.0

    # ---- model architecture (persisted into ckpt) ----
    vocab: int = 151936
    d: int = 512
    n_layers: int = 10
    loop_depth: int = 1
    ffn_ratio: float = 8.0
    sparsity: float = 0.95
    dropout: float = 0.0
    tie_lm_head: bool = True

    # ---- checkpoint / warmstart ----
    warmstart_ckpt: str = ""
    best_ckpt_track: bool = True

    # ---- device / dtype ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"  # one of {"float32", "bfloat16", "float16"}

    # ---- mode marker (set by subclasses) ----
    mode: str = "base"

    # ---- arbitrary extra kwargs (for subclasses) ----
    extra: dict = field(default_factory=dict)

    def torch_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]

    def to_ckpt_config(self) -> dict:
        """Architecture subset persisted into every ckpt's ``config`` dict.

        Mirrors the shape that ``train_100m_kd.py._build_config_dict``
        writes so an SFT/KD ckpt is bidirectionally compatible.
        """
        return {
            "vocab": int(self.vocab),
            "d": int(self.d),
            "n_layers": int(self.n_layers),
            "loop_depth": int(self.loop_depth),
            "max_seq": int(self.seq_len),
            "ffn_ratio": float(self.ffn_ratio),
            "sparsity": float(self.sparsity),
            "dropout": float(self.dropout),
            "tie_lm_head": bool(self.tie_lm_head),
        }


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def safe_mkdir(p: str) -> None:
    """``os.makedirs`` with ``exist_ok`` and quiet-on-race."""
    try:
        os.makedirs(p, exist_ok=True)
    except OSError:
        pass


def log_metrics(
    step: int,
    metrics: dict,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """Render a per-step log line from a metrics dict and emit it.

    Format::

        step=00100 loss=2.1345 ce=2.0987 kd=0.6543 lr=1.000e-04 ...

    Returns the full line (without trailing newline) so callers can
    also append it to an in-memory log buffer.
    """
    parts = [f"step={int(step):05d}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            if abs(v) < 1e-3 or abs(v) >= 1e5:
                parts.append(f"{k}={v:.3e}")
            else:
                parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    line = " ".join(parts)
    if log_fn is None:
        print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
    else:
        log_fn(line)
    return line


def gradient_clip(model, max_norm: float) -> float:
    """``torch.nn.utils.clip_grad_norm_`` wrapper that returns the
    total grad norm BEFORE clipping (Python float, not tensor)."""
    if max_norm <= 0:
        return 0.0
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total)


def lr_at(step: int, peak_lr: float, warmup: int, total: int,
          schedule: str = "constant_with_warmup") -> float:
    """Compute LR at a given step.

    Schedules:
      * ``"constant_with_warmup"``: linear warmup over ``warmup`` steps,
        then ``peak_lr`` constant. (Default — matches Anthropic Tulu 3
        and Run 3n recipe per ``feedback_cosine_lr_warmstart_replateau``.)
      * ``"cosine"``: linear warmup, then cosine decay to 0.1 * peak.
      * ``"constant"``: ``peak_lr`` regardless of step.
    """
    if schedule == "constant":
        return float(peak_lr)
    if warmup > 0 and step < warmup:
        return float(peak_lr) * (step + 1) / float(warmup)
    if schedule == "constant_with_warmup":
        return float(peak_lr)
    if schedule == "cosine":
        progress = (step - warmup) / max(1, total - warmup)
        progress = max(0.0, min(1.0, progress))
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        # decay from peak to 0.1 * peak
        return float(peak_lr) * (0.1 + 0.9 * cos_factor)
    raise ValueError(f"unknown schedule {schedule!r}")


def ema_update(ema_state: dict, model, decay: float) -> None:
    """Update an EMA state dict in-place: ``ema = decay*ema + (1-decay)*model``.

    No-op when ``decay <= 0``. Reuses the canonical recipe from
    ``synapforge.training.ema.EMATracker.update`` but stays here as a
    pure-function variant so the BaseTrainer doesn't have to construct
    an EMATracker object on every call.
    """
    if decay <= 0:
        return
    sd = model.state_dict()
    for k, v in sd.items():
        if k not in ema_state:
            ema_state[k] = v.detach().to("cpu", dtype=torch.float32).clone()
            continue
        ema_state[k].mul_(decay).add_(
            v.detach().to("cpu", dtype=torch.float32),
            alpha=(1.0 - decay),
        )


def save_ckpt(
    path: str,
    *,
    model,
    optim,
    step: int,
    config: dict,
    extra: Optional[dict] = None,
) -> None:
    """Save a ckpt with the standard payload shape.

    Schema matches ``train_100m_kd.py`` exactly so loaders
    (chat_demo.py / chat_repl.py) reconstruct without guessing::

        {
          "model": <state_dict>,
          "optim_state": <state_dict>,
          "step": int,
          "config": {vocab, d, n_layers, ...},
          ...extra...
        }
    """
    payload = {
        "model": model.state_dict(),
        "optim_state": optim.state_dict() if optim is not None else None,
        "step": int(step),
        "config": dict(config),
    }
    if extra:
        payload.update(extra)
    safe_mkdir(os.path.dirname(path) or ".")
    torch.save(payload, path)


def load_ckpt(path: str, map_location: Optional[str] = None) -> dict:
    """Load a ckpt to a Python dict. Thin wrapper to centralise
    ``map_location`` and the ``weights_only=False`` decision."""
    return torch.load(path, map_location=map_location, weights_only=False)


def warmstart_resume(
    model,
    optim,
    ckpt_path: str,
    *,
    map_location: Optional[str] = None,
    strict: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> int:
    """Load model + (optional) optimizer state from ``ckpt_path``.

    Returns the step number stored in the ckpt (0 if missing).

    Per ``feedback_no_random_init_use_warmstart``, every architecture
    upgrade should warmstart from the previous best ckpt. Strict=False
    by default so new layers / shape mismatches don't crash; mismatches
    are logged.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        if log_fn:
            log_fn(f"[warmstart] ckpt {ckpt_path!r} not found; skipping")
        return 0
    ck = load_ckpt(ckpt_path, map_location=map_location)
    if "model" in ck:
        result = model.load_state_dict(ck["model"], strict=strict)
        if log_fn and (
            getattr(result, "missing_keys", None)
            or getattr(result, "unexpected_keys", None)
        ):
            log_fn(
                f"[warmstart] missing_keys={len(result.missing_keys)} "
                f"unexpected_keys={len(result.unexpected_keys)}"
            )
    if optim is not None and ck.get("optim_state") is not None:
        try:
            optim.load_state_dict(ck["optim_state"])
        except Exception as exc:
            if log_fn:
                log_fn(f"[warmstart] optim load failed: {exc!r}")
    step = int(ck.get("step", 0))
    if log_fn:
        log_fn(f"[warmstart] loaded {ckpt_path!r} at step {step}")
    return step


def update_best_ckpt(
    *,
    out_dir: str,
    step: int,
    val_ppl: float,
    best_val_ppl: float,
    enabled: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[float, Optional[str]]:
    """Maintain a ``best_step_<N>.pt`` link tracking the best val_ppl.

    Lifted from ``train_100m_kd._update_best_ckpt`` so all subclasses
    benefit from the best-ckpt-track invariant
    (``feedback_best_ckpt_track_mandatory``: must be ON to recover GPU
    time after divergence).

    Returns (new_best_val_ppl, link_path_or_None).
    """
    if not enabled:
        return best_val_ppl, None
    val_f = float(val_ppl)
    if not (val_f < float(best_val_ppl)):
        return best_val_ppl, None

    src_name = f"step_{step:06d}.pt"
    src_path = os.path.join(out_dir, src_name)
    if not os.path.exists(src_path):
        if log_fn:
            log_fn(
                f"[best-ckpt] val={val_f:.2f} would improve from "
                f"{float(best_val_ppl):.2f} but {src_name!r} not on disk"
            )
        return best_val_ppl, None

    dst_name = f"best_step_{step:06d}.pt"
    dst_path = os.path.join(out_dir, dst_name)

    # Remove stale best_step_*.pt files.
    try:
        for fname in os.listdir(out_dir):
            if fname.startswith("best_step_") and fname.endswith(".pt"):
                stale = os.path.join(out_dir, fname)
                try:
                    os.remove(stale)
                except OSError:
                    pass
    except OSError:
        return best_val_ppl, None

    is_windows = os.name == "nt"
    try:
        if is_windows:
            import shutil
            shutil.copy2(src_path, dst_path)
        else:
            rel_src = os.path.relpath(src_path, os.path.dirname(dst_path))
            os.symlink(rel_src, dst_path)
    except OSError as exc:
        if log_fn:
            log_fn(f"[best-ckpt] link/copy failed: {exc!r}")
        return best_val_ppl, None

    if log_fn:
        log_fn(f"[best-ckpt] new best val_ppl={val_f:.2f} -> {dst_name}")
    return val_f, dst_path


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------


class BaseTrainer:
    """Minimal mode-agnostic training loop.

    Subclasses MUST override :meth:`compute_loss`. They MAY override:

    * :meth:`prepare_batch` -- transform a raw stream batch into the
      ``(model_inputs, targets, mask)`` triple the loss expects.
      Default: pass-through.
    * :meth:`val_step` -- per-batch validation loss. Default: same as
      :meth:`compute_loss` minus the KD term.
    * :meth:`evaluate` -- aggregate val loss into ppl. Default: token
      mean -> exp.

    Attributes set in __init__::

        self.cfg         -- TrainerConfig
        self.model       -- nn.Module on cfg.device
        self.optim       -- optimizer
        self.train_stream-- iterable yielding (x, y) or (x, y, mask)
        self.val_stream  -- same
        self.device      -- str
        self.dtype       -- torch.dtype
        self.step        -- int (current training step, post-warmstart)
        self.best_val_ppl-- float (track for best-ckpt link)
        self.ema_state   -- dict | None
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        model,
        optim,
        train_stream: Iterator,
        val_stream: Optional[Iterator] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optim = optim
        self.train_stream = train_stream
        self.val_stream = val_stream
        self.device = cfg.device
        self.dtype = cfg.torch_dtype()
        self.step = 0
        self.best_val_ppl = float("inf")
        self.ema_state: Optional[dict] = {} if cfg.ema_decay > 0 else None
        self.log_lines: list[str] = []
        self._user_log_fn = log_fn

        safe_mkdir(cfg.out_dir)

        if cfg.warmstart_ckpt:
            self.step = warmstart_resume(
                model, optim, cfg.warmstart_ckpt,
                map_location=self.device,
                log_fn=self._log,
            )

    # ----- override hooks -----

    def prepare_batch(self, raw):
        """Normalize a raw stream batch into ``(x, y[, mask])``.

        Default: pass through. KD uses (x, y); SFT uses
        (tokens_in, tokens_out, loss_mask). Override per mode.
        """
        return raw

    def compute_loss(self, batch) -> dict:
        """Subclass MUST override.

        Returns a dict ``{"loss": tensor, "ce": tensor, ...}`` where
        ``loss`` is the scalar to backward and the rest are component
        tensors / floats for logging. The dict's keys become metrics
        in :func:`log_metrics`.
        """
        raise NotImplementedError("subclass must override compute_loss")

    def val_step(self, batch) -> dict:
        """Per-batch validation loss. Default: pure CE on (x, y)."""
        x, y = batch[0], batch[1]
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.amp.autocast(
            device_type=self.device,
            dtype=self.dtype,
            enabled=self.device == "cuda",
        ):
            logits = self.model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
            )
        return {"loss": loss}

    def evaluate(self, n_batches: Optional[int] = None) -> float:
        """Run validation and return ppl = exp(token-mean CE).

        Mirrors ``train_100m_kd.evaluate``. Returns NaN when val_stream
        is exhausted before yielding any batches.
        """
        if self.val_stream is None:
            return float("nan")
        n_batches = int(n_batches or self.cfg.val_n_batches)
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for _ in range(n_batches):
                try:
                    raw = next(self.val_stream)
                except StopIteration:
                    break
                batch = self.prepare_batch(raw)
                out = self.val_step(batch)
                losses.append(float(out["loss"].item()))
        self.model.train()
        if not losses:
            return float("nan")
        return math.exp(sum(losses) / len(losses))

    # ----- internal -----

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        if self._user_log_fn is not None:
            self._user_log_fn(line)
        else:
            print(line, flush=True)

    def save(self, step: Optional[int] = None) -> str:
        step = int(step if step is not None else self.step)
        path = os.path.join(self.cfg.out_dir, f"step_{step:06d}.pt")
        extra = {}
        if self.ema_state:
            extra["ema_state"] = self.ema_state
        save_ckpt(
            path,
            model=self.model,
            optim=self.optim,
            step=step,
            config=self.cfg.to_ckpt_config(),
            extra=extra,
        )
        return path

    def _set_lr(self, lr: float) -> None:
        for pg in self.optim.param_groups:
            pg["lr"] = lr

    # ----- main loop -----

    def run(self, steps: Optional[int] = None) -> dict:
        """Run the main training loop.

        Each step:
          1. fetch next batch (StopIteration -> early-stop)
          2. self.prepare_batch
          3. self.compute_loss
          4. backward + grad-clip + optim.step
          5. ema_update if enabled
          6. log every cfg.log_every
          7. save every cfg.save_every
          8. evaluate every cfg.eval_every; track best

        Returns a metrics summary dict::

            {"final_step": int, "best_val_ppl": float, "log_lines": [...]}
        """
        n_steps = int(steps if steps is not None else self.cfg.steps)
        self.model.train()
        start_step = self.step
        end_step = start_step + n_steps

        self._log(
            f"[trainer.run] mode={self.cfg.mode} steps={n_steps} "
            f"bs={self.cfg.batch_size} seq={self.cfg.seq_len} "
            f"lr={self.cfg.lr} device={self.device} dtype={self.cfg.dtype}"
        )

        for s in range(start_step + 1, end_step + 1):
            self.step = s
            try:
                raw = next(self.train_stream)
            except StopIteration:
                self._log(f"[trainer.run] train stream exhausted at step {s}")
                break

            batch = self.prepare_batch(raw)

            # LR schedule
            cur_lr = lr_at(
                s, self.cfg.lr, self.cfg.warmup_steps, end_step,
                schedule=self.cfg.extra.get("lr_schedule",
                                            "constant_with_warmup"),
            )
            self._set_lr(cur_lr)

            self.optim.zero_grad(set_to_none=True)
            losses = self.compute_loss(batch)
            loss = losses["loss"]
            (loss / float(max(1, self.cfg.accum_steps))).backward()

            grad_norm = gradient_clip(self.model, self.cfg.grad_clip)
            self.optim.step()

            if self.ema_state is not None:
                ema_update(self.ema_state, self.model, self.cfg.ema_decay)

            # ---- logging ----
            if s % self.cfg.log_every == 0 or s == start_step + 1:
                metrics = {
                    "loss": float(loss.detach().item()),
                    "lr": cur_lr,
                    "grad_norm": grad_norm,
                }
                # Add component losses (ce / kd / etc.) when present.
                for k, v in losses.items():
                    if k == "loss":
                        continue
                    if hasattr(v, "item"):
                        try:
                            metrics[k] = float(v.detach().item())
                        except Exception:
                            pass
                    elif isinstance(v, (int, float)):
                        metrics[k] = float(v)
                log_metrics(s, metrics, log_fn=self._log)

            # ---- save ----
            if self.cfg.save_every > 0 and s % self.cfg.save_every == 0:
                self.save(s)

            # ---- evaluate ----
            if (
                self.cfg.eval_every > 0
                and s % self.cfg.eval_every == 0
                and self.val_stream is not None
            ):
                ppl = self.evaluate()
                self._log(f"[eval] step={s} val_ppl={ppl:.4f}")
                # Make sure a ckpt exists at this step before linking.
                ckpt_path = os.path.join(
                    self.cfg.out_dir, f"step_{s:06d}.pt"
                )
                if not os.path.exists(ckpt_path):
                    self.save(s)
                self.best_val_ppl, _link = update_best_ckpt(
                    out_dir=self.cfg.out_dir,
                    step=s,
                    val_ppl=ppl,
                    best_val_ppl=self.best_val_ppl,
                    enabled=self.cfg.best_ckpt_track,
                    log_fn=self._log,
                )

        # Final ckpt + log dump.
        if self.step > start_step:
            final_path = self.save(self.step)
            self._log(f"[trainer.run] saved final ckpt: {final_path}")
        return {
            "final_step": self.step,
            "best_val_ppl": self.best_val_ppl,
            "log_lines": list(self.log_lines),
        }


__all__ = [
    "BaseTrainer",
    "TrainerConfig",
    "ema_update",
    "gradient_clip",
    "load_ckpt",
    "log_metrics",
    "lr_at",
    "safe_mkdir",
    "save_ckpt",
    "update_best_ckpt",
    "warmstart_resume",
]
