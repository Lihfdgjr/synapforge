"""DEEP_MAINT_QUEUE.md T8.4 — exponential-moving-average weight tracker.

Top-LM training pipelines (DeepSeek, Llama, SmolLM2, ...) maintain a
shadow copy of model weights with the recurrence

    model_ema = decay * model_ema + (1 - decay) * model

after every optimizer step. ``decay`` typically lives in [0.99, 0.9999];
0.999 is the most common default. The EMA model is then used at
inference time (or for evaluation) instead of the live training weights
because it averages out the per-step noise of stochastic gradient
descent and tracks a smoother trajectory through the loss landscape.

Design notes for this implementation:

* **EMA state lives on CPU in fp32.** A 100M-param model in bf16 is
  ~200 MB on GPU; the live model already pays for that. Mirroring the
  EMA on GPU would push us into a second 200 MB allocation on top of
  the live model + optimizer state. CPU + fp32 is ~400 MB of RAM,
  which is essentially free on any training box, and using fp32 also
  avoids the slow bf16 drift that would compound across many small
  ``(1 - decay)`` updates.

* **Per-key tensor copy on update.** ``model.state_dict()`` is the
  source of truth for "what the optimizer just stepped"; we don't try
  to peek at ``p.data`` directly because that breaks for state_dict
  entries that are non-parameter buffers (RoPE bases, RMS norm scale
  buffers, etc.) -- those should also be in the EMA so the saved EMA
  ckpt is a drop-in replacement for the live ckpt.

* **Hot-path is per-step, so the loop is tight.** The CPU-fp32 update
  is a host-side mul + add per parameter; for 100M params that runs
  in single-digit milliseconds on a modern CPU. We do NOT GPU->CPU
  copy whole tensors per step: instead the live model's
  ``state_dict()`` items are transferred lazily inside the per-key
  ``mul_ + add_`` so the dispatch is one ``.to('cpu', dtype=fp32)``
  per param, which PyTorch internally fuses for tiny tensors.

* **Default OFF.** ``train_100m_kd.py``'s ``--ema-decay`` default is
  0.0; existing runs see zero behaviour change. Only when the user
  passes a positive decay does the tracker initialize and the per-step
  update fires.

* **Save format is identical to a live ckpt's "model" payload** so
  ``load_into(model)`` works without any custom unpacking, and so that
  ``chat_demo.py`` / ``chat_repl.py`` can load an EMA ckpt by exactly
  the same code path used for the live ``step_<N>.pt``.

* **swap() is a context manager.** For evaluation / generation inside
  the training loop without losing the live optimizer trajectory, the
  caller wraps ``with tracker.swap(model): ...`` — the live weights
  are transparently replaced with the EMA copy on enter, then restored
  bit-exact on exit (even if an exception is raised inside the block).

CPU-only-safe: imports torch lazily inside methods so unit tests that
``importorskip('torch')`` are still well-formed.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Optional

import torch


def _detach_to_cpu_fp32(t: "torch.Tensor") -> "torch.Tensor":
    """Return ``t`` detached, on CPU, cast to fp32 (or kept as int dtypes).

    Integer state_dict entries (e.g. ``num_batches_tracked`` in BatchNorm)
    are NOT EMA'd -- they're counters, not weights -- so we keep them as
    their original dtype on CPU. Floating-point entries are cast to fp32
    so the slow ``decay·old + (1-decay)·new`` accumulation doesn't drift
    in bf16/fp16.
    """
    cpu = t.detach().to("cpu", copy=True)
    if cpu.is_floating_point():
        cpu = cpu.to(dtype=torch.float32)
    return cpu


class EMATracker:
    """Exponential moving average of model weights (CPU shadow copy).

    Usage:

        tracker = EMATracker(model, decay=0.999)
        for step in range(N):
            ...
            optim.step()
            tracker.update(model)
        tracker.save("step_N_ema.pt")    # writes a state_dict dict.
        tracker.load_into(model)         # swap live weights to EMA copy.

    The tracker keeps an in-memory dict of ``{key: cpu_fp32_tensor}``
    parallel to ``model.state_dict()``. Non-floating-point entries are
    snapshotted but never updated (they're counters, not parameters).
    """

    def __init__(self, model: "torch.nn.Module", decay: float = 0.999) -> None:
        if decay < 0.0 or decay > 1.0:
            raise ValueError(
                f"EMATracker: decay must be in [0, 1]; got {decay!r}"
            )
        self.decay = float(decay)
        self.state: Dict[str, "torch.Tensor"] = {}
        # Snapshot every key in the live state_dict so the EMA mirror is
        # the same shape as the live model. CPU + fp32 for floats; original
        # dtype for integer counters.
        sd = model.state_dict()
        for k, v in sd.items():
            self.state[k] = _detach_to_cpu_fp32(v)

    @torch.no_grad()
    def update(self, model: "torch.nn.Module") -> None:
        """One EMA update step: ``state[k] = decay·state[k] + (1-decay)·model[k]``.

        Keys that don't appear in ``state`` (e.g. a buffer that materialised
        after init via lazy module construction) are inserted at full weight
        — i.e. as if they had been part of the EMA from the start. This is
        the behaviour of the standard timm / fairseq EMA: a never-seen
        parameter starts equal to its current value rather than zero.

        Integer-typed state entries (counters) are snapshotted to the
        latest live value, NOT EMA'd, because averaging counters is
        meaningless.
        """
        decay = self.decay
        one_minus = 1.0 - decay
        sd = model.state_dict()
        for k, v in sd.items():
            v_cpu = _detach_to_cpu_fp32(v)
            cur = self.state.get(k)
            if cur is None or cur.shape != v_cpu.shape:
                # First-time key (or shape change due to dynamic sparsity
                # rewires): take the live value as the EMA seed.
                self.state[k] = v_cpu
                continue
            if not v_cpu.is_floating_point():
                # Integer counters: just snapshot the latest value.
                self.state[k] = v_cpu
                continue
            # In-place to avoid re-allocating a 100M-param fp32 tensor every step.
            cur.mul_(decay).add_(v_cpu, alpha=one_minus)

    def state_dict(self) -> Dict[str, "torch.Tensor"]:
        """Return the underlying CPU-fp32 EMA state dict (live reference)."""
        return self.state

    def save(self, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """Save the EMA state to ``path``.

        Format matches a live ``step_<N>.pt`` ckpt's ``"model"`` payload so
        the loader (``chat_demo`` / ``chat_repl`` / ``adv_warmstart``) can
        consume the EMA ckpt with exactly the same code path it uses for
        the live ckpt. Optional ``extra`` keys (step, decay, config) are
        merged at the top level.
        """
        payload: Dict[str, Any] = {
            "model": dict(self.state),
            "ema_decay": self.decay,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @torch.no_grad()
    def load_into(self, model: "torch.nn.Module") -> None:
        """Copy the EMA weights into ``model`` (in-place, on the model's device).

        Cast back to each parameter's native dtype so a bf16 model stays
        bf16-resident after the swap; fp32 EMA tensor -> bf16 model param
        is a single copy_ on the model's device.
        """
        sd = model.state_dict()
        for k, live in sd.items():
            ema = self.state.get(k)
            if ema is None:
                # Key only on live model side -- leave the live value alone.
                continue
            if ema.shape != live.shape:
                continue
            # Move to the live param's device + dtype before copy_.
            ema_t = ema.to(device=live.device, dtype=live.dtype, copy=False)
            live.copy_(ema_t)

    @contextlib.contextmanager
    def swap(self, model: "torch.nn.Module") -> Iterator["torch.nn.Module"]:
        """Context manager: temporarily replace live model weights with EMA copy.

        On enter, the current live weights are snapshotted (CPU + fp32, same
        format as the EMA state itself) and the EMA tensors are copied into
        the model. On exit, the snapshot is restored bit-exact, even if the
        wrapped block raises.

        Usage:

            tracker = EMATracker(model, decay=0.999)
            ...
            with tracker.swap(model):
                # model now reflects the EMA-smoothed weights.
                val_loss = evaluate(model, val_iter)
            # model has been restored to its pre-swap live weights.

        This is the standard timm / DeepSpeed pattern for EMA-evaluation
        inside a training loop without losing optimizer/grad state.
        """
        # Snapshot the live weights into the same CPU-fp32 format we use for
        # the EMA mirror; this both preserves the exact pre-swap values and
        # keeps a single restore code path (state_dict copy_ in fp32).
        snapshot: Dict[str, "torch.Tensor"] = {}
        sd = model.state_dict()
        for k, v in sd.items():
            snapshot[k] = _detach_to_cpu_fp32(v)

        # Copy EMA -> live model (on its current device + dtype).
        self.load_into(model)
        try:
            yield model
        finally:
            # Restore the snapshot bit-exact onto the model -- even if the
            # wrapped block raised, the live trajectory is preserved.
            with torch.no_grad():
                cur = model.state_dict()
                for k, live in cur.items():
                    snap = snapshot.get(k)
                    if snap is None:
                        continue
                    if snap.shape != live.shape:
                        continue
                    snap_t = snap.to(device=live.device, dtype=live.dtype,
                                     copy=False)
                    live.copy_(snap_t)

    def load(self, path: str, map_location: str = "cpu") -> Dict[str, Any]:
        """Load EMA state from a previously saved ckpt (counterpart to ``save``).

        Reads the ckpt at ``path`` (expected to be the format produced by
        :meth:`save` -- a top-level dict with a ``"model"`` key carrying the
        EMA state_dict). The internal state dict is replaced (per-key
        cloned to CPU + fp32 for floats so the in-place ``mul_`` / ``add_``
        path stays well-typed). Returns the full ckpt payload so callers
        can read auxiliary fields like ``step`` / ``ema_decay``.
        """
        ck = torch.load(path, map_location=map_location)
        if not isinstance(ck, dict) or "model" not in ck:
            raise ValueError(
                f"EMATracker.load: ckpt at {path!r} must be a dict with "
                f"a 'model' key (EMATracker.save format); got "
                f"{type(ck).__name__}"
            )
        sd = ck["model"]
        new_state: Dict[str, "torch.Tensor"] = {}
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            new_state[k] = _detach_to_cpu_fp32(v)
        self.state = new_state
        decay = ck.get("ema_decay")
        if isinstance(decay, (int, float)) and 0.0 <= float(decay) <= 1.0:
            self.decay = float(decay)
        return ck


# ----- aliases / canonical names -----------------------------------------
# ``ModelEMA`` is the public-facing class name in the T8.4 spec / the
# README. Keep ``EMATracker`` as the historical name (the trainer + the
# pre-existing tests/integration/test_ema_weights.py both import that
# symbol; renaming would break those callers). Both names point at the
# same class, so user docs / new code can use ``ModelEMA`` while the
# existing call sites keep working.
ModelEMA = EMATracker


def load_ema(ckpt_path: str, model: "torch.nn.Module") -> Dict[str, Any]:
    """Top-level helper: load EMA state from disk into a fresh model.

    Reads the ckpt at ``ckpt_path`` (expected to be the format produced by
    :meth:`EMATracker.save` -- a top-level dict with a ``"model"`` key
    containing the EMA state_dict), then copies those weights into
    ``model`` via ``model.load_state_dict(..., strict=False)`` so that
    out-of-date keys do not crash the loader (consistent with how the
    rest of the trainer treats ckpt warmstart).

    Returns the full ckpt payload so callers can read auxiliary fields
    like ``ema_decay`` / ``step`` / ``config`` if they need them.

    Use this from inference scripts (``chat_demo.py`` etc.) to swap the
    live-trained weights to the EMA snapshot:

        model = build_synapforge_100m(...)
        load_ema("/runs/step_000500_ema.pt", model)
        model.eval()
        # ... use model for inference ...
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ck, dict) or "model" not in ck:
        raise ValueError(
            f"load_ema: ckpt at {ckpt_path!r} must be a dict with a 'model' "
            f"key (EMATracker.save format); got {type(ck).__name__}"
        )
    sd = ck["model"]
    model.load_state_dict(sd, strict=False)
    return ck
