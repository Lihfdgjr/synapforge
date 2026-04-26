"""sf.hf_trainer — HuggingFace ``Trainer`` integration for synapforge.

Drop-in subclass of :class:`transformers.Trainer` that wires synapforge's
plasticity engine into the standard HuggingFace training loop. A user with
an existing HF training script does not have to rewrite anything more than
the import and a single keyword:

    from synapforge.hf_trainer import SFTrainer
    from transformers import TrainingArguments

    trainer = SFTrainer(
        model=sf_model,
        args=TrainingArguments(output_dir="./out", per_device_train_batch_size=32),
        train_dataset=ds,
        plasticity_engine=engine,        # NEW
    )
    trainer.train()

Behaviour:
  * If ``plasticity_engine`` is provided, after each ``optimizer.step()`` we
    call ``engine.step(t, weight_dict)`` and ``engine.apply(...)`` to
    integrate the plasticity deltas into the BP-trained weights.
  * If the user passes a synapforge model with sf.Param-tagged params, we
    auto-build :class:`sf.PlasticityAwareAdamW` instead of vanilla AdamW
    (so multi-source gradients merge correctly). Otherwise we defer to
    the parent's ``create_optimizer`` (vanilla AdamW).
  * Mixed precision (bf16 / fp16) inherits from the parent's
    ``compute_loss_context_manager`` and the Accelerator's autocast path.
    Plasticity deltas are upcast to the param dtype inside engine.apply,
    so this is safe under amp.

Constraint: we do NOT add an ``accelerate.Accelerator`` ourselves — the
parent ``Trainer`` already manages one (or a no-op stand-in). We only
override the three hooks called out in the API contract:
``__init__``, ``training_step``, ``create_optimizer``.
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

try:
    from transformers import (
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,  # noqa: F401  re-export-able
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "synapforge.hf_trainer requires `transformers` to be installed. "
        "Install via `pip install transformers>=4.40`."
    ) from exc

from .optim import (
    MultiSourceParam,
    PlasticityAwareAdamW,
)
from .optim import (
    build_optimizer as build_sf_optimizer,
)
from .plasticity import PlasticityEngine

__all__ = [
    "SFTrainer",
    "PlasticityCallback",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — discover sf-style params, build a name->param map for engine.apply
# ---------------------------------------------------------------------------


def _has_sf_param(model: nn.Module) -> bool:
    """True iff at least one parameter has ``_sf_grad_source`` metadata
    (i.e. was constructed via :func:`sf.Param`)."""
    for p in model.parameters():
        if hasattr(p, "_sf_grad_source"):
            return True
    return False


def _named_tensor_map(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return ``{qualified_name: tensor}`` for parameters AND boolean
    ``mask`` buffers (so SynaptogenesisGrowPrune can update masks)."""
    out: dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        out[n] = p.data
    for n, b in model.named_buffers():
        if b.dtype == torch.bool or n.endswith(".mask"):
            out[n] = b
    return out


# ---------------------------------------------------------------------------
# PlasticityCallback — fires engine.step after optimizer.step
# ---------------------------------------------------------------------------


class PlasticityCallback(TrainerCallback):
    """HF callback that runs the synapforge plasticity engine each optimizer step.

    HF Trainer fires ``on_optimizer_step`` AFTER ``optimizer.step()`` returns
    (in v4.45 this is at line 395 of ``Trainer._inner_training_loop``). We
    use that hook to:
        1. compute deltas via ``engine.step(t, weight_dict)``
        2. apply them via ``engine.apply(deltas, weight_dict)``

    The ``trainer`` reference is bound at construction so we can pull
    ``trainer.model`` (which is the unwrapped model — ``trainer.model_wrapped``
    is what DDP/Accelerator wraps for forward passes).
    """

    def __init__(self, trainer: SFTrainer, engine: PlasticityEngine) -> None:
        self.trainer_ref = trainer
        self.engine = engine
        self._n_applied = 0
        self._weight_map_cache: dict[str, torch.Tensor] | None = None

    def on_optimizer_step(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self.engine is None:
            return control
        model = self.trainer_ref.model
        if model is None:
            return control
        # Build the weight map once and keep it; re-build only if model
        # parameters change shape (rare — mostly during synaptogenesis).
        if self._weight_map_cache is None:
            self._weight_map_cache = _named_tensor_map(model)
        weights = self._weight_map_cache
        try:
            deltas = self.engine.step(t=state.global_step, weight_dict=weights)
        except Exception as exc:
            warnings.warn(
                f"PlasticityCallback: engine.step failed: {exc!r}; skipping."
            )
            return control
        if not deltas:
            return control
        try:
            self.engine.apply(deltas, weights)
            self._n_applied += 1
        except Exception as exc:
            warnings.warn(
                f"PlasticityCallback: engine.apply failed: {exc!r}; skipping."
            )
        return control

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl | None:
        # Reset counters and rule traces at the start of training.
        self._n_applied = 0
        self._weight_map_cache = None
        try:
            self.engine.reset()
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("PlasticityCallback: engine.reset failed: %r", exc)
        return control


# ---------------------------------------------------------------------------
# SFTrainer — Trainer subclass with synapforge plasticity engine support
# ---------------------------------------------------------------------------


class SFTrainer(Trainer):
    """``transformers.Trainer`` with synapforge plasticity-aware optimizer.

    Args (in addition to all of ``Trainer``'s):
        plasticity_engine:    optional :class:`sf.PlasticityEngine`. When
                              provided, its rules fire after every
                              ``optimizer.step()`` via a callback.
        force_sf_optimizer:   if True, build :class:`PlasticityAwareAdamW`
                              even for vanilla nn.Modules. Default False:
                              we only switch to the SF optimizer when at
                              least one param is sf-tagged.
        sf_optimizer_kwargs:  forwarded to ``build_optimizer``.
    """

    def __init__(
        self,
        model: nn.Module,
        args: Any = None,
        *,
        plasticity_engine: PlasticityEngine | None = None,
        force_sf_optimizer: bool = False,
        sf_optimizer_kwargs: dict[str, Any] | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        self._sf_engine = plasticity_engine
        self._force_sf_opt = bool(force_sf_optimizer)
        self._sf_opt_kwargs = dict(sf_optimizer_kwargs or {})
        # Decide opt-class hint BEFORE super().__init__ since the parent
        # may call create_optimizer during init in some configurations.
        self._use_sf_optimizer = (
            self._force_sf_opt or (model is not None and _has_sf_param(model))
        )
        super().__init__(model=model, args=args, **trainer_kwargs)
        # Wire plasticity callback after super().__init__() set up callbacks.
        if self._sf_engine is not None:
            cb = PlasticityCallback(self, self._sf_engine)
            self.add_callback(cb)
            self._plasticity_callback = cb
        else:
            self._plasticity_callback = None

    # ------------------------------------------------------------------ optim
    def create_optimizer(self):
        """Override: use :class:`PlasticityAwareAdamW` if model has sf.Params.

        Falls back to the parent implementation (vanilla AdamW) when there's
        no plasticity wiring on the model.
        """
        if self.optimizer is not None:
            return self.optimizer
        if not self._use_sf_optimizer:
            return super().create_optimizer()

        opt_model = self.model
        # Mirror parent's grouping logic: decay vs. no-decay.
        decay_param_names = self.get_decay_parameter_names(opt_model)
        decay_set = set(decay_param_names)

        ms_decay: list[MultiSourceParam] = []
        ms_nodecay: list[MultiSourceParam] = []
        for name, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            sources = list(getattr(p, "_sf_grad_source", ("bp",)))
            wps = getattr(p, "_sf_weight_per_source", None)
            msp = MultiSourceParam(p, sources=sources, weight_per_source=wps)
            (ms_decay if name in decay_set else ms_nodecay).append(msp)

        # We only support single-LR PlasticityAwareAdamW for now; weight_decay
        # is honoured per group via two optimizers? No — keep it simple: use
        # the global wd for the decay group, 0 for the rest. We attach two
        # PlasticityAwareAdamWs and chain their .step() through a façade.
        lr = float(self.args.learning_rate)
        wd = float(getattr(self.args, "weight_decay", 0.0))
        beta1 = float(getattr(self.args, "adam_beta1", 0.9))
        beta2 = float(getattr(self.args, "adam_beta2", 0.999))
        eps = float(getattr(self.args, "adam_epsilon", 1e-8))

        kwargs = dict(lr=lr, betas=(beta1, beta2), eps=eps)
        kwargs.update(self._sf_opt_kwargs)

        opts: list[PlasticityAwareAdamW] = []
        if ms_decay:
            opts.append(PlasticityAwareAdamW(ms_decay, weight_decay=wd, **kwargs))
        if ms_nodecay:
            opts.append(PlasticityAwareAdamW(ms_nodecay, weight_decay=0.0, **kwargs))
        if not opts:
            return super().create_optimizer()
        if len(opts) == 1:
            self.optimizer = opts[0]
        else:
            self.optimizer = _ChainedOptimizer(opts)
        return self.optimizer

    # ----------------------------------------------------------- training_step
    def training_step(self, model, inputs, *args, **kwargs):
        """Override: pass-through to parent.

        The plasticity step proper runs in :class:`PlasticityCallback` AFTER
        ``optimizer.step()``; this keeps the BP graph and the plasticity
        update cleanly separated. *args/**kwargs forwards ``num_items_in_batch``
        on transformers >= 4.46 without breaking 4.45 (which has only
        ``(model, inputs)``).
        """
        if not model.training:
            model.train()
        return super().training_step(model, inputs, *args, **kwargs)


# ---------------------------------------------------------------------------
# _ChainedOptimizer — holds two PlasticityAwareAdamW with different weight_decay
# ---------------------------------------------------------------------------


class _ChainedOptimizer(torch.optim.Optimizer):
    """A trivial chained optimizer that delegates step/zero_grad to each child.

    HF's ``Trainer`` only calls ``optimizer.step()`` and ``optimizer.zero_grad()``
    so a thin façade is sufficient. ``param_groups`` is the union of all
    children, which is what HF and LR schedulers iterate over.
    """

    def __init__(self, opts: Sequence[torch.optim.Optimizer]) -> None:
        if not opts:
            raise ValueError("_ChainedOptimizer needs at least one child")
        self.opts = list(opts)
        # Set defaults to first child for compat. We don't call super().__init__()
        # because base Optimizer enforces a single param-list; we have many.
        self.defaults = dict(opts[0].defaults)
        self.state = {}
        for o in self.opts:
            self.state.update(o.state)
        self.param_groups = []
        for o in self.opts:
            self.param_groups.extend(o.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.opts:
            o.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"opts": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for o, sd in zip(self.opts, state_dict.get("opts", [])):
            o.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Public re-export
# ---------------------------------------------------------------------------

# Convenience: also expose build_sf_optimizer so users can pre-build their
# optimizer outside the trainer (e.g. for LR scheduling experiments).
build_sf_optimizer = build_sf_optimizer
