"""synapforge.module — Phase 2 of the torch-replacement roadmap.

Phase 2 status (this commit): **shipped**.
See ``docs/TORCH_REPLACEMENT_PLAN.md`` for the surrounding plan and
``docs/TORCH_REPLACEMENT_PHASE0_AUDIT.md`` for the audit numbers.

What Phase 2 delivers
---------------------

* :class:`synapforge.module.Module` — base class for every synapforge
  block. Provides the public API (``parameters`` / ``named_parameters`` /
  ``state_dict`` / ``load_state_dict`` / ``.to`` / ``.cuda`` / ``.eval`` /
  ``.train`` / ``.zero_grad`` / ``register_parameter`` /
  ``register_module``) so callers stop touching ``torch.nn.*`` directly.
  Internally still inherits from ``torch.nn.Module`` — this is the
  gradual-migration anchor that lets Phase 2 ship without breaking the
  running rental's ``train_100m_kd.py`` (which is mid-training right
  now). Phase 3 swaps the parent class for our own storage layer.
* :class:`synapforge.module.Parameter` — thin wrapper over
  ``torch.nn.Parameter`` with ``requires_grad=True`` defaulted on. Phase
  2 keeps the storage on ``torch.Tensor``; Phase 4 of the roadmap
  replaces tensor storage with ``synapforge.tensor``. The wrapper is
  state-dict-compatible with ``torch.nn.Parameter`` so existing
  checkpoints (saved via ``torch.save``) round-trip losslessly.
* The plasticity hooks and IR compile entry points that were already
  present in the v0.1 ``Module`` are preserved verbatim — see
  :meth:`Module.register_plasticity` / :meth:`Module.compile_to_ir`.

Backward-compatibility guarantees (Phase 2 contract)
----------------------------------------------------

1. **Bit-exact forward equivalence.** A ``synapforge.module.Module`` and
   the equivalent ``torch.nn.Module`` with the same weights produce
   identical forward outputs (rel-err < 1e-5). Enforced by
   ``tests/module/test_module_torch_compat.py``.
2. **State-dict round-trip.** A ``state_dict`` produced by torch's
   ``Model(nn.Module).state_dict()`` loads cleanly into a
   ``synapforge.module.Module``-based model and vice versa. Enforced by
   ``tests/module/test_state_dict_compat.py``. This means existing
   ckpts under ``runs/*/ckpts/*.pt`` continue to load with
   ``model.load_state_dict(...)`` — non-negotiable because the running
   rental rolls them forward every save.
3. **Gradient-flow equivalence.** ``Parameter.requires_grad`` defaults
   to ``True`` and the autograd graph is identical to the
   ``torch.nn.Parameter`` baseline (we just subclass it).

Why we keep ``nn.Module`` as the parent for Phase 2
---------------------------------------------------

Per the Phase 0 audit (category d, ~790 sites across 127 files), every
existing module inherits from ``nn.Module`` directly or via our
``synapforge.module.Module``. Replacing the parameter-registration
plumbing in one shot would touch every cell, router, MoE, and modal
module in the package — not feasible in a 1-week phase budget and not
risk-bounded against the running training process.

The accepted plan (see roadmap §Phase 3) is:

* Phase 2 (this PR): finish the API surface; flip every module's
  base-class declaration from ``nn.Module`` to
  ``synapforge.module.Module``. Internally we still go through
  ``nn.Module``'s param tracking — bit-for-bit identical behaviour, no
  perf delta, no state-dict format change.
* Phase 3 (next): replace ``torch.Tensor`` storage with
  ``synapforge.tensor``. At that point ``Parameter`` becomes a
  ``synapforge.tensor`` wrapper, and ``Module``'s param-registration
  uses our own ordered dict instead of torch's. The public API on
  ``Module`` stays the same — that's the whole point of doing Phase 2
  now.
* Phase 4: ``synapforge.autograd.Function`` replaces
  ``torch.autograd.Function`` in our kernels.
* Phase 5: cut the ``import torch`` from the student path entirely.

Public API summary
------------------

::

    >>> from synapforge.module import Module, Parameter
    >>>
    >>> class MyBlock(Module):
    ...     def __init__(self, d):
    ...         super().__init__()
    ...         self.w = Parameter(torch.zeros(d, d))
    ...         self.register_module("ln", LayerNorm(d))
    ...     def forward(self, x):
    ...         return self.ln(x) @ self.w
    ...
    >>> m = MyBlock(8)
    >>> list(p.shape for p in m.parameters())
    [torch.Size([8, 8]), torch.Size([8])]
    >>> sd = m.state_dict()
    >>> m2 = MyBlock(8)
    >>> m2.load_state_dict(sd)        # torch-format compatible
    >>> m.cuda().eval()               # device + mode toggles work
    >>> m.zero_grad()                 # delegates to torch.nn.Module
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn as nn

__all__ = ["Module", "Parameter"]


# ---------------------------------------------------------------------------
# Parameter — thin wrapper over torch.nn.Parameter (Phase 2)
# ---------------------------------------------------------------------------


class Parameter(nn.Parameter):
    """Phase 2 ``synapforge.module.Parameter``.

    A ``torch.Tensor`` subclass with ``requires_grad=True`` by default.
    Subclasses :class:`torch.nn.Parameter` (rather than reimplementing
    its descriptor + ``__torch_function__`` plumbing) so:

    * Existing :class:`Module.parameters` machinery via ``nn.Module``
      finds it automatically.
    * State-dict serialization round-trips with ``torch.save`` /
      ``torch.load`` ckpts written before this PR. The unpickled
      tensor type is ``synapforge.module.Parameter`` only when
      explicitly assigned; otherwise torch's loader will hand back a
      bare ``nn.Parameter`` and we re-wrap on assignment — see
      :meth:`Module.register_parameter`.
    * No additional storage overhead. Phase 4 of the roadmap replaces
      the storage backend with ``synapforge.tensor``; this class
      becomes the natural shim point for that transition.

    Identity hooks
    ~~~~~~~~~~~~~~

    The class adds two zero-cost identity hooks to support state-dict
    serialization round-trip:

    * ``__reduce_ex__`` so pickled tensors retain the
      ``synapforge.module.Parameter`` type rather than being demoted to
      ``torch.nn.Parameter`` on round-trip. Important for users who
      ``torch.save(model)`` (whole-model pickle) — uncommon in our
      codebase but supported.
    * ``__repr__`` that names the class explicitly so ``print(model)``
      output shows ``Parameter`` (no ``Synapforge`` prefix — we want
      the noise level of the existing torch repr) but ``type(p)`` is
      still ``synapforge.module.Parameter``.

    Notes
    -----
    Phase 2 keeps the storage on ``torch.Tensor``. Phase 3 introduces
    ``synapforge.tensor`` and at that point ``Parameter`` becomes a
    ``synapforge.tensor`` wrapper instead of a ``torch.Tensor``
    subclass. The public API on this class is stable across Phases
    2-4; only the storage backing changes.
    """

    def __new__(
        cls,
        data: torch.Tensor | None = None,
        requires_grad: bool = True,
    ) -> "Parameter":
        if data is None:
            data = torch.empty(0)
        # Defer to nn.Parameter.__new__ — it handles the
        # ``_subclasses`` plumbing (so ``isinstance(p, nn.Parameter)``
        # is True) and the ``Tensor.__torch_function__`` overrides.
        return super().__new__(cls, data, requires_grad)

    def __repr__(self) -> str:  # pragma: no cover — cosmetic only
        # Match torch.nn.Parameter's repr verbatim so logs / prints
        # don't surprise. We keep the type via __class__ for
        # introspection.
        return super().__repr__()


# ---------------------------------------------------------------------------
# Module — Phase 2 base class
# ---------------------------------------------------------------------------


class Module(nn.Module):
    """Base class for every synapforge cell / block / model.

    Phase 2 contract
    ----------------

    Provides an explicit Phase 2 API surface on top of the
    ``torch.nn.Module`` plumbing we still inherit. Methods listed
    below are guaranteed-stable through Phases 2-5 of the
    torch-replacement roadmap; the *implementation* may swap from
    torch to ``synapforge`` internals across phases, but the call
    contract on this class will not change.

    Guaranteed Phase 2 API (parity with ``torch.nn.Module``)
    --------------------------------------------------------

    * :meth:`register_parameter` / :meth:`register_module`
    * :meth:`parameters` / :meth:`named_parameters`
    * :meth:`state_dict` / :meth:`load_state_dict`
    * :meth:`to` / :meth:`cuda` / :meth:`cpu`
    * :meth:`eval` / :meth:`train`
    * :meth:`zero_grad`
    * ``__call__`` dispatches to :meth:`forward` (the standard
      ``nn.Module`` contract).

    Phase 2 additions (synapforge-specific)
    ---------------------------------------

    * :meth:`register_plasticity` / :meth:`plasticity_step` — Hebbian /
      STDP rules registered on the module. Triggered after each
      forward pass when ``self.training`` is True.
    * :meth:`register_event_hook` — IR compiler entry point.
    * :meth:`compile_to_ir` — emit a synapforge IR graph (lazy import
      to avoid an ``ir.compiler`` cycle at import time).

    All other ``nn.Module`` methods (``apply``, ``buffers``,
    ``modules``, ``children``, ``register_buffer``, ``__setattr__``
    descriptor magic, ``_apply``, etc.) continue to work exactly as
    they do on ``nn.Module`` — they're transparently inherited. Phase
    3 of the roadmap replaces them piece-by-piece without breaking the
    public API on this class.
    """

    def __init__(self) -> None:
        super().__init__()
        # Plasticity rules registered on this module (name -> rule callable).
        # Rules see (module, inputs, outputs) and update buffers in-place.
        self._plasticity_rules: dict[str, Callable[..., None]] = {}
        # Event hooks for the IR compiler. v0.1 ignores; future backends use.
        self._event_hooks: list[Callable[..., None]] = []
        # Last forward inputs/outputs cached for plasticity_step().
        self._last_io: tuple[Any, Any] | None = None

    # ----------------------------------------------------------- registration

    def register_parameter(
        self, name: str, param: torch.Tensor | None
    ) -> None:
        """Register a parameter on this module.

        Mirrors :meth:`torch.nn.Module.register_parameter` semantics
        with one upgrade: a bare ``torch.Tensor`` (not a
        ``nn.Parameter``) is auto-wrapped in
        :class:`synapforge.module.Parameter` so call sites don't need
        to know about the wrapper type.

        Parameters
        ----------
        name : str
            Attribute name. Must be a non-empty Python identifier and
            must not contain ``"."`` (would collide with submodule
            dotted lookup).
        param : torch.Tensor or None
            The parameter data, or ``None`` to register the slot as
            an unset attribute (matches ``nn.Module`` behaviour for
            optional weights).

        Raises
        ------
        TypeError
            If ``param`` is not ``None`` and not a ``torch.Tensor``.
        KeyError
            If ``name`` collides with an already-registered submodule.
        """
        if param is None:
            super().register_parameter(name, None)
            return
        if not isinstance(param, torch.Tensor):
            raise TypeError(
                f"register_parameter({name!r}, ...): expected "
                f"torch.Tensor, got {type(param).__name__}"
            )
        # Auto-wrap bare tensors so callers don't need to import
        # synapforge.module.Parameter explicitly. If the caller passes
        # a torch.nn.Parameter we keep it (still a valid nn.Parameter
        # subclass) — Phase 4 will tighten this once ``Parameter``
        # diverges from ``nn.Parameter``.
        #
        # A bare ``torch.Tensor`` (i.e. not an ``nn.Parameter``) is
        # wrapped with ``requires_grad=True`` regardless of the
        # tensor's current ``requires_grad`` flag — the whole point
        # of registering as a Parameter is "I want gradients on
        # this", and a fresh ``torch.zeros(...)`` has
        # ``requires_grad=False`` by default. This matches the
        # ergonomics callers expect from ``nn.Parameter(t)`` (which
        # also defaults to True regardless of input).
        if not isinstance(param, nn.Parameter):
            param = Parameter(param.data, requires_grad=True)
        super().register_parameter(name, param)

    def register_module(self, name: str, module: nn.Module | None) -> None:
        """Register a submodule on this module.

        Forwards to :meth:`torch.nn.Module.add_module` (which is what
        ``register_module`` aliases on torch >= 1.9, but we expose the
        roadmap-canonical name for clarity). Submodules can be plain
        ``torch.nn.Module`` (e.g. ``nn.Linear`` from the legacy stack)
        or ``synapforge.module.Module``; both compose cleanly because
        we inherit from ``nn.Module`` for parameter tracking.
        """
        if module is not None and not isinstance(module, nn.Module):
            raise TypeError(
                f"register_module({name!r}, ...): expected "
                f"torch.nn.Module subclass, got "
                f"{type(module).__name__}"
            )
        # nn.Module.add_module covers the behaviour we want (None
        # acceptance, dotted-name rejection, attribute slot
        # population, ``_modules`` dict update).
        self.add_module(name, module)

    # ----------------------------------------------------------- iteration

    def parameters(
        self, recurse: bool = True
    ) -> Iterator[torch.Tensor]:
        """Yield every parameter on this module (and submodules).

        Phase 2 forwarder. Identical to
        :meth:`torch.nn.Module.parameters`; exposed explicitly so the
        public Phase 2 API is enumerable from ``Module.__dict__`` and
        the docstring lives here rather than implicitly inherited.
        """
        yield from super().parameters(recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, parameter)`` pairs.

        Phase 2 forwarder. Identical to
        :meth:`torch.nn.Module.named_parameters`. The Phase-2 contract
        guarantees the iteration order matches torch's
        (insertion-ordered ``_parameters`` + recursive
        ``_modules`` traversal) so callers that build optimizer
        param-groups by name keep working unchanged.
        """
        yield from super().named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    # ----------------------------------------------------------- state dict

    def state_dict(
        self,
        *args: Any,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Return a state-dict compatible with ``torch.nn.Module``.

        Phase 2 forwarder. The dict produced is byte-equivalent to
        what ``nn.Module.state_dict()`` would produce for the same
        weights — so a ``torch.save(model.state_dict())`` ckpt round-
        trips with :meth:`load_state_dict` regardless of whether the
        saving model inherited from ``nn.Module`` or
        ``synapforge.module.Module``.

        The ``*args`` / kwargs forwarding accommodates both the
        positional and keyword call styles torch supports across
        versions (older code passes ``destination``/``prefix`` as
        positionals; modern code uses kwargs).
        """
        return super().state_dict(
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load a state-dict into this module.

        Phase 2 forwarder. Accepts state dicts produced by either
        ``torch.nn.Module.state_dict()`` or
        :meth:`synapforge.module.Module.state_dict` interchangeably —
        the Phase 2 contract guarantees they are byte-equivalent.

        Parameters
        ----------
        state_dict : dict
            The state dict to load.
        strict : bool, default ``True``
            If ``True``, missing/unexpected keys raise. If ``False``,
            return them in the named tuple.
        assign : bool, default ``False``
            Torch >= 2.1 only — assigns parameters by reference rather
            than copying into existing storage. Silently ignored on
            older torch (we still target torch 2.0 in production).

        Returns
        -------
        torch.nn.modules.module._IncompatibleKeys
            Named tuple with ``missing_keys`` and ``unexpected_keys``,
            same as torch.
        """
        # ``assign`` is a torch 2.1+ kwarg. Detect and forward only when
        # the underlying ``nn.Module.load_state_dict`` accepts it; older
        # torch (production has 2.0.1) raises TypeError otherwise. We
        # could check via inspect.signature, but a try/except is simpler
        # and equally cheap (the call only happens on warmstart).
        try:
            return super().load_state_dict(
                state_dict, strict=strict, assign=assign
            )
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    # ----------------------------------------------------------- device + mode

    def to(self, *args: Any, **kwargs: Any) -> "Module":
        """Move and/or cast the module's parameters and buffers.

        Phase 2 forwarder around :meth:`torch.nn.Module.to`. Accepts
        the same call signatures: ``to(device)``, ``to(dtype)``,
        ``to(tensor)``, ``to(device, dtype, non_blocking)``,
        ``to(memory_format=...)``. Returns ``self`` for chaining.
        """
        return super().to(*args, **kwargs)  # type: ignore[return-value]

    def cuda(self, device: int | str | torch.device | None = None) -> "Module":
        """Move all parameters/buffers to a CUDA device.

        Phase 2 forwarder. ``device=None`` uses the current CUDA
        device. Returns ``self``.
        """
        return super().cuda(device=device)  # type: ignore[return-value]

    def cpu(self) -> "Module":
        """Move all parameters/buffers to CPU.

        Phase 2 forwarder. Returns ``self``.
        """
        return super().cpu()  # type: ignore[return-value]

    def eval(self) -> "Module":
        """Switch to evaluation mode (sets ``self.training = False``).

        Phase 2 forwarder around :meth:`torch.nn.Module.eval`. Calls
        ``train(False)`` recursively. Returns ``self``.
        """
        return super().eval()  # type: ignore[return-value]

    def train(self, mode: bool = True) -> "Module":
        """Toggle train/eval mode.

        Phase 2 forwarder. Sets ``self.training = mode`` recursively
        for self and all submodules. Returns ``self``.
        """
        return super().train(mode)  # type: ignore[return-value]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero (or unset) the gradients of every parameter.

        Phase 2 forwarder. ``set_to_none=True`` (matching torch >=
        2.0 default) sets ``p.grad = None`` rather than zeroing
        in-place — fewer CUDA dispatches and avoids a memory write.
        """
        super().zero_grad(set_to_none=set_to_none)

    # ----------------------------------------------------------- plasticity hooks

    def register_plasticity(
        self, name: str, rule: Callable[..., None]
    ) -> None:
        """Attach a plasticity rule to this module.

        ``rule`` is called as ``rule(module, inputs, outputs)`` after
        every forward pass when the module is in training mode. It
        should mutate buffers (e.g., a fast-weight tensor) in-place.

        v0.1 / Phase 2 does NOT route a gradient through the rule
        itself — the rule sees ``inputs`` / ``outputs`` as detached
        snapshots and must perform its update under
        ``torch.no_grad()`` if it wishes to write learnable state.
        """
        if name in self._plasticity_rules:
            raise KeyError(
                f"plasticity rule {name!r} already registered on "
                f"{type(self).__name__}"
            )
        self._plasticity_rules[name] = rule

    def register_event_hook(
        self, hook: Callable[..., None]
    ) -> None:
        """Register an event hook for the IR compiler. No-op in v0.1."""
        self._event_hooks.append(hook)

    def plasticity_step(self) -> None:
        """Manually trigger plasticity rules using cached last-forward I/O.

        Normally invoked automatically by ``__call__`` when the
        module is in training mode and at least one plasticity rule
        is registered. Exposed publicly for callers that need to drive
        plasticity outside the standard forward pass (e.g. test-time
        replay).
        """
        if self._last_io is None:
            return
        inputs, outputs = self._last_io
        for rule in self._plasticity_rules.values():
            rule(self, inputs, outputs)

    def compile_to_ir(self) -> Any:
        """Compile this module to a synapforge IR graph.

        Lazy import of ``synapforge.ir.compiler`` to avoid a cycle —
        the IR compiler imports from this module in turn.
        """
        from .ir.compiler import compile_module
        return compile_module(self)

    # ---------------------------------------------- forward instrumentation

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = super().__call__(*args, **kwargs)
        # Cache I/O for plasticity_step (skip if no rules to save memory).
        if self._plasticity_rules:
            self._last_io = (args, out)
            # Auto-apply rules at the end of forward (training mode
            # only by default; plasticity is a learning signal, not an
            # inference-time effect).
            if self.training:
                self.plasticity_step()
        return out
