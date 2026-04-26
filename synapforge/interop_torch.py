"""sf.interop_torch — bidirectional bridge with vanilla nn.Module.

Designed so a developer with an existing PyTorch project can drop synapforge
cells in seamlessly and convert between the two worlds without rewriting:

  1. Use synapforge cells inside ``nn.Sequential`` / ``nn.ModuleList`` —
     since ``sf.Module`` already inherits from ``nn.Module``, this works
     directly. ``SFAsTorchModule`` is a thin wrapper for explicit "make
     this look like a vanilla nn.Module" semantics (and to surface a
     ``plasticity_step()`` proxy if the inner module has plasticity rules).
  2. Wrap a synapforge model with ``torch.compile`` / ``torch.jit`` /
     ``torch.optim.AdamW`` — works on the wrapped object the same as any
     other ``nn.Module``.
  3. Convert ``torch.nn.Linear`` to :class:`sf.SparseSynapse` (and back)
     for retrofit-style plasticity adoption.
  4. Save/load synapforge models via ``torch.save`` / ``torch.load`` using
     a regular ``state_dict`` — supported by inheritance.

This module never raises on a missing/extra layer: it is meant to be a
"pour-in" bridge for live PyTorch codebases. Verbose warnings are emitted
when something is dropped or when a substitution is unlikely to be sensible
(e.g., ``replace_relu_with_plif`` is rarely correct because PLIF is a
stateful spiking neuron, not a pointwise nonlinearity).
"""
from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .cells.plif import PLIF
from .cells.synapse import SparseSynapse
from .module import Module as SFModule

__all__ = [
    "SFAsTorchModule",
    "TorchAsSFModule",
    "replace_linear_with_sparse",
    "replace_relu_with_plif",
    "convert_sparse_to_linear",
    "list_replaceable_modules",
]


# ---------------------------------------------------------------------------
# SFAsTorchModule — make a sf.Module look like a vanilla nn.Module
# ---------------------------------------------------------------------------


class SFAsTorchModule(nn.Module):
    """Wraps a sf.Module so it looks/acts like a vanilla nn.Module.

    Use case: drop into a standard PyTorch training loop or compose with
    ``nn.Sequential`` / DDP / ``torch.compile``.

    Notes:
      * ``sf.Module`` already inherits from ``nn.Module``, so technically
        any sf.Module *is* an nn.Module already. The wrapper exists so
        callers can:
          - opt out of the ``Module.__call__`` plasticity auto-trigger
            (set ``auto_plasticity=False``),
          - keep the module addressable as ``self.inner`` in a parent that
            does not want to use synapforge-specific attributes.
      * If the inner module exposes ``plasticity_step()``, it is callable on
        the wrapper too. ``torch.compile`` can graph-trace plasticity-free
        forwards; plasticity buffers are skipped during tracing.

    Args:
        sf_module:        the synapforge module to wrap.
        auto_plasticity:  if False, suppress sf.Module's auto plasticity
                          trigger inside ``__call__`` (handy when running
                          ``torch.compile``, which dislikes Python branches).
    """

    def __init__(self, sf_module: nn.Module, *, auto_plasticity: bool = True) -> None:
        super().__init__()
        if not isinstance(sf_module, nn.Module):
            raise TypeError(
                f"SFAsTorchModule expects nn.Module/sf.Module, got "
                f"{type(sf_module).__name__}"
            )
        self.inner = sf_module
        self._auto_plasticity = bool(auto_plasticity)

    def forward(self, *args, **kwargs):
        if not self._auto_plasticity and isinstance(self.inner, SFModule):
            # Bypass sf.Module.__call__ so plasticity_step() is not auto-fired
            # and Python branches don't end up in the compile graph.
            return nn.Module.__call__(self.inner, *args, **kwargs)
        return self.inner(*args, **kwargs)

    def plasticity_step(self) -> None:
        if hasattr(self.inner, "plasticity_step"):
            self.inner.plasticity_step()

    # Ergonomic delegations — let users go ``wrapper.tau`` etc. transparently.
    def __getattr__(self, name: str):  # pragma: no cover - usual delegation
        # nn.Module __getattr__ is invoked only when the attribute is not on
        # the instance dict. Try inner only after falling through.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is not None and hasattr(inner, name):
                return getattr(inner, name)
            raise


# ---------------------------------------------------------------------------
# TorchAsSFModule — lift a vanilla nn.Module into sf-land
# ---------------------------------------------------------------------------


class TorchAsSFModule(SFModule):
    """Wraps a vanilla nn.Module so it is treated as a sf.Module.

    Use case: a HuggingFace ``BertModel`` (or any ``nn.Module``) becomes a
    valid sub-module under a synapforge model, can have a plasticity rule
    registered on top of it, and gets the IR compile hook for free.

    The inner module's forward signature is preserved verbatim. ``state_dict``
    keys are ``inner.<original_key>``.
    """

    def __init__(self, torch_module: nn.Module) -> None:
        super().__init__()
        if not isinstance(torch_module, nn.Module):
            raise TypeError(
                f"TorchAsSFModule expects an nn.Module, got "
                f"{type(torch_module).__name__}"
            )
        self.inner = torch_module

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


# ---------------------------------------------------------------------------
# Layer-replacement helpers
# ---------------------------------------------------------------------------


def list_replaceable_modules(
    model: nn.Module, target_cls: type
) -> List[Tuple[str, nn.Module]]:
    """Return ``(qualified_name, module)`` pairs for instances of ``target_cls``."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, target_cls)]


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace ``root.<qualified_name>`` with ``new_module``.

    ``qualified_name`` follows ``named_modules`` convention (``a.b.c``).
    Skips silently for the empty string (the root itself).
    """
    if qualified_name == "":
        raise ValueError("cannot set root module via _set_submodule")
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def replace_linear_with_sparse(
    model: nn.Module,
    density: float = 0.05,
    *,
    skip_modules: Optional[Iterable[str]] = None,
    copy_weights: bool = True,
    verbose: bool = False,
) -> List[str]:
    """Walk ``model``; replace each ``nn.Linear`` with :class:`sf.SparseSynapse`.

    Useful for retrofit-ing existing models to add structural plasticity
    without rebuilding them. Returns the list of qualified names that were
    actually replaced.

    Args:
        model:        ``nn.Module`` to mutate in-place.
        density:      target density for the sparse mask (1.0 = fully dense).
        skip_modules: qualified names to leave untouched (e.g.
                      ``["lm_head"]`` to keep the output projection dense).
        copy_weights: copy the original linear weights/bias into the new
                      sparse synapse before masking. Default True so the
                      forward output is approximately preserved.
        verbose:      print one line per replacement.

    Returns:
        List of qualified names of replaced linears.
    """
    skip = set(skip_modules or ())
    replaced: List[str] = []
    targets = list_replaceable_modules(model, nn.Linear)
    # SparseSynapse is itself a Linear-like; don't recurse into our own.
    for qn, mod in targets:
        if isinstance(mod, SparseSynapse):
            continue
        if qn in skip:
            continue
        if qn == "":
            warnings.warn(
                "replace_linear_with_sparse: root model is itself nn.Linear; "
                "wrap it inside an nn.Module before calling."
            )
            continue
        out_dim, in_dim = mod.out_features, mod.in_features
        bias = mod.bias is not None
        new = SparseSynapse(in_dim, out_dim, sparsity=float(density), bias=bias)
        new.to(device=mod.weight.device, dtype=mod.weight.dtype)
        if copy_weights:
            with torch.no_grad():
                new.weight.copy_(mod.weight.detach())
                if bias and mod.bias is not None:
                    new.bias.copy_(mod.bias.detach())
        _set_submodule(model, qn, new)
        replaced.append(qn)
        if verbose:
            print(
                f"[interop_torch] replaced {qn}: "
                f"Linear({in_dim}->{out_dim}) -> SparseSynapse(density={density})",
                flush=True,
            )
    return replaced


def replace_relu_with_plif(
    model: nn.Module,
    threshold: float = 1.0,
    *,
    skip_modules: Optional[Iterable[str]] = None,
    hidden_dim_hint: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """Walk ``model``; replace each ``nn.ReLU`` with :class:`sf.PLIF`.

    .. warning::
       PLIF is a STATEFUL spiking neuron with persistent membrane potential
       semantics. Replacing ``nn.ReLU`` (a pointwise stateless nonlinearity)
       with PLIF changes the network's dynamics fundamentally and is rarely
       what you want in a feed-forward model. We emit a warning by default;
       set ``verbose=False`` to silence it.

    The PLIF constructor needs a hidden-dim. We try to recover it from the
    closest preceding ``nn.Linear`` / ``nn.LayerNorm`` / ``nn.Conv*`` (last
    output channel). If we cannot infer it, fall back to ``hidden_dim_hint``;
    otherwise raise ``ValueError`` with the offending name.

    PLIF returns ``(spike, membrane)`` whereas ``nn.ReLU`` returns a single
    tensor. We wrap PLIF so it returns just the spike to keep the upstream
    API the same. The membrane is dropped — if you need it, attach a PLIF
    yourself rather than going through this helper.

    Returns:
        List of qualified names of replaced ReLUs.
    """
    skip = set(skip_modules or ())
    replaced: List[str] = []
    targets = list_replaceable_modules(model, nn.ReLU)
    if not targets:
        return replaced

    if verbose:
        warnings.warn(
            "replace_relu_with_plif: PLIF is a stateful spiking neuron — this "
            "swap changes network dynamics fundamentally. Verify behaviour."
        )

    # Pre-build a mapping from each name -> the latest dim seen prior.
    dim_by_name: dict[str, Optional[int]] = {}
    last_dim: Optional[int] = hidden_dim_hint
    for n, m in model.named_modules():
        # Update last_dim when we see something with a clear out-channel.
        if isinstance(m, nn.Linear):
            last_dim = m.out_features
        elif isinstance(m, nn.LayerNorm):
            normalized_shape = m.normalized_shape
            if isinstance(normalized_shape, tuple):
                last_dim = normalized_shape[0]
            else:
                last_dim = int(normalized_shape)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            last_dim = m.out_channels
        elif isinstance(m, SparseSynapse):
            last_dim = m.out_dim
        dim_by_name[n] = last_dim

    class _PLIFAsRelu(nn.Module):
        """Adapter: wraps a PLIF cell to return just the spike tensor."""

        def __init__(self, dim: int, threshold: float) -> None:
            super().__init__()
            self.plif = PLIF(dim, threshold=float(threshold))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            spk, _mem = self.plif(x)
            return spk

    # Pick a device/dtype reference from the first parameter of the parent
    # model so the new PLIF lives on the same device. PLIF allocates internal
    # tau_log / threshold / last_spike_rate buffers on CPU at __init__ time.
    ref_param = next(model.parameters(), None)
    ref_device = ref_param.device if ref_param is not None else torch.device("cpu")
    ref_dtype = ref_param.dtype if ref_param is not None else torch.float32

    for qn, _mod in targets:
        if qn in skip:
            continue
        dim = dim_by_name.get(qn)
        if dim is None:
            raise ValueError(
                f"replace_relu_with_plif: could not infer hidden_dim for "
                f"ReLU at {qn!r}; pass hidden_dim_hint=<int>."
            )
        new = _PLIFAsRelu(int(dim), threshold=float(threshold))
        new.to(device=ref_device, dtype=ref_dtype)
        _set_submodule(model, qn, new)
        replaced.append(qn)
        if verbose:
            print(
                f"[interop_torch] replaced {qn}: ReLU -> PLIF(dim={dim}, "
                f"threshold={threshold})",
                flush=True,
            )
    return replaced


def convert_sparse_to_linear(
    model: nn.Module, *, verbose: bool = False
) -> List[str]:
    """Walk ``model``; replace each :class:`sf.SparseSynapse` back to ``nn.Linear``.

    Multiplies the weight by the boolean mask before copying so the
    forward output is preserved bit-exact. Use this when you want to
    "freeze" a synaptogenesis-trained model into a vanilla PyTorch
    checkpoint (no ``mask`` buffer, no SF dependency on the consumer side).

    Returns:
        List of qualified names of replaced SparseSynapses.
    """
    replaced: List[str] = []
    targets = list_replaceable_modules(model, SparseSynapse)
    for qn, mod in targets:
        if qn == "":
            warnings.warn("convert_sparse_to_linear: root is SparseSynapse; skipping.")
            continue
        bias = mod.bias is not None
        new = nn.Linear(mod.in_dim, mod.out_dim, bias=bias)
        new.to(device=mod.weight.device, dtype=mod.weight.dtype)
        with torch.no_grad():
            masked_w = mod.weight.detach() * mod.mask.to(mod.weight.dtype)
            new.weight.copy_(masked_w)
            if bias and mod.bias is not None:
                new.bias.copy_(mod.bias.detach())
        _set_submodule(model, qn, new)
        replaced.append(qn)
        if verbose:
            print(f"[interop_torch] replaced {qn}: SparseSynapse -> Linear", flush=True)
    return replaced
