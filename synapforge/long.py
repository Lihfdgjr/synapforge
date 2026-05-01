"""synapforge.long — long-context inference utilities.

This module is the **stable public API surface** for the 50M effective-
context claim (MASTER_PLAN §11, P20/P21). The heavy machinery lives in
:mod:`synapforge.infinite` (5-tier memory hierarchy: RoPE / Local-GQA /
hierarchical compress / slow-tau / SSM scan / external vector store /
disk archive); this module wires it together with the inference-time
STDP toggle so callers do not have to know the internals.

Why a separate module
---------------------
The validation harness in ``tests/integration/test_long_context_50m.py``
needs ONE place to:

1. Force inference-time STDP **on** or **off** for the A/B that proves
   the monotonic-quality claim (P21).
2. Read the L2 norm of the running STDP fast-weight matrix to confirm
   it is not exploding past ~50M tokens (P20).
3. Reset all per-document plasticity state between independent runs so
   one length's accumulated weights do not pollute the next.

The single-source-of-truth for the gate is the ``SYNAPFORGE_STDP_INFERENCE``
environment variable, read inside :class:`STDPFastWeight.forward` at
``synapforge/bio/stdp_fast.py:127``. Setting it via this module's
:func:`set_stdp_inference` is preferred over poking ``os.environ``
directly so the toggle is grep-able and we have ONE call-site to update
when the gate is moved into a config object.

Usage
-----
.. code-block:: python

    import synapforge.long as long
    from synapforge.model_100m import build_synapforge_100m

    model = build_synapforge_100m(...)

    # A/B: same prompt at different lengths, with STDP on vs off.
    long.set_stdp_inference("off")
    ppl_off = eval_at_length(model, length=10_000)

    long.set_stdp_inference("on")     # default
    long.reset_stdp(model)             # clear plasticity state
    ppl_on = eval_at_length(model, length=10_000)

    # Read STDP weight L2 norm after the run for non-explosion check.
    stdp_norm = long.stdp_weight_norm(model)

This module imports lazily — touching it does NOT pull in torch unless
the caller actually invokes a function that needs it. Keeps
``pytest --collect-only`` cheap on a torch-less dev box.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:  # pragma: no cover -- type hints only
    import torch
    import torch.nn as nn


# Public env-var name. Read inside STDPFastWeight.forward.
STDP_ENV_VAR = "SYNAPFORGE_STDP_INFERENCE"
STDP_VALID_MODES = ("on", "off", "decay")


def set_stdp_inference(mode: str) -> str:
    """Toggle inference-time STDP plasticity globally.

    Parameters
    ----------
    mode : str
        One of ``"on"`` (forward-only Hebbian during eval, the default
        and the *paper claim*), ``"off"`` (freeze fast weights, classic
        transformer-style inference), or ``"decay"`` (on + per-step
        consolidation tick that pulls W toward zero between docs).

    Returns
    -------
    str
        The previous mode (or ``""`` if unset). Useful for nested
        save/restore in tests.

    Notes
    -----
    The change takes effect on the next ``STDPFastWeight.forward`` call.
    Modules already mid-forward keep the previous mode for that call.
    """
    if mode not in STDP_VALID_MODES:
        raise ValueError(
            f"mode must be one of {STDP_VALID_MODES}, got {mode!r}"
        )
    prev = os.environ.get(STDP_ENV_VAR, "")
    os.environ[STDP_ENV_VAR] = mode
    return prev


def get_stdp_inference() -> str:
    """Return the current inference-STDP mode (default ``"on"``)."""
    return os.environ.get(STDP_ENV_VAR, "on")


def disable_stdp_at_inference() -> str:
    """Convenience: ``set_stdp_inference("off")``. Returns prev mode."""
    return set_stdp_inference("off")


def enable_stdp_at_inference() -> str:
    """Convenience: ``set_stdp_inference("on")``. Returns prev mode."""
    return set_stdp_inference("on")


def reset_stdp(model) -> int:
    """Reset every STDPFastWeight (or compatible) module in ``model``.

    Walks ``model.modules()`` and calls ``reset_doc_state()`` on every
    module that has it (the ``STDPFastWeight`` class and any other
    module that opts into the same reset protocol).

    Returns the number of modules reset.
    """
    n = 0
    for m in model.modules():
        fn = getattr(m, "reset_doc_state", None)
        if callable(fn):
            fn()
            n += 1
    return n


def stdp_modules(model) -> "Iterator[nn.Module]":
    """Yield every STDP fast-weight module in ``model``.

    Identified by having a ``W`` buffer of rank 2 plus the
    ``reset_doc_state`` method (the STDPFastWeight protocol).
    """
    for m in model.modules():
        W = getattr(m, "W", None)
        if W is None:
            continue
        if hasattr(W, "dim") and W.dim() == 2 and callable(
            getattr(m, "reset_doc_state", None)
        ):
            yield m


def stdp_weight_norm(model) -> float:
    """Return Frobenius (L2) norm summed over every STDP fast-weight.

    Used by the long-context harness to assert ``W`` does not explode at
    50M tokens. A summed norm is a sufficient scalar — exploding in any
    single layer is captured.

    Returns 0.0 if the model has no STDP modules.
    """
    import torch  # local import to keep collection cheap

    total = 0.0
    for m in stdp_modules(model):
        W = m.W.detach()
        # use float() to dodge bf16 underflow on small magnitudes
        total += float(torch.linalg.norm(W.float()))
    return total


def stdp_weight_norms(model) -> dict[str, float]:
    """Per-module L2 norms keyed by ``model.named_modules()`` name."""
    import torch

    out: dict[str, float] = {}
    name_lookup = {id(m): n for n, m in model.named_modules()}
    for m in stdp_modules(model):
        n = name_lookup.get(id(m), "")
        out[n] = float(torch.linalg.norm(m.W.detach().float()))
    return out


def chunked_token_stream(
    total_len: int,
    chunk: int = 4096,
    vocab: int = 151_936,
    seed: int = 0,
) -> "Iterator[torch.Tensor]":
    """Yield ``[1, chunk]`` int64 token chunks summing to ``total_len``.

    A generator so the validation harness can stream 50M tokens without
    ever materializing them all in memory.  Tokens are pseudo-random
    per-chunk; the same seed reproduces the same stream.

    Last chunk is short if ``total_len`` is not a multiple of ``chunk``.
    """
    import torch

    if total_len <= 0:
        return
    g = torch.Generator()
    g.manual_seed(int(seed))
    remaining = int(total_len)
    chunk = max(1, int(chunk))
    while remaining > 0:
        n = min(chunk, remaining)
        yield torch.randint(0, int(vocab), (1, n), generator=g, dtype=torch.long)
        remaining -= n


__all__ = [
    "STDP_ENV_VAR",
    "STDP_VALID_MODES",
    "set_stdp_inference",
    "get_stdp_inference",
    "disable_stdp_at_inference",
    "enable_stdp_at_inference",
    "reset_stdp",
    "stdp_modules",
    "stdp_weight_norm",
    "stdp_weight_norms",
    "chunked_token_stream",
]
