"""synapforge.native.vjp -- closed-form Vector-Jacobian Product catalogue.

Every op exposes ``<op>_fwd`` and ``<op>_bwd`` pure numpy functions:

* ``embed``         token embedding lookup with sparse grad accumulation
* ``linear``        affine y = x @ W.T + b
* ``rmsnorm``       Root-Mean-Square LayerNorm (Zhang & Sennrich 2019)
* ``swiglu``        SwiGLU FFN (Shazeer 2020), 3-linear-layer composite
* ``cfc``           Liquid CfC continuous-time recurrence (Hasani et al. 2022)
* ``plif``          Parametric Leaky-Integrate-Fire (Fang et al. ICCV 2021)
* ``sew_shortcut``  SEW spike-element-wise residual (Fang et al. NeurIPS 2021)
* ``cross_entropy`` standard softmax cross-entropy

Forward functions return either a single tensor or a tuple
``(output, *saved_for_bwd)``. Backward functions accept ``grad_output``
plus the saved tensors and return parameter gradients as a tuple.

All ops are pure numpy. fp32 is the default precision; pass ``dtype=...``
to promote to / demote from bf16 explicitly.
"""

from __future__ import annotations

from synapforge.native.vjp import (
    cfc,
    cross_entropy,
    dtypes,
    embed,
    linear,
    plif,
    rmsnorm,
    sew_shortcut,
    swiglu,
)

__all__ = [
    "cfc",
    "cross_entropy",
    "dtypes",
    "embed",
    "linear",
    "plif",
    "rmsnorm",
    "sew_shortcut",
    "swiglu",
]
