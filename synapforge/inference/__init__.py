"""synapforge.inference — generation harness with R-fold CfC + Coconut latent.

Decouples the live decoder from training-time bulk-prefill code paths.
The two production entry points are:

* :func:`rfold_chat.generate_rfold` — autoregressive decode that carries
  per-block CfC + PLIF state forward and uses R-fold closed-form for the
  per-token CfC update (constant-time per-token regardless of context
  length).
* :func:`coconut_loop.generate_with_coconut` — wraps R-fold decode with
  k=8 latent thinking steps at ``<bot>`` markers (arxiv:2412.06769).

Both are OFF by default in the existing demo CLI (``synapforge-demo
chat``); pass the explicit flag to opt in.
"""

from __future__ import annotations

from .coconut_loop import (  # noqa: F401
    coconut_step,
    generate_with_coconut,
)
from .rfold_chat import (  # noqa: F401
    InferenceState,
    generate_rfold,
    incremental_step,
    prefill_state,
)

__all__ = [
    "InferenceState",
    "generate_rfold",
    "incremental_step",
    "prefill_state",
    "coconut_step",
    "generate_with_coconut",
]
