"""synapforge.native -- pure-numpy native kernels and gradient catalogues.

Sub-packages
------------
vjp/
    Closed-form vector-Jacobian products for every op in HybridBlock.
    Zero torch imports; fp32 default with optional dtype promotion.

This package is the long-term replacement for torch.autograd in the
SynapForge training and inference paths. The closed forms here avoid
the float-rounding drift introduced by chain-rule autograd traversal
of the LNN+SNN hybrid block.
"""

from __future__ import annotations

__all__ = ["vjp"]
