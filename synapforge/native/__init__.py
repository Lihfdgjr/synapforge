"""synapforge.native — native (single-dispatch) compute kernels.

Sub-packages:

* ``kernel/`` — fused HybridBlock Triton kernels (forward + closed-form
  backward) plus PyTorch glue (autograd.Function bridge).
"""
from __future__ import annotations
