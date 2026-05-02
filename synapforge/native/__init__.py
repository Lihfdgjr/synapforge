"""synapforge.native -- pure-numpy / closed-form-VJP training stack.

This package is the home for the zero-torch SynapForge native runtime:
op-level VJPs, dispatch, CUDA bridge (when torch is *not* in the
production hot path), and the saturation / roofline profiler under
``synapforge/native/bench``.

No ``import torch`` is allowed in production code under this package.
Test code may use torch.
"""
