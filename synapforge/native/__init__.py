"""synapforge.native -- torch-free numerical core (CPU + GPU).

Subpackages:
    cuda/   -- GPU side, backed by cupy + cuBLAS, no torch import.
               Bridges to existing Triton kernels.

Sister agents own (don't touch):
    vjp/        -- VJP catalog
    dispatch/   -- async dispatcher
    data/       -- numpy-only loader
"""
