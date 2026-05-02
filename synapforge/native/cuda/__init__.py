"""synapforge.native.cuda -- torch-free GPU compute layer.

Public API
----------

    from synapforge.native.cuda import (
        CudaTensor,         # numpy-API wrapper over cupy.ndarray
        ops,                # cuBLAS / cuDNN-backed elementwise + reductions
        CudaMemPool,        # block allocator on top of cupy.cuda.MemoryPool
        CudaStreamPool,     # 4-stream pool for compute / H2D / D2H / misc
        triton_glue,        # bridge so Triton kernels accept CudaTensor
        HAS_CUPY,           # bool: True iff a real GPU+cupy is reachable
        cupy_or_numpy,      # the active xp module (cp if HAS_CUPY else np)
    )

Hard rule: NO ``import torch`` lives anywhere under
``synapforge/native/cuda/``. Triton itself depends on torch internally,
but our wrapper API never *exposes* torch.Tensor -- the bridge in
``triton_glue.py`` materializes torch.Tensor only inside its own scope
to satisfy the kernel ABI, then unwraps the result back into CudaTensor.

cupy is OPTIONAL. When ``import cupy`` fails (e.g. CI runners without
GPU, or our Windows dev box), every public symbol still imports and
falls back transparently to numpy. Tests can guard with::

    pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="no GPU")

or simply call CudaTensor functions and accept the numpy fallback.
"""
from __future__ import annotations

# Lazy / fallback-tolerant import. cupy.ndarray is one of the most
# expensive Python imports in the ecosystem (loads CUDA libs at import
# time). We re-export ``HAS_CUPY`` so tests can skip gracefully.
HAS_CUPY = False
try:  # pragma: no cover -- depends on host
    import cupy as _cp

    # Probe a tiny op to confirm CUDA actually works (cupy may import
    # but the runtime is broken on machines with mismatched drivers).
    try:
        _cp.zeros(1, dtype=_cp.float32) + 0
        HAS_CUPY = True
    except Exception:
        HAS_CUPY = False
except Exception:  # pragma: no cover
    _cp = None
    HAS_CUPY = False

import numpy as _np  # always available

cupy_or_numpy = _cp if HAS_CUPY else _np

# Re-exports (defer module imports so optional deps don't break).
from .tensor import CudaTensor  # noqa: E402

# ops / allocator / streams / triton_glue are lighter; tests import
# them directly from the submodule, but expose them here for
# convenience too.
from . import allocator  # noqa: E402,F401
from . import lnn_ops  # noqa: E402,F401
from . import ops  # noqa: E402,F401
from . import streams  # noqa: E402,F401
from . import triton_glue  # noqa: E402,F401
from .allocator import CudaMemPool  # noqa: E402
from .streams import CudaStreamPool  # noqa: E402

__all__ = [
    "CudaTensor",
    "CudaMemPool",
    "CudaStreamPool",
    "ops",
    "lnn_ops",
    "triton_glue",
    "allocator",
    "streams",
    "HAS_CUPY",
    "cupy_or_numpy",
]
