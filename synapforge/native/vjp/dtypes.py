"""Shared dtype helpers for the VJP catalogue.

Centralises the bf16 / fp32 promotion logic so each op file stays
focused on its math. All functions are pure numpy.

Convention
----------
* "Compute dtype" = the dtype the math runs in. Default fp32.
* "Storage dtype" = the dtype the user passed in (could be bf16 simulated
  via numpy's float32 cast, or actual fp32).

We always do the *math* in fp32 because numpy bf16 is not native -- the
nearest-representable trick only saves memory in the hosts, not compute.
But we honour the storage dtype on output so callers can keep
mixed-precision invariants.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_dtype(x: np.ndarray, override: Optional[np.dtype] = None) -> np.dtype:
    """Pick the dtype the math should run in.

    Default: fp32. If ``override`` is provided, return that. If ``x`` is
    already fp64 we keep fp64 (callers know what they're doing).
    """
    if override is not None:
        return np.dtype(override)
    if x.dtype == np.float64:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def to_compute(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast to the compute dtype if needed; no copy when dtypes match."""
    if x.dtype == dtype:
        return x
    return x.astype(dtype, copy=False)


def to_storage(x: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Cast back to the storage dtype if needed."""
    if x.dtype == target_dtype:
        return x
    return x.astype(target_dtype, copy=False)


def matches_dtype(*arrays: np.ndarray) -> bool:
    """Return True iff all arrays share the same dtype.

    Used by callers that want to assert dtype consistency before kicking
    off a kernel where mixed dtypes would silently downcast.
    """
    if not arrays:
        return True
    first = arrays[0].dtype
    return all(a.dtype == first for a in arrays[1:])


__all__ = ["compute_dtype", "to_compute", "to_storage", "matches_dtype"]
