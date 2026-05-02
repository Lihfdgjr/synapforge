"""per_block_router.py -- per-block (per-layer) device routing.

What this enables
-----------------
Different blocks of the same model live on different devices at
runtime. Example config for a 16-block LNN+SNN model on a host with
80 GB GPU and 128 GB RAM::

    {
        "embed":       "cpu",     # huge V x d table -- park on CPU
        "layers_0_7":  "cuda",    # heavy attention/CfC kernels on GPU
        "layers_8_15": "cpu",     # bandwidth-bound layers on CPU MKL
        "lm_head":     "cuda",    # fused softmax+CE on GPU
    }

The router walks the block list at forward time, dispatches each
block to its assigned device, and (if the next block is on a
different device) issues a non-blocking H2D / D2H copy on the
transfer stream so the next block's compute overlaps with the move.

Why per-block routing matters for our model
-------------------------------------------
A 730M LNN+SNN block is dominated by:

    a) MatMul-heavy parts (CfC delta, FFN gate/up/down, lm_head)
       -- GPU-sweet (memory-bandwidth bounded by HBM).
    b) PLIF spike accumulator
       -- pure elementwise ATan-surrogate, ~constant-cost vs sequence
       length, and the GPU under-utilises while waiting for the
       upstream CfC. Running PLIF on CPU AVX2 actually *removes* a
       sync barrier on the GPU stream and lets H2D for the next
       layer overlap.
    c) RMSNorm/SwiGLU activation
       -- elementwise; small; either device works.

If ``t_cpu_layer >> t_gpu_layer`` then putting layers on CPU is a
loss; if ``t_cpu_layer ~= t_gpu_layer`` (typical for small d-model
LNN+SNN with sparse spikes) the throughput curve becomes the sum
of the two devices, which is the point.

Design
------
The router does NOT own the block functions; it owns the device
assignments and the cross-device transfer plumbing. The user passes:

    forward_fn(block_idx, x_dev, params_dev) -> y_dev

which is responsible for actually running the block's math on
``x_dev`` (already on the right device per ``device_for(block_idx)``).
The router handles the moves.

Hard constraint
---------------
**No ``import torch``.** numpy + cupy via the ``streams`` module.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from synapforge.native.dispatch.streams import (
    CUPY_AVAILABLE,
    StreamPair,
    asnumpy,
    to_device,
)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

# Range token: "layers_0_7", "layers_8_15", "blocks_0_3" etc.
_RANGE_RE = re.compile(r"^(?:layers|blocks)_(\d+)_(\d+)$")

# Single-block tokens (e.g. lm_head, embed).
_SINGLE_TOKENS = ("embed", "lm_head", "head", "proj_in", "proj_out", "norm")


def parse_routing_config(
    config: Dict[str, str], num_blocks: int,
) -> Dict[str, str]:
    """Expand a routing config into a per-block ``{name: device}`` map.

    Input
    -----
    ``config`` keys are either:

    * single-block names (``"embed"``, ``"lm_head"``, ``"layers_3"``)
    * range names (``"layers_0_7"`` -- inclusive on both ends)

    Output
    ------
    A dict with one entry per layer index plus any single-block names
    in the input. Layer keys are formatted as ``"layers_0"``,
    ``"layers_1"``, .... Devices are normalised to lowercase ``"cpu"``
    or ``"cuda"``.

    Validation
    ----------
    * Every layer index in [0, num_blocks) MUST be assigned by some
      key, or we raise ``ValueError`` (no implicit defaults -- forces
      the user to be explicit).
    * Overlapping ranges are an error.
    """
    if not isinstance(config, dict):
        raise TypeError(f"routing config must be dict, got {type(config).__name__}")
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be >= 0, got {num_blocks}")

    layer_to_device: Dict[int, str] = {}
    extra: Dict[str, str] = {}

    for key, dev in config.items():
        norm_dev = _normalise_device(dev)
        m = _RANGE_RE.match(key)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo < 0 or hi >= num_blocks or lo > hi:
                raise ValueError(
                    f"routing range {key!r} out of [0,{num_blocks-1}] or lo>hi")
            for i in range(lo, hi + 1):
                if i in layer_to_device:
                    raise ValueError(
                        f"layer {i} assigned twice in routing config")
                layer_to_device[i] = norm_dev
            continue
        # Single layer "layers_3"
        if key.startswith("layers_") or key.startswith("blocks_"):
            try:
                idx = int(key.split("_", 1)[1])
            except ValueError as exc:
                raise ValueError(f"bad layer key {key!r}") from exc
            if idx < 0 or idx >= num_blocks:
                raise ValueError(
                    f"layer index {idx} out of [0,{num_blocks-1}]")
            if idx in layer_to_device:
                raise ValueError(f"layer {idx} assigned twice in routing config")
            layer_to_device[idx] = norm_dev
            continue
        # Single tokens (embed, lm_head, ...)
        if key in _SINGLE_TOKENS or key.replace("_", "").isalpha():
            extra[key] = norm_dev
            continue
        raise ValueError(f"unrecognised routing key {key!r}")

    # Every layer must be assigned.
    missing = [i for i in range(num_blocks) if i not in layer_to_device]
    if missing:
        raise ValueError(
            f"routing config missing layers: {missing}. "
            f"add an entry like 'layers_{missing[0]}': 'cpu' (or 'cuda')")

    out: Dict[str, str] = {f"layers_{i}": d for i, d in sorted(layer_to_device.items())}
    out.update(extra)
    return out


def _normalise_device(dev: str) -> str:
    s = str(dev).strip().lower()
    s = s.split(":")[0]
    if s in ("cpu", "cuda"):
        return s
    if s in ("gpu",):
        return "cuda"
    raise ValueError(f"unknown device {dev!r}; expected cpu or cuda")


# ---------------------------------------------------------------------------
# PerBlockRouter
# ---------------------------------------------------------------------------

class PerBlockRouter:
    """Forward+backward dispatcher with per-block device assignments.

    Usage
    -----
    Build the router::

        router = PerBlockRouter(
            num_blocks=16,
            config={
                "embed":       "cpu",
                "layers_0_7":  "cuda",
                "layers_8_15": "cpu",
                "lm_head":     "cuda",
            },
        )

    Run the per-block forward::

        x = embedding_lookup(input_ids)            # CPU output
        x = router.move(x, "embed", "layers_0")    # CPU -> GPU async
        for i in range(16):
            x = router.move(x, f"layers_{i-1}" if i else "embed",
                            f"layers_{i}")
            x = block_fwd(i, x, block_params[i])
        x = router.move(x, "layers_15", "lm_head")
        logits = lm_head_fwd(x, lm_params)

    The router is intentionally orthogonal to the gradient bookkeeping
    in :mod:`synapforge.native.vjp` -- you pass the same routing
    sequence in reverse for backward.

    Notes
    -----
    * On a CPU-only host (``CUPY_AVAILABLE = False``) all ``move``
      calls are passthrough, so the same code runs unchanged.
    * The router holds *one* :class:`StreamPair` per device hop. With
      a small set of distinct device boundaries (typically 2-4) the
      stream pool is tiny; we don't bother to GC.
    """

    __slots__ = (
        "num_blocks",
        "expanded_config",
        "_layer_devices",
        "_extra_devices",
        "_streampairs",
        "_metrics",
    )

    def __init__(
        self,
        num_blocks: int,
        config: Dict[str, str],
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.expanded_config = parse_routing_config(config, self.num_blocks)
        self._layer_devices: List[str] = [
            self.expanded_config[f"layers_{i}"] for i in range(self.num_blocks)
        ]
        self._extra_devices: Dict[str, str] = {
            k: v for k, v in self.expanded_config.items()
            if not k.startswith("layers_")
        }
        # One StreamPair per (src_dev, dst_dev) pair; allocated lazily.
        self._streampairs: Dict[Tuple[str, str], StreamPair] = {}
        # Move metrics (bytes moved per direction; useful for cost analysis).
        self._metrics: Dict[str, int] = {
            "h2d_bytes": 0, "d2h_bytes": 0, "moves": 0, "noop_moves": 0,
        }

    # ----- public API ------------------------------------------------------

    def device_for(self, name: str) -> str:
        """Return the device assigned to a block name (or layer index)."""
        if isinstance(name, int):
            return self._layer_devices[name]
        if name in self._extra_devices:
            return self._extra_devices[name]
        if name.startswith("layers_") or name.startswith("blocks_"):
            try:
                idx = int(name.split("_", 1)[1])
            except ValueError as exc:
                raise KeyError(name) from exc
            return self._layer_devices[idx]
        raise KeyError(f"no device assigned for {name!r}")

    def move(self, x: Any, src_name: str, dst_name: str) -> Any:
        """Move tensor ``x`` from the device of ``src_name`` to that of
        ``dst_name``. Returns the new tensor.

        Same-device hops are passthrough (counted as ``noop_moves``).
        H2D / D2H hops use the cached :class:`StreamPair` for the
        ``(src, dst)`` direction so consecutive moves overlap with
        compute.
        """
        src = self.device_for(src_name) if not isinstance(src_name, str) or src_name in self.expanded_config or src_name.startswith("layers_") else _normalise_device(src_name)
        dst = self.device_for(dst_name) if not isinstance(dst_name, str) or dst_name in self.expanded_config or dst_name.startswith("layers_") else _normalise_device(dst_name)

        # Update accounting.
        if src == dst:
            self._metrics["noop_moves"] += 1
            return x
        self._metrics["moves"] += 1
        # Pick / cache stream pair.
        sp = self._streampairs.get((src, dst))
        if sp is None:
            sp = StreamPair()
            self._streampairs[(src, dst)] = sp
        # Bytes count (works on np or cupy arrays).
        try:
            nbytes = int(getattr(x, "nbytes", 0))
        except Exception:
            nbytes = 0
        if dst == "cuda":
            self._metrics["h2d_bytes"] += nbytes
            return sp.copy_to_device_async(asnumpy(x))
        # dst == "cpu"
        self._metrics["d2h_bytes"] += nbytes
        if not CUPY_AVAILABLE:
            return asnumpy(x)
        return sp.copy_to_host_async(x)

    def synchronize(self) -> None:
        """Wait for all in-flight transfers to finish."""
        for sp in self._streampairs.values():
            sp.synchronize()

    # ----- helpers ---------------------------------------------------------

    @property
    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        for k in self._metrics:
            self._metrics[k] = 0

    def is_uniform(self, device: Optional[str] = None) -> bool:
        """True if every block is on the same device.

        With ``device=None`` returns True iff all blocks share *some*
        common device. With ``device='cpu'`` / ``'cuda'`` returns True
        iff that device is the common one.
        """
        if not self._layer_devices:
            return True
        first = self._layer_devices[0]
        if any(d != first for d in self._layer_devices):
            return False
        if device is None:
            return True
        return _normalise_device(device) == first

    def __repr__(self) -> str:
        unique = sorted(set(self._layer_devices))
        return (f"PerBlockRouter(num_blocks={self.num_blocks}, "
                f"devices={unique}, "
                f"cuda_available={CUPY_AVAILABLE})")


# ---------------------------------------------------------------------------
# Convenience: forward driver
# ---------------------------------------------------------------------------

def run_routed_forward(
    router: PerBlockRouter,
    block_fwds: List[Callable[[Any], Any]],
    x: np.ndarray,
    src_name: str = "embed",
    dst_name: Optional[str] = None,
) -> Any:
    """Drive a sequence of block forwards under a router.

    ``block_fwds[i]`` is the closure for block ``i``; it accepts the
    on-device input tensor and returns the on-device output.

    Returns the final output (still on the device of the last block, OR
    on the device of ``dst_name`` if specified).
    """
    if len(block_fwds) != router.num_blocks:
        raise ValueError(
            f"len(block_fwds)={len(block_fwds)} but router expects "
            f"{router.num_blocks} blocks")
    cur_name = src_name
    cur = x
    for i, fwd in enumerate(block_fwds):
        next_name = f"layers_{i}"
        cur = router.move(cur, cur_name, next_name)
        cur = fwd(cur)
        cur_name = next_name
    if dst_name is not None and dst_name != cur_name:
        cur = router.move(cur, cur_name, dst_name)
    return cur
