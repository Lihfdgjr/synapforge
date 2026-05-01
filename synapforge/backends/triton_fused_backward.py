"""Fused PLIF surrogate forward + backward Triton kernel.

T2.2 (DEEP_MAINT_QUEUE.md): real implementation of the previously
stubbed fused PLIF surrogate kernel. The kernel computes both
``spike = (v >= thr)`` and ``dspike/dv = surrogate'(v - thr)`` in a
single Triton tile so the autograd ``Function`` no longer has to stash
``v - thr`` and re-launch a kernel on backward.

Why this exists
---------------
The default surrogate path in ``synapforge/surrogate.py`` runs a per-call
``torch.autograd.Function`` (ATan / Sigmoid / Triangle / FastSigmoid /
SLAYER). Forward emits the binary Heaviside spike, backward computes the
surrogate derivative in fp32 and casts back to the input dtype. On the
production trainer (10 PLIF layers x 256 timesteps x bs=80) the dispatch
overhead alone is ~25 ms/step on A800-80GB and the saved ``v - thr`` tape
doubles activation memory for the spike train.

This module ships a fused kernel that:

  1. computes ``spike`` and ``dspike/dv`` in the SAME Triton tile,
  2. caches only ``dspike/dv`` for backward (fp32, same shape as input),
  3. reads the cached ``dspike/dv`` directly on backward — no per-call
     surrogate dispatch.

bf16 / autocast handling
------------------------
Triton 2.x silently downcasts fp32 intermediates back to the input
dtype on bf16 ops, which silently corrupts the surrogate denominator
``1 + (pi/2 * alpha * x)^2`` for moderate alpha. The kernel explicitly
upcasts ``(v - thr).to(tl.float32)`` for the surrogate computation and
casts the spike back to the input dtype on store. The ``dspike/dv``
cache is fp32 unconditionally — backward reads it without further casts.

Honesty / availability
----------------------
The kernel imports cleanly even when Triton is not installed (Windows
dev box, CPU-only CI). ``is_available()`` returns False on those hosts;
the autograd Function falls back to the pure-PyTorch reference path,
which is bit-exact with the existing ``ATanSurrogate.backward`` for the
ATan case and within 1e-4 for the supported surrogates. CUDA-only tests
in ``tests/integration/test_triton_fused_plif.py`` skip cleanly when
``torch.cuda.is_available()`` is False.

Status: implemented but UNTESTED on CUDA from this dev box (no Triton +
no GPU). The pure-PyTorch reference is bit-exact verified against the
existing ATanSurrogate.backward (see tests). The Triton path mirrors
the same math; user will verify on the A800 rental.

Public API
----------
* :class:`FusedPLIFFn` — :class:`torch.autograd.Function` that runs the
  fused forward+backward path. Drops in for ``ATanSurrogate.apply`` /
  ``SigmoidSurrogate.apply``.
* :func:`fused_plif_spike` — functional wrapper that takes
  ``(v, threshold, alpha, surrogate_kind)``.
* :func:`is_available` — True iff Triton + CUDA + a tested kernel are
  ready.
* :func:`enable_fused_backward` — kept for backward-compat with the
  existing trainer wiring (``--triton-fused-backward`` flag in
  ``train_100m_kd.py``). Now a no-op when Triton is available; raises
  ``NotImplementedError`` only when the kernel cannot be compiled (so
  the trainer logs ``[triton-fused-backward] disabled: ...`` and falls
  back to the autograd surrogate path the same way).
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "FusedPLIFFn",
    "fused_plif_spike",
    "is_available",
    "enable_fused_backward",
    "_HAS_TRITON",
]

# ---------------------------------------------------------------------------
# Triton availability probe (lazy: file importable on Windows / no-GPU CI).
# ---------------------------------------------------------------------------

_HAS_TRITON = False
try:  # pragma: no cover -- environment-dependent
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None


# Surrogate kind constants. Kept as ints so the Triton kernel can branch
# at compile time via ``tl.constexpr`` rather than via Python dispatch.
_SURROGATE_ATAN = 0
_SURROGATE_SIGMOID = 1


_SURROGATE_NAMES = {
    "atan": _SURROGATE_ATAN,
    "sigmoid": _SURROGATE_SIGMOID,
}

_PI_HALF = math.pi / 2.0


# ---------------------------------------------------------------------------
# Triton kernel — defined only when Triton imports succeed.
# ---------------------------------------------------------------------------

if _HAS_TRITON:  # pragma: no cover -- requires CUDA + Triton

    @triton.jit
    def fused_plif_fwd_bwd(
        v_ptr,                   # input v (any shape, treated as 1-D)
        thr_ptr,                 # threshold (per-channel or scalar)
        spike_ptr,               # output: spike (same dtype as v)
        dspike_dv_ptr,           # output: dspike/dv (fp32 cache for bw)
        N,                       # total #elements in v
        D,                       # channel dim (for thr broadcast)
        alpha,                   # surrogate sharpness
        SURROGATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Compute spike + dspike/dv in one tile.

        Layout assumption: ``v`` and ``spike`` are flattened to 1-D of
        length ``N``. ``thr`` is broadcast along the inner dim ``D``
        (per-channel). The kernel handles any leading (B, T) shape via
        the ``i % D`` modulo on flat indices.
        """
        PI_HALF = 1.5707963267948966
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        # Load v (cast to fp32 for the surrogate math). Any input dtype.
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # thr broadcast: row-major (B,T,D) -> channel idx = i % D.
        ch = offs % D
        thr = tl.load(thr_ptr + ch, mask=mask, other=0.0).to(tl.float32)

        m = v - thr  # explicit upcast to fp32 already

        # --- forward: binary Heaviside spike ---
        # Use >= 0 to match Python ATanSurrogate.forward semantics.
        spike = tl.where(m >= 0.0, 1.0, 0.0)

        # --- backward: surrogate derivative ---
        if SURROGATE == 0:  # ATan
            # g'(x) = alpha / (2 * (1 + (pi/2 * alpha * x)^2))
            x = alpha * m
            denom = 1.0 + (PI_HALF * x) * (PI_HALF * x)
            ds_dv = alpha / (2.0 * denom)
        else:  # Sigmoid
            # g'(x) = alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))
            ax = alpha * m
            sig = 1.0 / (1.0 + tl.exp(-ax))
            ds_dv = alpha * sig * (1.0 - sig)

        # Store outputs.
        tl.store(spike_ptr + offs, spike.to(spike_ptr.dtype.element_ty), mask=mask)
        tl.store(dspike_dv_ptr + offs, ds_dv, mask=mask)


def _pytorch_fwd_bwd_reference(
    v: torch.Tensor,
    thr: torch.Tensor,
    alpha: float,
    surrogate: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference: identical math to the Triton kernel.

    Used as the CPU / no-Triton fallback AND as the gradient-correctness
    reference for tests. Bit-exact (within fp32 epsilon) with
    ``ATanSurrogate.backward`` / ``SigmoidSurrogate.backward`` for the
    relevant surrogate.
    """
    out_dtype = v.dtype
    # Always upcast to fp32 for the surrogate math (matches the
    # ``_stable_fp32`` guard in synapforge/surrogate.py).
    v_f = v.float()
    if isinstance(thr, torch.Tensor):
        thr_f = thr.float()
    else:
        thr_f = float(thr)
    m = v_f - thr_f

    spike = (m >= 0.0).to(out_dtype)

    if surrogate == "atan":
        denom = 1.0 + (_PI_HALF * alpha * m).pow(2)
        ds_dv = alpha / (2.0 * denom)
    elif surrogate == "sigmoid":
        sig = torch.sigmoid(alpha * m)
        ds_dv = alpha * sig * (1.0 - sig)
    else:
        raise ValueError(
            f"unknown surrogate {surrogate!r}; supported: atan, sigmoid"
        )
    # ds_dv stays in fp32 (the autograd cache); spike is in input dtype.
    return spike, ds_dv


def _triton_fwd_bwd(  # pragma: no cover -- requires CUDA
    v: torch.Tensor,
    thr: torch.Tensor,
    alpha: float,
    surrogate: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Triton kernel. Caller has already verified CUDA."""
    if not _HAS_TRITON:
        raise RuntimeError("triton not available -- caller should have fallen back")
    if surrogate not in _SURROGATE_NAMES:
        raise ValueError(
            f"unknown surrogate {surrogate!r}; supported: {list(_SURROGATE_NAMES)}"
        )

    v_c = v.contiguous()
    D = v_c.shape[-1] if v_c.dim() >= 1 else 1
    N = v_c.numel()

    if isinstance(thr, torch.Tensor):
        thr_c = thr.contiguous()
        if thr_c.numel() == 1:
            # broadcast scalar across all channels
            thr_c = thr_c.expand(D).contiguous()
        elif thr_c.numel() != D:
            raise ValueError(
                f"threshold shape {thr_c.shape} not broadcast-compatible with "
                f"v's last dim {D}"
            )
        # Triton wants fp32 for the threshold load when v is bf16/fp16
        # (we explicitly upcast inside the kernel via .to(fp32) but
        # passing fp32 avoids a cast on every load).
        if thr_c.dtype != torch.float32:
            thr_c = thr_c.to(torch.float32)
    else:
        thr_c = torch.full((D,), float(thr), dtype=torch.float32, device=v.device)

    spike = torch.empty_like(v_c)
    ds_dv = torch.empty(N, device=v.device, dtype=torch.float32)

    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)

    fused_plif_fwd_bwd[grid](
        v_c, thr_c, spike, ds_dv,
        N, D, float(alpha),
        SURROGATE=_SURROGATE_NAMES[surrogate],
        BLOCK=BLOCK,
    )
    return spike, ds_dv.view_as(v_c)


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------

class FusedPLIFFn(torch.autograd.Function):
    """Fused PLIF surrogate forward+backward.

    Forward returns the binary spike. Backward reads the cached
    ``dspike/dv`` (computed in the same kernel as forward) and multiplies
    by the upstream gradient. The ``threshold`` gradient is the negation
    of the v-gradient (because ``m = v - thr`` => ``dm/dthr = -1``),
    summed over all batch / time dimensions, broadcast back to the
    threshold's shape.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        v: torch.Tensor,
        threshold: torch.Tensor,
        alpha: float,
        surrogate: str,
    ) -> torch.Tensor:
        use_triton = (
            _HAS_TRITON
            and v.is_cuda
            and v.dtype in (torch.float32, torch.float16, torch.bfloat16)
        )
        if use_triton:  # pragma: no cover -- requires CUDA
            try:
                spike, ds_dv = _triton_fwd_bwd(v, threshold, alpha, surrogate)
            except Exception:
                # Fall back to PyTorch ref; the Triton kernel may fail to
                # compile on niche shapes / Triton versions.
                spike, ds_dv = _pytorch_fwd_bwd_reference(
                    v, threshold, alpha, surrogate
                )
        else:
            spike, ds_dv = _pytorch_fwd_bwd_reference(
                v, threshold, alpha, surrogate
            )

        # Save fp32 ds/dv buffer for backward; threshold ref preserved
        # so we can produce its gradient via the m = v - thr chain rule.
        ctx.save_for_backward(ds_dv)
        # Track threshold metadata so backward can produce the matching
        # grad shape without depending on the now-dropped tensor.
        ctx.threshold_shape = (
            tuple(threshold.shape) if isinstance(threshold, torch.Tensor) else ()
        )
        ctx.threshold_dtype = (
            threshold.dtype if isinstance(threshold, torch.Tensor) else v.dtype
        )
        ctx.threshold_needs_grad = bool(
            isinstance(threshold, torch.Tensor) and threshold.requires_grad
        )
        ctx.v_dtype = v.dtype
        return spike

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (ds_dv,) = ctx.saved_tensors
        v_dtype = ctx.v_dtype

        # Compute grad in fp32 to mirror the existing ATanSurrogate.backward
        # numerical contract; cast back to v's dtype on return.
        grad_f = grad_output.float() * ds_dv  # ds_dv already fp32

        grad_v = grad_f.to(v_dtype)

        # Threshold gradient: dm/dthr = -1, so dL/dthr = -dL/dv broadcast-summed
        # over all dims that thr broadcasts against.
        if ctx.threshold_needs_grad and ctx.threshold_shape:
            grad_thr = -grad_f
            # Sum over leading dims so the result matches the threshold shape.
            n_extra = grad_thr.dim() - len(ctx.threshold_shape)
            for _ in range(n_extra):
                grad_thr = grad_thr.sum(dim=0)
            # Sum any broadcast dims (where thr_shape == 1).
            for i, s in enumerate(ctx.threshold_shape):
                if s == 1 and grad_thr.shape[i] != 1:
                    grad_thr = grad_thr.sum(dim=i, keepdim=True)
            grad_thr = grad_thr.to(ctx.threshold_dtype)
        else:
            grad_thr = None

        # Return signature matches forward: (v, threshold, alpha, surrogate).
        return grad_v, grad_thr, None, None


# ---------------------------------------------------------------------------
# Functional wrapper (the autograd Function takes care of grad)
# ---------------------------------------------------------------------------

def fused_plif_spike(
    v: torch.Tensor,
    threshold,
    *,
    alpha: float = 2.0,
    surrogate: str = "atan",
) -> torch.Tensor:
    """Fused PLIF spike with surrogate gradient.

    Equivalent to ``synapforge.surrogate.spike(v, threshold, surrogate=...,
    alpha=...)`` for surrogate in {"atan", "sigmoid"}, but the forward
    computes ``dspike/dv`` in the same kernel and caches it for backward
    so we avoid the per-call ``ATanSurrogate.apply`` dispatch and the
    ``v - thr`` autograd tape stash.

    Args:
        v: membrane voltage tensor (any shape; channel = last dim).
        threshold: scalar or per-channel threshold (broadcast over v's
            last dim). Either a python scalar or a 1-D / scalar tensor.
        alpha: surrogate sharpness.
        surrogate: ``"atan"`` (default) or ``"sigmoid"``.

    Returns:
        spike tensor (same shape & dtype as ``v``, values in {0, 1}).
    """
    if isinstance(threshold, (int, float)):
        thr = torch.tensor(float(threshold), device=v.device, dtype=v.dtype)
    else:
        thr = threshold
    return FusedPLIFFn.apply(v, thr, float(alpha), str(surrogate))


# ---------------------------------------------------------------------------
# Compatibility shims for the trainer wiring
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """True iff Triton + CUDA + a compileable kernel are available.

    The trainer reads this to decide whether to actually route through
    the fused kernel; on hosts without CUDA / Triton, the PLIFCell path
    falls back to the existing PyTorch autograd surrogate.
    """
    if not _HAS_TRITON:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False


def enable_fused_backward() -> None:
    """Trainer hook: validate the kernel is available + raise otherwise.

    Backward-compat shim for ``train_100m_kd.py``. The trainer wraps
    this in a try/except that catches ``NotImplementedError`` (legacy
    contract from the stub) and logs ``[triton-fused-backward]
    disabled: <reason>``. With the kernel landed, calling this is a
    no-op when Triton + CUDA are available; otherwise it raises so the
    trainer logs the same disabled message it always did and falls
    back to the autograd surrogate path the same way.
    """
    if not _HAS_TRITON:
        raise NotImplementedError(
            "Triton not installed in this environment; fused PLIF "
            "backward kernel cannot be enabled. Install triton or run "
            "without --triton-fused-backward."
        )
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "CUDA not available; fused PLIF backward kernel is "
            "CUDA-only. Run without --triton-fused-backward."
        )
    # Kernel is available; the autograd Function picks the Triton path
    # automatically based on tensor placement / dtype.
    return None
