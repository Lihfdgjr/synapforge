"""sf.surrogate — autograd Functions for backprop through spike events.

Spikes are discrete: ``s = (v >= threshold)``. The Heaviside step has a
zero / undefined gradient, which kills SNN training under any standard
PyTorch op. The classical fix is the *surrogate gradient* (Neftci et al.
2019): keep the binary forward, but replace the backward with a smooth
approximation that lets gradient flow back through the spike event.

This module is the spike-autograd machinery for synapforge:

* Five built-in surrogates: ATan / Sigmoid / Triangular / FastSigmoid /
  SLAYER, each as a ``torch.autograd.Function``.
* A pluggable registry + ``register`` decorator for user-defined ones.
* The single public op ``spike(v, threshold, ...)`` that wraps them.
* ``PLIFCell`` — a Parametric LIF neuron (Fang et al. 2021) that ties
  ``spike()`` into a learnable cell with single-step + sequence interface.

All backwards compute the surrogate derivative in fp32 even when the
forward ran under bf16 / fp16 autocast, to avoid silent NaN / underflow
in the small-x regime where ``alpha * x`` blows up under low precision.

Public API
----------
    >>> from synapforge.surrogate import spike, PLIFCell, register
    >>> s = spike(v, threshold, surrogate="atan", alpha=2.0)

References
----------
* Neftci, Mostafa & Zenke (2019), Surrogate Gradient Learning in SNNs.
* Fang, Yu, Chen, Huang, Masquelier, Tian (2021), ICCV — Incorporating
  Learnable Membrane Time Constant (PLIF + ATan).
* Shrestha & Orchard (2018), SLAYER — Spike LAYer Error Reassignment.
* Zenke & Ganguli (2018), SuperSpike (fast-sigmoid surrogate).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

__all__ = [
    "ATanSurrogate",
    "SigmoidSurrogate",
    "TriangularSurrogate",
    "FastSigmoidSurrogate",
    "SLAYERSurrogate",
    "register",
    "spike",
    "PLIFCell",
    "list_surrogates",
]

_PI_HALF = math.pi / 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_fp32(x: torch.Tensor, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Return (x_fp32, grad_fp32, original_dtype) with autocast disabled.

    Mirrors the b200_gpu_prep ``install_fp32_surrogate`` pattern: the
    surrogate derivative is computed in fp32, then cast back to the input
    dtype on the way out. Without this, ``(alpha * x)**2`` overflows
    bf16's 7-bit mantissa for moderate ``alpha``, producing NaN gradients
    that silently corrupt the spike layer.
    """
    return x.float(), grad_output.float(), x.dtype


# ---------------------------------------------------------------------------
# Built-in surrogates
# ---------------------------------------------------------------------------

class _SurrogateBase(torch.autograd.Function):
    """Common forward: binary Heaviside on ``v - threshold``.

    Subclasses override only ``backward``. ``alpha`` is stashed on ``ctx``
    as a python float; it is NOT a tensor, so the matching backward
    return slot is ``None``.
    """

    @staticmethod
    def forward(ctx, v_minus_threshold: torch.Tensor, alpha: float):  # type: ignore[override]
        ctx.save_for_backward(v_minus_threshold)
        ctx.alpha = float(alpha)
        return (v_minus_threshold >= 0).to(v_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        raise NotImplementedError("subclass must implement backward")


class ATanSurrogate(_SurrogateBase):
    """ArcTangent surrogate (Fang et al. 2021).

    g(x)  = (1/pi) * atan(pi/2 * alpha * x) + 1/2
    g'(x) = alpha / (2 * (1 + (pi/2 * alpha * x)**2))

    Smooth, bounded peak gradient = ``alpha/2`` at x = 0. Decays as 1/x^2.
    Default workhorse — well-behaved under bf16 with the fp32 backward.
    """

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f, g_f, dtype = _stable_fp32(x, grad_output)
            denom = 1.0 + (_PI_HALF * alpha * x_f).pow(2)
            grad_x = g_f * (alpha / (2.0 * denom))
            grad_x = grad_x.to(dtype)
        return grad_x, None


class SigmoidSurrogate(_SurrogateBase):
    """Logistic-sigmoid surrogate.

    g(x)  = sigmoid(alpha * x)
    g'(x) = alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))

    Peak gradient = ``alpha/4`` at x = 0. Decays exponentially in |x|.
    More aggressive vanishing than ATan — use small alpha (~1.0) on
    deep stacks.
    """

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f, g_f, dtype = _stable_fp32(x, grad_output)
            # Stable sigmoid via clamp to keep exp() bounded under fp16-cast
            sig = torch.sigmoid(alpha * x_f)
            grad_x = g_f * alpha * sig * (1.0 - sig)
            grad_x = grad_x.to(dtype)
        return grad_x, None


class TriangularSurrogate(_SurrogateBase):
    """Piecewise-linear triangular surrogate (Esser et al. 2016).

    g'(x) = max(0, alpha * (1 - alpha * |x|))

    Compact support on |x| <= 1/alpha. Cheap, no transcendentals; ideal
    for event-driven backends. Zero gradient outside the support — can
    cause dead neurons if mem stays far from threshold.
    """

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f, g_f, dtype = _stable_fp32(x, grad_output)
            grad_x = g_f * torch.clamp(alpha * (1.0 - alpha * x_f.abs()), min=0.0)
            grad_x = grad_x.to(dtype)
        return grad_x, None


class FastSigmoidSurrogate(_SurrogateBase):
    """SuperSpike fast-sigmoid surrogate (Zenke & Ganguli 2018).

    g(x)  = x / (1 + alpha * |x|)            (rescaled)
    g'(x) = alpha / (1 + alpha * |x|)**2

    Peak gradient = ``alpha`` at x = 0. Heavier tails than ATan — keeps
    gradient alive farther from threshold, but the peak is twice as
    sharp. Good middle ground.
    """

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f, g_f, dtype = _stable_fp32(x, grad_output)
            grad_x = g_f * (alpha / (1.0 + alpha * x_f.abs()).pow(2))
            grad_x = grad_x.to(dtype)
        return grad_x, None


class SLAYERSurrogate(_SurrogateBase):
    """SLAYER exponential surrogate (Shrestha & Orchard 2018).

    g'(x) = alpha * exp(-alpha * |x|)

    Two-sided Laplacian. Peak gradient = ``alpha`` at x = 0. Decays
    exponentially in |x|, but slower than sigmoid and faster than
    fast-sigmoid. Original use: temporal-credit-assignment SNNs.
    """

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f, g_f, dtype = _stable_fp32(x, grad_output)
            # Clamp |x| to avoid huge negative exponent on bf16 cast-back
            grad_x = g_f * alpha * torch.exp(-alpha * x_f.abs().clamp(max=80.0))
            grad_x = grad_x.to(dtype)
        return grad_x, None


# ---------------------------------------------------------------------------
# Pluggable registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[_SurrogateBase]] = {
    "atan": ATanSurrogate,
    "sigmoid": SigmoidSurrogate,
    "triangle": TriangularSurrogate,
    "fast_sigmoid": FastSigmoidSurrogate,
    "slayer": SLAYERSurrogate,
}


def register(name: str) -> Callable[[type[_SurrogateBase]], type[_SurrogateBase]]:
    """Decorator: add a custom surrogate to the registry.

    >>> @register("my_ste")
    ... class MyShiftedSTE(_SurrogateBase):
    ...     @staticmethod
    ...     def backward(ctx, grad_output):
    ...         x, = ctx.saved_tensors
    ...         return grad_output * ((x.abs() < ctx.alpha).to(x.dtype)), None
    """
    def deco(cls: type[_SurrogateBase]) -> type[_SurrogateBase]:
        if not issubclass(cls, torch.autograd.Function):
            raise TypeError(f"{cls!r} must subclass torch.autograd.Function")
        _REGISTRY[name] = cls
        return cls
    return deco


def list_surrogates() -> list[str]:
    """Return registered surrogate names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Public spike op
# ---------------------------------------------------------------------------

def spike(
    v: torch.Tensor,
    threshold: torch.Tensor | float,
    *,
    surrogate: str | type[torch.autograd.Function] = "atan",
    alpha: float = 2.0,
) -> torch.Tensor:
    """Public spike op: ``v >= threshold`` with backward via surrogate.

    Forward is exact Heaviside (binary, dtype-preserving). Backward is
    the surrogate's smooth derivative — gradient flows through ``v`` and,
    when ``threshold`` is a tensor with ``requires_grad``, through it as
    well (autograd traces ``v - threshold`` automatically).

    Args:
        v: membrane voltage, any shape.
        threshold: scalar or per-channel tensor broadcastable to ``v``.
        surrogate: name from the registry, or a ``torch.autograd.Function``
            subclass implementing the ``_SurrogateBase`` contract.
        alpha: surrogate sharpness (peak-gradient scale at x = 0).

    Returns:
        Spike tensor with same shape & dtype as ``v``, values in {0, 1}.
    """
    if isinstance(surrogate, str):
        try:
            surrogate = _REGISTRY[surrogate]
        except KeyError as exc:
            raise KeyError(
                f"unknown surrogate {surrogate!r}; registered: {list_surrogates()}"
            ) from exc
    if not (isinstance(surrogate, type) and issubclass(surrogate, torch.autograd.Function)):
        raise TypeError(f"surrogate must be name or autograd.Function subclass, got {surrogate!r}")
    return surrogate.apply(v - threshold, alpha)


# ---------------------------------------------------------------------------
# PLIFCell — Parametric LIF integrating sf.surrogate.spike
# ---------------------------------------------------------------------------

class PLIFCell(nn.Module):
    """Parametric Leaky Integrate-and-Fire cell.

    Both the membrane time-constant ``tau`` and per-channel ``threshold``
    are learnable. Wraps :func:`spike` for the spike-emit step.

    Math (single step, dt = 1):
        decay = exp(-1 / exp(log_tau))
        v_t   = decay * v_{t-1} + (1 - decay) * x_t
        s_t   = spike(v_t, threshold; surrogate, alpha)
        v_t   = v_t - s_t * threshold        if reset == "subtract"
              = v_t * (1 - s_t)              if reset == "zero"

    Args:
        hidden: channel count C.
        tau_init: initial tau (steps); stored as log for positivity.
        threshold_init: initial firing threshold.
        surrogate: name in registry or autograd.Function subclass.
        alpha: surrogate sharpness.
        reset: "subtract" (soft) or "zero" (hard).
    """

    def __init__(
        self,
        hidden: int,
        tau_init: float = 10.0,
        threshold_init: float = 1.0,
        surrogate: str | type[torch.autograd.Function] = "atan",
        alpha: float = 2.0,
        reset: str = "subtract",
    ) -> None:
        super().__init__()
        if reset not in ("subtract", "zero"):
            raise ValueError(f"reset must be 'subtract' or 'zero', got {reset!r}")
        self.hidden = int(hidden)
        # log_tau is learnable; ensures tau > 0 via exp().
        self.log_tau = nn.Parameter(torch.full((hidden,), math.log(float(tau_init))))
        self.threshold = nn.Parameter(torch.full((hidden,), float(threshold_init)))
        self.surrogate = surrogate
        self.alpha = float(alpha)
        self.reset = reset

    # -- decays -----------------------------------------------------------

    def get_decay(self) -> torch.Tensor:
        """Return per-channel decay factor in (0, 1)."""
        return torch.exp(-1.0 / torch.exp(self.log_tau))

    # -- single step ------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        v_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single LIF step + spike emission.

        Args:
            x_t: input current, shape (B, hidden) (or any leading shape).
            v_prev: prior membrane voltage; zeros if None.

        Returns:
            (s_t, v_t): spike tensor and new voltage, same shape as x_t.
        """
        if v_prev is None:
            v_prev = torch.zeros_like(x_t)
        decay = self.get_decay()
        # Cast decay/threshold to input dtype for clean bf16/fp16 paths.
        decay_c = decay.to(x_t.dtype)
        thr_c = self.threshold.to(x_t.dtype)
        v_t = decay_c * v_prev + (1.0 - decay_c) * x_t
        s_t = spike(v_t, thr_c, surrogate=self.surrogate, alpha=self.alpha)
        if self.reset == "subtract":
            v_t = v_t - s_t * thr_c
        else:  # "zero"
            v_t = v_t * (1.0 - s_t)
        return s_t, v_t

    # -- vectorised over time --------------------------------------------

    def forward_seq(
        self,
        x_seq: torch.Tensor,
        v0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run T LIF steps.

        Args:
            x_seq: (B, T, hidden) input current sequence.
            v0: (B, hidden) initial voltage; zeros if None.

        Returns:
            (s_seq, v_final): (B, T, hidden) spike trace and final voltage.
        """
        if x_seq.dim() != 3:
            raise ValueError(f"x_seq must be (B, T, hidden); got shape {tuple(x_seq.shape)}")
        B, T, H = x_seq.shape
        if H != self.hidden:
            raise ValueError(f"x_seq hidden {H} != cell hidden {self.hidden}")
        v = v0 if v0 is not None else torch.zeros(B, H, dtype=x_seq.dtype, device=x_seq.device)
        spikes = []
        for t in range(T):
            s, v = self.forward(x_seq[:, t], v)
            spikes.append(s)
        return torch.stack(spikes, dim=1), v

    def extra_repr(self) -> str:
        sr = self.surrogate if isinstance(self.surrogate, str) else self.surrogate.__name__
        return (
            f"hidden={self.hidden}, surrogate={sr}, alpha={self.alpha}, reset={self.reset}"
        )
