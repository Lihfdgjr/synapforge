"""WaveFormer1D — wave-PDE depth mixer along the token axis (FFT spectral).

Adaptation of arxiv:2601.08602 (Shu et al. 2026) to a 1-D token sequence:

    H(omega; t) = exp(-alpha * t / 2) * cos(omega_d * t)
    omega_d     = sqrt( (v * omega)^2 - (alpha/2)^2 ).clamp_min(eps)

with ``t`` an externally-set RDT loop step (``set_step``), ``omega``
the per-bin angular frequency from ``rfft``, and ``v`` / ``alpha``
per-channel learnable parameters.  Total ~3D learnable scalars.

Why a damped wave?
------------------
Strong inductive bias: real depth-wise mixing in a recurrent depth
transformer has *some* characteristic propagation speed and *some*
damping rate.  Encoding both as the closed-form Fourier filter avoids
fitting them in the data while preserving most of the expressivity of
a free per-bin filter (Hyena1D, also included).

Two baselines:
  - ``Hyena1D`` — battle-tested long-conv via small MLP-parameterised
                  filter (arxiv:2302.10866).
  - ``FNet1D``  — fixed FFT mixing, zero learnable spectral params
                  (arxiv:2105.03824).

CPU-friendly: ``torch.fft.rfft`` is MKL-backed on Xeon.  bf16 OK for
the input; complex math is performed in float (FFT casts internally).

Composition with sf.HybridBlock
-------------------------------
``attach_wave_mixer_to_block(block, hidden, kind='wave')`` monkey-
patches the block to add ``y = block_out + mixer(block_out)`` post-
residual, similar to the existing MoE attach pattern.  The mixer reads
``block._coe_step_t`` (set by external RDT loop) for depth-aware
filters; ``Hyena1D`` and ``FNet1D`` ignore the step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .module import Module


# ---------------------------------------------------------------------------
# 1. WaveFormer1D — wave-PDE closed-form Fourier filter
# ---------------------------------------------------------------------------


class WaveFormer1D(Module):
    """Damped wave-equation depth mixer in Fourier domain.

    Per-frequency filter::

        H(omega; t) = exp(-alpha * t / 2) * cos(omega_d * t)
        omega_d = sqrt( (v * omega)^2 - (alpha/2)^2 ).clamp_min(eps)

    where ``t = self._step + 1`` is the RDT loop step (external) and
    ``omega`` is the per-bin frequency from ``rfft`` of the token axis.

    Trainable parameters per layer:
        - ``log_v``     (D,)  velocity init exp(0) = 1.0
        - ``log_alpha`` (D,)  damping init exp(-2.3) ~ 0.1
        - ``amp_gain``  (D,)  per-channel output gain init 1.0
    """

    def __init__(self, hidden: int, max_steps: int = 4, eps: float = 1e-5):
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.hidden = int(hidden)
        self.max_steps = int(max_steps)
        self.eps = float(eps)
        self.log_v = nn.Parameter(torch.zeros(hidden))
        self.log_alpha = nn.Parameter(torch.full((hidden,), -2.3))
        self.amp_gain = nn.Parameter(torch.ones(hidden))
        self._step: int = 0

    def set_step(self, t: int) -> None:
        """Set the current RDT loop step.  Clamped to ``[0, max_steps-1]``."""
        self._step = max(0, min(int(t), self.max_steps - 1))

    def forward(self, x: torch.Tensor, step_t: int | None = None) -> torch.Tensor:
        """x: [B, T, D]  ->  y: [B, T, D]."""
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, D], got {tuple(x.shape)}")
        if x.size(-1) != self.hidden:
            raise ValueError(
                f"last dim {x.size(-1)} != hidden {self.hidden}"
            )
        if step_t is not None:
            self.set_step(step_t)
        t = float(self._step + 1)
        B, T, D = x.shape

        # rFFT over the T axis; complex result has shape (B, T//2+1, D).
        # bf16 is upcast inside the FFT call; cast back at end.
        in_dtype = x.dtype
        Xf = torch.fft.rfft(x.to(torch.float32), dim=1)
        n_bins = Xf.shape[1]

        # Per-bin angular frequency in [0, pi].
        omega = torch.linspace(0.0, math.pi, n_bins, device=x.device, dtype=torch.float32)
        omega = omega.view(1, n_bins, 1)

        v = self.log_v.exp().to(torch.float32).view(1, 1, D)
        alpha = self.log_alpha.exp().to(torch.float32).view(1, 1, D)

        v_omega_sq = (v * omega) ** 2
        damp_sq = (alpha / 2.0) ** 2
        omega_d = torch.sqrt((v_omega_sq - damp_sq).clamp_min(self.eps))

        decay = torch.exp(-alpha * t / 2.0)
        modulation = torch.cos(omega_d * t)
        H = decay * modulation                       # (1, F, D), real

        Yf = Xf * H.to(Xf.dtype)
        y = torch.fft.irfft(Yf, n=T, dim=1)
        y = y.to(in_dtype) * self.amp_gain.view(1, 1, D)
        return y


# ---------------------------------------------------------------------------
# 2. Hyena1D — long-conv with learned per-bin filter (FFT-evaluated)
# ---------------------------------------------------------------------------


class _HyenaFilterMLP(nn.Module):
    """Tiny MLP mapping (positional embedding) -> per-bin filter."""

    def __init__(self, hidden: int, pos_dim: int = 16, mlp_hidden: int = 32):
        super().__init__()
        self.pos_dim = int(pos_dim)
        self.net = nn.Sequential(
            nn.Linear(pos_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden),
        )

    def forward(self, pos_emb: torch.Tensor) -> torch.Tensor:
        return self.net(pos_emb)


class Hyena1D(Module):
    """Long-conv with per-bin filter generated by a tiny MLP. arxiv:2302.10866."""

    def __init__(
        self,
        hidden: int,
        pos_dim: int = 16,
        mlp_hidden: int = 32,
        max_seq: int = 4096,
    ):
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        self.hidden = int(hidden)
        self.pos_dim = int(pos_dim)
        self.max_seq = int(max_seq)

        n_bins = max_seq // 2 + 1
        pos = torch.arange(n_bins, dtype=torch.float32).unsqueeze(1)
        inv_freq = torch.exp(
            torch.arange(0, pos_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / pos_dim)
        )
        pe = torch.zeros(n_bins, pos_dim)
        pe[:, 0::2] = torch.sin(pos * inv_freq)
        pe[:, 1::2] = torch.cos(pos * inv_freq[: pe[:, 1::2].shape[1]])
        self.register_buffer("pos_emb", pe)

        self.filter_mlp = _HyenaFilterMLP(hidden, pos_dim=pos_dim, mlp_hidden=mlp_hidden)
        self.amp_gain = nn.Parameter(torch.ones(hidden))

    def forward(self, x: torch.Tensor, step_t: int | None = None) -> torch.Tensor:
        del step_t  # depth-agnostic
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, D], got {tuple(x.shape)}")
        in_dtype = x.dtype
        B, T, D = x.shape
        n_bins = T // 2 + 1
        Xf = torch.fft.rfft(x.to(torch.float32), dim=1)
        if n_bins <= self.pos_emb.shape[0]:
            pe = self.pos_emb[:n_bins]
        else:
            pos = torch.arange(n_bins, dtype=torch.float32, device=x.device).unsqueeze(1)
            inv_freq = torch.exp(
                torch.arange(0, self.pos_dim, 2, dtype=torch.float32, device=x.device)
                * (-math.log(10000.0) / self.pos_dim)
            )
            pe = torch.zeros(n_bins, self.pos_dim, device=x.device)
            pe[:, 0::2] = torch.sin(pos * inv_freq)
            pe[:, 1::2] = torch.cos(pos * inv_freq[: pe[:, 1::2].shape[1]])
        H = self.filter_mlp(pe).to(torch.float32)
        Yf = Xf * H.unsqueeze(0)
        y = torch.fft.irfft(Yf, n=T, dim=1)
        return y.to(in_dtype) * self.amp_gain.view(1, 1, D)


# ---------------------------------------------------------------------------
# 3. FNet1D — fixed FFT mixing, no learnable spectral params
# ---------------------------------------------------------------------------


class FNet1D(Module):
    """Fixed FFT mixing (real part of token-axis FFT). arxiv:2105.03824."""

    def __init__(self, hidden: int):
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        self.hidden = int(hidden)
        self.amp_gain = nn.Parameter(torch.ones(hidden))

    def forward(self, x: torch.Tensor, step_t: int | None = None) -> torch.Tensor:
        del step_t
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, D], got {tuple(x.shape)}")
        in_dtype = x.dtype
        Y1 = torch.fft.fft(x.to(torch.float32), dim=2)
        Y2 = torch.fft.fft(Y1, dim=1)
        return Y2.real.to(in_dtype) * self.amp_gain.view(1, 1, -1)


# ---------------------------------------------------------------------------
# Attach helper
# ---------------------------------------------------------------------------


def _build_mixer(kind: str, hidden: int, max_steps: int = 4) -> nn.Module:
    k = kind.lower()
    if k == "wave":
        return WaveFormer1D(hidden, max_steps=max_steps)
    if k == "hyena":
        return Hyena1D(hidden)
    if k == "fnet":
        return FNet1D(hidden)
    raise ValueError(f"Unknown wave mixer kind: {kind}")


def attach_wave_mixer_to_block(
    block: nn.Module,
    hidden: int,
    kind: str = "wave",
    max_steps: int = 4,
) -> nn.Module:
    """Add a sequence-axis mixer post-residual to a block.

    The patched ``forward`` returns ``(combined + mixer(combined), ...)``.
    Reads ``block._coe_step_t`` (set by an external RDT loop) for
    depth-aware mixers (only ``WaveFormer1D`` uses it).

    Idempotent: re-attaching is a no-op.
    """
    if getattr(block, "_wave_attached", False):
        return block
    mixer = _build_mixer(kind, hidden, max_steps=max_steps)
    block.add_module("wave_mixer", mixer)
    orig_forward = block.forward

    def patched(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if isinstance(out, tuple) and len(out) >= 1 and torch.is_tensor(out[0]):
            combined = out[0]
            step_t = int(getattr(block, "_coe_step_t", 0))
            mixed = block.wave_mixer(combined, step_t=step_t)
            return (combined + mixed,) + tuple(out[1:])
        if torch.is_tensor(out):
            step_t = int(getattr(block, "_coe_step_t", 0))
            return out + block.wave_mixer(out, step_t=step_t)
        return out

    block.forward = patched  # type: ignore[assignment]
    block._wave_attached = True
    block._wave_kind = kind
    return block


__all__ = [
    "WaveFormer1D",
    "Hyena1D",
    "FNet1D",
    "attach_wave_mixer_to_block",
]
