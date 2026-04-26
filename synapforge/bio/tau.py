"""TauSplit + MultiBandTau — shared-leak time-constant parameterisation.

Why split?
----------
PLIF and CfC both want a learnable leak ``tau``, but if they share the
same parameter their gradient magnitudes collide (one is the spike
threshold's denominator, the other is the ODE step).  ``TauSplit`` lets
them share a *base* tensor and apply per-stream offsets, decoupling the
gradient contributions while keeping global tau drift coherent.

MultiBand?
----------
Cortical oscillations cluster into discrete bands (theta 4-8 Hz, alpha
8-12 Hz, beta 13-30 Hz, gamma 30-100 Hz).  ``MultiBandTau`` keeps one
tau parameter per band and a learnable router that decides which band
each token uses.  At inference the router argmax-selects; at training
it produces a soft mixture so all bands receive gradient.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module


class TauSplit(Module):
    """Shared base ``alpha`` with per-stream PLIF/CfC offsets.

    ``alpha`` is a single learnable tensor of shape ``[hidden_size]``.
    The two methods ``tau_plif`` and ``tau_cfc`` apply ``softplus(alpha)``
    plus their own scalar offset, so the *bias* between PLIF and CfC
    leak is fixed at init but the shared shape (which units leak fast,
    which leak slow) is learned jointly.

    Args:
        hidden_size:  feature dimension.
        plif_offset:  added to softplus(alpha) for the PLIF branch.
        cfc_offset:   added to softplus(alpha) for the CfC branch.
    """

    def __init__(
        self,
        hidden_size: int,
        plif_offset: float = 0.8,
        cfc_offset: float = 0.5,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = int(hidden_size)
        self.alpha = nn.Parameter(torch.zeros(hidden_size))
        self.plif_offset = float(plif_offset)
        self.cfc_offset = float(cfc_offset)

    def tau_plif(self) -> torch.Tensor:
        return F.softplus(self.alpha) + self.plif_offset

    def tau_cfc(self) -> torch.Tensor:
        return F.softplus(self.alpha) + self.cfc_offset

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: return ``(tau_plif, tau_cfc)`` together."""
        base = F.softplus(self.alpha)
        return base + self.plif_offset, base + self.cfc_offset

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"plif_offset={self.plif_offset}, cfc_offset={self.cfc_offset}"
        )


class MultiBandTau(Module):
    """Multi-frequency band split: theta / alpha / beta / gamma per token.

    Maintains one ``log_tau`` parameter per band (4 by default).  A
    learnable router projects the hidden state onto the bands; softmax
    over bands gives the mixture weight.  The output ``tau`` is the
    weighted sum of band-specific taus, clamped into a safe range.

    Args:
        hidden_size:  feature dimension.
        bands:        tuple of band names; defaults to 4-band cortical
                      decomposition.

    Forward:
        hidden:  [..., D]
        returns: (tau [..., D],  routing [..., n_bands])
    """

    BAND_INITS = {
        "theta": 50.0,   # ~6 Hz at 1ms steps
        "alpha": 25.0,   # ~10 Hz
        "beta":  12.5,   # ~20 Hz
        "gamma": 5.0,    # ~50 Hz
    }
    TAU_MIN = 0.05
    TAU_MAX = 200.0

    def __init__(
        self,
        hidden_size: int,
        bands: tuple[str, ...] = ("theta", "alpha", "beta", "gamma"),
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = int(hidden_size)
        self.bands = tuple(bands)
        # Validate all bands have init values
        unknown = [b for b in self.bands if b not in self.BAND_INITS]
        if unknown:
            raise ValueError(
                f"unknown bands {unknown}; valid: {list(self.BAND_INITS)}"
            )
        self.tau_logs = nn.ParameterDict({
            b: nn.Parameter(torch.full((hidden_size,), math.log(self.BAND_INITS[b])))
            for b in self.bands
        })
        self.router = nn.Linear(hidden_size, len(self.bands), bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden: [..., D].  Returns (tau [..., D], routing [..., nB])."""
        if hidden.size(-1) != self.hidden_size:
            raise ValueError(
                f"last dim {hidden.size(-1)} != hidden_size {self.hidden_size}"
            )
        logits = self.router(hidden)
        weights = F.softmax(logits, dim=-1)
        # Stack tau parameters into [nB, D] tensor.
        taus = torch.stack(
            [self.tau_logs[b].exp().clamp(self.TAU_MIN, self.TAU_MAX) for b in self.bands],
            dim=0,
        )
        tau = torch.einsum("...b,bd->...d", weights, taus)
        return tau, weights

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, bands={self.bands}"


__all__ = ["TauSplit", "MultiBandTau"]
