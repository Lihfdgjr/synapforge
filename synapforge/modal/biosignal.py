"""BioSignalEmbed -- EEG / EMG / ECG embedding via FFT-bands + 1D conv.

Input
-----
signal: (B, T_samples, C) float -- C electrodes/channels at sample_rate Hz.

Pipeline
--------
1. Window the signal into overlapping windows of `win_samples` samples
   with hop `hop_samples`.
2. Per-channel FFT magnitude per window. Bin into 5 standard EEG bands:
     delta (0.5-4 Hz), theta (4-8), alpha (8-13), beta (13-30), gamma (30-100).
   Sum (or mean) magnitudes per band -> (B, T_win, C, 5) feature.
3. Concat band-energy with raw-window mean/std stats -> (B, T_win, C, 7).
4. Per-channel learnable embed: 7 -> per_channel_dim, then concat across C.
5. Linear -> hidden.
6. Add 1D positional encoding.
7. Prepend learned <|bio|> marker.

Returns
-------
(B, 1 + T_win, hidden).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module
from .audio import sinusoidal_1d

# Standard EEG bands in Hz.
_EEG_BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 100.0),
)
_N_BANDS = len(_EEG_BANDS)
_N_STATS = 2  # mean, std
_FEATS_PER_CHAN = _N_BANDS + _N_STATS  # 7


class BioSignalEmbed(Module):
    """Biosignal (EEG/EMG/ECG) embedding via spectral bands + 1D conv mix.

    Forward
    -------
    signal: (B, T_samples, C) float.
    Returns (B, 1 + T_win, hidden) with T_win = (T_samples - win) // hop + 1.
    """

    def __init__(
        self,
        hidden: int = 512,
        sample_rate: int = 256,
        win_ms: int = 250,
        hop_ms: int = 125,
        max_channels: int = 64,
        per_channel_dim: int = 8,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.sample_rate = int(sample_rate)
        self.win = max(8, int(round(sample_rate * win_ms / 1000)))
        self.hop = max(1, int(round(sample_rate * hop_ms / 1000)))
        self.max_channels = int(max_channels)
        self.per_channel_dim = int(per_channel_dim)
        # Per-channel 7 -> per_channel_dim linear, then concat -> mc*pc -> hidden.
        self.chan_w = nn.Parameter(torch.zeros(max_channels, _FEATS_PER_CHAN, per_channel_dim))
        self.chan_b = nn.Parameter(torch.zeros(max_channels, per_channel_dim))
        nn.init.normal_(self.chan_w, std=0.02)
        nn.init.normal_(self.chan_b, std=0.02)
        # Mix across channels.
        self.mix = nn.Linear(max_channels * per_channel_dim, hidden, bias=False)
        nn.init.normal_(self.mix.weight, std=0.02)
        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)
        # Pre-compute band bin indices once at runtime per-call (cheap).
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    def _pos(self, T_win: int, device, dtype) -> torch.Tensor:
        key = (T_win, str(dtype))
        if key not in self._pos_cache:
            self._pos_cache[key] = sinusoidal_1d(T_win, self.hidden, device, dtype)
        return self._pos_cache[key]

    def _windowize(self, signal: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T_win, C, win)."""
        B, T, C = signal.shape
        win, hop = self.win, self.hop
        if T < win:
            # Pad up to one full window so we always emit at least 1 token.
            pad = win - T
            signal = F.pad(signal, (0, 0, 0, pad))
            T = win
        # Use unfold along time dim (dim=1).
        # Result shape: (B, T_win, C, win) via permute.
        # Note: unfold returns (B, T_win, C, win) directly when applied on dim=1.
        x = signal.unfold(1, win, hop)              # (B, T_win, C, win)
        return x

    def _spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_win, C, win) -> (B, T_win, C, _FEATS_PER_CHAN)."""
        B, Tw, C, win = x.shape
        # FFT along last dim.
        fft = torch.fft.rfft(x.float(), dim=-1)
        mag = fft.abs()                              # (B, Tw, C, win//2+1)
        n_bins = mag.shape[-1]
        # Frequencies of each bin.
        freqs = torch.fft.rfftfreq(win, d=1.0 / self.sample_rate).to(mag.device)
        # Band energies.
        band_feats = torch.zeros(
            B, Tw, C, _N_BANDS, device=x.device, dtype=mag.dtype
        )
        for i, (_name, lo, hi) in enumerate(_EEG_BANDS):
            sel = (freqs >= lo) & (freqs < hi)
            if sel.any():
                band_feats[..., i] = mag[..., sel].mean(dim=-1)
            else:
                band_feats[..., i] = 0.0
        # Raw-domain stats.
        mean = x.float().mean(dim=-1, keepdim=False).unsqueeze(-1)
        std = x.float().std(dim=-1, keepdim=False).unsqueeze(-1)
        feats = torch.cat([band_feats, mean, std], dim=-1)  # (B, Tw, C, 7)
        return feats

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.dim() != 3:
            raise ValueError(
                f"BioSignalEmbed expects (B,T,C); got {tuple(signal.shape)}"
            )
        B, T, C = signal.shape
        if C > self.max_channels:
            raise ValueError(
                f"channels {C} > max_channels {self.max_channels}"
            )
        # Pad-channels for batched einsum.
        if C < self.max_channels:
            pad = torch.zeros(
                B, T, self.max_channels - C,
                device=signal.device, dtype=signal.dtype,
            )
            signal_p = torch.cat([signal, pad], dim=-1)
        else:
            signal_p = signal
        windows = self._windowize(signal_p)          # (B, Tw, mc, win)
        feats = self._spectral_features(windows)     # (B, Tw, mc, 7)
        # Per-channel embed: einsum('btcf,cfp->btcp')
        feats_t = feats.to(self.chan_w.dtype)
        emb = torch.einsum("btcf,cfp->btcp", feats_t, self.chan_w)  # (B,Tw,mc,pc)
        emb = emb + self.chan_b.view(1, 1, self.max_channels, self.per_channel_dim)
        # Mask out unused channels.
        if C < self.max_channels:
            mask = torch.zeros(self.max_channels, device=emb.device, dtype=emb.dtype)
            mask[:C] = 1.0
            emb = emb * mask.view(1, 1, -1, 1)
        # Concat channel dim.
        Bw, Tw, _, _ = emb.shape
        flat = emb.reshape(Bw, Tw, self.max_channels * self.per_channel_dim)
        z = self.mix(flat)                           # (B, Tw, hidden)
        pos = self._pos(Tw, signal.device, z.dtype)
        z = z + pos.unsqueeze(0)
        marker = self.marker.to(z.dtype).expand(B, 1, self.hidden)
        z = torch.cat([marker, z], dim=1)
        return z

    @staticmethod
    def expected_token_count(T_samples: int, sample_rate: int = 256,
                             win_ms: int = 250, hop_ms: int = 125) -> int:
        win = max(8, int(round(sample_rate * win_ms / 1000)))
        hop = max(1, int(round(sample_rate * hop_ms / 1000)))
        if T_samples < win:
            return 1 + 1  # padded to one window
        return 1 + (T_samples - win) // hop + 1
