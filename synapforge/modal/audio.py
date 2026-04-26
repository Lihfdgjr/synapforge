"""AudioPatchEmbed -- 20ms-chunk audio patch embedding.

Two modes:
    mode="raw"  : 1D convolution over raw waveform (kernel = chunk_size, stride = chunk_size).
    mode="mel"  : torchaudio.MelSpectrogram + linear projection over mel-frame chunks.
                  Auto-falls-back to "raw" when torchaudio is missing.

Both produce (B, T, hidden) with T proportional to audio duration.
A learned <|audio|> marker is prepended.

Default geometry
----------------
- sample_rate=16000, chunk_ms=20 -> 320 samples / chunk
- mode="raw":  T = ceil(samples / chunk_size)
- mode="mel":  n_mels=80, hop=10ms (1600 frames per 16s of audio); we then group
                hop frames per chunk -> T = ceil(n_frames / hop_per_chunk)
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Module
from .image import sinusoidal_2d  # reused for 1D pos enc (rows=1, cols=T)

try:
    import torchaudio  # type: ignore
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False


def sinusoidal_1d(seq: int, dim: int, device, dtype) -> torch.Tensor:
    """1D sinusoidal pos encoding -> (seq, dim)."""
    if dim % 2 != 0:
        dim_eff = dim - 1
    else:
        dim_eff = dim
    pos = torch.arange(seq, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim_eff, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / dim_eff)
    )
    out = torch.zeros(seq, dim, device=device, dtype=torch.float32)
    out[:, 0:dim_eff:2] = torch.sin(pos * div)
    out[:, 1:dim_eff:2] = torch.cos(pos * div)
    return out.to(dtype)


class AudioPatchEmbed(Module):
    """20ms-chunk audio patch embedding (raw or mel-based).

    Forward
    -------
    waveform: (B, samples) float in [-1, 1] mono OR (B, 1, samples).
    Returns (B, 1 + T, hidden) where T depends on mode and duration.
    """

    def __init__(
        self,
        hidden: int = 512,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
        mode: Literal["raw", "mel"] = "raw",
        n_mels: int = 80,
        win_ms: int = 25,
        hop_ms: int = 10,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.sample_rate = int(sample_rate)
        self.chunk_ms = int(chunk_ms)
        self.chunk_size = int(round(sample_rate * chunk_ms / 1000))
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if mode == "mel" and not _HAS_TORCHAUDIO:
            mode = "raw"
        self.mode = mode

        if mode == "raw":
            # 1D conv with kernel = stride = chunk_size : equivalent to
            # patchify on time + linear projection.
            self.proj = nn.Conv1d(
                in_channels=1,
                out_channels=hidden,
                kernel_size=self.chunk_size,
                stride=self.chunk_size,
                bias=False,
            )
            nn.init.normal_(self.proj.weight, std=0.02)
            self.mel = None
            self.mel_proj = None
        else:  # mode == "mel"
            n_fft = max(512, int(sample_rate * win_ms / 1000) * 2)
            win_length = int(sample_rate * win_ms / 1000)
            hop_length = int(sample_rate * hop_ms / 1000)
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
                normalized=False,
            )
            self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
            self.hop_length = hop_length
            # Group `frames_per_chunk` mel frames into one chunk patch.
            self.frames_per_chunk = max(1, chunk_ms // hop_ms)
            self.mel_proj = nn.Linear(n_mels * self.frames_per_chunk, hidden, bias=False)
            nn.init.normal_(self.mel_proj.weight, std=0.02)
            self.proj = None

        self.marker = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.marker, std=0.02)
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    def _pos(self, T: int, device, dtype) -> torch.Tensor:
        key = (T, str(dtype))
        if key not in self._pos_cache:
            self._pos_cache[key] = sinusoidal_1d(T, self.hidden, device, dtype)
        return self._pos_cache[key]

    def _normalize_input(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        if waveform.dim() != 2:
            raise ValueError(
                f"AudioPatchEmbed expects (B, samples) or (B,1,samples); got {tuple(waveform.shape)}"
            )
        return waveform

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self._normalize_input(waveform)
        B, N = waveform.shape

        if self.mode == "raw":
            # Pad to multiple of chunk_size on the right.
            pad = (-N) % self.chunk_size
            if pad:
                waveform = F.pad(waveform, (0, pad))
            x = waveform.unsqueeze(1).to(self.proj.weight.dtype)   # (B, 1, N')
            z = self.proj(x)                                       # (B, hidden, T)
            z = z.transpose(1, 2).contiguous()                     # (B, T, hidden)
        else:
            mel_spec = self.mel(waveform)                          # (B, n_mels, F)
            mel_db = self.amp_to_db(mel_spec)
            # Group F frames into chunks of frames_per_chunk.
            B_, M, Fr = mel_db.shape
            pad_f = (-Fr) % self.frames_per_chunk
            if pad_f:
                mel_db = F.pad(mel_db, (0, pad_f))
                Fr = mel_db.shape[-1]
            T = Fr // self.frames_per_chunk
            mel_db = mel_db.reshape(B_, M, T, self.frames_per_chunk)
            # (B, T, M*frames_per_chunk)
            feat = mel_db.permute(0, 2, 1, 3).reshape(B_, T, M * self.frames_per_chunk)
            feat = feat.to(self.mel_proj.weight.dtype)
            z = self.mel_proj(feat)                                # (B, T, hidden)

        T = z.shape[1]
        pos = self._pos(T, z.device, z.dtype)                       # (T, hidden)
        z = z + pos.unsqueeze(0)
        marker = self.marker.to(z.dtype).expand(B, 1, self.hidden)
        z = torch.cat([marker, z], dim=1)                          # (B, 1+T, hidden)
        return z
