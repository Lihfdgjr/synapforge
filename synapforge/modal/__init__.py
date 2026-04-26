"""synapforge.modal -- first-class multimodal embedding for the LNN+SNN backbone.

Fuyu-style early fusion: every modality is patched then linearly projected into
the SAME hidden dimension. Modality-boundary tokens are prepended so the
temporal LNN+SNN block (HybridBlock) learns mode transitions itself.

Round 6A: text / image / audio / video.
Round 6B (this round): screen / point_cloud / time_series / graph / biosignal.

Public API
----------
    >>> import synapforge as sf
    >>> import synapforge.modal as sm
    >>> embed = sm.UnifiedEmbed(hidden=512)
    >>> z = embed({
    ...     "text_tokens": ids,
    ...     "image": img,
    ...     "audio": wav,
    ...     "video": clip,
    ...     "screen": screenshot,
    ...     "point_cloud": cloud,
    ...     "time_series": ts,
    ...     "graph": {"nodes": ..., "edges": ...},
    ...     "biosignal": eeg,
    ... })
    >>> # z: (B, T_unified, 512) ready for sf.HybridBlock(512)

Submodules
----------
    image       -- ImagePatchEmbed     (B, 3, H, W) -> (B, T_img, hidden)
    audio       -- AudioPatchEmbed     (B, samples) -> (B, T_aud, hidden)
    video       -- VideoPatchEmbed     (B, T_f, 3, H, W) -> (B, T_vid, hidden)
    screen      -- ScreenPatchEmbed    (B, 3, H, W) -> (B, T_scr, hidden)
    point_cloud -- PointCloudEmbed     (B, N, F)    -> (B, T_pc, hidden)
    time_series -- TimeSeriesEmbed     (B, T, C)    -> (B, T_ts, hidden)
    graph       -- GraphEmbed          dict         -> (B, T_g, hidden)
    biosignal   -- BioSignalEmbed      (B, T, C)    -> (B, T_bio, hidden)
    unified     -- UnifiedEmbed        orchestrator + early fusion + boundary markers
"""
from __future__ import annotations

from .audio import AudioPatchEmbed
from .biosignal import BioSignalEmbed
from .graph import GraphEmbed
from .image import ImagePatchEmbed
from .point_cloud import PointCloudEmbed
from .screen import ScreenPatchEmbed
from .time_series import TimeSeriesEmbed
from .unified import ModalityMarkers, UnifiedEmbed
from .video import VideoPatchEmbed

__all__ = [
    "ImagePatchEmbed",
    "AudioPatchEmbed",
    "VideoPatchEmbed",
    "ScreenPatchEmbed",
    "PointCloudEmbed",
    "TimeSeriesEmbed",
    "GraphEmbed",
    "BioSignalEmbed",
    "UnifiedEmbed",
    "ModalityMarkers",
]
