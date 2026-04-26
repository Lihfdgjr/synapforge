"""UnifiedEmbed -- early-fusion orchestrator for ALL 9 modalities.

Round 6A modalities: text / image / audio / video.
Round 6B modalities: screen / point_cloud / time_series / graph / biosignal.

Per the bet (memory feedback_native_multimodal_required.md) we do NOT use
LLaVA-style frozen vision encoders bolted on the side. Every modality is
patched + linearly projected into the SAME hidden dim, then concatenated
along the time axis. The downstream HybridBlock (CfC + PLIF + SparseSynapse)
sees ONE unified token stream and learns mode transitions itself.

Public API
----------
    embed = UnifiedEmbed(hidden=512, vocab=50257)
    z = embed({
        "text_tokens":  torch.long  (B, Tt)              optional,
        "image":        torch.float (B, 3, H, W)         optional,
        "audio":        torch.float (B, samples)         optional,
        "video":        torch.float (B, Tf, 3, H, W)     optional,
        "screen":       torch.float (B, 3, H, W)         optional,
        "point_cloud":  torch.float (B, N, F>=3)         optional,
        "time_series":  torch.float (B, T_raw, C)        optional,
        "graph":        dict {nodes, edges, ...}         optional,
        "biosignal":    torch.float (B, T_samp, C)       optional,
    })
    # z: (B, T_unified, hidden)
    # Output token order matches `order` arg.
    # A learned <|sep|> separator is inserted between modalities.

The text-embedding `Embedding` layer is exposed as `embed.token_embedding` so
callers can build a tied LM head.

The forward never branches on which modalities are present; missing keys are
simply skipped. This satisfies the constraint "same model handles all batch
types without code branches" -- the user code passes a dict, the orchestrator
loops.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..module import Module
from .audio import AudioPatchEmbed
from .biosignal import BioSignalEmbed
from .graph import GraphEmbed
from .image import ImagePatchEmbed
from .point_cloud import PointCloudEmbed
from .screen import ScreenPatchEmbed
from .time_series import TimeSeriesEmbed
from .video import VideoPatchEmbed


class ModalityMarkers(nn.Module):
    """Holds the modality boundary markers + one shared <|sep|>.

    Per-modality markers live INSIDE each *PatchEmbed module (ImagePatchEmbed
    has its own .marker, etc). This class only holds:
      - text marker (text branch is implemented inside UnifiedEmbed itself).
      - shared <|sep|> separator (inserted between any two adjacent modalities).
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.text = nn.Parameter(torch.zeros(hidden))
        self.sep = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.text, std=0.02)
        nn.init.normal_(self.sep, std=0.02)


class UnifiedEmbed(Module):
    """Orchestrator that fuses ALL 9 modalities into one token stream."""

    DEFAULT_ORDER = (
        "text", "image", "audio", "video",
        "screen", "point_cloud", "time_series", "graph", "biosignal",
    )

    def __init__(
        self,
        hidden: int = 512,
        vocab: int = 50257,
        # text
        max_text_seq: int = 4096,
        tokenizer: str = "gpt2",
        # image
        patch_image: int = 8,
        # audio
        patch_audio_ms: int = 20,
        audio_sample_rate: int = 16000,
        audio_mode: str = "raw",
        # video
        video_temporal_patch: int = 4,
        video_spatial_patch: int | None = None,
        # screen
        patch_screen: int = 32,
        # point cloud
        pc_voxel_grid: int = 8,
        pc_feat_dim: int = 6,
        # time series
        ts_patch_t: int = 8,
        ts_max_channels: int = 64,
        # graph
        graph_node_feat: int = 32,
        graph_edge_feat: int = 0,
        graph_rounds: int = 3,
        graph_pool: str = "set",
        # biosignal
        bio_sample_rate: int = 256,
        bio_win_ms: int = 250,
        bio_hop_ms: int = 125,
        bio_max_channels: int = 64,
        # ordering
        order: tuple[str, ...] = DEFAULT_ORDER,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.vocab = int(vocab)
        self.tokenizer_name = tokenizer
        self.order = tuple(order)
        for o in order:
            if o not in self.DEFAULT_ORDER:
                raise ValueError(f"unknown modality {o!r} in order")

        # ---- text branch ----
        self.token_embedding = nn.Embedding(vocab, hidden)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        self.text_pos = nn.Parameter(torch.zeros(max_text_seq, hidden))
        nn.init.normal_(self.text_pos, std=0.02)
        self.max_text_seq = int(max_text_seq)

        # ---- image / audio / video branches ----
        self.image_embed = ImagePatchEmbed(hidden=hidden, patch=patch_image)
        self.audio_embed = AudioPatchEmbed(
            hidden=hidden,
            sample_rate=audio_sample_rate,
            chunk_ms=patch_audio_ms,
            mode=audio_mode,
        )
        if video_spatial_patch is None:
            video_spatial_patch = patch_image
        self.video_embed = VideoPatchEmbed(
            hidden=hidden,
            spatial_patch=video_spatial_patch,
            temporal_patch=video_temporal_patch,
        )

        # ---- Round 6B branches ----
        self.screen_embed = ScreenPatchEmbed(hidden=hidden, patch=patch_screen)
        self.point_cloud_embed = PointCloudEmbed(
            hidden=hidden, voxel_grid=pc_voxel_grid, feat_dim=pc_feat_dim,
        )
        self.time_series_embed = TimeSeriesEmbed(
            hidden=hidden, patch_t=ts_patch_t, max_channels=ts_max_channels,
        )
        self.graph_embed = GraphEmbed(
            hidden=hidden, node_feat=graph_node_feat, edge_feat=graph_edge_feat,
            rounds=graph_rounds, pool=graph_pool,
        )
        self.biosignal_embed = BioSignalEmbed(
            hidden=hidden, sample_rate=bio_sample_rate,
            win_ms=bio_win_ms, hop_ms=bio_hop_ms, max_channels=bio_max_channels,
        )

        # ---- modality markers (text + shared sep) ----
        self.markers = ModalityMarkers(hidden)

    # -------------------------------------------------------- internal helpers

    def _embed_text(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, Tt) long -> (B, 1+Tt, hidden)."""
        if ids.dim() != 2:
            raise ValueError(f"text_tokens must be (B, Tt); got {tuple(ids.shape)}")
        B, Tt = ids.shape
        if Tt > self.max_text_seq:
            raise ValueError(f"text seq_len {Tt} > max_text_seq {self.max_text_seq}")
        z = self.token_embedding(ids) + self.text_pos[:Tt].unsqueeze(0)
        marker = self.markers.text.to(z.dtype).expand(B, 1, self.hidden)
        return torch.cat([marker, z], dim=1)

    def _modality_seq(self, batch: dict, key: str) -> torch.Tensor | None:
        if key not in batch or batch[key] is None:
            return None
        if key == "text_tokens":
            return self._embed_text(batch[key])
        if key == "image":
            return self.image_embed(batch[key])
        if key == "audio":
            return self.audio_embed(batch[key])
        if key == "video":
            return self.video_embed(batch[key])
        if key == "screen":
            # Allow tuple form (screen, cursor, roi) for ROI crops.
            v = batch[key]
            if isinstance(v, (tuple, list)):
                if len(v) == 3:
                    return self.screen_embed(v[0], cursor=v[1], roi=v[2])
                return self.screen_embed(v[0])
            return self.screen_embed(v)
        if key == "point_cloud":
            v = batch[key]
            if isinstance(v, dict):
                return self.point_cloud_embed(v["points"], mask=v.get("mask"))
            return self.point_cloud_embed(v)
        if key == "time_series":
            return self.time_series_embed(batch[key])
        if key == "graph":
            return self.graph_embed(batch[key])
        if key == "biosignal":
            return self.biosignal_embed(batch[key])
        return None

    def _key_for_modality(self, m: str) -> str:
        return "text_tokens" if m == "text" else m

    # -------------------------------------------------- forward

    def forward(self, batch: dict) -> torch.Tensor:
        if not isinstance(batch, dict):
            raise TypeError(f"UnifiedEmbed expects a dict batch; got {type(batch)}")

        # Determine batch size from the first present modality.
        present: list[tuple[str, torch.Tensor]] = []
        B: int | None = None
        device = None
        dtype = self.token_embedding.weight.dtype  # default

        for m in self.order:
            key = self._key_for_modality(m)
            seq = self._modality_seq(batch, key)
            if seq is not None:
                if B is None:
                    B = seq.shape[0]
                    device = seq.device
                    dtype = seq.dtype
                else:
                    if seq.shape[0] != B:
                        raise ValueError(
                            f"modality {m} batch size {seq.shape[0]} != {B}"
                        )
                present.append((m, seq))

        if not present:
            raise ValueError(
                "UnifiedEmbed.forward got an empty batch -- need at least one of "
                "text_tokens / image / audio / video / screen / point_cloud / "
                "time_series / graph / biosignal"
            )

        # Insert <|sep|> between modalities.
        sep = self.markers.sep.to(dtype).expand(B, 1, self.hidden)
        chunks: list[torch.Tensor] = []
        for i, (_m, seq) in enumerate(present):
            if i > 0:
                chunks.append(sep)
            chunks.append(seq.to(dtype))
        z = torch.cat(chunks, dim=1)
        return z

    # -------------------------------------------------- helpers for callers

    def tied_lm_head_weight(self) -> torch.Tensor:
        """Shortcut so the LM head can be tied to the text token embedding."""
        return self.token_embedding.weight

    def expected_unified_len(
        self,
        text_len: int = 0,
        image_hw: tuple[int, int] | None = None,
        audio_samples: int = 0,
        video_thw: tuple[int, int, int] | None = None,
        screen_hw: tuple[int, int] | None = None,
        point_cloud: bool = False,
        time_series_t: int = 0,
        graph_n_nodes: int = 0,
        biosignal_samples: int = 0,
    ) -> int:
        """Compute T_unified for a planning batch (no torch ops)."""
        present_count = 0
        T = 0
        if text_len > 0:
            T += 1 + text_len
            present_count += 1
        if image_hw is not None:
            H, W = image_hw
            T += 1 + (H // self.image_embed.patch) * (W // self.image_embed.patch)
            present_count += 1
        if audio_samples > 0:
            cs = self.audio_embed.chunk_size
            n_chunks = (audio_samples + cs - 1) // cs
            T += 1 + n_chunks
            present_count += 1
        if video_thw is not None:
            Tf, H, W = video_thw
            tp = self.video_embed.tp
            sp = self.video_embed.sp
            Tf_pad = ((Tf + tp - 1) // tp) * tp
            T += 1 + (Tf_pad // tp) * (H // sp) * (W // sp)
            present_count += 1
        if screen_hw is not None:
            H, W = screen_hw
            p = self.screen_embed.patch
            rows = (H + p - 1) // p
            cols = (W + p - 1) // p
            T += 1 + rows * cols
            present_count += 1
        if point_cloud:
            V = self.point_cloud_embed.V
            T += 1 + V * V * V
            present_count += 1
        if time_series_t > 0:
            pt = self.time_series_embed.patch_t
            T += 1 + (time_series_t + pt - 1) // pt
            present_count += 1
        if graph_n_nodes > 0:
            T += 1 + (1 if self.graph_embed.pool == "readout" else graph_n_nodes)
            present_count += 1
        if biosignal_samples > 0:
            be = self.biosignal_embed
            if biosignal_samples < be.win:
                Tw = 1
            else:
                Tw = (biosignal_samples - be.win) // be.hop + 1
            T += 1 + Tw
            present_count += 1
        if present_count > 1:
            T += present_count - 1  # <|sep|> tokens
        return T
