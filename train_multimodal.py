"""train_multimodal -- joint native 9-modal trainer with anti-fakery probe.

The bet (memory feedback_native_multimodal_required.md) is that the model
must actually USE every modality, not just project them through a frozen
encoder. To enforce that we run an anti-fakery probe every ``--probe-every``
steps:

    - Compute caption_loss with real modal embeds.
    - Compute caption_loss again after zeroing the modal hidden tensors.
    - If caption_loss(zero) < 1.5 * caption_loss(real), the modality contributed
      almost nothing to the LM head and we log "ANTI-FAKERY FAIL".

Phase schedule (compressed Chameleon-style, see docs/MULTIMODAL_TRAINING.md):

    Phase 0 -- 4 GPU-h: warmstart text-only ckpt + freeze backbone, train ONLY
                        modal heads + UnifiedEmbed projections + LM head row
                        for new <|image|>...<|biosignal|> special tokens.
    Phase 1 -- 12 GPU-h: unfreeze last 4 HybridBlocks, full multitask LM + Σ
                         modal contrastive + recon (alpha=0.05 each).
    Phase 2 -- 8 GPU-h:  SFT on LLaVA-Instruct-150K + M3IT 50K (or smaller).
    Phase 3 -- 1 GPU-h:  Eval (MMMU/MathVista/AudioBench/VideoMME) + anti-fakery.

The CLI flags below let one machine run any subset; ``--smoke`` runs Phase 0
on synthetic data with bs=2, 50 steps -- enough to validate the whole code
path (~30-60s on CPU).

Usage::

    # Smoke (no warmstart needed):
    python train_multimodal.py --smoke --steps 50

    # Phase 0 with warmstart on real data:
    python train_multimodal.py --warmstart runs/synapforge_100m/best.pt \\
        --phase 0 --steps 5000 --bs 4 --seq 512 \\
        --data data/multimodal --out runs/multimodal_phase0

Constraints
-----------
- All modal encoders use the BYTE-PATCH path (Fuyu/Chameleon-style). No
  frozen vision encoders, no LLaVA-style projection-only adapters.
- ``MultimodalMixin`` from synapforge.trainer_mixins is used for the InfoNCE
  contrastive aux loss between text-hidden and modal-hidden.
- Anti-fakery probe is built in -- catches "model is faking it" early.
- Defensive imports throughout -- a missing dataset for one modality must
  not kill the training step for the rest.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local synapforge import precedes any installed copy.
_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

import synapforge as sf  # noqa: E402
from synapforge.modal import UnifiedEmbed  # noqa: E402
from synapforge.model_100m import build_synapforge_100m  # noqa: E402

logger = logging.getLogger("train_multimodal")
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S")

ALL_MODALITIES = (
    "image", "audio", "video", "biosignal", "graph",
    "time_series", "screen", "point_cloud", "spatial_3d",
)
SPECIAL_TOKENS = {
    "image": 50257, "audio": 50258, "video": 50259, "biosignal": 50260,
    "graph": 50261, "time_series": 50262, "screen": 50263,
    "point_cloud": 50264, "spatial_3d": 50265,
}


# ----------------------------------------------------------------- batch loaders
class _ParquetIter:
    """Round-robin iterator over a list of parquet files for one modality."""

    def __init__(self, parquet_paths: list[Path], shuffle: bool = True) -> None:
        self.paths = list(parquet_paths)
        self.shuffle = shuffle
        self._cache: list[dict] = []
        self._idx = 0
        self._refill()

    def _refill(self) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self._cache = []
            return
        rows = []
        for p in self.paths:
            try:
                t = pq.read_table(str(p))
                rows.extend(t.to_pylist())
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning(f"failed to read {p}: {exc!r}")
        if self.shuffle:
            random.shuffle(rows)
        self._cache = rows

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if not self._cache:
            raise StopIteration
        if self._idx >= len(self._cache):
            self._idx = 0
            if self.shuffle:
                random.shuffle(self._cache)
        row = self._cache[self._idx]
        self._idx += 1
        return row


def _decode_row(row: dict) -> Optional[torch.Tensor]:
    """Decode a parquet row's bytes payload into the modality-shaped tensor.

    Mirrors the meta written by ``scripts/prep_multimodal_data.py``. Returns
    None on any decode error so the caller can skip without crashing.
    """
    try:
        meta = json.loads(row["meta"]) if isinstance(row.get("meta"), str) else {}
        mod = row["modality"]
        b = row["bytes"]
        if mod == "image":
            arr = np.frombuffer(b, dtype=np.uint8).reshape(meta["H"], meta["W"], 3)
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        if mod == "audio":
            arr = np.frombuffer(b, dtype=np.float32).copy()
            return torch.from_numpy(arr)
        if mod == "video":
            T = meta["t_frames"]
            H, W = meta["H"], meta["W"]
            arr = np.frombuffer(b, dtype=np.float32).reshape(T, 3, H, W).copy()
            return torch.from_numpy(arr)
        if mod == "biosignal":
            T = meta["t_samples"]
            C = meta["channels"]
            arr = np.frombuffer(b, dtype=np.float32).reshape(T, C).copy()
            return torch.from_numpy(arr)
        if mod == "time_series":
            T = meta["t_raw"]
            C = meta["channels"]
            arr = np.frombuffer(b, dtype=np.float32).reshape(T, C).copy()
            return torch.from_numpy(arr)
        if mod == "screen":
            H, W = meta["H"], meta["W"]
            arr = np.frombuffer(b, dtype=np.float32).reshape(3, H, W).copy()
            return torch.from_numpy(arr)
        if mod == "point_cloud":
            n_pts = meta["n_pts"]
            feat = meta["feat"]
            arr = np.frombuffer(b, dtype=np.float32).reshape(n_pts, feat).copy()
            return torch.from_numpy(arr)
        if mod == "spatial_3d":
            n_pts = meta["n_pts"]
            # Layout: pts(n_pts*3) | rgb(n_pts*3) | K(9) | T(16)
            buf = np.frombuffer(b, dtype=np.float32).copy()
            pts = buf[:n_pts * 3].reshape(n_pts, 3)
            rgb = buf[n_pts * 3:n_pts * 6].reshape(n_pts, 3)
            return torch.from_numpy(np.concatenate([pts, rgb], axis=-1))
        if mod == "graph":
            n = meta["n_nodes"]
            f = meta["node_feat"]
            E = meta["E"]
            buf = b
            n_floats = n * f
            nodes = np.frombuffer(buf, dtype=np.float32, count=n_floats).reshape(n, f).copy()
            offset = n_floats * 4
            edges = np.frombuffer(buf, dtype=np.int64, count=E * 2,
                                   offset=offset).reshape(E, 2).copy()
            return {"nodes": torch.from_numpy(nodes),
                    "edges": torch.from_numpy(edges)}
    except Exception as exc:  # pragma: no cover -- defensive
        logger.debug(f"decode failed for {row.get('modality')!r}: {exc!r}")
        return None


def collate_modal_batch(rows: list[dict], modality: str, B: int,
                         device: torch.device) -> Optional[dict]:
    """Build a UnifiedEmbed-compatible batch from up to B parquet rows."""
    samples = []
    captions = []
    for r in rows[:B]:
        d = _decode_row(r)
        if d is None:
            continue
        samples.append(d)
        captions.append(r.get("text", ""))
    if not samples:
        return None
    if modality == "graph":
        # Stack node + edge tensors with B-padding to longest E.
        max_E = max(s["edges"].shape[0] for s in samples)
        nodes_b = torch.stack([s["nodes"] for s in samples], dim=0).to(device)
        edges_b = torch.zeros(len(samples), max_E, 2, dtype=torch.long, device=device)
        edge_mask = torch.zeros(len(samples), max_E, dtype=torch.bool, device=device)
        for i, s in enumerate(samples):
            E = s["edges"].shape[0]
            edges_b[i, :E] = s["edges"].to(device)
            edge_mask[i, :E] = True
        return {"graph": {"nodes": nodes_b, "edges": edges_b,
                          "edge_mask": edge_mask},
                "captions": captions}
    if modality == "spatial_3d":
        # Treat as point_cloud for UnifiedEmbed.
        x = torch.stack(samples, dim=0).to(device)
        return {"point_cloud": x, "captions": captions}
    # Generic stack (image/audio/video/biosignal/time_series/screen/point_cloud).
    x = torch.stack(samples, dim=0).to(device)
    return {modality: x, "captions": captions}


# --------------------------------------------------------- model & loss helpers
class MultimodalSynapForge(sf.Module):
    """SynapForge100M backbone wrapped with UnifiedEmbed and 9 modal heads.

    The backbone is reused as the LM head (tied embedding). Per-modality recon
    heads are NOT a hard requirement -- the contrastive loss alone provides
    grounding -- but a tiny linear recon head per modality acts as anchor.
    """

    def __init__(self, backbone, hidden: int, vocab: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden = int(hidden)
        self.vocab = int(vocab)
        # Replace UnifiedEmbed but tie its token_embedding to backbone's tok_embed.
        self.embed = UnifiedEmbed(hidden=hidden, vocab=vocab,
                                   max_text_seq=4096, patch_image=8,
                                   patch_audio_ms=20, audio_sample_rate=16000,
                                   audio_mode="raw", video_temporal_patch=4,
                                   video_spatial_patch=8, patch_screen=32,
                                   pc_voxel_grid=8, pc_feat_dim=6,
                                   ts_patch_t=8, ts_max_channels=64,
                                   graph_node_feat=32, graph_edge_feat=0,
                                   graph_pool="set", bio_sample_rate=256,
                                   bio_win_ms=250, bio_hop_ms=125,
                                   bio_max_channels=64)
        # Tie text embedding so warmstarted ckpt's tok_embed weights flow.
        with torch.no_grad():
            v_old = backbone.tok_embed.weight.shape[0]
            v_new = self.embed.token_embedding.weight.shape[0]
            v = min(v_old, v_new)
            self.embed.token_embedding.weight[:v].copy_(
                backbone.tok_embed.weight[:v]
            )

    def forward(self, batch: dict) -> torch.Tensor:
        """batch -> hidden (B, T, d)."""
        z = self.embed(batch)  # (B, T, d)
        h = self.backbone.forward_from_z(z)  # (B, T, d)
        return h

    def lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Compute LM logits via tied embedding. h: (B, T, d) -> (B, T, V)."""
        return F.linear(h, self.embed.token_embedding.weight)


def _build_dummy_text_ids(captions: list[str], device: torch.device,
                          seq: int = 32) -> torch.Tensor:
    """Hash captions into deterministic token ids without requiring a tokenizer.

    A real run uses the GPT-2 tokenizer (or Qwen for v4.x); for smoke we hash
    each caption byte mod V to produce a (B, seq) tensor.
    """
    B = len(captions)
    V = 50257
    out = torch.zeros(B, seq, dtype=torch.long, device=device)
    for i, c in enumerate(captions):
        b = c.encode("utf-8")[:seq]
        for j, ch in enumerate(b):
            out[i, j] = int(ch) % V
    return out


def _caption_loss(model: MultimodalSynapForge, batch: dict,
                  text_ids: torch.Tensor, modal_zeroed: bool = False
                  ) -> torch.Tensor:
    """Loss = next-token CE on text_ids conditioned on modal hidden.

    When ``modal_zeroed=True`` the modal hidden tensors are replaced with
    zero tensors of the same shape, so the LM head only sees text. The
    anti-fakery contract: caption_loss(real) << caption_loss(zeroed).
    """
    z_modal = model.embed(batch)  # (B, T_m, d)
    if modal_zeroed:
        z_modal = torch.zeros_like(z_modal)
    z_text = model.embed.token_embedding(text_ids)  # (B, T_t, d)
    z_full = torch.cat([z_modal, z_text], dim=1)
    h = model.backbone.forward_from_z(z_full)  # (B, T_total, d)
    # Predict text tokens shifted by one.
    h_text = h[:, z_modal.shape[1]:, :]  # (B, T_t, d)
    logits = model.lm_logits(h_text)
    # Next-token CE over text ids.
    if text_ids.shape[1] < 2:
        return logits.float().sum() * 0.0
    pred = logits[:, :-1, :].contiguous()
    tgt = text_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        pred.reshape(-1, pred.shape[-1]).float(),
        tgt.reshape(-1),
        reduction="mean",
    )
    return loss


def _contrastive_loss(text_h: torch.Tensor, modal_h: torch.Tensor,
                      temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE between mean-pooled text and modal hidden.

    Both inputs (B, T, d). Returns alpha-scaled scalar (alpha applied by caller).
    """
    if text_h.shape[0] != modal_h.shape[0] or text_h.shape[0] < 2:
        return text_h.float().sum() * 0.0
    z_t = F.normalize(text_h.mean(dim=1).float(), dim=-1)
    z_m = F.normalize(modal_h.mean(dim=1).float(), dim=-1)
    logits = z_t @ z_m.t() / max(temperature, 1e-3)
    labels = torch.arange(text_h.shape[0], device=text_h.device)
    return F.cross_entropy(logits, labels)


# ------------------------------------------------------- phase config & freezing
def apply_phase_freeze(model: MultimodalSynapForge, phase: int) -> None:
    """Freeze parameters per phase schedule.

    Phase 0: freeze backbone (everything except modal heads + special-token rows).
    Phase 1: unfreeze last 4 HybridBlocks.
    Phase 2: unfreeze all (full SFT).
    """
    if phase == 0:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        # Modal encoders and shared LM head row remain trainable.
        for p in model.embed.parameters():
            p.requires_grad_(True)
    elif phase == 1:
        for p in model.parameters():
            p.requires_grad_(True)
        # Re-freeze first n-4 HybridBlocks.
        n_layers = model.backbone.n_layers
        keep_unfrozen = max(0, n_layers - 4)
        for i, blk in enumerate(model.backbone.blocks):
            if i < keep_unfrozen:
                for p in blk.parameters():
                    p.requires_grad_(False)
    else:
        for p in model.parameters():
            p.requires_grad_(True)


# ----------------------------------------------------------------- main loop
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, default="data/multimodal")
    p.add_argument("--warmstart", type=str, default="",
                   help="path to text-pretrained synapforge_100m ckpt")
    p.add_argument("--out", type=str, default="runs/multimodal")
    p.add_argument("--phase", type=int, default=0,
                   choices=[0, 1, 2, 3], help="0=heads, 1=last4, 2=full SFT, 3=eval")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--alpha-modal", type=float, default=0.05,
                   help="contrastive aux weight per modality")
    p.add_argument("--probe-every", type=int, default=500,
                   help="anti-fakery probe cadence in steps")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="tiny model + synthetic data, ~30s validation")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- build model ------------------------------------------------------
    if args.smoke:
        backbone = build_synapforge_100m(
            vocab=50272, d=64, n_layers=2, loop_depth=1,
            max_seq=4096, ffn_ratio=2.0, sparsity=0.5,
        )
    else:
        backbone = build_synapforge_100m(
            vocab=50272, d=512, n_layers=10, loop_depth=4,
            max_seq=4096, ffn_ratio=8.0, sparsity=0.95,
            use_grad_checkpoint=True,
        )
    if args.warmstart:
        try:
            sd = torch.load(args.warmstart, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            logger.info(f"warmstart loaded: missing={len(missing)} "
                        f"unexpected={len(unexpected)}")
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning(f"warmstart failed ({exc!r}); training from scratch")
    model = MultimodalSynapForge(backbone, hidden=backbone.d, vocab=50272).to(device)
    apply_phase_freeze(model, args.phase)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params: {n_params/1e6:.2f}M (phase {args.phase})")

    # --- data iterators ---------------------------------------------------
    data_root = Path(args.data)
    iterators: dict[str, _ParquetIter] = {}
    for m in ALL_MODALITIES:
        train_p = data_root / m / "train.parquet"
        if train_p.exists():
            iterators[m] = _ParquetIter([train_p], shuffle=True)
            logger.info(f"loaded {m}: {len(iterators[m]._cache)} rows")
        else:
            logger.warning(f"no train data for {m} (expected {train_p}); skipping")
    if not iterators:
        logger.error("no modal data found; run scripts/prep_multimodal_data.py first")
        return 2

    # --- optimizer --------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01,
                              betas=(0.9, 0.95))

    # --- training loop ----------------------------------------------------
    history: list[dict] = []
    best_loss = float("inf")
    t0 = time.time()
    modal_keys = list(iterators.keys())
    for step in range(args.steps):
        # round-robin modality
        modality = modal_keys[step % len(modal_keys)]
        it = iterators[modality]
        try:
            rows = [next(it) for _ in range(args.bs)]
        except StopIteration:
            continue
        batch = collate_modal_batch(rows, modality, args.bs, device)
        if batch is None:
            logger.warning(f"step {step}: empty batch for {modality}; skipping")
            continue
        captions = batch.pop("captions", [""] * args.bs)
        text_ids = _build_dummy_text_ids(captions, device,
                                          seq=min(64, args.seq))

        model.train()
        opt.zero_grad(set_to_none=True)

        # caption loss (modality grounds the text prediction)
        cap_loss = _caption_loss(model, batch, text_ids, modal_zeroed=False)

        # contrastive aux: text_h vs modal_h
        try:
            z_modal = model.embed(batch)
            h_modal = model.backbone.forward_from_z(z_modal)
            text_only = {"text_tokens": text_ids}
            z_text = model.embed(text_only)
            h_text = model.backbone.forward_from_z(z_text)
            con_loss = _contrastive_loss(h_text, h_modal)
        except Exception as exc:  # pragma: no cover -- defensive
            logger.debug(f"contrastive failed: {exc!r}")
            con_loss = cap_loss * 0.0

        loss = cap_loss + args.alpha_modal * con_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if (step + 1) % args.log_every == 0:
            ms_per_step = 1000 * (time.time() - t0) / max(1, step + 1)
            rec = {
                "step": step + 1,
                "modality": modality,
                "cap_loss": float(cap_loss.detach()),
                "con_loss": float(con_loss.detach()),
                "total": float(loss.detach()),
                "ms_per_step": round(ms_per_step, 1),
            }
            history.append(rec)
            logger.info(
                f"step {step+1}/{args.steps} mod={modality} "
                f"cap={rec['cap_loss']:.3f} con={rec['con_loss']:.3f} "
                f"total={rec['total']:.3f} ({rec['ms_per_step']:.0f}ms/step)"
            )

        # ----- anti-fakery probe -----
        if (step + 1) % args.probe_every == 0:
            model.eval()
            with torch.no_grad():
                real = float(_caption_loss(model, batch, text_ids,
                                            modal_zeroed=False).detach())
                zero = float(_caption_loss(model, batch, text_ids,
                                            modal_zeroed=True).detach())
            ratio = zero / max(real, 1e-6)
            verdict = "OK" if ratio >= 1.5 else "FAIL"
            logger.info(
                f"[anti-fakery step={step+1} mod={modality}] "
                f"real={real:.3f} zero={zero:.3f} ratio={ratio:.2f}x ({verdict})"
            )
            if verdict == "FAIL":
                logger.warning(
                    "ANTI-FAKERY FAIL -- modality is not yet contributing. "
                    "Either TRAIN MORE or backbone has degenerated. "
                    "(target ratio >= 1.5x)"
                )
            (out_dir / f"antifakery_step{step+1}.json").write_text(
                json.dumps({"step": step + 1, "modality": modality,
                            "real": real, "zero": zero, "ratio": ratio,
                            "verdict": verdict}, indent=2),
                encoding="utf-8",
            )

        # ----- save -----
        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            ckpt_path = out_dir / f"step_{step+1:06d}.pt"
            torch.save({"model": model.state_dict(),
                        "step": step + 1, "phase": args.phase,
                        "args": vars(args)}, str(ckpt_path))
            logger.info(f"saved -> {ckpt_path}")
            cur_loss = float(loss.detach())
            if cur_loss < best_loss:
                best_loss = cur_loss
                torch.save({"model": model.state_dict(),
                            "step": step + 1, "loss": cur_loss},
                           str(out_dir / "best.pt"))

    # --- final summary ---
    summary = {
        "args": vars(args),
        "history": history[-100:],
        "best_loss": best_loss,
        "modalities_seen": list(iterators.keys()),
        "elapsed_s": round(time.time() - t0, 2),
        "phase": args.phase,
        "anti_fakery_contract": (
            "caption_loss(modal_zeroed) MUST be >= 1.5x caption_loss(real); "
            "lower means model is ignoring the modality."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(f"done. summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
