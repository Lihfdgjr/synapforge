"""train_3d.py -- 3D world understanding scaffold (Task #236).

TODO: Real training requires ~140 GPU-h on A800 per docs/3D.md / docs/3D_PLAN.md.
This file is a SCAFFOLD: it validates the full code path end-to-end on a
synthetic CLEVR-3D parquet emitted by scripts/prep_3d_data.py, in ~30 minutes
on CPU. When GPU rental is available, point --train-data at real CLEVR-3D /
ScanQA mix and uncomment the phase-2 / phase-3 schedules.

Pipeline
--------
1. Load SynapForge100M warmstart ckpt (--warmstart). Backbone frozen by default.
2. Add three modal heads on top:
     - synapforge.modal.PluckerRayEmbed(grid=8, hidden=64)
     - synapforge.modal.EGNNAdapter(hidden=hidden, n_layers=3)
     - synapforge.modal.DUSt3RTeacher (frozen pseudo-label generator stub)
3. Multitask loss (weights from docs/3D.md):
     L = 0.4 L_lm + 0.3 L_pointmap_mse + 0.2 L_view_consistency + 0.1 L_qa
4. 4-phase schedule (docs/3D.md). For 24h budget run only phase 0 + smoke phase 1.

Usage
-----
    python train_3d.py --help
    python train_3d.py --train-data /path/to/clevr3d_synth.parquet \
        --warmstart /path/to/synapforge_100m.pt \
        --steps 200 --batch-size 4 --hidden 384 \
        --out runs/sf_3d_phase0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure local synapforge resolves before any site-packages copy.
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

import synapforge as sf  # noqa: E402
from synapforge.modal import DUSt3RTeacher, EGNNAdapter, PluckerRayEmbed  # noqa: E402


# -----------------------------------------------------------------------------
# Decoder for prep_3d_data.py SF3D-encoded image bytes
# -----------------------------------------------------------------------------
def _sf3d_decode(blob: bytes) -> np.ndarray:
    if blob[:4] != b"SF3D":
        raise ValueError(f"not SF3D blob, magic={blob[:4]!r}")
    h, w = struct.unpack(">HH", blob[4:8])
    payload = zlib.decompress(blob[8:])
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
    return arr


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CLEVR3DParquet(torch.utils.data.Dataset):
    """Reads parquet emitted by scripts/prep_3d_data.py."""

    def __init__(self, path: str) -> None:
        import pyarrow.parquet as pq
        self.path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"clevr3d parquet not found: {path}. Run "
                "`python scripts/prep_3d_data.py --n-examples 1000 --out {path}`"
            )
        self.tbl = pq.read_table(path)
        self._len = self.tbl.num_rows

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        row = self.tbl.slice(idx, 1).to_pylist()[0]
        H, W = int(row["image_h"]), int(row["image_w"])
        img_l = _sf3d_decode(row["image_left"]).astype(np.float32) / 255.0
        img_r = _sf3d_decode(row["image_right"]).astype(np.float32) / 255.0
        depth_l = np.frombuffer(row["depth_left"], dtype=np.float32).reshape(H, W).copy()
        pointmap = np.frombuffer(row["pointmap_gt"], dtype=np.float32).reshape(H, W, 3).copy()
        K = np.frombuffer(row["intrinsics"], dtype=np.float32).reshape(3, 3).copy()
        Tl = np.frombuffer(row["extrinsics_left"], dtype=np.float32).reshape(4, 4).copy()
        Tr = np.frombuffer(row["extrinsics_right"], dtype=np.float32).reshape(4, 4).copy()
        return {
            "image_left": torch.from_numpy(img_l).permute(2, 0, 1),       # (3, H, W)
            "image_right": torch.from_numpy(img_r).permute(2, 0, 1),
            "depth_left": torch.from_numpy(depth_l),
            "pointmap_gt": torch.from_numpy(pointmap),                    # (H, W, 3)
            "intrinsics": torch.from_numpy(K),
            "extrinsics_left": torch.from_numpy(Tl),
            "extrinsics_right": torch.from_numpy(Tr),
            "caption": row["caption"],
            "n_objects": int(row["n_objects"]),
        }


def _collate(rows: list[dict]) -> dict:
    out: dict = {}
    for k in rows[0]:
        if isinstance(rows[0][k], torch.Tensor):
            out[k] = torch.stack([r[k] for r in rows], dim=0)
        else:
            out[k] = [r[k] for r in rows]
    return out


# -----------------------------------------------------------------------------
# Model: SynapForge backbone + 3D heads
# -----------------------------------------------------------------------------
class Spatial3DModel(sf.Module):
    """Backbone (frozen by default) + Plucker + EGNN + pointmap/QA heads."""

    def __init__(
        self,
        hidden: int = 384,
        vocab: int = 50257,
        plucker_grid: int = 8,
        egnn_layers: int = 3,
        n_pointmap_pixels: int = 16,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.n_pm = n_pointmap_pixels
        # Shared embed (covers text + image).
        self.embed = sf.modal.UnifiedEmbed(hidden=hidden, vocab=vocab, max_text_seq=2048)
        # CfC + PLIF backbone (2 layers; mirrors train_full_modal.py).
        self.cfc1 = sf.LiquidCell(hidden, hidden)
        self.plif1 = sf.PLIFCell(hidden, threshold_init=0.3, tau_init="bimodal")
        self.cfc2 = sf.LiquidCell(hidden, hidden)
        self.plif2 = sf.PLIFCell(hidden, threshold_init=0.3, tau_init="bimodal")
        self.norm = nn.LayerNorm(hidden)
        # 3D heads.
        self.plucker = PluckerRayEmbed(grid=plucker_grid, hidden=hidden)
        self.egnn = EGNNAdapter(hidden=hidden, n_layers=egnn_layers, feat_dim=4, radius=2.0)
        # Project EGNN+Plucker to a pointmap (n*n*3) and to QA logits.
        # Pointmap head outputs world-frame xyz at low resolution.
        self.pm_head = nn.Linear(hidden, n_pointmap_pixels * n_pointmap_pixels * 3)
        # View-consistency: contrastive head -> 128-dim (InfoNCE).
        self.consistency = nn.Linear(hidden, 128)
        # QA head: small CE over a fixed answer vocab (caption length).
        self.qa_head = nn.Linear(hidden, vocab)
        # LM head tied to text embedding.
        self.lm_head = sf.tied_lm_head(hidden, vocab, embedding=self.embed.token_embedding)
        # Backbone freeze flag (mutable).
        self._frozen = False

    def freeze_backbone(self) -> None:
        for p in self.embed.parameters():
            p.requires_grad_(False)
        for p in self.cfc1.parameters():
            p.requires_grad_(False)
        for p in self.cfc2.parameters():
            p.requires_grad_(False)
        for p in self.plif1.parameters():
            p.requires_grad_(False)
        for p in self.plif2.parameters():
            p.requires_grad_(False)
        self._frozen = True

    def unfreeze_backbone(self, last_n_layers: int | None = None) -> None:
        # `last_n_layers` controls partial unfreeze (phase 1 unfreezes last 4
        # CfC blocks per docs/3D.md; we only have 2 here, so unfreeze both).
        del last_n_layers
        for p in self.parameters():
            p.requires_grad_(True)
        self._frozen = False

    def _backbone(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, H) -> (B, T, H). Two-layer CfC + PLIF."""
        x = self.cfc1(z)
        v = None
        outs = []
        for t in range(x.shape[1]):
            s, v = self.plif1(x[:, t, :], v)
            outs.append(s)
        x = torch.stack(outs, dim=1)
        x = self.cfc2(x)
        v = None
        outs = []
        for t in range(x.shape[1]):
            s, v = self.plif2(x[:, t, :], v)
            outs.append(s)
        x = torch.stack(outs, dim=1)
        return self.norm(x)

    def forward(self, batch: dict, teacher_pm: torch.Tensor | None = None) -> dict:
        """Returns dict {h, pm_pred, consistency_emb, lm_logits, qa_logits}.

        teacher_pm: (B, n_pm, n_pm, 3) DUSt3R pointmap at low res (already
                    pooled). Used only as input feature for EGNN seeding,
                    not for loss inside this fn.
        """
        device = batch["image_left"].device
        # Build a trivial scene point cloud from teacher pointmap (or GT pointmap).
        if teacher_pm is None:
            raise ValueError("teacher_pm required")
        B = teacher_pm.shape[0]
        N = teacher_pm.shape[1] * teacher_pm.shape[2]
        pts = teacher_pm.reshape(B, N, 3)
        feats = torch.cat(
            [pts, torch.ones(B, N, 1, device=device, dtype=pts.dtype)],
            dim=-1,
        )
        h_egnn, _ = self.egnn(pts, feats)                # (B, 1+N, H)
        h_pluc = self.plucker(batch["intrinsics"], batch["extrinsics_left"])  # (B, 1+g*g, H)
        # Build a tiny text input (caption) -- tokenize crudely (whitespace -> id).
        cap_ids = _crude_tokenise(batch["caption"], self.vocab, device, max_len=24)
        z_text = self.embed({"text_tokens": cap_ids, "image": batch["image_left"]})
        # Concat all token streams: text + image (already in z_text), + plucker + egnn.
        # NOTE: UnifiedEmbed already prepends per-modality markers, no double prepend.
        z = torch.cat([z_text, h_pluc, h_egnn], dim=1)
        h = self._backbone(z)
        # Pool: take last token for pointmap / consistency, mean for QA.
        h_last = h[:, -1, :]
        h_mean = h.mean(dim=1)
        pm_pred = self.pm_head(h_last).reshape(B, self.n_pm, self.n_pm, 3)
        cons = F.normalize(self.consistency(h_last), dim=-1)
        qa_logits = self.qa_head(h_mean)
        # LM logits from the text portion only.
        T_text = z_text.shape[1]
        lm_logits = self.lm_head(h[:, :T_text, :])
        return {
            "h": h,
            "pm_pred": pm_pred,
            "cons": cons,
            "qa_logits": qa_logits,
            "lm_logits": lm_logits,
            "cap_ids": cap_ids,
            "T_text": T_text,
        }


def _crude_tokenise(captions: list[str], vocab: int, device, max_len: int = 24) -> torch.Tensor:
    """Hash-based tokeniser for synthetic captions (placeholder).

    Real training uses the SynapForge tokenizer. This keeps the smoke run
    self-contained without spending a tokenizer download.
    """
    out = []
    for cap in captions:
        ids = []
        for w in cap.split()[:max_len]:
            ids.append(abs(hash(w)) % vocab)
        if not ids:
            ids = [0]
        # Pad / truncate.
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long, device=device)


# -----------------------------------------------------------------------------
# Loss assembly
# -----------------------------------------------------------------------------
def _pool_pointmap(pm: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """(B, H, W, 3) -> (B, h, w, 3) via adaptive avg pool."""
    pm_chw = pm.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(pm_chw, (target_h, target_w))
    return pooled.permute(0, 2, 3, 1).contiguous()


def compute_loss(out: dict, batch: dict, teacher_pm: torch.Tensor) -> dict:
    """Per docs/3D.md weights: 0.4 lm + 0.3 pm + 0.2 view + 0.1 qa."""
    pm_gt_low = _pool_pointmap(batch["pointmap_gt"], out["pm_pred"].shape[1], out["pm_pred"].shape[2])
    # confidence weighted MSE (uniform 1.0 here -- DUSt3RTeacher would supply real conf).
    l_pm = F.mse_loss(out["pm_pred"], pm_gt_low.to(out["pm_pred"].dtype))
    # view consistency: dummy InfoNCE -- treat batch as positive pairs (left/right share scene).
    cons = out["cons"]                                # (B, 128)
    sim = cons @ cons.t()                             # (B, B)
    labels = torch.arange(cons.shape[0], device=cons.device)
    l_view = F.cross_entropy(sim / 0.07, labels)
    # LM loss: shift caption ids by 1.
    lm_logits = out["lm_logits"]                     # (B, T_text, vocab)
    cap_ids = out["cap_ids"]                          # (B, max_len)
    # Align: lm covers text marker + cap_ids; predict cap_ids[1:] from cap_ids[:-1].
    if cap_ids.shape[1] >= 2:
        # take the last cap_ids.shape[1]-1 logits (post-marker) for prediction.
        T = cap_ids.shape[1]
        # h has the layout [marker, t0, ..., t_{T-1}, ...]; lm_logits has same offset
        # but is sliced to T_text = 1+T positions. Predict cap_ids[1:T] from positions [1:T-1]+1 etc.
        pred = lm_logits[:, 1:T, :]
        target = cap_ids[:, 1:T]
        l_lm = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
    else:
        l_lm = torch.zeros((), device=lm_logits.device)
    # QA: predict cap_ids[0] (first content word) from h_mean. Toy proxy.
    l_qa = F.cross_entropy(out["qa_logits"], cap_ids[:, 0])
    total = 0.4 * l_lm + 0.3 * l_pm + 0.2 * l_view + 0.1 * l_qa
    return {
        "total": total,
        "l_lm": l_lm.detach(),
        "l_pm": l_pm.detach(),
        "l_view": l_view.detach(),
        "l_qa": l_qa.detach(),
    }


# -----------------------------------------------------------------------------
# Phase schedule
# -----------------------------------------------------------------------------
PHASES = [
    # (name, gpu_hours, lr, action)
    ("phase_0_freeze", 10, 5e-4, "freeze_backbone"),
    ("phase_1_partial", 60, 2e-4, "unfreeze_partial"),
    ("phase_2_full", 50, 5e-5, "unfreeze_full"),
    ("phase_3_sft", 20, 1e-5, "sft_only"),
]


def lr_at(step: int, peak: float, warmup: int, total: int, lr_min: float = 1e-5) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return lr_min + (peak - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", type=str, default="runs/clevr3d_smoke.parquet",
                   help="Path to parquet from scripts/prep_3d_data.py")
    p.add_argument("--warmstart", type=str, default=None,
                   help="SynapForge100M ckpt to warmstart backbone from")
    p.add_argument("--out", type=str, default="runs/sf_3d_phase0")
    p.add_argument("--phase", type=str, default="phase_0_freeze",
                   choices=[ph[0] for ph in PHASES] + ["smoke"])
    p.add_argument("--steps", type=int, default=200,
                   help="Override schedule step count (24h budget = phase 0 + smoke phase 1)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512,
                   help="Max unified seq len (smaller because point cloud carries tokens)")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--lr", type=float, default=None,
                   help="Override phase peak lr")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Enable gradient checkpointing on backbone (recommended for B>=4 + point cloud)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logp = out / "train.log"
    metricsp = out / "metrics.jsonl"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logp, "a") as f:
            f.write(line + "\n")

    log(f"args: {vars(args)}")
    device = torch.device(args.device)

    # ---- model ----
    model = Spatial3DModel(hidden=args.hidden, vocab=50257).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"model params total: {n_total/1e6:.2f}M")

    # ---- warmstart ----
    if args.warmstart and os.path.exists(args.warmstart):
        sd = torch.load(args.warmstart, map_location=device)
        if "state" in sd:
            sd = sd["state"]
        # strict=False: backbone keys load, new heads (plucker/egnn/pm/qa) stay random.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"warmstart from {args.warmstart}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        if args.warmstart:
            log(f"WARN: --warmstart not found: {args.warmstart} (proceeding from scratch)")
        else:
            log("no --warmstart provided; backbone init random (smoke OK, real run NOT OK)")

    # ---- phase action ----
    phase_cfg = next(ph for ph in PHASES if ph[0] == args.phase) if args.phase != "smoke" else (
        "smoke", 0.01, 5e-4, "freeze_backbone"
    )
    name, hours, peak_lr, action = phase_cfg
    if args.lr is not None:
        peak_lr = args.lr
    if action == "freeze_backbone":
        model.freeze_backbone()
        log("backbone frozen (phase 0)")
    elif action == "unfreeze_partial":
        model.unfreeze_backbone(last_n_layers=4)
        log("backbone partially unfrozen (last 4 cfc blocks)")
    elif action == "unfreeze_full":
        model.unfreeze_backbone()
        log("backbone fully unfrozen")
    elif action == "sft_only":
        # Load ScanQA train here in real run; smoke uses same data.
        log("sft_only phase (uses same parquet in scaffold)")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params: {n_train/1e6:.2f}M")

    # ---- DUSt3R teacher (frozen) ----
    teacher = DUSt3RTeacher(out_h=16, out_w=16).to(device)
    if not teacher.is_real():
        warnings.warn(
            "DUSt3R teacher running on STUB weights -- pointmap pseudo-labels are "
            "only useful for code-path validation, NOT real metrics.",
            RuntimeWarning,
        )

    # ---- data ----
    ds = CLEVR3DParquet(args.train_data)
    log(f"dataset: {len(ds)} examples from {args.train_data}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )

    # ---- optimizer ----
    optim = sf.build_optimizer(model, lr=peak_lr, weight_decay=args.weight_decay)
    log(f"optim={type(optim).__name__} peak_lr={peak_lr} wd={args.weight_decay}")

    # ---- training loop ----
    model.train()
    t0 = time.time()
    step = 0
    nan_count = 0
    iter_loader = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)
        # Move to device.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Teacher pseudo-labels (frozen, no_grad).
        with torch.no_grad():
            teacher_out = teacher(batch["image_left"], batch["image_right"])
            teacher_pm = teacher_out["pointmap"]   # (B, 16, 16, 3)

        cur_lr = lr_at(step, peak_lr, args.warmup, args.steps)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr

        out_dict = model(batch, teacher_pm=teacher_pm)
        losses = compute_loss(out_dict, batch, teacher_pm)
        loss = losses["total"]

        optim.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optim.step()
        else:
            nan_count += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            line = (
                f"step {step:5d}/{args.steps} loss={float(loss):.4f} "
                f"lm={float(losses['l_lm']):.3f} pm={float(losses['l_pm']):.3f} "
                f"view={float(losses['l_view']):.3f} qa={float(losses['l_qa']):.3f} "
                f"lr={cur_lr:.6f} elapsed={elapsed:.1f}s nan={nan_count}"
            )
            log(line)
            with open(metricsp, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": float(loss),
                    "l_lm": float(losses["l_lm"]),
                    "l_pm": float(losses["l_pm"]),
                    "l_view": float(losses["l_view"]),
                    "l_qa": float(losses["l_qa"]),
                    "lr": cur_lr, "elapsed": elapsed, "nan": nan_count,
                }) + "\n")

        if (step + 1) % args.ckpt_every == 0:
            ck = out / f"ckpt_step{step+1}.pt"
            torch.save({"step": step + 1, "phase": name, "state": model.state_dict()}, ck)
            log(f"saved ckpt {ck}")

    # Final save.
    final = out / "ckpt_final.pt"
    torch.save({"step": args.steps, "phase": name, "state": model.state_dict()}, final)
    log(f"DONE: {args.steps} steps in {time.time()-t0:.1f}s, nan={nan_count}, saved {final}")


if __name__ == "__main__":
    main()
