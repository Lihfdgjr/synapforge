"""train_full_modal.py — joint native training across ALL 9 synapforge modalities.

Per the bet (memory feedback_native_multimodal_required.md):
- NO frozen encoders, NO conv layers anywhere in this script.
- All modalities go through `sf.modal.UnifiedEmbed` (byte-patch + Linear).
- One CfC + PLIF backbone receives the unified token stream.
- Per-modality recon heads (Linear only) supervise self-recon (denoising).
- Action head & NeuroMCP head exercise sf.action native paths.

Usage
-----
    python train_full_modal.py --steps 4000 --batch-size 16 --lr 2e-4 \\
        --warmup 200 --out /workspace/runs/synapforge_full_modal
"""
from __future__ import annotations

# ------------------------------------------------------------------ sys.path
import sys
sys.path[:] = [p for p in sys.path if p not in ("/workspace/synapforge", "")]
sys.path.insert(0, "/workspace")

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import synapforge as sf  # noqa: E402

# -------------------------------------------------------------- model
class FullModalModel(sf.Module):
    """One UnifiedEmbed -> CfC+PLIF x2 -> {LM head, per-modal recon, action, NeuroMCP}."""

    def __init__(self, hidden: int = 384, vocab: int = 50257) -> None:
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab

        # --- shared input fusion -------------------------------------------------
        self.embed = sf.modal.UnifiedEmbed(
            hidden=hidden,
            vocab=vocab,
            max_text_seq=4096,
            patch_image=8,
            patch_audio_ms=20,
            audio_sample_rate=16000,
            audio_mode="raw",
            video_temporal_patch=4,
            video_spatial_patch=8,
            patch_screen=32,
            pc_voxel_grid=8,
            pc_feat_dim=6,
            ts_patch_t=8,
            ts_max_channels=64,
            graph_node_feat=32,
            graph_edge_feat=0,
            graph_pool="set",
            bio_sample_rate=256,
            bio_win_ms=250,
            bio_hop_ms=125,
            bio_max_channels=64,
        )

        # --- two-layer CfC + PLIF backbone --------------------------------------
        self.cfc1 = sf.LiquidCell(hidden, hidden)
        self.plif1 = sf.PLIFCell(hidden, threshold_init=0.3, tau_init="bimodal")
        self.cfc2 = sf.LiquidCell(hidden, hidden)
        self.plif2 = sf.PLIFCell(hidden, threshold_init=0.3, tau_init="bimodal")
        self.norm = nn.LayerNorm(hidden)

        # --- LM head tied to text token embedding -------------------------------
        self.lm_head = sf.tied_lm_head(
            hidden, vocab, embedding=self.embed.token_embedding
        )

        # --- per-modal recon heads (Linear only, no conv) -----------------------
        # Each head projects the *modality-pooled* hidden vector to flat input dim.
        # image: 3*32*32 = 3072
        self.head_image = nn.Linear(hidden, 3 * 32 * 32)
        # audio: 16000 raw samples
        self.head_audio = nn.Linear(hidden, 16000)
        # video: 8*3*16*16 = 6144
        self.head_video = nn.Linear(hidden, 8 * 3 * 16 * 16)
        # screen: 3*64*64 = 12288
        self.head_screen = nn.Linear(hidden, 3 * 64 * 64)
        # point_cloud: 256*6 = 1536
        self.head_point_cloud = nn.Linear(hidden, 256 * 6)
        # time_series: 64*8 = 512
        self.head_time_series = nn.Linear(hidden, 64 * 8)
        # graph: node features 16*32 = 512
        self.head_graph = nn.Linear(hidden, 16 * 32)
        # biosignal: 1024*8 = 8192
        self.head_biosignal = nn.Linear(hidden, 1024 * 8)

        # --- action heads -------------------------------------------------------
        self.action = sf.action.ActionHead(hidden, sf.action.OSActionSpec.default())
        self.neuromcp = sf.action.NeuroMCPHead(
            hidden,
            codebook_initial=8,
            codebook_max=64,
            synapse_density=0.05,
            synapse_max_density=0.30,
        )

    # -- backbone --
    def _backbone(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, H) -> (B, T, H). Two-layer CfC + PLIF, time-axis serial PLIF."""
        # CfC #1 (parallel scan over T, internal).
        x = self.cfc1(z)  # (B, T, H)
        # PLIF #1: stepwise over T axis.
        v_prev = None
        outs = []
        for t in range(x.shape[1]):
            s_t, v_prev = self.plif1(x[:, t, :], v_prev)
            outs.append(s_t)
        x = torch.stack(outs, dim=1)
        # CfC #2 + PLIF #2.
        x = self.cfc2(x)
        v_prev = None
        outs = []
        for t in range(x.shape[1]):
            s_t, v_prev = self.plif2(x[:, t, :], v_prev)
            outs.append(s_t)
        x = torch.stack(outs, dim=1)
        return self.norm(x)

    def forward(self, batch: dict) -> torch.Tensor:
        z = self.embed(batch)        # (B, T_unified, H)
        h = self._backbone(z)        # (B, T_unified, H)
        return h


# -------------------------------------------------------------- batch builders
def make_text_batch(stream_iter, B: int, device) -> dict:
    tokens_in, tokens_out = next(stream_iter)
    return {
        "_kind": "text",
        "text_tokens": tokens_in.to(device),
        "labels": tokens_out.to(device),
    }


def _rand(*shape, device, dtype=torch.float32) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=dtype)


def make_image_batch(B, device):
    x = _rand(B, 3, 32, 32, device=device)
    return {"_kind": "image", "image": x, "labels": x.reshape(B, -1)}


def make_audio_batch(B, device):
    x = _rand(B, 16000, device=device)
    return {"_kind": "audio", "audio": x, "labels": x}


def make_video_batch(B, device):
    x = _rand(B, 8, 3, 16, 16, device=device)
    return {"_kind": "video", "video": x, "labels": x.reshape(B, -1)}


def make_screen_batch(B, device):
    x = _rand(B, 3, 64, 64, device=device)
    return {"_kind": "screen", "screen": x, "labels": x.reshape(B, -1)}


def make_point_cloud_batch(B, device):
    x = _rand(B, 256, 6, device=device)
    return {"_kind": "point_cloud", "point_cloud": x, "labels": x.reshape(B, -1)}


def make_time_series_batch(B, device):
    x = _rand(B, 64, 8, device=device)
    return {"_kind": "time_series", "time_series": x, "labels": x.reshape(B, -1)}


def make_graph_batch(B, device):
    nodes = _rand(B, 16, 32, device=device)
    # use minimal edges tensor; GraphEmbed expects edges (B, E, ...); use (B, 16, 16)
    # GraphEmbed expects edges (B, E, 2) long; build a simple ring graph (E=16).
    src = torch.arange(16, device=device)
    dst = (src + 1) % 16
    edges_one = torch.stack([src, dst], dim=-1)
    edges = edges_one.unsqueeze(0).expand(B, -1, -1).contiguous().long()
    return {
        "_kind": "graph",
        "graph": {"nodes": nodes, "edges": edges},
        "labels": nodes.reshape(B, -1),
    }


def make_biosignal_batch(B, device):
    x = _rand(B, 1024, 8, device=device)
    return {"_kind": "biosignal", "biosignal": x, "labels": x.reshape(B, -1)}


def make_action_batch(B, device, spec):
    # Use text as a stand-in upstream context so backbone has tokens.
    text = torch.randint(0, 50257, (B, 32), device=device, dtype=torch.long)
    n_types = len(spec.action_types)
    n_keys = spec.num_keys
    type_ids = torch.randint(0, n_types, (B,), device=device, dtype=torch.long)
    return {
        "_kind": "neuromcp_action",
        "text_tokens": text,
        "action_type": type_ids,
        "xy": torch.rand(B, 2, device=device),
        "scroll": (torch.rand(B, 2, device=device) * 2 - 1),
        "key": torch.randint(0, n_keys, (B,), device=device, dtype=torch.long),
        "text_trigger": torch.randint(0, 2, (B,), device=device).float(),
    }


# -------------------------------------------------------------- LR schedule
def lr_at(step: int, peak: float, warmup: int, total: int, lr_min: float = 1e-5) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (peak - lr_min) * cosine


# -------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--out", type=str, default="/workspace/runs/synapforge_full_modal")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument(
        "--text-glob", type=str,
        default="/workspace/data/wt103_raw/train-*.parquet",
    )
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logp = out / "train.log"
    metricsp = out / "metrics.jsonl"

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logp, "a") as f:
            f.write(line + "\n")

    log(f"args: {vars(args)}")
    device = torch.device(args.device)
    log(f"device={device}")

    # ---- model ----
    model = FullModalModel(hidden=args.hidden, vocab=50257).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model params: {n_params/1e6:.2f}M")

    # ---- optimizer (sf.optim native) ----
    optim = sf.build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
    )
    log(f"optim={type(optim).__name__} lr={args.lr} wd={args.weight_decay}")

    # ---- text data stream ----
    text_stream = sf.ParquetTokenStream(
        args.text_glob, seq_len=128, batch_size=args.batch_size, loop=True,
    )
    text_iter = iter(text_stream)
    log(f"text stream: {text_stream}")

    # ---- action loss helper ----
    spec = sf.action.OSActionSpec.default()
    action_loss = sf.action.head.ActionLoss(spec).to(device)

    # ---- task cycle (10 tasks) ----
    task_order = [
        "text", "image", "audio", "video", "screen",
        "point_cloud", "time_series", "graph", "biosignal", "neuromcp_action",
    ]

    builders = {
        "text": lambda: make_text_batch(text_iter, args.batch_size, device),
        "image": lambda: make_image_batch(args.batch_size, device),
        "audio": lambda: make_audio_batch(args.batch_size, device),
        "video": lambda: make_video_batch(args.batch_size, device),
        "screen": lambda: make_screen_batch(args.batch_size, device),
        "point_cloud": lambda: make_point_cloud_batch(args.batch_size, device),
        "time_series": lambda: make_time_series_batch(args.batch_size, device),
        "graph": lambda: make_graph_batch(args.batch_size, device),
        "biosignal": lambda: make_biosignal_batch(args.batch_size, device),
        "neuromcp_action": lambda: make_action_batch(args.batch_size, device, spec),
    }

    head_for = {
        "image": model.head_image,
        "audio": model.head_audio,
        "video": model.head_video,
        "screen": model.head_screen,
        "point_cloud": model.head_point_cloud,
        "time_series": model.head_time_series,
        "graph": model.head_graph,
        "biosignal": model.head_biosignal,
    }

    # ---- training loop ----
    model.train()
    t0 = time.time()
    nan_count = 0
    # Per-task running-avg loss (10-step EMA) for nicer logs.
    loss_avg = {t: 0.0 for t in task_order}
    loss_count = {t: 0 for t in task_order}
    for step in range(args.steps):
        # LR schedule
        cur_lr = lr_at(step, args.lr, args.warmup, args.steps, args.lr_min)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr

        task = task_order[step % len(task_order)]
        batch = builders[task]()
        kind = batch.pop("_kind")

        # Forward through embed + backbone.
        # For text/action tasks the input dict has text_tokens; for the others
        # only the modality field is present, so UnifiedEmbed only embeds that.
        embed_input = {k: v for k, v in batch.items()
                       if k not in ("labels", "action_type", "xy", "scroll",
                                    "key", "text_trigger")}
        h = model(embed_input)  # (B, T_uni, H)

        # Pool: last-token for everything; LM uses per-token alignment.
        if kind == "text":
            # h was built from text_tokens => layout: [text_marker, t0, t1, ... tT-1]
            # Skip the marker token (idx 0) and align to labels of length T.
            T = batch["labels"].shape[1]
            h_text = h[:, 1:1 + T, :]               # (B, T, H)
            logits = model.lm_head(h_text)          # (B, T, vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab),
                batch["labels"].reshape(-1),
                ignore_index=-100,
            )
            extra = {}
        elif kind == "neuromcp_action":
            h_pool = h[:, -1, :]                     # (B, H)
            # ActionHead expects (B, ..., H); produce action with T-axis squeezed.
            action_out = model.action(h_pool.unsqueeze(1))  # (B, 1, ...) per field
            # Build targets aligned with T=1.
            from synapforge.action.head import ActionTargets
            tgt = ActionTargets(
                action_type=batch["action_type"].unsqueeze(1),
                xy=batch["xy"].unsqueeze(1),
                scroll=batch["scroll"].unsqueeze(1),
                key=batch["key"].unsqueeze(1),
                text_trigger=batch["text_trigger"].unsqueeze(1),
            )
            l_action = action_loss(action_out, tgt)
            if isinstance(l_action, dict):
                l_action = l_action.get("total", sum(v for v in l_action.values() if torch.is_tensor(v)))
            mcp_out = model.neuromcp(h_pool)         # {logits, hidden_z}
            # Self-supervised: encourage entropy across the action codebook.
            mcp_logits = mcp_out["logits"]
            mcp_log_probs = F.log_softmax(mcp_logits, dim=-1)
            l_mcp = -(mcp_log_probs.exp() * mcp_log_probs).sum(-1).mean()
            loss = l_action + 0.1 * l_mcp
            extra = {"l_action": float(l_action.detach()), "l_mcp": float(l_mcp.detach())}
        else:
            # Modality recon: pool h to (B, H), project to flat input, MSE vs label.
            h_pool = h.mean(dim=1)                    # (B, H)
            head = head_for[kind]
            recon = head(h_pool)                       # (B, flat_dim)
            target = batch["labels"]
            if recon.shape != target.shape:
                # reshape recon to match
                recon = recon.reshape(target.shape)
            loss = F.mse_loss(recon, target)
            extra = {}

        # Optimize
        optim.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        else:
            nan_count += 1

        # NeuroMCP plasticity step
        if kind == "neuromcp_action":
            try:
                with torch.no_grad():
                    model.neuromcp.step_plasticity()
            except Exception as e:
                if step < 5:
                    log(f"  warn: neuromcp.step_plasticity skipped: {e}")

        # Log
        # Per-task EMA tracking
        loss_count[task] += 1
        loss_avg[task] = 0.95 * loss_avg[task] + 0.05 * float(loss.item()) if loss_count[task] > 1 else float(loss.item())
        if step % 50 == 0 and step > 0:
            summary = " ".join(f"{t[:4]}={loss_avg[t]:.2f}" for t in task_order)
            print(f"[{time.strftime('%H:%M:%S')}] EMA step={step:4d} | {summary}", flush=True)
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            line = (f"step {step:5d}/{args.steps} task={kind:<14s} "
                    f"loss={float(loss):.4f} lr={cur_lr:.6f} "
                    f"elapsed={elapsed:.1f}s nan={nan_count}")
            if extra:
                line += " " + " ".join(f"{k}={v:.4f}" for k, v in extra.items())
            log(line)
            with open(metricsp, "a") as f:
                f.write(json.dumps({
                    "step": step, "task": kind, "loss": float(loss),
                    "lr": cur_lr, "elapsed": elapsed, "nan": nan_count, **extra,
                }) + "\n")

        # Checkpoint
        if (step + 1) % args.ckpt_every == 0:
            ck = out / f"ckpt_step{step+1}.pt"
            torch.save({"step": step + 1, "state": model.state_dict()}, ck)
            log(f"saved ckpt {ck}")

    log(f"DONE: {args.steps} steps in {time.time()-t0:.1f}s, nan_count={nan_count}")


if __name__ == "__main__":
    main()
