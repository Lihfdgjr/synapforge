"""End-to-end test for synapforge.modal -- ALL 9 modalities.

Covers Round 6A (text/image/audio/video) + Round 6B (screen/point_cloud/
time_series/graph/biosignal).

Tests:
  - 9 single-modality forward shape + grad-flow tests.
  - 1 mega-test: bs=4 batch with ALL 9 modalities present, end-to-end.
  - Forward step ms reported for each modality.

Run:
  CUDA_VISIBLE_DEVICES=1 python /workspace/synapforge/test_full_modality.py
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

import synapforge as sf
import synapforge.modal as sm
from synapforge.cells.liquid import LiquidCell
from synapforge.surrogate import PLIFCell
from synapforge.cells.synapse import SparseSynapse


# -------------------- model under test (reused HybridBlock) --------------------

class _RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class HybridBlock(sf.Module):
    def __init__(self, hidden, sparsity=0.95):
        super().__init__()
        self.ln1 = _RMSNorm(hidden)
        self.liquid = LiquidCell(hidden, hidden, init="hasani")
        self.plif = PLIFCell(hidden, tau_init=1.5, threshold_init=0.3,
                             surrogate="atan", reset="subtract")
        self.synapse = SparseSynapse(hidden, hidden, sparsity=sparsity, bias=False)
        self.gate = nn.Linear(hidden, hidden, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.01)
        self.ln2 = _RMSNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden, bias=False),
        )

    def forward(self, x):
        a = self.ln1(x)
        h = self.liquid(a)
        s, _ = self.plif.forward_seq(h)
        g = self.synapse(s) * torch.sigmoid(self.gate(s))
        x = x + g
        x = x + self.ffn(self.ln2(x))
        return x


class FullModalLNN(sf.Module):
    """ONE backbone, any subset of 9 modalities."""

    def __init__(self, hidden=128, vocab=8192):
        super().__init__()
        self.embed = sm.UnifiedEmbed(
            hidden=hidden,
            vocab=vocab,
            patch_image=8,
            patch_audio_ms=20,
            video_temporal_patch=4,
            patch_screen=32,
            pc_voxel_grid=4,           # 64 voxels for tests (faster)
            pc_feat_dim=6,
            ts_patch_t=8,
            ts_max_channels=16,
            graph_node_feat=16,
            graph_edge_feat=4,
            graph_rounds=2,
            graph_pool="set",
            bio_sample_rate=256,
            bio_win_ms=250,
            bio_hop_ms=125,
            bio_max_channels=16,
        )
        self.block = HybridBlock(hidden)
        self.lm_head = sf.tied_lm_head(hidden, vocab,
                                       embedding=self.embed.token_embedding)

    def forward(self, batch):
        z = self.embed(batch)
        h = self.block(z)
        logits = self.lm_head(h)
        return logits


# -------------------- data factories --------------------

def _mk_text(B, Tt, vocab=8192, device="cuda"):
    return torch.randint(0, vocab, (B, Tt), device=device, dtype=torch.long)


def _mk_image(B, H=64, W=64, device="cuda"):
    return torch.rand(B, 3, H, W, device=device)


def _mk_audio(B, samples=16000, device="cuda"):
    return torch.rand(B, samples, device=device) * 2 - 1


def _mk_video(B, Tf=8, H=32, W=32, device="cuda"):
    return torch.rand(B, Tf, 3, H, W, device=device)


def _mk_screen(B, H=256, W=384, device="cuda"):
    # Lower res than 1080p for faster test, divisible by 32 so no pad needed.
    return torch.rand(B, 3, H, W, device=device)


def _mk_point_cloud(B, N=256, device="cuda"):
    # xyz in [-1,1], rgb in [0,1]
    xyz = (torch.rand(B, N, 3, device=device) * 2 - 1)
    rgb = torch.rand(B, N, 3, device=device)
    return torch.cat([xyz, rgb], dim=-1)


def _mk_time_series(B, T_raw=128, C=8, device="cuda"):
    return torch.randn(B, T_raw, C, device=device)


def _mk_graph(B, N=16, E=32, F_n=16, F_e=4, device="cuda"):
    nodes = torch.randn(B, N, F_n, device=device)
    edges = torch.randint(0, N, (B, E, 2), device=device, dtype=torch.long)
    edge_feat = torch.randn(B, E, F_e, device=device)
    node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    edge_mask = torch.ones(B, E, dtype=torch.bool, device=device)
    return {
        "nodes": nodes, "edges": edges, "edge_feat": edge_feat,
        "node_mask": node_mask, "edge_mask": edge_mask,
    }


def _mk_biosignal(B, T_samples=512, C=8, device="cuda"):
    # Simulate noisy multi-channel EEG-like signal at 256Hz.
    return torch.randn(B, T_samples, C, device=device) * 0.1


def _now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _time_forward(model, batch, n_warmup=2, n_iters=3):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(batch)
        t0 = _now()
        for _ in range(n_iters):
            _ = model(batch)
        t1 = _now()
    return (t1 - t0) / n_iters * 1000.0


# -------------------- test runner --------------------

def run(device="cuda", bs=4, hidden=128, vocab=8192, dtype=torch.bfloat16):
    print(f"[setup] device={device} bs={bs} hidden={hidden} dtype={dtype}")
    model = FullModalLNN(hidden=hidden, vocab=vocab).to(device)
    if dtype == torch.bfloat16 and device == "cuda":
        model = model.to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] params={n_params/1e6:.2f} M")

    # Each entry: (name, factory, grad_attr_path)
    cases = [
        ("text",        lambda: {"text_tokens": _mk_text(bs, 32, vocab, device)},
                       "embed.token_embedding.weight"),
        ("image",       lambda: {"image": _mk_image(bs, 32, 32, device).to(dtype)},
                       "embed.image_embed.proj.weight"),
        ("audio",       lambda: {"audio": _mk_audio(bs, 8000, device).to(dtype)},
                       "embed.audio_embed.proj.weight"),
        ("video",       lambda: {"video": _mk_video(bs, 4, 16, 16, device).to(dtype)},
                       "embed.video_embed.proj.weight"),
        ("screen",      lambda: {"screen": _mk_screen(bs, 128, 192, device).to(dtype)},
                       "embed.screen_embed.proj.weight"),
        ("point_cloud", lambda: {"point_cloud": _mk_point_cloud(bs, 64, device).to(dtype)},
                       "embed.point_cloud_embed.proj.weight"),
        ("time_series", lambda: {"time_series": _mk_time_series(bs, 64, 8, device).to(dtype)},
                       "embed.time_series_embed.tconv.weight"),
        ("graph",       lambda: {"graph": _mk_graph(bs, 8, 16, 16, 4, device)},
                       "embed.graph_embed.node_proj.weight"),
        ("biosignal",   lambda: {"biosignal": _mk_biosignal(bs, 256, 8, device).to(dtype)},
                       "embed.biosignal_embed.mix.weight"),
    ]

    pass_count = 0
    fail_count = 0
    timings = {}
    fwd_steps = {}

    def _resolve(model, dotted: str):
        obj = model
        for p in dotted.split("."):
            obj = getattr(obj, p)
        return obj

    for name, mk, grad_path in cases:
        try:
            batch = mk()
            # 1) shape
            with torch.no_grad():
                logits = model(batch)
            assert logits.dim() == 3, f"{name}: expected 3D logits; got {logits.shape}"
            assert logits.shape[0] == bs, f"{name}: B mismatch {logits.shape}"
            assert logits.shape[2] == vocab, f"{name}: V mismatch {logits.shape}"
            T_actual = logits.shape[1]
            # 2) grad flow
            model.train()
            batch_g = mk()
            logits = model(batch_g)
            tgt = torch.randint(0, vocab, (bs, logits.shape[1]), device=device)
            loss = F.cross_entropy(logits.reshape(-1, vocab).float(),
                                    tgt.reshape(-1))
            loss.backward()
            tensor = _resolve(model, grad_path)
            g = tensor.grad
            assert g is not None, f"{name}: grad is None at {grad_path}"
            v = g.detach()
            if torch.isfinite(v).all().item():
                assert v.abs().sum().item() > 0, f"{name}: grad all-zero"
            model.zero_grad(set_to_none=True)

            # 3) timing
            ms = _time_forward(model, batch, n_warmup=1, n_iters=3)
            timings[name] = ms
            fwd_steps[name] = T_actual
            print(f"  [{name:12s}] PASS  T={T_actual:5d}  fwd={ms:7.2f} ms  loss={loss.item():.3f}")
            pass_count += 1
        except Exception as e:
            traceback.print_exc()
            print(f"  [{name:12s}] FAIL: {e}")
            fail_count += 1
            timings[name] = None
            fwd_steps[name] = None

    # ---- mega test: ALL 9 modalities at once on bs=4 ----
    print("\n[mega] ALL 9 modalities in one batch, bs={}".format(bs))
    try:
        mega = {
            "text_tokens":  _mk_text(bs, 32, vocab, device),
            "image":        _mk_image(bs, 32, 32, device).to(dtype),
            "audio":        _mk_audio(bs, 8000, device).to(dtype),
            "video":        _mk_video(bs, 4, 16, 16, device).to(dtype),
            "screen":       _mk_screen(bs, 128, 192, device).to(dtype),
            "point_cloud":  _mk_point_cloud(bs, 64, device).to(dtype),
            "time_series":  _mk_time_series(bs, 64, 8, device).to(dtype),
            "graph":        _mk_graph(bs, 8, 16, 16, 4, device),
            "biosignal":    _mk_biosignal(bs, 256, 8, device).to(dtype),
        }
        with torch.no_grad():
            logits = model(mega)
        assert logits.dim() == 3 and logits.shape[0] == bs
        T_mega = logits.shape[1]
        # full backward path
        model.train()
        logits = model(mega)
        tgt = torch.randint(0, vocab, (bs, logits.shape[1]), device=device)
        loss = F.cross_entropy(logits.reshape(-1, vocab).float(), tgt.reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        ms = _time_forward(model, mega, n_warmup=1, n_iters=3)
        print(f"  [mega-9-modal] PASS  T={T_mega}  fwd={ms:.2f} ms  loss={loss.item():.3f}")
        pass_count += 1
        timings["mega"] = ms
        fwd_steps["mega"] = T_mega
    except Exception as e:
        traceback.print_exc()
        print(f"  [mega-9-modal] FAIL: {e}")
        fail_count += 1
        timings["mega"] = None
        fwd_steps["mega"] = None

    # ---- summary ----
    total = len(cases) + 1
    print(f"\n[summary] PASS={pass_count}/{total} FAIL={fail_count}")
    print(f"[summary] forward ms by modality (bs={bs}, hidden={hidden}, dtype={dtype}):")
    for k in [c[0] for c in cases] + ["mega"]:
        v = timings.get(k)
        T = fwd_steps.get(k)
        if v is None:
            print(f"  {k:14s}  FAIL")
        else:
            print(f"  {k:14s}  T={T:5d}  {v:7.2f} ms")
    return pass_count == total, pass_count, total, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if args.fp32 or device == "cpu" else torch.bfloat16
    ok, p, n, t = run(device=device, bs=args.bs, hidden=args.hidden, dtype=dtype)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
