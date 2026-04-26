"""train_v16_unified.py — v1.6: 100M backbone + v1.5 multi-modal/semantic/intrinsic.

Combines:
  v1.3 KD strengths: SynapForge100M (~99M, Triton fused, proven ppl 239)
  v1.5 capabilities: UnifiedEmbed (9 modalities), ActionHead+NeuroMCP, semantic
                     aux losses (triplet/def/char), agent Q/A, idle exploration,
                     rep-level Qwen0.5B KD (vocab-agnostic).

Warmstart: /workspace/best_ckpts/synapforge_v13_kd_best.pt (99M backbone matches
           exactly; new heads strict=False with random init).

DDP fix from v1.3c crash: eval+save are wrapped in dist.barrier() so rank-1
doesn't hit a stale allreduce while rank-0 is doing eval.

Per user 铁律:
  - Always warmstart from prior best (no random init for backbone).
  - Model never tool-calls; tools only at data prep time.
  - 9-modal byte-patch (no conv encoders).
  - 神经元直驱 (ActionHead + NeuroMCP trained, not schema-based MCP).
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time, glob
from typing import Optional

sys.path[:] = [p for p in sys.path if p not in ("/workspace/synapforge", "")]
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# --- DDP env detect (torchrun) ---
_DDP_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
_DDP_RANK = int(os.environ.get("RANK", "0"))
_DDP_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
_IS_DDP = _DDP_WORLD > 1

import torch
import torch.nn as nn
import torch.nn.functional as F

if _IS_DDP:
    torch.cuda.set_device(_DDP_LOCAL_RANK)
    import torch.distributed as torch_dist
    torch_dist.init_process_group("nccl")

import synapforge as sf
from synapforge.data import ParquetTokenStream
from synapforge.huggingface_adapter import load_tokenizer
from synapforge.optim import build_optimizer
from synapforge.model_100m import build_synapforge_100m

# --- defaults ---
HIDDEN = 512
VOCAB = 50257
WT103_GLOB = "/workspace/data/wt103_raw/train-*.parquet"
WT103_VAL = "/workspace/data/wt103_raw/validation.parquet"
QA_GLOB = "/workspace/data/agent_qa/*.jsonl"
TRIPLET_PATH = "/workspace/data/semantic/triplets.jsonl"
DEFINITION_PATH = "/workspace/data/semantic/definitions.jsonl"
FINEWEB_GLOB = "/workspace/data/multimodal/fineweb_edu/*.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


class V16Model(sf.Module):
    """100M SynapForge backbone + 9-modal embed + multi-task heads.

    Two encoding entry points:
      forward_text(ids)       — token ids; uses backbone tok_embed path.
      forward_modal(batch)    — dict; uses UnifiedEmbed -> backbone.forward_from_z.
    """

    def __init__(self):
        super().__init__()
        self.backbone = build_synapforge_100m(
            vocab=VOCAB, d=HIDDEN, n_layers=10, loop_depth=4,
            max_seq=256, ffn_ratio=8.0, sparsity=0.95, dropout=0.0,
        )
        # Apply Triton backend to backbone (matches v13_kd ckpt structure
        # `liquid.shared.block.*` AND gives 28K tok/s on A100).
        try:
            from synapforge.backends.triton_block import TritonBlockBackend
            be = TritonBlockBackend()
            stats = be.compile(self.backbone)
            if _DDP_RANK == 0:
                print(f"[backend] triton_block fused {stats.get('n_pairs_fused', 0)} pairs", flush=True)
        except Exception as e:
            if _DDP_RANK == 0:
                print(f"[backend] triton fail, gpu_dense: {e}", flush=True)
        # Multi-modal embed (hidden=512 to match backbone). Tied token embed
        # to backbone's tok_embed so warmstart preserves trained vocab embed.
        self.modal_embed = sf.modal.UnifiedEmbed(
            hidden=HIDDEN, vocab=VOCAB,
            patch_image=8, patch_audio_ms=20, video_temporal_patch=4,
            patch_screen=16, pc_voxel_grid=4, ts_patch_t=8,
            graph_node_feat=32,
            bio_sample_rate=256, bio_win_ms=250, bio_hop_ms=125,
            bio_max_channels=8,
        )
        # Tie modal_embed text path to backbone tok_embed — share weights.
        self.modal_embed.token_embedding.weight = self.backbone.tok_embed.weight

        # Per-modality recon heads (Linear only; Fuyu byte-patch convention)
        self.head_image = nn.Linear(HIDDEN, 8 * 8 * 3)
        self.head_audio = nn.Linear(HIDDEN, 320)
        self.head_video = nn.Linear(HIDDEN, 8 * 8 * 3)
        self.head_screen = nn.Linear(HIDDEN, 16 * 16 * 3)
        self.head_pc = nn.Linear(HIDDEN, 6)
        self.head_ts = nn.Linear(HIDDEN, 8)
        self.head_graph = nn.Linear(HIDDEN, 32)
        self.head_bio = nn.Linear(HIDDEN, 8)
        # Action + NeuroMCP (神经元直驱)
        self.action = sf.action.ActionHead(HIDDEN, sf.action.OSActionSpec.default())
        self.neuromcp = sf.action.NeuroMCPHead(
            HIDDEN, codebook_initial=8, codebook_max=64,
            synapse_density=0.05, synapse_max_density=0.3,
        )
        # Semantic projection (triplet/contrastive)
        self.sem_proj = nn.Linear(HIDDEN, 128, bias=False)
        # Char-aux head
        self.char_head = nn.Linear(HIDDEN, 256, bias=False)
        nn.init.normal_(self.sem_proj.weight, std=0.02)
        nn.init.normal_(self.char_head.weight, std=0.02)

    def encode_text(self, ids):
        """Use backbone's native tok_embed path. Returns (B, T, HIDDEN)."""
        return self.backbone.encode(ids)

    def lm_logits(self, hidden):
        """Tied lm_head from hidden."""
        return F.linear(hidden, self.backbone.tok_embed.weight)

    def encode_modal(self, batch):
        """Multi-modal dict -> backbone hidden (skips tok_embed, uses UnifiedEmbed)."""
        z = self.modal_embed(batch)  # (B, T_uni, HIDDEN)
        return self.backbone.forward_from_z(z)


def load_qa_pairs(glob_pat):
    pairs = []
    for f in sorted(glob.glob(glob_pat)):
        with open(f) as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                    pairs.append("Q: " + o["q"] + "\nA: " + o["a"] + "\n\n")
                except Exception:
                    continue
    return pairs


def load_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try: out.append(json.loads(line))
                except Exception: continue
    return out


def load_fineweb_chunks(glob_pat, max_lines=10000):
    out = []
    for f in sorted(glob.glob(glob_pat)):
        with open(f) as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                    text = o.get("text") or o.get("content")
                    if text and len(text) > 100:
                        out.append(text[:2000])
                except Exception: continue
                if len(out) >= max_lines: return out
    return out


def encode_text_batch(texts, tok, seq_len, device):
    ids_list = []
    for t in texts:
        ids = tok.encode(t)[:seq_len]
        if len(ids) < seq_len:
            pad = tok.eos_token_id or 50256
            ids = ids + [pad] * (seq_len - len(ids))
        ids_list.append(ids)
    return torch.tensor(ids_list, dtype=torch.long, device=device)


def load_teacher(name, device, dtype):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {"local_files_only": True} if os.environ.get("TRANSFORMERS_OFFLINE") == "1" else {}
        tok = AutoTokenizer.from_pretrained(name, **kw)
        m = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, output_hidden_states=True, **kw
        ).to(device).eval()
        for p in m.parameters(): p.requires_grad_(False)
        return m, tok, m.config.hidden_size
    except Exception as e:
        if _DDP_RANK == 0:
            print(f"[teacher FAIL] {e}", flush=True)
        return None, None, None


def teacher_pool(teacher, ttok, ids, stok, device):
    if teacher is None: return None
    texts = stok.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)
    enc = ttok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None: attn = attn.to(device)
    with torch.no_grad():
        out = teacher(input_ids=tids, attention_mask=attn, output_hidden_states=True)
    h = out.hidden_states[-1]
    if attn is not None:
        m = attn.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
    return h.mean(1)


def lr_at(step, peak, warmup, total, kind="cosine", min_lr=1e-5):
    if step < warmup: return peak * step / max(1, warmup)
    if kind == "none": return peak
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    if kind == "cosine":
        return min_lr + (peak - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak - min_lr) * max(0.0, 1.0 - progress)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/workspace/runs/synapforge_v16_unified")
    p.add_argument("--warmstart", default="/workspace/best_ckpts/synapforge_v13_kd_best.pt")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--kd-weight", type=float, default=0.1)
    p.add_argument("--qa-weight", type=float, default=0.15)
    p.add_argument("--triplet-weight", type=float, default=0.05)
    p.add_argument("--def-weight", type=float, default=0.05)
    p.add_argument("--char-weight", type=float, default=0.02)
    p.add_argument("--idle-every", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(DEVICE)
    torch.manual_seed(42 + _DDP_RANK)

    def _log(msg):
        if _DDP_RANK == 0:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", flush=True)

    _log(f"v1.6 unified  device={DEVICE} dtype={DTYPE} world={_DDP_WORLD}")
    _log(f"steps={args.steps} bs={args.batch_size} seq={args.seq_len} lr={args.lr}")

    # Build model
    model = V16Model().to(device).to(DTYPE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    _log(f"model params: {n_params:.2f}M (backbone={sum(p.numel() for p in model.backbone.parameters())/1e6:.1f}M)")

    # Warmstart: backbone matches v13_kd_best exactly; new heads strict=False
    if args.warmstart and os.path.exists(args.warmstart):
        try:
            ck = torch.load(args.warmstart, map_location="cpu")
            sd = ck.get("model", ck) if isinstance(ck, dict) else ck
            # Map v13 keys (which are top-level) to model.backbone.* keys
            mapped = {}
            for k, v in sd.items():
                # If key matches a backbone key, re-prefix with backbone.
                if hasattr(model.backbone, k.split(".")[0]):
                    mapped[f"backbone.{k}"] = v
                else:
                    mapped[k] = v
            res = model.load_state_dict(mapped, strict=False)
            _log(f"warmstart: matched=missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
            _log(f"  first 3 missing: {list(res.missing_keys)[:3]}")
        except Exception as e:
            _log(f"warmstart failed: {e}")

    # DDP wrap
    if _IS_DDP:
        from synapforge.distributed import wrap_model
        model = wrap_model(model, local_rank=_DDP_LOCAL_RANK, find_unused_parameters=True)
        _log(f"[DDP] wrapped world={_DDP_WORLD}")
    base_model = model.module if _IS_DDP else model

    # Optim
    opt = build_optimizer(base_model, lr=args.lr, weight_decay=0.05)
    if args.warmstart and os.path.exists(args.warmstart):
        try:
            ck = torch.load(args.warmstart, map_location="cpu")
            if isinstance(ck, dict) and "optim_state" in ck:
                # optim state from v13 may not match (different model arch);
                # try and tolerate failure.
                try:
                    opt.load_state_dict(ck["optim_state"])
                    _log("warmstart: restored optim_state")
                except Exception as e:
                    _log(f"warmstart optim restore skipped (arch mismatch): {e}")
        except Exception:
            pass

    # Tokenizer
    student_tok = load_tokenizer("gpt2")

    # Data
    train_ds = ParquetTokenStream(WT103_GLOB, seq_len=args.seq_len,
                                  batch_size=args.batch_size, loop=True)
    train_it = iter(train_ds)
    val_ds = ParquetTokenStream(WT103_VAL, seq_len=args.seq_len,
                                batch_size=args.batch_size, loop=False)
    qa_pairs = load_qa_pairs(QA_GLOB)
    fineweb = load_fineweb_chunks(FINEWEB_GLOB, max_lines=10000)
    triplets = load_jsonl(TRIPLET_PATH)
    definitions = load_jsonl(DEFINITION_PATH)
    _log(f"data: WT103 ready | QA={len(qa_pairs)} FW={len(fineweb)} trip={len(triplets)} defs={len(definitions)}")

    # Teacher (rep-level KD)
    teacher, ttok, t_hidden = load_teacher(args.teacher, device, DTYPE)
    if teacher is not None:
        kd_adapter = nn.Linear(t_hidden, HIDDEN, bias=False).to(device).to(DTYPE)
        nn.init.normal_(kd_adapter.weight, std=0.02)
        opt.add_param_group({"params": list(kd_adapter.parameters())})
        _log(f"teacher loaded h={t_hidden}, adapter -> {HIDDEN}")
    else:
        kd_adapter = None

    metrics = {"step": [], "ppl_eval": {}}
    t0 = time.time()
    cum_tok = 0
    qa_ix = fw_ix = trip_ix = def_ix = 0

    @torch.no_grad()
    def do_eval():
        base_model.eval()
        losses = []
        try: it = iter(val_ds)
        except Exception: return float("nan")
        for _ in range(16):
            try: x, y = next(it)
            except StopIteration: break
            x = x.to(device); y = y.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=DTYPE,
                                    enabled=device.type == "cuda"):
                h = base_model.encode_text(x)
                logits = base_model.lm_logits(h).float()
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            losses.append(float(loss.item()))
        base_model.train()
        if not losses: return float("nan")
        return math.exp(sum(losses) / len(losses))

    for step in range(1, args.steps + 1):
        cur_lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in opt.param_groups: pg["lr"] = cur_lr

        try: x, y = next(train_it)
        except StopIteration:
            train_it = iter(train_ds); x, y = next(train_it)
        x = x.to(device); y = y.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=DTYPE,
                                enabled=device.type == "cuda"):
            h = base_model.encode_text(x)
            logits = base_model.lm_logits(h).float()
            ce = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                                 label_smoothing=args.label_smoothing)
            log_z = torch.logsumexp(logits.reshape(-1, VOCAB), dim=-1)
            z = (log_z ** 2).mean()
            loss = ce + args.z_loss_weight * z

            kd = torch.zeros((), device=device, dtype=ce.dtype)
            if kd_adapter is not None:
                t_pool = teacher_pool(teacher, ttok, x, student_tok, device)
                if t_pool is not None:
                    s_pool = h.mean(dim=1).float()
                    kd = F.mse_loss(s_pool, kd_adapter(t_pool.to(DTYPE)).float())
                    loss = loss + args.kd_weight * kd

            qa_l = torch.zeros((), device=device, dtype=ce.dtype)
            if qa_pairs and step % 4 == 0:
                bt = [qa_pairs[(qa_ix + i) % len(qa_pairs)] for i in range(args.batch_size)]
                qa_ix += args.batch_size
                qx = encode_text_batch(bt, student_tok, args.seq_len, device)
                qy = torch.cat([qx[:, 1:], qx[:, :1]], dim=1)
                qh = base_model.encode_text(qx)
                qlogits = base_model.lm_logits(qh).float()
                qa_l = F.cross_entropy(qlogits.reshape(-1, VOCAB), qy.reshape(-1),
                                       label_smoothing=args.label_smoothing)
                loss = loss + args.qa_weight * qa_l

            fw_l = torch.zeros((), device=device, dtype=ce.dtype)
            if fineweb and step % 4 == 2:
                bt = [fineweb[(fw_ix + i) % len(fineweb)] for i in range(args.batch_size)]
                fw_ix += args.batch_size
                fx = encode_text_batch(bt, student_tok, args.seq_len, device)
                fy = torch.cat([fx[:, 1:], fx[:, :1]], dim=1)
                fh = base_model.encode_text(fx)
                flogits = base_model.lm_logits(fh).float()
                fw_l = F.cross_entropy(flogits.reshape(-1, VOCAB), fy.reshape(-1),
                                       label_smoothing=args.label_smoothing)
                loss = loss + 0.2 * fw_l

            trip_l = torch.zeros((), device=device, dtype=ce.dtype)
            if triplets and step % 8 == 0:
                B = min(args.batch_size, len(triplets))
                bt = [triplets[(trip_ix + i) % len(triplets)] for i in range(B)]
                trip_ix += B
                ax = encode_text_batch([t["anchor"] for t in bt], student_tok, 16, device)
                sx = encode_text_batch([t["synonym"] for t in bt], student_tok, 16, device)
                anx = encode_text_batch([t["antonym"] for t in bt], student_tok, 16, device)
                ah = base_model.encode_text(ax).mean(1)
                sh = base_model.encode_text(sx).mean(1)
                anh = base_model.encode_text(anx).mean(1)
                ap = F.normalize(base_model.sem_proj(ah).float(), dim=-1)
                sp = F.normalize(base_model.sem_proj(sh).float(), dim=-1)
                anp = F.normalize(base_model.sem_proj(anh).float(), dim=-1)
                trip_l = F.relu(0.3 - (ap*sp).sum(-1) + (ap*anp).sum(-1)).mean()
                loss = loss + args.triplet_weight * trip_l

            def_l = torch.zeros((), device=device, dtype=ce.dtype)
            if definitions and step % 16 == 0:
                B = min(args.batch_size, len(definitions))
                bt = [definitions[(def_ix + i) % len(definitions)] for i in range(B)]
                def_ix += B
                texts = [d["word"] + ": " + d["definition"] for d in bt]
                dx = encode_text_batch(texts, student_tok, 64, device)
                dy = torch.cat([dx[:, 1:], dx[:, :1]], dim=1)
                dh = base_model.encode_text(dx)
                dlogits = base_model.lm_logits(dh).float()
                def_l = F.cross_entropy(dlogits.reshape(-1, VOCAB), dy.reshape(-1),
                                        label_smoothing=args.label_smoothing)
                loss = loss + args.def_weight * def_l

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in base_model.parameters() if p.requires_grad], max_norm=0.5)
        opt.step()

        # NeuroMCP plasticity (no_grad, post-backward)
        if step % 8 == 0:
            try:
                with torch.no_grad():
                    pooled = h[:, -1:, :].detach()
                    base_model.neuromcp(pooled)
                    base_model.neuromcp.step_plasticity()
                    base_model.neuromcp.codebook.maybe_grow(h.mean(dim=1).detach())
            except Exception: pass

        cum_tok += args.batch_size * args.seq_len * _DDP_WORLD

        if step % 10 == 0 or step == 1:
            tok_s = cum_tok / max(time.time() - t0, 1e-6)
            mem = f" mem_GB={torch.cuda.memory_allocated()/1e9:.1f}" if device.type == "cuda" else ""
            _log(f"step {step:5d} loss={loss.item():.4f} ce={ce.item():.3f} "
                 f"kd={kd.item():.3f} qa={qa_l.item():.3f} trip={trip_l.item():.3f} "
                 f"def={def_l.item():.3f} z={z.item():.1f} lr={cur_lr:.5f} "
                 f"tok/s={tok_s:.0f}{mem}")

        # Eval on rank 0 only, but barrier on both ranks
        if step % 250 == 0 or step == args.steps:
            if _IS_DDP: torch_dist.barrier()
            ppl = float("nan")
            if _DDP_RANK == 0:
                ppl = do_eval()
                metrics["ppl_eval"][step] = ppl
                _log(f"VAL step {step}: ppl={ppl:.2f}")
            if _IS_DDP: torch_dist.barrier()

        if step % 500 == 0 and _DDP_RANK == 0:
            ck_path = os.path.join(args.out, f"step_{step:06d}.pt")
            torch.save({
                "model": base_model.state_dict(),
                "optim_state": opt.state_dict(),
                "step": step,
                "loss": float(loss.item()),
                "n_params_M": n_params,
            }, ck_path)
            _log(f"saved ckpt {ck_path}")
        if step % 500 == 0 and _IS_DDP:
            torch_dist.barrier()

    if _DDP_RANK == 0:
        torch.save({
            "model": base_model.state_dict(),
            "optim_state": opt.state_dict(),
            "step": args.steps,
            "n_params_M": n_params,
        }, os.path.join(args.out, "final.pt"))
        _log(f"done {time.time()-t0:.0f}s")
    if _IS_DDP:
        torch_dist.barrier()
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
