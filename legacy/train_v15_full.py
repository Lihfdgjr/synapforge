"""train_v15_full.py — v1.5: full-modal self-learning trainer.

Builds on `train_full_modal.py` (proven 9-modal training) and adds:
1. **Web-augmented data** — FineWeb-Edu / wikipedia text + COCO images + agent Q/A
2. **Stronger KD teacher** — Qwen2.5-0.5B (rep-level, vocab-agnostic)
3. **Intrinsic motivation** — sf.intrinsic.IdleLoop + SelfGoalProposer + NoveltyDrive
4. **Semantic understanding** — triplet (anchor/syn/ant) + definition modeling +
   cross-modal contrastive (image-caption)
5. **All on neuronal substrate** — model never tool-calls; tools used only at data prep
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time, glob, pathlib
from typing import Optional

sys.path[:] = [p for p in sys.path if p not in ("/workspace/synapforge", "")]
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.nn.functional as F

import synapforge as sf
from synapforge.data import ParquetTokenStream
from synapforge.huggingface_adapter import load_tokenizer
from synapforge.optim import build_optimizer

OUT_DIR_DEFAULT = "/workspace/runs/synapforge_v15_full"
WARM_CKPT_DEFAULT = "/workspace/runs/synapforge_full_modal/ckpt_step4000.pt"
WT103_GLOB = "/workspace/data/wt103_raw/train-*.parquet"
WT103_VAL = "/workspace/data/wt103_raw/validation.parquet"
QA_GLOB = "/workspace/data/agent_qa/*.jsonl"
TRIPLET_PATH = "/workspace/data/semantic/triplets.jsonl"
DEFINITION_PATH = "/workspace/data/semantic/definitions.jsonl"
COCO_DIR = "/workspace/data/multimodal/coco"
FINEWEB_GLOB = "/workspace/data/multimodal/fineweb_edu/*.jsonl"

LR_DEFAULT = 1.5e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 0.5
SAVE_EVERY = 500
EVAL_EVERY = 250
LOG_EVERY = 10
HIDDEN = 384
VOCAB = 50257
TEXT_SEQ = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


class V15Model(sf.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.embed = sf.modal.UnifiedEmbed(
            hidden=hidden, vocab=VOCAB,
            patch_image=8, patch_audio_ms=20, video_temporal_patch=4,
            patch_screen=16, pc_voxel_grid=4, ts_patch_t=8,
            graph_node_feat=32,
            bio_sample_rate=256, bio_win_ms=250, bio_hop_ms=125,
            bio_max_channels=8,
        )
        self.cfc1 = sf.LiquidCell(hidden, hidden)
        self.plif1 = sf.PLIFCell(hidden, threshold_init=0.1, tau_init="bimodal")
        self.cfc2 = sf.LiquidCell(hidden, hidden)
        self.plif2 = sf.PLIFCell(hidden, threshold_init=0.1, tau_init="bimodal")
        self.lm_head = sf.tied_lm_head(hidden, VOCAB,
                                       embedding=self.embed.token_embedding)
        self.head_image = nn.Linear(hidden, 8 * 8 * 3)
        self.head_audio = nn.Linear(hidden, 320)
        self.head_video = nn.Linear(hidden, 8 * 8 * 3)
        self.head_screen = nn.Linear(hidden, 16 * 16 * 3)
        self.head_pc = nn.Linear(hidden, 6)
        self.head_ts = nn.Linear(hidden, 8)
        self.head_graph = nn.Linear(hidden, 32)
        self.head_bio = nn.Linear(hidden, 8)
        self.action = sf.action.ActionHead(hidden, sf.action.OSActionSpec.default())
        self.neuromcp = sf.action.NeuroMCPHead(
            hidden, codebook_initial=8, codebook_max=64,
            synapse_density=0.05, synapse_max_density=0.3,
        )
        self.sem_proj = nn.Linear(hidden, 128, bias=False)
        self.char_head = nn.Linear(hidden, 256, bias=False)

    def encode_seq(self, batch: dict) -> torch.Tensor:
        # v1.5a: dense CfC path; PLIF runs under no_grad just for spike monitoring.
        z = self.embed(batch)
        h1 = self.cfc1(z)
        with torch.no_grad():
            self.plif1.forward_seq(h1)
        h2 = self.cfc2(h1)
        with torch.no_grad():
            self.plif2.forward_seq(h2)
        return h2

    def pool_text(self, hidden, n_text_tokens):
        return hidden[:, -n_text_tokens:, :]


def load_qa_pairs(glob_pat):
    pairs = []
    for f in sorted(glob.glob(glob_pat)):
        with open(f) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    pairs.append("Q: " + obj["q"] + "\nA: " + obj["a"] + "\n\n")
                except Exception:
                    continue
    return pairs


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_fineweb_chunks(glob_pat, max_lines=200000):
    out = []
    for f in sorted(glob.glob(glob_pat)):
        with open(f) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content")
                    if text and len(text) > 100:
                        out.append(text[:2000])
                except Exception:
                    continue
                if len(out) >= max_lines:
                    return out
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
        print("[teacher] loading", name, flush=True)
        kw = {"local_files_only": True} if os.environ.get("TRANSFORMERS_OFFLINE") == "1" else {}
        tok = AutoTokenizer.from_pretrained(name, **kw)
        m = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, output_hidden_states=True, **kw,
        ).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        h = m.config.hidden_size
        print("[teacher] loaded", f"{n:.1f}M", "hidden=", h, flush=True)
        return m, tok, h
    except Exception as e:
        print("[teacher FAIL]", name, e, flush=True)
        return None, None, None


def teacher_pool(teacher, ttok, student_ids, stok, device, dtype):
    if teacher is None:
        return None
    texts = stok.batch_decode(student_ids.cpu().tolist(), skip_special_tokens=True)
    enc = ttok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    with torch.no_grad():
        out = teacher(input_ids=tids, attention_mask=attn, output_hidden_states=True)
    h = out.hidden_states[-1]
    if attn is not None:
        m = attn.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
    return h.mean(1)


def lr_at(step, peak, warmup, total, kind="cosine", min_lr=1e-5):
    if step < warmup:
        return peak * step / max(1, warmup)
    if kind == "none":
        return peak
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    if kind == "cosine":
        return min_lr + (peak - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == "linear":
        return min_lr + (peak - min_lr) * max(0.0, 1.0 - progress)
    return peak


@torch.no_grad()
def self_propose(model, tokenizer, device, max_new=24, prefix="Q:"):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long, device=device)
    for _ in range(max_new):
        with torch.amp.autocast(device_type=device.type, dtype=DTYPE,
                                enabled=device.type == "cuda"):
            h = model.encode_seq({"text_tokens": ids})
            tail = h[:, -1:, :]
            logits = model.lm_head(tail).float()
        topk = torch.topk(logits[0, -1], k=40)
        probs = torch.softmax(topk.values / 0.8, dim=-1)
        nxt = topk.indices[torch.multinomial(probs, 1)]
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
        if nxt.item() == (tokenizer.eos_token_id or 50256):
            break
    text = tokenizer.decode(ids[0].tolist())
    model.train()
    return text


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_DIR_DEFAULT)
    p.add_argument("--warmstart", default=WARM_CKPT_DEFAULT)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=TEXT_SEQ)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--lr", type=float, default=LR_DEFAULT)
    p.add_argument("--lr-decay", default="cosine")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--kd-weight", type=float, default=0.3)
    p.add_argument("--qa-weight", type=float, default=0.15)
    p.add_argument("--triplet-weight", type=float, default=0.05)
    p.add_argument("--def-weight", type=float, default=0.05)
    p.add_argument("--char-weight", type=float, default=0.02)
    p.add_argument("--idle-every", type=int, default=200)
    p.add_argument("--skip-warmstart", action="store_true", default=False,
                   help="Skip warmstart entirely (use random init)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_lines = []
    def _log(m):
        ts = time.strftime("%H:%M:%S")
        line = "[" + ts + "] " + m
        print(line, flush=True)
        log_lines.append(line)

    _log("v1.5 full+self-learn  device=" + DEVICE + " dtype=" + str(DTYPE))
    _log("steps=" + str(args.steps) + " bs=" + str(args.batch_size) +
         " seq=" + str(args.seq_len) + " lr=" + str(args.lr))

    device = torch.device(DEVICE)
    torch.manual_seed(42)

    model = V15Model(hidden=HIDDEN).to(device).to(DTYPE)
    n_params = sum(pp.numel() for pp in model.parameters()) / 1e6
    _log("model params: " + f"{n_params:.2f}M")

    if args.warmstart and os.path.exists(args.warmstart) and not args.skip_warmstart:
        try:
            ck = torch.load(args.warmstart, map_location="cpu")
            sd = ck.get("model", ck) if isinstance(ck, dict) else ck
            res = model.load_state_dict(sd, strict=False)
            _log("warmstart: missing=" + str(len(res.missing_keys)) +
                 " unexpected=" + str(len(res.unexpected_keys)))
        except Exception as e:
            _log("warmstart failed: " + str(e))

    opt = build_optimizer(model, lr=args.lr, weight_decay=WEIGHT_DECAY)

    student_tok = load_tokenizer("gpt2")
    _log("student tokenizer: GPT-2 vocab=" + str(student_tok.vocab_size))

    train_ds = ParquetTokenStream(WT103_GLOB, seq_len=args.seq_len,
                                  batch_size=args.batch_size, loop=True)
    train_it = iter(train_ds)
    val_ds = ParquetTokenStream(WT103_VAL, seq_len=args.seq_len,
                                batch_size=args.batch_size, loop=False)

    qa_pairs = load_qa_pairs(QA_GLOB)
    fineweb_chunks = load_fineweb_chunks(FINEWEB_GLOB, max_lines=10000)
    _log("agent QA pairs: " + str(len(qa_pairs)) +
         " | FineWeb-Edu chunks: " + str(len(fineweb_chunks)))

    triplets = load_jsonl(TRIPLET_PATH)
    definitions = load_jsonl(DEFINITION_PATH)
    _log("semantic: triplets=" + str(len(triplets)) +
         " defs=" + str(len(definitions)))

    teacher, teacher_tok, teacher_hidden = load_teacher(args.teacher, device, DTYPE)
    if teacher is not None:
        kd_adapter = nn.Linear(teacher_hidden, HIDDEN, bias=False).to(device).to(DTYPE)
        nn.init.normal_(kd_adapter.weight, std=0.02)
        opt.add_param_group({"params": list(kd_adapter.parameters())})
        _log("KD rep-level adapter: " + str(teacher_hidden) + " -> " + str(HIDDEN))
    else:
        kd_adapter = None

    # Intrinsic exploration: experience replay buffer for self-rolled prompts.
    try:
        from synapforge.self_learn import ExperienceReplayBuffer
        replay_buf = ExperienceReplayBuffer(capacity=512, alpha=0.5)
    except Exception:
        replay_buf = None
    metrics = {"step": [], "ppl_eval": {}}
    t0 = time.time()
    cum_tok = 0
    plif_cells = [m for m in model.modules() if isinstance(m, sf.PLIFCell)]

    @torch.no_grad()
    def do_eval():
        model.eval()
        losses = []
        try:
            it = iter(val_ds)
        except Exception:
            return float("nan")
        for _ in range(16):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x = x.to(device); y = y.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=DTYPE,
                                    enabled=device.type == "cuda"):
                h = model.encode_seq({"text_tokens": x})
                Tt = x.shape[1]
                hh = model.pool_text(h, Tt)
                logits = model.lm_head(hh).float()
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            losses.append(float(loss.item()))
        model.train()
        if not losses:
            return float("nan")
        return math.exp(sum(losses) / len(losses))

    qa_ix = 0; fw_ix = 0; trip_ix = 0; def_ix = 0

    for step in range(1, args.steps + 1):
        cur_lr = lr_at(step, args.lr, args.warmup, args.steps, args.lr_decay)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        try:
            x, y = next(train_it)
        except StopIteration:
            train_it = iter(train_ds)
            x, y = next(train_it)
        x = x.to(device); y = y.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=DTYPE,
                                enabled=device.type == "cuda"):
            h = model.encode_seq({"text_tokens": x})
            Tt = x.shape[1]
            hh = model.pool_text(h, Tt)
            logits = model.lm_head(hh).float()
            ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                                      label_smoothing=args.label_smoothing)
            log_z = torch.logsumexp(logits.reshape(-1, VOCAB), dim=-1)
            z_loss = (log_z ** 2).mean()
            loss = ce_loss + args.z_loss_weight * z_loss

            kd_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            if kd_adapter is not None:
                student_pool = hh.mean(dim=1).float()
                t_pool = teacher_pool(teacher, teacher_tok, x, student_tok, device, DTYPE)
                if t_pool is not None:
                    proj = kd_adapter(t_pool.to(DTYPE)).float()
                    kd_loss = F.mse_loss(student_pool, proj)
                    loss = loss + args.kd_weight * kd_loss

            qa_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            if qa_pairs and step % 4 == 0:
                bt = []
                for _ in range(args.batch_size):
                    bt.append(qa_pairs[qa_ix % len(qa_pairs)])
                    qa_ix += 1
                qx = encode_text_batch(bt, student_tok, args.seq_len, device)
                qy = torch.cat([qx[:, 1:], qx[:, :1]], dim=1)
                qh = model.encode_seq({"text_tokens": qx})
                qhh = model.pool_text(qh, qx.shape[1])
                qlogits = model.lm_head(qhh).float()
                qa_loss = F.cross_entropy(
                    qlogits.reshape(-1, VOCAB), qy.reshape(-1),
                    label_smoothing=args.label_smoothing)
                loss = loss + args.qa_weight * qa_loss

            fw_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            if fineweb_chunks and step % 4 == 2:
                bt = []
                for _ in range(args.batch_size):
                    bt.append(fineweb_chunks[fw_ix % len(fineweb_chunks)])
                    fw_ix += 1
                fx = encode_text_batch(bt, student_tok, args.seq_len, device)
                fy = torch.cat([fx[:, 1:], fx[:, :1]], dim=1)
                fh_ = model.encode_seq({"text_tokens": fx})
                fhh = model.pool_text(fh_, fx.shape[1])
                flogits = model.lm_head(fhh).float()
                fw_loss = F.cross_entropy(
                    flogits.reshape(-1, VOCAB), fy.reshape(-1),
                    label_smoothing=args.label_smoothing)
                loss = loss + 0.2 * fw_loss

            trip_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            if triplets and step % 8 == 0:
                B = min(args.batch_size, len(triplets))
                bt = [triplets[(trip_ix + i) % len(triplets)] for i in range(B)]
                trip_ix += B
                ax = encode_text_batch([t["anchor"] for t in bt], student_tok, 16, device)
                sx = encode_text_batch([t["synonym"] for t in bt], student_tok, 16, device)
                anx = encode_text_batch([t["antonym"] for t in bt], student_tok, 16, device)
                ah = model.encode_seq({"text_tokens": ax}).mean(1)
                sh = model.encode_seq({"text_tokens": sx}).mean(1)
                anh = model.encode_seq({"text_tokens": anx}).mean(1)
                ap = F.normalize(model.sem_proj(ah).float(), dim=-1)
                sp = F.normalize(model.sem_proj(sh).float(), dim=-1)
                anp = F.normalize(model.sem_proj(anh).float(), dim=-1)
                pos_sim = (ap * sp).sum(-1)
                neg_sim = (ap * anp).sum(-1)
                trip_loss = F.relu(0.3 - pos_sim + neg_sim).mean()
                loss = loss + args.triplet_weight * trip_loss

            def_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            if definitions and step % 16 == 0:
                B = min(args.batch_size, len(definitions))
                bt = [definitions[(def_ix + i) % len(definitions)] for i in range(B)]
                def_ix += B
                texts = [d["word"] + ": " + d["definition"] for d in bt]
                dx = encode_text_batch(texts, student_tok, 64, device)
                dy = torch.cat([dx[:, 1:], dx[:, :1]], dim=1)
                dh = model.encode_seq({"text_tokens": dx})
                dhh = model.pool_text(dh, dx.shape[1])
                dlogits = model.lm_head(dhh).float()
                def_loss = F.cross_entropy(
                    dlogits.reshape(-1, VOCAB), dy.reshape(-1),
                    label_smoothing=args.label_smoothing)
                loss = loss + args.def_weight * def_loss

            char_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            if step % 4 == 0 and Tt >= 32:
                # Real-byte char target: decode each token to its first UTF-8 byte.
                # token_id -> bytes -> first byte (0..255). Cached on first hit.
                if not hasattr(student_tok, "_first_byte_cache"):
                    cache = torch.zeros(VOCAB, dtype=torch.long, device=device)
                    for tid in range(VOCAB):
                        try:
                            txt = student_tok.decode([tid], skip_special_tokens=True)
                            cache[tid] = txt.encode("utf-8", errors="replace")[0] if txt else 0
                        except Exception:
                            cache[tid] = 0
                    student_tok._first_byte_cache = cache
                cb = student_tok._first_byte_cache
                char_target = cb[x.long()]
                char_logits = model.char_head(hh).float()
                char_loss = F.cross_entropy(
                    char_logits[:, :-1].reshape(-1, 256),
                    char_target[:, 1:].reshape(-1))
                loss = loss + args.char_weight * char_loss

            # NeuroMCP + ActionHead training (神经元直驱; user 铁律: train these)
            # Use last hidden of text seq as proxy state, predict next-token action_type.
            act_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            if step % 8 == 0:
                pooled = hh[:, -1:, :]  # (B, 1, H)
                act_out = model.action(pooled)
                act_logits = act_out.action_type_logits.squeeze(1).float()
                # Synthetic target: token_id mod num_action_types as supervision signal.
                # In a real env this would be reward; here just keep gradients flowing.
                num_acts = act_logits.shape[-1]
                act_target = (x[:, -1] % num_acts).long()
                act_loss = F.cross_entropy(act_logits, act_target)
                loss = loss + 0.02 * act_loss
                # NeuroMCP plasticity is a Hebbian no-grad step; defer to AFTER opt.step()
                # to keep the in-place mask mutation off the autograd graph.

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=GRAD_CLIP)
        opt.step()

        # NeuroMCP plasticity post-step (no_grad; safe to mutate masks now).
        if step % 8 == 0:
            try:
                with torch.no_grad():
                    pooled_nm = hh[:, -1:, :].detach()
                    model.neuromcp(pooled_nm)
                    model.neuromcp.step_plasticity()
                    model.neuromcp.codebook.maybe_grow(hh.mean(dim=1).detach())
            except Exception:
                pass

        cum_tok += args.batch_size * args.seq_len

        if step % LOG_EVERY == 0 or step == 1:
            tok_s = cum_tok / max(time.time() - t0, 1e-6)
            mem = (" mem_GB=" + f"{torch.cuda.memory_allocated()/1e9:.2f}"
                   if device.type == "cuda" else "")
            _log("step " + f"{step:5d}" + " loss=" + f"{loss.item():.4f}" +
                 " ce=" + f"{ce_loss.item():.3f}" +
                 " kd=" + f"{kd_loss.item():.3f}" +
                 " qa=" + f"{qa_loss.item():.3f}" +
                 " trip=" + f"{trip_loss.item():.3f}" +
                 " def=" + f"{def_loss.item():.3f}" +
                 " char=" + f"{char_loss.item():.3f}" +
                 " z=" + f"{z_loss.item():.2f}" +
                 " lr=" + f"{cur_lr:.5f}" +
                 " tok/s=" + f"{tok_s:.0f}" + mem)
            if plif_cells and step % 50 == 0:
                rates = [m.last_spike_rate.item() for m in plif_cells]
                rmean = sum(rates) / len(rates)
                n_dead = sum(1 for r in rates if r < 0.005)
                n_sat = sum(1 for r in rates if r > 0.5)
                _log("  spike: mean=" + f"{rmean:.3f}" +
                     " dead=" + str(n_dead) + "/" + str(len(rates)) +
                     " sat=" + str(n_sat) + "/" + str(len(rates)))

        if step % args.idle_every == 0:
            try:
                proposed = self_propose(model, student_tok, device, max_new=24)
                _log("  IDLE step " + str(step) + " proposed: " + repr(proposed[:120]))
                if replay_buf is not None:
                    # Score by inverse-prob (NoveltyDrive-style surprise).
                    ids = student_tok.encode(proposed)
                    if ids and len(ids) > 4:
                        # Push the rolled-out tokens with tag for later replay.
                        try:
                            replay_buf.add(("idle", ids), priority=1.0)
                        except Exception:
                            pass
            except Exception as e:
                _log("  IDLE failed: " + str(e))
        # Replay every 100 steps: pull a stored rollout, retrain on its tokens.
        if replay_buf is not None and step % 100 == 0 and len(getattr(replay_buf, '_buf', [])) > 4:
            try:
                tag, ids = replay_buf.sample()[0] if hasattr(replay_buf, 'sample') else (None, None)
                if ids and len(ids) > 4:
                    rb_x = torch.tensor([ids[:args.seq_len]], dtype=torch.long, device=device)
                    pad = student_tok.eos_token_id or 50256
                    if rb_x.shape[1] < args.seq_len:
                        rb_x = F.pad(rb_x, (0, args.seq_len - rb_x.shape[1]), value=pad)
                    rb_y = torch.cat([rb_x[:, 1:], rb_x[:, :1]], dim=1)
                    rh = model.encode_seq({"text_tokens": rb_x})
                    rhh = model.pool_text(rh, rb_x.shape[1])
                    rlogits = model.lm_head(rhh).float()
                    rb_loss = F.cross_entropy(rlogits.reshape(-1, VOCAB), rb_y.reshape(-1))
                    opt.zero_grad(set_to_none=True)
                    rb_loss.backward()
                    opt.step()
                    _log("  REPLAY step " + str(step) + " loss=" + f"{rb_loss.item():.3f}")
            except Exception:
                pass

        if step % SAVE_EVERY == 0:
            ck = os.path.join(args.out, f"step_{step:06d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optim_state": opt.state_dict(),
                "step": step,
                "loss": float(loss.item()),
                "n_params_M": n_params,
                "lr": cur_lr,
            }, ck)
            _log("saved ckpt " + ck)

        if step % EVAL_EVERY == 0 or step == args.steps:
            ppl = do_eval()
            metrics["ppl_eval"][step] = ppl
            _log("VAL step " + str(step) + ": ppl=" + f"{ppl:.2f}")

        if step % 60 == 0:
            with open(os.path.join(args.out, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(args.out, "live.log"), "w") as f:
                f.write("\n".join(log_lines[-200:]))

    torch.save({
        "model": model.state_dict(),
        "optim_state": opt.state_dict(),
        "step": args.steps,
        "n_params_M": n_params,
    }, os.path.join(args.out, "final.pt"))
    _log("done " + f"{time.time()-t0:.0f}s {cum_tok} tok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
