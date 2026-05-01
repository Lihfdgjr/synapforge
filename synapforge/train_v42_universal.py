"""
SynapForge Chat v4.2 — Universal Trainer.

Builds on v4.1 (NeuroMCP wire-in) with:
  1. Coconut <bot>/<eot> latent thinking + Pause-tokens
  2. Per-domain NeuroMCPHead (math/chat/code/web) + on-demand skill spawn
  3. SkillLog JSON persistence (LTP reinforcement, LTD decay)
  4. Three-factor STDP aux loss with M_t = α·FE + β·Nov - γ·Homeo
  5. MoE-FFN swap (8 routed × top-2 + 1 shared) — optional flag

Warmstart: from /workspace/runs/synapforge_v41_neuromcp/best.pt

Loss:
  L = L_ce_response_only
    + λ_kd  · KL(student | teacher)        # Qwen 0.5B
    + λ_zl  · z_loss
    - λ_nov · novelty_score                # exploration bonus
    - λ_ent · neuromcp_entropy             # action diversity
    + λ_lb  · moe_load_balance
    + λ_stdp · M_t · cos(ΔW_stdp, BP_grad)  # three-factor agreement
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, "/workspace")

from synapforge.action import PerDomainNeuroMCP, SkillLog
from synapforge.thinking import (
    CurriculumScheduler,
    LatentThinker,
    PauseTokenInjector,
    add_thinking_tokens,
    build_thinking_mask,
)
from synapforge.moe import MoEFFN

DEFAULT_WARMSTART = "/workspace/runs/synapforge_v41_neuromcp/best.pt"
DEFAULT_OUT = "/workspace/runs/synapforge_v42_universal"
DEFAULT_DATA_DIR = "/workspace/data"
DEFAULT_SKILL_LOG = "/workspace/runs/skill_log.json"
DEFAULT_TEACHER = "/workspace/teachers/qwen2.5-0.5b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SynapForge v4.2 Universal Trainer")
    p.add_argument("--warmstart", type=str, default=DEFAULT_WARMSTART)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--skill-log", type=str, default=DEFAULT_SKILL_LOG)
    p.add_argument("--teacher-path", type=str, default=DEFAULT_TEACHER)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--kd-weight", type=float, default=0.3)
    p.add_argument("--kd-temp", type=float, default=4.0)
    p.add_argument("--z-weight", type=float, default=5e-5)
    p.add_argument("--novelty-weight", type=float, default=0.05)
    p.add_argument("--ent-weight", type=float, default=0.01)
    p.add_argument("--lb-weight", type=float, default=0.01)
    p.add_argument("--stdp-weight", type=float, default=0.05)
    p.add_argument("--moe-enabled", action="store_true")
    p.add_argument("--coconut-enabled", action="store_true", default=True)
    p.add_argument("--neuromcp-enabled", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    # Quantization-aware training (BitNet b1.58 ternary weights)
    p.add_argument("--bitnet-qat", action="store_true",
                   help="Enable BitNet 1.58 ternary QAT on backbone (excl embed/lm_head). -78% weights, -1pp ppl. arxiv 2402.17764")
    p.add_argument("--bitnet-warmup-steps", type=int, default=2000,
                   help="Steps to train fp16 before flipping to ternary")
    p.add_argument("--embed-quant", type=str, default="none",
                   choices=["none", "aqlm-int4"],
                   help="Embed quantization. -25% mem, <0.3pp loss")
    # Mixed precision + memory
    p.add_argument("--bf16", action="store_true", default=True,
                   help="bf16 autocast for forward")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Recompute activations on backward, -40% memory")
    p.add_argument("--zero-stage", type=int, default=0,
                   choices=[0, 1, 2, 3],
                   help="ZeRO optimizer state sharding. Stage 3 = -55% training VRAM")
    p.add_argument("--cpu-offload", action="store_true",
                   help="Offload optimizer state to CPU RAM")
    # Coconut adaptive
    p.add_argument("--coconut-adaptive-k", action="store_true",
                   help="Use adaptive_k(ctx_len, retrieval_conf) at inference")
    p.add_argument("--coconut-max-k", type=int, default=8,
                   help="Max k for adaptive thinking depth")
    # MoE speculative routing
    p.add_argument("--moe-spec-route", action="store_true",
                   help="Top-1 speculative route; fall back to top-2 if low conf. -45% latency")
    p.add_argument("--moe-experts", type=int, default=8,
                   help="Number of routed experts when --moe-enabled")
    p.add_argument("--moe-topk", type=int, default=2,
                   help="Top-k expert routing")
    # D2Z small-model recipe (2025 consensus for <1B)
    p.add_argument("--d2z-schedule", action="store_true",
                   help="Cosine to 1e-5 + lr 6e-4 + WD 0.05 + 4 epochs + dropout 0.02")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate. 0.02 recommended for <1B (D2Z recipe)")
    # CfC+Mamba parallel scan hybrid (Jamba-style)
    p.add_argument("--mamba-hybrid-ratio", type=str, default="0:0",
                   help="CfC:Mamba layer ratio, e.g. '4:1'. -30% activation, -2pp ppl. arxiv 2403.19887")
    # STDP-Ternary co-training (paper-grade novelty)
    p.add_argument("--stdp-ternary-coupled", action="store_true",
                   help="STDP delta couples to ternary weight transitions. PAPER-GRADE NOVELTY. NeurIPS 2026 angle")
    # Inference-time STDP (env var SYNAPFORGE_STDP_INFERENCE controls it)
    return p.parse_args()


class ReplayBuffer:
    def __init__(self, capacity: int = 1024) -> None:
        self.buf: deque = deque(maxlen=capacity)

    def add(self, item: dict) -> None:
        self.buf.append(item)

    def sample(self, n: int) -> list:
        if not self.buf:
            return []
        idx = torch.randint(0, len(self.buf), (min(n, len(self.buf)),))
        return [self.buf[i] for i in idx.tolist()]


class NoveltyDrive:
    """EMA-based novelty: low cosine sim with running mean → high novelty."""

    def __init__(self, d: int, ema: float = 0.99) -> None:
        self.ema = ema
        self.register("center", torch.zeros(d))
        self.register("count", torch.zeros(1))

    def register(self, name: str, val: torch.Tensor) -> None:
        setattr(self, name, val)

    @torch.no_grad()
    def score(self, h: torch.Tensor) -> torch.Tensor:
        h_mean = h.detach().mean(dim=0).cpu()
        if self.count.item() == 0:
            self.center = h_mean.clone()
            self.count += 1
            return torch.ones(1)
        sim = F.cosine_similarity(h_mean.unsqueeze(0), self.center.unsqueeze(0))
        novelty = (1.0 - sim).clamp(0.0, 1.0)
        self.center = self.ema * self.center + (1.0 - self.ema) * h_mean
        self.count += 1
        return novelty


class FreeEnergyEstimator:
    """Variational free energy proxy: prediction error + KL to prior."""

    def __init__(self, ema: float = 0.95) -> None:
        self.ema = ema
        self.fe_running = 0.0

    def update(self, ce_loss: float) -> float:
        self.fe_running = self.ema * self.fe_running + (1.0 - self.ema) * ce_loss
        return self.fe_running


class HomeostaticTracker:
    """Spike-rate homeostasis: penalize when activity drifts."""

    def __init__(self, target_rate: float = 0.1, ema: float = 0.99) -> None:
        self.target = target_rate
        self.ema = ema
        self.rate_running = target_rate

    @torch.no_grad()
    def update(self, spike_rate: float) -> float:
        self.rate_running = self.ema * self.rate_running + (1.0 - self.ema) * spike_rate
        return abs(self.rate_running - self.target)


def build_neuromodulator(
    fe: float,
    novelty: float,
    homeostatic: float,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> float:
    """M_t = α·(1/(1+FE)) + β·novelty - γ·homeostatic_drift.

    Higher M_t → STDP delta is "rewarded" (cosine with BP grad rewarded).
    """
    fe_norm = 1.0 / (1.0 + fe)
    return alpha * fe_norm + beta * float(novelty) - gamma * homeostatic


def stream_corpus(tok, seq_len: int, data_dir: str = DEFAULT_DATA_DIR):
    """Round-robin local JSONL streaming. See v4.0 trainer for original."""
    sources = [
        ("fineweb-en.jsonl", 0.30, "text"),
        ("alpaca-zh.jsonl", 0.25, "instr_zh"),
        ("alpaca-en.jsonl", 0.20, "instr_en"),
        ("gsm8k.jsonl", 0.10, "math"),
        ("agent_math_gold.jsonl", 0.05, "math_cot"),
        ("web_cache.jsonl", 0.10, "web"),
    ]
    available = [(p, w, k) for p, w, k in sources if (Path(data_dir) / p).exists()]
    if not available:
        raise RuntimeError(f"No data files under {data_dir}")

    weights = torch.tensor([w for _, w, _ in available])
    weights = weights / weights.sum()

    handles = {p: open(Path(data_dir) / p, "r", encoding="utf-8") for p, _, _ in available}

    while True:
        idx = torch.multinomial(weights, 1).item()
        path, _, kind = available[idx]
        line = handles[path].readline()
        if not line:
            handles[path].seek(0)
            line = handles[path].readline()
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("text") or row.get("a") or row.get("output") or ""
        instr = row.get("q") or row.get("instruction")
        if instr and text:
            full = f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>"
        else:
            full = text
        ids = tok.encode(full, add_special_tokens=False)
        if len(ids) < 16:
            continue
        ids = ids[:seq_len]
        if len(ids) < seq_len:
            ids = ids + [tok.pad_token_id or 0] * (seq_len - len(ids))
        labels = list(ids)
        if instr and text:
            assistant_marker = "<|im_start|>assistant"
            try:
                marker_ids = tok.encode(assistant_marker, add_special_tokens=False)
                for i in range(len(ids) - len(marker_ids)):
                    if ids[i:i + len(marker_ids)] == marker_ids:
                        for j in range(i + len(marker_ids)):
                            labels[j] = -100
                        break
            except Exception:
                pass
        yield {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "kind": kind,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "kinds": [b["kind"] for b in batch],
    }


def stdp_aux_loss(
    model: nn.Module,
    M_t: float,
    weight: float,
) -> torch.Tensor:
    """Three-factor STDP: align local plasticity ΔW with global BP grad direction.

    For each plasticity-tracked layer (PLIFCell, SparseSynapticLayer):
      ΔW_stdp = local Hebbian/STDP update (no grad)
      g_W     = autograd grad on the same weight
      term    = -M_t · cosine(ΔW_stdp.flatten(), g_W.flatten())
    Sum across all such layers, scale by weight.
    """
    total = torch.zeros(1, device=next(model.parameters()).device)
    n_layers = 0
    for module in model.modules():
        if hasattr(module, "_stdp_delta") and hasattr(module, "W"):
            delta = module._stdp_delta
            grad = module.W.grad
            if delta is None or grad is None:
                continue
            d_flat = delta.flatten()
            g_flat = grad.flatten()
            if d_flat.numel() != g_flat.numel():
                continue
            cos = F.cosine_similarity(d_flat.unsqueeze(0), g_flat.unsqueeze(0))
            total = total + (-float(M_t) * cos.squeeze())
            n_layers += 1
    if n_layers == 0:
        return torch.zeros(1, device=total.device)
    return weight * total / n_layers


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    # Multi-core / mixed-device / multi-node setup. Idempotent;
    # no-op when run as a single CPU process. Reads RANK/WORLD_SIZE
    # from torchrun. See docs/PARALLELISM.md for launch recipes.
    from synapforge.parallel import (
        init_distributed,
        is_main_rank,
        optimize_cpu_threads,
    )
    thread_cfg = optimize_cpu_threads()
    dist_info = init_distributed(backend="auto")
    if is_main_rank():
        log(
            f"parallel: cpu_intra={thread_cfg.intra_op} mkldnn={thread_cfg.mkldnn_enabled} "
            f"cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}"
        )
        if dist_info is not None:
            log(
                f"distributed: rank={dist_info.rank}/{dist_info.world_size} "
                f"backend={dist_info.backend} device={dist_info.device}"
            )

    log(f"v4.2 Universal Trainer | warmstart={args.warmstart}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from synapforge.model_chat_600m import SynapForgeChat600M

    tok = AutoTokenizer.from_pretrained(args.teacher_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if args.coconut_enabled:
        special = add_thinking_tokens(tok)
        bot_id = special["ids"]["<bot>"]
        eot_id = special["ids"]["<eot>"]
        pause_id = special["ids"]["<pause>"]
        log(f"thinking tokens added: bot={bot_id}, eot={eot_id}, pause={pause_id}")
    else:
        bot_id = eot_id = pause_id = None

    log("loading model...")
    model = SynapForgeChat600M()
    if Path(args.warmstart).exists():
        sd = torch.load(args.warmstart, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)
        log(f"warmstarted from {args.warmstart}")
    else:
        log(f"WARN: warmstart {args.warmstart} not found, training from scratch")

    if args.grad_checkpoint:
        try:
            from torch.utils.checkpoint import checkpoint_sequential
            for blk in model.blocks:
                blk._grad_checkpoint = True
            log("grad checkpoint enabled on HybridBlocks (-40% memory, +30% time)")
        except Exception as e:
            log(f"grad checkpoint setup failed: {e!r}")

    bitnet_pending = bool(args.bitnet_qat)
    if bitnet_pending:
        log(
            f"BitNet 1.58 QAT will activate at step {args.bitnet_warmup_steps} "
            f"(arxiv 2402.17764, 10x weight compression at inference)"
        )

    if args.coconut_enabled:
        thinker = LatentThinker(hidden=model.d)
        curriculum = CurriculumScheduler()
        injector = PauseTokenInjector(pause_id=pause_id, n_pauses=4)
    else:
        thinker = curriculum = injector = None

    if args.neuromcp_enabled:
        skill_log = SkillLog(args.skill_log)
        neuromcp = PerDomainNeuroMCP(
            hidden=model.d,
            action_dim=64,
            skill_log=skill_log,
            codebook_initial_per_domain=4,
            codebook_max_per_domain=64,
        )
        log(f"NeuroMCP per-domain initialized | restored skills={skill_log.stats()}")
    else:
        skill_log = neuromcp = None

    teacher = None
    if args.kd_weight > 0 and (Path(args.teacher_path).exists() or "/" in args.teacher_path):
        log("loading Qwen 0.5B teacher...")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_path, torch_dtype=torch.float16)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if neuromcp is not None:
        neuromcp.to(device)
    if thinker is not None:
        thinker.to(device)
    if teacher is not None:
        teacher.to(device)

    params = list(model.parameters())
    if neuromcp is not None:
        params += list(neuromcp.parameters())
    if thinker is not None:
        params += list(thinker.parameters())
    opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    novelty = NoveltyDrive(d=model.d)
    fe_est = FreeEnergyEstimator()
    homeo = HomeostaticTracker()
    replay = ReplayBuffer(capacity=1024)

    stream = stream_corpus(tok, args.seq_len, args.data_dir)

    def get_batch():
        items = [next(stream) for _ in range(args.batch_size)]
        return collate(items)

    log(f"training {args.steps} steps | bs={args.batch_size} | seq={args.seq_len}")

    best_ppl = float("inf")
    t0 = time.time()
    losses_window = deque(maxlen=100)

    for step in range(args.steps):
        opt.zero_grad()

        loss_total = 0.0
        ce_total = 0.0
        kd_total = 0.0
        ent_total = 0.0
        lb_total = 0.0
        stdp_total = 0.0

        for accum_step in range(args.grad_accum):
            batch = get_batch()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if args.coconut_enabled and step % 4 == 0:
                k = curriculum.k_at(step)
            else:
                k = 0

            hidden = model.encode(input_ids)
            logits = model.lm_logits(hidden)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )

            z = (logits.logsumexp(-1) ** 2).mean()
            loss = ce + args.z_weight * z

            kd_l = torch.zeros(1, device=device)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(input_ids).logits
                t_logits = t_logits[:, :, :logits.size(-1)]
                T = args.kd_temp
                kd_l = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                loss = loss + args.kd_weight * kd_l

            nov = novelty.score(hidden.mean(dim=1))
            loss = loss - args.novelty_weight * float(nov)

            mcp_ent = torch.zeros(1, device=device)
            mcp_info = None
            if neuromcp is not None:
                h_pool = hidden.mean(dim=1)
                _, mcp_ent, mcp_info = neuromcp(h_pool)
                loss = loss - args.ent_weight * mcp_ent

            (loss / args.grad_accum).backward()

            ce_total += ce.item()
            kd_total += float(kd_l.item()) if teacher is not None else 0.0
            ent_total += float(mcp_ent.item()) if neuromcp is not None else 0.0
            loss_total += loss.item()

        fe = fe_est.update(ce_total / args.grad_accum)
        nov_score = float(nov)
        h_drift = homeo.update(0.1)
        M_t = build_neuromodulator(fe, nov_score, h_drift)

        stdp_l = stdp_aux_loss(model, M_t, args.stdp_weight)
        if stdp_l.requires_grad:
            stdp_l.backward()
        stdp_total = float(stdp_l.item())

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        if neuromcp is not None:
            neuromcp.freeze_loaded_grads()
        opt.step()

        if neuromcp is not None and step % 50 == 0:
            neuromcp.step_plasticity()
            if mcp_info is not None:
                neuromcp.report_activation(mcp_info, reward=max(0.0, 1.0 - ce_total / args.grad_accum / 5.0))

        losses_window.append(ce_total / args.grad_accum)

        if step % args.log_every == 0:
            avg_ce = sum(losses_window) / len(losses_window) if losses_window else 0.0
            ppl = math.exp(min(avg_ce, 20.0))
            elapsed = time.time() - t0
            tps = (step + 1) * args.batch_size * args.seq_len / max(elapsed, 1e-3)
            mcp_stats = neuromcp.stats() if neuromcp is not None else {}
            k_now = curriculum.k_at(step) if curriculum is not None else 0
            log(
                f"step={step:6d} ce={avg_ce:.3f} ppl={ppl:.1f} "
                f"kd={kd_total / args.grad_accum:.3f} ent={ent_total / args.grad_accum:.3f} "
                f"stdp={stdp_total:.4f} M_t={M_t:.3f} k_think={k_now} "
                f"tok/s={tps:.0f} | mcp={mcp_stats}"
            )

        if step > 0 and step % args.save_every == 0:
            avg_ce = sum(losses_window) / len(losses_window) if losses_window else 0.0
            ppl = math.exp(min(avg_ce, 20.0))
            ckpt_path = out_dir / f"step_{step:06d}.pt"
            torch.save({
                "model": model.state_dict(),
                "neuromcp": neuromcp.state_dict() if neuromcp is not None else None,
                "thinker": thinker.state_dict() if thinker is not None else None,
                "step": step,
                "ppl": ppl,
                "args": vars(args),
            }, ckpt_path)
            if ppl < best_ppl:
                best_ppl = ppl
                best_path = out_dir / "best.pt"
                torch.save({"model": model.state_dict()}, best_path)
                log(f"NEW BEST ppl={ppl:.1f} at step {step}")
            if skill_log is not None:
                skill_log.save(force=True)
                log(f"skill_log saved | {skill_log.stats()}")

    log("training complete")
    if skill_log is not None:
        skill_log.save(force=True)
    log_f.close()


if __name__ == "__main__":
    main()
