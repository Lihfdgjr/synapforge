"""train_100m_kd -- KD distillation training of synapforge_100m on WT-103.

KD: KL(student || teacher) at temperature T, mixed with CE.

CLI:
    --out <dir>          override output dir
    --backend <name>     gpu_dense (default) or triton_block
    --warmstart <path>   override warm-start ckpt; "" disables
    --steps <int>        override N_STEPS

Wires together:
    * sf.model_100m.SynapForge100M (~99.3M params)
    * sf.huggingface_adapter.adv_warmstart
    * sf.optim.PlasticityAwareAdamW
    * sf.data.ParquetTokenStream on /workspace/data/wt103_raw

Runs on a single A100 80 GB (CUDA_VISIBLE_DEVICES from env).
"""

from __future__ import annotations

import os as _os_early
_os_early.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

# Strip script dir from sys.path so `import synapforge` finds the outer
# package (the nested /workspace/synapforge/synapforge/ has identical files
# but a different __init__.py — and pulls in unpatched copies of triton_block_kernel).
sys.path[:] = [p for p in sys.path if p not in ("/workspace/synapforge", "")]
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import synapforge as sf  # noqa: E402
from synapforge.surrogate import PLIFCell  # noqa: E402
from synapforge.data import ParquetTokenStream  # noqa: E402
from synapforge.huggingface_adapter import adv_warmstart  # noqa: E402
from synapforge.model_100m import build_synapforge_100m  # noqa: E402
from synapforge.optim import build_optimizer  # noqa: E402

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

OUT_DIR_DEFAULT = "/workspace/runs/synapforge_100m"
WARM_CKPT_DEFAULT = "/workspace/runs/step_001250.pt"
DATA_GLOB = "/workspace/data/wt103_raw/train-*.parquet"
VAL_GLOB = "/workspace/data/wt103_raw/validation.parquet"

N_STEPS_DEFAULT = 1000
SAVE_EVERY = 250
EVAL_EVERY = 500
LOG_EVERY = 10

BATCH_SIZE = 32
SEQ_LEN = 256
LR = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_DIR_DEFAULT)
    p.add_argument("--backend", default="gpu_dense",
                   choices=["gpu_dense", "triton_block"])
    p.add_argument("--warmstart", default=WARM_CKPT_DEFAULT)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=N_STEPS_DEFAULT)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr-decay", default="cosine", choices=["none","cosine","linear"])
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--grad-checkpoint", action="store_true", default=False,
                   help="trade compute for memory; use when B>=96 OOMs")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--skip-spike", action="store_true", default=True)
    p.add_argument("--z-loss-weight", type=float, default=1e-4,
                   help="weight for log-Z**2 stabilizer (PaLM/Gemma style)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="cross-entropy label smoothing alpha")
    p.add_argument("--spike-target", type=float, default=0.1,
                   help="target mean spike rate for PLIF cells; warn if drift > 0.05")
    p.add_argument("--teacher", default="gpt2",
                   help="HF model id for KD teacher (frozen); see HF_ENDPOINT for mirror")
    p.add_argument("--kd-weight", type=float, default=0.7,
                   help="alpha for KD loss; loss = (1-alpha)*ce + alpha*kd")
    p.add_argument("--kd-every", type=int, default=4)
    p.add_argument("--kd-temperature", type=float, default=4.0,
                   help="softmax temperature T; KD scaled by T*T")
    p.add_argument("--teacher-fallback-ckpt", default="",
                   help="path to a self-distill teacher ckpt (used if HF teacher load fails)")
    return p.parse_args()


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def evaluate(model, val_iter, n_batches: int = 16,
             plif_cells: Optional[list] = None) -> float:
    model.eval()
    losses = []
    rate_accum: list = []
    for _ in range(n_batches):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE,
                                enabled=DEVICE == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
            )
        losses.append(float(loss.item()))
        if plif_cells:
            rate_accum.append([m.last_spike_rate.item() for m in plif_cells])
    model.train()
    if plif_cells and rate_accum:
        n = len(plif_cells)
        avg_rates = [
            sum(r[i] for r in rate_accum) / len(rate_accum)
            for i in range(n)
        ]
        rate_min, rate_max = min(avg_rates), max(avg_rates)
        rate_mean = sum(avg_rates) / n
        n_dead = sum(1 for r in avg_rates if r < 0.005)
        n_sat = sum(1 for r in avg_rates if r > 0.5)
        _log(
            f"  [val] spike: mean={rate_mean:.3f} "
            f"range=[{rate_min:.3f}, {rate_max:.3f}] "
            f"dead={n_dead}/{n} sat={n_sat}/{n}"
        )
    if not losses:
        return float("nan")
    avg = sum(losses) / len(losses)
    return math.exp(avg)


@torch.no_grad()
def sample(model, tokenizer, prompt: str,
           max_new: int = 32, temperature: float = 0.8) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id or 50256]
    ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    max_seq = getattr(model, "max_seq", 256)
    for _ in range(max_new):
        ctx = ids[:, -max_seq:]
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE,
                                enabled=DEVICE == "cuda"):
            logits = model(ctx)
        logits = logits[0, -1].float() / max(temperature, 1e-3)
        topk = torch.topk(logits, k=50)
        probs = torch.softmax(topk.values, dim=-1)
        next_id = topk.indices[torch.multinomial(probs, 1)]
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
    text = tokenizer.decode(ids[0].tolist())
    model.train()
    return text


def lr_at(step: int, peak: float, warmup: int, total: int, kind: str = "cosine") -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    if kind == "none":
        return peak
    progress = (step - warmup) / max(1, total - warmup)
    if kind == "cosine":
        import math
        return peak * 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == "linear":
        return peak * max(0.0, 1.0 - progress)
    return peak




def _load_teacher(name: str, fallback_ckpt: str = "") -> "torch.nn.Module":
    """Load frozen GPT-2-class teacher; falls back to local ckpt on failure.

    Returns nn.Module whose forward(input_ids) yields logits (B, T, V).
    """
    import importlib
    try:
        AutoModelForCausalLM = importlib.import_module(
            "transformers"
        ).AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.to(DEVICE)
        return teacher
    except Exception as exc:  # pragma: no cover -- network-dependent
        if fallback_ckpt and os.path.exists(fallback_ckpt):
            print(f"[teacher] HF load failed ({exc!r}); using self-distill ckpt {fallback_ckpt}")
            ck = torch.load(fallback_ckpt, map_location="cpu")
            sd = ck.get("model", ck)
            t_model = build_synapforge_100m(
                vocab=151643, d=512, n_layers=10, loop_depth=1,
                max_seq=SEQ_LEN, ffn_ratio=8.0, sparsity=0.95, dropout=0.0,
                use_grad_checkpoint=False,
            )
            t_model.load_state_dict(sd, strict=False)
            t_model.eval()
            for p in t_model.parameters():
                p.requires_grad_(False)
            t_model.to(DEVICE)
            return t_model
        raise


def _teacher_logits(teacher, x: "torch.Tensor") -> "torch.Tensor":
    """Run teacher forward; handle HF (returns CausalLMOutput) vs nn.Module returning Tensor."""
    out = teacher(x)
    if hasattr(out, "logits"):
        return out.logits
    return out


def _kd_loss(student_logits, teacher_logits, T: float = 4.0):
    """Chunked KL(student_logp || teacher_p), avoids OOM on large vocab."""
    import torch
    import torch.nn.functional as F
    V = student_logits.size(-1); V_t = teacher_logits.size(-1)
    if V_t > V: teacher_logits = teacher_logits[..., :V]
    elif V > V_t: student_logits = student_logits[..., :V_t]
    bs = student_logits.size(0); chunk = max(1, bs // 4)
    total = 0.0
    for i in range(0, bs, chunk):
        sl = student_logits[i:i+chunk]; tl = teacher_logits[i:i+chunk].detach()
        slp = F.log_softmax(sl.float() / T, dim=-1)
        tp = F.softmax(tl.float() / T, dim=-1)
        total = total + F.kl_div(slp, tp, reduction='sum') / sl.size(0) / sl.size(1)
    n_chunks = (bs + chunk - 1) // chunk
    return (total / n_chunks) * (T * T)

def main() -> int:
    args = _parse_args()
    out_dir = args.out
    backend_name = args.backend
    warm_ckpt = args.warmstart
    n_steps = args.steps
    _safe_mkdir(out_dir)
    log_lines = []

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    print(f"device={DEVICE} dtype={DTYPE} out={out_dir} backend={backend_name}")
    print(f"steps={n_steps} bs={args.batch_size} seq={SEQ_LEN} lr={LR}")

    # ---------------- model ----------------
    model = build_synapforge_100m(
        vocab=151643, d=512, n_layers=10, loop_depth=1,
        max_seq=SEQ_LEN, ffn_ratio=8.0, sparsity=0.95, dropout=0.0,
        use_grad_checkpoint=args.grad_checkpoint,
    )
    n_params = model.num_parameters()
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")
    plif_cells = [m for m in model.modules() if isinstance(m, PLIFCell)]
    print(f"plif cells found: {len(plif_cells)}")

    # ---------------- warm-start ----------------
    if warm_ckpt and os.path.exists(warm_ckpt):
        try:
            rep = adv_warmstart(model, warm_ckpt,
                                name_map=[
                                    (r"\.cfc\.", ".liquid."),
                                    (r"\.embed\.text_embed\.", ".tok_embed."),
                                ])
            _log(
                f"warmstart matched={rep.matched}/{rep.total_target} "
                f"missing={len(rep.missing)} extra={len(rep.extra)} "
                f"shape_mismatch={len(rep.shape_mismatch)}"
            )
            for tgt, src in rep.matched_keys[:5]:
                _log(f"  warmstart match: {tgt} <- {src}")
        except Exception as exc:
            import traceback
            _log(f"warmstart skipped: {exc}")
            _log("warmstart traceback:")
            for line in traceback.format_exc().splitlines():
                _log(f"  {line}")
    else:
        _log(f"warmstart ckpt {warm_ckpt!r} not found; pure random init")

    model = model.to(DEVICE)
    model.train()

    # ---------------- KD teacher (frozen) ----------------
    teacher = None
    if args.kd_weight > 0:
        try:
            _log(f"loading KD teacher: {args.teacher!r} (HF_ENDPOINT={os.environ.get('HF_ENDPOINT','<unset>')})")
            teacher = _load_teacher(args.teacher, args.teacher_fallback_ckpt)
            t_params = sum(p.numel() for p in teacher.parameters())
            _log(f"teacher loaded: {type(teacher).__name__} params={t_params/1e6:.1f}M frozen=True")
        except Exception as exc:
            import traceback
            _log(f"teacher load FAILED: {exc}; disabling KD (alpha=0)")
            for line in traceback.format_exc().splitlines():
                _log(f"  {line}")
            args.kd_weight = 0.0
            teacher = None
    else:
        _log("KD disabled (kd-weight=0)")

    # ---------------- backend integration ----------------
    if backend_name == "triton_block":
        try:
            from synapforge.backends.triton_block import (
                TritonBlockBackend,
                _SharedTritonBlock,
            )
            from synapforge.backends.triton_block_kernel import _HAS_TRITON
            backend = TritonBlockBackend()
            stats = backend.compile(model)
            _log(
                f"[backend] triton_block enabled: triton_avail={_HAS_TRITON} "
                f"pairs_fused={stats.get('n_pairs_fused', 0)}"
            )
            # Sanity: must have fused at least one pair, else we're a no-op.
            if stats.get("n_pairs_fused", 0) == 0:
                _log("[backend] WARN no Liquid+PLIF pairs fused -- "
                     "triton_block is a no-op for this model")
        except Exception as exc:
            import traceback
            _log(f"[backend] triton_block compile FAILED: {exc}")
            for line in traceback.format_exc().splitlines():
                _log(f"  {line}")
            _log("[backend] falling back to gpu_dense (PyTorch native)")
            backend_name = "gpu_dense"
    else:
        _log("[backend] gpu_dense (PyTorch native passthrough)")

    # ---------------- optimizer ----------------
    optim = build_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"optimizer: {type(optim).__name__} lr={LR} wd={WEIGHT_DECAY}")

    # Load optimizer state from warmstart ckpt (preserves Adam m/v momentum;
    # without this, warmstart loses momentum state and the run cold-starts
    # the moment estimates -- a known cause of warmstart-divergence as seen
    # in v1.2 b80 (loss 5.45 -> 6.34 in 500 steps).
    if warm_ckpt and os.path.exists(warm_ckpt):
        try:
            _ck = torch.load(warm_ckpt, map_location="cpu")
            if isinstance(_ck, dict) and "optim_state" in _ck:
                optim.load_state_dict(_ck["optim_state"])
                _log(f"warmstart: loaded optim state from {warm_ckpt}")
            else:
                _log(f"warmstart: no optim_state in ckpt (legacy ckpt; momentum cold-start)")
        except Exception as exc:
            _log(f"warmstart optim load skipped: {exc}")

    # ---------------- data ----------------
    train_ds = ParquetTokenStream(DATA_GLOB, seq_len=SEQ_LEN, tokenizer_name="/workspace/teachers/qwen2.5-0.5b",
                                  batch_size=args.batch_size, loop=True)
    train_it = iter(train_ds)
    print(f"train stream: {train_ds!r}")

    val_ds = ParquetTokenStream(VAL_GLOB, seq_len=SEQ_LEN, tokenizer_name="/workspace/teachers/qwen2.5-0.5b",
                                batch_size=args.batch_size, loop=False)
    print(f"val stream:   {val_ds!r}")

    from synapforge.huggingface_adapter import load_tokenizer
    tok = load_tokenizer("/workspace/teachers/qwen2.5-0.5b")

    # ---------------- training ----------------
    metrics = {"step": [], "loss": [], "step_ms": [], "tok_per_s": [],
               "ppl_eval": {}, "samples": {}}
    t0 = time.time()
    last_log_t = t0
    cum_tok = 0
    sample_prompts = [
        "The first thing the agent did was",
        "In a quiet town nestled between",
        "Recurrent neural networks are",
    ]

    for step in range(1, n_steps + 1):
        cur_lr = lr_at(step, LR, args.warmup, n_steps, args.lr_decay)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr
        t_step = time.time()
        try:
            x, y = next(train_it)
        except StopIteration:
            _log(f"data exhausted at step {step}")
            break
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE,
                                enabled=DEVICE == "cuda"):
            logits = model(x)
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_y = y.reshape(-1)
            ce_loss = F.cross_entropy(flat_logits, flat_y,
                                      label_smoothing=args.label_smoothing)
            # z-loss: penalize large logsumexp (numerical stability + softmax temp control)
            log_z = torch.logsumexp(flat_logits, dim=-1)
            z_loss = (log_z ** 2).mean()
            base_loss = ce_loss + args.z_loss_weight * z_loss

            if teacher is not None and args.kd_weight > 0:
                with torch.no_grad():
                    t_logits = _teacher_logits(teacher, x)
                kd = _kd_loss(logits, t_logits, args.kd_temperature) if (step % args.kd_every == 0) else torch.tensor(0., device=logits.device)
                loss = (1.0 - args.kd_weight) * base_loss + args.kd_weight * kd
            else:
                kd = torch.zeros((), device=logits.device)
                loss = base_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=GRAD_CLIP,
            )
        optim.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.time() - t_step) * 1000.0
        cum_tok += args.batch_size * SEQ_LEN

        if step % LOG_EVERY == 0 or step == 1:
            tok_s = cum_tok / max(time.time() - t0, 1e-6)
            metrics["step"].append(step)
            metrics["loss"].append(float(loss.item()))
            metrics["step_ms"].append(step_ms)
            metrics["tok_per_s"].append(tok_s)
            mem_str = (
                f" mem_GB={torch.cuda.memory_allocated()/1e9:.2f}"
                if DEVICE == "cuda" else ""
            )
            _log(f"step {step:5d} loss={loss.item():.4f} ce={ce_loss.item():.3f} "
                 f"kd={kd.item():.3f} z={z_loss.item():.3f} lr={cur_lr:.5f} "
                 f"step_ms={step_ms:.1f} tok/s={tok_s:.0f}{mem_str}")
            if step % 50 == 0 and plif_cells:
                rates = [m.last_spike_rate.item() for m in plif_cells]
                rate_min, rate_max = min(rates), max(rates)
                rate_mean = sum(rates) / len(rates)
                n_dead = sum(1 for r in rates if r < 0.005)
                n_sat = sum(1 for r in rates if r > 0.5)
                _log(
                    f"  spike: mean={rate_mean:.3f} "
                    f"range=[{rate_min:.3f}, {rate_max:.3f}] "
                    f"dead={n_dead}/{len(rates)} sat={n_sat}/{len(rates)}"
                )
                if step % 100 == 0 and abs(rate_mean - args.spike_target) > 0.05:
                    direction = "high" if rate_mean > args.spike_target else "low"
                    _log(
                        f"  [WARN] spike rate {rate_mean:.3f} drifted {direction} "
                        f"of target {args.spike_target:.3f} (|delta|>0.05); "
                        f"consider threshold auto-adjust (not enabled)"
                    )

        if step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(out_dir, f"step_{step:06d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optim_state": optim.state_dict(),
                "step": step,
                "loss": float(loss.item()),
                "n_params": n_params,
                "lr": cur_lr,
            }, ckpt_path)
            _log(f"saved ckpt {ckpt_path}")

        if step % EVAL_EVERY == 0 or step == n_steps:
            val_it = iter(val_ds)
            ppl = evaluate(model, val_it, n_batches=16, plif_cells=plif_cells)
            metrics["ppl_eval"][step] = ppl
            _log(f"VAL step {step}: ppl={ppl:.2f}")
            samples = []
            for p in sample_prompts:
                try:
                    s = sample(model, tok, p, max_new=32, temperature=0.8)
                    samples.append({"prompt": p, "completion": s})
                    _log(f"  sample> {s[:140]!r}")
                except Exception as exc:
                    samples.append({"prompt": p, "error": str(exc)})
                    _log(f"  sample FAILED for {p!r}: {exc}")
            metrics["samples"][step] = samples

        if time.time() - last_log_t > 30:
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(out_dir, "live.log"), "w") as f:
                f.write("\n".join(log_lines[-200:]))
            last_log_t = time.time()

    torch.save({
        "model": model.state_dict(),
        "optim_state": optim.state_dict(),
        "step": n_steps,
        "n_params": n_params,
    }, os.path.join(out_dir, "final.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "live.log"), "w") as f:
        f.write("\n".join(log_lines))
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({cum_tok} tokens, "
         f"{cum_tok/max(elapsed,1e-6):.0f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
