"""train_100m -- 1000-step training of synapforge_100m on WT-103.

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

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

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
    return p.parse_args()


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def evaluate(model, val_iter, n_batches: int = 16) -> float:
    model.eval()
    losses = []
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
    model.train()
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

    _log(f"device={DEVICE} dtype={DTYPE} out={out_dir} backend={backend_name}")
    _log(f"steps={n_steps} bs={args.batch_size} seq={SEQ_LEN} lr={LR}")

    # ---------------- model ----------------
    model = build_synapforge_100m(
        vocab=50257, d=512, n_layers=10, loop_depth=4,
        max_seq=SEQ_LEN, ffn_ratio=8.0, sparsity=0.95, dropout=0.0,
    )
    n_params = model.num_parameters()
    _log(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")

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

    # ---------------- backend integration ----------------
    if backend_name == "triton_block":
        try:
            from synapforge.backends.triton_block import (
                TritonBlockBackend,
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
    _log(f"optimizer: {type(optim).__name__} lr={LR} wd={WEIGHT_DECAY}")

    # ---------------- data ----------------
    train_ds = ParquetTokenStream(DATA_GLOB, seq_len=SEQ_LEN,
                                  batch_size=args.batch_size, loop=True)
    train_it = iter(train_ds)
    _log(f"train stream: {train_ds!r}")

    val_ds = ParquetTokenStream(VAL_GLOB, seq_len=SEQ_LEN,
                                batch_size=args.batch_size, loop=False)
    _log(f"val stream:   {val_ds!r}")

    from synapforge.huggingface_adapter import load_tokenizer
    tok = load_tokenizer("gpt2")

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
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
            )

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
            _log(f"step {step:5d} loss={loss.item():.4f} "
                 f"step_ms={step_ms:.1f} tok/s={tok_s:.0f}{mem_str}")

        if step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(out_dir, f"step_{step:06d}.pt")
            torch.save({
                "model": model.state_dict(),
                "step": step,
                "loss": float(loss.item()),
                "n_params": n_params,
            }, ckpt_path)
            _log(f"saved ckpt {ckpt_path}")

        if step % EVAL_EVERY == 0 or step == n_steps:
            val_it = iter(val_ds)
            ppl = evaluate(model, val_it, n_batches=16)
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
        "step": n_steps,
        "n_params": n_params,
    }, os.path.join(out_dir, "final.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "live.log"), "w") as f:
        f.write("\n".join(log_lines))
    elapsed = time.time() - t0
    _log(f"done in {elapsed:.1f}s ({cum_tok} tokens, "
         f"{cum_tok/max(elapsed,1e-6):.0f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
