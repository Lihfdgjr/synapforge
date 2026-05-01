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
from synapforge.data import ParquetTokenStream, split_val_stream  # noqa: E402
from synapforge.huggingface_adapter import adv_warmstart  # noqa: E402
from synapforge.model_100m import build_synapforge_100m  # noqa: E402
from synapforge.optim import build_optimizer  # noqa: E402

# Phase signal + honest eval (best-effort imports; trainer must not die if
# these aren't on disk).
try:
    from synapforge import phase_signal  # noqa: E402
except Exception as _exc:  # pragma: no cover
    phase_signal = None  # type: ignore[assignment]
    print(f"[trainer] phase_signal unavailable: {_exc!r}", flush=True)
try:
    # scripts/ is sibling of train_100m_kd.py; ensure it's on sys.path.
    _SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    from honest_eval_hook import HonestEvalHook  # type: ignore  # noqa: E402
except Exception as _exc:  # pragma: no cover
    HonestEvalHook = None  # type: ignore[assignment]
    print(f"[trainer] honest_eval_hook unavailable: {_exc!r}", flush=True)

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

# 2026-05-01 perf bump: bs 32 -> 80 on A800-80GB. With z-loss `logsumexp`
# materialising a (B*T, V, fp32) intermediate at vocab=151936 / seq=256, we
# previously capped at bs=64 (~9.5 GiB intermediate) with ~8 GiB headroom.
# Sparse z-loss (top-K logits, default K=2048) drops that to ~3 GiB at bs=80
# and ~3.6 GiB at bs=96; the bs=80 sweet spot leaves real headroom for the
# KD activation chunks. Document: bs=96 still OOMs because the KD chunked
# `(chunk*T, V, fp32)` softmax stack stays full-vocab. See docs/PERF_KNOBS.md.
BATCH_SIZE = 80
SEQ_LEN = 256
LR = 1e-4  # 2026-05-01: 3e-4 was too aggressive — Run 3b VAL ppl exploded
           #             422 -> 4071 by step 5500.  1e-4 is the proven phase 0
           #             rate per Anthropic Tulu 3 / SmolLM2 recipes.
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Architecture constants used to build the model. These are SOURCE OF TRUTH:
# trainer-side and persisted into every ckpt's "config" dict so the loader
# (chat_demo.py / chat_repl.py) can reconstruct the exact model without
# guessing. P12 — see docs/MASTER_PLAN.md §6.
MODEL_VOCAB = 151936
MODEL_D = 512
MODEL_N_LAYERS = 10
MODEL_LOOP_DEPTH = 1
MODEL_FFN_RATIO = 8.0
MODEL_SPARSITY = 0.95
MODEL_DROPOUT = 0.0
MODEL_TIE_LM_HEAD = True


def _build_config_dict(args) -> dict:
    """Build the architecture-config dict persisted alongside every ckpt.

    Loader (chat_demo / chat_repl) reads this via ckpt["config"] and
    rebuilds the model with the exact shapes; falls back to its own
    hardcoded defaults if the key is missing (legacy ckpts). See P12 in
    docs/MASTER_PLAN.md §6.

    Reads from ``args`` (set by argparse) so P9-smoke and tiny-model runs
    persist the exact shapes they trained with — not the hardcoded 100M
    defaults. ``getattr`` fallbacks make this resilient to older callers
    that may not have run the new argparse (e.g. unit tests).
    """
    return {
        "vocab": int(getattr(args, "vocab", MODEL_VOCAB)),
        "d": int(getattr(args, "d", MODEL_D)),
        "n_layers": int(getattr(args, "n_layers", MODEL_N_LAYERS)),
        "loop_depth": int(getattr(args, "loop_depth", MODEL_LOOP_DEPTH)),
        # max_seq is what the trainer was actually built with -- not the arg
        # default, since args.max_seq may not exist on this trainer.
        "max_seq": int(getattr(args, "seq_len", SEQ_LEN)),
        "ffn_ratio": float(getattr(args, "ffn_ratio", MODEL_FFN_RATIO)),
        "sparsity": float(getattr(args, "sparsity", MODEL_SPARSITY)),
        "dropout": MODEL_DROPOUT,
        "tie_lm_head": MODEL_TIE_LM_HEAD,
    }


def _log(msg: str) -> None:
    """Module-level log shim. main() rebinds a closure that ALSO appends to
    log_lines, but evaluate() and the other module-level helpers may run
    before/after main()'s closure exists, so we always have a safe fallback."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _adaptive_kd_every(student_ce: float,
                       teacher_ce_estimate: float = 4.5,
                       base: int = 4) -> int:
    """Schedule KD frequency by gap between student and teacher CE.

    Intuition: when the student is far from the teacher (early training,
    big gap), every-2-step KD pulls harder; when the student is close
    to the teacher (late training, small gap), KD adds little signal,
    so we drop to every-8 or every-16 to save the teacher-forward tax.

    Args:
        student_ce: running mean CE of recent KD-OFF steps (so the
            measure is not artificially pulled down by KD itself).
        teacher_ce_estimate: rough teacher-on-train-distribution CE; the
            anchor we measure the gap against. 4.5 is a reasonable
            Qwen2.5-0.5B / wt103 figure; lower = more aggressive KD
            reduction.
        base: nominal --kd-every value (4 in production today). The
            adaptive schedule walks ``base // 2`` ... ``base * 4``
            around it.

    Returns:
        New ``kd_every`` integer in {1, 2, 4, 8, 16, ...}. Always >= 1
        (never disable KD outright via this path).

    Schedule (with ``base=4``):
        gap > 3.0    -> kd_every = 2     (more KD signal, early)
        gap > 1.5    -> kd_every = 4     (default mid-training)
        gap > 0.5    -> kd_every = 8     (small signal, save 50%)
        gap <= 0.5   -> kd_every = 16    (KD nearly converged)
    """
    base = max(1, int(base))
    gap = max(0.0, float(student_ce) - float(teacher_ce_estimate))
    if gap > 3.0:
        return max(1, base // 2)
    if gap > 1.5:
        return base
    if gap > 0.5:
        return base * 2
    return base * 4


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_DIR_DEFAULT)
    p.add_argument("--backend", default="gpu_dense",
                   choices=["gpu_dense", "triton_block"])
    p.add_argument("--warmstart", default=WARM_CKPT_DEFAULT)
    p.add_argument("--no-warmstart", dest="warmstart", action="store_const",
                   const="", help="force-disable warmstart (P9 smoke test).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"per-step batch size; default {BATCH_SIZE} bumped from "
                        "32 on 2026-05-01 (A800-80GB sweet spot with sparse "
                        "z-loss + expandable_segments). bs=96 still OOMs "
                        "because KD chunked softmax stays full-vocab.")
    p.add_argument("--steps", type=int, default=N_STEPS_DEFAULT)
    p.add_argument("--warmup", type=int, default=200)
    # ---- P9 data-pipeline / smoke overrides (defaults preserve rental paths) ----
    p.add_argument("--data-glob", default=DATA_GLOB,
                   help="glob for training parquets; overrides DATA_GLOB.")
    p.add_argument("--val-glob", default=VAL_GLOB,
                   help="glob for validation parquets; overrides VAL_GLOB.")
    p.add_argument("--tokenizer-name", default="/workspace/teachers/qwen2.5-0.5b",
                   help="HF tokenizer dir or model id (e.g. 'gpt2' for CI).")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY,
                   help="save step_*.pt every N steps (overrides SAVE_EVERY).")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                   help="run val ppl every N steps (overrides EVAL_EVERY).")
    p.add_argument("--log-every", type=int, default=LOG_EVERY,
                   help="emit a log line every N steps (overrides LOG_EVERY).")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN,
                   help="tokens per training example (overrides SEQ_LEN).")
    # ---- P9 architecture overrides for tiny-model smoke / unit testing ----
    p.add_argument("--vocab", type=int, default=MODEL_VOCAB,
                   help="model vocab size; default 151936 = Qwen padded dim.")
    p.add_argument("--d", type=int, default=MODEL_D,
                   help="model hidden width; default 512 (~100M params).")
    p.add_argument("--n-layers", type=int, default=MODEL_N_LAYERS,
                   help="number of HybridBlock layers.")
    p.add_argument("--loop-depth", type=int, default=MODEL_LOOP_DEPTH,
                   help="LoopLM Ouro recursion depth.")
    p.add_argument("--ffn-ratio", type=float, default=MODEL_FFN_RATIO,
                   help="FFN expansion ratio (8.0 default).")
    p.add_argument("--sparsity", type=float, default=MODEL_SPARSITY,
                   help="STDP plasticity-aware sparsity.")
    p.add_argument("--lr", type=float, default=LR,
                   help="peak learning rate.")
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
    p.add_argument("--kd-every", type=int, default=4,
                   help="run the (expensive) teacher forward every N steps "
                        "to amortise its cost; on KD-skip steps we use pure "
                        "base_loss (NOT (1-alpha)*base_loss) so the LM "
                        "gradient isn't silently scaled down by 0.3x on 75%% "
                        "of steps")
    p.add_argument("--kd-temperature", type=float, default=4.0,
                   help="softmax temperature T; KD scaled by T*T")
    p.add_argument("--kd-chunk", type=int, default=0,
                   help="MASTER_PLAN.md §6 P2/P13: KD chunk size override. "
                        "0 (default) = auto-tune from torch.cuda.mem_get_info() "
                        "with 50%% headroom. >0 forces fixed chunk size. "
                        "Only used when --kd-topk == 0 (full-vocab path).")
    p.add_argument("--kd-topk", type=int, default=2048,
                   help="Top-K teacher softmax for KD (memory-bounded). "
                        "Computes softmax over top-K teacher logits per row "
                        "instead of full vocab; student log_softmax restricted "
                        "to the same K indices. Captures ~99.99%% of mass at "
                        "K=2048 (BitNet/DistilBERT recipe). Memory at bs=80 "
                        "seq=256 V=151936 fp32: full=12GB -> top-2048=167MB "
                        "(70x less). Set 0 to fall back to full-vocab "
                        "chunked-softmax path (controlled by --kd-chunk). "
                        "See docs/PERF_KNOBS.md for the math test.")
    p.add_argument("--teacher-fallback-ckpt", default="",
                   help="path to a self-distill teacher ckpt (used if HF teacher load fails)")
    # ---- Phase-gated opt-in components (default OFF, see docs/PHASE_TRAINING.md) ----
    p.add_argument("--modal-list", default="",
                   help="comma-separated modalities to enable for contrastive aux "
                        "(e.g. 'image,audio'); empty = no multimodal aux")
    p.add_argument("--modal-data-dir", default="",
                   help="root directory holding pre-tokenised modal samples "
                        "(image/, audio/ subdirs of *.pt); empty = mixin no-ops")
    p.add_argument("--modal-alpha", type=float, default=0.05,
                   help="weight of InfoNCE contrastive aux loss (text vs modal)")
    p.add_argument("--self-learn-ttt", action="store_true", default=False,
                   help="at val time, TTT-adapt on top-K high-CE examples and "
                        "log lift; weights restored after probe (training "
                        "trajectory unchanged unless this flag is on)")
    p.add_argument("--self-learn-k", type=int, default=8,
                   help="K = number of hardest val examples to probe per eval")
    p.add_argument("--ttt-val-fraction", type=float, default=0.8,
                   help="P3 (MASTER_PLAN.md §6): fraction of val chunks "
                        "routed to the TTT side. The remaining (1 - frac) "
                        "goes to a holdout side that --self-learn-ttt "
                        "NEVER sees, so its ppl is the leak-free signal "
                        "phase_manager gates on. Default 0.8 (4:1 split).")
    p.add_argument("--curiosity-weight", type=float, default=0.0,
                   help="weight of 6-signal curiosity loss; 0 = disabled. "
                        "Recommended ~0.05 once base ppl<250 (Phase 1)")
    # ---- Reliability hooks (default-safe) ----
    p.add_argument("--phase-aware", action="store_true", default=False,
                   help="poll out_dir/.phase every 100 steps; on phase change "
                        "save ckpt + sys.exit(101) so an outer relauncher can "
                        "re-spawn with new flags. Default OFF -- doesn't affect "
                        "the default --warmstart=... --kd-weight=0.3 launch.")
    p.add_argument("--honest-eval", action="store_true", default=True,
                   help="dump 5 EN + 5 ZH chat samples every EVAL_EVERY steps "
                        "to honest_eval.jsonl (catches 'ppl-going-down but model "
                        "is word-salad' hallucination). On by default; takes "
                        "~5s per eval cycle on A100; never crashes the trainer.")
    p.add_argument("--no-honest-eval", action="store_false", dest="honest_eval",
                   help="disable the honest-eval hook")
    # P1: PLIF homeostatic threshold control (default ON).
    p.add_argument("--no-plif-homeostasis", action="store_true", default=False,
                   help="disable PLIF threshold homeostasis (P1). When ON "
                        "(default), the trainer calls "
                        "PLIFCell.homeostatic_step every 50 steps and "
                        "clamp_threshold every 100 steps to prevent "
                        "threshold drift past the input distribution. "
                        "Set this flag to A/B-disable for diagnosis.")
    # ---- Perf knobs (2026-05-01; see docs/PERF_KNOBS.md) ------------------
    p.add_argument("--z-loss-topk", type=int, default=2048,
                   help="top-K logits used for sparse z-loss logsumexp; "
                        "0 disables (full-vocab path). Default 2048 captures "
                        ">=99.99%% of softmax mass post-warmup at vocab=151936 "
                        "and shrinks the (B*T, V, fp32) intermediate ~74x. "
                        "Math is regularization-only (z-loss penalises large "
                        "logsumexp); the top-K approximation is a rigorous "
                        "lower bound. See tests/integration/test_sparse_z_loss.py.")
    p.add_argument("--kd-async-teacher", action="store_true", default=False,
                   help="overlap teacher forward with student backward on a "
                        "side CUDA stream. Teacher is frozen (eval()) so the "
                        "side stream has no autograd interaction; we sync "
                        "before computing KL. ~5-10%% step wins on bs=80. "
                        "Default OFF -- enable after validating no NaN drift "
                        "vs the synchronous path on your specific GPU/driver.")
    p.add_argument("--torch-compile", default="off",
                   choices=["off", "reduce-overhead", "max-autotune"],
                   help="wrap model.forward in torch.compile() at trainer "
                        "init. 'reduce-overhead' is the safer choice for the "
                        "PLIFCell sequence loop (no CUDA-graph capture under "
                        "dynamic seq_len). 'max-autotune' may NaN under bf16 "
                        "and triton_block backend; rolls back to 'off' "
                        "automatically on compile failure. Default OFF.")
    # ---- P24 (MASTER_PLAN.md §6) data-shuffle ----------------------------
    # Three Run 3a/3b/3c divergences at step ~2500 traced to deterministic
    # parquet ordering (see feedback_data_ordering_divergence_2026q2.md).
    # Default is now ON (10000) — train stream only; val stays deterministic.
    p.add_argument("--shuffle-buffer", type=int, default=10000,
                   help="P24: streaming Fisher-Yates reservoir size for the "
                        "TRAIN ParquetTokenStream. <=1 disables (legacy "
                        "deterministic order — caused divergence at step "
                        "~2500 in three runs). Default 10000 covers ~100x "
                        "the per-epoch correlation window at WT-103 row "
                        "rate and adds <40MB RAM. Validation stream stays "
                        "deterministic regardless.")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="P24: deterministic seed for the train-stream "
                        "reservoir RNG and per-epoch file-order shuffle. "
                        "Same seed + same buffer => identical yield order, "
                        "so reruns reproduce token sequence exactly.")
    # ---- Perf v2 knobs (2026-05-01; see docs/PERF_KNOBS.md) ---------------
    # Background-thread dataloader prefetch + pinned-memory H2D overlap.
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="size of the dataloader prefetch queue. >=2 spawns "
                        "a daemon producer thread that pre-builds batches "
                        "in pinned memory while the GPU step runs. 0/1 "
                        "keeps the legacy single-thread iterator (each "
                        "next() blocks on tokenizer + parquet decode). "
                        "Pairs with --pin-memory for the H2D overlap to "
                        "actually fire. Default 2 (~10-20%% step win on "
                        "the 21k tok/s baseline).")
    p.add_argument("--pin-memory", action="store_true", default=True,
                   help="allocate yielded (tokens_in, tokens_out) tensors "
                        "in pinned host memory so the trainer's "
                        "x.to(device, non_blocking=True) actually overlaps "
                        "with compute. Silent no-op without CUDA. Default "
                        "ON when the underlying torch build supports it.")
    p.add_argument("--no-pin-memory", action="store_false", dest="pin_memory",
                   help="disable pinned-memory dataloader output (use on "
                        "machines where pinned RAM is contested or torch "
                        "build doesn't support it).")
    # Adaptive KD frequency: scale --kd-every with running student-teacher
    # CE gap so converged students don't pay the teacher-forward tax.
    p.add_argument("--kd-every-adaptive", action="store_true", default=False,
                   help="recompute --kd-every every 100 steps from the "
                        "running mean CE of the last 50 KD-off steps. "
                        "Big gap (early) -> halve --kd-every (more KD); "
                        "small gap (late) -> double or quadruple it (KD "
                        "adds little, save ~50%% of step time). Default "
                        "OFF; explicit opt-in. Anchor target is teacher CE "
                        "estimate via --kd-teacher-ce-estimate.")
    p.add_argument("--kd-teacher-ce-estimate", type=float, default=4.5,
                   help="rough teacher-on-train-distribution CE used as "
                        "the gap anchor for --kd-every-adaptive. 4.5 is a "
                        "reasonable Qwen2.5-0.5B / wt103 estimate; "
                        "lower = more aggressive KD-frequency reduction.")
    # Speculative: fused PLIF surrogate backward Triton kernel.
    p.add_argument("--triton-fused-backward", action="store_true", default=False,
                   help="opt into the fused CfC+PLIF surrogate forward+"
                        "backward Triton kernel. Stubbed in "
                        "synapforge/backends/triton_fused_backward.py "
                        "(NotImplementedError on enable). Default OFF; "
                        "flip on once the kernel lands.")
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
        # eos_token_id is the right anchor for any HF tokenizer (Qwen,
        # GPT-2, etc); only fall back to 0 if eos is genuinely missing.
        eos = getattr(tokenizer, "eos_token_id", None)
        ids = [int(eos) if eos is not None else 0]
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




def _sparse_z_loss(logits: "torch.Tensor", k: int = 2048) -> "torch.Tensor":
    """logsumexp over top-k logits as a sparse approximation of the full one.

    The z-loss term ``(logsumexp(logits))**2`` is a regularizer (PaLM/Gemma
    style) that pulls the partition function toward 1. Materialising
    ``logsumexp`` over the full vocab forces a ``(B*T, V, fp32)`` tensor
    that costs ~9.5 GiB at bs=64 / V=151936 / seq=256 -- the binding
    constraint that capped us at bs=64 on A800-80GB.

    Top-K logsumexp is a tight lower bound on the full one
    (``logsumexp_K = log sum_{i in top-k} exp(x_i) <= log sum_i exp(x_i)``).
    The gap ``log(1 + sum_{i not in top-k} exp(x_i - x_max) / sum_{top-k}
    exp(x_i - x_max))`` is bounded by ``log(1 + (V-K) * exp(x_K - x_max) /
    sum_{top-k} exp(x_i - x_max))``. Empirically at vocab=151936 and any
    reasonable post-warmup logit distribution, top 2048 captures >=99.99%%
    of the softmax mass, so the gap is sub-1e-4 nats per row and the
    z-loss-squared error is sub-1e-8 nats^2 -- numerically irrelevant
    next to the CE loss (~5 nats early, ~2 nats late).

    For this to be safely a regularizer (and not bias the partition):
    * The penalty pulls log-Z DOWN. Underestimating log-Z pulls less
      strongly, which is the safe direction (no spurious gradient bias).
    * Returns a (B*T,) tensor matching the full-vocab path's shape.

    Args:
        logits: (B*T, V) fp32 (already flattened/upcast by caller).
        k: number of top logits to keep. ``k <= 0`` falls through to the
           full-vocab path (caller's responsibility -- see main()).
    Returns:
        log_z of shape (B*T,) -- same as ``torch.logsumexp(logits, dim=-1)``.
    """
    if k <= 0 or k >= logits.size(-1):
        return torch.logsumexp(logits, dim=-1)
    # logits.topk is O(V log K) and runs entirely on-device; no extra alloc
    # of a (V,) bool mask, so the (B*T, V) tensor never crystallises in fp32.
    top_k_vals, _ = logits.topk(k, dim=-1)
    return torch.logsumexp(top_k_vals, dim=-1)


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
                vocab=151936, d=512, n_layers=10, loop_depth=1,
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


# MASTER_PLAN.md §6 P2/P13 -- one-shot guard so the auto-tune banner
# only prints on the first call (avoids spamming train.log every step).
_KD_CHUNK_BANNER_PRINTED = False
_KD_TOPK_BANNER_PRINTED = False


def _kd_chunk_size(batch_size: int, seq_len: int, vocab: int,
                   headroom: float = 0.5) -> int:
    """Auto-pick KD chunk size from current GPU free VRAM.

    MASTER_PLAN.md §6 P2 + P13: the prior fixed ``chunk = bs // 4`` OOM'd
    at bs=128 / vocab=151936 on A800-80GB because the (B*T, V, fp32)
    intermediate for ``logsumexp`` + KL is ~18.55 GiB. This sizes the
    chunk so that intermediate fits in ``headroom`` * free_vram.

    Each chunk materializes ``(chunk * seq_len, vocab)`` in fp32 for
    log_softmax + KL, so per-row cost is ``seq_len * vocab * 4`` bytes.
    Reserve ``headroom`` fraction of free mem; size chunk to fit. Floor
    at 1, cap at ``batch_size`` (so VRAM-rich cards behave like the
    pre-fix single-pass path).
    """
    if not torch.cuda.is_available():
        return max(1, batch_size // 4)  # CPU fallback: keep prior behavior
    free_b, _ = torch.cuda.mem_get_info()
    budget_b = int(free_b * headroom)
    per_row_b = max(1, int(seq_len) * int(vocab) * 4)  # fp32 intermediate
    chunk = max(1, min(int(batch_size), budget_b // per_row_b))
    return chunk


def _kd_topk_loss(student_logits, teacher_logits, T: float, k: int):
    """Memory-bounded top-K teacher softmax KD.

    Math: take top-K teacher logits per row, compute teacher softmax
    over the top-K only (renormalised), gather student logits at the
    same K indices and run log_softmax over those K. KL is then
    `-sum_k top_p * student_logp` averaged over (B, T) and scaled by
    T**2 (Hinton). Memory is ``(B, T, K) * 4`` bytes for fp32 instead
    of ``(B, T, V) * 4`` -- at bs=80, seq=256, V=151936, K=2048 that's
    167 MiB vs 12 GiB for the full path (~70x reduction).

    Why this is mathematically defensible: BitNet/DistilBERT/SmolLM
    all use this exact trick. At V=151936, the top-2048 captures
    >=99.99% of the softmax mass once the model is past random init,
    so the KL approximation error is dominated by the renormalisation
    of the residual mass which the test harness pins below 5%.

    Edge cases:
        * k >= V: degrades to the full-vocab KL (test asserts equality).
        * k == 1: still finite (log_softmax over a single element is 0,
          KL term reduces to 0 + the log-renormalisation; doesn't NaN).
    """
    # Pin K to vocab size (handles k=2048 with V=1024 toy vocab).
    V = student_logits.size(-1)
    k = max(1, min(int(k), V))
    # Top-K teacher logits + their vocab indices.
    top_vals, top_idx = teacher_logits.detach().topk(k, dim=-1)  # (B, T, k)
    # Teacher distribution restricted to those K indices, renormalised.
    top_p = F.softmax(top_vals.float() / T, dim=-1)              # (B, T, k)
    # Gather student logits at the same K indices and run log_softmax
    # over only those K -- this is the key memory win.
    s_top = student_logits.gather(-1, top_idx)                   # (B, T, k)
    s_logp = F.log_softmax(s_top.float() / T, dim=-1)            # (B, T, k)
    # KL term: -sum_k top_p * s_logp; subtract the constant entropy
    # of top_p so the loss is the true KL not just the cross-entropy.
    # F.kl_div(s_logp, top_p) computes sum top_p*(log top_p - s_logp)
    # in one pass. reduction='batchmean' divides by B*T (leading dims).
    kl = F.kl_div(s_logp, top_p, reduction='sum')
    n_tokens = top_p.size(0) * top_p.size(1)
    n_tokens = max(n_tokens, 1)
    return (kl / n_tokens) * (T * T)


def _kd_loss(student_logits, teacher_logits, T: float = 4.0,
             chunk_override: int = 0, topk: int = 2048):
    """KL(student_logp || teacher_p), memory-bounded for large vocab.

    Returns a scalar token-mean (over batch * time) scaled by T**2 (Hinton).

    Two paths:
      * ``topk > 0`` (default 2048): top-K teacher softmax. Memory
        bounded at ``(B, T, K) * 4`` bytes regardless of vocab. ~70x
        less memory than full-vocab at V=151936, K=2048. See
        ``_kd_topk_loss`` for math.
      * ``topk == 0``: full-vocab chunked softmax (legacy path).
        ``chunk_override``: if > 0, use that chunk size verbatim (CLI
        ``--kd-chunk N`` path). If 0/absent, auto-tune from current GPU
        free VRAM via ``_kd_chunk_size``.
    """
    global _KD_CHUNK_BANNER_PRINTED, _KD_TOPK_BANNER_PRINTED
    V = student_logits.size(-1); V_t = teacher_logits.size(-1)
    # Vocabulary mismatch (e.g., student=Qwen 151643, teacher=GPT-2 50257).
    # Truncate to common prefix; both vocabs must share a token-id ordering
    # over [0, min(V,V_t)). Caller is responsible for ensuring this.
    if V_t > V:
        teacher_logits = teacher_logits[..., :V]
    elif V > V_t:
        student_logits = student_logits[..., :V_t]
    vocab = student_logits.size(-1)

    # ---- top-K path (default, memory-bounded) ----
    if topk and topk > 0:
        if not _KD_TOPK_BANNER_PRINTED:
            try:
                bs = student_logits.size(0)
                seq = (student_logits.size(1)
                       if student_logits.dim() >= 3 else 1)
                k_eff = max(1, min(int(topk), vocab))
                full_b = bs * seq * vocab * 4
                topk_b = bs * seq * k_eff * 4
                savings_gb = (full_b - topk_b) / (1024 ** 3)
                print(f"[kd] using top-{k_eff} softmax "
                      f"(saves {savings_gb:.2f} GB vs full-vocab "
                      f"V={vocab})", flush=True)
            except Exception:
                pass
            _KD_TOPK_BANNER_PRINTED = True
        return _kd_topk_loss(student_logits, teacher_logits, T, topk)

    # ---- legacy full-vocab chunked path (topk == 0) ----
    bs = student_logits.size(0)
    seq = student_logits.size(1) if student_logits.dim() >= 3 else 1
    if chunk_override and chunk_override > 0:
        chunk = max(1, min(int(chunk_override), bs))
    else:
        chunk = _kd_chunk_size(bs, seq, vocab)
    if not _KD_CHUNK_BANNER_PRINTED:
        try:
            if torch.cuda.is_available():
                free_b, _tot_b = torch.cuda.mem_get_info()
                free_gb = free_b / (1024 ** 3)
                print(f"[kd] chunk={chunk} (free={free_gb:.1f}GB, "
                      f"vocab={vocab})", flush=True)
            else:
                print(f"[kd] chunk={chunk} (cpu, vocab={vocab})", flush=True)
        except Exception:
            pass
        _KD_CHUNK_BANNER_PRINTED = True
    total = 0.0
    n_tokens = 0
    for i in range(0, bs, chunk):
        sl = student_logits[i:i + chunk]
        tl = teacher_logits[i:i + chunk].detach()
        slp = F.log_softmax(sl.float() / T, dim=-1)
        tp = F.softmax(tl.float() / T, dim=-1)
        # Reduce 'sum' then divide by n_tokens at the end -- gives a true
        # token-mean even when the final chunk is smaller than the rest
        # (was a bias source in the per-chunk-then-average path).
        total = total + F.kl_div(slp, tp, reduction='sum')
        n_tokens += sl.size(0) * sl.size(1)
    n_tokens = max(n_tokens, 1)
    return (total / n_tokens) * (T * T)

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

    # P9 / smoke overrides: pull architecture + cadence values from args
    # so the trainer can run on a tiny model (e.g. d=128, n_layers=2) and
    # save every 50 steps. Defaults preserve the prior 100M behaviour.
    seq_len = int(args.seq_len)
    save_every = int(args.save_every)
    eval_every = int(args.eval_every)
    log_every = int(args.log_every)
    peak_lr = float(args.lr)

    print(f"device={DEVICE} dtype={DTYPE} out={out_dir} backend={backend_name}")
    print(f"steps={n_steps} bs={args.batch_size} seq={seq_len} lr={peak_lr}")

    # ---------------- model ----------------
    model = build_synapforge_100m(
        vocab=int(args.vocab), d=int(args.d), n_layers=int(args.n_layers),
        loop_depth=int(args.loop_depth), max_seq=seq_len,
        ffn_ratio=float(args.ffn_ratio), sparsity=float(args.sparsity),
        dropout=MODEL_DROPOUT,
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

    # ---------------- torch.compile (default OFF) ----------------
    # 2026-05-01 perf knob: wrap model.forward in torch.compile() for the
    # PLIFCell sequence loop + lm_head + FFN paths. triton_block backend
    # already fuses Liquid+PLIF; torch.compile here covers the non-fused
    # surface area. Roll back to OFF on compile failure rather than crash
    # the trainer (perf knob, not a correctness gate).
    if args.torch_compile != "off":
        try:
            mode = args.torch_compile  # 'reduce-overhead' | 'max-autotune'
            _log(f"[torch.compile] wrapping model with mode={mode!r}")
            # `dynamic=True` is necessary because eval batches may be
            # smaller than train batches; static would force recompile.
            model = torch.compile(model, mode=mode, dynamic=True)
            _log(f"[torch.compile] compiled (mode={mode!r}, dynamic=True)")
        except Exception as exc:
            import traceback
            _log(f"[torch.compile] FAILED ({exc!r}); rolling back to off")
            for line in traceback.format_exc().splitlines():
                _log(f"  {line}")
            args.torch_compile = "off"
    else:
        _log("[torch.compile] disabled (default; pass --torch-compile "
             "reduce-overhead to enable)")

    # ---------------- KD async teacher stream (default OFF) ------------
    # 2026-05-01 perf knob: when on, teacher forward pushes onto a side
    # CUDA stream so it can overlap with the student backward (which still
    # owns stream 0). Teacher is frozen so the side stream has no autograd
    # interaction; we sync before KL. Default OFF.
    kd_teacher_stream = None
    if args.kd_async_teacher and DEVICE == "cuda" and teacher is not None:
        try:
            kd_teacher_stream = torch.cuda.Stream()
            _log("[kd-async] enabled: teacher forward on side stream "
                 f"(stream id={id(kd_teacher_stream)})")
        except Exception as exc:
            _log(f"[kd-async] stream alloc failed ({exc!r}); disabled")
            kd_teacher_stream = None
    elif args.kd_async_teacher:
        _log("[kd-async] requested but unavailable "
             f"(device={DEVICE}, teacher={'set' if teacher else 'None'})")

    # ---------------- optimizer ----------------
    optim = build_optimizer(model, lr=peak_lr, weight_decay=WEIGHT_DECAY)
    print(f"optimizer: {type(optim).__name__} lr={peak_lr} wd={WEIGHT_DECAY}")

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
    # P9: data globs and tokenizer are CLI-overridable so the smoke test
    # can point at a tiny synth parquet + the gpt2 tokenizer (no rental
    # path required). Defaults preserve the WT-103 + Qwen rental setup.
    # P24 (MASTER_PLAN.md §6): TRAIN stream gets a streaming reservoir
    # shuffle; VAL stream stays deterministic so eval ppl is comparable
    # across runs.
    train_ds = ParquetTokenStream(args.data_glob, seq_len=seq_len,
                                  tokenizer_name=args.tokenizer_name,
                                  batch_size=args.batch_size, loop=True,
                                  shuffle_buffer=int(args.shuffle_buffer),
                                  shuffle_seed=int(args.shuffle_seed),
                                  prefetch_factor=int(args.prefetch_factor),
                                  pin_memory=bool(args.pin_memory))
    train_it = iter(train_ds)
    print(f"train stream: {train_ds!r}")

    # If --val-glob points at a missing path (smoke runs only have a single
    # parquet), reuse the train glob so eval still runs.
    _val_glob = args.val_glob if args.val_glob else args.data_glob
    try:
        val_ds = ParquetTokenStream(_val_glob, seq_len=seq_len,
                                    tokenizer_name=args.tokenizer_name,
                                    batch_size=args.batch_size, loop=False)
    except FileNotFoundError:
        _log(f"[data] val glob {_val_glob!r} not found; falling back to train glob for eval")
        val_ds = ParquetTokenStream(args.data_glob, seq_len=seq_len,
                                    tokenizer_name=args.tokenizer_name,
                                    batch_size=args.batch_size, loop=False)
    print(f"val stream:   {val_ds!r}")
    # P3 (MASTER_PLAN.md §6): split val into TTT-side (touched by
    # --self-learn-ttt inner step) and holdout-side (NEVER touched by
    # TTT). The holdout ppl is the honest, leak-free signal that
    # phase_manager gates on. See tests/integration/test_ttt_val_split.py.
    val_ds_ttt, val_ds_holdout = split_val_stream(
        val_ds, ttt_fraction=args.ttt_val_fraction, denom=5,
    )
    _log(f"val split: ttt_fraction={args.ttt_val_fraction:.2f} "
         f"(ttt rows mod 5 in {sorted(val_ds_ttt.keep_indices)}, "
         f"holdout rows mod 5 in {sorted(val_ds_holdout.keep_indices)})")

    from synapforge.huggingface_adapter import load_tokenizer
    tok = load_tokenizer(args.tokenizer_name)

    # ---------------- honest eval hook (default ON, fail-soft) ----------------
    eval_hook = None
    if args.honest_eval:
        if HonestEvalHook is None:
            _log("[honest-eval] hook unavailable (import failed at startup); skipping")
        else:
            try:
                eval_hook = HonestEvalHook(
                    model=model,
                    tokenizer=tok,
                    out_dir=out_dir,
                    every_steps=eval_every,
                    max_new_tokens=40,
                    device=DEVICE,
                )
                _log(f"[honest-eval] enabled: every {eval_every} steps, "
                     f"out={eval_hook.jsonl}")
            except Exception as exc:
                _log(f"[honest-eval] init failed (continuing without): {exc}")
                eval_hook = None
    else:
        _log("[honest-eval] disabled by --no-honest-eval")

    # ---------------- phase-aware polling state ----------------
    current_phase_id: int | None = None
    if args.phase_aware:
        if phase_signal is None:
            _log("[phase-aware] phase_signal module unavailable; flag is a no-op")
        else:
            initial = phase_signal.read_phase(out_dir)
            current_phase_id = initial.get("phase_id") if isinstance(initial, dict) else None
            _log(f"[phase-aware] enabled; initial phase_id={current_phase_id}")

    # ---------------- opt-in mixins (default OFF) ----------------
    # See docs/PHASE_TRAINING.md for when each phase becomes safe to enable.
    modal_mixin = None
    self_learn_mixin = None
    curiosity_mixin = None
    try:
        from synapforge.trainer_mixins import (
            MultimodalMixin, SelfLearnMixin, CuriosityMixin,
        )
        modal_list = tuple(
            m.strip() for m in args.modal_list.split(",") if m.strip()
        )
        if modal_list and args.modal_data_dir:
            try:
                modal_mixin = MultimodalMixin(
                    model, modal_list=modal_list,
                    modal_data_dir=args.modal_data_dir,
                    hidden=model.d, alpha=args.modal_alpha,
                )
                _log(f"[mixin] MultimodalMixin enabled: {modal_list} "
                     f"alpha={args.modal_alpha} dir={args.modal_data_dir}")
            except Exception as exc:
                _log(f"[mixin] MultimodalMixin DISABLED: {exc}")
                modal_mixin = None
        else:
            _log("[mixin] multimodal: disabled (no --modal-list/--modal-data-dir)")
        if args.self_learn_ttt:
            try:
                self_learn_mixin = SelfLearnMixin(
                    model, optimizer=optim, k_failures=args.self_learn_k,
                )
                _log(f"[mixin] SelfLearnMixin enabled: K={args.self_learn_k} "
                     f"(eval-time only, weights restored)")
            except Exception as exc:
                _log(f"[mixin] SelfLearnMixin DISABLED: {exc}")
                self_learn_mixin = None
        else:
            _log("[mixin] self-learn-ttt: disabled")
        if args.curiosity_weight > 0:
            try:
                curiosity_mixin = CuriosityMixin(model, hidden=model.d)
                _log(f"[mixin] CuriosityMixin enabled: weight={args.curiosity_weight}")
            except Exception as exc:
                _log(f"[mixin] CuriosityMixin DISABLED: {exc}")
                curiosity_mixin = None
        else:
            _log("[mixin] curiosity: disabled (weight=0)")
    except Exception as exc:
        _log(f"[mixin] mixins import failed; continuing without: {exc}")

    # ---------------- training ----------------
    # P3: track BOTH val_ppl_ttt (set TTT trains on; drops artificially
    # after TTT inner step) and val_ppl_holdout (set TTT never sees;
    # honest signal phase_manager gates on). ``ppl_eval`` retained as
    # alias of holdout for legacy parsers/tooling.
    metrics = {"step": [], "loss": [], "step_ms": [], "tok_per_s": [],
               "ppl_eval": {}, "ppl_eval_ttt": {}, "ppl_eval_holdout": {},
               "samples": {}}
    t0 = time.time()
    last_log_t = t0
    cum_tok = 0
    sample_prompts = [
        "The first thing the agent did was",
        "In a quiet town nestled between",
        "Recurrent neural networks are",
    ]
    # ---- adaptive --kd-every state ----
    # ``args.kd_every`` is the user-provided BASE schedule. When the
    # adaptive flag is on, we keep a separate ``effective_kd_every``
    # that is recomputed every 100 steps from the running mean CE of
    # the last ``KD_ADAPT_WINDOW`` KD-OFF steps (KD-OFF only so the
    # measure isn't artificially pulled down by KD itself).
    effective_kd_every = int(args.kd_every)
    kd_off_ce_window: list[float] = []
    KD_ADAPT_WINDOW = 50    # rolling buffer of KD-OFF step CEs
    KD_ADAPT_PERIOD = 100   # recompute the schedule every N steps
    if args.kd_every_adaptive:
        _log(f"[kd-adaptive] enabled: base={args.kd_every} "
             f"teacher_ce_estimate={args.kd_teacher_ce_estimate:.2f} "
             f"window={KD_ADAPT_WINDOW} period={KD_ADAPT_PERIOD}")
    if args.triton_fused_backward:
        # Speculative knob — kernel is stubbed only.
        try:
            from synapforge.backends.triton_fused_backward import (
                enable_fused_backward,
            )
            enable_fused_backward()
        except NotImplementedError as exc:
            _log(f"[triton-fused-backward] disabled: {exc}; falling back "
                 f"to PyTorch autograd surrogate path")
        except Exception as exc:
            _log(f"[triton-fused-backward] import failed: {exc}; falling back")

    for step in range(1, n_steps + 1):
        cur_lr = lr_at(step, peak_lr, args.warmup, n_steps, args.lr_decay)
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
            # When a mixin needs the hidden state, take the encode -> head
            # path so we can pass `text_hidden` into the mixin without a
            # second forward. Default path (no mixin) is unchanged.
            need_hidden = (modal_mixin is not None
                           or curiosity_mixin is not None)
            if need_hidden:
                text_hidden = model.encode(x)              # (B, T, d)
                logits = F.linear(text_hidden, model.tok_embed.weight) \
                    if model.tie_lm_head else model.lm_head(text_hidden)
            else:
                text_hidden = None
                logits = model(x)
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_y = y.reshape(-1)
            ce_loss = F.cross_entropy(flat_logits, flat_y,
                                      label_smoothing=args.label_smoothing)
            # z-loss: penalize large logsumexp (numerical stability + softmax
            # temp control). 2026-05-01: default ``--z-loss-topk 2048`` uses
            # the sparse approximation so the (B*T, V, fp32) intermediate
            # never materialises -- this is what unlocks bs=64 -> bs=80 on
            # A800-80GB. Set --z-loss-topk 0 to recover the full-vocab path.
            log_z = _sparse_z_loss(flat_logits, k=int(args.z_loss_topk))
            z_loss = (log_z ** 2).mean()
            base_loss = ce_loss + args.z_loss_weight * z_loss

            if teacher is not None and args.kd_weight > 0:
                # Only run the (expensive) teacher forward on KD-active steps.
                # On KD-skip steps fall back to pure base_loss without the
                # (1-α) downweight that would otherwise scale the LM gradient
                # by 0.3× on 3/4 steps (effectively dropping LR; see review).
                # ``effective_kd_every`` == ``args.kd_every`` unless
                # --kd-every-adaptive is on, in which case it's
                # auto-tuned from the student-teacher CE gap (see
                # ``_adaptive_kd_every`` and the post-step update below).
                if step % effective_kd_every == 0:
                    if kd_teacher_stream is not None:
                        # Push teacher forward onto side stream so it can
                        # overlap with the student backward (which still runs
                        # on the default stream). Teacher is frozen + .eval()
                        # so the side stream needs no autograd interaction.
                        # We sync before KL to guarantee t_logits is ready.
                        kd_teacher_stream.wait_stream(
                            torch.cuda.current_stream())
                        with torch.cuda.stream(kd_teacher_stream), \
                                torch.no_grad():
                            t_logits = _teacher_logits(teacher, x)
                        torch.cuda.current_stream().wait_stream(
                            kd_teacher_stream)
                    else:
                        with torch.no_grad():
                            t_logits = _teacher_logits(teacher, x)
                    kd = _kd_loss(logits, t_logits, args.kd_temperature,
                                  chunk_override=args.kd_chunk,
                                  topk=args.kd_topk)
                    loss = (1.0 - args.kd_weight) * base_loss + args.kd_weight * kd
                else:
                    kd = torch.zeros((), device=logits.device)
                    loss = base_loss
                    # ---- adaptive --kd-every: track CE on KD-OFF steps ----
                    # We sample CE from KD-OFF steps only so the gap measure
                    # isn't artificially pulled down by the KD loss itself.
                    # ``ce_loss.item()`` triggers a host-sync but only on KD-
                    # OFF steps, which already pay no teacher-forward cost.
                    if args.kd_every_adaptive:
                        kd_off_ce_window.append(float(ce_loss.detach().item()))
                        if len(kd_off_ce_window) > KD_ADAPT_WINDOW:
                            kd_off_ce_window = kd_off_ce_window[-KD_ADAPT_WINDOW:]
            else:
                kd = torch.zeros((), device=logits.device)
                loss = base_loss

            # ---- opt-in mixin contributions (default OFF) ----
            modal_aux = torch.zeros((), device=logits.device)
            cur_aux = torch.zeros((), device=logits.device)
            if modal_mixin is not None and text_hidden is not None:
                try:
                    modal_aux = modal_mixin.contrastive_loss(text_hidden)
                    loss = loss + modal_aux
                except Exception as exc:
                    _log(f"[mixin] modal step {step} skipped: {exc}")
                    modal_aux = torch.zeros((), device=logits.device)
            if curiosity_mixin is not None and text_hidden is not None:
                try:
                    # use timestep t and t+1 as h_prev, h_next for ICM
                    h_prev = text_hidden[:, :-1].contiguous()
                    h_next = text_hidden[:, 1:].contiguous()
                    cur_state = {
                        "h_prev": h_prev,
                        "h_next": h_next,
                        "plif_modules": plif_cells,
                    }
                    cur_aux = curiosity_mixin.curiosity_loss(cur_state)
                    loss = loss + args.curiosity_weight * cur_aux
                except Exception as exc:
                    _log(f"[mixin] curiosity step {step} skipped: {exc}")
                    cur_aux = torch.zeros((), device=logits.device)

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
        cum_tok += args.batch_size * seq_len

        # ---- adaptive --kd-every: recompute schedule every N steps ----
        # Only mutates ``effective_kd_every``; ``args.kd_every`` (the
        # base) stays the user input. We need at least KD_ADAPT_WINDOW//2
        # samples in the running window to avoid noisy early flips.
        if (
            args.kd_every_adaptive
            and step % KD_ADAPT_PERIOD == 0
            and len(kd_off_ce_window) >= KD_ADAPT_WINDOW // 2
        ):
            mean_ce = sum(kd_off_ce_window) / len(kd_off_ce_window)
            new_kd_every = _adaptive_kd_every(
                mean_ce,
                teacher_ce_estimate=args.kd_teacher_ce_estimate,
                base=int(args.kd_every),
            )
            if new_kd_every != effective_kd_every:
                _log(f"[kd-adaptive] step={step} mean_ce={mean_ce:.3f} "
                     f"gap={mean_ce - args.kd_teacher_ce_estimate:+.2f} "
                     f"kd_every {effective_kd_every} -> {new_kd_every}")
                effective_kd_every = int(new_kd_every)

        if step % log_every == 0 or step == 1:
            tok_s = cum_tok / max(time.time() - t0, 1e-6)
            metrics["step"].append(step)
            metrics["loss"].append(float(loss.item()))
            metrics["step_ms"].append(step_ms)
            metrics["tok_per_s"].append(tok_s)
            mem_str = (
                f" mem_GB={torch.cuda.memory_allocated()/1e9:.2f}"
                if DEVICE == "cuda" else ""
            )
            mixin_str = ""
            if modal_mixin is not None:
                mixin_str += f" modal={float(modal_aux.detach()):.4f}"
            if curiosity_mixin is not None:
                mixin_str += f" cur={float(cur_aux.detach()):.4f}"
            _log(f"step {step:5d} loss={loss.item():.4f} ce={ce_loss.item():.3f} "
                 f"kd={kd.item():.3f} z={z_loss.item():.3f} lr={cur_lr:.5f} "
                 f"step_ms={step_ms:.1f} tok/s={tok_s:.0f}{mem_str}{mixin_str}")
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
                        f"of target {args.spike_target:.3f} (|delta|>0.05)"
                    )
                # P1: PLIF homeostatic threshold control.
                # Run 1 (April) ce 5.72->8.46 word-salad and Run 2 (May 1)
                # dead-10/10 spike rate were both rooted in unbounded
                # threshold drift past the input distribution. The legacy
                # all-dead auto-revive only fired in the worst case --
                # partial death never triggered. Replaced with:
                #   * every 50 steps: homeostatic_step toward spike_target
                #   * every 100 steps: clamp_threshold defensively bounds
                #     drift to [0.005, 0.5].
                # Both run under torch.no_grad in PLIFCell methods so the
                # surrogate-gradient path is untouched.
                if not args.no_plif_homeostasis:
                    if step % 50 == 0:
                        thr_before = float(plif_cells[0].threshold.mean())
                        for m in plif_cells:
                            m.homeostatic_step(
                                rate_mean,
                                target_rate=args.spike_target,
                                gain=0.01,
                            )
                        thr_after = float(plif_cells[0].threshold.mean())
                        if abs(thr_after - thr_before) > 1e-6:
                            _log(
                                f"  [PLIF-REVIVE] homeostatic step "
                                f"rate={rate_mean:.3f} -> thr "
                                f"{thr_before:.4f} -> {thr_after:.4f}"
                            )
                    if step % 100 == 0:
                        thr_before = float(plif_cells[0].threshold.mean())
                        for m in plif_cells:
                            m.clamp_threshold(0.005, 0.5)
                        thr_after = float(plif_cells[0].threshold.mean())
                        if abs(thr_after - thr_before) > 1e-6:
                            _log(
                                f"  [PLIF-REVIVE] clamp [0.005, 0.5] thr "
                                f"{thr_before:.4f} -> {thr_after:.4f}"
                            )

        if step % save_every == 0:
            ckpt_path = os.path.join(out_dir, f"step_{step:06d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optim_state": optim.state_dict(),
                "step": step,
                "loss": float(loss.item()),
                "n_params": n_params,
                "lr": cur_lr,
                "config": _build_config_dict(args),
            }, ckpt_path)
            _log(f"saved ckpt {ckpt_path}")

        # ---- phase-aware polling (default OFF) ----
        # Cheap: read_phase is a stat + ~1KB read. Done every 100 steps.
        if (
            args.phase_aware
            and phase_signal is not None
            and step % 100 == 0
        ):
            try:
                payload = phase_signal.consume_phase(out_dir)
                if payload is not None:
                    new_id = payload.get("phase_id")
                    if new_id is not None and new_id != current_phase_id:
                        _log(f"[phase-aware] phase change "
                             f"{current_phase_id} -> {new_id}: "
                             f"{payload.get('phase_name')!r} flags="
                             f"{payload.get('next_phase_flags')}")
                        ckpt_path = os.path.join(
                            out_dir, f"phase_change_step_{step:06d}.pt"
                        )
                        try:
                            torch.save({
                                "model": model.state_dict(),
                                "optim_state": optim.state_dict(),
                                "step": step,
                                "loss": float(loss.item()),
                                "n_params": n_params,
                                "lr": cur_lr,
                                "phase_id": new_id,
                                "phase_name": payload.get("phase_name"),
                                "config": _build_config_dict(args),
                            }, ckpt_path)
                            _log(f"[phase-aware] saved ckpt {ckpt_path}")
                        except Exception as exc:
                            _log(f"[phase-aware] ckpt save failed: {exc}")
                        # Persist final metrics so the relauncher has them.
                        try:
                            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                                json.dump(metrics, f, indent=2)
                        except Exception:
                            pass
                        _log("[phase-aware] sys.exit(101) -- relauncher should "
                             "re-spawn with new flags appended")
                        sys.exit(101)
            except SystemExit:
                raise
            except Exception as exc:
                _log(f"[phase-aware] poll failed (continuing): {exc}")

        if step % eval_every == 0 or step == n_steps:
            # P3: TWO val ppls. ttt is the set --self-learn-ttt trains
            # on (artificially low after TTT inner step). holdout is
            # the leak-free signal phase_manager gates on. Both are
            # logged to train.log and persisted in metrics.json.
            ppl_ttt = evaluate(
                model, iter(val_ds_ttt), n_batches=16, plif_cells=plif_cells,
            )
            ppl_holdout = evaluate(
                model, iter(val_ds_holdout), n_batches=16,
                plif_cells=None,  # plif rate already logged on the ttt pass
            )
            metrics["ppl_eval_ttt"][step] = ppl_ttt
            metrics["ppl_eval_holdout"][step] = ppl_holdout
            # Legacy alias: phase_manager parses the FIRST regex match per
            # ckpt; keep ``ppl_eval`` pointing at the holdout (honest)
            # number so older parsers do the right thing by default.
            metrics["ppl_eval"][step] = ppl_holdout
            _log(
                f"VAL step {step}: val_ppl_ttt={ppl_ttt:.2f} "
                f"val_ppl_holdout={ppl_holdout:.2f} (honest)"
            )
            # Honest eval (default ON, fail-soft). Dumps real chat samples to
            # honest_eval.jsonl alongside this run -- catches the case where
            # ppl monotonically decreases but the model emits word-salad.
            if eval_hook is not None:
                try:
                    eval_hook.maybe_eval(
                        step,
                        float(ppl_holdout) if ppl_holdout is not None else None,
                    )
                except Exception as exc:  # never let eval kill training
                    _log(f"  [honest-eval] step {step} skipped: {exc}")
            # Opt-in: probe TTT lift on top-K hardest val examples.
            # CRITICAL: only the TTT side is exposed -- holdout stays clean.
            if self_learn_mixin is not None:
                try:
                    val_it_probe = iter(val_ds_ttt)
                    xv, yv = next(val_it_probe)
                    xv = xv.to(DEVICE)
                    yv = yv.to(DEVICE)
                    rec = self_learn_mixin.adapt_on_failures(xv, yv)
                    metrics.setdefault("self_learn_lift", {})[step] = rec
                    _log(f"  [self-learn] k={rec['k']} "
                         f"ce_before={rec['before']:.3f} "
                         f"ce_after={rec['after']:.3f} "
                         f"lift={rec['lift']:+.4f}")
                except Exception as exc:
                    _log(f"  [self-learn] probe failed: {exc}")
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
        "config": _build_config_dict(args),
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
