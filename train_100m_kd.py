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

BATCH_SIZE = 32
SEQ_LEN = 256
LR = 3e-4
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
    """
    return {
        "vocab": MODEL_VOCAB,
        "d": MODEL_D,
        "n_layers": MODEL_N_LAYERS,
        "loop_depth": MODEL_LOOP_DEPTH,
        # max_seq is what the trainer was actually built with -- not the arg
        # default, since args.max_seq may not exist on this trainer.
        "max_seq": SEQ_LEN,
        "ffn_ratio": MODEL_FFN_RATIO,
        "sparsity": MODEL_SPARSITY,
        "dropout": MODEL_DROPOUT,
        "tie_lm_head": MODEL_TIE_LM_HEAD,
    }


def _log(msg: str) -> None:
    """Module-level log shim. main() rebinds a closure that ALSO appends to
    log_lines, but evaluate() and the other module-level helpers may run
    before/after main()'s closure exists, so we always have a safe fallback."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    p.add_argument("--kd-every", type=int, default=4,
                   help="run the (expensive) teacher forward every N steps "
                        "to amortise its cost; on KD-skip steps we use pure "
                        "base_loss (NOT (1-alpha)*base_loss) so the LM "
                        "gradient isn't silently scaled down by 0.3x on 75%% "
                        "of steps")
    p.add_argument("--kd-temperature", type=float, default=4.0,
                   help="softmax temperature T; KD scaled by T*T")
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


def _kd_loss(student_logits, teacher_logits, T: float = 4.0):
    """Chunked KL(student_logp || teacher_p), avoids OOM on large vocab.

    Returns a scalar token-mean (over batch * time) scaled by T**2 (Hinton).
    """
    V = student_logits.size(-1); V_t = teacher_logits.size(-1)
    # Vocabulary mismatch (e.g., student=Qwen 151643, teacher=GPT-2 50257).
    # Truncate to common prefix; both vocabs must share a token-id ordering
    # over [0, min(V,V_t)). Caller is responsible for ensuring this.
    if V_t > V:
        teacher_logits = teacher_logits[..., :V]
    elif V > V_t:
        student_logits = student_logits[..., :V_t]
    bs = student_logits.size(0); chunk = max(1, bs // 4)
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

    print(f"device={DEVICE} dtype={DTYPE} out={out_dir} backend={backend_name}")
    print(f"steps={n_steps} bs={args.batch_size} seq={SEQ_LEN} lr={LR}")

    # ---------------- model ----------------
    model = build_synapforge_100m(
        vocab=MODEL_VOCAB, d=MODEL_D, n_layers=MODEL_N_LAYERS,
        loop_depth=MODEL_LOOP_DEPTH, max_seq=SEQ_LEN,
        ffn_ratio=MODEL_FFN_RATIO, sparsity=MODEL_SPARSITY,
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
                    every_steps=EVAL_EVERY,
                    max_new_tokens=40,
                    device=DEVICE,
                )
                _log(f"[honest-eval] enabled: every {EVAL_EVERY} steps, "
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
            # z-loss: penalize large logsumexp (numerical stability + softmax temp control)
            log_z = torch.logsumexp(flat_logits, dim=-1)
            z_loss = (log_z ** 2).mean()
            base_loss = ce_loss + args.z_loss_weight * z_loss

            if teacher is not None and args.kd_weight > 0:
                # Only run the (expensive) teacher forward on KD-active steps.
                # On KD-skip steps fall back to pure base_loss without the
                # (1-α) downweight that would otherwise scale the LM gradient
                # by 0.3× on 3/4 steps (effectively dropping LR; see review).
                if step % args.kd_every == 0:
                    with torch.no_grad():
                        t_logits = _teacher_logits(teacher, x)
                    kd = _kd_loss(logits, t_logits, args.kd_temperature)
                    loss = (1.0 - args.kd_weight) * base_loss + args.kd_weight * kd
                else:
                    kd = torch.zeros((), device=logits.device)
                    loss = base_loss
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
                # Auto-revive dead PLIF: if every cell is silent for 1000+ steps,
                # the leaky-to-steady form has settled below threshold for all
                # channels. Scale threshold down 10% per check until rate>=0.01.
                # Bounded by LearnableThreshold min_val=1e-3 spirit.
                if step >= 1000 and step % 200 == 0 and n_dead == len(rates):
                    with torch.no_grad():
                        for m in plif_cells:
                            m.threshold.mul_(0.9).clamp_(min=5e-3)
                    new_thr = float(plif_cells[0].threshold.mean())
                    _log(f"  [PLIF-REVIVE] all cells dead; thr×0.9 -> mean={new_thr:.4f}")

        if step % SAVE_EVERY == 0:
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

        if step % EVAL_EVERY == 0 or step == n_steps:
            val_it = iter(val_ds)
            ppl = evaluate(model, val_it, n_batches=16, plif_cells=plif_cells)
            metrics["ppl_eval"][step] = ppl
            _log(f"VAL step {step}: ppl={ppl:.2f}")
            # Honest eval (default ON, fail-soft). Dumps real chat samples to
            # honest_eval.jsonl alongside this run -- catches the case where
            # ppl monotonically decreases but the model emits word-salad.
            if eval_hook is not None:
                try:
                    eval_hook.maybe_eval(step, float(ppl) if ppl is not None else None)
                except Exception as exc:  # never let eval kill training
                    _log(f"  [honest-eval] step {step} skipped: {exc}")
            # Opt-in: probe TTT lift on top-K hardest val examples.
            if self_learn_mixin is not None:
                try:
                    val_it_probe = iter(val_ds)
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
