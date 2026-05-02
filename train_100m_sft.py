"""train_100m_sft -- Phase 2 SFT trainer for synapforge_100m.

Goal
----
Switch from LM-only training (``train_100m_kd.py``) to instruction-tune
SFT in order to break the val ppl plateau (~1000-2000 on raw LM data
per scaling laws) toward <= 10. This is the trainer the Phase autopilot
(``docs/DEEP_MAINT_QUEUE.md`` H5 phase trigger table) expects to find
on disk; the LM trainer alone cannot push ppl past the natural-text
floor because raw-text CE has nothing to push down once the model
matches text statistics.

Key differences vs ``train_100m_kd.py``
---------------------------------------
1. **Data**: instruction parquets (alpaca-zh + Qwen-tokenized) read via
   :class:`synapforge.training.sft_loop.InstructionParquetStream`. The
   stream emits ``(tokens_in, tokens_out, loss_mask)`` triples; the mask
   is 1 only on response tokens (default) so the model learns the
   answer distribution, not the prompt distribution -- the standard
   Stanford-Alpaca / DeepSeek-V2 / Qwen-Chat recipe. Two on-disk schemas
   are auto-detected: ``prompt_input_ids+response_input_ids`` (preferred)
   or ``input_ids+response_mask`` (legacy compat with the v2.6 trainer).
2. **Loss**: :func:`synapforge.training.sft_loop.response_only_ce_loss`
   is masked CE. With ``--no-response-only-loss`` (ablation only) it
   degrades to full-CE on all non-pad positions, recovering the
   instruction-LM baseline -- documented as the off-path so we can A/B
   the mask's effect on ppl.
3. **No KD teacher**: SFT is a pure CE step. KD distillation already
   happened in Phase 1 (``train_100m_kd.py``); Phase 2 hardens the
   model on instruction-following format. Adding a teacher here would
   double-train the same signal at additional GPU cost.
4. **Warmstart**: explicit ``--warmstart <path>`` of a Phase 1 ckpt
   (Synap-1 Ultra ``step_X.pt`` or ``best_step_X.pt``). The ckpt format
   matches ``train_100m_kd.py`` exactly (model+optim_state+step+config),
   so warmstart is bidirectional: an SFT ckpt can warmstart back into
   the KD trainer (e.g. for a second LM pass after Phase 2 plateau breaks).
5. **Honest VAL**: 5% of the alpaca-zh stream is held out (deterministic
   row-modulo split) for in-domain ppl, plus an optional second parquet
   (``--cross-val``) for cross-domain ppl (e.g. wikitext-100). Both are
   evaluated under the same response-only mask so numbers are comparable
   across domains.

References
----------
* DEEP_MAINT_QUEUE H5 -- the phase trigger table that expects this file
  to exist; bumped by T9.4.
* memory ``v2.6 response-only loss masking`` -- the original
  response-mask landing on a 100M trainer (2026-04, task #214).
* memory ``feedback_50m_context_monotonic_quality.md`` -- inference-
  STDP claim that motivates breaking the LM-only ppl plateau.

CLI
---
Defaults are configured for the rental Phase 1 -> Phase 2 hand-off
launch. Synap-1 architecture flags from ``train_100m_kd.py`` carry the
same name and default so launch scripts can swap entry points by
changing the trainer module name only.
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

# Strip script dir from sys.path so ``import synapforge`` finds the outer
# package even when this file is in the worktree root (mirror of
# ``train_100m_kd.py`` P9 nested-pkg-shadow fix).
sys.path[:] = [p for p in sys.path if p not in ("/workspace/synapforge", "")]
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import synapforge as sf  # noqa: E402,F401  -- side-effects + sanity
from synapforge.surrogate import PLIFCell  # noqa: E402
from synapforge.huggingface_adapter import adv_warmstart  # noqa: E402
from synapforge.model_100m import build_synapforge_100m  # noqa: E402
from synapforge.optim import build_optimizer  # noqa: E402
from synapforge.training.sft_loop import (  # noqa: E402
    InstructionParquetStream,
    response_only_ce_loss,
)


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

OUT_DIR_DEFAULT = "/workspace/runs/synapforge_100m_sft"
WARM_CKPT_DEFAULT = "/workspace/runs/synap1_ultra/best_step.pt"
DATA_DEFAULT = "/workspace/data/alpaca_zh_qwen_tokenized.parquet"
# Cross-domain val parquet (optional). Empty string disables the second-
# domain probe; the in-domain holdout is always on.
CROSS_VAL_DEFAULT = "/workspace/data/wikitext_100_qwen_tokenized.parquet"

# 5000 steps * bs=16 * seq=512 ~= 40M tokens -- about 1 epoch over a
# 48k-row alpaca-zh shard. Sized so the run completes in ~6h on A800.
N_STEPS_DEFAULT = 5000
SAVE_EVERY = 250
EVAL_EVERY = 250
LOG_EVERY = 10

# bs=16, seq=512 is the SFT sweet spot on A800-80GB: response-only loss
# means a much higher fraction of token positions actually carry
# gradient (vs LM-only), so we get more signal per step at smaller bs.
BATCH_SIZE = 16
SEQ_LEN = 512
LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Architecture constants used to build the model (mirror train_100m_kd.py).
MODEL_VOCAB = 151936
MODEL_D = 512
MODEL_N_LAYERS = 10
MODEL_LOOP_DEPTH = 1
MODEL_FFN_RATIO = 8.0
MODEL_SPARSITY = 0.95
MODEL_DROPOUT = 0.0
MODEL_TIE_LM_HEAD = True

# In-domain val holdout: rows where ``row_idx % VAL_HOLDOUT_DENOM in
# VAL_HOLDOUT_KEEP`` are routed to val (5% by default).  Deterministic
# so different runs evaluate on the same examples.
VAL_HOLDOUT_DENOM = 20
VAL_HOLDOUT_KEEP = {0}  # 1/20 = 5% holdout


def _build_config_dict(args) -> dict:
    """Build the architecture-config dict persisted alongside every ckpt.

    Same shape as ``train_100m_kd.py`` so an SFT ckpt is a drop-in
    warmstart for the KD trainer. The loader (chat_demo.py) reads
    ``ckpt["config"]`` and rebuilds the model with the exact shapes;
    falls back to its own hardcoded defaults if the key is missing
    (legacy ckpts). See P12 in ``docs/MASTER_PLAN.md`` §6.
    """
    return {
        "vocab": int(getattr(args, "vocab", MODEL_VOCAB)),
        "d": int(getattr(args, "d", MODEL_D)),
        "n_layers": int(getattr(args, "n_layers", MODEL_N_LAYERS)),
        "loop_depth": int(getattr(args, "loop_depth", MODEL_LOOP_DEPTH)),
        "max_seq": int(getattr(args, "seq_len", SEQ_LEN)),
        "ffn_ratio": float(getattr(args, "ffn_ratio", MODEL_FFN_RATIO)),
        "sparsity": float(getattr(args, "sparsity", MODEL_SPARSITY)),
        "dropout": MODEL_DROPOUT,
        "tie_lm_head": MODEL_TIE_LM_HEAD,
        # Phase tag persists in the ckpt so the loader / resume logic
        # can branch on "this came from SFT, drop SFT-only state".
        "phase": "sft",
    }


def _log(msg: str) -> None:
    """Module-level log shim. ``main()`` rebinds a closure that ALSO
    appends to ``log_lines``, but module-level helpers may run before
    that closure exists, so we always have a safe fallback.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# Eval -- response-only ppl
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_sft(
    model,
    val_iter,
    n_batches: int = 16,
    plif_cells: Optional[list] = None,
    response_only: bool = True,
    pad_id: int = 0,
) -> float:
    """Compute response-only ppl over up to ``n_batches`` val batches.

    The val iterator must yield ``(tokens_in, tokens_out, loss_mask)``
    triples (the SFT format). Pad positions get mask=0 by construction
    so this is a leak-free in-domain holdout signal.

    Parameters
    ----------
    response_only:
        If True, ppl is computed only on positions where mask=1 (i.e.
        response tokens). If False, ppl is computed on all non-pad
        positions (the ablation comparison).
    pad_id:
        Used only when ``response_only=False``; we override the mask
        to ``(tokens_out != pad_id)`` so trailing pads never enter the
        denominator.

    Returns
    -------
    Float ppl ``= exp(mean_loss)``. Returns NaN when the iterator is
    empty (degenerate val set).
    """
    model.eval()
    losses: list[float] = []
    rate_accum: list = []
    # Device-aware: take cue from the model itself so a CPU-only test
    # harness (CI runners, integration tests) can call evaluate_sft on
    # a CUDA-built process without device mismatch on F.embedding.
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(DEVICE)
    use_cuda_path = model_device.type == "cuda"
    for _ in range(n_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        if len(batch) == 3:
            x, y, m = batch
        else:
            # Defensive: if some caller passes a pure (in, out) pair
            # (LM stream), treat it as full-CE on all non-pad positions.
            x, y = batch
            m = (y != pad_id).float()
        x = x.to(model_device)
        y = y.to(model_device)
        m = m.to(model_device)
        if not response_only:
            m = (y != pad_id).float()  # all non-pad positions
        if use_cuda_path:
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                logits = model(x).float()
                loss = response_only_ce_loss(logits, y, m)
        else:
            # CPU: torch 2.0.1 raises if autocast_cpu_dtype != bfloat16
            # even when enabled=False, so skip the context manager
            # entirely on CPU and rely on fp32 throughout.
            logits = model(x).float()
            loss = response_only_ce_loss(logits, y, m)
        losses.append(float(loss.item()))
        if plif_cells:
            rate_accum.append(
                [c.last_spike_rate().item() for c in plif_cells]
            )
    model.train()
    if plif_cells and rate_accum:
        n = len(plif_cells)
        avg = [
            sum(r[i] for r in rate_accum) / len(rate_accum) for i in range(n)
        ]
        rmin, rmax = min(avg), max(avg)
        rmean = sum(avg) / n
        n_dead = sum(1 for r in avg if r < 0.005)
        n_sat = sum(1 for r in avg if r > 0.5)
        _log(
            f"  [val] spike: mean={rmean:.3f} "
            f"range=[{rmin:.3f}, {rmax:.3f}] "
            f"dead={n_dead}/{n} sat={n_sat}/{n}"
        )
    if not losses:
        return float("nan")
    return math.exp(sum(losses) / len(losses))


# ---------------------------------------------------------------------------
# In-domain val split: row-modulo holdout
# ---------------------------------------------------------------------------


class _AlpacaHoldoutStream:
    """Wraps an :class:`InstructionParquetStream` and routes a 1/N
    fraction of rows to a separate holdout iterator.

    Row-deterministic: each row's index in the parent's underlying row
    iterator decides whether it's in train or holdout. Same recipe as
    :func:`synapforge.data.split_val_stream` but applied at the SFT
    triple level.
    """

    def __init__(
        self,
        parent: InstructionParquetStream,
        keep: set,
        denom: int,
        side: str,
    ) -> None:
        self._parent = parent
        self._keep = set(int(i) for i in keep)
        self._denom = int(denom)
        self._side = str(side)

    @property
    def batch_size(self) -> int:
        return self._parent.batch_size

    def __iter__(self):
        rows: list = []
        for row_idx, row in enumerate(self._parent._iter_rows()):
            if (row_idx % self._denom) not in self._keep:
                continue
            rows.append(row)
            if len(rows) >= self._parent.batch_size:
                yield self._parent._build_batch(rows)
                rows = []
        # Yield trailing partial batch (val is loop=False so important).
        if rows:
            yield self._parent._build_batch(rows)


def split_alpaca(
    parent: InstructionParquetStream,
    holdout_denom: int = VAL_HOLDOUT_DENOM,
    holdout_keep: set = VAL_HOLDOUT_KEEP,
):
    """Return ``(train_stream, val_holdout_stream)`` over the same parent.

    The two streams' row sets are disjoint by construction (row index
    modulo ``denom``), so the holdout ppl is a leak-free signal.
    """
    train_keep = set(range(holdout_denom)) - set(holdout_keep)
    train = _AlpacaHoldoutStream(parent, train_keep, holdout_denom, "train")
    holdout = _AlpacaHoldoutStream(
        parent, holdout_keep, holdout_denom, "holdout"
    )
    return train, holdout


# ---------------------------------------------------------------------------
# LR schedule (mirror kd trainer)
# ---------------------------------------------------------------------------


def lr_at(
    step: int, peak: float, warmup: int, total: int, kind: str = "cosine"
) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    if kind == "none":
        return peak
    progress = (step - warmup) / max(1, total - warmup)
    if kind == "cosine":
        return peak * 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == "linear":
        return peak * max(0.0, 1.0 - progress)
    return peak


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_DIR_DEFAULT,
                   help="run output dir (ckpts + metrics.json + train.log)")
    p.add_argument("--data", default=DATA_DEFAULT,
                   help="path or glob of instruction parquets (alpaca-zh "
                        "+ Qwen-tokenized layout). Each file must have "
                        "either prompt_input_ids+response_input_ids OR "
                        "input_ids+response_mask columns.")
    p.add_argument("--cross-val", default=CROSS_VAL_DEFAULT,
                   dest="cross_val",
                   help="path or glob of a SECOND parquet for cross-"
                        "domain val ppl (e.g. wikitext-100 tokenized "
                        "with the same tokenizer). Empty string = "
                        "disable; only in-domain holdout is reported.")
    p.add_argument("--warmstart", default=WARM_CKPT_DEFAULT,
                   help="path to a Phase 1 ckpt (Synap-1 Ultra "
                        "step_X.pt / best_step_X.pt) to warmstart from. "
                        "Must match the architecture defined by the "
                        "--vocab/--d/--n-layers flags. Empty string = "
                        "disabled (cold start; ill-advised for SFT).")
    p.add_argument("--no-warmstart", dest="warmstart",
                   action="store_const", const="",
                   help="force-disable warmstart (cold-start SFT; "
                        "ablation only -- production SFT must always "
                        "warmstart from the best Phase 1 ckpt).")
    p.add_argument("--steps", type=int, default=N_STEPS_DEFAULT)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--save-every", type=int, default=SAVE_EVERY)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    # Architecture overrides for tiny-model smoke / unit testing.
    p.add_argument("--vocab", type=int, default=MODEL_VOCAB)
    p.add_argument("--d", type=int, default=MODEL_D)
    p.add_argument("--n-layers", type=int, default=MODEL_N_LAYERS)
    p.add_argument("--loop-depth", type=int, default=MODEL_LOOP_DEPTH)
    p.add_argument("--ffn-ratio", type=float, default=MODEL_FFN_RATIO)
    p.add_argument("--sparsity", type=float, default=MODEL_SPARSITY)
    p.add_argument("--lr", type=float, default=LR,
                   help="peak LR. SFT default 1e-4; warmstart from a "
                        "trained Phase 1 ckpt does not need warmup-driven "
                        "LR discovery -- we already know the loss surface.")
    p.add_argument("--lr-decay", default="cosine",
                   choices=["none", "cosine", "linear"])
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--tokenizer-name",
                   default="/workspace/teachers/qwen2.5-0.5b",
                   help="tokenizer used to derive pad_id/eos_id when "
                        "the parquet has no explicit metadata. Defaults "
                        "to the rental Qwen2.5-0.5B path; 'gpt2' is a "
                        "fine fallback for CI / smoke.")
    p.add_argument("--pad-id", type=int, default=-1,
                   help="explicit pad token id; -1 = auto from tokenizer "
                        "(eos if pad missing). Only used if the parquet "
                        "has no padded rows already.")
    # Response-only loss is the SFT recipe (default ON). Off path is
    # the ablation -- documented in the docstring at the top of file.
    p.add_argument("--response-only-loss", action="store_true",
                   default=True, dest="response_only_loss",
                   help="mask out prompt tokens from CE; only response "
                        "tokens contribute to the loss. Default ON. "
                        "This is the standard SFT recipe (Stanford "
                        "Alpaca / DeepSeek V2 / Qwen-Chat).")
    p.add_argument("--no-response-only-loss", action="store_false",
                   dest="response_only_loss",
                   help="ABLATION ONLY: train on full instruction LM "
                        "(prompt + response). Used to A/B against the "
                        "default response-mask recipe so we can show "
                        "the mask is what unlocks the plateau break.")
    # Synap-1 model knobs inherited from train_100m_kd.py (default OFF).
    p.add_argument("--lm-head-spectral-norm", action="store_true",
                   default=False, dest="lm_head_spectral_norm")
    p.add_argument("--lm-head-pre-ln", action="store_true",
                   default=False, dest="lm_head_pre_ln")
    p.add_argument("--quant-cfc-weights", default="none",
                   choices=["none", "ternary"])
    p.add_argument("--latent-k", type=int, default=0, dest="latent_k")
    p.add_argument("--high-pass-residual-weight", type=float, default=0.0,
                   dest="high_pass_residual_weight")
    p.add_argument("--plif-tau-init", default="unimodal",
                   choices=["unimodal", "bimodal", "trimodal", "log_uniform"],
                   dest="plif_tau_init")
    p.add_argument("--freeze-vocab-tail", action="store_true", default=True,
                   dest="freeze_vocab_tail")
    p.add_argument("--no-freeze-vocab-tail", action="store_false",
                   dest="freeze_vocab_tail")
    p.add_argument("--grad-checkpoint", action="store_true", default=False)
    # P24 shuffle (mirror kd trainer).
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    p.add_argument("--shuffle-seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# pad/eos resolution
# ---------------------------------------------------------------------------


def _resolve_pad_eos_ids(args) -> tuple[int, int]:
    """Best-effort load of pad/eos ids from the tokenizer.

    Falls back to (0, 0) if the tokenizer can't be loaded (CI / no-net
    environment). Trainer rationality: pad=0 collides with valid token
    ids in some tokenizers, but in our SFT loop pad positions only
    occur in the loss mask which already has them as 0 -- so pad
    ids never reach the model in a position where their identity
    matters for the gradient.
    """
    pad = int(args.pad_id) if args.pad_id >= 0 else -1
    eos = -1
    try:
        from synapforge.huggingface_adapter import load_tokenizer
        tok = load_tokenizer(args.tokenizer_name)
        if pad < 0:
            pad = int(getattr(tok, "pad_token_id", None) or 0)
        eos = int(getattr(tok, "eos_token_id", None) or pad)
    except Exception as exc:  # pragma: no cover -- network/CI dep
        _log(f"[tokenizer] load failed ({exc!r}); using pad=0 eos=0")
        pad = 0 if pad < 0 else pad
        eos = 0 if eos < 0 else eos
    return pad, eos


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out
    warm_ckpt = args.warmstart
    n_steps = int(args.steps)
    _safe_mkdir(out_dir)
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    seq_len = int(args.seq_len)
    save_every = int(args.save_every)
    eval_every = int(args.eval_every)
    log_every = int(args.log_every)
    peak_lr = float(args.lr)

    print(
        f"device={DEVICE} dtype={DTYPE} out={out_dir} "
        f"steps={n_steps} bs={args.batch_size} seq={seq_len} lr={peak_lr}"
    )
    print(
        f"response_only_loss={bool(args.response_only_loss)} "
        f"warmstart={warm_ckpt!r}"
    )

    # ---------------- model ----------------
    # Mirror train_100m_kd.py's plif_tau_init translation: PLIFCell does
    # not understand "unimodal" -- that string is a CLI sentinel meaning
    # "use the legacy 2.5 uniform-warm init". Pass the float through
    # otherwise (bimodal/trimodal/log_uniform are PLIFCell-native strings).
    plif_tau = float(2.5) if args.plif_tau_init == "unimodal" else args.plif_tau_init
    model = build_synapforge_100m(
        vocab=int(args.vocab), d=int(args.d), n_layers=int(args.n_layers),
        loop_depth=int(args.loop_depth), max_seq=seq_len,
        ffn_ratio=float(args.ffn_ratio), sparsity=float(args.sparsity),
        dropout=MODEL_DROPOUT,
        use_grad_checkpoint=bool(args.grad_checkpoint),
        freeze_vocab_tail=bool(args.freeze_vocab_tail),
        lm_head_spectral_norm=bool(args.lm_head_spectral_norm),
        lm_head_pre_ln=bool(args.lm_head_pre_ln),
        weight_quant_cfc=str(args.quant_cfc_weights),
        plif_tau_init=plif_tau,
        high_pass_residual_weight=float(args.high_pass_residual_weight),
        latent_k=int(args.latent_k),
    )
    n_params = model.num_parameters()
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")
    plif_cells = [m for m in model.modules() if isinstance(m, PLIFCell)]
    print(f"plif cells found: {len(plif_cells)}")

    # ---------------- warm-start ----------------
    if warm_ckpt and os.path.exists(warm_ckpt):
        try:
            rep = adv_warmstart(
                model, warm_ckpt,
                name_map=[
                    (r"\.cfc\.", ".liquid."),
                    (r"\.embed\.text_embed\.", ".tok_embed."),
                ],
            )
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
            for line in traceback.format_exc().splitlines():
                _log(f"  {line}")
    else:
        _log(f"warmstart ckpt {warm_ckpt!r} not found; pure random init "
             "(SFT cold-start, ablation only)")

    model = model.to(DEVICE)
    model.train()

    # ---------------- optimizer ----------------
    optim = build_optimizer(model, lr=peak_lr, weight_decay=WEIGHT_DECAY)
    print(f"optimizer: {type(optim).__name__} lr={peak_lr} wd={WEIGHT_DECAY}")

    # Load optim state from warmstart ckpt if present (preserves Adam m/v
    # momentum so SFT doesn't cold-start the moment estimates -- one of
    # the proven recipes from feedback_warmstart_explicit_step_not_mtime).
    if warm_ckpt and os.path.exists(warm_ckpt):
        try:
            _ck = torch.load(warm_ckpt, map_location="cpu")
            if isinstance(_ck, dict) and "optim_state" in _ck:
                optim.load_state_dict(_ck["optim_state"])
                _log(f"warmstart: loaded optim state from {warm_ckpt}")
            else:
                _log("warmstart: no optim_state in ckpt (legacy ckpt; "
                     "momentum cold-start)")
        except Exception as exc:
            _log(f"warmstart optim load skipped: {exc}")

    # ---------------- data ----------------
    pad_id, eos_id = _resolve_pad_eos_ids(args)
    _log(f"[data] pad_id={pad_id} eos_id={eos_id}")
    parent_stream = InstructionParquetStream(
        args.data,
        seq_len=seq_len,
        batch_size=args.batch_size,
        response_only_loss=bool(args.response_only_loss),
        pad_id=pad_id,
        eos_id=eos_id,
        loop=True,
        shuffle_buffer=int(args.shuffle_buffer),
        shuffle_seed=int(args.shuffle_seed),
    )
    val_parent = InstructionParquetStream(
        args.data,
        seq_len=seq_len,
        batch_size=args.batch_size,
        response_only_loss=bool(args.response_only_loss),
        pad_id=pad_id,
        eos_id=eos_id,
        loop=False,           # val never loops
        shuffle_buffer=0,     # val is deterministic
    )
    train_stream, _val_unused = split_alpaca(parent_stream)
    val_holdout_stream = _AlpacaHoldoutStream(
        val_parent,
        keep=set(VAL_HOLDOUT_KEEP),
        denom=VAL_HOLDOUT_DENOM,
        side="holdout",
    )
    train_iter = iter(train_stream)
    print(f"train stream: {parent_stream!r}")

    # Optional cross-domain val (e.g. wikitext for OOD ppl).
    cross_val_stream = None
    if args.cross_val and os.path.exists(args.cross_val):
        try:
            cross_val_stream = InstructionParquetStream(
                args.cross_val,
                seq_len=seq_len,
                batch_size=args.batch_size,
                response_only_loss=bool(args.response_only_loss),
                pad_id=pad_id,
                eos_id=eos_id,
                loop=False,
                shuffle_buffer=0,
            )
            _log(f"[cross-val] enabled: {args.cross_val}")
        except Exception as exc:
            _log(f"[cross-val] load failed ({exc!r}); disabling")
            cross_val_stream = None
    else:
        _log("[cross-val] disabled (no path or file missing)")

    # ---------------- training loop ----------------
    metrics: dict = {
        "step": [], "loss": [], "step_ms": [], "tok_per_s": [],
        "ppl_alpaca_holdout": {}, "ppl_cross_val": {},
    }
    t0 = time.time()
    cum_tok = 0
    last_log_t = t0

    for step in range(1, n_steps + 1):
        cur_lr = lr_at(step, peak_lr, args.warmup, n_steps, args.lr_decay)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr
        t_step = time.time()

        try:
            x, y, m = next(train_iter)
        except StopIteration:
            _log(f"data exhausted at step {step}")
            break
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        if DEVICE == "cuda":
            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
                logits = model(x).float()
                loss = response_only_ce_loss(
                    logits, y, m,
                    label_smoothing=float(args.label_smoothing),
                )
        else:
            # CPU: torch 2.0.1 raises if autocast_cpu_dtype != bfloat16
            # even when enabled=False (see evaluate_sft for context).
            logits = model(x).float()
            loss = response_only_ce_loss(
                logits, y, m, label_smoothing=float(args.label_smoothing)
            )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(args.grad_clip),
            )
        optim.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.time() - t_step) * 1000.0
        # Effective tokens-per-step counts the response tokens that
        # actually carried gradient (the unmasked positions). When
        # response-only is on, that's much less than B*T -- documenting
        # the higher-signal-density of SFT vs LM.
        n_response_tokens = int(m.sum().item())
        cum_tok += n_response_tokens

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
            _log(
                f"step {step:5d} loss={loss.item():.4f} "
                f"lr={cur_lr:.5f} step_ms={step_ms:.1f} "
                f"resp_tok/s={tok_s:.0f} "
                f"resp_per_step={n_response_tokens}{mem_str}"
            )

        if step % save_every == 0:
            ckpt_path = os.path.join(out_dir, f"step_{step:06d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "step": step,
                    "loss": float(loss.item()),
                    "n_params": n_params,
                    "lr": cur_lr,
                    "config": _build_config_dict(args),
                },
                ckpt_path,
            )
            _log(f"saved ckpt {ckpt_path}")

        if step % eval_every == 0 or step == n_steps:
            ppl_h = evaluate_sft(
                model, iter(val_holdout_stream),
                n_batches=16, plif_cells=plif_cells,
                response_only=bool(args.response_only_loss),
                pad_id=pad_id,
            )
            metrics["ppl_alpaca_holdout"][step] = ppl_h
            _log(f"VAL step {step}: alpaca_holdout_ppl={ppl_h:.2f}")

            if cross_val_stream is not None:
                try:
                    ppl_x = evaluate_sft(
                        model, iter(cross_val_stream),
                        n_batches=16, plif_cells=None,
                        response_only=bool(args.response_only_loss),
                        pad_id=pad_id,
                    )
                    metrics["ppl_cross_val"][step] = ppl_x
                    _log(f"VAL step {step}: cross_val_ppl={ppl_x:.2f}")
                except Exception as exc:
                    _log(f"  [cross-val] eval failed: {exc}")

        if time.time() - last_log_t > 30:
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(out_dir, "live.log"), "w") as f:
                f.write("\n".join(log_lines[-200:]))
            last_log_t = time.time()

    # ---------------- final dump ----------------
    torch.save(
        {
            "model": model.state_dict(),
            "optim_state": optim.state_dict(),
            "step": n_steps,
            "n_params": n_params,
            "config": _build_config_dict(args),
        },
        os.path.join(out_dir, "final.pt"),
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "live.log"), "w") as f:
        f.write("\n".join(log_lines))
    elapsed = time.time() - t0
    print(
        f"done in {elapsed:.1f}s ({cum_tok} response tokens, "
        f"{cum_tok/max(elapsed,1e-6):.0f} resp_tok/s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
