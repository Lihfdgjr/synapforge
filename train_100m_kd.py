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


def _format_loss_pct(
    *,
    step_loss: float,
    step_ce: float,
    step_kd: float,
    step_z: float,
    z_loss_weight: float,
    kd_weight: float = 0.0,
    step_modal_aux: float = 0.0,
    has_modal: bool = False,
    step_cur_aux: float = 0.0,
    cur_weight: float = 0.0,
    has_curiosity: bool = False,
) -> str:
    """T5.1 — compute the ``pct_*`` log columns for the per-step log line.

    Returns a string starting with a leading space (or empty if there is
    nothing to add). The math:

        denom    = max(step_loss, 1e-9)
        pct_ce    = step_ce          / denom * 100
        pct_kd    = step_kd          / denom * 100
        pct_z     = step_z * z_w     / denom * 100
        pct_modal = step_modal_aux   / denom * 100   (modal added unweighted)
        pct_cur   = step_cur_aux*c_w / denom * 100

    ``z`` and ``cur`` accumulators in the trainer hold RAW (un-weighted)
    means so we re-weight here to reflect the actual fraction of
    ``loss``. ``modal_aux`` is added directly to ``loss`` un-weighted.

    KD-OFF steps (``step_kd == 0``) print ``pct_kd=0.0`` rather than
    omitting the column so downstream parsers see a stable schema.

    Floors by 1e-9 so the pre-backward step (loss == 0.0 fallback) does
    not divide by zero.

    Pure function -- no torch, no IO -- so it is unit-testable on CPU.
    """
    denom = max(float(step_loss), 1e-9)
    # Trainer applies (1-α)·base + α·kd on KD-on steps where base = ce + z·zw
    # and α = ``kd_weight``. On KD-off steps loss = ce + z·zw (step_kd is 0).
    #
    # Reweighting only kicks in when ``kd_weight > 0`` (the production path
    # where the trainer actually mixes a KD loss in). When ``kd_weight == 0``
    # — the unit-test default, where each ``step_*`` already represents its
    # raw fraction of ``step_loss`` — we treat all components as raw
    # contributions and avoid scaling them, so the printed pct columns sum
    # back to ~100 % of the loss as the docstring promises.
    kd_w = float(kd_weight)
    if kd_w > 0.0 and float(step_kd) > 0.0:
        base_w = float(1.0 - kd_w)
        kd_factor = kd_w
    else:
        base_w = 1.0
        kd_factor = 1.0
    pct_ce = base_w * float(step_ce) / denom * 100.0
    pct_kd = kd_factor * float(step_kd) / denom * 100.0
    pct_z = base_w * float(step_z) * float(z_loss_weight) / denom * 100.0
    out = f" pct_ce={pct_ce:.1f} pct_kd={pct_kd:.1f} pct_z={pct_z:.1f}"
    if has_modal:
        pct_modal = float(step_modal_aux) / denom * 100.0
        out += f" pct_modal={pct_modal:.1f}"
    if has_curiosity:
        pct_cur = float(step_cur_aux) * float(cur_weight) / denom * 100.0
        out += f" pct_cur={pct_cur:.1f}"
    return out


def _format_spike_rates_per_layer(rates: list[float]) -> str:
    """T5.2 — render the per-layer spike-rate log line.

    Returns a string of the form

        spike_rates_per_layer: l0=0.000 l1=0.000 ... l<N-1>=0.000

    where ``rates`` is the list of per-PLIF-cell spike rates (already
    pulled via ``last_spike_rate().item()`` by the caller). The label
    width tracks the layer count so the helper handles any ``N`` without
    truncation -- a 4-layer model emits ``l0=...l3=...``, a 12-layer
    model emits ``l0=...l11=...``. Each value is rendered to 3-decimal
    precision to match the existing ``spike: mean=...`` line.

    Pure function -- no torch, no IO -- so it is unit-testable on CPU.

    Args:
        rates: per-layer spike rates as Python floats. May be empty, in
               which case the helper returns the prefix label only with
               no entries (``"spike_rates_per_layer:"``).

    Returns:
        Single-line log string (no trailing newline).
    """
    parts = [f"l{i}={float(r):.3f}" for i, r in enumerate(rates)]
    if not parts:
        return "spike_rates_per_layer:"
    return "spike_rates_per_layer: " + " ".join(parts)


def _compute_grad_norm_per_named_module(model) -> "list[tuple[str, float]]":
    """T5.3 — compute total gradient L2 norm per top-level named child of ``model``.

    Returns a list of ``(name, total_norm)`` pairs preserving the order in
    which the children were registered. The norm for one module is

        sqrt( sum_{p in module.parameters() if p.grad is not None} ||p.grad||^2 )

    which is the same Frobenius-style aggregation that
    ``torch.nn.utils.clip_grad_norm_`` uses internally. Modules whose
    every parameter has ``p.grad is None`` (e.g. frozen submodules, the
    untouched LM head when tied, any layer that hasn't seen a backward
    yet) are SKIPPED entirely so we don't emit a meaningless ``0.0``
    that might be confused with a dead-grad layer.

    The "top-level children" rule is what makes this safe to call every
    100 steps without flooding the log: we explicitly do NOT recurse via
    ``named_modules()`` -- a 100M model has thousands of submodules, and
    that volume of grad reduction every 100 steps would itself slow
    training noticeably. ``named_children()`` returns only one level
    (e.g. ``tok_embed``, ``ln_f``, ``lm_head``) PLUS one entry per
    ``ModuleList``: SynapForge100M's ``blocks`` is a ``nn.ModuleList``
    so its direct child appears as the single name ``blocks``. To match
    the documented format ``... block_0=Y.YY ... block_9=Z.ZZ ...`` we
    expand any ``ModuleList`` / ``ModuleDict`` child one level: each
    entry becomes ``<list-name>_<idx>`` (or ``<dict-name>_<key>``). This
    is the EXACT recursion stop -- no further drill-in.

    Pure-Python aside from the ``p.grad.norm()`` call which is a single
    fused CUDA reduction per parameter; total wall is sub-millisecond at
    the 100M scale. CPU-fallback path (no CUDA) is the same code.

    Args:
        model: the model whose named_children we walk.

    Returns:
        Ordered list of ``(name, total_norm)`` pairs (one per surviving
        top-level / one-level-expanded module). Empty list when the
        model has no parameters with grads (e.g. before the first
        backward, or all-frozen models).
    """
    import torch.nn as nn

    pairs: list[tuple[str, float]] = []

    def _module_total_norm(module) -> "float | None":
        """Return sqrt(sum_p ||p.grad||^2) over a single module's params, or
        None when *every* parameter's grad is None (skip-clean signal)."""
        sq_sum = 0.0
        any_grad = False
        for p in module.parameters():
            g = p.grad
            if g is None:
                continue
            any_grad = True
            # ``.norm()`` returns a 0-d tensor; ``.item()`` is the safe
            # cross-device pull. ``** 2`` keeps it Python-side cheap; we
            # accumulate doubles to avoid bf16-overflow drift on big
            # FFN params.
            n = float(g.norm().item())
            sq_sum += n * n
        if not any_grad:
            return None
        return sq_sum ** 0.5

    for name, child in model.named_children():
        # Expand ModuleList / ModuleDict by exactly one level so the
        # documented ``block_0 ... block_9`` format works without
        # uncontrolled recursion. Any deeper nesting (e.g. a
        # ModuleList-of-ModuleLists) stops here -- the inner list's
        # total norm is rolled into ``<outer>_<idx>``.
        if isinstance(child, nn.ModuleList):
            for idx, sub in enumerate(child):
                norm = _module_total_norm(sub)
                if norm is None:
                    continue
                pairs.append((f"{name}_{idx}", norm))
        elif isinstance(child, nn.ModuleDict):
            for key, sub in child.items():
                norm = _module_total_norm(sub)
                if norm is None:
                    continue
                pairs.append((f"{name}_{key}", norm))
        else:
            norm = _module_total_norm(child)
            if norm is None:
                continue
            pairs.append((name, norm))
    return pairs


def _format_grad_norm_per_module(pairs: "list[tuple[str, float]]") -> str:
    """T5.3 — render the per-named-module grad-norm log line.

    Returns a string of the form

        grad_norm: tok_embed=X.XXXe+YY blocks_0=... ... ln_f=W.WWWe+ZZ

    where ``pairs`` is the list of ``(name, total_norm)`` produced by
    :func:`_compute_grad_norm_per_named_module`. Each value is rendered
    in scientific notation (``.3e``) because gradient norms span orders
    of magnitude across modules and across training (early ``lm_head``
    grads dwarf early ``tok_embed`` grads by 10x+; later in training
    they cross). Linear ``.3f`` would lose precision on the small end.

    Pure function -- no torch, no IO -- so it is unit-testable on CPU.

    Args:
        pairs: ordered list of ``(name, total_norm)`` pairs. May be
               empty, in which case the helper returns the prefix label
               only with no entries (``"grad_norm:"``). Names with NaN
               or inf norms are still rendered (``.3e`` formats them
               as ``"nan"`` / ``"inf"``) -- the caller should NOT
               pre-filter these so divergence shows up in the log.

    Returns:
        Single-line log string (no trailing newline).
    """
    if not pairs:
        return "grad_norm:"
    parts = [f"{name}={float(norm):.3e}" for name, norm in pairs]
    return "grad_norm: " + " ".join(parts)


def _update_best_ckpt(
    *,
    out_dir: str,
    step: int,
    val_ppl: float,
    best_val_ppl: float,
    enabled: bool = True,
    log_fn=None,
    os_name: Optional[str] = None,
) -> tuple[float, Optional[str]]:
    """T5.4 — track best val_ppl_holdout and maintain a ``best_step_*.pt`` link.

    On Linux/Mac creates a *relative* symlink ``best_step_<N>.pt -> step_<N>.pt``;
    on Windows (where symlinks need admin) falls back to a file copy.  Idempotent:
    if the new ``val_ppl`` does NOT improve on ``best_val_ppl``, returns the old
    value unchanged and the link is left alone (so re-running with the same val
    multiple times is a no-op).

    Args:
        out_dir:        directory containing ``step_<N>.pt`` ckpts.
        step:           current step (used to format ``best_step_<N>.pt``).
        val_ppl:        the just-computed ``val_ppl_holdout`` for this step.
        best_val_ppl:   running min so far (use ``float('inf')`` on the first eval).
        enabled:        if False this is a no-op and ``best_val_ppl`` is returned as is.
        log_fn:         optional logger; receives one human-readable string when an
                        improvement is recorded. Defaults to ``_log``.
        os_name:        override for ``os.name`` (``'nt'`` forces the Windows copy
                        fallback). Used by tests; production callers leave None.

    Returns:
        ``(new_best_val_ppl, link_path_or_None)``. ``link_path`` is the path of
        the freshly created ``best_step_<N>.pt`` if the val improved, else None.

    Notes:
        * The source ckpt ``step_<N>.pt`` MUST already exist on disk -- this
          helper does NOT save the model, it only links/copies an already-saved
          ckpt. Caller is responsible for ordering: save first, then call us.
        * Stale ``best_step_*.pt`` files in ``out_dir`` (from previous runs or
          prior bests) are removed BEFORE the new one is created so there is
          always exactly one ``best_step_*.pt`` after a successful improvement.
    """
    if not enabled:
        return best_val_ppl, None
    if log_fn is None:
        log_fn = _log
    val_f = float(val_ppl)
    if not (val_f < float(best_val_ppl)):  # also handles NaN -> no update
        return best_val_ppl, None

    src_name = f"step_{step:06d}.pt"
    src_path = os.path.join(out_dir, src_name)
    if not os.path.exists(src_path):
        # ckpt wasn't saved at this step (eval_every doesn't divide save_every);
        # nothing to link to, so we can't actually mark this as best yet.
        log_fn(
            f"[best-ckpt] val={val_f:.2f} would improve from "
            f"{float(best_val_ppl):.2f} but {src_name!r} not on disk; "
            f"skipped"
        )
        return best_val_ppl, None

    dst_name = f"best_step_{step:06d}.pt"
    dst_path = os.path.join(out_dir, dst_name)

    # Remove any stale best_step_*.pt files (and the new dst if a partial run
    # left a turd) so there is exactly one after we're done.
    try:
        for fname in os.listdir(out_dir):
            if fname.startswith("best_step_") and fname.endswith(".pt"):
                stale = os.path.join(out_dir, fname)
                try:
                    os.remove(stale)
                except OSError:
                    pass
    except OSError:
        # out_dir vanished -- nothing we can do.
        return best_val_ppl, None

    is_windows = (os_name if os_name is not None else os.name) == "nt"
    used_copy = False
    if is_windows:
        # Windows symlinks need admin; copy is the portable fallback.
        import shutil
        shutil.copyfile(src_path, dst_path)
        used_copy = True
    else:
        try:
            # Relative target so the link survives moving the run dir.
            os.symlink(src_name, dst_path)
        except (OSError, NotImplementedError):
            # POSIX symlink failed (no perms / weird FS) -> fall back to copy.
            import shutil
            shutil.copyfile(src_path, dst_path)
            used_copy = True

    prev_str = (
        f"{float(best_val_ppl):.2f}"
        if not math.isinf(float(best_val_ppl))
        else "inf"
    )
    method = "copy" if used_copy else "symlink"
    log_fn(
        f"[best-ckpt] val={val_f:.2f} improved from {prev_str} at step "
        f"{step} -> {dst_name} ({method})"
    )
    return val_f, dst_path


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
    # T2.7 — gradient accumulation. ``--grad-accum-steps N`` runs N
    # micro-batches per optimizer step, each scaled by ``1/N`` so the
    # accumulated gradient equals what a single ``batch_size * N`` batch
    # would produce — without the VRAM cost. ``N=1`` (default) preserves
    # the legacy code path verbatim. The trainer reads the value via
    # ``getattr(args, "grad_accum_steps", 1)`` so older launch scripts
    # that don't pass the flag still work.
    p.add_argument("--grad-accum-steps", type=int, default=1, dest="grad_accum_steps",
                   help="micro-batches per optimizer step (T2.7); "
                        "effective bs = --batch-size * --grad-accum-steps. "
                        "Default 1 (no accumulation).")
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
    # T5.1 (DEEP_MAINT_QUEUE.md) — append loss-component % columns to the
    # per-step log line so root-cause work can see at a glance whether ce/kd/
    # z/modal/cur dominates total. Default ON; toggle off with
    # ``--no-log-loss-pct`` if downstream tooling parses a fixed prefix.
    p.add_argument("--log-loss-pct", action="store_true", default=True,
                   dest="log_loss_pct",
                   help="append pct_ce/pct_kd/pct_z (and pct_modal/pct_cur "
                        "when those mixins are active) columns to the "
                        "per-step log line. Default ON.")
    p.add_argument("--no-log-loss-pct", action="store_false",
                   dest="log_loss_pct",
                   help="disable the T5.1 pct_ce/pct_kd/... columns; the "
                        "log line reverts to the legacy "
                        "loss=/ce=/kd=/z=/lr=/step_ms=/tok/s= format.")
    # T5.2 (DEEP_MAINT_QUEUE.md) — when debugging dead PLIF layers the
    # aggregated ``spike: mean=... range=[a,b] dead=D/T`` line hides which
    # individual layer is dead. Behind this opt-in flag, immediately
    # below the existing ``spike:`` line we also emit
    #     spike_rates_per_layer: l0=0.000 l1=0.000 ... l<N-1>=0.000
    # at the same cadence (every 50 steps). Default OFF so existing
    # downstream tooling that parses a fixed log schema is undisturbed;
    # enable for dead-layer root-cause work.
    p.add_argument("--log-spike-per-layer", action="store_true", default=False,
                   dest="log_spike_per_layer",
                   help="emit a per-layer spike-rate log line "
                        "``spike_rates_per_layer: l0=... l<N-1>=...`` "
                        "immediately after the aggregated ``spike:`` line, "
                        "at the same cadence. Useful for detecting which "
                        "individual PLIF layer is dead. Default OFF.")
    # T5.3 (DEEP_MAINT_QUEUE.md) — Run 3l/3m diverged at step 14000+ but
    # the existing aggregated ``loss=... ce=... kd=...`` line couldn't
    # show WHICH submodule's grad blew up first. Behind this opt-in
    # flag, every 100 steps -- AFTER backward / clip_grad_norm_ but
    # BEFORE optim.step() so we capture the gradients about to be
    # applied -- emit one extra line:
    #     grad_norm: tok_embed=X.XXXe+YY blocks_0=... ... ln_f=W.WWWe+ZZ
    # Top-level named_children only (PLUS a one-level ModuleList/Dict
    # expansion for ``blocks``) so the line is bounded; values in
    # scientific notation since grad norms span orders of magnitude.
    # Default OFF so the cadence-100 cost (one ``norm().item()`` per
    # parameter) only fires when the user is actually root-causing a
    # diverge.
    p.add_argument("--log-grad-norm", action="store_true", default=False,
                   dest="log_grad_norm",
                   help="emit a ``grad_norm: <module>=<.3e> ...`` log line "
                        "every 100 steps after backward + clip but before "
                        "optim.step() to detect per-module gradient "
                        "imbalance (T5.3, DEEP_MAINT_QUEUE.md). Top-level "
                        "named_children of the model are reported, with a "
                        "one-level expansion of any ``nn.ModuleList`` (so "
                        "SynapForge100M's ``blocks`` produces ``blocks_0`` "
                        "... ``blocks_9`` rather than a single rolled-up "
                        "``blocks=``). Modules whose every param has "
                        "``p.grad is None`` are skipped cleanly. Default "
                        "OFF; opt-in for divergence root-cause work.")
    # T5.4 (DEEP_MAINT_QUEUE.md) — track best val_ppl_holdout across the
    # run and maintain a ``best_step_*.pt`` symlink (Linux) or copy
    # (Windows) so warmstart resilience does not depend on remembering
    # which step had the lowest val. Default ON; toggle off with
    # ``--no-best-ckpt-track`` for ablations or for runs where the
    # extra disk write is unwanted.
    p.add_argument("--best-ckpt-track", action="store_true", default=True,
                   dest="best_ckpt_track",
                   help="after each val eval, if val_ppl_holdout improved, "
                        "create/update a ``best_step_<N>.pt`` link pointing "
                        "at the matching step ckpt. Default ON.")
    p.add_argument("--no-best-ckpt-track", action="store_false",
                   dest="best_ckpt_track",
                   help="disable T5.4 best ckpt tracking. No best_step_*.pt "
                        "link is created; warmstart must explicitly pick a "
                        "step_*.pt to resume from.")
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
    # T2.4 — freeze the vocab tail so unused-by-Qwen-tokenizer rows
    # [151643, 151936) of tok_embed.weight (and lm_head.weight if untied)
    # don't drift under Adam noise + weight decay. Default ON.
    p.add_argument("--freeze-vocab-tail", action="store_true", default=True,
                   dest="freeze_vocab_tail",
                   help="zero gradient on vocab rows >= live_vocab (151643) "
                        "to keep ~75M unused params at their init values. "
                        "Qwen2.5 tokenizer never emits ids >= 151643, so "
                        "those rows only see noise; freezing them removes "
                        "that noise gradient. Default ON.")
    p.add_argument("--no-freeze-vocab-tail", action="store_false",
                   dest="freeze_vocab_tail",
                   help="disable T2.4 vocab tail freeze (unused rows drift "
                        "under Adam noise). Use only for ablations.")
    # T2.6 — bound LM head Lipschitz to mitigate P28 z-loss linear drift.
    # When tied (default), this wraps tok_embed via spectral_norm so the
    # shared weight is reparameterised as W / sigma each forward; when
    # untied, it wraps lm_head directly. Opt-in because spectral_norm has
    # known bf16 quirks (power-iter buffers stay fp32).
    # T2.8 / MATMUL_FREE.md M1 — BitNet b1.58 ternary QAT on CfC input
    # projections only (delta_proj + b_proj inside every LiquidCell).
    # Default "none" preserves the historical fp baseline. "ternary"
    # turns on AbsMean ternary QAT with STE backward; gamma stays in
    # fp32 so bf16 autocast is safe.
    p.add_argument("--quant-cfc-weights", default="none",
                   choices=["none", "ternary"],
                   help="quantization scheme for CfC input projections "
                        "(LiquidCell.delta_proj + LiquidCell.b_proj). "
                        "'ternary' = BitNet b1.58 AbsMean QAT (arXiv: "
                        "2402.17764). Untouched: A_log (recurrent decay), "
                        "PLIF tau/threshold, SparseSynapse, FFN, LM head.")
    p.add_argument("--lm-head-spectral-norm", action="store_true",
                   default=False, dest="lm_head_spectral_norm",
                   help="apply torch.nn.utils.spectral_norm to the LM head "
                        "(tied: tok_embed; untied: lm_head). Bounds the "
                        "Lipschitz constant so logsumexp(logits) -- and "
                        "therefore the z-loss term -- cannot grow "
                        "unboundedly during long training. Default OFF "
                        "(opt-in; see docs/PERF_KNOBS.md and T2.6 / P28).")
    p.add_argument("--lm-head-pre-ln", action="store_true",
                   default=False, dest="lm_head_pre_ln",
                   help="insert nn.LayerNorm(d, elementwise_affine=False) "
                        "immediately before the final lm_head projection. "
                        "Bounds the INPUT to lm_head: every row is "
                        "re-projected to a fixed sqrt(d)-norm sphere "
                        "regardless of how the affine RMSNorm scale in "
                        "ln_f drifted under Adam. Stops z-loss from "
                        "trending linearly upward across long training. "
                        "Orthogonal to --lm-head-spectral-norm (which "
                        "bounds the OPERATOR norm); both can be on. "
                        "Default OFF (opt-in; see T7.3 / P28 primary "
                        "plan in docs/TRAINING_ISSUES_RETROSPECTIVE.md "
                        "section 2.d).")
    # T2.9 / arxiv:2412.06769 — Coconut latent thinking budget.
    p.add_argument("--latent-k", type=int, default=0,
                   dest="latent_k",
                   help="Coconut latent thinking steps (k). When > 0, the "
                        "model runs k extra forward passes on the last-token "
                        "hidden as a continuous-thought vector (no token "
                        "sampling) before ln_f. Default 0 disables (zero "
                        "overhead, identity behaviour vs pre-T2.9). Recipe "
                        "from arxiv:2412.06769 / DEEP_MAINT_QUEUE.md T2.9.")
    # ---- NeurIPS 2025 (Fang et al., arXiv:2505.18608) frequency fixes ----
    # See docs/SNN_FREQUENCY_LIMIT_NEURIPS25.md.  All three flags default
    # to OFF / legacy behaviour so existing launch scripts are unaffected.
    p.add_argument("--byte-patch-pool", default="avg",
                   choices=["avg", "max", "max+avg"],
                   dest="byte_patch_pool",
                   help="A1 (Fang et al. 2505.18608): pool branch in the "
                        "BytePatch primitive (synapforge.modal.byte_patch). "
                        "'avg' (default) preserves the legacy low-pass "
                        "behaviour.  'max' adds a high-pass / edge-detector "
                        "branch.  'max+avg' concatenates both and lets a "
                        "1x1 conv learn the mix.  Wired into multimodal "
                        "callers; for text-only LM training this currently "
                        "has no effect (recorded in the trainer config and "
                        "honoured by callers that build UnifiedEmbed via "
                        "BytePatch).")
    p.add_argument("--high-pass-residual-weight", type=float, default=0.0,
                   dest="high_pass_residual_weight",
                   help="A2 (Fang et al. 2505.18608): per-block high-pass "
                        "residual lambda init.  When > 0 the HybridBlock "
                        "wraps its forward as out = block(x) + lambda * "
                        "(x - LowPass(x)) where LowPass is a depth-wise "
                        "causal Conv1d (kernel=3 default) and lambda is "
                        "per-channel learnable initialised to this scalar. "
                        "Default 0.0 = OFF (zero new params, zero new "
                        "modules; bit-identical to the legacy code path).")
    p.add_argument("--plif-tau-init", default="unimodal",
                   choices=["unimodal", "bimodal", "trimodal", "log_uniform"],
                   dest="plif_tau_init",
                   help="A3 (Fang et al. 2505.18608): PLIF tau init mode. "
                        "'unimodal' (default) keeps the legacy 2.5-uniform "
                        "init bit-for-bit.  'bimodal' = DA-LIF fast/slow "
                        "split.  'trimodal' = short(0.5)/mid(2.0)/long(8.0) "
                        "30/40/30 split spanning the full frequency band "
                        "from init.  'log_uniform' = uniform in log-space "
                        "over [2, 50].")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--skip-spike", action="store_true", default=True)
    p.add_argument("--z-loss-weight", type=float, default=1e-4,
                   help="weight for log-Z**2 stabilizer (PaLM/Gemma style)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="cross-entropy label smoothing alpha")
    p.add_argument("--spike-target", type=float, default=0.1,
                   help="target mean spike rate for PLIF cells; warn if drift > 0.05")
    # T2.5: spike-rate-target auxiliary loss term (addresses P25 PLIF dead).
    # When the PLIF spike rate falls outside [low, high], add a quadratic
    # penalty to the total loss that backprops through the surrogate
    # gradient to push ``threshold`` and ``log_tau`` back into the band.
    # Independent of the (no-grad) ``--no-plif-homeostasis`` threshold
    # control: this term participates in the autograd graph of the same
    # backward pass as ce/kd/z, so weight = 0.001 default keeps it
    # subordinate to the LM signal but still nudges live training.
    p.add_argument("--spike-target-loss-weight", type=float, default=0.001,
                   help="weight of spike-rate-target auxiliary loss (T2.5). "
                        "0 disables. Penalty = sum_layers ((rate - high).clamp(min=0)**2 "
                        "+ (low - rate).clamp(min=0)**2). Backprops through "
                        "the surrogate gradient to ``threshold`` / ``log_tau``.")
    p.add_argument("--spike-target-loss-low", type=float, default=0.05,
                   help="lower bound of dead-zone for spike-target loss (T2.5). "
                        "rate < low -> linear penalty proportional to (low - rate)**2.")
    p.add_argument("--spike-target-loss-high", type=float, default=0.20,
                   help="upper bound of dead-zone for spike-target loss (T2.5). "
                        "rate > high -> linear penalty proportional to (rate - high)**2.")
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
    # ---- NeuroMCP wire-in (T9.2 in DEEP_MAINT_QUEUE.md) -------------------
    # NeuroMCPHead = SparseSynapticLayer + DynamicActionCodebook. Replaces
    # MCP / function-calling tool tokens with a neuroplastic head that
    # grows synapses (co-activation EMA) and prototypes (cosine-novelty).
    # PoC at synapforge.demo.four_button hits density 4.5->39.9% / K=9->12 /
    # 100% hit-rate on the 4-button env. Default OFF here because the LM
    # trainer has no real action labels yet -- placeholder targets are used
    # (next-token first byte mod K) which is enough to drive the
    # plasticity rules and prove the head learns. See
    # synapforge.training.neuromcp_mixin.NeuroMCPMixin docstring.
    p.add_argument("--neuromcp-weight", type=float, default=0.0,
                   help="weight of NeuroMCP action-codebook CE loss; "
                        "0 = disabled (default). Recommended ~0.05 for "
                        "first wire-in runs. The mixin runs the LM hidden "
                        "state through SparseSynapticLayer + "
                        "DynamicActionCodebook, computes CE against a "
                        "placeholder action target (next-token first byte "
                        "mod K), and adds neuromcp_weight * action_loss "
                        "to the total. Density / K grow via the same PoC "
                        "rules validated at synapforge.demo.four_button.")
    p.add_argument("--neuromcp-codebook-size", type=int, default=16,
                   dest="neuromcp_codebook_size",
                   help="initial alive prototype count for the dynamic "
                        "action codebook. ``max_size = 4 * codebook_size`` "
                        "so growth has headroom. Default 16 (max 64).")
    p.add_argument("--neuromcp-action-dim", type=int, default=64,
                   dest="neuromcp_action_dim",
                   help="placeholder for a future fixed-action-dim head "
                        "(e.g. when real OS-actuator labels are wired in); "
                        "currently unused -- the head's effective action "
                        "space equals the dynamic codebook size. Default 64.")
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
    # ---- Run 5 PLIF-dead fixes (default OFF; see docs/PLIF_DEAD_DIAGNOSIS.md) ----
    p.add_argument("--plif-dense-bypass-steps", type=int, default=0,
                   help="Run 5 PLIF-dead fix #1: emit continuous "
                        "tanh(v_t - thr) instead of binary spikes for the "
                        "first N training steps. After step N the cell "
                        "switches back to binary spike + ATan surrogate. "
                        "0 disables (Run 5 baseline). Recommended: 5000 "
                        "for a 60k-step run; lifts the LM gradient through "
                        "synapse(s) so liquid weights don't decay to 0 "
                        "before spikes wake up.")
    p.add_argument("--sew-shortcut", action="store_true", default=False,
                   help="Run 5 PLIF-dead fix #3 (arxiv:2102.04159 SEW): "
                        "spike branch becomes synapse(s + h) * sigmoid(...), "
                        "providing a non-zero LM-gradient path through "
                        "the LiquidCell output even when s=0. Default OFF "
                        "keeps Run 5 bit-identical. Strongly recommended "
                        "with --plif-dense-bypass-steps 0 (gradient-only "
                        "fix without forward-time mode change).")
    p.add_argument("--sparse-spike-synapse", action="store_true", default=False,
                   dest="sparse_spike_synapse",
                   help="Sparse-spike synapse path (2026-05-02): exploit "
                        "the SNN-architecture-unique fact that the PLIF "
                        "spike train ``s`` is binary AND sparse. When "
                        "post-revival spike density is 5-15%%, the "
                        "synapse contribution costs O(K * out_dim) via "
                        "embedding-bag row-gather instead of O(d^2) dense "
                        "GEMM (~2-3x synapse speedup at 10%% density on "
                        "A800). Auto-falls-back to dense when measured "
                        "spike density >= --sparse-spike-threshold "
                        "(default 0.30), so this flag is safe under any "
                        "PLIF state (dead, healthy, saturated). Default "
                        "OFF since current Run 6 PLIF is dead (density~0); "
                        "enable AFTER PLIF revives. See "
                        "synapforge.kernels.sparse_spike_matmul.")
    p.add_argument("--sparse-spike-threshold", type=float, default=0.30,
                   dest="sparse_spike_threshold",
                   help="Density threshold for the sparse-spike-synapse "
                        "auto-fallback. Below this density the sparse "
                        "row-gather path is used; above it cuBLAS dense "
                        "GEMM wins so we fall back. Default 0.30 (the "
                        "empirical crossover at d=1280 on A800 80GB). "
                        "Set to 1.0 to force-always-sparse, 0.0 to "
                        "force-always-dense (debug only).")
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
    # ---- Perf audit 2026-05-02 knobs (see docs/PERF_AUDIT_2026-05-02.md) ----
    # Both default OFF (back-compat: same step-by-step sync semantics as
    # the live Run 5 launch). Opt-in for the next-restart perf combo.
    p.add_argument("--cuda-sync-every", type=int, default=1,
                   help="run torch.cuda.synchronize() once every N global "
                        "steps instead of every step. Default 1 = current "
                        "behaviour (sync every step for accurate per-step "
                        "timing). N>1 lets the GPU pipeline several steps "
                        "ahead of the host clock - big win when paired with "
                        "--kd-async-teacher because the side-stream teacher "
                        "fwd can overlap with multiple student bwds. "
                        "step_ms reported per logged step is then the "
                        "average over the last sync window. NaN guards "
                        "intact (loss.item() at log boundary still "
                        "host-syncs). Recommended: 10 on bs<=32, 1 on "
                        "bs>=80.")
    p.add_argument("--clip-grad-cache", action="store_true", default=False,
                   help="cache the requires_grad=True parameter list once "
                        "at trainer init and reuse for clip_grad_norm_ "
                        "instead of rebuilding [p for p in "
                        "model.parameters() if p.requires_grad] every "
                        "step. Saves ~50us/step CPU-side dispatch on Ultra "
                        "(~340 params at d=1280/n=16). Default OFF; "
                        "opt-in once your run isn't unfreezing params "
                        "mid-training (NeuroMCP plasticity, phase-aware "
                        "exit-101 reload). Re-cache fires on phase change.")
    # ---- Speedup audit 2026-05-02 knobs (3 ships, see
    # docs/SPEEDUP_AUDIT_2026-05-02.md) ----
    # Each default OFF (back-compat: identical Run 5 step semantics).
    p.add_argument("--lazy-host-sync-accum", action="store_true",
                   default=False, dest="lazy_host_sync_accum",
                   help="defer per-microbatch loss-component .item() calls "
                        "to log boundary. The 6 .item() calls in the inner "
                        "accum loop (loss/ce/kd/z/modal/cur) each implicit-"
                        "syncs stream 0; at accum=2 that's 12 host stalls/"
                        "step just for logging. When ON, replace "
                        "``accum_X += float(t.item())`` with GPU tensor "
                        "accumulators and only materialize Python floats at "
                        "log boundaries. Default OFF; opt-in for ~2-4 % "
                        "step-time win on Ultra. NaN/inf surfaces unchanged "
                        "at log boundary. The kd-every-adaptive ce window "
                        "still calls .item() directly (intentional, runs "
                        "every step).")
    p.add_argument("--fused-adamw", action="store_true", default=False,
                   dest="fused_adamw",
                   help="when ON and no plasticity sources are wired into "
                        "any model parameter (i.e. every requires_grad "
                        "param has only _sf_grad_source=['bp'] or no tag), "
                        "build torch.optim.AdamW(fused=True) instead of "
                        "PlasticityAwareAdamW. Numerically equivalent to "
                        "vanilla AdamW (Adam moment update is "
                        "kernel-implementation-independent) but fused "
                        "kernel is ~3x faster than the per-param Python "
                        "loop in PlasticityAwareAdamW.step. Default OFF "
                        "(preserves PlasticityAwareAdamW path). Opt-in "
                        "buys ~2-3 % step-time on Ultra (~535M, ~340 "
                        "params). Auto-falls back to PlasticityAwareAdamW "
                        "if any param is plasticity-tagged. Warmstart "
                        "compat: ckpts saved by PlasticityAwareAdamW have "
                        "different state_dict keys (m/v vs exp_avg/"
                        "exp_avg_sq); fused-adamw cold-starts moments on "
                        "v1 ckpt warmstart (one-step penalty, then "
                        "bit-exact).")
    p.add_argument("--synapforge-adamw", action="store_true", default=False,
                   dest="synapforge_adamw",
                   help="Phase 1 of the torch-replacement roadmap (see "
                        "docs/TORCH_REPLACEMENT_PLAN.md). When ON and no "
                        "plasticity sources are wired, build "
                        "synapforge.optim.AdamW (pure-python AdamW, NO "
                        "torch.optim inheritance, NO fused kernel) "
                        "instead of torch.optim.AdamW or "
                        "PlasticityAwareAdamW. Numerics match "
                        "torch.optim.AdamW(fused=True) within 1e-5 "
                        "(see tests/optim/test_synapforge_adamw.py). "
                        "Step time is ~2-3 % slower than fused-adamw "
                        "(bandwidth-bound, no fused kernel) -- "
                        "intentional Phase 1 trade-off; Phase 4 of the "
                        "roadmap will dispatch this same loop to a "
                        "Triton kernel. Mutually exclusive with "
                        "--fused-adamw; if both are passed, "
                        "--synapforge-adamw wins (we are migrating off "
                        "torch.optim, not towards it). Auto-falls back "
                        "to PlasticityAwareAdamW if any param is "
                        "plasticity-tagged (correctness > Phase 1 "
                        "milestone). Default OFF (current default = "
                        "PlasticityAwareAdamW unless --fused-adamw is "
                        "set).")
    p.add_argument("--skip-warmstart-eval-N", type=int, default=0,
                   dest="skip_warmstart_eval_n",
                   help="when warmstarting from a known-good ckpt, skip "
                        "the first N val evaluations. Each val pass on "
                        "Ultra costs ~30s (2 evals: ttt + holdout); on a "
                        "phase-aware exit-101 relaunch chain this runs "
                        "many times for no quality benefit (we already "
                        "know the baseline ppl). Default 0 = current "
                        "behaviour (every eval-every step does both ttt "
                        "and holdout val). N=1 typical for a single "
                        "relaunch; N=2 if you also want to skip the "
                        "second eval boundary. Step counter, save_every, "
                        "chat samples are unaffected.")
    # ---- Remote data warehouse (mohuanfang) -----------------------------
    # Tiered storage: rental SSD = compute-only with bounded LRU cache,
    # mohuanfang holds the canonical corpus. See
    # docs/REMOTE_DATA_WAREHOUSE.md.
    p.add_argument("--remote-warehouse-host", default="",
                   help="ssh host spec for the mohuanfang warehouse "
                        "(e.g. 'liu@mohuanfang.com'). Empty (default) = "
                        "off; trainer reads parquets from --data-glob "
                        "directly on local SSD. Set to enable lazy-fetch "
                        "from the remote warehouse into a bounded local "
                        "cache (--cache-max-gb).")
    p.add_argument("--remote-warehouse-base", default="/home/liu/synapforge_data",
                   help="directory on the warehouse host that contains "
                        "<dataset>/ subdirs of parquet shards. Combined "
                        "with --remote-warehouse-dataset to form the "
                        "remote path.")
    p.add_argument("--remote-warehouse-dataset", default="",
                   help="dataset name (subdir under --remote-warehouse-base "
                        "and the local cache). Required when "
                        "--remote-warehouse-host is set. Inferred from "
                        "--data-glob's parent dir name when empty.")
    p.add_argument("--remote-warehouse-cache-dir", default="/workspace/data_cache",
                   help="rental-local SSD cache root for warehouse fetches. "
                        "Created if missing; capped at --cache-max-gb.")
    p.add_argument("--cache-max-gb", type=float, default=30.0,
                   help="soft cap on the local warehouse cache (GiB). "
                        "After each fetch, oldest-atime shards are "
                        "evicted until the cache is below the cap. "
                        "Default 30 GiB — rule of thumb is ~10x a single "
                        "shard size at WT-103 / FineWeb scale, so the "
                        "trainer keeps roughly 1 epoch's worth of "
                        "lookahead resident on local SSD.")
    # T8.4 (DEEP_MAINT_QUEUE.md) — exponential moving average of weights.
    # Standard practice in DeepSeek / Llama / SmolLM2: track a shadow
    # copy ``model_ema = decay·model_ema + (1-decay)·model`` after each
    # optim.step() and use the EMA at inference. EMA state lives on CPU
    # in fp32 to avoid bf16 drift (~400MB at 100M params, ~free vs the
    # GPU live model). Default 0.0 = disabled (zero behaviour change for
    # existing runs); recommended 0.999 for a 100M-1B class run.
    p.add_argument("--ema-decay", type=float, default=0.0,
                   dest="ema_decay",
                   help="EMA decay for shadow weight tracking (T8.4). "
                        "Default 0.0 = OFF; 0.999 is the standard recipe. "
                        "When > 0, after each optim.step() the trainer "
                        "updates a CPU-fp32 EMA copy. At every "
                        "--save-every step the EMA state is BOTH "
                        "embedded inside step_<N>.pt under the "
                        "``ema_state`` key (preferred path, used by "
                        "``chat_demo --use-ema``) AND written to a sibling "
                        "step_<N>_ema.pt file for legacy "
                        "synapforge.training.ema.load_ema() loaders.")
    # T9.3 (DEEP_MAINT_QUEUE.md) -- multi-seq-len validation + monotonic
    # quality check. When --val-seq-lens is passed (CSV of ints), at each
    # VAL step the trainer ALSO runs validation at every additional
    # seq-len, building a fresh ParquetTokenStream per length. Output:
    #     VAL step N seq=256 ppl=X / seq=512 ppl=Y / seq=1024 ppl=Z
    #     quality_grows_with_seq=True
    # The monotonic flag is True iff ppl strictly DECREASES across the
    # provided lengths sorted ascending (more left-context => lower
    # per-token ppl). This is the user 铁律: "quality must grow with
    # context length, not just stay flat or degrade." Default: empty
    # string => behaviour unchanged (only val at training seq_len).
    p.add_argument("--val-seq-lens", type=str, default="",
                   dest="val_seq_lens",
                   help="comma-separated extra seq-lens for VAL "
                        "(e.g. '256,512,1024'). When non-empty, at each "
                        "VAL step the trainer runs validation at EACH of "
                        "these lengths in addition to training seq_len. "
                        "Logs one ``seq=N ppl=X`` token per length and a "
                        "``quality_grows_with_seq=True/False`` flag. "
                        "Default '' = no extra val (legacy behaviour).")
    p.add_argument("--val-seq-lens-bs", type=str, default="",
                   dest="val_seq_lens_bs",
                   help="comma-separated batch sizes paired 1:1 with "
                        "--val-seq-lens (e.g. '24,12,4' for 256/512/1024). "
                        "If empty AND --no-val-seq-lens-auto-scale, falls "
                        "back to --batch-size for every length (may OOM at "
                        "longer seq). Length must match --val-seq-lens.")
    p.add_argument("--val-seq-lens-auto-scale", action="store_true",
                   default=True, dest="val_seq_lens_auto_scale",
                   help="when --val-seq-lens-bs not provided, auto-scale "
                        "per-length batch as ``int(base_bs * (train_seq / "
                        "seq_len))`` floor 1, so longer seq drops batch "
                        "to fit VRAM (256->1024 = 16x activation). "
                        "Default ON.")
    p.add_argument("--no-val-seq-lens-auto-scale", action="store_false",
                   dest="val_seq_lens_auto_scale",
                   help="disable T9.3 multi-seq val auto-scaling; reuse "
                        "--batch-size at every seq-len (may OOM).")
    return p.parse_args()


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _detect_optim_state_layout(state: dict) -> str:
    """Speedup audit 2026-05-02 (Ship #2): detect optimizer state_dict layout.

    PlasticityAwareAdamW uses ``state[<id>] = {"step": int, "m": Tensor, "v":
    Tensor}`` (see synapforge/optim.py:279-280). torch.optim.AdamW uses
    ``state[<id>] = {"step": Tensor, "exp_avg": Tensor, "exp_avg_sq": Tensor}``
    (PyTorch upstream). When swapping between the two via ``--fused-adamw``
    the cross-load is unsafe -- ``optim.load_state_dict`` silently ignores
    mismatched keys and the result is a moment cold-start. Detect the
    layout up front so the trainer can warn + skip instead.

    Args:
        state: A ``optim.state_dict()`` dict with a top-level ``"state"`` key.

    Returns:
        ``"torch_adamw"`` if the per-param state has ``exp_avg`` /
        ``exp_avg_sq``; ``"plasticity_adamw"`` if it has ``m`` / ``v``;
        ``"unknown"`` if neither pattern matches (empty state dict or a
        third-party optimizer's state -- skip the load and warn).
    """
    if not isinstance(state, dict):
        return "unknown"
    inner = state.get("state", state)
    if not isinstance(inner, dict) or not inner:
        return "unknown"
    # Sample the first per-param entry to identify the layout.
    for _pid, pstate in inner.items():
        if not isinstance(pstate, dict):
            continue
        keys = set(pstate.keys())
        if "exp_avg" in keys or "exp_avg_sq" in keys:
            return "torch_adamw"
        if "m" in keys and "v" in keys:
            return "plasticity_adamw"
    return "unknown"


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
            # T2.5: ``last_spike_rate`` is now a method (returns the live
            # autograd-attached tensor when one is available, else the
            # detached buffer). ``.item()`` works on either.
            rate_accum.append([m.last_spike_rate().item() for m in plif_cells])
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


# ---------------------------------------------------------------------------
# T9.3 (DEEP_MAINT_QUEUE.md) -- multi-seq-len VAL + monotonic-quality check.
# ---------------------------------------------------------------------------
# Defaults to OFF (empty --val-seq-lens) so existing runs see zero behaviour
# change. When the user passes --val-seq-lens "256,512,1024", at every VAL
# step the trainer ALSO runs validation at those lengths, builds a per-seq
# val stream, and asserts the user 铁律: "quality grows with context length".
#
# The 铁律 (memory: feedback_50m_context_monotonic_quality.md and
# feedback_long_context_quality.md) is that an inference-STDP model should
# get LOWER per-token ppl as context grows, because more left-context means
# more bits to predict from. If ppl(L=1024) >= ppl(L=256), the model is
# wrong (or the STDP path is dead). The monotonic flag surfaces that
# regression as a single bool in train.log + metrics.json so phase_manager
# / dashboards can trip on it.


def _parse_val_seq_lens(
    val_seq_lens_csv: str,
    val_seq_lens_bs_csv: str,
    train_seq_len: int,
    train_batch_size: int,
    auto_scale: bool,
) -> list:
    """Resolve ``--val-seq-lens`` + ``--val-seq-lens-bs`` into pairs.

    Parameters
    ----------
    val_seq_lens_csv:
        Raw CSV from ``--val-seq-lens`` (e.g. ``"256,512,1024"``).
        Empty string => no extra val => return ``[]`` (caller will only
        validate at training seq_len, the legacy path).
    val_seq_lens_bs_csv:
        Raw CSV from ``--val-seq-lens-bs`` (e.g. ``"24,12,4"``). When
        non-empty, length MUST equal that of ``val_seq_lens_csv``;
        explicit pairing wins over auto-scale.
    train_seq_len:
        The training-time ``--seq-len``. Used for auto-scaling and as
        the implicit baseline in the monotonic check.
    train_batch_size:
        The training-time ``--batch-size``. The "base" for auto-scale.
    auto_scale:
        When True AND ``val_seq_lens_bs_csv`` is empty, derive batch as
        ``max(1, int(train_batch_size * train_seq_len / seq_len))``.
        When False, fall back to ``train_batch_size`` for every length
        (caller is responsible for any OOMs).

    Returns
    -------
    List of ``(seq_len, batch_size)`` tuples in user-supplied order. The
    monotonic check sorts by ``seq_len`` separately, so ordering only
    affects log presentation.

    Raises
    ------
    ValueError if the CSV is malformed or if ``--val-seq-lens-bs`` length
    does not match ``--val-seq-lens``.
    """
    s_lens = [t.strip() for t in (val_seq_lens_csv or "").split(",") if t.strip()]
    if not s_lens:
        return []
    try:
        seq_lens = [int(s) for s in s_lens]
    except ValueError as exc:
        raise ValueError(
            f"--val-seq-lens must be a CSV of ints; got {val_seq_lens_csv!r}"
        ) from exc
    for L in seq_lens:
        if L <= 0:
            raise ValueError(f"--val-seq-lens entries must be > 0; got {L}")

    bs_raw = [t.strip() for t in (val_seq_lens_bs_csv or "").split(",") if t.strip()]
    if bs_raw:
        if len(bs_raw) != len(seq_lens):
            raise ValueError(
                f"--val-seq-lens-bs has {len(bs_raw)} entries but "
                f"--val-seq-lens has {len(seq_lens)} entries; must match 1:1"
            )
        try:
            batch_sizes = [int(b) for b in bs_raw]
        except ValueError as exc:
            raise ValueError(
                f"--val-seq-lens-bs must be a CSV of ints; got "
                f"{val_seq_lens_bs_csv!r}"
            ) from exc
        for B in batch_sizes:
            if B <= 0:
                raise ValueError(f"--val-seq-lens-bs entries must be > 0; got {B}")
    else:
        if auto_scale:
            base_b = max(1, int(train_batch_size))
            base_s = max(1, int(train_seq_len))
            batch_sizes = [
                max(1, int(base_b * base_s / max(1, int(L))))
                for L in seq_lens
            ]
        else:
            batch_sizes = [int(train_batch_size)] * len(seq_lens)

    return list(zip(seq_lens, batch_sizes))


def _monotonic_quality_grows(per_seq_ppl: dict) -> bool:
    """Compute the user 铁律 flag: ppl strictly DECREASES with seq_len.

    The dict ``per_seq_ppl`` maps ``seq_len -> ppl``. We sort by seq_len
    ascending and check ``ppl[i+1] < ppl[i]`` for every consecutive pair.
    NaN/Inf entries fail the check (cannot conclude ppl improved). Single
    entry returns True trivially (cannot regress).

    Returns True iff longer context yielded strictly lower ppl across
    every adjacent pair. The "strict <" is intentional: equal ppl across
    different context lengths means the model is not USING the extra
    context, which is itself a regression vs. the inference-STDP claim.
    """
    if not per_seq_ppl:
        return True
    items = sorted(per_seq_ppl.items(), key=lambda kv: int(kv[0]))
    for i in range(len(items) - 1):
        p_short = float(items[i][1])
        p_long = float(items[i + 1][1])
        if not (math.isfinite(p_short) and math.isfinite(p_long)):
            return False
        if not (p_long < p_short):
            return False
    return True


def _eval_at_seq_lens(
    model,
    pairs: list,
    *,
    val_glob: str,
    data_glob: str,
    tokenizer_name: str,
    n_batches: int = 16,
    remote_warehouse=None,
):
    """Run :func:`evaluate` once per ``(seq_len, batch_size)`` pair.

    Builds a fresh non-looping ``ParquetTokenStream`` for each length so
    that each call sees the same first ``n_batches`` chunks (modulo the
    seq_len-driven re-tokenization). Catches FileNotFoundError on the
    val glob and falls back to the train glob, mirroring the main()
    bootstrap logic so smoke runs with a single parquet still work.

    Returns a dict ``seq_len -> ppl`` in the order given by ``pairs``.
    Empty ``pairs`` returns ``{}`` and the caller must handle that case
    (legacy single-seq path).
    """
    out: dict = {}
    if not pairs:
        return out
    for (seq_len_i, bs_i) in pairs:
        try:
            ds_i = ParquetTokenStream(
                val_glob, seq_len=int(seq_len_i),
                tokenizer_name=tokenizer_name,
                batch_size=int(bs_i), loop=False,
                remote_warehouse=remote_warehouse,
            )
        except FileNotFoundError:
            ds_i = ParquetTokenStream(
                data_glob, seq_len=int(seq_len_i),
                tokenizer_name=tokenizer_name,
                batch_size=int(bs_i), loop=False,
                remote_warehouse=remote_warehouse,
            )
        ppl_i = evaluate(model, iter(ds_i), n_batches=int(n_batches),
                          plif_cells=None)
        out[int(seq_len_i)] = float(ppl_i)
    return out


def _format_per_seq_log(
    step: int, per_seq_ppl: dict, monotonic: bool,
) -> str:
    """Render the multi-seq VAL log line.

    Format:
        VAL step N seq=256 ppl=X / seq=512 ppl=Y / seq=1024 ppl=Z
            quality_grows_with_seq=True

    The single-line `seq=...` column block is sorted by ``seq_len``
    ascending so the longest context is always rightmost; downstream
    parsers that care about ordering can rely on that. The
    quality-grows flag is on its own line below for grep-friendliness.
    """
    items = sorted(per_seq_ppl.items(), key=lambda kv: int(kv[0]))
    parts = [f"seq={int(L)} ppl={float(p):.2f}" for (L, p) in items]
    head = f"VAL step {int(step)} " + " / ".join(parts)
    return f"{head}\n  quality_grows_with_seq={bool(monotonic)}"


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
                plif_tau_init=(2.5 if getattr(args, "plif_tau_init", "unimodal") == "unimodal" else getattr(args, "plif_tau_init", "unimodal")),
                high_pass_residual_weight=getattr(args, "high_pass_residual_weight", 0.0),
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
        freeze_vocab_tail=bool(args.freeze_vocab_tail),
        lm_head_spectral_norm=bool(args.lm_head_spectral_norm),
        lm_head_pre_ln=bool(args.lm_head_pre_ln),
        weight_quant_cfc=str(args.quant_cfc_weights),
        latent_k=int(args.latent_k),
        sew_shortcut=bool(args.sew_shortcut),
        sparse_spike_synapse=bool(getattr(args, "sparse_spike_synapse", False)),
        sparse_spike_threshold=float(getattr(args, "sparse_spike_threshold", 0.30)),
    )
    # Log the sparse-spike-synapse flag state so post-mortems can verify
    # the dispatch is live (matches the existing kwta/sew/rfold pattern).
    if bool(getattr(args, "sparse_spike_synapse", False)):
        print(f"[sparse-spike] synapse path enabled "
              f"(threshold={float(getattr(args, 'sparse_spike_threshold', 0.30)):.2f}); "
              f"auto-fallback to dense above threshold spike density")
    n_params = model.num_parameters()
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")
    if str(args.quant_cfc_weights) == "ternary":
        # MATMUL_FREE.md M1 — log how many CfC input projections the
        # constructor wired as TernaryLinear so post-mortems can confirm
        # the flag actually took effect (and the QAT path isn't a no-op).
        from synapforge.quantize import TernaryLinear
        n_tern = sum(
            1 for m in model.modules() if isinstance(m, TernaryLinear)
        )
        print(f"[quant] CfC ternary QAT enabled: "
              f"{n_tern} TernaryLinear layers wired (delta_proj + b_proj)")
    plif_cells = [m for m in model.modules() if isinstance(m, PLIFCell)]
    print(f"plif cells found: {len(plif_cells)}")
    if int(args.latent_k) > 0:
        # T2.9 / arxiv:2412.06769 — log Coconut latent thinking is live so
        # post-mortems can confirm the flag took effect (the LatentThinker
        # adds 2 (d,d) Linear layers to state_dict; no overhead when k==0).
        print(f"[coconut] latent thinking enabled: k={args.latent_k} "
              f"(arxiv:2412.06769)")
    else:
        print("[coconut] latent thinking disabled (k=0, default)")

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
    # Speedup audit 2026-05-02 (Ship #2): when --fused-adamw is ON AND no
    # parameter is tagged with a non-bp plasticity source, build vanilla
    # torch.optim.AdamW(fused=True) instead of PlasticityAwareAdamW. The
    # fused kernel is ~3x faster than the per-param Python loop in
    # PlasticityAwareAdamW.step on Ultra (~340 trainable params).
    # Numerically equivalent to vanilla AdamW (Adam moment update is
    # kernel-implementation-independent). Detection rule: any param with
    # ``_sf_grad_source`` set to anything other than ``["bp"]`` (or with
    # plasticity sources mixed in) forces the safe PlasticityAwareAdamW
    # fallback so STDP / Hebb gradients are not silently dropped.
    _use_fused_adamw = bool(getattr(args, "fused_adamw", False))
    _use_synapforge_adamw = bool(getattr(args, "synapforge_adamw", False))
    _adamw_safe = True
    # Same plasticity-source safety check applies to both --fused-adamw
    # and --synapforge-adamw paths: if ANY param is tagged with a
    # non-bp grad source we fall back to PlasticityAwareAdamW so the
    # plasticity stream is not silently dropped.
    if _use_fused_adamw or _use_synapforge_adamw:
        _bad_param_names: list[str] = []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            sources = getattr(p, "_sf_grad_source", None)
            if sources is None:
                continue
            if list(sources) != ["bp"]:
                _adamw_safe = False
                _bad_param_names.append(f"{pname}={list(sources)}")
        if not _adamw_safe:
            _log("[plain-adamw] plasticity sources detected on "
                 f"{len(_bad_param_names)} params (e.g. "
                 f"{_bad_param_names[:3]!r}); falling back to "
                 "PlasticityAwareAdamW for correctness.")
    if _use_synapforge_adamw and _adamw_safe:
        # Phase 1 of the torch-replacement roadmap: pure-python AdamW
        # (no torch.optim inheritance). See docs/TORCH_REPLACEMENT_PLAN.md.
        # Wins over --fused-adamw: gets us out of torch.optim. Loses to
        # --fused-adamw: ~2-3 % slower step (no fused CUDA kernel).
        # Phase 4 of the roadmap will dispatch this same code path to a
        # Triton kernel, closing the perf gap without bringing torch.optim
        # back. Mutually exclusive with --fused-adamw (synapforge wins).
        from synapforge.optim import AdamW as _SynapforgeAdamW
        optim = _SynapforgeAdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=peak_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=WEIGHT_DECAY,
        )
        if _use_fused_adamw:
            _log("[synapforge-adamw] both --synapforge-adamw and "
                 "--fused-adamw passed; --synapforge-adamw wins (we are "
                 "migrating off torch.optim, not towards it).")
        _log("[synapforge-adamw] using synapforge.optim.AdamW "
             "(pure-python, NO torch.optim inheritance; ms_param_table "
             "absent so plasticity ticks rely on model-side state).")
    elif _use_fused_adamw and _adamw_safe and DEVICE == "cuda":
        # Vanilla fused AdamW. WEIGHT_DECAY mirrors build_optimizer's default.
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=peak_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=WEIGHT_DECAY,
            fused=True,
        )
        _log("[fused-adamw] using torch.optim.AdamW(fused=True) "
             "(ms_param_table absent; plasticity ticks must rely on "
             "model-side state, not optim.ms_param_table).")
    elif _use_fused_adamw and _adamw_safe and DEVICE != "cuda":
        _log("[fused-adamw] requires CUDA; falling back to "
             "PlasticityAwareAdamW.")
        optim = build_optimizer(model, lr=peak_lr, weight_decay=WEIGHT_DECAY)
    else:
        optim = build_optimizer(model, lr=peak_lr, weight_decay=WEIGHT_DECAY)
    print(f"optimizer: {type(optim).__name__} lr={peak_lr} wd={WEIGHT_DECAY}")

    # Load optimizer state from warmstart ckpt (preserves Adam m/v momentum;
    # without this, warmstart loses momentum state and the run cold-starts
    # the moment estimates -- a known cause of warmstart-divergence as seen
    # in v1.2 b80 (loss 5.45 -> 6.34 in 500 steps).
    # Ship #2: fused AdamW uses ``exp_avg`` / ``exp_avg_sq`` keys; vanilla
    # PlasticityAwareAdamW uses ``m`` / ``v``. Cross-load is unsafe (silently
    # ignored by torch.optim) so we detect a layout mismatch and warn instead
    # of crashing -- moments cold-start is a one-step penalty.
    if warm_ckpt and os.path.exists(warm_ckpt):
        try:
            _ck = torch.load(warm_ckpt, map_location="cpu")
            if isinstance(_ck, dict) and "optim_state" in _ck:
                _ck_state = _ck["optim_state"]
                _state_layout = _detect_optim_state_layout(_ck_state)
                # synapforge.optim.AdamW uses the same exp_avg/exp_avg_sq
                # state layout as torch.optim.AdamW, so it lives in the
                # ``torch_adamw`` bucket for layout-compat purposes.
                from synapforge.optim import AdamW as _SfAdamW  # noqa: PLC0415
                _running_layout = (
                    "torch_adamw"
                    if isinstance(optim, (torch.optim.AdamW, _SfAdamW))
                    else "plasticity_adamw"
                )
                if _state_layout != "unknown" and _state_layout != _running_layout:
                    _log(f"warmstart: optim_state layout mismatch "
                         f"(ckpt={_state_layout}, running={_running_layout}); "
                         "skipping load (Adam moments will cold-start; "
                         "one-step momentum penalty)")
                else:
                    optim.load_state_dict(_ck_state)
                    _log(f"warmstart: loaded optim state from {warm_ckpt}")
            else:
                _log(f"warmstart: no optim_state in ckpt (legacy ckpt; momentum cold-start)")
        except Exception as exc:
            _log(f"warmstart optim load skipped: {exc}")

    # ---------------- EMA tracker (T8.4) ----------------
    # Shadow CPU-fp32 copy of model weights, updated after each optim.step()
    # by ``ema_tracker.update(model)``. Default --ema-decay=0.0 = OFF.
    # When >0, every --save-every step also dumps step_<N>_ema.pt next to
    # the live ckpt; chat_demo / chat_repl can swap to the EMA copy via
    # synapforge.training.ema.load_ema().
    ema_tracker = None
    if float(getattr(args, "ema_decay", 0.0)) > 0.0:
        try:
            from synapforge.training.ema import EMATracker
            ema_tracker = EMATracker(model, decay=float(args.ema_decay))
            _log(f"[ema] tracker enabled: decay={args.ema_decay} "
                 f"(state on CPU fp32)")
        except Exception as exc:
            _log(f"[ema] init FAILED ({exc!r}); disabling EMA")
            ema_tracker = None
    else:
        _log("[ema] disabled (default; pass --ema-decay 0.999 to enable)")

    # ---------------- data ----------------
    # P9: data globs and tokenizer are CLI-overridable so the smoke test
    # can point at a tiny synth parquet + the gpt2 tokenizer (no rental
    # path required). Defaults preserve the WT-103 + Qwen rental setup.
    # P24 (MASTER_PLAN.md §6): TRAIN stream gets a streaming reservoir
    # shuffle; VAL stream stays deterministic so eval ppl is comparable
    # across runs.
    # 2026-05-01: optional remote warehouse — see
    # docs/REMOTE_DATA_WAREHOUSE.md. When --remote-warehouse-host is
    # set, parquet shards are lazy-fetched from mohuanfang into a
    # bounded local LRU cache instead of read directly from local disk.
    _warehouse = None
    if args.remote_warehouse_host:
        from synapforge.data import RemoteDataWarehouse
        _wh_dataset = args.remote_warehouse_dataset.strip()
        if not _wh_dataset:
            # Infer from the parent dir basename of --data-glob.
            _wh_dataset = os.path.basename(
                os.path.dirname(args.data_glob.rstrip("/"))
            ) or "default"
        _warehouse = RemoteDataWarehouse(
            dataset=_wh_dataset,
            cache_dir=args.remote_warehouse_cache_dir,
            max_cache_gb=float(args.cache_max_gb),
            remote_host=args.remote_warehouse_host,
            remote_base=args.remote_warehouse_base,
        )
        _log(f"[data] remote warehouse ON: {_warehouse!r}")

    train_ds = ParquetTokenStream(args.data_glob, seq_len=seq_len,
                                  tokenizer_name=args.tokenizer_name,
                                  batch_size=args.batch_size, loop=True,
                                  shuffle_buffer=int(args.shuffle_buffer),
                                  shuffle_seed=int(args.shuffle_seed),
                                  prefetch_factor=int(args.prefetch_factor),
                                  pin_memory=bool(args.pin_memory),
                                  remote_warehouse=_warehouse)
    train_it = iter(train_ds)
    print(f"train stream: {train_ds!r}")

    # If --val-glob points at a missing path (smoke runs only have a single
    # parquet), reuse the train glob so eval still runs.
    _val_glob = args.val_glob if args.val_glob else args.data_glob
    try:
        val_ds = ParquetTokenStream(_val_glob, seq_len=seq_len,
                                    tokenizer_name=args.tokenizer_name,
                                    batch_size=args.batch_size, loop=False,
                                    remote_warehouse=_warehouse)
    except FileNotFoundError:
        _log(f"[data] val glob {_val_glob!r} not found; falling back to train glob for eval")
        val_ds = ParquetTokenStream(args.data_glob, seq_len=seq_len,
                                    tokenizer_name=args.tokenizer_name,
                                    batch_size=args.batch_size, loop=False,
                                    remote_warehouse=_warehouse)
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
    neuromcp_mixin = None
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

    # NeuroMCPMixin lives in ``synapforge.training`` (alongside the EMA
    # tracker), NOT in the umbrella ``synapforge.trainer_mixins`` module --
    # this keeps the wire-in independent of the older mixin import block
    # so a missing ``intrinsic`` / ``modal`` import there never blocks
    # NeuroMCP. Default OFF; only constructed when --neuromcp-weight > 0.
    if float(args.neuromcp_weight) > 0:
        try:
            from synapforge.training import NeuroMCPMixin
            neuromcp_mixin = NeuroMCPMixin(
                model,
                hidden=model.d,
                codebook_size=int(args.neuromcp_codebook_size),
                action_dim=int(args.neuromcp_action_dim),
            )
            if neuromcp_mixin.head is None:
                _log("[mixin] NeuroMCPMixin head failed to build; disabling")
                neuromcp_mixin = None
            else:
                # Push the head's parameters into the optimizer so the
                # synaptic mask + codebook prototypes actually train. We
                # add a fresh param group so existing per-group LR /
                # weight-decay state stays untouched.
                try:
                    optim.add_param_group({
                        "params": list(neuromcp_mixin.parameters()),
                        "lr": float(args.lr),
                        "weight_decay": 0.0,
                    })
                except Exception as _exc:
                    _log(f"[mixin] NeuroMCP param-group add failed (head still "
                         f"runs forward but won't update): {_exc!r}")
                _log(f"[mixin] NeuroMCPMixin enabled: weight={args.neuromcp_weight} "
                     f"codebook_size={args.neuromcp_codebook_size} "
                     f"action_dim={args.neuromcp_action_dim} "
                     f"hidden={model.d} initial_density={neuromcp_mixin.density:.3f} "
                     f"initial_K={neuromcp_mixin.K}")
        except Exception as exc:
            _log(f"[mixin] NeuroMCPMixin DISABLED: {exc}")
            neuromcp_mixin = None
    else:
        _log("[mixin] neuromcp: disabled (weight=0)")

    # ---------------- training ----------------
    # P3: track BOTH val_ppl_ttt (set TTT trains on; drops artificially
    # after TTT inner step) and val_ppl_holdout (set TTT never sees;
    # honest signal phase_manager gates on). ``ppl_eval`` retained as
    # alias of holdout for legacy parsers/tooling.
    metrics = {"step": [], "loss": [], "step_ms": [], "tok_per_s": [],
               "ppl_eval": {}, "ppl_eval_ttt": {}, "ppl_eval_holdout": {},
               "samples": {}}
    # ---- Perf audit 2026-05-02: cached trainable-param list for clip ----
    # When --clip-grad-cache is on, capture requires_grad=True params ONCE
    # and reuse the list for clip_grad_norm_ every step, instead of
    # rebuilding [p for p in model.parameters() if p.requires_grad] inside
    # the hot loop. Saves ~50us/step CPU-side dispatch on Ultra (~340
    # params at d=1280/n=16). Default OFF; phase-aware exit-101 path
    # causes the trainer to exit + relauncher rebuilds anyway, so the
    # cache lifecycle is bounded by one process.
    _clip_param_cache = None
    if bool(getattr(args, "clip_grad_cache", False)):
        _clip_param_cache = [
            p for p in model.parameters() if p.requires_grad
        ]
        _log(f"[clip-grad-cache] cached {len(_clip_param_cache)} trainable "
             f"params at init; will reuse every step (saves listcomp).")
    # T5.4 -- track lowest val_ppl_holdout we've seen this run. The
    # `_update_best_ckpt` helper compares against this and (on improvement)
    # creates/updates a ``best_step_<N>.pt`` symlink/copy so a relauncher can
    # warmstart from the actual best ckpt without grepping the log.
    best_val_ppl_holdout = float("inf")
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

    # T2.7: gradient accumulation. accum_steps>=1 micro-batches per optim
    # step. Loss is divided by accum_steps so the accumulated grad equals
    # the gradient that bs_eff = batch_size * accum_steps would produce in
    # a single big batch. optim.step() / zero_grad() run once per global
    # step; data, autocast, KD, mixins all run per micro-batch.
    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    if accum_steps > 1:
        _log(f"[grad-accum] {accum_steps} micro-batches/step "
             f"-> bs_eff = {args.batch_size * accum_steps} "
             f"(VRAM stays at bs={args.batch_size})")

    # Speedup audit 2026-05-02 (Ship #1): cache the lazy-host-sync flag once.
    # When True the inner-loop accumulators are GPU tensors; when False they
    # are Python floats (current behaviour). The branch is per-step (not
    # per-microbatch) so the cost is one int compare/step, free.
    _lazy_host_sync_accum = bool(getattr(args, "lazy_host_sync_accum", False))
    if _lazy_host_sync_accum:
        _log("[lazy-host-sync-accum] ON: 6 .item() calls/microbatch deferred "
             "to log boundary (saves ~12 host-syncs/step at accum=2)")
    # Skip-warmstart-eval-N (Ship #3): track how many evals to skip.
    _skip_eval_remaining = max(0, int(getattr(args, "skip_warmstart_eval_n", 0)))
    if _skip_eval_remaining > 0:
        if warm_ckpt and os.path.exists(warm_ckpt):
            _log(f"[skip-warmstart-eval-N] will skip first "
                 f"{_skip_eval_remaining} val evaluation(s) "
                 f"(warmstart from {warm_ckpt})")
        else:
            _log(f"[skip-warmstart-eval-N] requested {_skip_eval_remaining} "
                 "but no warmstart ckpt -- ignoring (every eval will run)")
            _skip_eval_remaining = 0

    # Run 5 PLIF-dead fix #1: capture dense-bypass window so the toggle
    # below is O(1) per step.  When --plif-dense-bypass-steps == 0 the
    # window is empty and PLIFCell.dense_bypass stays False forever.
    _plif_dense_bypass_steps = int(getattr(args, "plif_dense_bypass_steps", 0))
    if _plif_dense_bypass_steps > 0:
        for _m in plif_cells:
            _m.dense_bypass = True
        _log(f"[plif-fix #1] dense-bypass ON for steps 1..{_plif_dense_bypass_steps} "
             f"({len(plif_cells)} PLIF cells emit tanh(v - thr) instead of binary spikes; "
             f"breaks dead-PLIF positive feedback per docs/PLIF_DEAD_DIAGNOSIS.md)")
    if bool(getattr(args, "sew_shortcut", False)):
        # T2.5/T2.6 cross-check: count blocks that built the SEW residual.
        from synapforge.model_100m import HybridBlock as _HB
        _n_sew = sum(1 for m in model.modules() if isinstance(m, _HB) and getattr(m, "sew_shortcut", False))
        _log(f"[plif-fix #3] SEW (Spike-Element-Wise) shortcut ENABLED on {_n_sew} HybridBlocks "
             f"(arxiv:2102.04159; gated = synapse(s + h) * sigmoid(...))")

    for step in range(1, n_steps + 1):
        # Run 5 PLIF-dead fix #1 -- toggle dense_bypass off at the boundary.
        if _plif_dense_bypass_steps > 0 and step == _plif_dense_bypass_steps + 1:
            for _m in plif_cells:
                _m.dense_bypass = False
            _log(f"[plif-fix #1] dense-bypass OFF at step {step} "
                 f"-- PLIF cells now emit binary spikes + ATan surrogate")
        cur_lr = lr_at(step, peak_lr, args.warmup, n_steps, args.lr_decay)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr
        t_step = time.time()
        # Per-step loss-component accumulators. We log the *mean* across the
        # N micro-batches so the values are comparable to the bs=accum_steps*B
        # equivalent single-batch run. (Sum-and-divide because each
        # micro-batch's loss is already pre-divided by accum_steps before
        # backward; for logging we want the un-scaled mean.)
        # Speedup audit 2026-05-02 (Ship #1): when --lazy-host-sync-accum is
        # ON, accumulators are GPU tensors (no .item() in inner loop) and
        # only materialize at log boundary; when OFF, accumulators are
        # Python floats (current behaviour).
        if _lazy_host_sync_accum:
            # Lazy mode: GPU tensor scalars. Initialised to fp32 zeros on
            # DEVICE; tensor adds keep them on-stream. Materialised once
            # per global step below.
            _device_for_accum = torch.device(DEVICE) if DEVICE != "cpu" else torch.device("cpu")
            accum_total_loss_t = torch.zeros((), dtype=torch.float32,
                                              device=_device_for_accum)
            accum_ce_t = torch.zeros_like(accum_total_loss_t)
            accum_kd_t = torch.zeros_like(accum_total_loss_t)
            accum_z_t = torch.zeros_like(accum_total_loss_t)
            accum_modal_aux_t = torch.zeros_like(accum_total_loss_t)
            accum_cur_aux_t = torch.zeros_like(accum_total_loss_t)
        # Always-defined (used by both modes; default-mode reads them, lazy
        # mode rebinds to materialised values after the inner loop).
        accum_total_loss = 0.0
        accum_ce = 0.0
        accum_kd = 0.0
        accum_z = 0.0
        accum_modal_aux = 0.0
        accum_cur_aux = 0.0
        data_exhausted = False

        # ---- gradient accumulation inner loop ----
        # zero_grad ONCE per global step, before the inner loop. Each inner
        # iter calls .backward() which adds into .grad in place.
        optim.zero_grad(set_to_none=True)
        for _accum_iter in range(accum_steps):
            try:
                x, y = next(train_it)
            except StopIteration:
                _log(f"data exhausted at step {step} "
                     f"(accum_iter={_accum_iter}/{accum_steps})")
                data_exhausted = True
                break
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE,
                                    enabled=DEVICE == "cuda"):
                # When a mixin needs the hidden state, take the encode -> head
                # path so we can pass `text_hidden` into the mixin without a
                # second forward. Default path (no mixin) is unchanged.
                need_hidden = (modal_mixin is not None
                               or curiosity_mixin is not None
                               or neuromcp_mixin is not None)
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
                # z-loss: penalize large logsumexp (numerical stability +
                # softmax temp control). 2026-05-01: default ``--z-loss-topk
                # 2048`` uses the sparse approximation so the
                # (B*T, V, fp32) intermediate never materialises -- this is
                # what unlocks bs=64 -> bs=80 on A800-80GB. Set
                # --z-loss-topk 0 to recover the full-vocab path.
                log_z = _sparse_z_loss(flat_logits, k=int(args.z_loss_topk))
                z_loss = (log_z ** 2).mean()
                base_loss = ce_loss + args.z_loss_weight * z_loss

                if teacher is not None and args.kd_weight > 0:
                    # Only run the (expensive) teacher forward on KD-active
                    # steps. On KD-skip steps fall back to pure base_loss
                    # without the (1-α) downweight that would otherwise scale
                    # the LM gradient by 0.3× on 3/4 steps (effectively
                    # dropping LR; see review).
                    # ``effective_kd_every`` == ``args.kd_every`` unless
                    # --kd-every-adaptive is on, in which case it's
                    # auto-tuned from the student-teacher CE gap (see
                    # ``_adaptive_kd_every`` and the post-step update below).
                    if step % effective_kd_every == 0:
                        if kd_teacher_stream is not None:
                            # Push teacher forward onto side stream so it
                            # can overlap with the student backward (which
                            # still runs on the default stream). Teacher is
                            # frozen + .eval() so the side stream needs no
                            # autograd interaction. We sync before KL to
                            # guarantee t_logits is ready.
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
                        loss = (1.0 - args.kd_weight) * base_loss \
                            + args.kd_weight * kd
                    else:
                        kd = torch.zeros((), device=logits.device)
                        loss = base_loss
                        # ---- adaptive --kd-every: track CE on KD-OFF
                        # steps ---- We sample CE from KD-OFF steps only so
                        # the gap measure isn't artificially pulled down by
                        # the KD loss itself. ``ce_loss.item()`` triggers a
                        # host-sync but only on KD-OFF steps, which already
                        # pay no teacher-forward cost.
                        if args.kd_every_adaptive:
                            kd_off_ce_window.append(
                                float(ce_loss.detach().item()))
                            if len(kd_off_ce_window) > KD_ADAPT_WINDOW:
                                kd_off_ce_window = (
                                    kd_off_ce_window[-KD_ADAPT_WINDOW:])
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
                # ---- NeuroMCPMixin contribution (default OFF) ----
                # action_loss runs the LM hidden through SparseSynapticLayer +
                # DynamicActionCodebook and CE-trains the head against a
                # placeholder action target derived from y. Pass y_next so
                # the mixin can synthesise targets without a 2nd forward.
                neuromcp_aux = torch.zeros((), device=logits.device)
                if neuromcp_mixin is not None and text_hidden is not None:
                    try:
                        neuromcp_aux = neuromcp_mixin.action_loss(
                            text_hidden, y_next=y,
                        )
                        loss = loss + args.neuromcp_weight * neuromcp_aux
                    except Exception as exc:
                        _log(f"[mixin] neuromcp step {step} skipped: {exc}")
                        neuromcp_aux = torch.zeros((), device=logits.device)

            # T2.5: spike-rate-target auxiliary loss (graph-attached).
            # MUST run inside inner loop, BEFORE backward, while graph alive.
            spike_target_loss = torch.zeros((), device=logits.device)
            if args.spike_target_loss_weight > 0 and plif_cells:
                low = float(args.spike_target_loss_low)
                high = float(args.spike_target_loss_high)
                terms = []
                for m in plif_cells:
                    rate = m.last_spike_rate()
                    if rate.dim() > 0:
                        rate = rate.squeeze()
                    rate = rate.float()
                    over = (rate - high).clamp(min=0.0).pow(2)
                    under = (low - rate).clamp(min=0.0).pow(2)
                    terms.append(over + under)
                if terms:
                    spike_target_loss = torch.stack(terms).sum()
                    loss = loss + args.spike_target_loss_weight * spike_target_loss

            # T2.7: scale loss so the accumulated gradient over N micro-
            # batches equals the gradient bs_eff=batch_size*N would produce
            # in one big batch. backward() runs N times per global step,
            # zero_grad runs once before the inner loop, optim.step()
            # runs once after. accum_steps=1 leaves behaviour unchanged.
            (loss / float(accum_steps)).backward()

            # Track raw (un-scaled) loss components for logging. We sum
            # then divide by accum_steps_done so the logged value equals
            # the un-divided per-batch mean (comparable to bs=B*N runs).
            # Speedup audit 2026-05-02 (Ship #1): when --lazy-host-sync-accum
            # is ON, sum the detached tensors on GPU (no host stall).
            # When OFF, sum the materialised Python floats (current
            # behaviour, 6 .item() host syncs per microbatch).
            if _lazy_host_sync_accum:
                accum_total_loss_t = accum_total_loss_t + loss.detach().float()
                accum_ce_t = accum_ce_t + ce_loss.detach().float()
                accum_kd_t = accum_kd_t + kd.detach().float()
                accum_z_t = accum_z_t + z_loss.detach().float()
                accum_modal_aux_t = accum_modal_aux_t + modal_aux.detach().float()
                accum_cur_aux_t = accum_cur_aux_t + cur_aux.detach().float()
            else:
                accum_total_loss += float(loss.detach().item())
                accum_ce += float(ce_loss.detach().item())
                accum_kd += float(kd.detach().item())
                accum_z += float(z_loss.detach().item())
                accum_modal_aux += float(modal_aux.detach().item())
                accum_cur_aux += float(cur_aux.detach().item())

        # Compute the number of micro-batches actually run this step
        # (StopIteration mid-accum partial step). We still optim.step on
        # whatever grad we have so we don't waste the partial.
        accum_done = max(1, _accum_iter + (0 if data_exhausted else 1))
        if data_exhausted and _accum_iter == 0:
            # Truly nothing this step -- exit outer loop.
            break

        # P30 fix: zero_grad ran before the inner accum loop (line ~1245);
        # backward ran inside the inner loop (T2.7 grad-accum + T2.5 spike
        # target merged in). Just clip + step here.
        if GRAD_CLIP > 0:
            # Perf audit 2026-05-02 (RECO #2): when --clip-grad-cache is on
            # reuse the cached trainable-param list captured before the
            # training loop (saves the listcomp per step). Default OFF =>
            # rebuild fresh every step (back-compat).
            _clip_params = (
                _clip_param_cache
                if _clip_param_cache is not None
                else [p for p in model.parameters() if p.requires_grad]
            )
            torch.nn.utils.clip_grad_norm_(
                _clip_params,
                max_norm=GRAD_CLIP,
            )

        # T5.3 (DEEP_MAINT_QUEUE.md) — per-named-module grad-norm log.
        # Emitted AFTER clip_grad_norm_ (so the rendered numbers are the
        # actually-applied post-clip grads) and BEFORE optim.step() (so
        # ``p.grad`` still references the live gradient about to update
        # the parameters; ``optim.step()`` does NOT zero grads but it
        # does mutate ``p.data``, which is irrelevant here since we only
        # read ``p.grad``). Cadence 100 steps so the per-parameter
        # ``norm().item()`` reduction does not cost real wall time on
        # the hot loop. Behind ``--log-grad-norm`` (default OFF) so
        # production launches see no schema change unless they ask.
        if args.log_grad_norm and step % 100 == 0:
            _grad_pairs = _compute_grad_norm_per_named_module(model)
            _log("  " + _format_grad_norm_per_module(_grad_pairs))

        optim.step()

        # ---- NeuroMCP plasticity tick (after optim.step per the
        # OBSERVE/DELTA/APPLY contract documented in
        # synapforge.action.neuromcp.NeuroMCPHead.step_plasticity).
        # Grows the SparseSynapticLayer mask via co-activation EMA and
        # the DynamicActionCodebook prototype set via cosine novelty.
        # Cheap (no_grad, only when mixin enabled).
        if neuromcp_mixin is not None:
            try:
                neuromcp_mixin.step_plasticity()
            except Exception as exc:
                _log(f"[mixin] neuromcp step_plasticity step {step} failed: {exc}")

        # T8.4 — EMA update right after optim.step() (model params now reflect
        # the just-applied gradient). No-op when --ema-decay 0.0 (default).
        if ema_tracker is not None:
            try:
                ema_tracker.update(model)
            except Exception as exc:  # never let EMA error kill training
                _log(f"[ema] update failed step={step} ({exc!r}); disabling")
                ema_tracker = None

        # Perf audit 2026-05-02 (RECO #1): when --cuda-sync-every N>1, only
        # call torch.cuda.synchronize() every N global steps instead of
        # every step. This unblocks the host clock loop so the GPU can
        # pipeline several student bwds ahead of the host -- big win when
        # paired with --kd-async-teacher (the side-stream teacher fwd
        # finally has room to overlap). Default 1 = current behaviour.
        # NaN guards intact: loss.item() at log boundary still host-syncs.
        _sync_period = max(1, int(getattr(args, "cuda_sync_every", 1)))
        if DEVICE == "cuda" and (step % _sync_period == 0):
            torch.cuda.synchronize()
        step_ms = (time.time() - t_step) * 1000.0
        # T2.7: count tokens across all completed micro-batches this step.
        # When accum_steps=1, accum_done=1 (back-compat).
        cum_tok += args.batch_size * seq_len * accum_done

        # Speedup audit 2026-05-02 (Ship #1): when --lazy-host-sync-accum is
        # ON, materialise the GPU tensor accumulators to Python floats here
        # (one host-sync per step instead of 6×accum_steps host-syncs/step).
        # Done at the natural sync boundary -- after optim.step() and the
        # cuda.synchronize gate. The 6 .item() calls below run on the
        # already-syncrhonised stream so they are essentially free.
        if _lazy_host_sync_accum:
            accum_total_loss = float(accum_total_loss_t.item())
            accum_ce = float(accum_ce_t.item())
            accum_kd = float(accum_kd_t.item())
            accum_z = float(accum_z_t.item())
            accum_modal_aux = float(accum_modal_aux_t.item())
            accum_cur_aux = float(accum_cur_aux_t.item())

        # Per-step mean loss components (raw, un-divided) for logging.
        # These mirror what bs=B*N would log in a single big batch.
        # Underscore-prefixed to avoid shadowing inner ``mean_ce`` used by
        # the kd-every-adaptive branch below.
        _step_loss = accum_total_loss / accum_done
        _step_ce = accum_ce / accum_done
        _step_kd = accum_kd / accum_done
        _step_z = accum_z / accum_done
        _step_modal_aux = accum_modal_aux / accum_done
        _step_cur_aux = accum_cur_aux / accum_done

        # ---- NeuroMCP periodic stats (every 100 steps when enabled) ----
        # Reads cheap counters from the mixin: density, K, last_action_loss,
        # last_hit_rate. Logged separately from the per-step `_log` line so
        # parsers don't need to handle a variable-width column.
        if neuromcp_mixin is not None and step % 100 == 0:
            _nstats = neuromcp_mixin.stats()
            _log(
                f"  [neuromcp] step={step} density={_nstats['density']:.4f} "
                f"K={int(_nstats['K'])} action_loss={_nstats['last_action_loss']:.3f} "
                f"hit_rate={_nstats['last_hit_rate']:.3f}"
            )

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
            metrics["loss"].append(_step_loss)
            metrics["step_ms"].append(step_ms)
            metrics["tok_per_s"].append(tok_s)
            mem_str = (
                f" mem_GB={torch.cuda.memory_allocated()/1e9:.2f}"
                if DEVICE == "cuda" else ""
            )
            mixin_str = ""
            if modal_mixin is not None:
                mixin_str += f" modal={_step_modal_aux:.4f}"
            if curiosity_mixin is not None:
                mixin_str += f" cur={float(cur_aux.detach()):.4f}"
            if neuromcp_mixin is not None:
                mixin_str += (
                    f" nmcp={float(neuromcp_aux.detach()):.4f}"
                    f"|d={neuromcp_mixin.density:.3f}|K={neuromcp_mixin.K}"
                )
            if args.spike_target_loss_weight > 0:
                mixin_str += f" stl={float(spike_target_loss.detach()):.4f}"
            # T5.1 — loss component %. Computed from the per-step accumulator
            # means (``_step_*``) so the percentages match the same window
            # whatever ``--accum`` is set to. ``_format_loss_pct`` re-weights
            # the un-weighted ``z`` / ``cur`` accumulators and skips
            # ``pct_modal`` / ``pct_cur`` when the corresponding mixin is
            # disabled. Off when ``--no-log-loss-pct`` is passed.
            pct_str = ""
            if args.log_loss_pct:
                pct_str = _format_loss_pct(
                    step_loss=_step_loss,
                    step_ce=_step_ce,
                    step_kd=_step_kd,
                    step_z=_step_z,
                    z_loss_weight=args.z_loss_weight,
                    kd_weight=args.kd_weight,
                    step_modal_aux=_step_modal_aux,
                    has_modal=modal_mixin is not None,
                    step_cur_aux=_step_cur_aux,
                    cur_weight=args.curiosity_weight,
                    has_curiosity=curiosity_mixin is not None,
                )
            _log(f"step {step:5d} loss={loss.item():.4f} ce={ce_loss.item():.3f} "
                 f"kd={kd.item():.3f} z={z_loss.item():.3f} lr={cur_lr:.5f} "
                 f"step_ms={step_ms:.1f} tok/s={tok_s:.0f}{mem_str}{mixin_str}"
                 f"{pct_str}")
            if step % 50 == 0 and plif_cells:
                # T2.5: ``last_spike_rate`` is a method now.
                rates = [m.last_spike_rate().item() for m in plif_cells]
                rate_min, rate_max = min(rates), max(rates)
                rate_mean = sum(rates) / len(rates)
                n_dead = sum(1 for r in rates if r < 0.005)
                n_sat = sum(1 for r in rates if r > 0.5)
                _log(
                    f"  spike: mean={rate_mean:.3f} "
                    f"range=[{rate_min:.3f}, {rate_max:.3f}] "
                    f"dead={n_dead}/{len(rates)} sat={n_sat}/{len(rates)}"
                )
                # T5.2: per-layer spike rate breakdown. Behind opt-in flag
                # so existing aggregated-line parsers stay undisturbed.
                # Right BELOW the aggregated line for visual locality.
                if args.log_spike_per_layer:
                    _log("  " + _format_spike_rates_per_layer(rates))
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
            ckpt_payload = {
                "model": model.state_dict(),
                "optim_state": optim.state_dict(),
                "step": step,
                "loss": float(loss.item()),
                "n_params": n_params,
                "lr": cur_lr,
                "config": _build_config_dict(args),
            }
            # T8.4 — embed the EMA shadow state inside the live ckpt under
            # the ``ema_state`` key (per the T8.4 spec). chat_demo's
            # ``--use-ema`` flag prefers this embedded copy because it's
            # bit-exact paired with the live state. We ALSO write a
            # sibling step_<N>_ema.pt below so legacy loaders keep working.
            if ema_tracker is not None:
                try:
                    ckpt_payload["ema_state"] = dict(ema_tracker.state_dict())
                    ckpt_payload["ema_decay"] = float(ema_tracker.decay)
                except Exception as exc:
                    _log(f"[ema] embed in ckpt failed step={step}: {exc!r}")
            torch.save(ckpt_payload, ckpt_path)
            _log(f"saved ckpt {ckpt_path}")
            # T8.4 — when EMA is enabled, dump a sibling _ema.pt with the
            # EMA-smoothed weights using the same dict layout so the ckpt
            # loaders (chat_demo / chat_repl / load_ema) can consume it
            # by exactly the same code path.
            if ema_tracker is not None:
                try:
                    ema_path = os.path.join(out_dir, f"step_{step:06d}_ema.pt")
                    ema_tracker.save(
                        ema_path,
                        extra={
                            "step": step,
                            "n_params": n_params,
                            "lr": cur_lr,
                            "config": _build_config_dict(args),
                        },
                    )
                    _log(f"saved ema ckpt {ema_path}")
                except Exception as exc:
                    _log(f"[ema] save failed step={step}: {exc!r}")

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
            # Speedup audit 2026-05-02 (Ship #3): when warmstarting from a
            # known-good ckpt, skip the first N val evaluations. We already
            # know the baseline ppl from the prior run's metrics.json --
            # re-running eval at step 10500 (when warm-loaded from
            # step_10000) is wall-time waste (~30s/eval × 2 evals = ~60s).
            # Default behaviour (N=0) preserved: every eval-every step
            # still runs both ttt + holdout val. Step counter, save_every,
            # multi-seq val, chat samples, best-ckpt tracking unaffected.
            # NOTE: keeping ``step == n_steps`` as a hard force so the
            # final-step eval ALWAYS runs (otherwise a short relaunch
            # could finish without persisting any val_ppl_holdout in
            # metrics.json).
            if _skip_eval_remaining > 0 and step != n_steps:
                _log(f"VAL step {step}: SKIPPED "
                     f"(--skip-warmstart-eval-N "
                     f"{int(args.skip_warmstart_eval_n)}, "
                     f"{_skip_eval_remaining - 1} remaining after this)")
                _skip_eval_remaining -= 1
                # Continue to save/phase-aware paths below; just bypass
                # the (expensive) eval + best-ckpt update.
                continue
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
            # T9.3 -- multi-seq-len VAL + monotonic-quality flag.
            # Default OFF (--val-seq-lens empty); when set, run
            # evaluate() at every requested length with auto-scaled
            # batch and emit the user 铁律 flag. Fail-soft: any error
            # (OOM at long seq, malformed CSV) logs and falls through.
            try:
                _vsl_pairs = _parse_val_seq_lens(
                    val_seq_lens_csv=str(getattr(args, "val_seq_lens", "") or ""),
                    val_seq_lens_bs_csv=str(getattr(args, "val_seq_lens_bs", "") or ""),
                    train_seq_len=int(args.seq_len),
                    train_batch_size=int(args.batch_size),
                    auto_scale=bool(getattr(args, "val_seq_lens_auto_scale", True)),
                )
            except ValueError as exc:
                _log(f"  [multi-seq-val] arg parse failed (skipping): {exc}")
                _vsl_pairs = []
            if _vsl_pairs:
                try:
                    _per_seq_ppl = _eval_at_seq_lens(
                        model, _vsl_pairs,
                        val_glob=(args.val_glob or args.data_glob),
                        data_glob=args.data_glob,
                        tokenizer_name=args.tokenizer_name,
                        n_batches=16,
                        remote_warehouse=_warehouse,
                    )
                    _grows = _monotonic_quality_grows(_per_seq_ppl)
                    _log(_format_per_seq_log(step, _per_seq_ppl, _grows))
                    metrics.setdefault("ppl_eval_per_seq", {})[step] = _per_seq_ppl
                    metrics.setdefault("quality_grows_with_seq", {})[step] = bool(_grows)
                except Exception as exc:  # pragma: no cover
                    _log(f"  [multi-seq-val] eval failed step={step}: {exc!r}")
            # T5.4 -- maintain best_step_*.pt link/copy. Fail-soft: any
            # error (disk full, missing src ckpt, FS without symlinks) is
            # logged but does not interrupt training.
            try:
                best_val_ppl_holdout, _best_link = _update_best_ckpt(
                    out_dir=out_dir,
                    step=step,
                    val_ppl=float(ppl_holdout),
                    best_val_ppl=best_val_ppl_holdout,
                    enabled=bool(getattr(args, "best_ckpt_track", True)),
                    log_fn=_log,
                )
            except Exception as exc:  # pragma: no cover
                _log(f"[best-ckpt] update failed (continuing): {exc}")
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
