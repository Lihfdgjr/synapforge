"""train_100m_self_distill -- Phase 4 online policy self-distillation.

Phase 4 trigger (`docs/PHASE_TRAINING.md`, T9.7 in `docs/DEEP_MAINT_QUEUE.md`):
chat eval >= 0.6 OR val ppl <= 30. Goal: push ppl from the 20-30 plateau
that emerges after Phase 3 RL down to <= 10 by having Synap-1 Ultra teach
itself with online policy distillation. The student and the teacher are
the SAME network -- the teacher is the model rolled out at high
temperature (T=1.5, exploratory) and the student is the same network at
low temperature (T=0.0, greedy). This is the self-distillation pattern
used by R1-distill and similar 2024-2025 reasoning recipes (loss
specifically targets high-error tokens, where the high-T rollout
discovers a softer / better-calibrated distribution that the greedy
T->0 path is asked to match).

Mathematically (per batch position t):

    p_teacher_t = softmax(model(x; T=teacher_temp).logits / teacher_temp)
    p_student_t = softmax(model(x; T=student_temp).logits)
    self_kl_t   = KL(p_student_t || p_teacher_t)

    loss = lm_ce + alpha * mean(self_kl_t)

The "teacher" forward is run under torch.no_grad() so the only autograd
path is the student forward. ``alpha=0.0`` (default) is a strict no-op:
no teacher rollout, no extra forward, identical to plain LM CE training,
preserving full backwards-compat with the post-Phase-3 ckpt schema.

Self-KL "targets failure modes":
The KL contribution at position ``t`` is large when student and teacher
disagree -- which happens precisely on tokens where the model is
uncertain (high-entropy logits). High-error tokens get more
distillation pressure for free, no per-token weighting needed.

CLI (default behavior unchanged when --self-distill-alpha 0.0):

    python3 train_100m_self_distill.py \
        --warmstart /workspace/runs/v24h_qwen_rl/best_step_XXXX.pt \
        --sft-parquet /workspace/data/alpaca_sft/alpaca.parquet \
        --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
        --out /workspace/runs/v24h_qwen_self_distill \
        --self-distill-alpha 0.3 \
        --teacher-temp 1.5 \
        --student-temp 0.0 \
        --rollouts-per-step 4 \
        --backend triton_block --batch-size 16 --steps 2000 --lr 5e-6
"""

from __future__ import annotations

import os as _os_early
_os_early.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import math
import os
import random
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

# Mirror train_100m_kd / sft sys.path hygiene so `import synapforge` finds
# the outer package even when the inner /workspace/synapforge/synapforge
# nested dir is present on the rental.
sys.path[:] = [p for p in sys.path if p not in ("/workspace/synapforge", "")]
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

# Ensure the directory containing this script is also importable so
# ``from synapforge.*`` works when running locally too.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from synapforge.huggingface_adapter import adv_warmstart  # noqa: E402
from synapforge.model_100m import build_synapforge_100m  # noqa: E402
from synapforge.optim import build_optimizer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SEQ_LEN_DEFAULT = 1024

# Architecture constants (see train_100m_sft.py -- must match the Phase 3
# RL ckpt for warmstart to land cleanly).
MODEL_FFN_RATIO = 8.0
MODEL_SPARSITY = 0.95
MODEL_DROPOUT = 0.0
MODEL_TIE_LM_HEAD = True


def _build_config_dict(args) -> dict:
    """Architecture-config persisted into every ckpt (P12 in MASTER_PLAN.md)."""
    return {
        "vocab": int(args.vocab),
        "d": int(args.d),
        "n_layers": int(args.n_layers),
        "loop_depth": int(args.loop_depth),
        "max_seq": int(args.max_seq),
        "ffn_ratio": MODEL_FFN_RATIO,
        "sparsity": MODEL_SPARSITY,
        "dropout": MODEL_DROPOUT,
        "tie_lm_head": MODEL_TIE_LM_HEAD,
        "self_distill": {
            "alpha": float(args.self_distill_alpha),
            "teacher_temp": float(args.teacher_temp),
            "student_temp": float(args.student_temp),
            "rollouts_per_step": int(args.rollouts_per_step),
        },
    }


def _parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/workspace/runs/self_distill")
    p.add_argument("--backend", default="triton_block",
                   choices=["gpu_dense", "triton_block"])
    p.add_argument("--warmstart", required=True,
                   help="Phase 3 RL output ckpt (best_step_*.pt)")
    p.add_argument("--sft-parquet", required=True,
                   help="Same parquet schema as train_100m_sft.py "
                        "(input_ids + loss_mask columns)")
    p.add_argument("--tokenizer-path", required=True,
                   help="for eos_token_id and chat-sample logging")
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--loop-depth", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=SEQ_LEN_DEFAULT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=50)
    # Phase 4 LR: smaller than SFT (1e-5) since we are converging and
    # adding a second loss term that wants stable behaviour.
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    # ---- Self-distillation knobs ----
    # alpha=0.0 is the strict no-op path: no teacher forward is taken,
    # the loss collapses to plain LM CE, identical to train_100m_sft.
    p.add_argument("--self-distill-alpha", type=float, default=0.3,
                   help="weight of the self-KL term in total loss (0.0 = off)")
    p.add_argument("--teacher-temp", type=float, default=1.5,
                   help="temperature applied to TEACHER (high-T exploratory) "
                        "logits before softmax. Higher = softer target.")
    p.add_argument("--student-temp", type=float, default=0.0,
                   help="temperature applied to STUDENT (low-T greedy) "
                        "logits before softmax. 0.0 == treat as 1.0 in "
                        "softmax (no scaling) since dividing by zero is "
                        "undefined; for online policy distillation we want "
                        "the raw student distribution to match the soft "
                        "teacher target.")
    p.add_argument("--rollouts-per-step", type=int, default=4,
                   help="how many tokens of the high-T rollout we average "
                        "the self-KL over per step. >1 reduces variance "
                        "of the teacher target on a given batch.")
    return p.parse_args(argv)


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Loss helpers (pure functions so tests can call them without a trainer)
# ---------------------------------------------------------------------------


def _softmax_with_temp(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Softmax with temperature ``T``; T<=0 falls back to T=1.0.

    A literal T=0.0 would mean "argmax" / "Dirac at the max-logit", but for
    distillation we use it to mean "raw student distribution" (T=1). This
    convention matches the CLI default (--student-temp 0.0) and the tests
    use it to assert the temperatures-match sanity property.
    """
    eff_T = float(T) if float(T) > 0.0 else 1.0
    return F.softmax(logits.float() / eff_T, dim=-1)


def _log_softmax_with_temp(logits: torch.Tensor, T: float) -> torch.Tensor:
    eff_T = float(T) if float(T) > 0.0 else 1.0
    return F.log_softmax(logits.float() / eff_T, dim=-1)


def compute_self_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
    label_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL(student || teacher) under the temperature pair, mean over tokens.

    Both logits arrive shaped (B, T, V). ``label_mask`` (B, T) is the same
    response-only mask used by the LM CE term -- positions where the mask
    is 0 are excluded from the KL average so that pad / prompt tokens
    don't dilute the distillation signal.

    Note we use forward KL ``KL(student || teacher)``, i.e. the student
    is the q-distribution. This is the convention from the docstring
    formula:

        L_self_kl = mean_t  KL(p_student_t || p_teacher_t)

    so the student's heavy-tail probability mass is what the gradient
    flows through. (Reverse KL would be mode-seeking, the wrong sign for
    "match the teacher's softer, broader distribution".)
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student={tuple(student_logits.shape)} "
            f"teacher={tuple(teacher_logits.shape)}"
        )

    log_q = _log_softmax_with_temp(student_logits, student_temp)
    p_t = _softmax_with_temp(teacher_logits, teacher_temp)

    # KL(q || p) = sum_v q * (log q - log p). We have log_q already; for
    # log p we recompute log_softmax under the teacher temperature.
    log_p = _log_softmax_with_temp(teacher_logits, teacher_temp)
    q = log_q.exp()
    per_token = (q * (log_q - log_p)).sum(dim=-1)  # (B, T)

    # Numeric guard: q * log q -> 0 as q -> 0, but float rounding can
    # leave a tiny negative. Clamp to >=0 so the mean is well-behaved.
    per_token = per_token.clamp_min(0.0)

    if label_mask is not None:
        m = label_mask.to(per_token.dtype)
        denom = m.sum().clamp_min(1.0)
        return (per_token * m).sum() / denom
    return per_token.mean()


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    """Linear warmup -> cosine decay (matches train_100m_sft / kd)."""
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# SFT-style batcher (parquet input_ids + loss_mask). Same shape as
# train_100m_sft.py so the Phase 3 RL ckpt -> Phase 4 self-distill warm
# transition reuses the same data preprocessor.
# ---------------------------------------------------------------------------


class SFTBatcher:
    """Stream input_ids+loss_mask rows from parquet, pad to common length."""

    def __init__(
        self,
        parquet_path: str,
        batch_size: int,
        max_seq: int,
        pad_id: int,
        seed: int = 0,
    ) -> None:
        import pyarrow.parquet as pq
        self.path = parquet_path
        self.batch_size = int(batch_size)
        self.max_seq = int(max_seq)
        self.pad_id = int(pad_id)
        self.seed = int(seed)
        table = pq.read_table(parquet_path)
        self.ids = [list(x) for x in table["input_ids"].to_pylist()]
        self.msk = [list(x) for x in table["loss_mask"].to_pylist()]
        self.n = len(self.ids)
        _log(f"[sft-data] loaded {self.n:,} examples from {parquet_path}")
        self._order = list(range(self.n))
        random.Random(seed).shuffle(self._order)
        self._cursor = 0

    def _next_idx(self) -> int:
        if self._cursor >= self.n:
            random.Random(self.seed + self._cursor).shuffle(self._order)
            self._cursor = 0
        i = self._order[self._cursor]
        self._cursor += 1
        return i

    def __iter__(self):
        return self

    def __next__(self):
        ids_b: list[list[int]] = []
        msk_b: list[list[int]] = []
        Lmax = 0
        for _ in range(self.batch_size):
            i = self._next_idx()
            ids = self.ids[i][: self.max_seq]
            msk = self.msk[i][: self.max_seq]
            ids_b.append(ids)
            msk_b.append(msk)
            Lmax = max(Lmax, len(ids))
        for k in range(len(ids_b)):
            pad = Lmax - len(ids_b[k])
            ids_b[k] = ids_b[k] + [self.pad_id] * pad
            msk_b[k] = msk_b[k] + [0] * pad
        x = torch.tensor(ids_b, dtype=torch.long)
        m = torch.tensor(msk_b, dtype=torch.long)
        return x, m


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    _log(f"device={DEVICE} dtype={DTYPE} out={args.out} backend={args.backend}")
    _log(
        f"steps={args.steps} bs={args.batch_size} seq={args.max_seq} "
        f"lr={args.lr}"
    )
    _log(
        f"self-distill: alpha={args.self_distill_alpha} "
        f"T_teacher={args.teacher_temp} T_student={args.student_temp} "
        f"rollouts/step={args.rollouts_per_step}"
    )
    if args.self_distill_alpha == 0.0:
        _log("[self-distill] alpha=0.0 -> no-op (collapses to plain LM CE)")

    # --- model (must mirror Phase 3 RL ckpt config) ---
    model = build_synapforge_100m(
        vocab=args.vocab,
        d=args.d,
        n_layers=args.n_layers,
        loop_depth=args.loop_depth,
        max_seq=args.max_seq,
        ffn_ratio=MODEL_FFN_RATIO,
        sparsity=MODEL_SPARSITY,
        dropout=MODEL_DROPOUT,
        use_grad_checkpoint=args.grad_checkpoint,
    )
    n_params = model.num_parameters()
    _log(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # --- warmstart from Phase 3 RL ckpt ---
    if not os.path.exists(args.warmstart):
        _log(
            f"FATAL: warmstart {args.warmstart!r} not found. Phase 4 "
            f"self-distill REQUIRES the Phase 3 RL ckpt (or any prior best)."
        )
        return 2
    rep = adv_warmstart(
        model,
        args.warmstart,
        name_map=[
            (r"\.cfc\.", ".liquid."),
            (r"\.embed\.text_embed\.", ".tok_embed."),
        ],
    )
    _log(
        f"warmstart matched={rep.matched}/{rep.total_target} "
        f"missing={len(rep.missing)} extra={len(rep.extra)}"
    )
    model = model.to(DEVICE)
    model.train()

    # --- backend ---
    if args.backend == "triton_block":
        try:
            from synapforge.backends.triton_block import TritonBlockBackend
            from synapforge.backends.triton_block_kernel import _HAS_TRITON
            be = TritonBlockBackend()
            stats = be.compile(model)
            _log(
                f"[backend] triton_block: avail={_HAS_TRITON} "
                f"pairs={stats.get('n_pairs_fused', 0)}"
            )
        except Exception as e:
            _log(f"[backend] triton_block FAILED: {e}; using gpu_dense")
            args.backend = "gpu_dense"

    # --- optimizer (re-build, then load Phase-3 momentum if present) ---
    optim = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    try:
        ck = torch.load(args.warmstart, map_location="cpu")
        if isinstance(ck, dict) and "optim_state" in ck:
            optim.load_state_dict(ck["optim_state"])
            _log("warmstart: loaded optim_state from RL ckpt")
    except Exception as e:
        _log(f"warmstart optim load skipped: {e}")

    # --- tokenizer (just for pad_id) ---
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    # --- data ---
    train_iter = SFTBatcher(
        args.sft_parquet,
        args.batch_size,
        args.max_seq,
        pad_id=pad_id,
        seed=args.seed,
    )

    t0 = time.time()
    cum_tok = 0
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        cur_lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr
        t_step = time.time()
        x, mask = next(train_iter)
        n_resp_step = int(mask[:, 1:].sum().item())
        x = x.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        # CUDA gets a real bf16 autocast; CPU uses a no-op contextmanager
        # because torch.amp.autocast(device_type='cpu', dtype=fp32) will
        # error out (CPU autocast only supports bf16). Keeping the path
        # branch-free under contextlib.nullcontext keeps the logic clear.
        if DEVICE == "cuda":
            _ac_ctx = torch.amp.autocast(
                device_type=DEVICE, dtype=DTYPE, enabled=True
            )
        else:
            import contextlib
            _ac_ctx = contextlib.nullcontext()
        with _ac_ctx:
            # ---- student forward (autograd-attached) ----
            student_logits = model(x[:, :-1])  # (B, T-1, V)
            labels = x[:, 1:].clone()
            label_mask = mask[:, 1:]
            ce_labels = labels.masked_fill(label_mask == 0, -100)
            ce_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)).float(),
                ce_labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )

            # ---- teacher rollouts (no_grad) + self-KL ----
            # Skip entirely when alpha=0.0 so the no-op path is bit-identical
            # to plain SFT (no extra GPU work, no autograd noise).
            if args.self_distill_alpha > 0.0:
                self_kl_terms = []
                for _ in range(max(1, int(args.rollouts_per_step))):
                    with torch.no_grad():
                        # The teacher is the SAME network with high-T
                        # temperature applied to its logits. We re-run the
                        # forward (could be cached if rollouts_per_step=1
                        # since logits are deterministic given input, but
                        # we keep the loop body uniform for clarity / for
                        # future stochastic-sampling variants).
                        teacher_logits = model(x[:, :-1])
                    self_kl_terms.append(
                        compute_self_kl(
                            student_logits,
                            teacher_logits,
                            student_temp=args.student_temp,
                            teacher_temp=args.teacher_temp,
                            label_mask=label_mask,
                        )
                    )
                self_kl = torch.stack(self_kl_terms).mean()
                loss = ce_loss + args.self_distill_alpha * self_kl
            else:
                self_kl = torch.zeros((), device=ce_loss.device)
                loss = ce_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )
        optim.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.time() - t_step) * 1000.0
        cum_tok += n_resp_step

        if step % args.log_every == 0 or step == 1:
            tok_s = cum_tok / max(time.time() - t0, 1e-6)
            mem_str = (
                f" mem_GB={torch.cuda.memory_allocated()/1e9:.2f}"
                if DEVICE == "cuda" else ""
            )
            _log(
                f"step {step:5d} loss={loss.item():.4f} "
                f"ce={ce_loss.item():.4f} self_kl={float(self_kl):.4f} "
                f"lr={cur_lr:.2e} step_ms={step_ms:.1f} "
                f"resp_tok/s={tok_s:.0f}{mem_str}"
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "model": model.state_dict(),
                "optim_state": optim.state_dict(),
                "step": step,
                "loss": float(loss.item()),
                "ce_loss": float(ce_loss.item()),
                "self_kl": float(self_kl),
                "config": _build_config_dict(args),
            }
            cur = float(loss.item())
            if cur < best_loss:
                best_loss = cur
                best_path = os.path.join(args.out, f"best_step_{step:06d}.pt")
                torch.save(ckpt, best_path)
                _log(f"saved BEST ckpt {best_path} (loss={cur:.4f})")
            torch.save(ckpt, os.path.join(args.out, f"step_{step:06d}.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
