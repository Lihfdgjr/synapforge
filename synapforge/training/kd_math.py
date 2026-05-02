"""synapforge.training.kd_math -- KD distillation loss math (chunked + top-K).

Lifted verbatim from ``train_100m_kd.py`` so :class:`KDTrainer` and the
legacy ``train_100m_kd.py`` produce IDENTICAL loss values on the same
inputs (the bit-exactness contract documented in core_trainer.py).

Two paths:
  * **Top-K teacher softmax** (default, ``topk > 0``). Materialises
    only ``(B, T, K) * 4`` bytes -- ~70x less memory than the full
    vocab at K=2048, V=151936. Mathematically: take top-K teacher
    logits, renormalise teacher softmax to those K positions, gather
    student logits at the same indices, run log_softmax over those K,
    compute KL. BitNet / DistilBERT / SmolLM all use this trick.
  * **Full-vocab chunked** (``topk == 0``). Walks the batch in chunks
    so the ``(chunk*T, V, fp32)`` intermediate fits in ``headroom``
    fraction of free VRAM. Auto-tunes chunk size via
    :func:`kd_chunk_size`, or use the explicit ``chunk_override``
    argument.

Both paths return a Hinton-scaled scalar: ``KL * T^2``, averaged over
all token positions.

Vocab-mismatch handling: when teacher.vocab > student.vocab (e.g.
student=Qwen 151643, teacher=GPT-2 50257), the larger vocab is sliced
to ``[:V_min]``. Caller is responsible for ensuring the vocabularies
share a common prefix tokenization.

This module is a strict subset of train_100m_kd.py: NOTHING here
imports synapforge proper, so unit tests run on a stock dev laptop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Module-level guards so the auto-tune banner only prints once per process.
_KD_CHUNK_BANNER_PRINTED = False
_KD_TOPK_BANNER_PRINTED = False


def kd_chunk_size(
    batch_size: int,
    seq_len: int,
    vocab: int,
    headroom: float = 0.5,
) -> int:
    """Auto-pick KD chunk size from current GPU free VRAM.

    MASTER_PLAN.md §6 P2 + P13: the prior fixed ``chunk = bs // 4``
    OOM'd at bs=128 / vocab=151936 on A800-80GB because the
    ``(B*T, V, fp32)`` intermediate for ``logsumexp`` + KL is ~18.55
    GiB. This sizes the chunk so the intermediate fits in
    ``headroom * free_vram``.

    Each chunk materializes ``(chunk * seq_len, vocab)`` in fp32, so
    per-row cost is ``seq_len * vocab * 4`` bytes. Floor at 1, cap at
    ``batch_size``.
    """
    if not torch.cuda.is_available():
        return max(1, batch_size // 4)
    free_b, _ = torch.cuda.mem_get_info()
    budget_b = int(free_b * headroom)
    per_row_b = max(1, int(seq_len) * int(vocab) * 4)
    chunk = max(1, min(int(batch_size), budget_b // per_row_b))
    return chunk


def kd_topk_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float,
    k: int,
) -> torch.Tensor:
    """Memory-bounded top-K teacher softmax KD (Hinton, scaled by T^2).

    Edge cases:
      * ``k >= V`` -- equivalent to full-vocab KL (test asserts equality).
      * ``k == 1`` -- finite (log_softmax over a single element is 0).
    """
    V = student_logits.size(-1)
    k = max(1, min(int(k), V))
    top_vals, top_idx = teacher_logits.detach().topk(k, dim=-1)
    top_p = F.softmax(top_vals.float() / T, dim=-1)
    s_top = student_logits.gather(-1, top_idx)
    s_logp = F.log_softmax(s_top.float() / T, dim=-1)
    kl = F.kl_div(s_logp, top_p, reduction="sum")
    n_tokens = top_p.size(0) * top_p.size(1)
    n_tokens = max(n_tokens, 1)
    return (kl / n_tokens) * (T * T)


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 4.0,
    chunk_override: int = 0,
    topk: int = 2048,
) -> torch.Tensor:
    """KL(student_logp || teacher_p), memory-bounded for large vocab.

    Returns a scalar token-mean (over batch * time) scaled by T**2 (Hinton).
    BIT-EXACT match to ``train_100m_kd._kd_loss`` -- this is the same
    function lifted into a reusable module.

    Two paths:
      * ``topk > 0`` (default 2048): top-K teacher softmax. Memory
        bounded at ``(B, T, K) * 4`` bytes regardless of vocab.
      * ``topk == 0``: full-vocab chunked softmax (legacy path).
        ``chunk_override``: if > 0, use that chunk size verbatim.
        Else auto-tune from current GPU free VRAM.
    """
    global _KD_CHUNK_BANNER_PRINTED, _KD_TOPK_BANNER_PRINTED
    V = student_logits.size(-1)
    V_t = teacher_logits.size(-1)
    # Vocabulary mismatch handling: truncate to common prefix.
    if V_t > V:
        teacher_logits = teacher_logits[..., :V]
    elif V > V_t:
        student_logits = student_logits[..., :V_t]
    vocab = student_logits.size(-1)

    # ---- top-K path (default) ----
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
                print(
                    f"[kd] using top-{k_eff} softmax "
                    f"(saves {savings_gb:.2f} GB vs full-vocab V={vocab})",
                    flush=True,
                )
            except Exception:
                pass
            _KD_TOPK_BANNER_PRINTED = True
        return kd_topk_loss(student_logits, teacher_logits, T, topk)

    # ---- legacy full-vocab chunked path (topk == 0) ----
    bs = student_logits.size(0)
    seq = student_logits.size(1) if student_logits.dim() >= 3 else 1
    if chunk_override and chunk_override > 0:
        chunk = max(1, min(int(chunk_override), bs))
    else:
        chunk = kd_chunk_size(bs, seq, vocab)
    if not _KD_CHUNK_BANNER_PRINTED:
        try:
            if torch.cuda.is_available():
                free_b, _tot_b = torch.cuda.mem_get_info()
                free_gb = free_b / (1024 ** 3)
                print(
                    f"[kd] chunk={chunk} (free={free_gb:.1f}GB, "
                    f"vocab={vocab})",
                    flush=True,
                )
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
        total = total + F.kl_div(slp, tp, reduction="sum")
        n_tokens += sl.size(0) * sl.size(1)
    n_tokens = max(n_tokens, 1)
    return (total / n_tokens) * (T * T)


def reset_banners() -> None:
    """Test helper: reset the once-per-process banner guards."""
    global _KD_CHUNK_BANNER_PRINTED, _KD_TOPK_BANNER_PRINTED
    _KD_CHUNK_BANNER_PRINTED = False
    _KD_TOPK_BANNER_PRINTED = False


__all__ = [
    "kd_chunk_size",
    "kd_loss",
    "kd_topk_loss",
    "reset_banners",
]
