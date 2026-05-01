"""HellaSwag commonsense continuation accuracy.

Dataset: 4-choice next-sentence selection (Zellers et al. 2019). 10042 dev examples.

Standard scoring uses the lm-eval-harness rule:
    For each candidate continuation c_i, score = sum_t log P(c_i_t | ctx, c_i_<t)
    pick argmax_i score (length-normalised).

We compute the per-token log-probability of each candidate by a single forward
pass over `ctx + c_i` and reading the logits at the boundary positions.

CLI:
    python -m synapforge.bench.hellaswag --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \\
        --out hellaswag.json [--n 200]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------- dataset


def _load_problems(local_path: Optional[str], n: Optional[int]) -> List[Dict[str, Any]]:
    if local_path and Path(local_path).exists():
        probs = []
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    probs.append(json.loads(line))
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "HellaSwag needs either --data PATH or `datasets`."
            ) from e
        ds = load_dataset("hellaswag", split="validation")
        probs = list(ds)
    if n is not None:
        probs = probs[: int(n)]
    return probs


def build_context(ex: Dict[str, Any]) -> str:
    """HellaSwag standard preprocessing: activity_label + ctx_a + ctx_b."""
    label = ex.get("activity_label") or ""
    ctx_a = ex.get("ctx_a") or ""
    ctx_b = ex.get("ctx_b") or ""
    if label:
        return f"{label}: {ctx_a} {ctx_b.capitalize() if ctx_b else ''}".strip()
    return f"{ctx_a} {ctx_b}".strip()


# ---------------------------------------------------------------- scoring

def _score_candidate(model, tok, context: str, ending: str) -> float:
    """Return mean per-token log P(ending | context). Length-normalised."""
    import torch
    import torch.nn.functional as F
    from synapforge.eval.generate import _forward_logits

    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    ctx_ids = tok.encode(context, add_special_tokens=False) or [
        tok.eos_token_id or 0
    ]
    end_ids = tok.encode(" " + ending if not ending.startswith(" ") else ending,
                         add_special_tokens=False)
    if not end_ids:
        return float("-inf")
    full = ctx_ids + end_ids
    max_seq = getattr(model, "max_seq", None)
    if max_seq is not None and len(full) > max_seq:
        # keep the tail — we always need the ending fully visible.
        full = full[-max_seq:]
        ctx_len_eff = max(len(full) - len(end_ids), 1)
    else:
        ctx_len_eff = len(ctx_ids)

    inp = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = _forward_logits(model, inp)
    # logits[0, t] predicts token full[t+1].
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    end_log_p = 0.0
    n = 0
    for t in range(ctx_len_eff - 1, len(full) - 1):
        target = full[t + 1]
        end_log_p += float(log_probs[t, target].item())
        n += 1
    return end_log_p / max(n, 1)


def _pick_continuation(model, tok, ex: Dict[str, Any]) -> int:
    ctx = build_context(ex)
    endings = ex.get("endings") or []
    scores = []
    for e in endings:
        try:
            scores.append(_score_candidate(model, tok, ctx, e))
        except Exception:
            scores.append(float("-inf"))
    if not scores:
        return 0
    return int(max(range(len(scores)), key=scores.__getitem__))


# ---------------------------------------------------------------- main entry

def run_bench(ckpt: Optional[str] = None, tokenizer: Optional[str] = None,
              model: Any = None, tok: Any = None,
              data_path: Optional[str] = None, n: Optional[int] = None,
              out: Optional[str] = None, **_kw) -> Dict[str, Any]:
    if model is None and ckpt:
        from synapforge.eval.generate import load_synapforge
        model, _ = load_synapforge(ckpt)
    if tok is None and tokenizer:
        from synapforge.eval.generate import load_tokenizer
        tok = load_tokenizer(tokenizer)
    if model is None or tok is None:
        return {"name": "hellaswag", "error": "model/tokenizer required"}

    probs = _load_problems(data_path, n)
    n_total = len(probs)
    n_correct = 0
    t0 = time.time()
    for i, p in enumerate(probs):
        gold = int(p.get("label", -1))
        pick = _pick_continuation(model, tok, p)
        if pick == gold:
            n_correct += 1
        if (i + 1) % 50 == 0:
            print(f"[hellaswag] {i+1}/{n_total}  acc={n_correct}/{i+1}", flush=True)

    summary = {
        "name":      "hellaswag",
        "acc":       n_correct / max(n_total, 1),
        "n_correct": n_correct,
        "n_total":   n_total,
        "wall_s":    time.time() - t0,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    summary = run_bench(
        ckpt=args.ckpt, tokenizer=args.tokenizer,
        data_path=args.data, n=args.n, out=args.out,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
