"""LAMBADA last-word prediction accuracy.

Dataset: 5153 dev passages where the model must predict the final word given the
preceding context (Paperno et al. 2016). Word boundaries are non-trivial — the
target word may tokenize into multiple BPE pieces.

Score: greedy-decode token-by-token until the model produces the full target
word. Match if the decoded suffix.strip() == target.strip().

CLI:
    python -m synapforge.bench.lambada --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \\
        --out lambada.json [--n 200]
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
                "LAMBADA needs either --data PATH or `datasets`."
            ) from e
        # 'lambada' is the standard split; openai-released variant is 'lambada_openai'.
        try:
            ds = load_dataset("lambada", split="test")
        except Exception:
            ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        probs = list(ds)
    if n is not None:
        probs = probs[: int(n)]
    return probs


def split_passage(text: str) -> tuple[str, str]:
    """Split text into (context, target_last_word). Strips trailing whitespace."""
    text = (text or "").rstrip()
    if not text:
        return "", ""
    # The target is the last whitespace-delimited token.
    parts = text.rsplit(maxsplit=1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[1]


# ---------------------------------------------------------------- prediction

def _predict_last_word(model, tok, context: str, target: str,
                       max_new: int = 8) -> str:
    """Greedy-generate up to len(target_tokens)+slack tokens, return decoded suffix."""
    import torch
    import torch.nn.functional as F
    from synapforge.eval.generate import _forward_logits

    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    # Always include a leading space when feeding context — LAMBADA targets are
    # space-prefixed in BPE space.
    ctx_text = context if context.endswith(" ") else context + " "
    ids = tok.encode(ctx_text, add_special_tokens=False)
    if not ids:
        ids = [tok.eos_token_id or 0]
    target_ids = tok.encode(" " + target if not target.startswith(" ") else target,
                            add_special_tokens=False)
    n_target = max(len(target_ids), 1)
    budget = min(max_new, n_target + 2)

    out_ids: List[int] = []
    work = list(ids)
    max_seq = getattr(model, "max_seq", None)
    eos_id = getattr(tok, "eos_token_id", None)
    for _ in range(budget):
        ctx = work
        if max_seq is not None and len(ctx) > max_seq:
            ctx = ctx[-max_seq:]
        inp = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = _forward_logits(model, inp)
        nxt = int(logits[0, -1].argmax().item())
        # Stop on EOS first (decoding the EOS token can be empty or noise).
        if eos_id is not None and nxt == eos_id and out_ids:
            break
        out_ids.append(nxt)
        work.append(nxt)
        # Stop after we've emitted at least the expected number of target
        # tokens AND the most recent piece ends with a word boundary.
        try:
            piece = tok.decode([nxt])
        except Exception:
            piece = ""
        # Don't bail on the very first token — leading-space pieces ("hello")
        # don't *end* with space anyway, so this rarely fires too early.
        if (len(out_ids) >= n_target
                and piece
                and piece.endswith(("\n", " ", ".", "!", "?"))):
            break
    try:
        return tok.decode(out_ids).strip()
    except Exception:
        return ""


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
        return {"name": "lambada", "error": "model/tokenizer required"}

    probs = _load_problems(data_path, n)
    n_total = len(probs)
    n_correct = 0
    t0 = time.time()
    for i, p in enumerate(probs):
        text = p.get("text") or p.get("passage") or ""
        ctx, target = split_passage(text)
        if not target:
            continue
        # Strip trailing punctuation off the gold target — many sources keep it.
        gold = target.strip().strip(".,!?;:'\"")
        pred = _predict_last_word(model, tok, ctx, gold)
        # Strip trailing punctuation from pred too.
        pred_clean = pred.split()[0].strip(".,!?;:'\"") if pred.strip() else ""
        ok = (pred_clean.lower() == gold.lower())
        n_correct += int(ok)
        if (i + 1) % 100 == 0:
            print(f"[lambada] {i+1}/{n_total}  acc={n_correct}/{i+1}", flush=True)

    summary = {
        "name":      "lambada",
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
