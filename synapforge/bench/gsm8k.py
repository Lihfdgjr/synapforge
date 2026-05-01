"""GSM8K (grade-school math) exact-match accuracy.

Dataset: 1319 multi-step grade-school word problems with chain-of-thought
solutions ending in "#### N" where N is the integer answer (Cobbe et al. 2021).

Score: greedy generate up to 256 tokens, regex-extract the last integer that
appears after `####` (or as a final number), exact-match vs ground truth.

CLI:
    python -m synapforge.bench.gsm8k --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \\
        --max-new 256 --out gsm8k.json [--n 50]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Standard 8-shot CoT prompt would be ~1.5K tokens. Tiny LMs blow past their
# context window if we use it. Default to 0-shot with a short instruction; let
# users opt into few-shot via --shots when prompt budget allows.
ZERO_SHOT_PROMPT = (
    "Solve the following problem step by step. End your reasoning with "
    "'#### <integer>' where <integer> is the final numeric answer.\n\n"
    "Problem: {q}\n"
    "Answer:"
)


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
                "GSM8K needs either --data PATH or `datasets`."
            ) from e
        ds = load_dataset("gsm8k", "main", split="test")
        probs = list(ds)
    if n is not None:
        probs = probs[: int(n)]
    return probs


# ---------------------------------------------------------------- answer parsing

_ANS_RE   = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_LAST_INT = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse_answer(text: str) -> Optional[str]:
    """Pull the model's final numeric answer.

    Prefer `#### N` syntax; otherwise fall back to the last bare number in the
    completion (which is what Llama2 / Mistral usually emit).
    """
    m = _ANS_RE.search(text)
    if m:
        return _normalize(m.group(1))
    nums = _LAST_INT.findall(text.replace(",", ""))
    return _normalize(nums[-1]) if nums else None


def parse_gold(answer_field: str) -> Optional[str]:
    """GSM8K's `answer` field looks like '...rationale...\\n#### 18'."""
    m = _ANS_RE.search(answer_field or "")
    if m:
        return _normalize(m.group(1))
    nums = _LAST_INT.findall((answer_field or "").replace(",", ""))
    return _normalize(nums[-1]) if nums else None


def _normalize(s: str) -> str:
    s = s.strip()
    if s.endswith(".0"):
        s = s[:-2]
    if "." in s:
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return str(f)
        except ValueError:
            return s
    return s


# ---------------------------------------------------------------- generation

def _gen_completion(model, tok, prompt: str, max_new: int = 256) -> str:
    if model is None or tok is None:
        return ""
    from synapforge.eval.generate import generate
    out = generate(
        model, tok, prompt, max_new=max_new,
        top_k=1, top_p=1.0, temperature=0.0,
    )
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out


# ---------------------------------------------------------------- main entry

def run_bench(ckpt: Optional[str] = None, tokenizer: Optional[str] = None,
              model: Any = None, tok: Any = None,
              data_path: Optional[str] = None, n: Optional[int] = None,
              max_new: int = 256,
              out: Optional[str] = None, **_kw) -> Dict[str, Any]:
    if model is None and ckpt:
        from synapforge.eval.generate import load_synapforge
        model, _ = load_synapforge(ckpt)
    if tok is None and tokenizer:
        from synapforge.eval.generate import load_tokenizer
        tok = load_tokenizer(tokenizer)

    probs = _load_problems(data_path, n)
    n_total = len(probs)
    n_correct = 0
    samples: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, p in enumerate(probs):
        q = p.get("question") or p.get("problem") or ""
        gold = parse_gold(p.get("answer") or p.get("gold") or "")
        prompt = ZERO_SHOT_PROMPT.format(q=q)
        comp = _gen_completion(model, tok, prompt, max_new=max_new)
        pred = parse_answer(comp)
        ok = (pred is not None and gold is not None and pred == gold)
        n_correct += int(ok)
        samples.append({"i": i, "gold": gold, "pred": pred, "pass": ok})
        if (i + 1) % 25 == 0:
            print(f"[gsm8k] {i+1}/{n_total}  acc={n_correct}/{i+1}", flush=True)

    summary = {
        "name":       "gsm8k",
        "acc":        n_correct / max(n_total, 1),
        "n_correct":  n_correct,
        "n_total":    n_total,
        "wall_s":     time.time() - t0,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": samples}, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    summary = run_bench(
        ckpt=args.ckpt, tokenizer=args.tokenizer,
        data_path=args.data, n=args.n, max_new=args.max_new, out=args.out,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
