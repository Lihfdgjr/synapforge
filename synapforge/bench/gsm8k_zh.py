"""GSM8K-Chinese (translated grade-school math) exact-match accuracy.

Same 1319 grade-school word problems as GSM8K-EN, machine-translated to
Chinese (e.g. `meta-math/MetaMathQA-Chinese` or `WizardLMTeam/GSM8K_zh`).
Score: greedy generate up to 256 tokens, regex-extract the final integer
answer. We allow either Chinese punctuation around the answer or the
canonical "#### N" suffix (which translators usually preserve).

Chinese-aware quirks vs English GSM8K:
    * Numbers may be written with full-width digits ("１２３") or in Chinese
      number words ("一百二十三"). We normalise full-width to ASCII; Chinese
      number words are only recognised for very small magnitudes.
    * Final-answer phrase often is "答案是 N" / "因此答案为 N" / "所以 N".

CLI:
    python -m synapforge.bench.gsm8k_zh --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \
        --max-new 256 --out gsm8k_zh.json [--n 50]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

ZERO_SHOT_PROMPT = (
    "请逐步分析并解答下列数学题，最后用 “#### <整数>” 给出最终答案。\n\n"
    "题目：{q}\n"
    "解答："
)

# Patterns
_ANS_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_LAST_INT = re.compile(r"(?<![A-Za-z_])(-?\d+(?:\.\d+)?)(?![A-Za-z_])")
# Chinese cue phrases that often precede the answer.
_ZH_CUE = re.compile(
    r"(?:答案[是为：:]?|因此[，,]?|所以[，,]?|最终[，,]?|"
    r"故答案[为是：:]?|总共[有为是]?|一共[是有为]?)\s*(-?\d+(?:\.\d+)?)"
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
                "GSM8K-zh needs either --data PATH or `datasets`."
            ) from e
        # Best-effort across the most common Chinese mirrors.
        ds = None
        for cand in ("WizardLMTeam/GSM8K_zh", "meta-math/MetaMathQA-Chinese",
                     "Iker/gsm8k-zh"):
            try:
                ds = load_dataset(cand, split="test")
                break
            except Exception:
                continue
        if ds is None:
            raise RuntimeError("no GSM8K-zh source could be loaded")
        probs = list(ds)
    if n is not None:
        probs = probs[: int(n)]
    return probs


# ---------------------------------------------------------------- answer parsing


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


def _to_ascii_digits(text: str) -> str:
    """Full-width digits / Chinese punctuation → ASCII (NFKC)."""
    return unicodedata.normalize("NFKC", text or "").replace(",", "")


def parse_answer(text: str) -> Optional[str]:
    text = _to_ascii_digits(text)
    m = _ANS_RE.search(text)
    if m:
        return _normalize(m.group(1))
    m = _ZH_CUE.search(text)
    if m:
        return _normalize(m.group(1))
    nums = _LAST_INT.findall(text)
    return _normalize(nums[-1]) if nums else None


def parse_gold(answer_field: str) -> Optional[str]:
    af = _to_ascii_digits(answer_field)
    m = _ANS_RE.search(af)
    if m:
        return _normalize(m.group(1))
    m = _ZH_CUE.search(af)
    if m:
        return _normalize(m.group(1))
    nums = _LAST_INT.findall(af)
    return _normalize(nums[-1]) if nums else None


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
        # Schemas vary across mirrors: try a few common keys.
        q = (p.get("question_zh") or p.get("question_zh-cn") or
             p.get("question") or p.get("query") or p.get("problem") or "")
        gold_field = (p.get("answer_zh") or p.get("answer") or
                      p.get("response") or p.get("gold") or "")
        gold = parse_gold(gold_field)
        prompt = ZERO_SHOT_PROMPT.format(q=q)
        comp = _gen_completion(model, tok, prompt, max_new=max_new)
        pred = parse_answer(comp)
        ok = (pred is not None and gold is not None and pred == gold)
        n_correct += int(ok)
        samples.append({"i": i, "gold": gold, "pred": pred, "pass": ok})
        if (i + 1) % 25 == 0:
            print(f"[gsm8k_zh] {i+1}/{n_total}  acc={n_correct}/{i+1}", flush=True)

    summary = {
        "name":       "gsm8k_zh",
        "acc":        n_correct / max(n_total, 1),
        "n_correct":  n_correct,
        "n_total":    n_total,
        "wall_s":     time.time() - t0,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": samples}, f,
                      indent=2, ensure_ascii=False)
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
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
