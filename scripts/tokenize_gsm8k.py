"""tokenize_gsm8k.py — fetch + tokenize GSM8K math chain-of-thought (T3.5).

Pulls the GSM8K grade-school math word problem set from HuggingFace
``datasets`` and tokenizes 1000 chain-of-thought examples with the same
Qwen 2.5 tokenizer the Synap-1 100M trainer uses. Output is one parquet
ready for the **phase-4 RL verifier** stage (`--rl-grpo`) where the
model rolls out chain-of-thought solutions and a verifier compares the
extracted final answer against ``final_answer``.

Dataset
-------
* ``openai/gsm8k`` config ``"main"``  (7,473 train + 1,319 test examples)
  Each example has columns ``question`` and ``answer``. The ``answer``
  field is *both* the chain-of-thought reasoning AND a final boxed
  numeric answer marked by the canonical ``"#### N"`` suffix.

We split each ``answer`` into ``(cot, final_answer)`` along that marker.

Output schema (parquet)
-----------------------
    question      str        problem statement
    cot           str        chain-of-thought (everything before ``####``)
    answer        str        full original answer field (kept for audit)
    input_ids     list<int>  Qwen tokenization of "Q: <question>\\nA: <cot>"
    final_answer  int        extracted ground-truth number (negative-aware)

Companion ``.manifest.json`` records row count + tokenizer source.

Usage
-----
    python scripts/tokenize_gsm8k.py \\
        --out /workspace/data/math/gsm8k_qwen.parquet \\
        --n 1000

    python scripts/tokenize_gsm8k.py --smoke  # 10 hand-rolled rows, no net
    python scripts/tokenize_gsm8k.py --help

Determinism
-----------
The HuggingFace dataset is deterministically ordered, so taking the first
N examples gives a stable subset across machines. ``--seed`` is currently
unused but reserved for future shuffled subsamples.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


# Canonical GSM8K final-answer marker. Each example ends with "#### NUM"
# where NUM is an integer (possibly negative, possibly with commas as
# thousands separators).
_FINAL_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+)\s*$")


def extract_final_answer(answer: str) -> Optional[int]:
    """Pull the integer following ``####`` from a GSM8K answer string.

    Returns ``None`` if no marker found or the captured token isn't an
    integer (rare malformed entries). Strips comma thousand-separators
    so e.g. ``"#### 1,234"`` -> ``1234``.
    """
    if not answer:
        return None
    # Search anywhere — some entries have trailing whitespace / newlines.
    m = _FINAL_ANSWER_RE.search(answer.rstrip())
    if not m:
        return None
    raw = m.group(1).replace(",", "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


def split_cot_and_answer(answer: str) -> str:
    """Return chain-of-thought portion of an answer string (text before ####).

    Falls back to the full string if no marker found, so consumers always
    get a non-empty CoT field.
    """
    idx = answer.rfind("####")
    if idx < 0:
        return answer.strip()
    return answer[:idx].rstrip()


def _format_prompt(question: str, cot: str) -> str:
    """Compose the tokenizer input: standard Q/A formatting."""
    return f"Q: {question.strip()}\nA: {cot.strip()}"


def _load_tokenizer(name: str = "Qwen/Qwen2.5-0.5B"):  # pragma: no cover -- network
    """Lazy import; returns an HF AutoTokenizer or raises."""
    from transformers import AutoTokenizer
    candidates = [
        "/workspace/teachers/qwen2.5-0.5b",
        name,
    ]
    last = None
    for p in candidates:
        try:
            return AutoTokenizer.from_pretrained(p, trust_remote_code=True)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"could not load Qwen tokenizer: {last!r}")


def _load_gsm8k(split: str = "train") -> Iterable[dict]:  # pragma: no cover -- network
    """Yield raw GSM8K records ({question, answer}) from HF datasets."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    for ex in ds:
        yield {"question": str(ex["question"]), "answer": str(ex["answer"])}


def _build_rows(
    raw_records: Iterable[dict],
    tokenizer,
    n: int,
) -> List[dict]:
    """Apply CoT split + tokenize + final-answer extract to ``n`` records."""
    rows: List[dict] = []
    for rec in raw_records:
        if len(rows) >= n:
            break
        q = rec.get("question") or ""
        a = rec.get("answer") or ""
        cot = split_cot_and_answer(a)
        final = extract_final_answer(a)
        if final is None:
            # Skip malformed examples — they'd break the verifier anyway.
            continue
        prompt = _format_prompt(q, cot)
        if tokenizer is None:
            ids: List[int] = []
        else:
            ids = list(map(int, tokenizer.encode(prompt, add_special_tokens=False)))
        rows.append(
            dict(
                question=q,
                cot=cot,
                answer=a,
                input_ids=ids,
                final_answer=int(final),
            )
        )
    return rows


def write_parquet(path: str, rows: Sequence[dict],
                  tokenizer_name: str = "Qwen/Qwen2.5-0.5B") -> int:
    if not _HAVE_ARROW:
        raise RuntimeError("pyarrow required to write parquet")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "question": [r["question"] for r in rows],
        "cot": [r["cot"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "input_ids": [r["input_ids"] for r in rows],
        "final_answer": [int(r["final_answer"]) for r in rows],
    }
    pq.write_table(pa.table(cols), path, compression="zstd")
    with open(str(path) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "gsm8k_tokenized",
                "rows": len(rows),
                "tokenizer": tokenizer_name,
                "schema": {
                    "question": "string",
                    "cot": "string",
                    "answer": "string",
                    "input_ids": "list<int32>",
                    "final_answer": "int64",
                },
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
        )
    return len(rows)


# --- smoke-mode stand-in dataset (no network) -------------------------------
SMOKE_RECORDS = [
    {
        "question": "Janet has 3 apples. She buys 4 more. How many apples does she have?",
        "answer": "She starts with 3 and buys 4, so 3+4=7 apples.\n#### 7",
    },
    {
        "question": "A train travels 60 miles in 2 hours. What is its speed?",
        "answer": "Speed = distance/time = 60/2 = 30 mph.\n#### 30",
    },
    {
        "question": "Tom has $25 and spends $7 on lunch. How much is left?",
        "answer": "25 - 7 = 18 dollars remaining.\n#### 18",
    },
    {
        "question": "If 5 pencils cost $1.50, what is the cost of one pencil in cents?",
        "answer": "150 cents / 5 = 30 cents each.\n#### 30",
    },
    {
        "question": "Nina has 12 cookies and gives away 5. How many remain?",
        "answer": "12 - 5 = 7 cookies left.\n#### 7",
    },
]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="/workspace/data/math/gsm8k_qwen.parquet")
    ap.add_argument("--n", type=int, default=1000,
                    help="How many examples to keep (after final-answer filter)")
    ap.add_argument("--split", default="train",
                    choices=("train", "test"),
                    help="GSM8K split to draw from")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seed", type=int, default=42,
                    help="Reserved for future shuffled subsamples")
    ap.add_argument("--smoke", action="store_true",
                    help="Use 5 hand-rolled records (no network), n<=10")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        # No network, no real tokenizer (input_ids stay empty).
        rows = _build_rows(iter(SMOKE_RECORDS), tokenizer=None,
                           n=min(args.n, len(SMOKE_RECORDS)))
        n = write_parquet(args.out, rows, tokenizer_name="smoke-no-tokenizer")
        print(f"[gsm8k] smoke wrote {n} rows -> {args.out}", flush=True)
        return 0

    tokenizer = _load_tokenizer(args.tokenizer)  # pragma: no cover -- network
    raw = _load_gsm8k(split=args.split)         # pragma: no cover -- network
    rows = _build_rows(raw, tokenizer=tokenizer, n=args.n)
    if not rows:
        print("[gsm8k] no rows produced", file=sys.stderr)
        return 1
    n = write_parquet(args.out, rows, tokenizer_name=args.tokenizer)
    print(f"[gsm8k] wrote {n} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
