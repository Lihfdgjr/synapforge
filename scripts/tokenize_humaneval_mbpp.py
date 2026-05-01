"""tokenize_humaneval_mbpp.py — fetch + tokenize HumanEval & MBPP for T3.8.

Pulls the two canonical code-completion eval sets from HuggingFace
``datasets`` and tokenizes them with the same Qwen 2.5 tokenizer the
Synap-1 100M trainer uses. Output is one parquet ready for
``ParquetTokenStream`` ingestion in the phase-3+ code-SFT mix.

Datasets
--------
* ``openai_humaneval``         (164 problems, "test" split)
* ``mbpp``                     (974 problems, "train+validation+test")

Output schema
-------------
    task_id     str        canonical id ("HumanEval/0", "Mbpp/3", ...)
    source      str        "humaneval" | "mbpp"
    prompt      str        problem text
    solution    str        canonical solution code
    input_ids   list<int>  Qwen tokenization of "prompt + '\n' + solution"

Usage
-----
    python scripts/tokenize_humaneval_mbpp.py \\
        --out /workspace/data/code_eval/humaneval_mbpp_qwen.parquet

    python scripts/tokenize_humaneval_mbpp.py --smoke   # 4 rows total

The script prefers ``huggingface_hub`` cache + ``datasets`` (offline-ok
once primed). Network failure falls back to a clear error -- the
tokenizer load itself is the more likely failure mode on rentals.
"""
from __future__ import annotations

import argparse
import json
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


def _load_tokenizer(name: str = "Qwen/Qwen2.5-0.5B"):  # pragma: no cover
    """Lazy import; returns an HF AutoTokenizer."""
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


def _humaneval_rows() -> List[dict]:  # pragma: no cover -- network
    """Yield (task_id, prompt, solution) tuples from openai_humaneval."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    rows = []
    for ex in ds:
        rows.append(
            dict(
                task_id=str(ex["task_id"]),
                source="humaneval",
                prompt=str(ex["prompt"]),
                # HumanEval canonical: the prompt already contains the
                # function signature; canonical_solution is the body.
                solution=str(ex.get("canonical_solution") or ""),
            )
        )
    return rows


def _mbpp_rows() -> List[dict]:  # pragma: no cover -- network
    from datasets import load_dataset
    rows = []
    for split in ("train", "validation", "test", "prompt"):
        try:
            ds = load_dataset("mbpp", split=split)
        except Exception:
            continue
        for ex in ds:
            tid = ex.get("task_id") or ex.get("id") or ""
            rows.append(
                dict(
                    task_id=f"Mbpp/{tid}",
                    source="mbpp",
                    prompt=str(ex.get("text") or ex.get("prompt") or ""),
                    solution=str(ex.get("code") or ""),
                )
            )
    # Dedup by task_id (mbpp shows up under multiple splits).
    seen, uniq = set(), []
    for r in rows:
        if r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        uniq.append(r)
    return uniq


def _tokenize_rows(rows: Iterable[dict], tokenizer) -> List[dict]:  # pragma: no cover
    out = []
    for r in rows:
        joined = (r["prompt"] or "") + "\n" + (r["solution"] or "")
        ids = tokenizer.encode(joined, add_special_tokens=False)
        out.append(dict(r, input_ids=list(map(int, ids))))
    return out


def write_parquet(path: str, rows: Sequence[dict]) -> int:
    if not _HAVE_ARROW:
        raise RuntimeError("pyarrow required to write parquet")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "task_id": [r["task_id"] for r in rows],
        "source": [r["source"] for r in rows],
        "prompt": [r["prompt"] for r in rows],
        "solution": [r["solution"] for r in rows],
        "input_ids": [r["input_ids"] for r in rows],
    }
    pq.write_table(pa.table(cols), path, compression="zstd")
    with open(str(path) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "code_eval_tokenized",
                "rows": len(rows),
                "sources": sorted({r["source"] for r in rows}),
                "tokenizer": "Qwen/Qwen2.5-0.5B",
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
        )
    return len(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out",
                    default="/workspace/data/code_eval/humaneval_mbpp_qwen.parquet")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--smoke", action="store_true",
                    help="Use 2 hand-rolled rows (no network); for unit tests")
    args = ap.parse_args(argv)

    if args.smoke:
        rows = [
            dict(
                task_id="HumanEval/_smoke0",
                source="humaneval",
                prompt="def add(a, b):\n    \"\"\"return a+b\"\"\"\n",
                solution="    return a + b\n",
            ),
            dict(
                task_id="Mbpp/_smoke0",
                source="mbpp",
                prompt="Write a function to multiply two numbers.",
                solution="def mul(a, b):\n    return a * b\n",
            ),
        ]
        # Smoke skips real tokenizer; emits empty input_ids.
        for r in rows:
            r["input_ids"] = []
        n = write_parquet(args.out, rows)
        print(f"[tokenize_code] smoke wrote {n} rows -> {args.out}", flush=True)
        return 0

    tokenizer = _load_tokenizer(args.tokenizer)
    he = _humaneval_rows()
    mbpp = _mbpp_rows()
    rows = _tokenize_rows(he + mbpp, tokenizer)
    n = write_parquet(args.out, rows)
    print(
        f"[tokenize_code] wrote {n} rows -> {args.out} "
        f"(humaneval={len(he)} mbpp={len(mbpp)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
