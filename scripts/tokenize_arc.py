"""tokenize_arc.py — fetch + tokenize ARC-Easy + ARC-Challenge for T3.9.

Pulls the AI2 Reasoning Challenge multiple-choice science question sets
from HuggingFace ``datasets`` and tokenizes them with the same Qwen 2.5
tokenizer the Synap-1 100M trainer uses. Output is one parquet ready
for **multiple-choice log-likelihood eval**: for each example we keep
both the prompt-only token IDs and the prompt+answer token IDs for
each of the 4 candidate choices, so an evaluator can pick
``argmax_i logP(choice_i | prompt)``.

Datasets
--------
* ``allenai/ai2_arc`` config ``"ARC-Easy"``       (5,197 train + val + test)
* ``allenai/ai2_arc`` config ``"ARC-Challenge"``  (2,590 train + val + test)

Each example has::

    {
        "id":         "Mercury_7220990",
        "question":   "Which factor will most likely cause a person to develop a fever?",
        "choices":    {
            "label": ["A", "B", "C", "D"],
            "text":  ["a leg muscle relaxing after exercise",
                      "a bacterial population in the bloodstream",
                      "several viral particles on the skin",
                      "carbohydrates being digested in the stomach"],
        },
        "answerKey":  "B",
    }

Output schema (parquet)
-----------------------
    task                    str          "Easy" | "Challenge"
    split                   str          "train" | "validation"
    id                      str          ARC question id
    question                str          stem text
    choices                 list<str>    4 choice texts (A..D order)
    answerKey               str          one of "A".."D"
    answer_idx              int32        0..3 — which choice is correct
    prompt_input_ids        list<int>    Qwen tokenization of "Question: ...\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nAnswer:"
    prompt_plus_answer_ids  list<list<int>>  4 sequences — prompt + " <letter>" for each candidate
                                              shape ``[4][seq]``

Companion ``.manifest.json`` records row count + tokenizer source.

Usage
-----
    python scripts/tokenize_arc.py \\
        --out /workspace/data/reasoning/arc_qwen.parquet

    python scripts/tokenize_arc.py --smoke   # 5 hand-rolled rows, no net
    python scripts/tokenize_arc.py --help

Determinism
-----------
HuggingFace dataset order is deterministic; we keep all train+validation
rows. Splits are carried through to the parquet so downstream eval can
filter on them.
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


# ARC choices are sometimes labelled "1".."4" or "A".."D"; normalise to
# A..D for prompt formatting.
_LETTERS = ("A", "B", "C", "D")


def extract_answer_idx(answer_key: str, labels: Sequence[str]) -> Optional[int]:
    """Return the 0-based index of ``answer_key`` inside ``labels``.

    ``answer_key`` is the canonical label from the dataset ("A".."D" or
    "1".."4"); ``labels`` is the per-row label list. We trust the row's
    own label order (some examples come labelled "1234" not "ABCD").
    Returns ``None`` for malformed rows.
    """
    if not answer_key:
        return None
    key = str(answer_key).strip()
    for i, lbl in enumerate(labels):
        if str(lbl).strip() == key:
            return i
    return None


def format_prompt(question: str, choices: Sequence[str]) -> str:
    """Compose a multiple-choice prompt with A..D labelled choices.

    Pads with empty strings if fewer than 4 choices are provided so the
    output has a stable layout. Trailing ``"Answer:"`` (no space) is
    where the per-choice continuation token will attach.
    """
    lines = [f"Question: {question.strip()}"]
    for i, ch in enumerate(choices[:4]):
        letter = _LETTERS[i] if i < len(_LETTERS) else str(i)
        lines.append(f"{letter}. {ch}")
    # Pad missing choices so the prompt always shows A..D.
    for i in range(len(choices), 4):
        letter = _LETTERS[i]
        lines.append(f"{letter}. ")
    lines.append("Answer:")
    return "\n".join(lines)


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


def _load_arc(task: str, split: str) -> Iterable[dict]:  # pragma: no cover -- network
    """Yield raw ARC records ({id, question, choices, answerKey}) from HF."""
    from datasets import load_dataset
    cfg = "ARC-Easy" if task == "Easy" else "ARC-Challenge"
    ds = load_dataset("allenai/ai2_arc", cfg, split=split)
    for ex in ds:
        yield {
            "id": str(ex.get("id") or ""),
            "question": str(ex.get("question") or ""),
            "choices": dict(ex.get("choices") or {}),
            "answerKey": str(ex.get("answerKey") or ""),
        }


def _build_rows(
    raw_records: Iterable[dict],
    tokenizer,
    task: str,
    split: str,
) -> List[dict]:
    """Format each example into the multiple-choice eval shape.

    Drops examples that don't have exactly the labels listed in the
    record OR whose ``answerKey`` doesn't appear (rare malformed rows).
    """
    rows: List[dict] = []
    for rec in raw_records:
        q = rec.get("question") or ""
        ch_obj = rec.get("choices") or {}
        labels = list(ch_obj.get("label") or [])
        texts = list(ch_obj.get("text") or [])
        if not q or not texts:
            continue
        idx = extract_answer_idx(rec.get("answerKey") or "", labels)
        if idx is None or idx >= len(texts):
            continue
        # Pad / truncate to exactly 4 choices for a uniform eval shape.
        choices4 = list(texts[:4]) + [""] * max(0, 4 - len(texts))
        prompt = format_prompt(q, choices4)
        if tokenizer is None:
            prompt_ids: List[int] = []
            plus_ids: List[List[int]] = [[] for _ in range(4)]
        else:
            prompt_ids = list(map(int, tokenizer.encode(
                prompt, add_special_tokens=False)))
            plus_ids = []
            for i in range(4):
                full = prompt + " " + _LETTERS[i]
                plus_ids.append(list(map(int, tokenizer.encode(
                    full, add_special_tokens=False))))
        rows.append(
            dict(
                task=task,
                split=split,
                id=str(rec.get("id") or ""),
                question=q,
                choices=choices4,
                answerKey=str(rec.get("answerKey") or ""),
                answer_idx=int(idx),
                prompt_input_ids=prompt_ids,
                prompt_plus_answer_ids=plus_ids,
            )
        )
    return rows


def write_parquet(path: str, rows: Sequence[dict],
                  tokenizer_name: str = "Qwen/Qwen2.5-0.5B") -> int:
    if not _HAVE_ARROW:
        raise RuntimeError("pyarrow required to write parquet")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "task": [r["task"] for r in rows],
        "split": [r["split"] for r in rows],
        "id": [r["id"] for r in rows],
        "question": [r["question"] for r in rows],
        "choices": [r["choices"] for r in rows],
        "answerKey": [r["answerKey"] for r in rows],
        "answer_idx": [int(r["answer_idx"]) for r in rows],
        "prompt_input_ids": [r["prompt_input_ids"] for r in rows],
        "prompt_plus_answer_ids": [r["prompt_plus_answer_ids"] for r in rows],
    }
    pq.write_table(pa.table(cols), path, compression="zstd")
    with open(str(path) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "arc_tokenized",
                "rows": len(rows),
                "tasks": sorted({r["task"] for r in rows}),
                "splits": sorted({r["split"] for r in rows}),
                "tokenizer": tokenizer_name,
                "schema": {
                    "task": "string",
                    "split": "string",
                    "id": "string",
                    "question": "string",
                    "choices": "list<string>",
                    "answerKey": "string",
                    "answer_idx": "int32",
                    "prompt_input_ids": "list<int32>",
                    "prompt_plus_answer_ids": "list<list<int32>>",
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
        "id": "_smoke_easy_0",
        "question": "Which planet is closest to the sun?",
        "choices": {"label": ["A", "B", "C", "D"],
                    "text": ["Earth", "Mercury", "Venus", "Mars"]},
        "answerKey": "B",
    },
    {
        "id": "_smoke_easy_1",
        "question": "What gas do plants absorb during photosynthesis?",
        "choices": {"label": ["A", "B", "C", "D"],
                    "text": ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"]},
        "answerKey": "C",
    },
    {
        "id": "_smoke_challenge_0",
        "question": "Which is a renewable energy source?",
        "choices": {"label": ["A", "B", "C", "D"],
                    "text": ["Coal", "Oil", "Solar", "Natural gas"]},
        "answerKey": "C",
    },
    {
        "id": "_smoke_challenge_1",
        "question": "How many bones are in the adult human body?",
        "choices": {"label": ["A", "B", "C", "D"],
                    "text": ["106", "206", "306", "406"]},
        "answerKey": "B",
    },
    {
        "id": "_smoke_challenge_2",
        "question": "Which is the largest organ in the human body?",
        "choices": {"label": ["1", "2", "3", "4"],
                    "text": ["Liver", "Heart", "Skin", "Lungs"]},
        "answerKey": "3",
    },
]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="/workspace/data/reasoning/arc_qwen.parquet")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cache-dir", default=None,
                    help="Optional HF datasets cache dir (sets HF_DATASETS_CACHE)")
    ap.add_argument("--smoke", action="store_true",
                    help="Use 5 hand-rolled records (no network); for unit tests")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.cache_dir:
        import os
        os.environ["HF_DATASETS_CACHE"] = str(args.cache_dir)

    if args.smoke:
        # 5 mocked rows: 2 Easy, 3 Challenge — no network, no tokenizer.
        easy_rows = _build_rows(iter(SMOKE_RECORDS[:2]),
                                tokenizer=None, task="Easy", split="train")
        chall_rows = _build_rows(iter(SMOKE_RECORDS[2:]),
                                 tokenizer=None, task="Challenge", split="train")
        rows = easy_rows + chall_rows
        n = write_parquet(args.out, rows, tokenizer_name="smoke-no-tokenizer")
        print(f"[arc] smoke wrote {n} rows -> {args.out}", flush=True)
        return 0

    tokenizer = _load_tokenizer(args.tokenizer)  # pragma: no cover -- network
    all_rows: List[dict] = []
    for task in ("Easy", "Challenge"):
        for split in ("train", "validation"):
            try:
                raw = _load_arc(task, split=split)  # pragma: no cover -- network
            except Exception as exc:
                print(f"[arc] WARN: failed to load {task}/{split}: {exc!r}",
                      file=sys.stderr)
                continue
            all_rows.extend(_build_rows(raw, tokenizer=tokenizer,
                                        task=task, split=split))
    if not all_rows:
        print("[arc] no rows produced", file=sys.stderr)
        return 1
    n = write_parquet(args.out, all_rows, tokenizer_name=args.tokenizer)
    print(f"[arc] wrote {n} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
