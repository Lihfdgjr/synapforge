"""tokenize_swebench_mini.py — subset + tokenize SWE-bench-Lite (T3.10).

Pulls 50 issues from ``princeton-nlp/SWE-bench_Lite`` (test split, 300
total examples) deterministically — sort by ``instance_id`` and take the
first ``--n`` (default 50). Tokenizes both the prompt and the patch
target with the Qwen 2.5 0.5B tokenizer used by the Synap-1 100M
trainer. Output is one parquet ready for the **code-fix demo** stage.

Dataset
-------
* ``princeton-nlp/SWE-bench_Lite`` (test split = 300 examples)
  Each row carries:
    - ``instance_id``        (canonical id, e.g. "django__django-12345")
    - ``repo``               (e.g. "django/django")
    - ``base_commit``        (commit SHA the patch is applied against)
    - ``problem_statement``  (the GitHub issue body / bug report)
    - ``patch``              (golden unified-diff fix)
    - ``test_patch``         (test added in the same PR; not used here)
    - ``hints_text``         (optional hints; not used here)

Prompt format
-------------
::

    Repo: <repo>
    Problem:
    <problem_statement>

    Provide a unified diff patch:

The target is the raw ``patch`` field. Both prompt and patch are
tokenized independently to ``max_length=4096`` with ``truncation=True``
(SWE-bench patches can be very long — graceful truncation matters).

Output schema (parquet)
-----------------------
    instance_id        str        canonical SWE-bench id
    repo               str        owner/name
    prompt             str        formatted prompt sent to the model
    patch_target       str        ground-truth unified-diff fix
    prompt_input_ids   list<int>  Qwen tokenization of ``prompt``
    target_input_ids   list<int>  Qwen tokenization of ``patch_target``

Companion ``.manifest.json`` records row count + tokenizer source +
truncation settings.

Usage
-----
    python scripts/tokenize_swebench_mini.py \\
        --out /workspace/data/code_fix/swebench_mini_qwen.parquet \\
        --n 50

    python scripts/tokenize_swebench_mini.py --smoke   # 5 mocked rows, no net
    python scripts/tokenize_swebench_mini.py --help

Determinism
-----------
Sort by ``instance_id`` before taking the first ``n`` so the same 50
issues are picked across machines. The full SWE-bench-Lite test split
is fixed at 300 rows, so the subset is reproducible without seeding.
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


PROMPT_TEMPLATE = (
    "Repo: {repo}\n"
    "Problem:\n"
    "{problem_statement}\n\n"
    "Provide a unified diff patch:\n"
)


def format_prompt(repo: str, problem_statement: str) -> str:
    """Compose the SWE-bench code-fix prompt (deterministic, no f-string surprises)."""
    return PROMPT_TEMPLATE.format(
        repo=(repo or "").strip(),
        problem_statement=(problem_statement or "").strip(),
    )


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


def _load_swebench_lite(
    split: str = "test",
    cache_dir: Optional[str] = None,
) -> Iterable[dict]:  # pragma: no cover -- network
    """Yield raw SWE-bench-Lite records from HF datasets."""
    from datasets import load_dataset
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split, **kwargs)
    for ex in ds:
        yield {
            "instance_id": str(ex.get("instance_id") or ""),
            "repo": str(ex.get("repo") or ""),
            "base_commit": str(ex.get("base_commit") or ""),
            "problem_statement": str(ex.get("problem_statement") or ""),
            "patch": str(ex.get("patch") or ""),
        }


def _encode_truncate(tokenizer, text: str, max_length: int) -> List[int]:
    """Tokenize with explicit truncation. Returns a flat list of ints.

    ``tokenizer.encode`` historically does NOT respect ``max_length`` /
    ``truncation`` on every backend, so we slice as a belt-and-braces
    fallback (cheap on lists).
    """
    if tokenizer is None:
        return []
    try:
        ids = tokenizer.encode(
            text or "",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
    except TypeError:
        # Some HF builds reject keyword args on .encode; fall back.
        ids = tokenizer.encode(text or "", add_special_tokens=False)
    ids = [int(x) for x in ids]
    if len(ids) > max_length:
        ids = ids[:max_length]
    return ids


def build_rows(
    raw_records: Iterable[dict],
    tokenizer,
    n: int,
    max_length: int = 4096,
) -> List[dict]:
    """Sort records by instance_id, take first ``n``, format + tokenize."""
    records = list(raw_records)
    records.sort(key=lambda r: str(r.get("instance_id") or ""))
    rows: List[dict] = []
    for rec in records:
        if len(rows) >= n:
            break
        repo = rec.get("repo") or ""
        problem = rec.get("problem_statement") or ""
        patch = rec.get("patch") or ""
        prompt = format_prompt(repo, problem)
        prompt_ids = _encode_truncate(tokenizer, prompt, max_length)
        target_ids = _encode_truncate(tokenizer, patch, max_length)
        rows.append(
            dict(
                instance_id=str(rec.get("instance_id") or ""),
                repo=str(repo),
                prompt=prompt,
                patch_target=patch,
                prompt_input_ids=prompt_ids,
                target_input_ids=target_ids,
            )
        )
    return rows


def write_parquet(
    path: str,
    rows: Sequence[dict],
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    max_length: int = 4096,
) -> int:
    if not _HAVE_ARROW:
        raise RuntimeError("pyarrow required to write parquet")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "instance_id": [r["instance_id"] for r in rows],
        "repo": [r["repo"] for r in rows],
        "prompt": [r["prompt"] for r in rows],
        "patch_target": [r["patch_target"] for r in rows],
        "prompt_input_ids": [r["prompt_input_ids"] for r in rows],
        "target_input_ids": [r["target_input_ids"] for r in rows],
    }
    pq.write_table(pa.table(cols), path, compression="zstd")
    with open(str(path) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "swebench_lite_mini_tokenized",
                "rows": len(rows),
                "tokenizer": tokenizer_name,
                "max_length": int(max_length),
                "truncation": True,
                "dataset": "princeton-nlp/SWE-bench_Lite",
                "split": "test",
                "schema": {
                    "instance_id": "string",
                    "repo": "string",
                    "prompt": "string",
                    "patch_target": "string",
                    "prompt_input_ids": "list<int32>",
                    "target_input_ids": "list<int32>",
                },
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
        )
    return len(rows)


# --- smoke-mode stand-in dataset (no network) -------------------------------
SMOKE_RECORDS: List[dict] = [
    {
        "instance_id": "smoke__demo-0001",
        "repo": "smoke/demo",
        "base_commit": "deadbeef",
        "problem_statement": (
            "Calling foo(None) raises AttributeError instead of returning 0."
        ),
        "patch": (
            "--- a/foo.py\n+++ b/foo.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-def foo(x):\n-    return x.value\n"
            "+def foo(x):\n+    return x.value if x is not None else 0\n"
        ),
    },
    {
        "instance_id": "smoke__demo-0002",
        "repo": "smoke/demo",
        "base_commit": "cafef00d",
        "problem_statement": "off-by-one in range slice; loses last element.",
        "patch": (
            "--- a/slice.py\n+++ b/slice.py\n"
            "@@ -10,1 +10,1 @@\n"
            "-    return xs[: n - 1]\n"
            "+    return xs[:n]\n"
        ),
    },
    {
        "instance_id": "smoke__demo-0003",
        "repo": "smoke/example",
        "base_commit": "0123abcd",
        "problem_statement": "leak: file handle not closed in error path.",
        "patch": (
            "--- a/io.py\n+++ b/io.py\n"
            "@@ -5,3 +5,4 @@\n"
            "     f = open(p)\n"
            "+    try:\n"
            "         data = f.read()\n"
            "+    finally:\n         f.close()\n"
        ),
    },
    {
        "instance_id": "smoke__demo-0004",
        "repo": "smoke/example",
        "base_commit": "ff00ff00",
        "problem_statement": "TypeError when input is bytes; expected str only.",
        "patch": (
            "--- a/parse.py\n+++ b/parse.py\n"
            "@@ -2,1 +2,3 @@\n"
            "-def parse(s):\n+def parse(s):\n"
            "+    if isinstance(s, bytes):\n+        s = s.decode('utf-8')\n"
        ),
    },
    {
        "instance_id": "smoke__demo-0005",
        "repo": "smoke/x",
        "base_commit": "abcd1234",
        "problem_statement": "negative index loops forever in counter.",
        "patch": (
            "--- a/counter.py\n+++ b/counter.py\n"
            "@@ -3,1 +3,1 @@\n"
            "-    while i != stop:\n+    while i < stop:\n"
        ),
    },
]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out",
        default="/workspace/data/code_fix/swebench_mini_qwen.parquet",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=50,
        help="How many SWE-bench-Lite issues to keep (sorted by instance_id)",
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Max token length for prompt and target (truncation=True)",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace datasets cache directory",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Use 5 hand-rolled mocked records (no network); for unit tests",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        # No network, no real tokenizer (input_ids stay empty).
        rows = build_rows(
            iter(SMOKE_RECORDS),
            tokenizer=None,
            n=min(args.n, len(SMOKE_RECORDS)),
            max_length=args.max_length,
        )
        n = write_parquet(
            args.out,
            rows,
            tokenizer_name="smoke-no-tokenizer",
            max_length=args.max_length,
        )
        print(f"[swebench_mini] smoke wrote {n} rows -> {args.out}", flush=True)
        return 0

    tokenizer = _load_tokenizer(args.tokenizer)  # pragma: no cover -- network
    raw = _load_swebench_lite(split="test", cache_dir=args.cache_dir)  # pragma: no cover -- network
    rows = build_rows(raw, tokenizer=tokenizer, n=args.n, max_length=args.max_length)
    if not rows:
        print("[swebench_mini] FATAL: no rows produced", file=sys.stderr)
        return 1
    n = write_parquet(
        args.out,
        rows,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
    )
    print(f"[swebench_mini] wrote {n} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
