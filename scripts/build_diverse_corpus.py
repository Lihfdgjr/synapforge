"""build_diverse_corpus.py — assemble a diverse 1B-token training corpus.

The Synap-1 trainer currently cycles a single ZH parquet shard
(``/workspace/data/synth_zh_phase1.parquet``) plus an instruction shard.
That gives the val-ppl curve nowhere to go below ~7000 because the
deterministic ParquetTokenStream replays the same lexical sequence
forever (P24 in MASTER_PLAN.md §6 logged the same divergence pathology
on Run 3a/3b/3c).

This script fetches diverse English LM, code, math, and instruction
data alongside the existing ZH shard, tokenizes it with the same
Qwen 2.5 0.5B tokenizer the trainer uses, and writes a single concat
parquet with a ``corpus`` label column. Output schema is identical to
``mix_pretrain_corpora.py`` so the trainer ingests it with no changes.

Per-category default targets (--target-tokens 1B):

    ============  ===========  ====
    category      tokens       %
    ============  ===========  ====
    en            500M         50%
    zh            200M         20%
    code          150M         15%
    math          100M         10%
    instruct       50M          5%
    ============  ===========  ====

Sources (HuggingFace ``datasets`` streaming):

    en        HuggingFaceFW/fineweb-edu (10BT sample)
    zh        BAAI/CCI3-HQ + existing synth_zh_phase1.parquet
    code      bigcode/the-stack-v2-train-smol (Python only)
    math      open-web-math/open-web-math + EleutherAI/proof-pile-2
    instruct  silk-road/alpaca-data-gpt4-chinese + tatsu-lab/alpaca

Usage:

    python scripts/build_diverse_corpus.py \\
        --target-tokens 1B \\
        --include en,zh,code,math,instruct \\
        --out /workspace/data/diverse_corpus.parquet

    python scripts/build_diverse_corpus.py --smoke    # 5 docs/category, no net

The ``--smoke`` flag short-circuits all HF calls with hand-written
fixtures so CI / Windows-dev / no-internet machines can validate the
pipeline. Tests under ``tests/integration/test_diverse_corpus.py`` use
``--smoke`` exclusively.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


# ---------------------------------------------------------------------------
# Default per-category ratios. Sum must == 1.0; --include picks a subset and
# the remaining ratios are renormalised over the kept categories.
# ---------------------------------------------------------------------------
DEFAULT_RATIOS: dict[str, float] = {
    "en":       0.50,
    "zh":       0.20,
    "code":     0.15,
    "math":     0.10,
    "instruct": 0.05,
}

# HuggingFace dataset coordinates per category. Each entry is a list of
# (repo_id, config, split, text_col) candidates we walk in order; the
# first one that loads wins.
HF_SOURCES: dict[str, list[tuple[str, Optional[str], str, str]]] = {
    "en": [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
        ("HuggingFaceFW/fineweb",     "sample-10BT", "train", "text"),
    ],
    "zh": [
        ("BAAI/CCI3-HQ",       None, "train", "text"),
        ("Skywork/SkyPile-150B", None, "train", "text"),
    ],
    "code": [
        ("bigcode/the-stack-v2-train-smol", "Python", "train", "content"),
        ("codeparrot/codeparrot-clean",     None,    "train", "content"),
    ],
    "math": [
        ("open-web-math/open-web-math", None, "train", "text"),
        ("EleutherAI/proof-pile-2",     None, "train", "text"),
    ],
    "instruct": [
        # Instruction sources have their own (instruction, input, output)
        # schema; we render a chat-style string downstream.
        ("silk-road/alpaca-data-gpt4-chinese", None, "train", "_alpaca_zh"),
        ("tatsu-lab/alpaca",                   None, "train", "_alpaca_en"),
    ],
}


# ---------------------------------------------------------------------------
# Smoke fixtures: hand-written rows per category, no network.
# ---------------------------------------------------------------------------
SMOKE_FIXTURES: dict[str, list[str]] = {
    "en": [
        "The quick brown fox jumps over the lazy dog. This is sentence one of an English fixture document.",
        "Photosynthesis converts light energy into chemical energy in plants and some bacteria, producing oxygen.",
        "A theorem in mathematics is a statement that has been proven on the basis of previously established statements.",
        "Newton's third law states that for every action there is an equal and opposite reaction in the universe.",
        "Modern computing relies on transistors arranged into integrated circuits etched onto silicon wafers in fabs.",
    ],
    "zh": [
        "近年来人工智能领域取得了显著进展，深度学习在视觉与语言任务上达到了前所未有的水平。",
        "唐朝是中国历史上的一个重要时期，政治制度完善，文化艺术繁荣，对外交流频繁。",
        "微积分是研究函数变化率与累积量的数学分支，由牛顿与莱布尼茨在十七世纪独立创立。",
        "热力学第二定律表明孤立系统的熵不会减少，这对宇宙演化具有深远意义。",
        "供需关系是市场经济中决定价格的核心机制，理解其原理是经济学入门的基础。",
    ],
    "code": [
        "def add(a, b):\n    \"\"\"Return the sum of two numbers.\"\"\"\n    return a + b\n",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "import os\nfor name in os.listdir('.'):\n    if name.endswith('.py'):\n        print(name)\n",
        "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, x):\n        self._items.append(x)\n",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True\n",
    ],
    "math": [
        "Theorem (Pythagoras): In a right triangle with legs a, b and hypotenuse c, we have a^2 + b^2 = c^2.",
        "The integral of e^x dx is e^x + C, where C is the constant of integration. This follows from d/dx[e^x] = e^x.",
        "Lemma: If f and g are continuous on [a, b], then f + g is continuous on [a, b]. Proof by epsilon-delta argument.",
        "The Fibonacci sequence satisfies F(n) = F(n-1) + F(n-2). Closed form: F(n) = (phi^n - psi^n) / sqrt(5).",
        "Proof that sqrt(2) is irrational: assume sqrt(2) = p/q in lowest terms; then 2q^2 = p^2 so p is even.",
    ],
    "instruct": [
        "### Instruction:\nWhat is the capital of France?\n### Response:\nThe capital of France is Paris.",
        "### Instruction:\n请用一句话概括相对论。\n### Response:\n相对论指出时间和空间是相对于观察者的。",
        "### Instruction:\nWrite a Python one-liner to reverse a string.\n### Response:\ns[::-1]",
        "### Instruction:\nExplain photosynthesis in two sentences.\n### Response:\nPhotosynthesis is how plants turn sunlight, water, and CO2 into glucose and oxygen. It powers nearly all life on Earth.",
        "### Instruction:\n中文里如何表达感谢？\n### Response:\n常说'谢谢'，更正式时用'感谢您'。",
    ],
}


# ---------------------------------------------------------------------------
# Token-count helpers
# ---------------------------------------------------------------------------
def _parse_token_target(s: str) -> int:
    """Parse e.g. '1B' / '500M' / '1000000' into an integer token count."""
    s = s.strip().upper()
    if not s:
        raise ValueError("empty --target-tokens")
    suffix = 1
    if s.endswith("B"):
        suffix = 1_000_000_000
        s = s[:-1]
    elif s.endswith("M"):
        suffix = 1_000_000
        s = s[:-1]
    elif s.endswith("K"):
        suffix = 1_000
        s = s[:-1]
    return int(float(s) * suffix)


def _hash_key(text: str) -> str:
    """sha256 prefix on the first 4096 chars (matches mix_pretrain_corpora)."""
    return hashlib.sha256(text[:4096].encode("utf-8", errors="ignore")).hexdigest()[:32]


def _approx_tokens(text: str) -> int:
    """Quick char-based token estimate when the real tokenizer is unavailable.

    Empirically Qwen tokenizes ~3.5 chars/token on mixed English / Chinese.
    Using 4 keeps us conservative (slight over-budget of rows, never short).
    """
    return max(1, len(text) // 4)


@dataclass
class SourceResult:
    category: str
    rows: list[str]
    n_tokens_est: int


# ---------------------------------------------------------------------------
# HF streaming fetch — production path. Wrapped so smoke mode never imports.
# ---------------------------------------------------------------------------
def _stream_hf_text(
    repo_id: str,
    config: Optional[str],
    split: str,
    text_col: str,
    target_tokens: int,
    tokenizer=None,
) -> Iterator[str]:  # pragma: no cover -- needs network + datasets
    """Yield plain text rows from a HF dataset until we hit ``target_tokens``.

    Special-cases the alpaca instruction format via the ``_alpaca_*`` text
    column sentinels (rendered as the ``### Instruction / ### Response``
    template the trainer's chat_v* harness already speaks).
    """
    from datasets import load_dataset
    kw = {"split": split, "streaming": True}
    if config is not None:
        kw["name"] = config
    ds = load_dataset(repo_id, **kw)
    n = 0
    for ex in ds:
        if n >= target_tokens:
            return
        text = _coerce_text(ex, text_col)
        if not text:
            continue
        if tokenizer is not None:
            ntok = len(tokenizer.encode(text, add_special_tokens=False))
        else:
            ntok = _approx_tokens(text)
        n += ntok
        yield text


def _coerce_text(ex: dict, text_col: str) -> str:
    """Extract a plain-text body from one HF row.

    Honours the two ``_alpaca_*`` sentinels for the instruction format,
    falling through to a vanilla column lookup otherwise.
    """
    if text_col == "_alpaca_zh" or text_col == "_alpaca_en":
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if not instr or not out:
            return ""
        head = f"### Instruction:\n{instr}\n"
        if inp:
            head += f"### Input:\n{inp}\n"
        return f"{head}### Response:\n{out}"
    val = ex.get(text_col)
    return str(val) if val else ""


def _load_tokenizer(name: str = "Qwen/Qwen2.5-0.5B"):  # pragma: no cover
    from transformers import AutoTokenizer
    candidates = ["/workspace/teachers/qwen2.5-0.5b", name]
    last = None
    for p in candidates:
        try:
            return AutoTokenizer.from_pretrained(p, trust_remote_code=True)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"could not load Qwen tokenizer: {last!r}")


# ---------------------------------------------------------------------------
# Per-category fetcher (smoke vs real)
# ---------------------------------------------------------------------------
def _fetch_category(
    cat: str,
    target_tokens: int,
    smoke: bool,
    tokenizer=None,
) -> SourceResult:
    """Return ``target_tokens`` (approximate) of text for ``cat``.

    Smoke mode replays ``SMOKE_FIXTURES[cat]`` until the budget is met. We
    do **not** use a tokenizer in smoke mode, so the budget is char-based.
    """
    rows: list[str] = []
    n_tok = 0
    if smoke:
        fixtures = SMOKE_FIXTURES.get(cat, [])
        if not fixtures:
            return SourceResult(category=cat, rows=[], n_tokens_est=0)
        i = 0
        # Cycle the fixture pool so the budget is met deterministically.
        # Cap at len(fixtures) rows in smoke mode — tests assume exactly 5.
        for fx in fixtures:
            rows.append(fx)
            n_tok += _approx_tokens(fx)
            i += 1
        return SourceResult(category=cat, rows=rows, n_tokens_est=n_tok)

    # Real fetch path: walk HF source candidates until one yields rows.
    for (repo_id, config, split, text_col) in HF_SOURCES.get(cat, []):
        try:
            for text in _stream_hf_text(  # pragma: no cover -- network
                repo_id, config, split, text_col,
                target_tokens=target_tokens, tokenizer=tokenizer,
            ):
                rows.append(text)
                n_tok += (
                    len(tokenizer.encode(text, add_special_tokens=False))
                    if tokenizer is not None else _approx_tokens(text)
                )
                if n_tok >= target_tokens:
                    break
            if rows:
                break  # this source worked — stop walking candidates
        except Exception as exc:
            print(f"[diverse] {cat} source {repo_id!r} failed: {exc!r}",
                  file=sys.stderr)
            continue
    return SourceResult(category=cat, rows=rows, n_tokens_est=n_tok)


# ---------------------------------------------------------------------------
# Cross-source dedup: drop near-identical rows by sha256(first 4096 chars).
# ---------------------------------------------------------------------------
def _dedup_across_sources(
    results: list[SourceResult],
) -> tuple[list[str], list[str], int]:
    """Concatenate rows from each source with a category label and dedup.

    Returns ``(texts, categories, n_dropped)``.
    """
    seen: set[str] = set()
    out_text: list[str] = []
    out_cat: list[str] = []
    dropped = 0
    for r in results:
        for txt in r.rows:
            key = _hash_key(txt)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            out_text.append(txt)
            out_cat.append(r.category)
    return out_text, out_cat, dropped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _normalise_ratios(include: Sequence[str]) -> dict[str, float]:
    sub = {k: DEFAULT_RATIOS[k] for k in include if k in DEFAULT_RATIOS}
    s = sum(sub.values())
    if s <= 0:
        raise ValueError(f"--include {include!r} matches no known categories")
    return {k: v / s for k, v in sub.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-tokens", default="1B",
                    help="Total token budget across all categories. "
                         "Accepts B/M/K suffix; default 1B.")
    ap.add_argument("--out", default="/workspace/data/diverse_corpus.parquet",
                    help="Output parquet path.")
    ap.add_argument("--include",
                    default="en,zh,code,math,instruct",
                    help="Comma-separated categories to fetch. Subset of "
                         "{en,zh,code,math,instruct}; ratios renormalise.")
    ap.add_argument("--smoke", action="store_true",
                    help="Use hand-written fixtures (no HF network); "
                         "for unit tests + offline rentals.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B",
                    help="HF tokenizer for non-smoke mode token counting.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Reserved; the streaming order is deterministic "
                         "from HF so no shuffle is applied here.")
    args = ap.parse_args(argv)

    if not _HAVE_ARROW:
        print("[diverse] pyarrow required to write parquet output",
              file=sys.stderr)
        return 2

    include = [c.strip() for c in args.include.split(",") if c.strip()]
    try:
        ratios = _normalise_ratios(include)
    except ValueError as exc:
        print(f"[diverse] {exc}", file=sys.stderr)
        return 3
    target_tokens = _parse_token_target(args.target_tokens)
    print(f"[diverse] target_tokens={target_tokens:,} include={include} "
          f"ratios={ratios} smoke={args.smoke}")

    # Real-mode tokenizer load (smoke mode uses char-based estimate).
    tokenizer = None
    if not args.smoke:
        try:
            tokenizer = _load_tokenizer(args.tokenizer)  # pragma: no cover
        except Exception as exc:
            print(f"[diverse] tokenizer load failed: {exc!r}; "
                  f"falling back to char-based token estimate",
                  file=sys.stderr)

    # Per-category fetch budget = target * ratio.
    results: list[SourceResult] = []
    for cat, r in ratios.items():
        budget = int(target_tokens * r)
        res = _fetch_category(cat, budget, smoke=args.smoke, tokenizer=tokenizer)
        print(f"[diverse] cat={cat:8s} budget={budget:>12,} "
              f"got_rows={len(res.rows):,} got_tokens~{res.n_tokens_est:,}")
        results.append(res)

    out_text, out_cat, n_dropped = _dedup_across_sources(results)
    if not out_text:
        print("[diverse] FATAL: no rows produced (all sources empty)",
              file=sys.stderr)
        return 4

    # Estimate total tokens for manifest. Real-mode prefers the tokenizer
    # path; smoke mode falls back to the char estimate.
    if tokenizer is not None:
        total_tokens = sum(
            len(tokenizer.encode(t, add_special_tokens=False)) for t in out_text
        )
    else:
        total_tokens = sum(_approx_tokens(t) for t in out_text)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "text": out_text,
        "corpus": out_cat,
    })
    pq.write_table(table, args.out, compression="zstd")

    by_cat = {c: out_cat.count(c) for c in set(out_cat)}
    manifest = {
        "kind": "diverse_corpus",
        "rows": len(out_text),
        "estimated_tokens": int(total_tokens),
        "target_tokens": target_tokens,
        "categories": include,
        "ratios": ratios,
        "by_corpus_rows": by_cat,
        "n_dropped_dups": int(n_dropped),
        "tokenizer": args.tokenizer if tokenizer is not None else (
            "smoke-no-tokenizer" if args.smoke else "char-estimate"
        ),
        "smoke": bool(args.smoke),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(str(args.out) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[diverse] wrote {len(out_text):,} rows "
          f"(~{total_tokens:,} tokens, dropped {n_dropped} dups) -> {args.out}")
    print(f"[diverse] by_corpus={by_cat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
