"""build_kd_prompts.py — assemble the curated KD prompt set.

Produces ``data/kd_prompts.jsonl`` (~10K prompts) covering 6 buckets:

  - wikipedia / encyclopedic facts ......... 3000  (English, fineweb_edu)
  - story / narrative openings ............. 2000  (English, fineweb_edu)
  - technical / how-to documentation ....... 2000  (English, fineweb_edu)
  - code prefixes .......................... 1000  (English, hand-templated)
  - Q&A turns (instruction prompts) ........ 1000  (mixed, alpaca_zh + en)
  - Chinese mix ............................ 1000  (Chinese, synth_zh + alpaca_zh)
                                            =====
                                            10000

Sources (rental-resident, no fresh download):
  - /workspace/data/fineweb_edu/000_00000.parquet  (726K rows, 'text' col, with
                                                    'score' int_score)
  - /workspace/data/synth_zh_phase1.parquet        (50K rows, 'text','title','topic')
  - /workspace/data/alpaca_zh/alpaca_zh.json       (49K rows, 'instruction','input','output')

Each output row is a JSON object:
    {"id": "kd-00001",
     "bucket": "wiki|story|techdoc|code|qa|zh",
     "prompt": "...",
     "src": "fineweb_edu|alpaca_zh|synth_zh|template",
     "lang": "en|zh"}

The prompt is a 32-128 token *continuation seed* (NOT a full instruction):
the teacher will continue from where it ends. For Q&A, the prompt is
``Q: {instruction}\\nA:`` so the teacher generates the answer.

This script runs on the RENTAL (where the source parquets live) but is
designed to be small (one file, no extra deps beyond pyarrow + json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover -- only run on rental
    print("[build_kd_prompts] pyarrow required", file=sys.stderr)
    raise

# ---- defaults --------------------------------------------------------------
FINEWEB_PATH = "/workspace/data/fineweb_edu/000_00000.parquet"
SYNTH_ZH_PATH = "/workspace/data/synth_zh_phase1.parquet"
ALPACA_ZH_PATH = "/workspace/data/alpaca_zh/alpaca_zh.json"

DEFAULT_OUT = "/workspace/data/kd_prompts.jsonl"

BUCKET_TARGETS = {
    "wiki": 3000,
    "story": 2000,
    "techdoc": 2000,
    "code": 1000,
    "qa": 1000,
    "zh": 1000,
}

# Heuristic keyword groups for English bucket routing on fineweb_edu rows.
# Keep the list short and high-precision so we can scan ~10K rows without
# pulling NLP deps.
WIKI_HINTS = re.compile(
    r"\b(was born|is a (city|country|species|river|mountain|king|queen|element|planet|gas)|"
    r"founded in \d{4}|capital of|located in|known as|named after|"
    r"is one of the|first discovered|in greek mythology)\b",
    re.IGNORECASE,
)
STORY_HINTS = re.compile(
    r"\b(once upon a time|she walked|he said|the door (opened|closed)|"
    r"the wind|her eyes|his face|stared at|smiled at|in the morning)\b",
    re.IGNORECASE,
)
TECHDOC_HINTS = re.compile(
    r"\b(how to|step \d|first you|in this tutorial|the following|"
    r"to install|configure|requires|the user|the system|by default)\b",
    re.IGNORECASE,
)


# ---- helpers ---------------------------------------------------------------
def _word_count(s: str) -> int:
    return len(s.split())


def _truncate_words(s: str, n: int) -> str:
    parts = s.split()
    return " ".join(parts[:n])


def _stable_id(prefix: str, payload: str) -> str:
    h = hashlib.md5(payload.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}-{h}"


def _build_code_prompts(n: int, rng: random.Random) -> list[dict]:
    """Hand-templated code prefixes — Python / JS / SQL / shell. Synthetic
    but realistic openings the teacher can complete.
    """
    py_topics = [
        "compute the n-th Fibonacci number iteratively",
        "merge two sorted lists",
        "count word frequencies in a text file",
        "implement binary search on a sorted list",
        "parse a JSON file and pretty-print it",
        "compute the Levenshtein distance between two strings",
        "shuffle a list using the Fisher-Yates algorithm",
        "find duplicates in a list",
        "compute the average of a list of floats",
        "remove all whitespace from a string",
        "check if a number is prime",
        "convert a temperature from Fahrenheit to Celsius",
        "fetch a URL and return the response body as text",
        "list all files in a directory recursively",
        "compute the dot product of two vectors",
    ]
    js_topics = [
        "debounce a function call",
        "fetch a JSON endpoint and log the result",
        "filter an array of objects by a predicate",
        "deep-clone an object",
        "validate an email address with a regex",
    ]
    sql_topics = [
        "find the top 5 customers by total order amount",
        "count rows per status across a table with WHERE created_at filter",
        "join orders to customers and select pending shipments",
    ]
    sh_topics = [
        "delete all .pyc files older than 7 days under /tmp",
        "tail the last 100 lines of every .log file in a directory",
        "find the largest 10 files under the home directory",
    ]
    out: list[dict] = []
    for topic in py_topics:
        out.append({
            "bucket": "code",
            "src": "template",
            "lang": "en",
            "prompt": f"# Python: {topic}.\ndef ",
        })
    for topic in js_topics:
        out.append({
            "bucket": "code",
            "src": "template",
            "lang": "en",
            "prompt": f"// JavaScript: {topic}.\nfunction ",
        })
    for topic in sql_topics:
        out.append({
            "bucket": "code",
            "src": "template",
            "lang": "en",
            "prompt": f"-- SQL: {topic}.\nSELECT ",
        })
    for topic in sh_topics:
        out.append({
            "bucket": "code",
            "src": "template",
            "lang": "en",
            "prompt": f"# shell: {topic}.\n#!/bin/bash\nset -euo pipefail\n",
        })

    # Diversify with light topic mutations until we hit n.
    base = list(out)
    while len(out) < n:
        topic = rng.choice(base)
        # Re-frame as either docstring opener or comment-then-impl.
        variant = rng.choice([
            "# Implement: " + topic["prompt"].split(":", 1)[1].strip().rstrip(".").rstrip("\ndef ").rstrip("\nfunction ").rstrip("\nSELECT ") + ".\n",
            topic["prompt"],
        ])
        out.append({
            "bucket": "code",
            "src": "template",
            "lang": "en",
            "prompt": variant,
        })
    return out[:n]


def _scan_fineweb(path: str, target: dict[str, int], rng: random.Random,
                  min_words: int = 60, max_words: int = 800) -> dict[str, list[dict]]:
    """Pull rows from fineweb_edu and route to wiki/story/techdoc buckets.

    Strategy: stream the parquet in batches; for each row text, take the
    first 32-96 words as the *prompt seed*. Route by regex hint: rows
    that match WIKI_HINTS go to wiki bucket, STORY_HINTS to story,
    TECHDOC_HINTS to techdoc; rows that match nothing fall back to a
    rotating bucket so we hit targets even when hints under-index.

    fineweb_edu has a 'score' float (0-5, edu quality from FineWeb-Edu
    classifier). We prefer rows with score >= 3.0 when available.
    """
    needed = {k: target[k] for k in ("wiki", "story", "techdoc") if k in target}
    out: dict[str, list[dict]] = {k: [] for k in needed}

    pf = pq.ParquetFile(path)
    fallback_keys = list(needed.keys())  # for round-robin overflow
    fallback_idx = 0

    seen_rows = 0
    for batch in pf.iter_batches(batch_size=2048, columns=["text", "score", "int_score"]):
        if all(len(out[k]) >= needed[k] for k in needed):
            break
        d = batch.to_pydict()
        texts = d.get("text") or []
        scores = d.get("score") or [None] * len(texts)
        # Prefer high-score rows -- iterate in shuffled order, keep the
        # high-score ones first by sorting per-batch.
        order = list(range(len(texts)))
        rng.shuffle(order)
        # Stable secondary sort by score desc (None treated as -1).
        order.sort(key=lambda i: (-(scores[i] if scores[i] is not None else -1.0)))
        for i in order:
            if all(len(out[k]) >= needed[k] for k in needed):
                break
            txt = texts[i]
            if not txt or _word_count(txt) < min_words:
                continue
            seen_rows += 1
            # Truncate to a continuation seed: 48-96 words randomly.
            n_words = rng.randint(48, 96)
            seed = _truncate_words(txt, n_words).strip()
            if not seed or len(seed) < 80:
                continue
            # Bucket route by regex on first 1500 chars for speed.
            head = txt[:1500]
            target_bucket = None
            if WIKI_HINTS.search(head) and len(out["wiki"]) < needed["wiki"]:
                target_bucket = "wiki"
            elif STORY_HINTS.search(head) and len(out["story"]) < needed["story"]:
                target_bucket = "story"
            elif TECHDOC_HINTS.search(head) and len(out["techdoc"]) < needed["techdoc"]:
                target_bucket = "techdoc"
            else:
                # Round-robin fallback into whichever bucket has room.
                for _ in range(len(fallback_keys)):
                    cand = fallback_keys[fallback_idx % len(fallback_keys)]
                    fallback_idx += 1
                    if len(out[cand]) < needed[cand]:
                        target_bucket = cand
                        break
            if target_bucket is None:
                continue
            out[target_bucket].append({
                "bucket": target_bucket,
                "src": "fineweb_edu",
                "lang": "en",
                "prompt": seed,
            })

    print(f"[fineweb] scanned ~{seen_rows} rows, "
          f"yielded wiki={len(out.get('wiki', []))} "
          f"story={len(out.get('story', []))} "
          f"techdoc={len(out.get('techdoc', []))}", flush=True)
    return out


def _build_qa_prompts(alpaca_path: str, n: int, rng: random.Random) -> list[dict]:
    """Use alpaca_zh's *English-translatable* style — but alpaca_zh is
    Chinese only. So we build n_en + n_zh from alpaca_zh as Q&A rows where
    half use Chinese prompt formatting and half use English-translated
    instruction shells. Simpler: take Chinese alpaca rows and format as
    bilingual Q: ... \\nA: prompts. Teacher generates answer.

    For genuinely English Q&A we synthesize from a curated bank.
    """
    out: list[dict] = []

    # Half English Q&A (synthesized from a curated bank to ensure quality).
    en_questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Explain the difference between TCP and UDP.",
        "What are the main causes of World War I?",
        "Define the term 'machine learning' in two sentences.",
        "Why is the sky blue?",
        "What is the difference between a list and a tuple in Python?",
        "How do you make a cup of tea?",
        "What is the speed of light?",
        "Describe the structure of a typical cell.",
        "Who wrote the play 'Hamlet'?",
        "What does HTTP stand for?",
        "Explain how a refrigerator works.",
        "What is the boiling point of water at sea level?",
        "List three benefits of regular exercise.",
        "What is the difference between weather and climate?",
        "How does a vaccine work?",
        "What is dark matter?",
        "What are the primary colors of light?",
        "Explain Newton's third law of motion.",
        "What is the largest desert in the world?",
        "How does a microwave oven heat food?",
        "What is GDP?",
        "Who painted the Mona Lisa?",
        "What is the chemical formula for water?",
        "What is the difference between an alligator and a crocodile?",
        "Why do leaves change color in autumn?",
        "What is artificial intelligence?",
        "How does an electric motor work?",
        "What does DNA stand for?",
    ]
    rng.shuffle(en_questions)
    n_en = max(1, n // 2)
    for q in (en_questions * (n_en // len(en_questions) + 1))[:n_en]:
        out.append({
            "bucket": "qa",
            "src": "template",
            "lang": "en",
            "prompt": f"Q: {q}\nA:",
        })

    # Other half from alpaca_zh translated style — we use the Chinese
    # instruction itself but it's already covered in the zh bucket. Use
    # a small English bank of how-to questions instead.
    howto = [
        "How do I bake a chocolate cake?",
        "How do I change a flat tire?",
        "How do I write a resume for a software engineer position?",
        "How do I improve my public speaking skills?",
        "How do I set up SSH key authentication?",
        "How do I learn a new language quickly?",
        "How do I deploy a Flask app to a Linux server?",
        "How do I file my taxes as a freelancer?",
        "How do I train a dog to sit?",
        "How do I plant tomatoes in my garden?",
    ]
    rng.shuffle(howto)
    n_howto = n - n_en
    for q in (howto * (n_howto // len(howto) + 1))[:n_howto]:
        out.append({
            "bucket": "qa",
            "src": "template",
            "lang": "en",
            "prompt": f"Q: {q}\nA: ",
        })

    return out[:n]


def _build_zh_prompts(synth_zh_path: str, alpaca_path: str, n: int,
                      rng: random.Random) -> list[dict]:
    """Chinese mix: half from synth_zh (encyclopedic continuations), half
    from alpaca_zh (Q&A turns).
    """
    out: list[dict] = []
    n_synth = n // 2
    n_alp = n - n_synth

    # Pull synth_zh: take 'text' field, truncate to ~60-120 chars (Chinese
    # is char-dense, ~2-3 tokens per char).
    pf = pq.ParquetFile(synth_zh_path)
    seen = 0
    for batch in pf.iter_batches(batch_size=512, columns=["text", "topic"]):
        if len(out) >= n_synth:
            break
        d = batch.to_pydict()
        texts = d.get("text") or []
        topics = d.get("topic") or [""] * len(texts)
        order = list(range(len(texts)))
        rng.shuffle(order)
        for i in order:
            if len(out) >= n_synth:
                break
            t = texts[i]
            if not t or len(t) < 40:
                continue
            seed_len = rng.randint(30, 80)
            seed = t[:seed_len].rstrip()
            if not seed:
                continue
            out.append({
                "bucket": "zh",
                "src": "synth_zh",
                "lang": "zh",
                "prompt": seed,
            })
            seen += 1

    # Pull alpaca_zh as Q: / A: pairs.
    with open(alpaca_path, "r", encoding="utf-8") as f:
        alp = json.load(f)
    rng.shuffle(alp)
    for row in alp:
        if len(out) - n_synth >= n_alp:
            break
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        if not instr or len(instr) > 200:
            continue
        if inp:
            prompt = f"问：{instr}\n输入：{inp}\n答："
        else:
            prompt = f"问：{instr}\n答："
        out.append({
            "bucket": "zh",
            "src": "alpaca_zh",
            "lang": "zh",
            "prompt": prompt,
        })

    return out[:n]


# ---- main ------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fineweb", default=FINEWEB_PATH)
    ap.add_argument("--synth-zh", default=SYNTH_ZH_PATH)
    ap.add_argument("--alpaca-zh", default=ALPACA_ZH_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--smoke", action="store_true",
                    help="Build a tiny 60-row set for quick local testing.")
    args = ap.parse_args(argv)

    if args.smoke:
        target = {k: max(5, v // 200) for k, v in BUCKET_TARGETS.items()}
    else:
        target = dict(BUCKET_TARGETS)

    rng = random.Random(args.seed)

    # Build buckets.
    print("[build_kd_prompts] building English fineweb buckets ...", flush=True)
    en_buckets = _scan_fineweb(args.fineweb, target, rng)

    print("[build_kd_prompts] building code bucket ...", flush=True)
    code = _build_code_prompts(target["code"], rng)

    print("[build_kd_prompts] building Q&A bucket ...", flush=True)
    qa = _build_qa_prompts(args.alpaca_zh, target["qa"], rng)

    print("[build_kd_prompts] building Chinese mix ...", flush=True)
    zh = _build_zh_prompts(args.synth_zh, args.alpaca_zh, target["zh"], rng)

    # Assemble + dedupe by prompt content.
    rows = []
    for k in ("wiki", "story", "techdoc"):
        rows.extend(en_buckets.get(k, []))
    rows.extend(code)
    rows.extend(qa)
    rows.extend(zh)

    seen: set[str] = set()
    final: list[dict] = []
    for r in rows:
        h = hashlib.md5(r["prompt"].encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        r["id"] = _stable_id("kd", r["prompt"])
        final.append(r)

    rng.shuffle(final)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in final:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    print(f"[build_kd_prompts] wrote {len(final)} prompts -> {args.out}")
    for k in ("wiki", "story", "techdoc", "code", "qa", "zh"):
        print(f"  {k:>8s}: {counts.get(k, 0):>5d}  (target {target.get(k, 0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
