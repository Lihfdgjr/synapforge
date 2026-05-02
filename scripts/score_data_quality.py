"""score_data_quality.py — quantitative quality score for a parquet corpus.

D2 of the 2026-05-02 quality-data push.

Why this script exists
----------------------
The trainer's 21000-step "the the the..." TOKEN_SOUP failure has two
suspected co-factors:

  1. PLIF dead — the spiking neurons never fire, so gradient flows only
     through the dense bypass and the model collapses to a unigram.
  2. Low-quality data — the trainer is replaying a SINGLE parquet shard
     of likely raw web crawl. Top-K token frequency is heavily skewed,
     so even a healthy model would learn just "the / 0 / 1 / ,".

This scorer attacks (2): given any parquet file with a string column,
compute six quality signals so we can OBJECTIVELY say "this corpus is
better than that corpus" and a quality gate can fail bad data BEFORE we
burn 4-6 GPU hours on a doomed run.

Metrics
-------
1. **Top-K token frequency**  — top-50 most common token IDs and their
   share of the corpus. Healthy corpora are Zipfian: top token is
   ~5-7%, top-5 is ~15-20%, top-50 is ~40-50%. The current
   ``synth_zh_phase1.parquet`` has top-1 > 25% and top-5 > 50%
   (signature of repeated boilerplate / single-source web crawl).
2. **Vocabulary coverage**  — fraction of the tokenizer vocab actually
   used (any non-zero count). Qwen 2.5 has 151643 live tokens; healthy
   pretrain hits >40% coverage on a 100M-token sample. < 30% means the
   corpus is too narrow.
3. **Avg / median / p95 sequence length** (in tokens). For our seq_len=
   256 trainer, median should be > 256 (so rows are long enough that the
   sliding window doesn't constantly cross EOT boundaries).
4. **Unique-token-per-100-tokens ratio**  — readability proxy. Low values
   (<40%) flag boilerplate ("the the the..." or repeated nav menus); high
   (>70%) flags hallucinated/random text. Healthy text sits 55-75%.
5. **Bigram repetition ratio**  — fraction of bigrams that recur within
   a 50-token sliding window. >5% in a 100-token sample is repetitive
   boilerplate. The current bad data fails this hard.
6. **Char-set distribution**  — alpha / digit / punct / CJK / whitespace
   ratios. Helps spot code-corpora-mislabeled-as-text and "all
   numbers" datasets.

Usage
-----
    # Score a single file
    python scripts/score_data_quality.py /path/to/file.parquet

    # Score multiple files via the audit JSON
    python scripts/score_data_quality.py --audit docs/RENTAL_DATA_AUDIT.json

    # Score a remote file via ssh (auto on if path starts with 'host:')
    python scripts/score_data_quality.py myserver:/home/liu/lnn-train/data/wiki_zh/train-00000.parquet

Output
------
A markdown table to stdout AND to ``docs/DATA_QUALITY_REPORT.md``. The
report has one section per scored file and a final "summary verdict"
table sorted by composite score.

The composite score (0..1, higher better) is a hand-tuned mean of:

    composite = 0.25 * (1 - top1_share)            # less skew = better
              + 0.20 * vocab_coverage              # broader = better
              + 0.20 * (1 - bigram_repeat)         # less repeat = better
              + 0.15 * unique_token_ratio          # closer to 0.65 = best
              + 0.10 * (median_len / 256)          # longer = better
              + 0.10 * (alpha_ratio if english else cjk_or_alpha)

The trainer's quality gate (D5 test) checks:

    top1_share < 0.10    AND  vocab_coverage > 0.30
    bigram_repeat < 0.05  AND  unique_token_ratio > 0.40

Sample budget
-------------
We sample up to ``--sample-rows 50000`` rows per file. At 1k tokens/row
average, that's ~50M tokens — more than enough to estimate Zipfian
slopes and bigram statistics with <1% error.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence


_QWEN_TOKENIZER_PATHS = [
    # Local rental teacher dir (production training path)
    "/workspace/teachers/qwen2.5-0.5b",
    # Hugging Face repo id for the lab + Windows dev fallback
    "Qwen/Qwen2.5-0.5B",
]

# Compile once for reuse
_RE_CJK = re.compile(r"[一-鿿㐀-䶿]")
_RE_ALPHA = re.compile(r"[A-Za-z]")
_RE_DIGIT = re.compile(r"\d")
_RE_PUNCT = re.compile(r"[\.,!?;:'\"()\[\]{}<>/\\\-_=+*@#$%^&|~`]")


def _load_tokenizer(name: str | None = None):
    """Load a Qwen tokenizer; fail gracefully so the script still runs
    char-level when transformers/HF is unavailable.

    Returns ``None`` to mean "no tokenizer; use char heuristic".
    """
    candidates = []
    if name:
        candidates.append(name)
    candidates.extend(_QWEN_TOKENIZER_PATHS)
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[score] transformers unavailable ({exc!r}); using char-level",
              file=sys.stderr)
        return None
    for c in candidates:
        try:
            return AutoTokenizer.from_pretrained(c, trust_remote_code=True)
        except Exception as exc:
            print(f"[score] tokenizer {c!r} unavailable: {exc!r}",
                  file=sys.stderr)
    return None


@dataclass
class FileScore:
    """One file's quality scorecard."""
    path: str
    n_rows_sampled: int = 0
    n_tokens_total: int = 0
    text_column: str | None = None
    error: str | None = None
    # Token stats
    top50_tokens: list[tuple[str, int, float]] = field(default_factory=list)
    top1_share: float = 0.0
    top5_share: float = 0.0
    top50_share: float = 0.0
    vocab_coverage: float = 0.0  # fraction of Qwen vocab seen
    # Sequence stats
    avg_seq_len: float = 0.0
    median_seq_len: float = 0.0
    p95_seq_len: float = 0.0
    # Readability
    unique_token_ratio: float = 0.0  # mean unique_tokens_per_100 / 100
    bigram_repeat_ratio: float = 0.0  # mean repeated bigrams in 50-tok window
    # Char-set distribution
    alpha_ratio: float = 0.0
    digit_ratio: float = 0.0
    punct_ratio: float = 0.0
    cjk_ratio: float = 0.0
    # Composite
    composite_score: float = 0.0
    pass_quality_gate: bool = False

    def dump(self) -> dict[str, Any]:
        return asdict(self)


def _iter_rows(path: str, text_col_hint: str | None,
               sample_rows: int) -> tuple[Iterable[str], str]:
    """Yield up to ``sample_rows`` text strings from a parquet/jsonl path.

    Returns ``(iterator, text_column_used)``. Tolerant of missing
    columns: returns an empty iterator if no string column is present.
    """
    low = path.lower()
    if low.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise RuntimeError(f"pyarrow required for parquet: {e!r}") from e
        pf = pq.ParquetFile(path)
        # Pick text column
        col = text_col_hint
        if col is None:
            for f in pf.schema_arrow:
                if str(f.type) == "string" and f.name in (
                        "text", "content", "raw_content", "document", "code"):
                    col = f.name
                    break
            if col is None:
                # First string column
                for f in pf.schema_arrow:
                    if str(f.type) == "string":
                        col = f.name
                        break
        if col is None:
            return iter([]), "(no string column)"

        def _gen() -> Iterable[str]:
            n = 0
            for batch in pf.iter_batches(batch_size=512, columns=[col]):
                for v in batch.column(col).to_pylist():
                    if not v:
                        continue
                    yield str(v)
                    n += 1
                    if n >= sample_rows:
                        return
        return _gen(), col

    if low.endswith(".jsonl") or low.endswith(".jsonl.gz"):
        opener = open
        if low.endswith(".gz"):
            import gzip
            opener = gzip.open  # type: ignore[assignment]

        def _gen_jsonl() -> Iterable[str]:
            n = 0
            with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # Common shapes: {"text": ...}, {"content": ...},
                    # {"messages": [{"role":..,"content":..}, ...]}
                    if "text" in obj and isinstance(obj["text"], str):
                        yield obj["text"]
                    elif "content" in obj and isinstance(obj["content"], str):
                        yield obj["content"]
                    elif "messages" in obj and isinstance(obj["messages"], list):
                        # Concatenate role+content; this matches how the
                        # SFT trainer renders chat for tokenization.
                        bits = []
                        for m in obj["messages"]:
                            if isinstance(m, dict):
                                role = m.get("role", "")
                                content = m.get("content", "")
                                if isinstance(content, str):
                                    bits.append(f"<|{role}|>{content}")
                        if bits:
                            yield "\n".join(bits)
                    n += 1
                    if n >= sample_rows:
                        return
        return _gen_jsonl(), "(jsonl)"
    return iter([]), "(unknown extension)"


def score_file(
    path: str,
    text_col_hint: str | None = None,
    sample_rows: int = 50000,
    tokenizer=None,
) -> FileScore:
    """Compute the full scorecard for one file."""
    fs = FileScore(path=path)
    try:
        rows_iter, col = _iter_rows(path, text_col_hint, sample_rows)
        fs.text_column = col
    except Exception as e:
        fs.error = repr(e)[:200]
        return fs

    # --- token frequency ---
    tok_freq: collections.Counter[int] = collections.Counter()
    seq_lens: list[int] = []
    unique_per_100: list[float] = []
    bigram_repeat_samples: list[float] = []
    char_alpha = char_digit = char_punct = char_cjk = char_total = 0
    n_rows = 0

    for txt in rows_iter:
        n_rows += 1
        # Char stats (fast & locale-aware enough)
        char_total += len(txt)
        char_alpha += sum(1 for c in txt if c.isalpha() and ord(c) < 128)
        char_digit += sum(1 for c in txt if c.isdigit())
        char_punct += len(_RE_PUNCT.findall(txt))
        char_cjk += len(_RE_CJK.findall(txt))

        # Tokenize. Qwen tokenizer dominates on bytes-per-call but we're
        # I/O-bound on parquet decode anyway.
        if tokenizer is not None:
            try:
                ids = tokenizer.encode(txt, add_special_tokens=False)
            except Exception:
                ids = []
        else:
            # Char-level fallback "tokens": split on whitespace + punct.
            ids_str = re.findall(r"\w+|[^\w\s]", txt)
            # Hash each str-token to int for the freq counter
            ids = [hash(s) & 0x3FFFFFFF for s in ids_str]
        if not ids:
            continue
        tok_freq.update(ids)
        seq_lens.append(len(ids))

        # Readability proxy: take 100-token chunks and count unique tokens
        for i in range(0, len(ids) - 100 + 1, 100):
            chunk = ids[i:i + 100]
            unique_per_100.append(len(set(chunk)) / len(chunk))
            if len(unique_per_100) >= 200:
                break

        # Bigram repetition: in each 50-token window, count what fraction
        # of bigrams already appeared earlier in that window.
        for i in range(0, len(ids) - 50 + 1, 50):
            chunk = ids[i:i + 50]
            seen: set[tuple[int, int]] = set()
            n_repeat = 0
            for j in range(len(chunk) - 1):
                bg = (chunk[j], chunk[j + 1])
                if bg in seen:
                    n_repeat += 1
                else:
                    seen.add(bg)
            bigram_repeat_samples.append(n_repeat / max(1, len(chunk) - 1))
            if len(bigram_repeat_samples) >= 200:
                break

    fs.n_rows_sampled = n_rows
    fs.n_tokens_total = sum(tok_freq.values())
    if fs.n_tokens_total == 0:
        fs.error = fs.error or "no tokens (empty corpus)"
        return fs

    # Top-K shares
    most = tok_freq.most_common(50)
    if tokenizer is not None:
        try:
            decoded = tokenizer.batch_decode([[t[0]] for t in most],
                                             skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
        except Exception:
            decoded = [str(t[0]) for t in most]
    else:
        decoded = [str(t[0]) for t in most]
    total = float(fs.n_tokens_total)
    fs.top50_tokens = [
        (decoded[i].replace("\n", "\\n")[:24], int(c), round(c / total, 4))
        for i, (_, c) in enumerate(most)
    ]
    fs.top1_share = round(most[0][1] / total, 4) if most else 0.0
    fs.top5_share = round(sum(c for _, c in most[:5]) / total, 4)
    fs.top50_share = round(sum(c for _, c in most[:50]) / total, 4)

    # Vocab coverage. Use 151643 (Qwen 2.5 live vocab) when tokenizer
    # is the Qwen one, else fall back to len(tok_freq) only.
    if tokenizer is not None:
        try:
            vocab_size = getattr(tokenizer, "vocab_size", None) or len(
                getattr(tokenizer, "get_vocab", lambda: {})()
            )
        except Exception:
            vocab_size = 151643
        vocab_size = max(int(vocab_size), 32000)
    else:
        vocab_size = max(len(tok_freq), 32000)
    fs.vocab_coverage = round(len(tok_freq) / vocab_size, 4)

    # Sequence stats
    if seq_lens:
        fs.avg_seq_len = round(statistics.mean(seq_lens), 1)
        fs.median_seq_len = round(statistics.median(seq_lens), 1)
        fs.p95_seq_len = round(
            sorted(seq_lens)[max(0, int(len(seq_lens) * 0.95) - 1)], 1
        )
    fs.unique_token_ratio = (
        round(statistics.mean(unique_per_100), 4) if unique_per_100 else 0.0
    )
    fs.bigram_repeat_ratio = (
        round(statistics.mean(bigram_repeat_samples), 4)
        if bigram_repeat_samples else 0.0
    )

    if char_total > 0:
        fs.alpha_ratio = round(char_alpha / char_total, 4)
        fs.digit_ratio = round(char_digit / char_total, 4)
        fs.punct_ratio = round(char_punct / char_total, 4)
        fs.cjk_ratio = round(char_cjk / char_total, 4)

    # Composite score (see module docstring for formula).
    is_zh = fs.cjk_ratio > 0.10
    char_signal = fs.cjk_ratio if is_zh else fs.alpha_ratio
    median_norm = min(1.0, fs.median_seq_len / 256.0)
    # Sweet-spot for unique_token_ratio is 0.55-0.75 -- penalize both
    # the low end (boilerplate) and very high end (random/hallucinated).
    if fs.unique_token_ratio <= 0:
        ut_score = 0.0
    elif fs.unique_token_ratio < 0.40:
        ut_score = fs.unique_token_ratio / 0.40 * 0.7
    elif fs.unique_token_ratio < 0.75:
        ut_score = 1.0
    else:
        ut_score = max(0.0, 1.0 - (fs.unique_token_ratio - 0.75) * 2.0)
    fs.composite_score = round(
        0.25 * (1.0 - fs.top1_share)
        + 0.20 * min(1.0, fs.vocab_coverage / 0.40)
        + 0.20 * (1.0 - min(1.0, fs.bigram_repeat_ratio * 20))
        + 0.15 * ut_score
        + 0.10 * median_norm
        + 0.10 * char_signal,
        4,
    )

    fs.pass_quality_gate = bool(
        fs.top1_share < 0.10
        and fs.vocab_coverage > 0.30
        and fs.bigram_repeat_ratio < 0.05
        and fs.unique_token_ratio > 0.40
    )
    return fs


def _format_md_table(scores: list[FileScore]) -> str:
    """Build the Markdown summary table sorted by composite_score desc."""
    lines: list[str] = []
    lines.append("| File | rows | tokens | top1 | top5 | vocab_cov | "
                 "uniq/100 | bigram_rep | median_len | composite | gate |")
    lines.append("|------|------|--------|------|------|-----------|"
                 "----------|------------|------------|-----------|------|")
    for s in sorted(scores, key=lambda s: s.composite_score, reverse=True):
        if s.error:
            lines.append(
                f"| `{os.path.basename(s.path)}` | (err: {s.error[:40]}) | "
                "0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 | FAIL |"
            )
            continue
        lines.append(
            f"| `{os.path.basename(s.path)}` "
            f"| {s.n_rows_sampled:,} "
            f"| {s.n_tokens_total:,} "
            f"| {s.top1_share:.3f} "
            f"| {s.top5_share:.3f} "
            f"| {s.vocab_coverage:.3f} "
            f"| {s.unique_token_ratio:.3f} "
            f"| {s.bigram_repeat_ratio:.4f} "
            f"| {int(s.median_seq_len)} "
            f"| {s.composite_score:.3f} "
            f"| {'PASS' if s.pass_quality_gate else 'FAIL'} |"
        )
    return "\n".join(lines)


def _md_per_file_section(s: FileScore) -> str:
    """Detailed Markdown per-file block."""
    out: list[str] = []
    out.append(f"### `{s.path}`")
    if s.error:
        out.append(f"\n**ERROR**: {s.error}\n")
        return "\n".join(out)
    out.append(f"\n- text_column: `{s.text_column}`")
    out.append(f"- rows_sampled: {s.n_rows_sampled:,}")
    out.append(f"- tokens_total: {s.n_tokens_total:,}")
    out.append(f"- composite_score: **{s.composite_score:.3f}**  "
               f"(quality_gate: **{'PASS' if s.pass_quality_gate else 'FAIL'}**)")
    out.append(f"\n**Token frequency**")
    out.append(f"- top-1 token share: {s.top1_share:.4f}  "
               f"(target < 0.10)")
    out.append(f"- top-5 token share: {s.top5_share:.4f}")
    out.append(f"- top-50 token share: {s.top50_share:.4f}")
    out.append(f"- vocab coverage: {s.vocab_coverage:.4f}  (target > 0.30)")
    out.append(f"- top-10 tokens: " + ", ".join(
        f"`{tok}`={cnt:,} ({pct:.2%})" for tok, cnt, pct in s.top50_tokens[:10]
    ))
    out.append(f"\n**Sequence stats**")
    out.append(f"- avg seq_len: {s.avg_seq_len}, median: {s.median_seq_len}, "
               f"p95: {s.p95_seq_len}")
    out.append(f"\n**Readability**")
    out.append(f"- unique-token-per-100 ratio: {s.unique_token_ratio:.4f}  "
               f"(target 0.40 < x < 0.75; sweet spot 0.55-0.65)")
    out.append(f"- bigram repetition (50-tok windows): "
               f"{s.bigram_repeat_ratio:.4f}  (target < 0.05)")
    out.append(f"\n**Char distribution**")
    out.append(f"- alpha (ascii): {s.alpha_ratio:.3f}  "
               f"digit: {s.digit_ratio:.3f}  "
               f"punct: {s.punct_ratio:.3f}  "
               f"CJK: {s.cjk_ratio:.3f}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*",
                    help="Parquet/jsonl paths to score (local or 'host:/path').")
    ap.add_argument("--audit", default=None,
                    help="Path to RENTAL_DATA_AUDIT.json; auto-include all "
                         "parquet files with text columns from there.")
    ap.add_argument("--out",
                    default="docs/DATA_QUALITY_REPORT.md",
                    help="Markdown output path.")
    ap.add_argument("--out-json",
                    default="docs/DATA_QUALITY_REPORT.json",
                    help="JSON sidecar with raw scores.")
    ap.add_argument("--sample-rows", type=int, default=20000,
                    help="Max rows to sample per file (50k = ~50M tokens).")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer path/id (default: try Qwen 2.5 0.5B).")
    ap.add_argument("--max-files", type=int, default=20,
                    help="Cap files-from-audit at N (skip multi-shard noise).")
    args = ap.parse_args(argv)

    paths: list[tuple[str, str | None]] = []
    for p in args.paths:
        paths.append((p, None))
    if args.audit:
        try:
            m = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[score] could not load audit {args.audit!r}: {e}",
                  file=sys.stderr)
            return 2
        n_added = 0
        for h in m.get("hosts", []):
            host_name = h.get("host", "?")
            for e in h.get("entries", []):
                if e.get("kind") != "parquet":
                    continue
                path = e["path"]
                col = e.get("text_column")
                if not col or col in ("video_id", "filepath", "blob_id"):
                    # Skip non-content columns auto-detected by audit.
                    continue
                # Skip very small (size < 1MB) files to reduce noise.
                if int(e.get("size_bytes", 0)) < 1_000_000:
                    continue
                # Prefix with host name so the scorer tries SSH if needed.
                tag_path = (
                    f"{host_name}:{path}" if host_name not in (None, "local")
                    else path
                )
                paths.append((tag_path, col))
                n_added += 1
                if n_added >= args.max_files:
                    break
            if n_added >= args.max_files:
                break

    if not paths:
        print("[score] no paths to score; pass paths or --audit", file=sys.stderr)
        return 3

    tokenizer = _load_tokenizer(args.tokenizer)
    if tokenizer is None:
        print("[score] WARNING: no tokenizer; falling back to char-level "
              "frequency. Top-K and vocab metrics will be regex-token based.",
              file=sys.stderr)

    scores: list[FileScore] = []
    print(f"[score] scoring {len(paths)} files...", file=sys.stderr)
    for path, col in paths:
        # Remote path? "host:/...." — fetch via ssh+pyarrow on remote, then
        # download the score, NOT the data. We just run the scorer remotely.
        if ":" in path and not path[1:3].lstrip().startswith("\\"):
            host, _, rpath = path.partition(":")
            print(f"[score] {host}:{rpath} ...", file=sys.stderr)
            scores.append(_score_remote(host, rpath, col, args))
        else:
            t0 = time.time()
            print(f"[score] {path} ...", file=sys.stderr)
            fs = score_file(path, text_col_hint=col,
                            sample_rows=args.sample_rows,
                            tokenizer=tokenizer)
            dt = time.time() - t0
            verdict = "PASS" if fs.pass_quality_gate else "FAIL"
            print(f"[score]   composite={fs.composite_score:.3f} "
                  f"top1={fs.top1_share:.3f} cov={fs.vocab_coverage:.3f} "
                  f"gate={verdict} ({dt:.1f}s)",
                  file=sys.stderr)
            scores.append(fs)

    # Write outputs
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Data Quality Report\n\n")
        f.write(f"_Generated UTC: "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}_\n\n")
        f.write(
            "Token-soup root-cause analysis 2026-05-02. Higher composite "
            "score = better. The trainer's quality gate **PASSes** when "
            "all four hard thresholds are met:\n\n"
            "    top1_share  < 0.10\n"
            "    vocab_cov   > 0.30\n"
            "    bigram_rep  < 0.05\n"
            "    uniq/100    > 0.40\n\n"
        )
        f.write("## Summary table (sorted by composite, descending)\n\n")
        f.write(_format_md_table(scores))
        f.write("\n\n## Per-file detail\n\n")
        for s in sorted(scores, key=lambda s: s.composite_score, reverse=True):
            f.write(_md_per_file_section(s) + "\n\n")
    out_json = Path(args.out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([s.dump() for s in scores], f, indent=2, ensure_ascii=False)

    # Console one-liner summary
    n_pass = sum(1 for s in scores if s.pass_quality_gate)
    print(f"\n[score] wrote {out_md}", file=sys.stderr)
    print(f"[score] PASS={n_pass}/{len(scores)} (quality_gate)",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Remote scoring — ship the inline scorer to the host, parse json back.
# ---------------------------------------------------------------------------
_REMOTE_SCORE_PY = r"""
import json, os, sys, statistics, collections, re, time

# Force-load this module's own scorer body. We re-define just the
# numeric path so we don't depend on uploading the file.
def char_alpha(c): return c.isalpha() and ord(c) < 128
RE_CJK = re.compile("[一-鿿㐀-䶿]")
RE_PUNCT = re.compile(r"[\.,!?;:'\"()\[\]{}<>/\\\-_=+*@#$%^&|~`]")

def score(path, text_col_hint, sample_rows, tokenizer_path):
    out = {"path": path, "n_rows_sampled": 0, "n_tokens_total": 0,
           "text_column": None, "error": None}
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        out["error"] = "pyarrow not available: " + repr(e)[:200]
        return out
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        tok = None
        out["tokenizer_warn"] = repr(e)[:120]

    if path.lower().endswith(".jsonl"):
        # JSONL fallback (rare on remote)
        n = 0
        rows = []
        with open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "text" in obj: rows.append(obj["text"])
                    elif "messages" in obj:
                        rows.append("\n".join(
                            str(m.get("content","")) for m in obj["messages"] if isinstance(m, dict)))
                except Exception: pass
                n += 1
                if n >= sample_rows: break
        col = "(jsonl)"
        rows_iter = iter(rows)
    else:
        pf = pq.ParquetFile(path)
        col = text_col_hint
        if col is None:
            for f in pf.schema_arrow:
                if str(f.type) == "string" and f.name in ("text","content","raw_content","document","code"):
                    col = f.name; break
            if col is None:
                for f in pf.schema_arrow:
                    if str(f.type) == "string":
                        col = f.name; break
        if col is None:
            out["error"] = "no string column"
            return out
        def _gen():
            n = 0
            for batch in pf.iter_batches(batch_size=512, columns=[col]):
                for v in batch.column(col).to_pylist():
                    if not v: continue
                    yield str(v); n += 1
                    if n >= sample_rows: return
        rows_iter = _gen()

    out["text_column"] = col

    tok_freq = collections.Counter()
    seq_lens = []
    unique_per_100 = []
    bigram_repeat = []
    char_alpha_n = char_digit = char_punct = char_cjk = char_total = 0
    n_rows = 0
    for txt in rows_iter:
        n_rows += 1
        char_total += len(txt)
        char_alpha_n += sum(1 for c in txt if c.isalpha() and ord(c) < 128)
        char_digit += sum(1 for c in txt if c.isdigit())
        char_punct += len(RE_PUNCT.findall(txt))
        char_cjk += len(RE_CJK.findall(txt))
        if tok is not None:
            try: ids = tok.encode(txt, add_special_tokens=False)
            except Exception: ids = []
        else:
            ids = [hash(s)&0x3FFFFFFF for s in re.findall(r"\w+|[^\w\s]", txt)]
        if not ids: continue
        tok_freq.update(ids)
        seq_lens.append(len(ids))
        for i in range(0, len(ids)-100+1, 100):
            chunk = ids[i:i+100]
            unique_per_100.append(len(set(chunk))/len(chunk))
            if len(unique_per_100) >= 200: break
        for i in range(0, len(ids)-50+1, 50):
            chunk = ids[i:i+50]
            seen = set(); n_rep = 0
            for j in range(len(chunk)-1):
                bg = (chunk[j], chunk[j+1])
                if bg in seen: n_rep += 1
                else: seen.add(bg)
            bigram_repeat.append(n_rep/max(1, len(chunk)-1))
            if len(bigram_repeat) >= 200: break

    out["n_rows_sampled"] = n_rows
    out["n_tokens_total"] = sum(tok_freq.values())
    if out["n_tokens_total"] == 0:
        out["error"] = out.get("error") or "no tokens"
        return out

    most = tok_freq.most_common(50)
    if tok is not None:
        try:
            decoded = tok.batch_decode([[t[0]] for t in most], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            decoded = [str(t[0]) for t in most]
    else:
        decoded = [str(t[0]) for t in most]
    total = float(out["n_tokens_total"])
    out["top50_tokens"] = [(decoded[i].replace("\n","\\n")[:24], int(c), round(c/total,4))
                          for i,(_,c) in enumerate(most)]
    out["top1_share"] = round(most[0][1]/total, 4) if most else 0.0
    out["top5_share"] = round(sum(c for _,c in most[:5])/total, 4)
    out["top50_share"] = round(sum(c for _,c in most[:50])/total, 4)
    if tok is not None:
        try:
            vsize = getattr(tok, "vocab_size", None) or len(tok.get_vocab())
        except Exception:
            vsize = 151643
        vsize = max(int(vsize), 32000)
    else:
        vsize = max(len(tok_freq), 32000)
    out["vocab_coverage"] = round(len(tok_freq)/vsize, 4)
    out["avg_seq_len"] = round(statistics.mean(seq_lens), 1) if seq_lens else 0
    out["median_seq_len"] = round(statistics.median(seq_lens), 1) if seq_lens else 0
    out["p95_seq_len"] = round(sorted(seq_lens)[max(0,int(len(seq_lens)*0.95)-1)], 1) if seq_lens else 0
    out["unique_token_ratio"] = round(statistics.mean(unique_per_100), 4) if unique_per_100 else 0
    out["bigram_repeat_ratio"] = round(statistics.mean(bigram_repeat), 4) if bigram_repeat else 0
    out["alpha_ratio"] = round(char_alpha_n/char_total, 4) if char_total else 0
    out["digit_ratio"] = round(char_digit/char_total, 4) if char_total else 0
    out["punct_ratio"] = round(char_punct/char_total, 4) if char_total else 0
    out["cjk_ratio"] = round(char_cjk/char_total, 4) if char_total else 0

    is_zh = out["cjk_ratio"] > 0.10
    char_signal = out["cjk_ratio"] if is_zh else out["alpha_ratio"]
    median_norm = min(1.0, out["median_seq_len"]/256.0)
    if out["unique_token_ratio"] <= 0:
        ut_score = 0.0
    elif out["unique_token_ratio"] < 0.40:
        ut_score = out["unique_token_ratio"]/0.40 * 0.7
    elif out["unique_token_ratio"] < 0.75:
        ut_score = 1.0
    else:
        ut_score = max(0.0, 1.0-(out["unique_token_ratio"]-0.75)*2.0)
    out["composite_score"] = round(
        0.25*(1.0-out["top1_share"])
        + 0.20*min(1.0, out["vocab_coverage"]/0.40)
        + 0.20*(1.0-min(1.0, out["bigram_repeat_ratio"]*20))
        + 0.15*ut_score
        + 0.10*median_norm
        + 0.10*char_signal,
        4)
    out["pass_quality_gate"] = bool(
        out["top1_share"]<0.10 and out["vocab_coverage"]>0.30
        and out["bigram_repeat_ratio"]<0.05 and out["unique_token_ratio"]>0.40
    )
    return out

if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    res = score(args["path"], args.get("col"), int(args.get("sample_rows", 20000)), args["tokenizer"])
    print("__JSON_BEGIN__")
    print(json.dumps(res, ensure_ascii=False, default=str))
    print("__JSON_END__")
"""


def _score_remote(host: str, rpath: str, col: str | None, args) -> FileScore:
    """SSH to host, ship the inline scorer + args via stdin to `python3 -`.

    We embed the path/col/sample_rows/tokenizer as a top-of-script
    assignment block so the script doesn't have to parse stdin a second
    time (which is what kept breaking with the heredoc approach). The
    `python3 -` interpreter reads the entire stdin and execs it, so any
    encoding shenanigans in shell quoting are bypassed.
    """
    if host == "myserver":
        py = "/home/liu/lnn-train/.venv/bin/python3"
    else:
        py = "/usr/bin/python3"

    args_block = (
        "_PATH = " + repr(rpath) + "\n"
        + "_COL = " + repr(col) + "\n"
        + "_SAMPLE = " + str(int(args.sample_rows)) + "\n"
        + "_TOK = " + repr(args.tokenizer or _QWEN_TOKENIZER_PATHS[0]) + "\n"
    )
    # Replace the __main__ block in REMOTE_SCORE_PY: drop the stdin parse
    # and call score() directly with the embedded args.
    body = _REMOTE_SCORE_PY.replace(
        'if __name__ == "__main__":\n'
        '    args = json.loads(sys.stdin.read())\n'
        '    res = score(args["path"], args.get("col"), int(args.get("sample_rows", 20000)), args["tokenizer"])\n'
        '    print("__JSON_BEGIN__")\n'
        '    print(json.dumps(res, ensure_ascii=False, default=str))\n'
        '    print("__JSON_END__")',
        'if __name__ == "__main__":\n'
        '    res = score(_PATH, _COL, _SAMPLE, _TOK)\n'
        '    print("__JSON_BEGIN__")\n'
        '    print(json.dumps(res, ensure_ascii=False, default=str))\n'
        '    print("__JSON_END__")'
    )
    full = args_block + body

    cmd = ["ssh", host, f"{py} -"]
    try:
        proc = subprocess.run(
            cmd, input=full, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return FileScore(path=f"{host}:{rpath}", error="ssh score timeout")
    if proc.returncode != 0:
        return FileScore(
            path=f"{host}:{rpath}",
            error=f"ssh fail {proc.returncode}: " + (proc.stderr or "")[:300],
        )
    body_out = proc.stdout
    if "__JSON_BEGIN__" not in body_out or "__JSON_END__" not in body_out:
        return FileScore(
            path=f"{host}:{rpath}",
            error=f"no delim in remote output: tail={body_out[-200:]!r}",
        )
    j = body_out.split("__JSON_BEGIN__", 1)[1].split("__JSON_END__", 1)[0].strip()
    try:
        d = json.loads(j)
    except json.JSONDecodeError as e:
        return FileScore(path=f"{host}:{rpath}",
                         error=f"json parse: {e}")
    fs = FileScore(path=f"{host}:{rpath}")
    for k, v in d.items():
        if hasattr(fs, k):
            try:
                setattr(fs, k, v)
            except Exception:
                pass
    return fs


if __name__ == "__main__":
    sys.exit(main())
