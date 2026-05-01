"""tokenize_wikitext103.py -- pre-tokenize wikitext-103 with Qwen vocab (T3.7).

Locates the wikitext-103 raw train splits on the rental, concatenates them,
runs the Qwen 2.5 0.5B tokenizer with ``max_length=512`` + ``truncation=True``
and emits BOTH a Python pickle (list of int32 token IDs per chunk) AND a
single-column ``input_ids`` parquet for direct ``ParquetTokenStream``
ingestion in ``train_100m_kd.py``.

Earlier "wt103 files: 0" bug
---------------------------
The rental's wikitext-103 archive has lived under three different layouts
across data refreshes. We probe in priority order::

    1. /workspace/data/wikitext-103/*.txt
    2. /workspace/data/wikitext-103/wt103_raw/*.txt
    3. /workspace/data/wt103/*.txt

If none of the three glob patterns match, the script exits with a clear
error (``[tokenize_wt103] FATAL: no wikitext-103 source files found``) and
returns code ``1``. Use ``--smoke`` to bypass the FS scan entirely and emit
100 mock token sequences -- handy for unit tests + CI runs without the HF
``transformers`` package installed.

Output schema
-------------
* ``--output-pkl``      list[list[int]]  (int32-castable)  via pickle.dump
* ``--output-parquet``  one column ``input_ids: list<int32>``

CLI
---
    python scripts/tokenize_wikitext103.py
    python scripts/tokenize_wikitext103.py --smoke   # 100 mock seqs
    python scripts/tokenize_wikitext103.py \\
        --input-glob '/workspace/data/wikitext-103/*.txt' \\
        --output-pkl /workspace/data/wt103_qwen_tokens.pkl \\
        --output-parquet /workspace/data/wt103_qwen_tokens.parquet \\
        --max-length 512

Notes
-----
* Tokenizer load is lazy; if ``transformers`` isn't installed the script
  falls back to a tiny deterministic mock tokenizer (each text chunk maps
  to ``[hash(chunk) % vocab] * len(chunk)``-style IDs). This is ONLY used
  when ``--smoke`` is set OR when transformers import fails AND the smoke
  flag is set; non-smoke runs without transformers exit ``2``.
* Concatenation happens *before* tokenization so a single doc that spans
  several lines becomes one long stream the trainer can window-slide
  across at runtime.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
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


# --- candidate input layouts (ordered) ---------------------------------------
DEFAULT_INPUT_GLOBS: tuple = (
    "/workspace/data/wikitext-103/*.txt",
    "/workspace/data/wikitext-103/wt103_raw/*.txt",
    "/workspace/data/wt103/*.txt",
)


# --- file discovery ----------------------------------------------------------
def find_input_files(globs: Sequence[str]) -> List[str]:
    """Return the first glob's matches, or the union if multiple supplied.

    When the caller passes multiple ``--input-glob`` values explicitly we
    union the matches (caller-driven). When using the default candidate
    list we short-circuit on the first non-empty match (priority order).
    """
    if len(globs) == 1:
        return sorted(glob.glob(globs[0]))
    # Multi-pattern: try each in order; first non-empty wins.
    for pat in globs:
        found = sorted(glob.glob(pat))
        if found:
            return found
    return []


def read_text_concat(paths: Sequence[str]) -> str:
    """Concatenate UTF-8 text from every path with a single ``\\n\\n`` joiner."""
    chunks: List[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            chunks.append(f.read())
    return "\n\n".join(chunks)


# --- tokenizer (lazy) --------------------------------------------------------
def _load_qwen_tokenizer(name: str = "Qwen/Qwen2.5-0.5B"):  # pragma: no cover
    """Load Qwen 2.5 tokenizer, preferring local rental cache first."""
    from transformers import AutoTokenizer
    candidates = [
        "/workspace/teachers/qwen2.5-0.5b",
        name,
    ]
    last = None
    for path in candidates:
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"could not load Qwen tokenizer: {last!r}")


class _MockTokenizer:
    """Deterministic mock used by ``--smoke`` and when ``transformers`` missing.

    ``encode(text, ...)`` returns a list of ints in ``[0, 151643)``; the
    same text always gives the same IDs (collision-cheap).
    """

    vocab_size = 151643

    def encode(self, text: str, add_special_tokens: bool = False,
               max_length: Optional[int] = None,
               truncation: bool = False) -> List[int]:
        # Each character becomes one token id (mod vocab) so different
        # prompts produce different lengths, matching real tokenizer
        # behaviour on length-sensitive smoke tests.
        ids = [(ord(c) * 7919 + 1009) % self.vocab_size for c in text]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids


def _chunk_into_sequences(token_ids: Sequence[int],
                          max_length: int) -> List[List[int]]:
    """Slice the flat token stream into ``max_length``-sized chunks.

    The trainer wants an iterable of fixed-length samples (``max_length``
    is the seq_len passed to PyTorch). The last partial window is kept as
    long as it has at least 8 tokens -- shorter tails are dropped to
    avoid noisy gradient on near-empty windows.
    """
    out: List[List[int]] = []
    n = len(token_ids)
    for start in range(0, n, max_length):
        end = min(start + max_length, n)
        if end - start < 8:
            continue
        out.append(list(token_ids[start:end]))
    return out


# --- output writers ----------------------------------------------------------
def write_pkl(path: str, sequences: Sequence[Sequence[int]]) -> int:
    """Write list-of-list[int] to a Python pickle. Returns row count."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = [list(map(int, s)) for s in sequences]
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(payload)


def write_parquet(path: str, sequences: Sequence[Sequence[int]]) -> int:
    """Write a single ``input_ids`` column parquet (list<int32>)."""
    if not _HAVE_ARROW:
        raise RuntimeError("pyarrow required for parquet output")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [list(map(int, s)) for s in sequences]
    table = pa.table({"input_ids": rows})
    pq.write_table(table, path, compression="zstd")
    # Companion manifest, mirroring synth_chinese_pretrain + collect_kd_data.
    with open(str(path) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "wikitext103_qwen_tokenized",
                "rows": len(rows),
                "tokenizer": "Qwen/Qwen2.5-0.5B",
                "max_length": max(len(r) for r in rows) if rows else 0,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
        )
    return len(rows)


# --- smoke generator ---------------------------------------------------------
def _smoke_sequences(n: int = 100, max_length: int = 512) -> List[List[int]]:
    """Emit N deterministic mock token sequences for unit tests."""
    tok = _MockTokenizer()
    out: List[List[int]] = []
    for i in range(n):
        # Each "doc" is a short sentence with repeating structure so the
        # mock tokenizer's char->id mapping stays bounded but distinct.
        text = f"wikitext mock line {i} -- {'lorem ipsum ' * (i % 7 + 1)}"
        ids = tok.encode(text, max_length=max_length, truncation=True)
        # Pad short ones up to a reasonable size so chunking has work to do
        # but stay under max_length.
        while len(ids) < 32:
            ids.append((len(ids) * 31) % tok.vocab_size)
        out.append(ids[:max_length])
    return out


# --- main --------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--input-glob",
        action="append",
        default=None,
        help="Glob for raw .txt files (repeat flag for multiple). "
             "Default: probe rental's three known layouts in order.",
    )
    ap.add_argument(
        "--output-pkl",
        default="/workspace/data/wt103_qwen_tokens.pkl",
        help="Output Python pickle path",
    )
    ap.add_argument(
        "--output-parquet",
        default="/workspace/data/wt103_qwen_tokens.parquet",
        help="Output parquet path (one column 'input_ids')",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max_length + chunk size (default 512)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Skip FS scan + Qwen download; emit 100 mock token sequences",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        sequences = _smoke_sequences(n=100, max_length=args.max_length)
        n_pkl = write_pkl(args.output_pkl, sequences)
        n_parquet = write_parquet(args.output_parquet, sequences) \
            if _HAVE_ARROW else 0
        print(
            f"[tokenize_wt103] smoke: wrote {n_pkl} sequences -> "
            f"{args.output_pkl} (parquet={n_parquet})",
            flush=True,
        )
        return 0

    # Real run: locate files via globs.
    globs = args.input_glob if args.input_glob else list(DEFAULT_INPUT_GLOBS)
    files = find_input_files(globs)
    if not files:
        # Earlier "wt103 files: 0" bug: clear error + actionable.
        print(
            "[tokenize_wt103] FATAL: no wikitext-103 source files found.\n"
            f"  searched globs: {globs}\n"
            "  fix: download wikitext-103 raw to one of the candidate paths,\n"
            "       or pass --input-glob explicitly.\n"
            "  (use --smoke for a CI-friendly mock run.)",
            file=sys.stderr,
        )
        return 1

    print(
        f"[tokenize_wt103] found {len(files)} input file(s) "
        f"(first: {files[0]}); concatenating + tokenizing...",
        flush=True,
    )
    raw_text = read_text_concat(files)
    print(
        f"[tokenize_wt103] {len(raw_text):,} characters concatenated; "
        f"loading Qwen tokenizer...",
        flush=True,
    )
    try:  # pragma: no cover -- depends on host transformers
        tokenizer = _load_qwen_tokenizer()
    except Exception as exc:
        print(
            f"[tokenize_wt103] FATAL: tokenizer load failed: {exc!r}.\n"
            "  fix: pip install transformers, OR pre-cache "
            "/workspace/teachers/qwen2.5-0.5b, OR use --smoke",
            file=sys.stderr,
        )
        return 2

    flat_ids = tokenizer.encode(  # pragma: no cover
        raw_text,
        add_special_tokens=False,
    )
    sequences = _chunk_into_sequences(flat_ids, args.max_length)

    n_pkl = write_pkl(args.output_pkl, sequences)
    n_parquet = write_parquet(args.output_parquet, sequences) \
        if _HAVE_ARROW else 0
    total_tokens = sum(len(s) for s in sequences)
    print(
        f"[tokenize_wt103] wrote {n_pkl} sequences "
        f"({total_tokens:,} tokens, max_length={args.max_length}) -> "
        f"{args.output_pkl} + parquet={n_parquet}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
