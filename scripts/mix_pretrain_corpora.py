"""mix_pretrain_corpora.py — combine downloaded pretrain corpora into one file.

Reads `<root>/<corpus>/train.parquet` for each corpus, mixes by ratio, applies
sha256-based dedup + length filter + light language consistency check, writes
a single `pretrain_mix.parquet` ready for the trainer.

Default mix (ratio sums to 1.0):
    fineweb_edu          0.35   (English web)
    wudao + skypile      0.35   (Chinese web)         — 0.20 + 0.15
    the_stack_v2_python  0.20   (Code)
    cosmopedia_v2        0.10   (Synth educational)

Quality filter:
    min_chars=50, max_tokens=8192 (estimated 4 chars/token)
    drop blank, all-punctuation, dedup by sha256(text[:4096])
    optional language id (regex char-set ratio: zh / en / mixed)

Usage:
    python scripts/mix_pretrain_corpora.py \\
        --root /workspace/data/pretrain \\
        --out  /workspace/data/pretrain/pretrain_mix.parquet \\
        --target-rows 200000

    python scripts/mix_pretrain_corpora.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

# pyarrow + pandas are stdlib for the rental but optional here so --help works
# in CI without those wheels installed.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


DEFAULT_RATIOS = {
    "fineweb_edu": 0.35,
    "wudao": 0.20,
    "skypile": 0.15,
    "the_stack_v2_python": 0.20,
    "cosmopedia_v2": 0.10,
    # cci3_hq is optional; if present it absorbs from wudao share
}

# Heuristic language id: ratio of CJK characters per row.
def _lang_id(text: str) -> str:
    if not text:
        return "empty"
    n = 0
    for ch in text[:1000]:
        if "一" <= ch <= "鿿":
            n += 1
    ratio = n / max(1, min(1000, len(text)))
    if ratio > 0.30:
        return "zh"
    if ratio < 0.02:
        return "en"
    return "mixed"


def _quality_ok(text: str, min_chars: int, max_chars: int) -> bool:
    if not text or not text.strip():
        return False
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    # Reject all-punctuation / very low alpha ratio.
    alnum = sum(1 for c in text if c.isalnum())
    if alnum / max(1, n) < 0.20:
        return False
    return True


def _hash_key(text: str) -> str:
    h = hashlib.sha256(text[:4096].encode("utf-8", errors="ignore")).hexdigest()
    return h[:32]


def _read_corpus_rows(corpus_dir: Path, text_col_candidates: list[str]) -> list[str]:
    """Read text column from a corpus directory's train.parquet.

    Falls back to JSONL if parquet absent. Returns list of strings (no rows
    parsed/tokenized — that is the trainer's job).
    """
    if not _HAVE_ARROW:
        return []
    p = corpus_dir / "train.parquet"
    if p.exists():
        try:
            tbl = pq.read_table(p)
            # find a text column
            for col in text_col_candidates + tbl.column_names:
                if col in tbl.column_names:
                    return [str(x) for x in tbl[col].to_pylist() if x is not None]
        except Exception as e:
            print(f"[mix] parquet read failed for {p}: {e}", file=sys.stderr)
    # JSONL fallback (gzipped or not)
    for fname in ("train.jsonl", "train.jsonl.gz"):
        jp = corpus_dir / fname
        if not jp.exists():
            continue
        rows: list[str] = []
        opener = open
        if fname.endswith(".gz"):
            import gzip
            opener = gzip.open  # type: ignore[assignment]
        try:
            with opener(jp, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    for col in text_col_candidates:
                        if col in obj and isinstance(obj[col], str):
                            rows.append(obj[col])
                            break
            return rows
        except Exception as e:
            print(f"[mix] jsonl read failed for {jp}: {e}", file=sys.stderr)
    return []


def _normalise_ratios(user: dict[str, float]) -> dict[str, float]:
    s = sum(v for v in user.values() if v > 0)
    if s <= 0:
        return DEFAULT_RATIOS
    return {k: v / s for k, v in user.items() if v > 0}


def _parse_ratio_arg(s: str | None) -> dict[str, float] | None:
    if not s:
        return None
    out: dict[str, float] = {}
    for part in s.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"bad --ratios entry: {part}")
        k, v = part.split(":", 1)
        out[k.strip()] = float(v)
    return out


def _read_direct_parquet(path: Path, text_col_candidates: list[str]) -> list[str]:
    """Read text column from a direct parquet path (P9 ``--corpora`` mode).

    Differs from ``_read_corpus_rows`` in that the input is a plain file
    (not a corpus directory containing ``train.parquet``). Falls back to
    JSONL if the path lacks a ``.parquet`` suffix; otherwise treats the
    extension as authoritative.
    """
    if not path.exists():
        print(f"[mix] direct path does not exist: {path}", file=sys.stderr)
        return []
    if path.suffix.lower() == ".parquet" and _HAVE_ARROW:
        try:
            tbl = pq.read_table(path)
            for col in text_col_candidates + tbl.column_names:
                if col in tbl.column_names:
                    return [str(x) for x in tbl[col].to_pylist() if x is not None]
        except Exception as e:
            print(f"[mix] direct parquet read failed for {path}: {e}",
                  file=sys.stderr)
            return []
    if path.suffix.lower() in (".jsonl", ".json"):
        rows: list[str] = []
        try:
            with open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    for col in text_col_candidates:
                        if col in obj and isinstance(obj[col], str):
                            rows.append(obj[col])
                            break
            return rows
        except Exception as e:
            print(f"[mix] direct jsonl read failed for {path}: {e}",
                  file=sys.stderr)
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="/workspace/data/pretrain",
                    help="Directory containing per-corpus subdirs.")
    # P9: direct parquet inputs for the smoke test. When set, --corpora
    # bypasses the per-corpus directory layout entirely and treats each
    # comma-separated path as one corpus (the basename is used as the
    # corpus label in the manifest). Single input is fine; the script
    # then becomes "filter + dedup + write" rather than "mix".
    ap.add_argument("--corpora", default=None,
                    help="Comma-separated parquet/jsonl paths (P9 smoke "
                         "mode). When set, --root and --ratios are ignored "
                         "and each path is one corpus weighted equally.")
    ap.add_argument("--out",  default="/workspace/data/pretrain/pretrain_mix.parquet",
                    help="Output parquet path.")
    ap.add_argument("--target-rows", type=int, default=200000,
                    help="Approximate output row count.")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=32_000,
                    help="~8K tokens at 4 chars/token.")
    ap.add_argument("--ratios", default=None,
                    help="Override default mix, e.g. fineweb_edu:0.5,wudao:0.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any selected corpus is missing or empty.")
    args = ap.parse_args()

    if not _HAVE_ARROW:
        print("[mix] pyarrow + pandas required to write parquet output",
              file=sys.stderr)
        return 2

    random.seed(args.seed)

    text_cols = ["text", "content", "raw_content", "document", "code", "instruction"]
    corpus_rows: dict[str, list[str]] = {}

    # P9 direct-parquet path: --corpora overrides --root + --ratios.
    if args.corpora:
        paths = [Path(p.strip()) for p in args.corpora.split(",") if p.strip()]
        if not paths:
            print("[mix] --corpora was empty after parsing", file=sys.stderr)
            return 4
        ratios = {p.stem: 1.0 / len(paths) for p in paths}  # equal-weight
        for p in paths:
            rows = _read_direct_parquet(p, text_cols)
            corpus_rows[p.stem] = rows
            print(f"[mix] direct corpus={p.stem:24s} rows={len(rows):,} src={p}")
            if args.strict and not rows:
                print(f"[mix] STRICT: direct corpus {p} empty; aborting",
                      file=sys.stderr)
                return 3
    else:
        user = _parse_ratio_arg(args.ratios)
        ratios = _normalise_ratios(user) if user else DEFAULT_RATIOS

        root = Path(args.root)
        for c in ratios:
            rows = _read_corpus_rows(root / c, text_cols)
            corpus_rows[c] = rows
            print(f"[mix] corpus={c:24s} rows={len(rows):,}")
            if args.strict and not rows:
                print(f"[mix] STRICT: corpus {c} empty; aborting", file=sys.stderr)
                return 3

    # Allocate target rows across corpora by ratio.
    plan: dict[str, int] = {}
    for c, r in ratios.items():
        plan[c] = int(args.target_rows * r)

    seen: set[str] = set()
    out_text: list[str] = []
    out_corpus: list[str] = []
    out_lang: list[str] = []
    dropped_dup = dropped_qual = 0

    for c, want in plan.items():
        rows = corpus_rows.get(c, [])
        if not rows:
            print(f"[mix] {c}: no rows available, skipping its quota={want}")
            continue
        random.shuffle(rows)
        kept = 0
        for txt in rows:
            if kept >= want:
                break
            if not _quality_ok(txt, args.min_chars, args.max_chars):
                dropped_qual += 1
                continue
            key = _hash_key(txt)
            if key in seen:
                dropped_dup += 1
                continue
            seen.add(key)
            lang = _lang_id(txt)
            # weak language consistency check: zh/en/code corpora vs lang label
            # do not reject; just record. trainer can re-balance from `lang`.
            out_text.append(txt)
            out_corpus.append(c)
            out_lang.append(lang)
            kept += 1
        print(f"[mix] {c}: emitted {kept}/{want}")

    print(f"[mix] dropped: dup={dropped_dup} qual={dropped_qual}; total_out={len(out_text)}")

    # Final shuffle so trainer iter is well-mixed.
    idx = list(range(len(out_text)))
    random.shuffle(idx)
    out_text   = [out_text[i]   for i in idx]
    out_corpus = [out_corpus[i] for i in idx]
    out_lang   = [out_lang[i]   for i in idx]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "text": out_text,
        "corpus": out_corpus,
        "lang": out_lang,
    })
    pq.write_table(pa.Table.from_pandas(df), args.out)
    print(f"[mix] wrote {len(df):,} rows -> {args.out}")

    # Sidecar manifest.
    mfile = str(args.out) + ".manifest.json"
    with open(mfile, "w", encoding="utf-8") as f:
        json.dump({
            "out": str(args.out),
            "rows": len(df),
            "ratios": ratios,
            "by_corpus": {c: out_corpus.count(c) for c in ratios},
            "by_lang":   {l: out_lang.count(l)   for l in set(out_lang)},
            "dropped":   {"dup": dropped_dup, "qual": dropped_qual},
            "seed": args.seed,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    print(f"[mix] manifest -> {mfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
