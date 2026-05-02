"""assemble_quality_corpus.py — pick the BEST parquets and emit a mixed corpus.

D3 of the 2026-05-02 quality-data push. Reads the audit JSON
(``docs/RENTAL_DATA_AUDIT.json``) and the quality scorecard
(``docs/DATA_QUALITY_REPORT.json``), then assembles a mixed multi-domain
corpus for Synap-1 training.

Two output modes:

1. **Manifest mode** (default — fast, no copies). Writes a manifest at
   ``--out`` describing which files to mix at which weights, plus a
   training-side ``--data-files file:weight,file:weight,...`` snippet
   ready to paste into the Run 7 launcher. Designed for the new
   ``ParquetTokenStream(files_with_weights=...)`` constructor wired in
   ``synapforge/data/__init__.py`` (D4 of this push).

2. **Materialise mode** (``--materialise``). Streams the chosen rows
   through ``mix_pretrain_corpora.py``-style filtering + dedup and writes
   a single coalesced parquet at ``--out-parquet``. Slower (the trainer
   doesn't need this if it accepts file lists, but the smoke test in
   ``tests/data/test_corpus_quality.py`` uses the materialised output to
   assert the quality gates pass on the *assembled* corpus, not the
   individual files).

Default mixture (over English LM + Chinese + instruction):

    domain    weight   typical source
    -------  --------  --------------------------------------------------
    web_edu   0.50     fineweb_edu/000_*.parquet (English educational)
    wiki      0.30     wiki_zh/train-*.parquet  (Chinese encyclopedia)
    instr     0.20     sft_combined.jsonl + sft_seed_v3.jsonl

Auto-fallbacks:
  - If a domain has no PASS-quality file, drop it from the mixture and
    renormalise. Print a warning so the operator knows the assembled
    corpus is narrower than nominal.
  - If FineWeb-Edu is missing entirely (e.g. on Windows-dev), accept
    WikiText-103 (300MB, 0.71 non-empty) as a degraded English source.

The manifest contains:

    {
      "corpus_version": "v7",
      "total_files": 4,
      "estimated_tokens": 1_200_000_000,
      "mixture": [
         {"path": "myserver:/.../fineweb_edu/000_00000.parquet",
          "domain": "web_edu", "weight": 0.5, "rows": 726000,
          "tokens_est": 700M, "quality_score": 0.85},
         ...
      ],
      "data_files_arg": "file1.parquet:0.50,file2.parquet:0.30,...",
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Domain rules. Each rule scores a filename against a domain; first match
# wins. Order matters (more specific first).
# ---------------------------------------------------------------------------
DOMAIN_RULES: list[tuple[str, list[str]]] = [
    # web_edu: educational filtered web crawl
    ("web_edu", ["fineweb_edu", "fineweb-edu", "open-web-math", "openwebmath"]),
    # web_general: noisier crawl
    ("web_general", ["slimpajama", "fineweb", "common_crawl"]),
    # wiki: encyclopedic, well-formatted long-form
    ("wiki", ["wiki_zh", "wikitext", "wt103", "wikipedia"]),
    # code: mostly Python
    ("code", ["smollm_py", "the_stack", "starcoder", "codeparrot"]),
    # math: formal proofs, math word problems
    ("math", ["math", "gsm8k", "proof"]),
    # instruct: instruction-following datasets
    ("instruct", ["alpaca", "sft_", "tulu", "ultrachat", "instruct"]),
    # books / long-form (Gutenberg)
    ("books", ["gutenberg", "bookcorpus", "books"]),
]


def _classify_domain(path: str) -> str:
    """Return the domain label for a filename or 'other' if unknown."""
    low = path.lower().replace("\\", "/")
    for domain, keys in DOMAIN_RULES:
        for k in keys:
            if k in low:
                return domain
    return "other"


# Default mixture by domain. Sums to 1.0; missing domains renormalise.
# Tilts toward English educational web (proven highest signal-to-noise
# per FineWeb-Edu paper) + Chinese wiki + a small slice of instruction
# data so the model sees question/answer format from day one.
DEFAULT_MIXTURE: dict[str, float] = {
    "web_edu":   0.50,
    "wiki":      0.30,
    "instruct":  0.20,
    # If others exist they are merged with reduced weight:
    "code":      0.05,
    "math":      0.05,
    "books":     0.03,
    "web_general": 0.02,
    "other":     0.0,
}


@dataclass
class Candidate:
    """One file from the audit + scorer with a domain label."""
    path: str
    host: str
    domain: str
    rows: int = 0
    size_bytes: int = 0
    text_column: str | None = None
    quality_score: float = 0.0
    pass_quality_gate: bool = False
    tokens_est: int = 0


def _load_audit(audit_path: Path) -> list[Candidate]:
    """Walk the audit JSON and return candidate records.

    Includes:
      * parquets with a content-bearing string column (text/content/...)
      * jsonl files whose first sample looks like {"text":...} or
        {"messages":[...]} -- the SFT format the trainer's chat harness
        already speaks.

    Skipped:
      * parquets where the auto-detected text column is metadata
        (video_id / filepath / blob_id — these never carry text).
      * jsonl whose first sample is a video-caption or tool-error blob.
    """
    cands: list[Candidate] = []
    if not audit_path.exists():
        return cands
    m = json.loads(audit_path.read_text(encoding="utf-8"))
    for h in m.get("hosts", []):
        host_name = h.get("host", "?")
        for e in h.get("entries", []):
            kind = e.get("kind")
            path = e.get("path", "")
            if kind == "parquet":
                tc = e.get("text_column")
                # Skip junk text-cols (the audit picks first-string-col).
                if tc in (None, "video_id", "filepath", "blob_id"):
                    continue
                cands.append(Candidate(
                    path=path,
                    host=host_name,
                    domain=_classify_domain(path),
                    rows=int(e.get("rows", 0) or 0),
                    size_bytes=int(e.get("size_bytes", 0)),
                    text_column=tc,
                ))
            elif kind == "jsonl":
                # Heuristic: keep jsonl if any sample contains "text" or
                # "messages" or "content" keys. Skip MSRVTT video
                # captions, synth/error blobs (just "seed_hash"/"error"),
                # and other non-LM artefacts.
                samples = e.get("samples", [])
                ok = False
                for s in samples:
                    if not s:
                        continue
                    sl = s[:300].lower()
                    if (
                        '"text"' in sl
                        or '"messages"' in sl
                        or '"content"' in sl
                    ) and (
                        '"video_id"' not in sl
                        and '"video_path"' not in sl
                        and '"error"' not in sl
                    ):
                        ok = True
                        break
                if not ok:
                    continue
                cands.append(Candidate(
                    path=path,
                    host=host_name,
                    domain=_classify_domain(path),
                    rows=int(e.get("rows", 0) or 0),
                    size_bytes=int(e.get("size_bytes", 0)),
                    text_column="(jsonl)",
                ))
    return cands


def _merge_scores(cands: list[Candidate], scores_path: Path) -> None:
    """Update candidate quality_score / pass_quality_gate from scorer output."""
    if not scores_path.exists():
        return
    by_path: dict[str, dict[str, Any]] = {}
    for s in json.loads(scores_path.read_text(encoding="utf-8")):
        # Score paths may be prefixed with 'host:'; normalise to bare path
        sp = s.get("path", "")
        bare = sp.split(":", 1)[1] if ":" in sp and not sp.startswith("/") else sp
        # Also strip "host:" prefix if present
        if ":" in sp and not sp[1:3].startswith("/"):
            bare = sp.split(":", 1)[1]
        by_path[bare] = s
    for c in cands:
        s = by_path.get(c.path)
        if s is None:
            continue
        c.quality_score = float(s.get("composite_score", 0.0) or 0.0)
        c.pass_quality_gate = bool(s.get("pass_quality_gate", False))
        # Refine token estimate from the actual sample
        n_rows_sampled = int(s.get("n_rows_sampled", 0) or 0)
        n_tokens_sampled = int(s.get("n_tokens_total", 0) or 0)
        if n_rows_sampled > 0 and c.rows > 0:
            c.tokens_est = int(n_tokens_sampled / n_rows_sampled * c.rows)


def _bucket_by_domain(cands: list[Candidate]) -> dict[str, list[Candidate]]:
    """Group candidates by domain and sort each bucket by quality desc."""
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        buckets[c.domain].append(c)
    for k in buckets:
        buckets[k].sort(key=lambda x: (x.pass_quality_gate, x.quality_score),
                        reverse=True)
    return dict(buckets)


def _renormalise_mixture(
    requested: dict[str, float], available: list[str]
) -> dict[str, float]:
    """Drop domains with no candidates; renormalise the rest to sum 1.0."""
    sub = {d: w for d, w in requested.items() if w > 0 and d in available}
    s = sum(sub.values())
    if s <= 0:
        return {}
    return {d: w / s for d, w in sub.items()}


def _select_candidates(
    buckets: dict[str, list[Candidate]],
    mixture: dict[str, float],
    target_gb: float,
) -> list[tuple[Candidate, float]]:
    """For each domain, pick enough top-quality files to fill its byte
    budget. Returns ``[(candidate, weight_within_corpus), ...]``.

    The weight_within_corpus is how much of the mixed stream this file
    occupies, taking both the domain weight and within-domain
    rank-by-quality into account: top file in a domain gets larger
    fraction than secondary shards.
    """
    target_bytes = int(target_gb * 1e9)
    selected: list[tuple[Candidate, float]] = []

    for domain, w in mixture.items():
        budget = int(target_bytes * w)
        chosen: list[Candidate] = []
        running = 0
        # Prefer PASS files first; allow FAIL fallback to fill the budget
        # if no PASS exists in this domain (warning printed by caller).
        for c in buckets.get(domain, []):
            if running >= budget:
                break
            chosen.append(c)
            running += c.size_bytes
        if not chosen:
            continue
        # Within-domain weights: scale by file size (so a 2GB shard gets
        # more samples than a 100MB shard). Then multiply by the domain
        # weight so all selected files sum to the domain's weight.
        total_size = sum(c.size_bytes for c in chosen) or 1
        for c in chosen:
            sub_w = c.size_bytes / total_size
            selected.append((c, w * sub_w))
    return selected


def _format_data_files_arg(selected: list[tuple[Candidate, float]]) -> str:
    """Build the ``--data-files PATH:W,PATH:W,...`` string for the trainer.

    The trainer's wire-in (D4) splits on commas, then on the LAST colon
    (so file paths with colons survive — though Linux paths normally
    don't have any).

    Filters entries with weight rounding to 0.0000 (4 decimals) — the
    weighted ``ParquetTokenStream`` would still try to open them but
    the file would never actually be sampled.  Zero-weight entries
    arise when a tiny shard (e.g. 8-row sft_gold_v1.jsonl) gets
    allocated less than 0.005% of the corpus weight.
    """
    parts: list[str] = []
    for cand, w in selected:
        if round(w, 4) <= 0.0:
            continue
        # Strip "host:" prefix if present (the trainer expects bare paths
        # OR ``RemoteDataWarehouse``-relative basenames; the host part is
        # informational here).
        path = cand.path
        if ":" in path[:32] and not path.startswith("/"):
            path = path.split(":", 1)[1]
        parts.append(f"{path}:{w:.4f}")
    return ",".join(parts)


def _materialise_corpus(
    selected: list[tuple[Candidate, float]],
    out_parquet: Path,
    target_rows: int,
    seed: int,
) -> dict[str, Any]:
    """Stream rows from each selected file at the chosen weight, dedup +
    quality-filter, write a single mixed parquet.

    This is intentionally lightweight (pyarrow + hashlib + a tiny
    quality filter); the heavy lifting in production is the trainer's
    streaming ``ParquetTokenStream`` so we don't need a full
    ``mix_pretrain_corpora.py`` rewrite. Used by D5's smoke test +
    operators who want a single self-contained shard.
    """
    import hashlib
    import random
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pyarrow required for --materialise: {e!r}")

    rng = random.Random(seed)
    all_text: list[str] = []
    all_corpus: list[str] = []
    all_domain: list[str] = []
    seen: set[str] = set()
    n_dropped_dup = n_dropped_qual = 0
    total_size = sum(w for _, w in selected) or 1
    for cand, w in selected:
        # Locally-accessible path required for materialise; bail
        # otherwise. The default mode (--materialise OFF) does NOT need
        # this and works against remote paths via the manifest.
        if not os.path.exists(cand.path):
            print(f"[assemble] SKIP (not local) {cand.path}", file=sys.stderr)
            continue
        try:
            pf = pq.ParquetFile(cand.path)
        except Exception as exc:
            print(f"[assemble] SKIP (parquet fail) {cand.path}: {exc}",
                  file=sys.stderr)
            continue
        col = cand.text_column or "text"
        # Per-file row quota: ``target_rows * (w / total_size)``.
        quota = max(50, int(target_rows * (w / total_size)))
        kept = 0
        for batch in pf.iter_batches(batch_size=512, columns=[col]):
            for txt in batch.column(col).to_pylist():
                if kept >= quota:
                    break
                if not txt or not str(txt).strip():
                    n_dropped_qual += 1
                    continue
                txt = str(txt)
                # min 50 char to keep, max 32k for memory.
                if len(txt) < 50 or len(txt) > 32_000:
                    n_dropped_qual += 1
                    continue
                key = hashlib.sha256(
                    txt[:4096].encode("utf-8", errors="ignore")
                ).hexdigest()[:32]
                if key in seen:
                    n_dropped_dup += 1
                    continue
                seen.add(key)
                all_text.append(txt)
                all_corpus.append(os.path.basename(cand.path).split(".")[0])
                all_domain.append(cand.domain)
                kept += 1
            if kept >= quota:
                break
        print(f"[assemble] {cand.domain:10s}  emitted {kept:>5}/{quota}  "
              f"<- {os.path.basename(cand.path)}")

    if not all_text:
        raise RuntimeError("assembly produced zero rows")

    # Final shuffle so the trainer's streaming reader doesn't see all
    # web_edu rows in a row then all wiki, which would defeat the purpose
    # of the quality-mix.
    idx = list(range(len(all_text)))
    rng.shuffle(idx)
    all_text = [all_text[i] for i in idx]
    all_corpus = [all_corpus[i] for i in idx]
    all_domain = [all_domain[i] for i in idx]

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "text": all_text,
        "corpus": all_corpus,
        "domain": all_domain,
    })
    pq.write_table(table, out_parquet, compression="zstd")
    print(f"[assemble] wrote {len(all_text):,} rows -> {out_parquet}")
    return {
        "rows": len(all_text),
        "dropped_dup": n_dropped_dup,
        "dropped_qual": n_dropped_qual,
        "by_domain": {d: all_domain.count(d) for d in set(all_domain)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit", default="docs/RENTAL_DATA_AUDIT.json",
                    help="Audit JSON from scripts/audit_rental_data.py.")
    ap.add_argument("--scores", default="docs/DATA_QUALITY_REPORT.json",
                    help="Scorecard JSON from scripts/score_data_quality.py.")
    ap.add_argument("--out", default="docs/QUALITY_CORPUS_MANIFEST.json",
                    help="Where to write the corpus manifest.")
    ap.add_argument("--target-gb", type=float, default=5.0,
                    help="Total corpus byte budget (raw bytes summed; "
                         "the actual sampled size depends on the trainer's "
                         "shuffle/sample policy). Default 5 GB.")
    ap.add_argument(
        "--mixture",
        default=None,
        help="Override domain weights, e.g. 'web_edu:0.6,wiki:0.3,instruct:0.1'. "
             "Domains are renormalised to sum 1.0.",
    )
    ap.add_argument("--include-failed", action="store_true",
                    help="Include files that failed the quality gate "
                         "(default: PASS-only, with a fallback to FAILed "
                         "files only when no PASS exists in a domain).")
    ap.add_argument("--materialise", action="store_true",
                    help="Stream the selected rows into a single output "
                         "parquet (slower; ~2-5 GB output).")
    ap.add_argument("--out-parquet",
                    default="/workspace/data/corpus_v7/mixed.parquet",
                    help="Output parquet path when --materialise is set.")
    ap.add_argument("--target-rows", type=int, default=400_000,
                    help="Target row count when --materialise is set "
                         "(distributed across domains by weight).")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    audit = _load_audit(Path(args.audit))
    if not audit:
        print(f"[assemble] no candidates in {args.audit}", file=sys.stderr)
        return 2
    _merge_scores(audit, Path(args.scores))

    # Apply --include-failed filter + bucket
    if not args.include_failed:
        # Build PASS-only first; fallback to all if a domain has no PASS.
        pass_only = [c for c in audit if c.pass_quality_gate]
        if not pass_only:
            print(
                "[assemble] WARNING: no PASS candidates; using all "
                "(scoring may be incomplete) — pass --include-failed "
                "to silence this warning.",
                file=sys.stderr,
            )
            cands_to_use = audit
        else:
            cands_to_use = pass_only
    else:
        cands_to_use = audit

    buckets = _bucket_by_domain(cands_to_use)

    # If a domain in DEFAULT_MIXTURE has NO files in pass_only, fall
    # back to that domain's full bucket from `audit` (quality < gate but
    # at least content-bearing). Print a warning.
    full_buckets = _bucket_by_domain(audit)
    augmented = dict(buckets)
    for domain, w in DEFAULT_MIXTURE.items():
        if w > 0 and domain not in augmented and domain in full_buckets:
            augmented[domain] = full_buckets[domain]
            print(f"[assemble] WARNING: domain {domain!r} had no PASS "
                  f"file; falling back to best non-PASS shard.",
                  file=sys.stderr)
    buckets = augmented

    # Parse --mixture override
    if args.mixture:
        mix_user: dict[str, float] = {}
        for part in args.mixture.split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ValueError(f"bad --mixture entry: {part!r}")
            k, v = part.rsplit(":", 1)
            mix_user[k.strip()] = float(v)
        requested = mix_user
    else:
        requested = DEFAULT_MIXTURE

    available = list(buckets.keys())
    mixture = _renormalise_mixture(requested, available)
    if not mixture:
        print(f"[assemble] no overlap between mixture {requested!r} and "
              f"available domains {available!r}", file=sys.stderr)
        return 3
    print(f"[assemble] using mixture: " +
          ", ".join(f"{d}:{w:.3f}" for d, w in mixture.items()))

    selected = _select_candidates(buckets, mixture, args.target_gb)
    if not selected:
        print("[assemble] selection produced zero files (target_gb too low?)",
              file=sys.stderr)
        return 4

    # Build the manifest
    total_bytes = sum(c.size_bytes for c, _ in selected)
    total_tokens = sum(c.tokens_est or _estimate_tokens(c) for c, _ in selected)
    by_domain: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"weight": 0.0, "files": [], "rows": 0, "bytes": 0,
                 "tokens_est": 0}
    )
    for c, w in selected:
        d = by_domain[c.domain]
        d["weight"] += w
        d["files"].append(os.path.basename(c.path))
        d["rows"] += c.rows
        d["bytes"] += c.size_bytes
        d["tokens_est"] += c.tokens_est or _estimate_tokens(c)

    manifest = {
        "kind": "quality_corpus_manifest",
        "corpus_version": "v7",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_gb": float(args.target_gb),
        "mixture_requested": requested,
        "mixture_actual": mixture,
        "total_files": len(selected),
        "total_size_gb": round(total_bytes / 1e9, 2),
        "total_rows": sum(c.rows for c, _ in selected),
        "estimated_tokens": int(total_tokens),
        "by_domain": dict(by_domain),
        "files": [
            {
                "path": c.path,
                "host": c.host,
                "domain": c.domain,
                "weight": round(w, 4),
                "rows": c.rows,
                "size_mb": round(c.size_bytes / 1e6, 1),
                "tokens_est": c.tokens_est or _estimate_tokens(c),
                "quality_score": c.quality_score,
                "pass_quality_gate": c.pass_quality_gate,
                "text_column": c.text_column,
            }
            for c, w in selected
        ],
        "data_files_arg": _format_data_files_arg(selected),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Optional materialisation
    if args.materialise:
        result = _materialise_corpus(
            selected, Path(args.out_parquet),
            target_rows=args.target_rows, seed=args.seed,
        )
        manifest["materialised"] = {
            "out_parquet": args.out_parquet,
            **result,
        }
        Path(args.out).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Console summary
    print()
    print("=" * 70)
    print(f"corpus v7 — {len(selected)} files, "
          f"{manifest['total_size_gb']:.2f} GB, "
          f"~{manifest['estimated_tokens']:,} tokens")
    for domain, d in by_domain.items():
        print(f"  {domain:12s}  weight={d['weight']:.3f}  "
              f"rows={d['rows']:,}  ~{d['tokens_est']:,} tokens  "
              f"({len(d['files'])} files)")
    print("-" * 70)
    print(f"manifest -> {args.out}")
    print()
    print("To use in training, add this flag to train_100m_kd.py:")
    print()
    arg = manifest["data_files_arg"]
    if len(arg) > 80:
        arg = arg[:77] + "..."
    print(f"  --data-files \"{manifest['data_files_arg']}\"")
    print()
    print("=" * 70)
    return 0


def _estimate_tokens(c: Candidate) -> int:
    """Fallback token estimate when scorer wasn't run on this file.

    Approximation: 1 byte ≈ 0.30 tokens at Qwen tokenization (English
    averages ~3.5 chars/token; Chinese ~1.5 char/token). 0.30 is the
    midpoint that gets within ~25% on mixed corpora and is good enough
    for budget allocation.
    """
    return int(c.size_bytes * 0.30)


if __name__ == "__main__":
    sys.exit(main())
