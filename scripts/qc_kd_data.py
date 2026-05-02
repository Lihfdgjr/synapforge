"""qc_kd_data.py — quality gate for KD distillation data.

Reads:
  - the .samples.jsonl sidecar (one (prompt, completion) JSON per row)
  - the .parquet KD cache (for shape sanity check)

Reports:
  - row count, total tokens, avg seq len
  - distribution by bucket / lang / src
  - 5 random (prompt, completion) pairs (head)
  - TOKEN_SOUP detection: rows where the completion is a 4+ repeating
    n-gram pattern (e.g. "the the the the") get flagged
  - degenerate-empty rows (completion empty or whitespace only)
  - **teacher self-perplexity**: re-load Qwen 2.5 0.5B and compute its
    own ppl over the (prompt+completion) sequences. A healthy teacher
    should land at ppl 1.5-5 on its own continuations; if it's higher
    that means the teacher itself is unstable on this prompt set.

Optionally drops bad rows by writing a filtered parquet to
``--out-clean`` and re-emitting samples sidecar with the same drops.

CLI
---
    python scripts/qc_kd_data.py \\
        --parquet /workspace/data/kd_distill_v1.parquet \\
        --samples /workspace/data/kd_distill_v1.parquet.samples.jsonl \\
        --teacher /workspace/teachers/qwen2.5-0.5b \\
        --sample-n 100 \\
        --out-clean /workspace/data/kd_distill_v1.clean.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


# --- TOKEN_SOUP detector ----------------------------------------------------
# We're looking for the failure mode "the the the the..." (a single
# *word-shaped* token repeated 8+ times in a row), or its bigram form
# ("foo bar foo bar..."), or a single Chinese character glyph repeated
# ≥12 times in a row. We deliberately DON'T flag:
#   - whitespace runs in code (4-space indent)
#   - punctuation runs (".....", "::::")
#   - "}}}}}}", ")))" etc.
# because those are normal code patterns, not LM degeneration.
_WORD_RE = re.compile(r"[A-Za-z]+|[一-鿿]+|[0-9]+")


def is_token_soup(text: str, min_repeats: int = 8) -> bool:
    """Detect "the the the..." or "garbled garbled..." token-loop patterns.

    Strategy: extract *word tokens only* (alpha runs, CJK runs, digit runs);
    ignore whitespace, punctuation, and code symbols. Then look for any
    unigram or bigram that repeats consecutively at least ``min_repeats``
    times. Empty / very short completions are reported separately.
    """
    if not text or len(text) < 8:
        return False
    toks = _WORD_RE.findall(text)
    if len(toks) >= min_repeats:
        # Unigram repetition: "the the the the the the the the".
        for i in range(len(toks) - min_repeats + 1):
            if all(toks[i + k] == toks[i] for k in range(min_repeats)):
                return True
        # Bigram repetition: "foo bar foo bar foo bar ...".
        for i in range(len(toks) - 2 * min_repeats + 1):
            if all(toks[i + 2 * k] == toks[i] and toks[i + 2 * k + 1] == toks[i + 1]
                   for k in range(min_repeats)):
                return True
    # Single-char glyph repetition (Chinese / ad-hoc): only fire when
    # ≥ 12 of the SAME CJK character in a row. Skip ASCII (would flag
    # whitespace + code symbols as soup).
    for m in re.finditer(r"([一-鿿])\1{11,}", text):
        return True
    return False


def _empty_completion(text: str) -> bool:
    return not text or not text.strip()


# --- distribution stats -----------------------------------------------------
def _summarise_distribution(samples: list[dict]) -> dict:
    by_bucket = Counter(r.get("bucket", "?") for r in samples)
    by_lang = Counter(r.get("lang", "?") for r in samples)
    by_src = Counter(r.get("src", "?") for r in samples)
    total_toks = sum(int(r.get("total_tok_len") or 0) for r in samples)
    avg_total = total_toks / max(len(samples), 1)
    avg_prompt = sum(int(r.get("prompt_tok_len") or 0) for r in samples) / max(len(samples), 1)
    return {
        "n_rows": len(samples),
        "total_tokens": total_toks,
        "avg_total_tok_len": round(avg_total, 1),
        "avg_prompt_tok_len": round(avg_prompt, 1),
        "avg_completion_tok_len": round(avg_total - avg_prompt, 1),
        "by_bucket": dict(by_bucket),
        "by_lang": dict(by_lang),
        "by_src": dict(by_src),
    }


# --- teacher perplexity self-eval ------------------------------------------
def _teacher_self_ppl(
    teacher: str, samples: list[dict], n_eval: int, batch_size: int, device: str,
):  # pragma: no cover -- live torch
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(20260502)
    pool = [r for r in samples
            if not _empty_completion(r.get("completion", ""))
            and not is_token_soup(r.get("completion", ""))]
    if not pool:
        return {"error": "no clean rows for self-ppl"}
    rng.shuffle(pool)
    eval_rows = pool[:n_eval]

    tok = AutoTokenizer.from_pretrained(teacher, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        teacher,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    nlls: list[float] = []
    n_seen_tokens = 0
    for start in range(0, len(eval_rows), batch_size):
        batch = eval_rows[start:start + batch_size]
        texts = [r["prompt"] + r["completion"] for r in batch]
        enc = tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1024,
        )
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = mdl(input_ids=ids, attention_mask=attn)
            logits = out.logits[:, :-1, :].float()  # predict positions 1..T
            target = ids[:, 1:]
            mask = attn[:, 1:].float()
            log_probs = F.log_softmax(logits, dim=-1)
            tgt_lp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
            nll = -(tgt_lp * mask).sum().item()
            n_tok = mask.sum().item()
        nlls.append(nll)
        n_seen_tokens += int(n_tok)

    ppl = math.exp(sum(nlls) / max(n_seen_tokens, 1))
    return {
        "ppl": round(ppl, 3),
        "n_rows_evaluated": len(eval_rows),
        "n_tokens_evaluated": n_seen_tokens,
    }


# --- parquet sanity ----------------------------------------------------------
def _check_parquet(path: str) -> dict:
    if not _HAVE_ARROW:
        return {"error": "no pyarrow"}
    pf = pq.ParquetFile(path)
    schema = {f.name: str(f.type) for f in pf.schema_arrow}
    return {
        "rows": pf.metadata.num_rows,
        "schema": schema,
        "size_bytes": Path(path).stat().st_size,
    }


# --- main ------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parquet", required=True, help="Distill parquet path")
    ap.add_argument("--samples", required=True, help=".samples.jsonl sidecar")
    ap.add_argument("--teacher", default="/workspace/teachers/qwen2.5-0.5b",
                    help="Teacher model dir for self-ppl (set 'none' to skip)")
    ap.add_argument("--sample-n", type=int, default=100,
                    help="Random sample N rows for self-ppl computation")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-clean", default=None,
                    help="If set: write a parquet with TOKEN_SOUP/empty rows removed")
    ap.add_argument("--seed", type=int, default=20260502)
    args = ap.parse_args(argv)

    # Read sidecar.
    samples: list[dict] = []
    with open(args.samples, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                pass
    if not samples:
        print("[qc_kd_data] no samples loaded", file=sys.stderr)
        return 1

    print("=" * 64)
    print(f"[qc_kd_data] file: {args.parquet}")
    print(f"[qc_kd_data] sidecar: {args.samples} ({len(samples)} rows)")

    pq_info = _check_parquet(args.parquet) if Path(args.parquet).exists() else {"error": "parquet not found"}
    print(f"[qc_kd_data] parquet sanity: {json.dumps(pq_info)}")

    dist = _summarise_distribution(samples)
    print("\n[qc_kd_data] distribution:")
    for k, v in dist.items():
        print(f"  {k}: {v}")

    # Bad-row scan.
    n_empty = 0
    n_soup = 0
    soup_buckets: Counter = Counter()
    bad_idxs: set[int] = set()
    for i, r in enumerate(samples):
        c = r.get("completion") or ""
        if _empty_completion(c):
            n_empty += 1
            bad_idxs.add(i)
        elif is_token_soup(c):
            n_soup += 1
            bad_idxs.add(i)
            soup_buckets[r.get("bucket", "?")] += 1
    print(f"\n[qc_kd_data] BAD ROW SCAN:")
    print(f"  empty:       {n_empty} ({100.0 * n_empty / len(samples):.2f}%)")
    print(f"  token-soup:  {n_soup} ({100.0 * n_soup / len(samples):.2f}%)")
    if soup_buckets:
        print(f"  soup-by-bucket: {dict(soup_buckets)}")

    # Sample 5 random good rows.
    rng = random.Random(args.seed)
    good = [r for i, r in enumerate(samples) if i not in bad_idxs]
    rng.shuffle(good)
    print("\n[qc_kd_data] 5 RANDOM SAMPLES:")
    for i, r in enumerate(good[:5]):
        prompt = r["prompt"][:200].replace("\n", "\\n")
        comp = r["completion"][:300].replace("\n", "\\n")
        print(f"  --- sample {i} (bucket={r.get('bucket')} lang={r.get('lang')}) ---")
        print(f"  prompt:     {prompt}")
        print(f"  completion: {comp}")

    # Teacher self-ppl.
    if args.teacher and args.teacher.lower() != "none":
        print(f"\n[qc_kd_data] computing teacher self-ppl on {args.sample_n} clean rows ...")
        try:
            import torch  # noqa: E402
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            t0 = time.time()
            ppl_info = _teacher_self_ppl(
                args.teacher, samples, n_eval=args.sample_n,
                batch_size=8, device=device,
            )
            ppl_info["elapsed_seconds"] = round(time.time() - t0, 1)
            print(f"  result: {json.dumps(ppl_info)}")
            if isinstance(ppl_info.get("ppl"), (int, float)):
                p = ppl_info["ppl"]
                if p < 5.0:
                    print("  status: GOOD (teacher ppl on its own continuations < 5)")
                elif p < 15.0:
                    print(f"  status: ACCEPTABLE (ppl={p}, somewhat noisy but usable)")
                else:
                    print(f"  status: WARN (ppl={p}; teacher unstable on this prompt set)")
        except Exception as e:
            print(f"  self-ppl failed: {e!r}")

    # Optional: emit cleaned parquet.
    if args.out_clean:
        if not _HAVE_ARROW:
            print("[qc_kd_data] cannot emit cleaned parquet without pyarrow", file=sys.stderr)
            return 1
        if not Path(args.parquet).exists():
            print("[qc_kd_data] source parquet missing; skipping --out-clean", file=sys.stderr)
            return 1
        # Re-read rows; drop ones at bad_idxs *as ordered by sidecar*.
        # Sidecar order matches parquet write order in gen_kd_data.py
        # (with the OOM-drop adjustment), which means the i-th sample
        # is the i-th parquet row (after gen_kd's keep_idx prune).
        # Safest: reload, take by index.
        tab = pq.read_table(args.parquet)
        n_pq = tab.num_rows
        keep = [i for i in range(n_pq) if i not in bad_idxs]
        if len(keep) == n_pq:
            print("[qc_kd_data] no bad rows; out-clean = copy")
        new_tab = tab.take(keep)
        Path(args.out_clean).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_tab, args.out_clean, compression="zstd")
        # Also re-emit sidecar.
        clean_sidecar = args.out_clean + ".samples.jsonl"
        with open(clean_sidecar, "w", encoding="utf-8") as f:
            for i in keep:
                if i < len(samples):
                    f.write(json.dumps(samples[i], ensure_ascii=False) + "\n")
        # Manifest.
        clean_manifest = args.out_clean + ".manifest.json"
        with open(clean_manifest, "w", encoding="utf-8") as f:
            json.dump({
                "kind": "kd_distill_cache_clean",
                "rows": len(keep),
                "source": args.parquet,
                "dropped_indices": sorted(bad_idxs),
                "n_dropped": len(bad_idxs),
                "soup_dropped": n_soup,
                "empty_dropped": n_empty,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, f, indent=2)
        print(f"\n[qc_kd_data] wrote cleaned parquet: {args.out_clean} "
              f"({len(keep)}/{n_pq} rows kept)")
        print(f"[qc_kd_data] sidecar: {clean_sidecar}")
        print(f"[qc_kd_data] manifest: {clean_manifest}")

    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
