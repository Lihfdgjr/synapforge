"""collect_kd_data.py — pre-compute teacher top-K logits for offline KD.

Pattern: DistilBERT / MiniLM / TinyBERT.
The teacher (Qwen 2.5 0.5B by default) is run **once** over the source
corpus; for each token position we cache only the top-K teacher logits +
their softmax probabilities. The student trainer then loads this cache
and computes KL(student_topk || teacher_topk_renormalised) without ever
re-running the teacher — many epochs at a fraction of the cost.

Storage math (V=151936, fp16 = 2 bytes):
    full-vocab logits per token   = 151936 * 2 = 303,872 bytes
    top-64 indices + log-probs    = 64 * (4 + 2)  = 384 bytes
    reduction                       ~= 791x

The reduction is fundamental to the BitNet / DistilBERT KD recipe: the
teacher distribution is concentrated (top-64 captures >99% mass on a
trained LM), so the renormalised top-K KL is a near-exact gradient
proxy for the full-vocab KL. See ``tests/integration/test_kd_topk_softmax.py``
for the math justification at multiple K values.

Output schema (parquet, one row = one input window):

    input_ids       int32    [seq_len]
    topk_indices    int32    [seq_len, K]
    topk_log_probs  float16  [seq_len, K]
    seq_len         int32    scalar (denormalised for fast ParquetTokenStream
                              sample-len checks without unpacking arrays)

Trainer side (``train_100m_kd.py``) consumes ``topk_indices``+``topk_log_probs``
directly when ``--cached-kd-parquet`` is supplied: it reconstructs a
sparse teacher distribution at the K positions, renormalises, and feeds
``_kd_topk_loss`` exactly the same way as live-teacher KD does today.

CLI
---
    python scripts/collect_kd_data.py \\
        --teacher Qwen/Qwen2.5-0.5B \\
        --input  /workspace/data/wikitext-103/wt103_train.parquet \\
        --output /workspace/data/kd_cache/wt103_qwen05_top64.parquet \\
        --topk 64 --seq-len 512 --batch-size 8 --max-rows 0

    python scripts/collect_kd_data.py --help

``--max-rows 0`` = process the whole input corpus (default).
``--max-rows 100`` = smoke-test on N rows.

The script defers imports of ``torch`` + ``transformers`` until ``main()``
so its module-level body parses on a torch-less dev box (the unit tests
exercise the storage-format logic via mocks without ever loading a real
teacher).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Module-level pyarrow import is fine — it's already a hard dep of the
# trainer's ``ParquetTokenStream`` data path. We avoid pulling torch /
# transformers here so this file imports cleanly in CI / tests.
try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


# --- candidate teacher locations -------------------------------------------
# Order matters: try local on-rental cache first (no network), then remote.
DEFAULT_TEACHER_CANDIDATES = (
    "/workspace/teachers/qwen2.5-0.5b",
    "Qwen/Qwen2.5-0.5B",
)


# --- storage-format helpers (testable without torch) ----------------------
def topk_storage_bytes(seq_len: int, k: int) -> int:
    """Bytes per sample under the top-K scheme.

    int32 input_ids [seq_len]      -> 4 * seq_len
    int32 topk_indices [seq_len,K] -> 4 * seq_len * K
    fp16  topk_log_probs[seq_len,K]-> 2 * seq_len * K
    Plus tiny overhead for parquet headers (ignored — drowns in the rows).
    """
    return seq_len * 4 + seq_len * k * (4 + 2)


def full_vocab_bytes(seq_len: int, vocab_size: int) -> int:
    """Bytes per sample under naive full-vocab fp16 logit caching."""
    return seq_len * 4 + seq_len * vocab_size * 2  # input_ids + logits fp16


def storage_reduction(seq_len: int, k: int, vocab_size: int) -> float:
    """``full_vocab / top_k`` ratio. Shipped as a sanity assert in tests."""
    return full_vocab_bytes(seq_len, vocab_size) / max(topk_storage_bytes(seq_len, k), 1)


# --- parquet I/O ---------------------------------------------------------
def write_kd_parquet(
    out_path: str,
    input_ids: Sequence[Sequence[int]],
    topk_indices: Sequence[object],   # 2D arrays, one per row
    topk_log_probs: Sequence[object], # 2D arrays, one per row
    seq_lens: Sequence[int],
    extra_meta: Optional[dict] = None,
) -> int:
    """Write the KD cache as parquet. Returns row count.

    All four lists must have the same outer length. ``topk_indices`` rows
    are flattened to int32 lists [seq_len*K]; ``topk_log_probs`` to fp16
    lists [seq_len*K]. We also store per-row K + seq_len so the trainer
    can reshape on read without a separate sidecar manifest.
    """
    if not _HAVE_ARROW:
        raise RuntimeError("pyarrow + numpy required for KD parquet output")
    n = len(input_ids)
    assert n == len(topk_indices) == len(topk_log_probs) == len(seq_lens), (
        f"row-count mismatch: ids={n} idx={len(topk_indices)} "
        f"lp={len(topk_log_probs)} sl={len(seq_lens)}"
    )

    # Flatten 2D rows -> 1D int32 / fp16 arrays for parquet list<scalar>.
    flat_idx, flat_lp, ks = [], [], []
    for i in range(n):
        idx_arr = np.asarray(topk_indices[i], dtype=np.int32).reshape(-1)
        lp_arr = np.asarray(topk_log_probs[i], dtype=np.float16).reshape(-1)
        flat_idx.append(idx_arr.tolist())
        flat_lp.append(lp_arr.tolist())
        # K is the per-token slot count: total / seq_len. We pin it
        # per-row so heterogeneous K is permitted (e.g. K trimmed at the
        # end of a short doc).
        ks.append(int(idx_arr.size // max(seq_lens[i], 1)))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "input_ids": [list(map(int, x)) for x in input_ids],
            "topk_indices": flat_idx,
            "topk_log_probs": flat_lp,
            "seq_len": [int(x) for x in seq_lens],
            "topk": ks,
        }
    )
    pq.write_table(table, out_path, compression="zstd")

    # Companion manifest -- like the synth_zh script, parallel discoverable.
    meta = {
        "kind": "kd_topk_cache",
        "rows": n,
        "topk": int(ks[0]) if ks else 0,
        "seq_len_first": int(seq_lens[0]) if seq_lens else 0,
        "schema": {
            "input_ids": "list<int32>",
            "topk_indices": "list<int32> flattened seq_len*K",
            "topk_log_probs": "list<float16> flattened seq_len*K",
            "seq_len": "int32",
            "topk": "int32",
        },
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(str(out_path) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return n


# --- teacher loader (lazy torch import) -----------------------------------
def _load_teacher(candidates: Sequence[str], device: str = "cpu"):  # pragma: no cover
    """Load Qwen 2.5 teacher from the first working location.

    Lazy: only called from ``main()``. ``pragma: no cover`` because we
    test the storage-format helpers + writer in isolation; live teacher
    forward is exercised on the rental run.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    last_err = None
    for path in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                trust_remote_code=True,
            ).to(device).eval()
            print(f"[collect_kd] loaded teacher from {path!r} on {device}", flush=True)
            return tok, mdl, path
        except Exception as exc:
            print(f"[collect_kd] candidate {path!r} failed: {exc!r}", file=sys.stderr)
            last_err = exc
    raise RuntimeError(f"no teacher loadable from {list(candidates)}: {last_err!r}")


def _read_text_corpus(path: str) -> Iterable[str]:  # pragma: no cover
    """Yield text strings from parquet (column 'text') OR plain .txt file."""
    p = Path(path)
    if p.suffix == ".parquet":
        if not _HAVE_ARROW:
            raise RuntimeError("pyarrow needed to read parquet input")
        table = pq.read_table(path)
        col = table.column("text").to_pylist()
        for s in col:
            if s:
                yield str(s)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line


def _run_teacher(
    teacher_candidates: Sequence[str],
    text_iter: Iterable[str],
    topk: int,
    seq_len: int,
    batch_size: int,
    max_rows: int,
    device: str,
    progress: bool,
) -> Tuple[list, list, list, list, dict]:  # pragma: no cover -- live torch
    """Run teacher forward and harvest top-K. Returns parallel lists ready
    to hand to ``write_kd_parquet`` plus a metadata dict.

    Caller must ensure ``transformers`` + ``torch`` are importable.
    """
    import numpy as _np
    import torch
    import torch.nn.functional as F

    tok, mdl, src = _load_teacher(teacher_candidates, device=device)

    # Vocab + padding strategy: Qwen 2.5 0.5B reports ~151643 effective
    # vocab; the model.config.vocab_size is 151936 (padded). We store the
    # un-padded indices (model emits up to 151935 anyway).
    vocab_size = int(getattr(mdl.config, "vocab_size", 151936))

    in_ids_rows: list = []
    idx_rows: list = []
    lp_rows: list = []
    sl_rows: list = []

    try:
        from tqdm.auto import tqdm
        bar_iter = tqdm if progress else (lambda x, **kw: x)
    except Exception:
        bar_iter = lambda x, **kw: x  # noqa: E731

    pending: List[str] = []
    n_done = 0

    def _flush(batch: List[str]) -> None:
        nonlocal n_done
        if not batch:
            return
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=seq_len,
        )
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = mdl(input_ids=ids, attention_mask=attn)
            logits = out.logits  # [B, T, V]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            top_lp, top_idx = log_probs.topk(topk, dim=-1)
        ids_cpu = ids.cpu().numpy().astype(_np.int32)
        attn_cpu = attn.cpu().numpy().astype(_np.int32)
        idx_cpu = top_idx.cpu().numpy().astype(_np.int32)
        lp_cpu = top_lp.cpu().numpy().astype(_np.float16)
        for b in range(ids_cpu.shape[0]):
            valid = int(attn_cpu[b].sum())
            if valid <= 1:
                continue
            in_ids_rows.append(ids_cpu[b, :valid].tolist())
            idx_rows.append(idx_cpu[b, :valid, :])
            lp_rows.append(lp_cpu[b, :valid, :])
            sl_rows.append(valid)
            n_done += 1

    bar = bar_iter(text_iter, desc="kd-collect")
    for txt in bar:
        if max_rows and n_done >= max_rows:
            break
        pending.append(txt)
        if len(pending) >= batch_size:
            _flush(pending)
            pending = []
    if pending and (not max_rows or n_done < max_rows):
        _flush(pending)

    meta = {
        "teacher_source": src,
        "vocab_size": vocab_size,
        "topk": topk,
        "seq_len_max": seq_len,
        "batch_size": batch_size,
        "device": device,
        "rows_emitted": n_done,
    }
    return in_ids_rows, idx_rows, lp_rows, sl_rows, meta


# --- main ----------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True,
                    help="Source corpus parquet (col 'text') or txt file")
    ap.add_argument("--output", required=True,
                    help="Output parquet path for KD cache")
    ap.add_argument("--teacher", default=None,
                    help="Override teacher path/HF id; tries the default "
                         "candidate list when unset")
    ap.add_argument("--topk", type=int, default=64,
                    help="Top-K logits to cache per token (default 64)")
    ap.add_argument("--seq-len", type=int, default=512,
                    help="Truncate teacher inputs to this many tokens")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = process whole input corpus")
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES")
                                       else "cpu")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Override max-rows to 16 for fast end-to-end check")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        args.max_rows = 16
    if not _HAVE_ARROW:
        print("[collect_kd] FATAL: pyarrow + numpy required", file=sys.stderr)
        return 2

    candidates = (args.teacher,) if args.teacher else DEFAULT_TEACHER_CANDIDATES
    text_iter = _read_text_corpus(args.input)

    t0 = time.time()
    ids_rows, idx_rows, lp_rows, sl_rows, meta = _run_teacher(
        candidates,
        text_iter,
        topk=args.topk,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        device=args.device,
        progress=not args.no_progress,
    )
    if not ids_rows:
        print("[collect_kd] no rows produced -- empty input?", file=sys.stderr)
        return 1
    n = write_kd_parquet(
        args.output,
        ids_rows, idx_rows, lp_rows, sl_rows,
        extra_meta=dict(
            meta,
            input=str(args.input),
            elapsed_seconds=round(time.time() - t0, 2),
        ),
    )
    # Storage sanity report.
    if sl_rows:
        per_row = topk_storage_bytes(sl_rows[0], args.topk)
        v = meta.get("vocab_size", 151936)
        ratio = storage_reduction(sl_rows[0], args.topk, v)
        print(
            f"[collect_kd] wrote {n:,} rows -> {args.output} "
            f"(top-{args.topk}, vocab={v}, ~{per_row}B/row, "
            f"{ratio:.0f}x smaller than full-vocab)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
