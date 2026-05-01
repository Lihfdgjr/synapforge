"""curriculum_sort.py -- sort training data easy->hard by reference-model ppl (T8.3).

Curriculum learning recipe: feed the student easy samples first, hard ones
last. We score "easy" with a small reference model (Qwen 0.5B by default)
running a single forward pass per row of the input parquet; lower per-token
perplexity means the row is more predictable, hence easier for any LM.

Output schema preserves *all* input columns and adds:

* ``ref_ppl``         float32  -- per-row perplexity from the reference model
* ``curriculum_idx``  int32    -- 0 = easiest, N-1 = hardest

Sorting is stable + ascending by ``ref_ppl``; rows with identical ppl keep
their original input order.

CLI
---
    python scripts/curriculum_sort.py \\
        --input  /workspace/data/wt103_qwen_tokens.parquet \\
        --output /workspace/data/wt103_curriculum.parquet \\
        --ref-model Qwen/Qwen2.5-0.5B \\
        --batch-size 8

    python scripts/curriculum_sort.py --smoke   # 5 mocked rows, no torch

Notes
-----
* For unit tests + CI, ``--smoke`` (or any caller that injects a
  ``ppl_fn`` into ``compute_curriculum_order``) skips torch entirely;
  perplexities are derived from the input rows deterministically.
* The reference-model loader is identical in spirit to ``collect_kd_data``
  (same candidate path list -- local rental cache first, then HF id).
* Per-token ppl, NOT per-row ppl: we mean the cross-entropy averaged over
  *valid* token positions, then ``exp``'d. Rows with <2 valid tokens get
  ppl = ``inf`` and sort to the end of the curriculum.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


DEFAULT_REF_CANDIDATES = (
    "/workspace/teachers/qwen2.5-0.5b",
    "Qwen/Qwen2.5-0.5B",
)


# ---------- pure-python sort core (testable, no torch) ----------------------
def compute_curriculum_order(
    input_ids_rows: Sequence[Sequence[int]],
    ppl_fn: Callable[[Sequence[int]], float],
) -> List[int]:
    """Return a permutation of ``range(len(input_ids_rows))`` that sorts by
    ``ppl_fn(row)`` ascending. Stable sort, so equal-ppl rows keep order.

    Caller injects ``ppl_fn`` so tests can run without torch / a real
    reference model. Production wires ``_qwen_ppl_fn`` from a loaded model.
    """
    ppls = [ppl_fn(r) for r in input_ids_rows]
    # Stable sort by (ppl, original_index): equal ppl -> original order.
    order = sorted(range(len(ppls)), key=lambda i: (ppls[i], i))
    return order


def sort_table_by_ppl(
    table,
    ppl_fn: Callable[[Sequence[int]], float],
    input_ids_col: str = "input_ids",
):
    """Re-order an Arrow ``Table`` rows by ``ppl_fn(input_ids)`` ascending.

    Returns a new ``pyarrow.Table`` with all original columns + two new
    columns appended: ``ref_ppl`` (float32) + ``curriculum_idx`` (int32).
    """
    if not _HAVE_ARROW:  # pragma: no cover
        raise RuntimeError("pyarrow required for sort_table_by_ppl")
    if input_ids_col not in table.column_names:
        raise KeyError(
            f"input table missing required column {input_ids_col!r}; "
            f"have {table.column_names}"
        )
    rows = table.column(input_ids_col).to_pylist()
    # Compute ppl for every row up-front (we need full mapping for sort + reattach).
    ppls = [ppl_fn(r) for r in rows]
    n = len(ppls)

    # Stable sort by (ppl, idx). Inf ppls (empty rows) end up last.
    order = sorted(range(n), key=lambda i: (ppls[i], i))

    # Reorder every original column, then append the two new ones.
    sorted_cols = {}
    for name in table.column_names:
        col = table.column(name).to_pylist()
        sorted_cols[name] = [col[i] for i in order]
    sorted_cols["ref_ppl"] = pa.array(
        [float(ppls[i]) for i in order], type=pa.float32()
    )
    sorted_cols["curriculum_idx"] = pa.array(list(range(n)), type=pa.int32())

    return pa.table(sorted_cols)


# ---------- parquet I/O wrapper ---------------------------------------------
def write_curriculum_parquet(out_path: str, table, extra_meta: Optional[dict] = None) -> int:
    """Write the sorted table + a companion ``.manifest.json``."""
    if not _HAVE_ARROW:  # pragma: no cover
        raise RuntimeError("pyarrow required for parquet output")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")

    meta = {
        "kind": "curriculum_sorted",
        "rows": int(table.num_rows),
        "columns": list(table.column_names),
        "sort_key": "ref_ppl ASC",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(str(out_path) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return int(table.num_rows)


# ---------- reference-model ppl (lazy torch import) -------------------------
def _load_ref_model(candidates: Sequence[str], device: str = "cpu"):  # pragma: no cover
    """Load Qwen 2.5 (or any HF causal LM) from the first working candidate.

    Lazy: only called from ``main()``. Tests inject a synthetic ``ppl_fn``
    via ``compute_curriculum_order`` / ``sort_table_by_ppl`` instead.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    last = None
    for path in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                trust_remote_code=True,
            ).to(device).eval()
            print(f"[curriculum] loaded ref model {path!r} on {device}", flush=True)
            return tok, mdl, path
        except Exception as exc:
            print(f"[curriculum] candidate {path!r} failed: {exc!r}", file=sys.stderr)
            last = exc
    raise RuntimeError(f"no ref model loadable from {list(candidates)}: {last!r}")


def _qwen_ppl_fn(model, device: str) -> Callable[[Sequence[int]], float]:  # pragma: no cover
    """Build a ``ppl_fn(ids)`` closed over a loaded HF model.

    Per-token cross-entropy averaged over valid positions, then exp'd.
    """
    import torch
    import torch.nn.functional as F

    def _ppl(ids: Sequence[int]) -> float:
        ids_list = list(int(x) for x in ids)
        if len(ids_list) < 2:
            return float("inf")
        x = torch.tensor([ids_list], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=x)
            logits = out.logits[:, :-1, :].float()
            target = x[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction="mean",
            )
        return float(math.exp(min(float(loss.item()), 50.0)))  # clamp exp

    return _ppl


# ---------- smoke fixtures (no torch) ---------------------------------------
def _smoke_table(n: int = 5):
    """Five mock rows of varying length; ppl_fn = inverse-length proxy.

    Schema: ``input_ids: list<int32>``. The smoke ``ppl_fn`` returns
    ``1000 / (len + 1)`` so longer rows are "easier" -- gives a clear
    deterministic monotone result for the post-sort assert.
    """
    if not _HAVE_ARROW:  # pragma: no cover
        raise RuntimeError("pyarrow required for smoke table")
    rows = [
        [10, 20, 30],                    # len 3 -> ppl 250
        [11],                            # len 1 -> ppl 500
        [12, 22, 32, 42, 52, 62, 72],    # len 7 -> ppl 125
        [13, 23],                        # len 2 -> ppl 333.33
        [14, 24, 34, 44, 54],            # len 5 -> ppl 166.67
    ][:n]
    return pa.table({"input_ids": rows})


def _smoke_ppl_fn(ids: Sequence[int]) -> float:
    """Deterministic mock ppl: shorter rows = harder. No torch."""
    n = len(ids)
    return 1000.0 / (n + 1)


# ---------- main ------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default=None,
                    help="Input parquet (must contain 'input_ids' column)")
    ap.add_argument("--output", default=None,
                    help="Output parquet path (sorted curriculum)")
    ap.add_argument("--ref-model", default=None,
                    help="Override reference model path/HF id; default tries "
                         "/workspace/teachers/qwen2.5-0.5b then Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cpu",
                    help="Device for ref model forward (default cpu)")
    ap.add_argument("--input-ids-col", default="input_ids",
                    help="Name of the input_ids column to score (default 'input_ids')")
    ap.add_argument("--smoke", action="store_true",
                    help="Run with 5 mocked rows + mock ppl; no torch needed")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if not _HAVE_ARROW:
        print("[curriculum] FATAL: pyarrow + numpy required", file=sys.stderr)
        return 2

    if args.smoke:
        # Smoke path: no torch, no real input file. Output to args.output if
        # provided, else /tmp/curriculum_smoke.parquet.
        out_path = args.output or str(
            Path(args.input or "").parent / "curriculum_smoke.parquet"
        ) if args.input else "curriculum_smoke.parquet"
        if args.output:
            out_path = args.output
        table = _smoke_table()
        sorted_table = sort_table_by_ppl(table, _smoke_ppl_fn, args.input_ids_col)
        n = write_curriculum_parquet(
            out_path, sorted_table,
            extra_meta={"smoke": True, "ref_model": "mock"},
        )
        print(
            f"[curriculum] smoke: wrote {n} rows -> {out_path} "
            f"(ppl range "
            f"{sorted_table.column('ref_ppl')[0].as_py():.2f}.."
            f"{sorted_table.column('ref_ppl')[-1].as_py():.2f})",
            flush=True,
        )
        return 0

    # Real path: requires --input + --output + torch + transformers.
    if not args.input or not args.output:
        print(
            "[curriculum] FATAL: --input and --output required (or use --smoke)",
            file=sys.stderr,
        )
        return 2

    # Lazy heavy imports.
    candidates = (args.ref_model,) if args.ref_model else DEFAULT_REF_CANDIDATES
    try:  # pragma: no cover -- depends on host torch + transformers
        _, model, src = _load_ref_model(candidates, device=args.device)
    except Exception as exc:
        print(f"[curriculum] FATAL: ref model load failed: {exc!r}", file=sys.stderr)
        return 1

    table = pq.read_table(args.input)  # pragma: no cover (live)
    ppl_fn = _qwen_ppl_fn(model, args.device)  # pragma: no cover (live)
    t0 = time.time()  # pragma: no cover
    sorted_table = sort_table_by_ppl(table, ppl_fn, args.input_ids_col)  # pragma: no cover
    n = write_curriculum_parquet(  # pragma: no cover
        args.output, sorted_table,
        extra_meta={
            "input": str(args.input),
            "ref_model_source": src,
            "device": args.device,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
    )
    print(  # pragma: no cover
        f"[curriculum] wrote {n} rows sorted by ref_ppl ASC -> {args.output} "
        f"(ref={src}, device={args.device})",
        flush=True,
    )
    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
