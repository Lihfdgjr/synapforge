"""gen_kd_data.py — pre-generate teacher continuations for KD distillation.

Why
---
The student trainer (``train_100m_kd.py``) currently runs the teacher
*live* on the same web tokens the student sees. On weak/short pre-train
runs this lets the student collapse to TOKEN_SOUP (e.g. "the the the
...") because the teacher's signal on raw web noise is wide and the KD
loss alone isn't strong enough to lift the student off the unigram
manifold.

This script side-steps that by pre-generating teacher continuations on a
*curated* prompt set, and saving:

  - ``input_ids``     -- the prompt+completion tokens (full context)
  - ``topk_indices``  -- per-position top-K teacher token IDs (K=64 default)
  - ``topk_log_probs``-- per-position top-K teacher log-probs
  - ``seq_len`` / ``topk`` -- denormalised metadata for trainer reshape

The schema is **identical** to ``scripts/collect_kd_data.py`` output, so
the existing ``--cached-kd-parquet`` consumer code path in the trainer
(once wired) reads it without modification. We additionally write a
sidecar ``.jsonl`` with the (prompt, completion) pairs in plain text so
quality gating + the trainer's plain-text ``ParquetTokenStream`` can both
ingest it.

Diff from ``collect_kd_data.py``
--------------------------------
``collect_kd_data.py`` takes web *text* and computes top-K teacher
distributions over that text -- it caches a teacher's view of an existing
corpus.

``gen_kd_data.py`` takes *prompts* and runs ``model.generate(...)`` to
*sample* fresh continuations from the teacher (``temperature=0.7``,
``top_p=0.9``), then computes top-K teacher distributions over the
generated context. The point is to feed the student a *clean teacher
manifold* trajectory, not the raw web noise.

CLI
---
    python scripts/gen_kd_data.py \\
        --teacher /workspace/teachers/qwen2.5-0.5b \\
        --prompts /workspace/data/kd_prompts.jsonl \\
        --output  /workspace/data/kd_distill_v1.parquet \\
        --gen-tokens 256 --topk 64 --batch-size 16

Outputs
-------
    {output}                              -- parquet (KD top-K cache schema)
    {output}.manifest.json                -- metadata sidecar
    {output}.samples.jsonl                -- (prompt, completion) text rows
                                              (one JSON per row, all rows)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Defer torch / transformers imports to ``main()`` so this file imports
# cleanly on torch-less dev boxes (matches collect_kd_data.py pattern).
try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False


DEFAULT_TEACHER_CANDIDATES = (
    "/workspace/teachers/qwen2.5-0.5b",
    "Qwen/Qwen2.5-0.5B",
)


# ---- prompt loader ----------------------------------------------------------
def load_prompts(path: str) -> list[dict]:
    """Read kd_prompts.jsonl, return list of dicts with at least 'prompt'."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict) and d.get("prompt"):
                rows.append(d)
    return rows


# ---- parquet writer (top-K cache schema, identical to collect_kd_data) ----
def write_kd_distill_parquet(
    out_path: str,
    input_ids: Sequence[Sequence[int]],
    topk_indices: Sequence[object],
    topk_log_probs: Sequence[object],
    seq_lens: Sequence[int],
    extra_meta: Optional[dict] = None,
) -> int:
    if not _HAVE_ARROW:
        raise RuntimeError("pyarrow + numpy required for KD parquet output")
    n = len(input_ids)
    assert n == len(topk_indices) == len(topk_log_probs) == len(seq_lens), (
        f"row-count mismatch: ids={n} idx={len(topk_indices)} "
        f"lp={len(topk_log_probs)} sl={len(seq_lens)}"
    )

    flat_idx, flat_lp, ks = [], [], []
    for i in range(n):
        idx_arr = np.asarray(topk_indices[i], dtype=np.int32).reshape(-1)
        lp_arr = np.asarray(topk_log_probs[i], dtype=np.float16).reshape(-1)
        flat_idx.append(idx_arr.tolist())
        flat_lp.append(lp_arr.tolist())
        ks.append(int(idx_arr.size // max(seq_lens[i], 1)))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "input_ids": [list(map(int, x)) for x in input_ids],
        "topk_indices": flat_idx,
        "topk_log_probs": flat_lp,
        "seq_len": [int(x) for x in seq_lens],
        "topk": ks,
    })
    pq.write_table(table, out_path, compression="zstd")

    meta = {
        "kind": "kd_distill_cache",
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


# ---- teacher loader ---------------------------------------------------------
def _load_teacher(candidates: Sequence[str], device: str = "cuda"):  # pragma: no cover
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    last_err = None
    for path in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, padding_side="left",
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                trust_remote_code=True,
            ).to(device).eval()
            print(f"[gen_kd] loaded teacher from {path!r} on {device}", flush=True)
            # Pad token: Qwen 2.5 doesn't define one; reuse eos. With
            # padding_side='left' the pads are at the front, so the
            # generation continues correctly from the *right* end of
            # each row (the actual prompt boundary). Right-padding
            # would force every row to generate from a pad token,
            # which produces empty completions for short prompts.
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            return tok, mdl, path
        except Exception as exc:
            print(f"[gen_kd] candidate {path!r} failed: {exc!r}", file=sys.stderr)
            last_err = exc
    raise RuntimeError(f"no teacher loadable: {last_err!r}")


# ---- generation kernel ------------------------------------------------------
def _generate_batch(
    tok, mdl, prompts: list[str],
    gen_tokens: int, topk: int,
    temperature: float, top_p: float,
    max_prompt_len: int,
    device: str,
) -> Tuple[list, list, list, list, list, list]:  # pragma: no cover -- live torch
    """Generate continuations + harvest top-K teacher distributions.

    Returns parallel lists per row:
        input_ids, topk_idx, topk_lp, seq_lens,
        completion_text, prompt_token_lens
    """
    import torch
    import torch.nn.functional as F

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    # With padding_side='left', the leading region of each row is pads
    # and the real prompt content starts at index ``T_in - prompt_len``.
    # ``prompt_lens[b]`` = count of non-pad tokens (real prompt length).
    prompt_lens = attn_mask.sum(dim=1).tolist()
    T_in = input_ids.shape[1]  # padded prompt-batch length

    with torch.no_grad():
        out_ids = mdl.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=gen_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    # out_ids shape: [B, T_in + gen_steps] -- generate appends to the
    # left-padded prompt batch. Generated tokens start at column T_in
    # for every row (eos in row b => the rest of row b is pad_token_id).
    full_ids = out_ids  # [B, T_total]
    # Build a mask that's 1 over (real prompt + actual generated tokens),
    # 0 over (left pads + post-eos right pads). To distinguish post-eos
    # pads from intra-prompt pads, we OR (col >= T_in) with (col_in_prompt
    # AND not pad). The simpler, correct approach: keep the original
    # ``attn_mask`` for the prompt region, and for the generated region
    # mark a token as valid until we hit the first eos OR a run of
    # consecutive pad tokens.
    B, T_total = full_ids.shape
    gen_region = full_ids[:, T_in:]  # [B, T_gen]
    is_eos = (gen_region == tok.eos_token_id)
    # First-eos position per row; positions strictly AFTER first-eos are
    # ignored. Tokens AT first-eos are kept (the eos itself is part of
    # the generated stream).
    eos_idx = is_eos.float().argmax(dim=1)  # [B] (0 if no eos found)
    no_eos_mask = (~is_eos.any(dim=1)).long()  # 1 where no eos found
    # Effective generated length per row.
    gen_T = gen_region.shape[1]
    eff_gen_len = torch.where(no_eos_mask.bool(),
                              torch.full_like(eos_idx, gen_T),
                              eos_idx + 1)
    # Build full-sequence mask: 1 for real prompt tokens (use attn_mask),
    # 1 for valid generated tokens (col_in_gen < eff_gen_len), 0 elsewhere.
    full_attn = torch.zeros_like(full_ids)
    full_attn[:, :T_in] = attn_mask
    cols = torch.arange(gen_T, device=full_ids.device).unsqueeze(0).expand(B, -1)
    gen_mask = (cols < eff_gen_len.unsqueeze(1)).long()
    full_attn[:, T_in:] = gen_mask

    # Re-run forward to extract teacher top-K over the FULL sequence.
    with torch.no_grad():
        out = mdl(input_ids=full_ids, attention_mask=full_attn)
        logits = out.logits  # [B, T, V]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        top_lp, top_idx = log_probs.topk(topk, dim=-1)

    full_ids_np = full_ids.cpu().numpy().astype(np.int32)
    full_attn_np = full_attn.cpu().numpy().astype(np.int32)
    top_idx_np = top_idx.cpu().numpy().astype(np.int32)
    top_lp_np = top_lp.cpu().numpy().astype(np.float16)

    in_ids_rows: list = []
    idx_rows: list = []
    lp_rows: list = []
    sl_rows: list = []
    completion_text: list = []

    for b in range(full_ids_np.shape[0]):
        # The valid contiguous range is [pad_start_b, T_in + eff_gen_len_b).
        # pad_start_b = T_in - prompt_lens[b] (left-padding).
        plen_b = int(prompt_lens[b])
        pad_start_b = T_in - plen_b
        gen_end_b = T_in + int(eff_gen_len[b].item())
        valid = gen_end_b - pad_start_b
        if valid <= 1:
            in_ids_rows.append([])
            idx_rows.append(np.zeros((0, topk), dtype=np.int32))
            lp_rows.append(np.zeros((0, topk), dtype=np.float16))
            sl_rows.append(0)
            completion_text.append("")
            continue
        # Slice contiguous valid window.
        in_ids_rows.append(full_ids_np[b, pad_start_b:gen_end_b].tolist())
        idx_rows.append(top_idx_np[b, pad_start_b:gen_end_b, :])
        lp_rows.append(top_lp_np[b, pad_start_b:gen_end_b, :])
        sl_rows.append(valid)
        # Decode just the generated tail (positions T_in:gen_end_b in the
        # full row) for sample inspection.
        gen_ids = full_ids_np[b, T_in:gen_end_b].tolist()
        completion_text.append(tok.decode(gen_ids, skip_special_tokens=True))

    return in_ids_rows, idx_rows, lp_rows, sl_rows, completion_text, prompt_lens


# ---- main ------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--teacher", default=None,
                    help="Override teacher path/HF id; else tries default candidates")
    ap.add_argument("--prompts", required=True,
                    help="kd_prompts.jsonl (one JSON object per row, must have 'prompt')")
    ap.add_argument("--output", required=True,
                    help="Output parquet path")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=256,
                    help="max_new_tokens per prompt (default 256)")
    ap.add_argument("--max-prompt-len", type=int, default=128,
                    help="Truncate prompts to this many tokens (default 128)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-prompts", type=int, default=0,
                    help="0 = process all; else cap (smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="Override max-prompts to 8 + smaller batch for quick check")
    ap.add_argument("--report-every", type=int, default=50,
                    help="Print throughput every N batches")
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        args.max_prompts = 8
        args.batch_size = min(args.batch_size, 4)
    if not _HAVE_ARROW:
        print("[gen_kd] FATAL: pyarrow + numpy required", file=sys.stderr)
        return 2

    import torch  # noqa: E402

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    candidates = (args.teacher,) if args.teacher else DEFAULT_TEACHER_CANDIDATES

    prompts_data = load_prompts(args.prompts)
    if args.max_prompts and args.max_prompts > 0:
        prompts_data = prompts_data[:args.max_prompts]
    print(f"[gen_kd] loaded {len(prompts_data)} prompts from {args.prompts}", flush=True)
    if not prompts_data:
        print("[gen_kd] no prompts; aborting", file=sys.stderr)
        return 1

    tok, mdl, src = _load_teacher(candidates, device=device)

    # Outputs
    all_input_ids: list = []
    all_idx: list = []
    all_lp: list = []
    all_sl: list = []
    samples_path = str(args.output) + ".samples.jsonl"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    samples_f = open(samples_path, "w", encoding="utf-8")

    t0 = time.time()
    n_batches = 0
    n_total_tokens = 0

    bs = args.batch_size
    try:
        for start in range(0, len(prompts_data), bs):
            batch = prompts_data[start:start + bs]
            prompts = [r["prompt"] for r in batch]
            try:
                ids, idx, lp, sl, comps, plens = _generate_batch(
                    tok, mdl, prompts,
                    gen_tokens=args.gen_tokens,
                    topk=args.topk,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_prompt_len=args.max_prompt_len,
                    device=device,
                )
            except RuntimeError as e:
                # OOM safety: if a batch blows VRAM, retry with bs=1.
                if "out of memory" in str(e).lower() and bs > 1:
                    print(f"[gen_kd] OOM at bs={bs}, falling back to bs=1 for this batch", file=sys.stderr)
                    torch.cuda.empty_cache()
                    ids, idx, lp, sl, comps, plens = [], [], [], [], [], []
                    for j in range(len(prompts)):
                        ids_j, idx_j, lp_j, sl_j, comps_j, plens_j = _generate_batch(
                            tok, mdl, prompts[j:j + 1],
                            gen_tokens=args.gen_tokens,
                            topk=args.topk,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_prompt_len=args.max_prompt_len,
                            device=device,
                        )
                        ids.extend(ids_j); idx.extend(idx_j); lp.extend(lp_j)
                        sl.extend(sl_j); comps.extend(comps_j); plens.extend(plens_j)
                else:
                    raise

            for r_in, c, p_in, p_len, sl_v in zip(batch, comps, prompts, plens, sl):
                samples_f.write(json.dumps({
                    "id": r_in.get("id", ""),
                    "bucket": r_in.get("bucket", ""),
                    "lang": r_in.get("lang", ""),
                    "src": r_in.get("src", ""),
                    "prompt": p_in,
                    "completion": c,
                    "prompt_tok_len": int(p_len),
                    "total_tok_len": int(sl_v),
                }, ensure_ascii=False) + "\n")

            all_input_ids.extend(ids)
            all_idx.extend(idx)
            all_lp.extend(lp)
            all_sl.extend(sl)
            n_batches += 1
            n_total_tokens += sum(sl)

            if not args.no_progress and n_batches % args.report_every == 0:
                el = time.time() - t0
                tps = n_total_tokens / max(el, 1e-9)
                done = start + len(batch)
                pct = 100.0 * done / len(prompts_data)
                print(
                    f"[gen_kd] batch {n_batches} done={done}/{len(prompts_data)} "
                    f"({pct:.1f}%) tokens={n_total_tokens:,} "
                    f"throughput={tps:,.0f} tok/s elapsed={el:.0f}s",
                    flush=True,
                )
    finally:
        samples_f.close()

    # Drop fully-empty rows (OOM holes). Keep input order.
    keep_idx = [i for i, sl in enumerate(all_sl) if sl > 1]
    all_input_ids = [all_input_ids[i] for i in keep_idx]
    all_idx = [all_idx[i] for i in keep_idx]
    all_lp = [all_lp[i] for i in keep_idx]
    all_sl = [all_sl[i] for i in keep_idx]

    if not all_input_ids:
        print("[gen_kd] no rows produced", file=sys.stderr)
        return 1

    n = write_kd_distill_parquet(
        args.output,
        all_input_ids, all_idx, all_lp, all_sl,
        extra_meta=dict(
            teacher_source=src,
            topk=args.topk,
            seq_len_max=int(max(all_sl)),
            gen_tokens=args.gen_tokens,
            max_prompt_len=args.max_prompt_len,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            device=device,
            rows_emitted=len(keep_idx),
            rows_dropped=len(all_sl) - len(keep_idx) if False else 0,
            prompt_set=str(args.prompts),
            elapsed_seconds=round(time.time() - t0, 2),
        ),
    )

    el = time.time() - t0
    completion_tokens = sum(all_sl) - sum(plens for plens in [args.max_prompt_len] * 0)  # rough
    # Better: total tokens including prompts emitted.
    print(
        f"[gen_kd] wrote {n:,} rows -> {args.output} "
        f"total_tokens={sum(all_sl):,} "
        f"avg_seq_len={sum(all_sl)/max(n,1):.1f} "
        f"throughput={sum(all_sl)/max(el,1e-9):,.0f} tok/s "
        f"elapsed={el:.0f}s",
        flush=True,
    )
    print(f"[gen_kd] samples sidecar -> {samples_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
