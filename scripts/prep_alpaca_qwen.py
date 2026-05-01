"""prep_alpaca_qwen — tokenize Stanford-Alpaca-format JSON to SFT parquet.

Output schema:
    input_ids:  list[int32]  full prompt+response token ids (no padding)
    loss_mask:  list[int8]   1 = response token (compute loss), 0 = prompt

Template (matches what scripts/chat_repl.py emits):

    ### Instruction:
    {instruction}
    [### Input:\n{input}\n]    # only when input is non-empty
    ### Response:
    {output}<|im_end|>

`<|im_end|>` is the Qwen-tokenizer EOS marker; if not present in vocab we
fall back to the tokenizer's `eos_token`.

Examples that exceed --max-seq are skipped (with a counter).

Usage:
    python scripts/prep_alpaca_qwen.py \
        --alpaca-en /workspace/data/alpaca_zh/alpaca_zh.json \
        --alpaca-zh /workspace/data/alpaca_zh/alpaca_zh.json \
        --out /workspace/data/alpaca_sft/alpaca.parquet \
        --tokenizer-path /workspace/teachers/qwen2.5-0.5b \
        --max-seq 1024
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n"
PROMPT_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n### Input:\n{input}\n"
RESPONSE_PREFIX = "### Response:\n"
END_MARK = "<|im_end|>"


def _load_alpaca(path: str | None) -> list[dict]:
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[prep] alpaca file not found, skipping: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list at {path}, got {type(data).__name__}")
    print(f"[prep] loaded {len(data):,} examples from {path}")
    return data


def _build_text_pair(ex: dict) -> tuple[str, str]:
    """Return (prompt_only_str, full_str) so we can mask prompt tokens."""
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    if inp:
        head = PROMPT_INPUT_TEMPLATE.format(instruction=instr, input=inp)
    else:
        head = PROMPT_TEMPLATE.format(instruction=instr)
    prompt = head + RESPONSE_PREFIX
    full = prompt + out + END_MARK
    return prompt, full


def _resolve_end_id(tok) -> int:
    # try the literal `<|im_end|>` marker (Qwen2 has it)
    try:
        ids = tok.encode(END_MARK, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    except Exception:
        pass
    # fall back to eos
    return int(getattr(tok, "eos_token_id", None) or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca-en", type=str, default="",
                    help="Stanford-Alpaca-format JSON list of {instruction,input,output}")
    ap.add_argument("--alpaca-zh", type=str, default="",
                    help="Same format; merged with alpaca-en (en first then zh).")
    ap.add_argument("--out", type=str, required=True,
                    help="output .parquet path")
    ap.add_argument("--tokenizer-path", type=str, required=True,
                    help="HF AutoTokenizer-loadable path or hub id (e.g. /workspace/teachers/qwen2.5-0.5b)")
    ap.add_argument("--max-seq", type=int, default=1024,
                    help="drop examples whose tokenized length exceeds this")
    args = ap.parse_args()

    examples: list[dict] = []
    examples += _load_alpaca(args.alpaca_en)
    examples += _load_alpaca(args.alpaca_zh)
    if not examples:
        raise SystemExit("no alpaca examples loaded; pass --alpaca-en and/or --alpaca-zh")

    # tokenizer (Qwen-compatible via AutoTokenizer)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    end_id = _resolve_end_id(tok)
    print(f"[prep] tokenizer={args.tokenizer_path} end_id={end_id} (<|im_end|>)")

    rows_ids: list[list[int]] = []
    rows_mask: list[list[int]] = []
    n_skip_long = 0
    n_skip_empty = 0

    for i, ex in enumerate(examples):
        prompt_str, full_str = _build_text_pair(ex)
        prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
        full_ids = tok.encode(full_str, add_special_tokens=False)
        # full_ids may not have end_id appended cleanly if tokenizer mangles "<|im_end|>"
        if not full_ids or len(full_ids) <= len(prompt_ids):
            n_skip_empty += 1
            continue
        if len(full_ids) > args.max_seq:
            n_skip_long += 1
            continue
        # loss_mask: 0 over prompt tokens, 1 over response tokens
        mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        assert len(mask) == len(full_ids)
        rows_ids.append([int(x) for x in full_ids])
        rows_mask.append(mask)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "input_ids": pa.array(rows_ids, type=pa.list_(pa.int32())),
        "loss_mask": pa.array(rows_mask, type=pa.list_(pa.int8())),
    })
    pq.write_table(table, args.out, compression="zstd")

    n_resp = sum(sum(m) for m in rows_mask)
    n_total = sum(len(m) for m in rows_mask)
    print(f"[prep] wrote {len(rows_ids):,} examples to {args.out}")
    print(f"[prep] skipped: {n_skip_long:,} too long, {n_skip_empty:,} empty/degenerate")
    print(f"[prep] tokens: {n_total:,} total, {n_resp:,} response (loss-bearing) "
          f"= {n_resp/max(n_total,1):.1%}")


if __name__ == "__main__":
    main()
