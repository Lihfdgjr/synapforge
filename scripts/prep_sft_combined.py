"""prep_sft_combined -- unify and tokenize all SFT corpora into a single parquet.

Reads every corpus produced by ``scripts/download_sft_extended.sh`` (plus the
older ``scripts/download_alpaca.sh`` outputs) and emits one parquet file with
``input_ids`` + ``loss_mask`` columns ready for the SFT trainer.

Common schema (all sources are normalised into this):

    {
        instruction: str,             # the user turn
        input:       Optional[str],   # extra context, may be empty
        output:      str,             # the assistant turn (target)
        source:      str,             # "alpaca_en" / "sharegpt_zh" / ...
        language:    str,             # "en" / "zh" / "code" / "math" / "mix"
    }

Per-source upsampling/downsampling controls language balance and lets us
under-weight noisy corpora (sharegpt-zh, wizard-zh) without dropping them
entirely. See ``DEFAULT_RATIOS`` below.

Template (matches ``scripts/chat_repl.py`` and ``prep_alpaca_qwen.py``):

    ### Instruction:
    {instruction}
    [### Input:\n{input}\n]      # only if input is non-empty
    ### Response:
    {output}<|im_end|>

The ``loss_mask`` is 1 only on response tokens, 0 on the prompt scaffold.

Length filter: 64 <= tokenized_len <= 1024 (default --max-seq).
Examples shorter than 64 tokens are dropped (too noisy for chat training).

Usage:
    python scripts/prep_sft_combined.py \
        --in-root /workspace/data/sft \
        --out /workspace/data/sft/combined.parquet \
        --tokenizer /workspace/teachers/qwen2.5-0.5b \
        --max-seq 1024
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n"
PROMPT_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n### Input:\n{input}\n"
RESPONSE_PREFIX = "### Response:\n"
END_MARK = "<|im_end|>"

# Default per-source mixing ratios. Multiplier applied to that source's
# example count before the global cap (--max-examples). Values >1 upsample,
# values <1 downsample.
DEFAULT_RATIOS = {
    "alpaca_en":       1.0,
    "alpaca_zh":       1.5,
    "lima":            2.0,   # tiny but very high quality, upsample
    "sharegpt_zh":     0.8,   # noisy, slight downsample
    "coig":            1.0,
    "oa_zh":           1.0,
    "wizard_zh":       0.5,   # repetitive evol-instruct, heavy downsample
    "gsm8k_cot":       1.0,
    "codealpaca":      0.7,
    "self_instruct":   3.0,   # only ~200 seeds, upsample 3x
    "math_qa":         1.0,
    "tool_use_traces": 1.0,   # synthetic agent traces from synapforge.data.tool_use_traces
}

# Source-name -> language tag for mix reporting.
LANG_MAP = {
    "alpaca_en":       "en",
    "alpaca_zh":       "zh",
    "lima":            "en",
    "sharegpt_zh":     "zh",
    "coig":            "zh",
    "oa_zh":           "zh",
    "wizard_zh":       "zh",
    "gsm8k_cot":       "math",
    "codealpaca":      "code",
    "self_instruct":   "en",
    "math_qa":         "math",
    "tool_use_traces": "tool",
}


# --------------------------------------------------------------------------- #
# Per-corpus normalisers                                                       #
# --------------------------------------------------------------------------- #

def _to_alpaca(ex: dict) -> dict | None:
    """Coerce to {instruction, input, output}; return None on bad row."""
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or ex.get("response") or "").strip()
    if not instr or not out:
        return None
    return {"instruction": instr, "input": inp, "output": out}


def _from_sharegpt(ex: dict) -> dict | None:
    """ShareGPT format: {conversations: [{from: human/gpt, value: ...}, ...]}.

    We pick the first human/gpt pair; multi-turn handling is left to a
    later trainer iteration.
    """
    convs = ex.get("conversations") or ex.get("conv") or []
    if not isinstance(convs, list) or len(convs) < 2:
        return None
    user = next((c.get("value") for c in convs
                 if c.get("from") in ("human", "user")), None)
    asst = next((c.get("value") for c in convs
                 if c.get("from") in ("gpt", "assistant", "bot")), None)
    if not user or not asst:
        return None
    return {"instruction": user.strip(), "input": "", "output": asst.strip()}


def _from_oa(ex: dict) -> dict | None:
    """OpenAssistant: tree of messages; we want the (prompter, assistant) root pair."""
    role = ex.get("role")
    if role == "prompter" and ex.get("text"):
        # We can't pair it without the tree context; treated as unmatched.
        return None
    if role == "assistant":
        prompt = ex.get("parent_text") or ex.get("prompt")
        if prompt and ex.get("text"):
            return {"instruction": prompt.strip(), "input": "",
                    "output": ex["text"].strip()}
    # Fallback: try direct alpaca-shape.
    return _to_alpaca(ex)


def _from_gsm8k(ex: dict) -> dict | None:
    q = (ex.get("question") or ex.get("instruction") or "").strip()
    a = (ex.get("answer") or ex.get("output") or "").strip()
    if not q or not a:
        return None
    return {"instruction": q, "input": "", "output": a}


def _normalise(source: str, ex: dict) -> dict | None:
    if source == "sharegpt_zh":
        return _from_sharegpt(ex) or _to_alpaca(ex)
    if source == "oa_zh":
        return _from_oa(ex) or _to_alpaca(ex)
    if source == "gsm8k_cot":
        return _from_gsm8k(ex)
    return _to_alpaca(ex)


def _load_one(path: Path, source: str) -> list[dict]:
    if not path.exists():
        print(f"[prep] {source}: missing {path}, skip")
        return []
    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    text_stripped = text.lstrip()
    try:
        if text_stripped.startswith("["):
            raw = json.loads(text)
        elif text_stripped.startswith("{"):
            # Either a single dict or JSONL where line 1 starts with {.
            try:
                obj = json.loads(text)
                raw = obj if isinstance(obj, list) else [obj]
            except json.JSONDecodeError:
                raw = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        print(f"[prep] {source}: JSON parse failed ({e}); skipping")
        return []
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        norm = _normalise(source, ex)
        if norm is None:
            continue
        norm["source"] = source
        norm["language"] = LANG_MAP.get(source, "mix")
        rows.append(norm)
    print(f"[prep] {source}: loaded {len(rows):,} normalised examples from {path}")
    return rows


def _resolve_layout(in_root: Path) -> dict[str, Path]:
    """Map source name -> path of its train.json (or .jsonl)."""
    layout: dict[str, Path] = {}
    # Old alpaca layout: in_root/alpaca_en.json, alpaca_zh.json, lima.json, math_qa.json
    for legacy in ("alpaca_en", "alpaca_zh", "lima", "math_qa"):
        p_json = in_root / f"{legacy}.json"
        if p_json.exists():
            layout[legacy] = p_json
    # New extended layout: in_root/<name>/train.json
    for name in DEFAULT_RATIOS:
        p_dir = in_root / name / "train.json"
        if p_dir.exists():
            layout[name] = p_dir
    return layout


# --------------------------------------------------------------------------- #
# Templating + tokenisation                                                    #
# --------------------------------------------------------------------------- #

def _build_text_pair(ex: dict) -> tuple[str, str]:
    instr = ex["instruction"].strip()
    inp = (ex.get("input") or "").strip()
    out = ex["output"].strip()
    if inp:
        head = PROMPT_INPUT_TEMPLATE.format(instruction=instr, input=inp)
    else:
        head = PROMPT_TEMPLATE.format(instruction=instr)
    prompt = head + RESPONSE_PREFIX
    full = prompt + out + END_MARK
    return prompt, full


def _resolve_end_id(tok) -> int:
    try:
        ids = tok.encode(END_MARK, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    except Exception:
        pass
    return int(getattr(tok, "eos_token_id", None) or 0)


def _resample(rows: list[dict], multiplier: float, rng: random.Random) -> list[dict]:
    if multiplier == 1.0 or not rows:
        return list(rows)
    full = int(multiplier)
    frac = multiplier - full
    out = list(rows) * full
    if frac > 0:
        k = int(round(frac * len(rows)))
        out += rng.sample(rows, min(k, len(rows)))
    elif full == 0 and 0 < multiplier < 1:
        k = int(round(multiplier * len(rows)))
        out = rng.sample(rows, min(k, len(rows)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=str, default="/workspace/data/sft",
                    help="Root dir from download_sft_extended.sh + download_alpaca.sh")
    ap.add_argument("--out", type=str, required=True,
                    help="Output parquet path")
    ap.add_argument("--tokenizer", type=str, required=True,
                    help="HF AutoTokenizer-loadable path")
    ap.add_argument("--max-seq", type=int, default=1024,
                    help="upper token length filter (drop longer)")
    ap.add_argument("--min-seq", type=int, default=64,
                    help="lower token length filter (drop shorter)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-examples", type=int, default=0,
                    help="cap final example count (0 = no cap)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_root = Path(args.in_root)
    if not in_root.exists():
        raise SystemExit(f"--in-root does not exist: {in_root}")

    layout = _resolve_layout(in_root)
    if not layout:
        raise SystemExit(f"no SFT corpora found under {in_root}; "
                         "run scripts/download_sft_extended.sh first")
    print(f"[prep] discovered {len(layout)} corpora: {sorted(layout)}")

    pooled: list[dict] = []
    for source, path in sorted(layout.items()):
        rows = _load_one(path, source)
        ratio = DEFAULT_RATIOS.get(source, 1.0)
        sampled = _resample(rows, ratio, rng)
        if ratio != 1.0:
            print(f"[prep] {source}: {len(rows):,} -> {len(sampled):,} after x{ratio:.2f}")
        pooled.extend(sampled)
    rng.shuffle(pooled)
    print(f"[prep] pooled {len(pooled):,} examples across {len(layout)} sources")

    if args.max_examples and args.max_examples < len(pooled):
        pooled = pooled[: args.max_examples]
        print(f"[prep] capped to {len(pooled):,} (--max-examples)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    end_id = _resolve_end_id(tok)
    print(f"[prep] tokenizer={args.tokenizer} end_id={end_id} (<|im_end|>)")

    rows_ids: list[list[int]] = []
    rows_mask: list[list[int]] = []
    rows_lang: list[str] = []
    rows_source: list[str] = []
    n_skip_long = 0
    n_skip_short = 0
    n_skip_empty = 0

    for ex in pooled:
        prompt_str, full_str = _build_text_pair(ex)
        prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
        full_ids = tok.encode(full_str, add_special_tokens=False)
        if not full_ids or len(full_ids) <= len(prompt_ids):
            n_skip_empty += 1
            continue
        if len(full_ids) < args.min_seq:
            n_skip_short += 1
            continue
        if len(full_ids) > args.max_seq:
            n_skip_long += 1
            continue
        mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        assert len(mask) == len(full_ids)
        rows_ids.append([int(x) for x in full_ids])
        rows_mask.append(mask)
        rows_lang.append(ex["language"])
        rows_source.append(ex["source"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "input_ids": pa.array(rows_ids, type=pa.list_(pa.int32())),
        "loss_mask": pa.array(rows_mask, type=pa.list_(pa.int8())),
        "source":    pa.array(rows_source, type=pa.string()),
        "language":  pa.array(rows_lang, type=pa.string()),
    })
    pq.write_table(table, args.out, compression="zstd")

    n_resp = sum(sum(m) for m in rows_mask)
    n_total = sum(len(m) for m in rows_mask)
    print(f"[prep] wrote {len(rows_ids):,} examples to {args.out}")
    print(f"[prep] skipped: {n_skip_long:,} too long, "
          f"{n_skip_short:,} too short, {n_skip_empty:,} empty")
    print(f"[prep] tokens: {n_total:,} total, "
          f"{n_resp:,} response (loss-bearing) = {n_resp / max(n_total, 1):.1%}")

    # Per-language balance verifier.
    from collections import Counter
    lang_counts = Counter(rows_lang)
    src_counts = Counter(rows_source)
    print("[prep] language mix:")
    for lang, n in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:>5}: {n:>8,} ({n / len(rows_lang):.1%})")
    print("[prep] source mix:")
    for src, n in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:>16}: {n:>8,} ({n / len(rows_source):.1%})")


if __name__ == "__main__":
    main()
