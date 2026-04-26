"""sf.eval.compare_chat — head-to-head: synapforge 100M vs mscfc adv29 baseline.

Runs 5 standard prompts through both backends, computes validation perplexity
on /workspace/data/wt103_raw/validation.parquet, and writes a side-by-side
artifact at /workspace/runs/chat_compare.txt (and a copy under
/workspace/investor_demo_output/synapforge_vs_mscfc_chat.md).

Both models share the same gpt2 tokenizer, so the prompts/completions are
directly comparable.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import List, Tuple

_WS = "/workspace"
if _WS not in sys.path:
    sys.path.insert(0, _WS)

import torch
import torch.nn.functional as F

from synapforge.eval.generate import (  # type: ignore
    generate,
    load_mscfc,
    load_synapforge,
    load_tokenizer,
)


PROMPTS: List[str] = [
    "The capital of France is",
    "To compute the factorial of 5 in Python, you write",
    "Once upon a time, in a small village,",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Photosynthesis is the process by which plants",
]

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
SF_CKPT_DIR = "/workspace/runs/synapforge_100m"
MSCFC_CKPT = "/workspace/runs/step_001250.pt"  # adv29 32M, ppl ~565
OUT_TXT = "/workspace/runs/chat_compare.txt"
OUT_MD = "/workspace/investor_demo_output/synapforge_vs_mscfc_chat.md"
VAL_PARQUET = "/workspace/data/wt103_raw/validation.parquet"

GEN_KW = dict(max_new=60, top_k=40, top_p=0.9, temperature=0.7)


# ------------------------------------------------------------- helpers


def _latest_sf_ckpt() -> str:
    """Find latest synapforge ckpt; return empty string if none saved yet."""
    if not os.path.isdir(SF_CKPT_DIR):
        return ""
    cands = []
    for name in os.listdir(SF_CKPT_DIR):
        if name.endswith(".pt"):
            cands.append(os.path.join(SF_CKPT_DIR, name))
    cands.sort()
    return cands[-1] if cands else ""


def _count_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def _val_loader(tokenizer, seq_len: int = 256, batch_size: int = 4,
                max_batches: int = 32):
    """Yield (input, target) pairs from validation.parquet, gpt2-tokenized."""
    import pyarrow.parquet as pq

    buf: List[int] = []
    table = pq.read_table(VAL_PARQUET)
    rows = table.to_pylist()
    eos = tokenizer.eos_token_id or 50256
    for r in rows:
        text = r.get("text") or ""
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        buf.extend(ids + [eos])
    # pack into (batch_size, seq_len) tiles
    block = seq_len + 1
    n_total = len(buf) // block
    n_total = min(n_total, batch_size * max_batches)
    if n_total < 1:
        return
    chunks = []
    for i in range(n_total):
        chunks.append(buf[i * block:(i + 1) * block])
    for i in range(0, n_total, batch_size):
        batch = chunks[i:i + batch_size]
        if len(batch) < 1:
            break
        t = torch.tensor(batch, dtype=torch.long)
        yield t[:, :-1], t[:, 1:]


@torch.no_grad()
def compute_val_ppl(model, tokenizer, label: str, max_seq: int = 256,
                    max_batches: int = 32) -> float:
    model.eval()
    dev = next(model.parameters()).device
    tot_loss = 0.0
    tot_tok = 0
    pos_limit = getattr(model, "max_seq", None) or max_seq
    seq_len = min(max_seq, pos_limit) - 1  # leave room for shift
    for inp, tgt in _val_loader(tokenizer, seq_len=seq_len, batch_size=4,
                                max_batches=max_batches):
        inp = inp.to(dev)
        tgt = tgt.to(dev)
        try:
            out = model(inp)
        except TypeError:
            out = model(tokens=inp)
        if isinstance(out, tuple):
            out = out[0]
        loss = F.cross_entropy(
            out.float().reshape(-1, out.size(-1)),
            tgt.reshape(-1),
            reduction="sum",
        )
        tot_loss += float(loss.item())
        tot_tok += int(tgt.numel())
    if tot_tok == 0:
        return float("nan")
    avg_loss = tot_loss / tot_tok
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    print(f"[ppl] {label}: avg_loss={avg_loss:.4f} ppl={ppl:.2f} on {tot_tok} tokens")
    return ppl


# ------------------------------------------------------------- main


def main() -> None:
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)

    print(f"[compare] device={DEVICE} torch={torch.__version__}")
    tok = load_tokenizer("gpt2")

    # --- synapforge ---
    sf_ckpt = _latest_sf_ckpt()
    if sf_ckpt:
        print(f"[compare] synapforge: loading ckpt {sf_ckpt}")
        sf_model, sf_src = load_synapforge(ckpt_path=sf_ckpt, device=DEVICE)
    else:
        print(f"[compare] synapforge: no ckpt in {SF_CKPT_DIR}, using fresh-init "
              f"(warmstart from {MSCFC_CKPT}, best-effort overlap copy)")
        sf_model, sf_src = load_synapforge(
            ckpt_path=None, fresh_warmstart_from=MSCFC_CKPT, device=DEVICE,
        )
    sf_params = _count_params(sf_model)
    print(f"[compare] synapforge: params={sf_params:,} src={sf_src}")

    # --- mscfc baseline ---
    print(f"[compare] mscfc: loading {MSCFC_CKPT}")
    mscfc_model, _ = load_mscfc(MSCFC_CKPT, device=DEVICE)
    mscfc_params = _count_params(mscfc_model)
    print(f"[compare] mscfc: params={mscfc_params:,}")

    # --- val ppl ---
    sf_ppl = float("nan")
    mscfc_ppl = float("nan")
    try:
        sf_ppl = compute_val_ppl(sf_model, tok, "synapforge")
    except Exception as e:
        print(f"[ppl] synapforge FAILED: {e}\n{traceback.format_exc()}")
    try:
        mscfc_ppl = compute_val_ppl(mscfc_model, tok, "mscfc")
    except Exception as e:
        print(f"[ppl] mscfc FAILED: {e}\n{traceback.format_exc()}")

    # --- generation ---
    sf_completions: List[Tuple[str, str, float]] = []
    mscfc_completions: List[Tuple[str, str, float]] = []

    for p in PROMPTS:
        print(f"\n[gen] prompt={p!r}")
        # synapforge
        try:
            t0 = time.time()
            out = generate(sf_model, tok, p, device=DEVICE, **GEN_KW)
            sf_completions.append((p, out, time.time() - t0))
            print(f"  sf [{sf_completions[-1][2]:.1f}s] -> {out[:120]!r}")
        except Exception as e:
            sf_completions.append((p, f"<error: {e}>", 0.0))
            print(f"  sf FAILED: {e}")
        # mscfc
        try:
            t0 = time.time()
            out = generate(mscfc_model, tok, p, device=DEVICE, **GEN_KW)
            mscfc_completions.append((p, out, time.time() - t0))
            print(f"  mscfc [{mscfc_completions[-1][2]:.1f}s] -> {out[:120]!r}")
        except Exception as e:
            mscfc_completions.append((p, f"<error: {e}>", 0.0))
            print(f"  mscfc FAILED: {e}")

    # --- write artifacts ---
    header = (
        "synapforge vs mscfc — head-to-head completion comparison\n"
        f"  date           : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  synapforge     : {sf_params:,} params  src={sf_src}  val_ppl={sf_ppl:.2f}\n"
        f"  mscfc adv29    : {mscfc_params:,} params  ckpt={MSCFC_CKPT}  val_ppl={mscfc_ppl:.2f}\n"
        f"  sampling       : max_new=60 temperature=0.7 top_k=40 top_p=0.9\n"
        f"  tokenizer      : gpt2 (50257 vocab) — both backbones share it\n"
        f"  device         : {DEVICE}\n"
    )

    txt_lines = [header, "=" * 72, ""]
    for i, p in enumerate(PROMPTS):
        sf_p, sf_out, sf_dt = sf_completions[i]
        ms_p, ms_out, ms_dt = mscfc_completions[i]
        txt_lines.append(f"[Prompt {i + 1}] {p!r}")
        txt_lines.append("-" * 72)
        txt_lines.append(f"  synapforge ({sf_dt:.1f}s):")
        txt_lines.append(f"    {sf_out}")
        txt_lines.append("")
        txt_lines.append(f"  mscfc-adv29 ({ms_dt:.1f}s):")
        txt_lines.append(f"    {ms_out}")
        txt_lines.append("")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    # markdown
    md = []
    md.append("# synapforge vs mscfc — chat completion head-to-head\n")
    md.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on `{DEVICE}` "
              f"with gpt2 tokenizer._\n")
    md.append("\n## Models\n")
    md.append("| Model | Params | Source | Val PPL (wt103) |")
    md.append("| --- | --- | --- | --- |")
    md.append(f"| synapforge 100M | {sf_params:,} | {sf_src} | {sf_ppl:.2f} |")
    md.append(f"| mscfc adv29 (baseline) | {mscfc_params:,} | "
              f"`{MSCFC_CKPT}` | {mscfc_ppl:.2f} |")
    md.append("\n_Sampling: max_new=60, temperature=0.7, top_k=40, top_p=0.9._\n")
    for i, p in enumerate(PROMPTS):
        sf_p, sf_out, sf_dt = sf_completions[i]
        ms_p, ms_out, ms_dt = mscfc_completions[i]
        md.append(f"\n### Prompt {i + 1}\n")
        md.append("```")
        md.append(p)
        md.append("```")
        md.append(f"\n**synapforge** ({sf_dt:.1f}s)\n")
        md.append("```")
        md.append(sf_out)
        md.append("```")
        md.append(f"\n**mscfc-adv29** ({ms_dt:.1f}s)\n")
        md.append("```")
        md.append(ms_out)
        md.append("```")

    md.append("\n## Honest assessment\n")
    if (sf_ppl == sf_ppl) and (mscfc_ppl == mscfc_ppl) and mscfc_ppl > 0:  # NaN-safe
        ratio = (mscfc_ppl / sf_ppl) * 100.0 if sf_ppl > 0 else float("nan")
        md.append(
            f"- mscfc baseline ppl = **{mscfc_ppl:.0f}**, synapforge ppl = "
            f"**{sf_ppl:.0f}**.\n"
            f"- At this snapshot synapforge is roughly **{ratio:.0f}%** as "
            f"language-model coherent as the mscfc baseline (lower ppl = "
            f"more coherent; ratio is mscfc/synapforge).\n"
        )
    else:
        md.append("- ppl numbers are partial; see val_ppl in the table above.\n")
    md.append(
        "- **Both models produce wordy salad at this scale (≤1k training "
        "steps, 32–100M params).** That is expected. The point of this run "
        "is to validate that the synapforge framework can train and "
        "generate end-to-end on the same tokenizer/data as the original "
        "PyTorch mscfc, with a side-by-side artifact suitable for an "
        "investor demo.\n"
    )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # tiny machine-readable manifest for easy parse
    manifest = {
        "synapforge_params": sf_params,
        "synapforge_ppl": sf_ppl,
        "synapforge_src": sf_src,
        "mscfc_params": mscfc_params,
        "mscfc_ppl": mscfc_ppl,
        "mscfc_ckpt": MSCFC_CKPT,
        "prompts": PROMPTS,
        "synapforge_completions": [c for _, c, _ in sf_completions],
        "mscfc_completions": [c for _, c, _ in mscfc_completions],
        "sampling": GEN_KW,
        "device": DEVICE,
    }
    with open(os.path.join(os.path.dirname(OUT_MD),
                           "synapforge_vs_mscfc_chat.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[compare] wrote {OUT_TXT}")
    print(f"[compare] wrote {OUT_MD}")
    print(f"[compare] DONE.")


if __name__ == "__main__":
    main()
