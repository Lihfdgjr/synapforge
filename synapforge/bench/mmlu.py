"""MMLU (Massive Multitask Language Understanding) accuracy.

57 subjects, 4-choice exam questions. Standard format: prompt the model with the
question and labelled choices, take the highest-likelihood letter (A/B/C/D) at
the next-token position.

We do NOT generate text — we read the logits directly. This is much faster and
matches the standard lm-eval-harness recipe.

Score: macro accuracy = mean over subjects of mean over questions per subject.

CLI:
    python -m synapforge.bench.mmlu --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \\
        --out mmlu.json [--n 200] [--subjects=abstract_algebra,...]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# All 57 MMLU subjects in canonical order.
MMLU_SUBJECTS: List[str] = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

PREAMBLE = (
    "The following is a multiple-choice question. Reply with just one letter "
    "(A, B, C, or D) for the correct answer.\n\n"
)

# ---------------------------------------------------------------- dataset


def _load_subject(subject: str, local_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Return list of {question, choices: [A,B,C,D], answer: 'A'|'B'|...}."""
    # Try local first.
    if local_dir:
        p = Path(local_dir) / f"{subject}.jsonl"
        if p.exists():
            return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Fall back to HF.
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("cais/mmlu", subject, split="test")
    except Exception:
        return []
    out = []
    for ex in ds:
        choices = ex.get("choices") or [ex.get(k) for k in ("A", "B", "C", "D")]
        ans = ex.get("answer")
        if isinstance(ans, int):
            ans = "ABCD"[ans]
        out.append({
            "question": ex.get("question"),
            "choices":  choices,
            "answer":   ans,
        })
    return out


def build_prompt(ex: Dict[str, Any]) -> str:
    letters = "ABCD"
    body = ex["question"].strip() + "\n"
    for letter, choice in zip(letters, ex["choices"]):
        body += f"{letter}. {choice}\n"
    body += "Answer:"
    return PREAMBLE + body


# ---------------------------------------------------------------- letter-logit pick

def _letter_token_ids(tok) -> Dict[str, int]:
    """Token id of " A", " B", " C", " D" — the leading space matters for BPE."""
    ids = {}
    for letter in "ABCD":
        # Try with leading space first, fall back without.
        for cand in (f" {letter}", letter):
            try:
                t = tok.encode(cand, add_special_tokens=False)
            except Exception:
                t = []
            if len(t) == 1:
                ids[letter] = t[0]
                break
        else:
            # Last resort: take first token of the encoding.
            t = tok.encode(f" {letter}", add_special_tokens=False) or [0]
            ids[letter] = t[0]
    return ids


def _argmax_letter(model, tok, prompt: str, letter_ids: Dict[str, int]) -> str:
    import torch
    from synapforge.eval.generate import _forward_logits
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    max_seq = getattr(model, "max_seq", None)
    if max_seq is not None and inp.shape[1] > max_seq:
        inp = inp[:, -max_seq:]
    with torch.no_grad():
        logits = _forward_logits(model, inp)
    last = logits[0, -1].float()
    cands = {L: float(last[i].item()) for L, i in letter_ids.items()}
    return max(cands, key=cands.get)


# ---------------------------------------------------------------- main entry

def run_bench(ckpt: Optional[str] = None, tokenizer: Optional[str] = None,
              model: Any = None, tok: Any = None,
              data_dir: Optional[str] = None,
              subjects: Optional[List[str]] = None,
              n_per_subject: Optional[int] = None,
              out: Optional[str] = None, **_kw) -> Dict[str, Any]:
    if model is None and ckpt:
        from synapforge.eval.generate import load_synapforge
        model, _ = load_synapforge(ckpt)
    if tok is None and tokenizer:
        from synapforge.eval.generate import load_tokenizer
        tok = load_tokenizer(tokenizer)
    if model is None or tok is None:
        return {"name": "mmlu", "error": "model/tokenizer required"}

    letter_ids = _letter_token_ids(tok)
    subjects = subjects or MMLU_SUBJECTS

    by_subject: Dict[str, Dict[str, Any]] = {}
    total_correct = 0
    total_n = 0
    t0 = time.time()
    for s_i, subject in enumerate(subjects):
        examples = _load_subject(subject, data_dir)
        if n_per_subject:
            examples = examples[: int(n_per_subject)]
        if not examples:
            by_subject[subject] = {"n": 0, "correct": 0, "acc": 0.0}
            continue
        sub_correct = 0
        for ex in examples:
            prompt = build_prompt(ex)
            try:
                pick = _argmax_letter(model, tok, prompt, letter_ids)
            except Exception:
                pick = "A"
            if pick == (ex.get("answer") or "").upper().strip():
                sub_correct += 1
        n = len(examples)
        by_subject[subject] = {
            "n": n, "correct": sub_correct, "acc": sub_correct / max(n, 1),
        }
        total_correct += sub_correct
        total_n += n
        print(f"[mmlu] {s_i+1}/{len(subjects)} {subject}: {sub_correct}/{n}", flush=True)

    macro = (
        sum(s["acc"] for s in by_subject.values()) / max(len(by_subject), 1)
    )
    summary = {
        "name":       "mmlu",
        "acc":        total_correct / max(total_n, 1),
        "macro_acc":  macro,
        "n_total":    total_n,
        "n_correct":  total_correct,
        "by_subject": by_subject,
        "wall_s":     time.time() - t0,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data-dir", default=None,
                    help="Directory of {subject}.jsonl files")
    ap.add_argument("--subjects", default=None,
                    help="Comma-separated subset of subjects")
    ap.add_argument("--n", type=int, default=None,
                    help="Cap examples per subject (for smoke runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    subjects = args.subjects.split(",") if args.subjects else None
    summary = run_bench(
        ckpt=args.ckpt, tokenizer=args.tokenizer,
        data_dir=args.data_dir, subjects=subjects,
        n_per_subject=args.n, out=args.out,
    )
    # Skip the per-subject blob in stdout dump.
    light = {k: v for k, v in summary.items() if k != "by_subject"}
    print(json.dumps(light, indent=2))


if __name__ == "__main__":
    main()
