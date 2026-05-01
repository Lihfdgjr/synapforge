"""CMMLU (Chinese Massive Multitask Language Understanding) accuracy.

67 Chinese subjects spanning humanities, social sciences, STEM, and Chinese-
specific knowledge (e.g. ancient_chinese, chinese_food_culture). Each item is a
4-choice exam question. Standard recipe: read the next-token logits at the
"答案：" boundary and pick the highest letter A/B/C/D.

Source: `haonan-li/cmmlu` HF dataset (or local JSONL fallback).

CLI:
    python -m synapforge.bench.cmmlu --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \
        --out cmmlu.json [--n 20] [--subjects=anatomy,...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# All 67 CMMLU subjects in canonical order (haonan-li/cmmlu).
CMMLU_SUBJECTS: List[str] = [
    "agronomy", "anatomy", "ancient_chinese", "arts", "astronomy",
    "business_ethics", "chinese_civil_service_exam", "chinese_driving_rule",
    "chinese_food_culture", "chinese_foreign_policy", "chinese_history",
    "chinese_literature", "chinese_teacher_qualification", "clinical_knowledge",
    "college_actuarial_science", "college_education", "college_engineering_hydrology",
    "college_law", "college_mathematics", "college_medical_statistics",
    "college_medicine", "computer_science", "computer_security",
    "conceptual_physics", "construction_project_management", "economics",
    "education", "electrical_engineering", "elementary_chinese",
    "elementary_commonsense", "elementary_information_and_technology",
    "elementary_mathematics", "ethnology", "food_science", "genetics",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_geography", "high_school_mathematics", "high_school_physics",
    "high_school_politics", "human_sexuality", "international_law",
    "journalism", "jurisprudence", "legal_and_moral_basis", "logical",
    "machine_learning", "management", "marketing", "marxist_theory",
    "modern_chinese", "nutrition", "philosophy", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_study", "sociology", "sports_science",
    "traditional_chinese_medicine", "virology", "world_history", "world_religions",
]

PREAMBLE = (
    "以下是关于{subject}的单项选择题，请直接给出正确答案的选项字母（A、B、C 或 D）。\n\n"
)


# ---------------------------------------------------------------- dataset


def _load_subject(subject: str, local_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Return list of {question, choices: [A,B,C,D], answer: 'A'|'B'|...}."""
    if local_dir:
        p = Path(local_dir) / f"{subject}.jsonl"
        if p.exists():
            return [
                json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("haonan-li/cmmlu", subject, split="test")
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for ex in ds:
        # CMMLU schema: Question, A, B, C, D, Answer
        choices = [ex.get(k) for k in ("A", "B", "C", "D")]
        ans = ex.get("Answer") or ex.get("answer")
        if isinstance(ans, int):
            ans = "ABCD"[ans]
        out.append({
            "question": ex.get("Question") or ex.get("question"),
            "choices":  choices,
            "answer":   (ans or "").strip().upper(),
        })
    return out


def build_prompt(ex: Dict[str, Any], subject: str) -> str:
    letters = "ABCD"
    body = (ex["question"] or "").strip() + "\n"
    for letter, choice in zip(letters, ex["choices"]):
        body += f"{letter}. {choice}\n"
    body += "答案："
    return PREAMBLE.format(subject=subject) + body


# ---------------------------------------------------------------- letter-logit pick


def _letter_token_ids(tok) -> Dict[str, int]:
    """Token id of " A", " B", " C", " D" — leading space matters for BPE."""
    ids: Dict[str, int] = {}
    for letter in "ABCD":
        for cand in (f" {letter}", letter):
            try:
                t = tok.encode(cand, add_special_tokens=False)
            except Exception:
                t = []
            if len(t) == 1:
                ids[letter] = t[0]
                break
        else:
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
        return {"name": "cmmlu", "error": "model/tokenizer required"}

    letter_ids = _letter_token_ids(tok)
    subjects = subjects or CMMLU_SUBJECTS

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
            prompt = build_prompt(ex, subject)
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
        print(f"[cmmlu] {s_i+1}/{len(subjects)} {subject}: {sub_correct}/{n}", flush=True)

    macro = sum(s["acc"] for s in by_subject.values()) / max(len(by_subject), 1)
    summary = {
        "name":       "cmmlu",
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
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data-dir", default=None,
                    help="Directory of {subject}.jsonl files (CMMLU local mirror)")
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
    light = {k: v for k, v in summary.items() if k != "by_subject"}
    print(json.dumps(light, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
