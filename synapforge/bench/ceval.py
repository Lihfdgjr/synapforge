"""C-Eval (Chinese benchmark, 52 subjects) accuracy.

C-Eval covers four high-level categories: STEM, Humanities, Social Sciences,
and Other. Each item is a 4-choice exam question scored by next-token logit
pick at A/B/C/D — same recipe as MMLU/CMMLU. Source: `ceval/ceval-exam` HF
dataset (or local JSONL fallback). The official `val` split has gold labels
released; `test` split labels are held out by the leaderboard.

CLI:
    python -m synapforge.bench.ceval --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \
        --out ceval.json [--n 20] [--split val]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# All 52 C-Eval subjects (ceval/ceval-exam).
CEVAL_SUBJECTS: List[str] = [
    "computer_network", "operating_system", "computer_architecture",
    "college_programming", "college_physics", "college_chemistry",
    "advanced_mathematics", "probability_and_statistics", "discrete_mathematics",
    "electrical_engineer", "metrology_engineer", "high_school_mathematics",
    "high_school_physics", "high_school_chemistry", "high_school_biology",
    "middle_school_mathematics", "middle_school_biology", "middle_school_physics",
    "middle_school_chemistry", "veterinary_medicine", "college_economics",
    "business_administration", "marxism", "mao_zedong_thought", "education_science",
    "teacher_qualification", "high_school_politics", "high_school_geography",
    "middle_school_politics", "middle_school_geography", "modern_chinese_history",
    "ideological_and_moral_cultivation", "logic", "law", "chinese_language_and_literature",
    "art_studies", "professional_tour_guide", "legal_professional",
    "high_school_chinese", "high_school_history", "middle_school_history",
    "civil_servant", "sports_science", "plant_protection", "basic_medicine",
    "clinical_medicine", "urban_and_rural_planner", "accountant", "fire_engineer",
    "environmental_impact_assessment_engineer", "tax_accountant", "physician",
]

# Subject -> high-level category for partial reporting.
CATEGORY: Dict[str, str] = {
    s: "STEM" for s in [
        "computer_network", "operating_system", "computer_architecture",
        "college_programming", "college_physics", "college_chemistry",
        "advanced_mathematics", "probability_and_statistics", "discrete_mathematics",
        "electrical_engineer", "metrology_engineer", "high_school_mathematics",
        "high_school_physics", "high_school_chemistry", "high_school_biology",
        "middle_school_mathematics", "middle_school_biology", "middle_school_physics",
        "middle_school_chemistry", "veterinary_medicine",
    ]
}
for s in [
    "college_economics", "business_administration", "marxism", "mao_zedong_thought",
    "education_science", "teacher_qualification", "high_school_politics",
    "high_school_geography", "middle_school_politics", "middle_school_geography",
]:
    CATEGORY[s] = "Social"
for s in [
    "modern_chinese_history", "ideological_and_moral_cultivation", "logic",
    "law", "chinese_language_and_literature", "art_studies", "professional_tour_guide",
    "legal_professional", "high_school_chinese", "high_school_history",
    "middle_school_history",
]:
    CATEGORY[s] = "Humanities"
for s in [
    "civil_servant", "sports_science", "plant_protection", "basic_medicine",
    "clinical_medicine", "urban_and_rural_planner", "accountant", "fire_engineer",
    "environmental_impact_assessment_engineer", "tax_accountant", "physician",
]:
    CATEGORY[s] = "Other"


PREAMBLE = (
    "以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案（A、B、C 或 D）。\n\n"
)


# ---------------------------------------------------------------- dataset


def _load_subject(subject: str, local_dir: Optional[str], split: str) -> List[Dict[str, Any]]:
    if local_dir:
        p = Path(local_dir) / split / f"{subject}.jsonl"
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
        ds = load_dataset("ceval/ceval-exam", subject, split=split)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for ex in ds:
        choices = [ex.get(k) for k in ("A", "B", "C", "D")]
        ans = ex.get("answer") or ex.get("Answer") or ""
        if isinstance(ans, int):
            ans = "ABCD"[ans]
        out.append({
            "question": ex.get("question") or ex.get("Question"),
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
              split: str = "val",
              out: Optional[str] = None, **_kw) -> Dict[str, Any]:
    if model is None and ckpt:
        from synapforge.eval.generate import load_synapforge
        model, _ = load_synapforge(ckpt)
    if tok is None and tokenizer:
        from synapforge.eval.generate import load_tokenizer
        tok = load_tokenizer(tokenizer)
    if model is None or tok is None:
        return {"name": "ceval", "error": "model/tokenizer required"}

    letter_ids = _letter_token_ids(tok)
    subjects = subjects or CEVAL_SUBJECTS

    by_subject: Dict[str, Dict[str, Any]] = {}
    by_category: Dict[str, Dict[str, int]] = {}
    total_correct = 0
    total_n = 0
    t0 = time.time()
    for s_i, subject in enumerate(subjects):
        examples = _load_subject(subject, data_dir, split)
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
        cat = CATEGORY.get(subject, "Other")
        by_subject[subject] = {
            "n": n, "correct": sub_correct, "acc": sub_correct / max(n, 1),
            "category": cat,
        }
        cat_d = by_category.setdefault(cat, {"n": 0, "correct": 0})
        cat_d["n"] += n
        cat_d["correct"] += sub_correct
        total_correct += sub_correct
        total_n += n
        print(f"[ceval] {s_i+1}/{len(subjects)} {subject} ({cat}): "
              f"{sub_correct}/{n}", flush=True)

    macro = sum(s["acc"] for s in by_subject.values()) / max(len(by_subject), 1)
    cat_acc = {c: d["correct"] / max(d["n"], 1) for c, d in by_category.items()}
    summary = {
        "name":         "ceval",
        "split":        split,
        "acc":          total_correct / max(total_n, 1),
        "macro_acc":    macro,
        "by_category":  cat_acc,
        "n_total":      total_n,
        "n_correct":    total_correct,
        "by_subject":   by_subject,
        "wall_s":       time.time() - t0,
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
                    help="Directory of {split}/{subject}.jsonl files")
    ap.add_argument("--subjects", default=None,
                    help="Comma-separated subset of subjects")
    ap.add_argument("--split", default="val", choices=["val", "dev", "test"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    subjects = args.subjects.split(",") if args.subjects else None
    summary = run_bench(
        ckpt=args.ckpt, tokenizer=args.tokenizer,
        data_dir=args.data_dir, subjects=subjects,
        n_per_subject=args.n, split=args.split, out=args.out,
    )
    light = {k: v for k, v in summary.items() if k != "by_subject"}
    print(json.dumps(light, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
