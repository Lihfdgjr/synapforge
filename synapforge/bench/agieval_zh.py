"""AGIEval Chinese subset accuracy.

AGIEval (Microsoft, Zhong et al. 2023) is a human-centric exam benchmark
covering Gaokao, SAT, LSAT, and Civil Service exams. The Chinese subset is
~1700 multi-choice items spanning ~10 sub-tasks (gaokao-{chinese,english,
biology,chemistry,history,...} + logic + civil service). We evaluate with
the same letter-logit recipe as MMLU/CMMLU/C-Eval.

Source: `microsoft/AGIEval` raw JSONL files mirrored on HF, or local fallback.

CLI:
    python -m synapforge.bench.agieval_zh --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \
        --out agieval_zh.json [--n 50]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Chinese-language sub-tasks in AGIEval. Names match the JSONL filenames in
# the official microsoft/AGIEval repo (data/v1).
AGIEVAL_ZH_TASKS: List[str] = [
    "gaokao-chinese", "gaokao-english", "gaokao-geography", "gaokao-history",
    "gaokao-biology", "gaokao-chemistry", "gaokao-physics",
    "gaokao-mathqa", "gaokao-mathcloze",
    "logiqa-zh", "jec-qa-kd", "jec-qa-ca", "civil-service-exam",
]

PREAMBLE_ZH = (
    "请阅读以下题目，从给出的选项中选出正确答案，仅给出选项字母。\n\n"
)


# ---------------------------------------------------------------- dataset


def _load_task(task: str, local_dir: Optional[str]) -> List[Dict[str, Any]]:
    if local_dir:
        p = Path(local_dir) / f"{task}.jsonl"
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
        ds = load_dataset("microsoft/AGIEval", task, split="test")
    except Exception:
        try:
            ds = load_dataset("hails/agieval", task, split="test")
        except Exception:
            return []
    out: List[Dict[str, Any]] = []
    for ex in ds:
        # AGIEval schema: passage (optional) + question + options + label/answer
        question = ex.get("question") or ex.get("Question") or ""
        passage = ex.get("passage") or ""
        options = ex.get("options") or ex.get("choices") or []
        ans = (ex.get("label") or ex.get("answer") or "").strip().upper()
        if isinstance(ans, int):
            ans = "ABCDEF"[ans]
        if passage:
            question = passage.strip() + "\n" + question.strip()
        out.append({
            "question": question,
            "choices":  list(options),
            "answer":   ans[:1] if ans else "",
        })
    return out


def build_prompt(ex: Dict[str, Any]) -> str:
    n = len(ex["choices"])
    letters = "ABCDEF"[:max(n, 4)]
    body = (ex["question"] or "").strip() + "\n"
    for letter, choice in zip(letters, ex["choices"]):
        body += f"{letter}. {choice}\n"
    body += "答案："
    return PREAMBLE_ZH + body


# ---------------------------------------------------------------- letter-logit pick


def _letter_token_ids(tok, n: int = 4) -> Dict[str, int]:
    """Token id of " A".." F" for variable-arity questions."""
    ids: Dict[str, int] = {}
    for letter in "ABCDEF"[:n]:
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
              tasks: Optional[List[str]] = None,
              n_per_task: Optional[int] = None,
              out: Optional[str] = None, **_kw) -> Dict[str, Any]:
    if model is None and ckpt:
        from synapforge.eval.generate import load_synapforge
        model, _ = load_synapforge(ckpt)
    if tok is None and tokenizer:
        from synapforge.eval.generate import load_tokenizer
        tok = load_tokenizer(tokenizer)
    if model is None or tok is None:
        return {"name": "agieval_zh", "error": "model/tokenizer required"}

    tasks = tasks or AGIEVAL_ZH_TASKS
    by_task: Dict[str, Dict[str, Any]] = {}
    total_correct = 0
    total_n = 0
    t0 = time.time()
    for t_i, task in enumerate(tasks):
        items = _load_task(task, data_dir)
        if n_per_task:
            items = items[: int(n_per_task)]
        if not items:
            by_task[task] = {"n": 0, "correct": 0, "acc": 0.0}
            continue
        # Variable arity per task; build letter ids once per task for speed.
        max_arity = max((len(x["choices"]) for x in items), default=4)
        max_arity = max(4, min(6, max_arity))
        letter_ids = _letter_token_ids(tok, n=max_arity)
        sub_correct = 0
        for ex in items:
            prompt = build_prompt(ex)
            try:
                pick = _argmax_letter(model, tok, prompt, letter_ids)
            except Exception:
                pick = "A"
            if pick == (ex.get("answer") or "").upper().strip():
                sub_correct += 1
        n = len(items)
        by_task[task] = {
            "n": n, "correct": sub_correct, "acc": sub_correct / max(n, 1),
        }
        total_correct += sub_correct
        total_n += n
        print(f"[agieval_zh] {t_i+1}/{len(tasks)} {task}: {sub_correct}/{n}", flush=True)

    macro = sum(s["acc"] for s in by_task.values()) / max(len(by_task), 1)
    summary = {
        "name":      "agieval_zh",
        "acc":       total_correct / max(total_n, 1),
        "macro_acc": macro,
        "n_total":   total_n,
        "n_correct": total_correct,
        "by_task":   by_task,
        "wall_s":    time.time() - t0,
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
                    help="Directory of {task}.jsonl files (AGIEval mirror)")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated subset of zh-tasks")
    ap.add_argument("--n", type=int, default=None,
                    help="Cap items per task (smoke runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = args.tasks.split(",") if args.tasks else None
    summary = run_bench(
        ckpt=args.ckpt, tokenizer=args.tokenizer,
        data_dir=args.data_dir, tasks=tasks,
        n_per_task=args.n, out=args.out,
    )
    light = {k: v for k, v in summary.items() if k != "by_task"}
    print(json.dumps(light, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
