"""MBPP (Mostly Basic Python Problems) pass@1 evaluator.

Dataset: 974 short Python problems with three test cases each (Austin et al. 2021).
Same harness shape as humaneval.py — different prompt format.

Each MBPP example has:
    task_id   — int
    text      — natural-language description
    code      — reference solution (we don't use it)
    test_list — list of `assert ...` lines
    test_setup_code — optional setup (rare)

We construct a prompt of:
    "You are an expert Python programmer, and here is your task: {text}
     Your code should pass these tests: {first 1-2 tests}
     [BEGIN]"
and stop at "[DONE]".

CLI:
    python -m synapforge.bench.mbpp --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \\
        --max-new 384 --out mbpp.json [--n 50]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- dataset


def _load_problems(local_path: Optional[str], n: Optional[int]) -> List[Dict[str, Any]]:
    if local_path and Path(local_path).exists():
        probs = []
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    probs.append(json.loads(line))
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "MBPP needs either --data PATH (local JSONL) or `datasets`."
            ) from e
        ds = load_dataset("mbpp", split="test")
        probs = list(ds)
    if n is not None:
        probs = probs[: int(n)]
    return probs


def build_prompt(prob: Dict[str, Any]) -> str:
    text = prob.get("text") or prob.get("prompt") or ""
    tests = prob.get("test_list") or []
    tests_blurb = "\n".join(tests[:2])
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{text}\n"
        "Your code should pass these tests:\n\n"
        f"{tests_blurb}\n[BEGIN]\n"
    )


# ---------------------------------------------------------------- code extraction

_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    """Return the body between [BEGIN]…[DONE] or the first fenced block."""
    body = completion
    if "[DONE]" in body:
        body = body.split("[DONE]")[0]
    if "[BEGIN]" in body:
        body = body.split("[BEGIN]", 1)[1]
    m = _FENCE_RE.search(body)
    if m:
        body = m.group(1)
    return body.strip()


# ---------------------------------------------------------------- sandbox

_HARNESS = """\
import sys, signal
def _handler(signum, frame): raise TimeoutError()
try:
    signal.signal(signal.SIGALRM, _handler); signal.alarm(5)
except Exception:
    pass
__SETUP__
__SOLUTION__
__TESTS__
print("__OK__")
"""


def run_tests(solution: str, tests: List[str], setup: str = "", timeout_s: int = 5) -> bool:
    code = (
        _HARNESS
        .replace("__SETUP__",    setup)
        .replace("__SOLUTION__", solution)
        .replace("__TESTS__",    "\n".join(tests))
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            timeout=timeout_s, capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    return r.returncode == 0 and "__OK__" in (r.stdout or "")


# ---------------------------------------------------------------- generation

def _gen_completion(model, tok, prompt: str, max_new: int = 384) -> str:
    if model is None or tok is None:
        return ""
    from synapforge.eval.generate import generate
    out = generate(
        model, tok, prompt, max_new=max_new,
        top_k=1, top_p=1.0, temperature=0.0,
    )
    if out.startswith(prompt):
        out = out[len(prompt):]
    return out


# ---------------------------------------------------------------- main entry

def run_bench(ckpt: Optional[str] = None, tokenizer: Optional[str] = None,
              model: Any = None, tok: Any = None,
              data_path: Optional[str] = None, n: Optional[int] = None,
              max_new: int = 384, timeout_s: int = 5,
              out: Optional[str] = None, **_kw) -> Dict[str, Any]:
    if model is None and ckpt:
        from synapforge.eval.generate import load_synapforge
        model, _ = load_synapforge(ckpt)
    if tok is None and tokenizer:
        from synapforge.eval.generate import load_tokenizer
        tok = load_tokenizer(tokenizer)

    probs = _load_problems(data_path, n)
    n_total = len(probs)
    n_passed = 0
    samples: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, p in enumerate(probs):
        prompt = build_prompt(p)
        comp = _gen_completion(model, tok, prompt, max_new=max_new)
        sol  = extract_code(comp)
        ok   = run_tests(sol, p.get("test_list") or [],
                         setup=p.get("test_setup_code") or "", timeout_s=timeout_s)
        n_passed += int(ok)
        samples.append({"task_id": p.get("task_id"), "pass": ok})
        if (i + 1) % 25 == 0:
            print(f"[mbpp] {i+1}/{n_total}  pass={n_passed}/{i+1}", flush=True)

    summary = {
        "name":     "mbpp",
        "pass@1":   n_passed / max(n_total, 1),
        "n_passed": n_passed,
        "n_total":  n_total,
        "wall_s":   time.time() - t0,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": samples}, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--timeout-s", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    summary = run_bench(
        ckpt=args.ckpt, tokenizer=args.tokenizer,
        data_path=args.data, n=args.n,
        max_new=args.max_new, timeout_s=args.timeout_s, out=args.out,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
