"""HumanEval pass@1 evaluator.

Dataset: 164 hand-written Python problems with unit tests (Chen et al. 2021).
Loads from local JSONL (preferred) or HuggingFace `openai_humaneval` fallback.

Each problem has:
    task_id  — "HumanEval/0", ...
    prompt   — function signature + docstring
    test     — assertion-style unit test
    entry_point — the function name to call

Score: pass@1 = fraction of problems where the model's first sample passes the
unit test in a 5-second sandboxed subprocess.

CLI:
    python -m synapforge.bench.humaneval --ckpt PATH --tokenizer Qwen/Qwen2.5-0.5B \\
        --max-new 384 --out humaneval.json [--n 20]
"""

from __future__ import annotations

import argparse
import json
import os
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
                "HumanEval needs either --data PATH (local JSONL) or `datasets`."
            ) from e
        ds = load_dataset("openai_humaneval", split="test")
        probs = list(ds)
    if n is not None:
        probs = probs[: int(n)]
    return probs


# ---------------------------------------------------------------- code extraction

_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


def extract_code(completion: str, prompt: str, entry_point: str) -> str:
    """Pull a runnable Python function out of a model completion.

    Strategy:
        1. Strip ``` fences if present.
        2. If the completion already contains `def {entry_point}`, take from there.
        3. Otherwise prepend the original prompt (signature + docstring).
        4. Truncate at the first top-level construct after the function body.
    """
    body = completion
    m = _FENCE_RE.search(body)
    if m:
        body = m.group(1)

    if f"def {entry_point}" in body:
        idx = body.find(f"def {entry_point}")
        body = body[idx:]
    else:
        body = prompt + body

    # Truncate at next blank-line top-level def / class / if __name__.
    out_lines: List[str] = []
    in_func = False
    for line in body.splitlines():
        if not in_func:
            if line.startswith(f"def {entry_point}"):
                in_func = True
                out_lines.append(line)
                continue
            out_lines.append(line)
        else:
            if line.startswith(("def ", "class ", "if __name__")) and out_lines:
                break
            out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------- sandbox runner

_HARNESS = """\
import sys, signal
def _handler(signum, frame): raise TimeoutError()
try:
    signal.signal(signal.SIGALRM, _handler); signal.alarm(5)
except Exception:
    pass
__SOLUTION__
__TEST__
check({entry_point})
print("__OK__")
"""


def run_one_test(solution: str, test: str, entry_point: str, timeout_s: int = 5) -> bool:
    """Run `solution + test + check(entry_point)` in a fresh subprocess.

    Returns True if the subprocess prints "__OK__" within the timeout, False
    otherwise (timeout, exception, assertion failure, anything).
    """
    code = (
        _HARNESS
        .replace("__SOLUTION__", solution)
        .replace("__TEST__", test)
        .format(entry_point=entry_point)
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    return r.returncode == 0 and "__OK__" in (r.stdout or "")


# ---------------------------------------------------------------- generation

def _gen_completion(model, tok, prompt: str, max_new: int = 384) -> str:
    """Greedy or low-temp sampling. Falls back to deterministic stub if no model."""
    if model is None or tok is None:
        # Useful for unit-testing the harness itself with a known-passing stub.
        return ""
    from synapforge.eval.generate import generate
    out = generate(
        model, tok, prompt, max_new=max_new,
        top_k=1, top_p=1.0, temperature=0.0,
    )
    # generate() returns prompt+completion; strip the prompt prefix if present.
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
        comp = _gen_completion(model, tok, p["prompt"], max_new=max_new)
        sol = extract_code(comp, p["prompt"], p["entry_point"])
        ok = run_one_test(sol, p["test"], p["entry_point"], timeout_s=timeout_s)
        n_passed += int(ok)
        samples.append({"task_id": p["task_id"], "pass": ok})
        if (i + 1) % 10 == 0:
            print(f"[humaneval] {i+1}/{n_total}  pass={n_passed}/{i+1}", flush=True)

    summary = {
        "name":     "humaneval",
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
    ap.add_argument("--data", default=None, help="local JSONL of HumanEval problems")
    ap.add_argument("--n", type=int, default=None, help="cap problems for smoke")
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
