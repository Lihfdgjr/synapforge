"""scripts/synth_eval_data.py — generate synthetic eval data when downloads fail.

Pure programmatic generation, deterministic seed=42. No LLM call. Purpose:
provide a *smoke* baseline so eval harnesses never blow up on missing files
during CPU-only investor demos and CI dry runs.

Generates four packs:
  * trivia (100): multi-choice 4-option items mixing general knowledge and
    technical content (encoder/decoder, MoE, quantization, etc.).
  * math (50): GSM8K-style word problems with verifiable integer answer.
  * code (30): function completions given docstring + signature.
  * niah (20): "Needle in a Haystack" — insert a unique sentinel inside a
    ~4K token Lorem-ipsum-like haystack at varied depths.

Output: $OUT/{trivia,math,code,niah}.json (default OUT=data/eval/synth_smoke).

CLI:
    python scripts/synth_eval_data.py             # full
    python scripts/synth_eval_data.py --smoke     # 10/5/3/2 instead
    python scripts/synth_eval_data.py --out PATH
    python scripts/synth_eval_data.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Tuple

SEED = 42

# ---------------------------------------------------------------- trivia bank
TRIVIA_BANK: List[Tuple[str, List[str], int]] = [
    ("Capital of France?",
     ["Paris", "London", "Berlin", "Madrid"], 0),
    ("What does GPU stand for?",
     ["Graphics Processing Unit", "General Purpose Unit",
      "Gradient Processing Unit", "Graph Pattern Unit"], 0),
    ("Which is faster RAM access?",
     ["L1 cache", "L2 cache", "L3 cache", "DRAM"], 0),
    ("Default Python sort algorithm?",
     ["Timsort", "Quicksort", "Mergesort", "Heapsort"], 0),
    ("ReLU activation outputs?",
     ["max(0, x)", "min(0, x)", "1/(1+e^-x)", "tanh(x)"], 0),
    ("BPE stands for?",
     ["Byte Pair Encoding", "Backward Pass Engine",
      "Batch Processing Engine", "Bit Per Element"], 0),
    ("Adam optimizer combines?",
     ["momentum + adaptive lr", "SGD + warmup",
      "L1 + L2 reg", "rmsprop only"], 0),
    ("Liquid neural network introduced by?",
     ["Hasani et al.", "Hochreiter et al.",
      "Vaswani et al.", "Schmidhuber et al."], 0),
    ("Which is a spiking neuron model?",
     ["LIF", "ReLU", "GELU", "Swish"], 0),
    ("Hebbian rule abbreviated?",
     ["fire together, wire together", "loss must go down",
      "gradient flows backward", "weights initialize to zero"], 0),
    ("Free energy principle author?",
     ["Karl Friston", "Geoffrey Hinton",
      "Yann LeCun", "Yoshua Bengio"], 0),
    ("What is STDP?",
     ["spike-timing dependent plasticity", "softmax temporal duration policy",
      "stochastic time differential propagation", "self-training double pass"], 0),
    ("Transformer attention complexity?",
     ["O(n^2 d)", "O(n d^2)", "O(n d)", "O(n log n)"], 0),
    ("MoE means?",
     ["mixture of experts", "memory of events",
      "model on edge", "multi-objective evaluation"], 0),
    ("Common LLM context window 2024?",
     ["128K", "1K", "10M", "1B"], 0),
    ("Which is a vector DB?",
     ["FAISS", "MySQL", "Redis", "Postgres"], 0),
    ("Who proposed RoPE?",
     ["Su et al.", "Devlin et al.",
      "Touvron et al.", "Brown et al."], 0),
    ("PPO is a?",
     ["RL algorithm", "tokenizer", "loss function only", "weight init"], 0),
    ("Loss for classification?",
     ["cross-entropy", "MSE", "L1", "Huber"], 0),
    ("CLIP is a?",
     ["vision-language model", "tokenizer",
      "vector DB", "scheduler"], 0),
    ("What's a butterfly factor in math?",
     ["FFT building block", "RNN gate",
      "tokenization unit", "regularizer"], 0),
    ("GeLU is similar to?",
     ["smooth ReLU", "softmax",
      "sigmoid only", "tanh only"], 0),
    ("Layer norm normalizes over?",
     ["features", "batch only",
      "channels only", "time only"], 0),
    ("BatchNorm differs from LayerNorm by?",
     ["normalizes over batch", "normalizes over features",
      "uses bias only", "uses sigmoid"], 0),
    ("Residual connection helps?",
     ["gradient flow", "memory savings",
      "data augmentation", "tokenization"], 0),
]


def gen_trivia(n: int, rng: random.Random) -> List[Dict]:
    """Sample up to n trivia. If n > bank, repeat with shuffled distractors."""
    items: List[Dict] = []
    for i in range(n):
        q, choices, ans = TRIVIA_BANK[i % len(TRIVIA_BANK)]
        # Shuffle choices on repeats to avoid degenerate "always A" overfit.
        order = list(range(4))
        if i >= len(TRIVIA_BANK):
            rng.shuffle(order)
        shuf = [choices[j] for j in order]
        new_ans = order.index(ans)
        items.append({
            "id": f"trivia_{i:04d}",
            "question": q,
            "choices": shuf,
            "answer_idx": new_ans,
            "answer_letter": "ABCD"[new_ans],
        })
    return items


# ---------------------------------------------------------------- math bank


def gen_math(n: int, rng: random.Random) -> List[Dict]:
    """GSM8K-style: 1- or 2-step word problem, integer answer."""
    items: List[Dict] = []
    templates = [
        ("{a} apples plus {b} apples is how many?", lambda a, b: a + b),
        ("{a} books each cost {b} yuan, total?", lambda a, b: a * b),
        ("Subtract {b} from {a}, what's left?", lambda a, b: a - b),
        ("Divide {a} candies among {b} kids equally, each gets?",
         lambda a, b: a // max(1, b)),
        ("{a} groups of {b} students each, total students?",
         lambda a, b: a * b),
        ("If a runner does {a}km/h for {b} hours, distance?",
         lambda a, b: a * b),
        ("Train moves {a}km in {b} hours, speed?",
         lambda a, b: a // max(1, b)),
        ("Rectangle {a}x{b}, area?", lambda a, b: a * b),
        ("Half of {a}, then plus {b}, equals?",
         lambda a, b: a // 2 + b),
        ("Double {a}, then minus {b}, equals?",
         lambda a, b: 2 * a - b),
    ]
    for i in range(n):
        tpl, fn = templates[i % len(templates)]
        a = rng.randint(2, 99)
        b = rng.randint(2, 20)
        q = tpl.format(a=a, b=b)
        ans = fn(a, b)
        items.append({
            "id": f"math_{i:04d}",
            "question": q,
            "answer": int(ans),
            "answer_str": str(int(ans)),
        })
    return items


# ---------------------------------------------------------------- code bank


def gen_code(n: int, rng: random.Random) -> List[Dict]:
    """Function-completion items. Each item has signature + docstring + tests."""
    bank = [
        ("def add(a, b):",
         "Return a + b.",
         [(2, 3, 5), (10, -1, 9), (-5, -5, -10)]),
        ("def reverse(s):",
         "Return string s reversed.",
         [("abc", "cba"), ("Python", "nohtyP")]),
        ("def is_prime(n):",
         "Return True if n is prime, else False.",
         [(2, True), (15, False), (7, True)]),
        ("def fib(n):",
         "Return the nth Fibonacci number, fib(0)=0.",
         [(0, 0), (5, 5), (10, 55)]),
        ("def gcd(a, b):",
         "Return greatest common divisor.",
         [(12, 8, 4), (15, 25, 5)]),
        ("def is_palindrome(s):",
         "Return True if s reads the same forward and backward.",
         [("aba", True), ("abc", False)]),
        ("def count_vowels(s):",
         "Return number of vowels (aeiou, case-insensitive).",
         [("hello", 2), ("xyz", 0)]),
        ("def fizzbuzz(n):",
         "Return list of strings 1..n with fizz/buzz/fizzbuzz substitutions.",
         [(5, ["1", "2", "Fizz", "4", "Buzz"])]),
        ("def factorial(n):",
         "Return n! ; factorial(0) = 1.",
         [(0, 1), (5, 120)]),
        ("def square(n):",
         "Return n*n.",
         [(3, 9), (-2, 4)]),
    ]
    items: List[Dict] = []
    for i in range(n):
        sig, doc, tests = bank[i % len(bank)]
        items.append({
            "id": f"code_{i:04d}",
            "signature": sig,
            "docstring": doc,
            "tests": [list(t) for t in tests],
            "prompt": f'{sig}\n    """{doc}"""\n',
        })
    return items


# ---------------------------------------------------------------- NIAH


HAYSTACK_BLOCK = (
    "The brain processes information through neurons connected by synapses. "
    "Hebb proposed that neurons that fire together wire together, forming "
    "the basis of associative memory. Modern artificial neural networks borrow "
    "this idea via gradient descent on differentiable losses, although biological "
    "plasticity is local and event-driven. Spiking neural networks attempt to "
    "bridge this gap. "
)


def gen_niah(n: int, rng: random.Random) -> List[Dict]:
    """Needle in a Haystack — insert sentinel at varying depths."""
    items: List[Dict] = []
    target_chars = 4096  # ~1K tokens with English text
    for i in range(n):
        depth = (i % 10) / 10.0  # 0.0..0.9
        # Each sentinel encodes a unique 4-digit secret.
        secret = f"{rng.randint(1000, 9999)}-{rng.choice(['alpha','bravo','charlie','delta'])}"
        sentinel = f"NEEDLE-SECRET-CODE-{secret}"
        question = (
            "What is the NEEDLE-SECRET-CODE in the passage above? "
            "Answer with the exact code."
        )
        # Build haystack of ~target_chars chars.
        body_chars = max(target_chars - len(sentinel) - 8, 64)
        repeats = max(1, body_chars // len(HAYSTACK_BLOCK) + 1)
        body = HAYSTACK_BLOCK * repeats
        body = body[:body_chars]
        cut = int(len(body) * depth)
        haystack = body[:cut] + " " + sentinel + " " + body[cut:]
        items.append({
            "id": f"niah_{i:04d}",
            "depth": depth,
            "haystack_chars": len(haystack),
            "haystack": haystack,
            "question": question,
            "answer": sentinel,
        })
    return items


# ---------------------------------------------------------------- entrypoint


def write(outdir: str, name: str, items: List[Dict]) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "n": len(items), "items": items},
                  f, indent=2, ensure_ascii=False)
    # Append checksum to manifest.
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    print(f"wrote {len(items):>4} {name:<6} -> {path}  sha256={digest}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", default=os.path.join("data", "eval", "synth_smoke"))
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny pack: 10/5/3/2 items (instead of 100/50/30/20).")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.smoke:
        n_trivia, n_math, n_code, n_niah = 10, 5, 3, 2
    else:
        n_trivia, n_math, n_code, n_niah = 100, 50, 30, 20

    write(args.out, "trivia", gen_trivia(n_trivia, rng))
    write(args.out, "math",   gen_math(n_math, rng))
    write(args.out, "code",   gen_code(n_code, rng))
    write(args.out, "niah",   gen_niah(n_niah, rng))

    manifest = {
        "version": "1.0",
        "seed": args.seed,
        "smoke": args.smoke,
        "out_dir": args.out,
        "packs": {"trivia": n_trivia, "math": n_math,
                  "code": n_code, "niah": n_niah},
    }
    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
