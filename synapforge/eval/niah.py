"""
Needle-in-a-Haystack (NIAH) eval harness for verbatim long-context recall.

Two task variants:
  - UUID NIAH: insert a UUID at random depth, ask "what was the UUID?"
  - PASSKEY: insert a 5-digit passkey, ask "what is the passkey?"

Used to verify the claim: "STDP-inference makes longer context yield BETTER
recall, monotonically." Run at lengths {1K, 10K, 100K, 1M} × {A baseline,
B +STDP-inference}. Pass = B dominates A at every length.

Generates synthetic long contexts from a filler corpus + a needle inserted
at depth d ∈ [0.05, 0.95]. Measures exact-match accuracy on the needle.

Quick smoke test (no model):
    python -m synapforge.eval.niah --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


@dataclass
class NIAHExample:
    context_len_target: int
    actual_len: int
    needle: str
    needle_position: int
    needle_depth_frac: float
    prompt: str
    expected_answer: str
    task_kind: str = "uuid"


@dataclass
class NIAHResult:
    example: NIAHExample
    model_answer: str = ""
    pass_: bool = False
    latency_s: float = 0.0


def random_uuid(rng: random.Random) -> str:
    """Random UUID-like string. 32 hex chars with dashes."""
    h = "".join(rng.choices("0123456789abcdef", k=32))
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def random_passkey(rng: random.Random) -> str:
    return "".join(rng.choices(string.digits, k=5))


def build_filler_text(rng: random.Random, n_words: int) -> str:
    """Cheap filler: random sampling from an English wordlist.

    For real use, replace with PG-19 / Wikipedia text. For smoke test we use
    a small bag of common words.
    """
    BAG = (
        "the quick brown fox jumps over the lazy dog and went down to the river "
        "where it found a small fish swimming gracefully through the cool water "
        "while a bird sang from the branches above and the wind whispered softly "
        "carrying the scent of pine and freshly cut grass through the meadow "
        "where children played with their kites under the bright summer sun "
    ).split()
    chunks = []
    while sum(len(c.split()) for c in chunks) < n_words:
        chunks.append(" ".join(rng.choices(BAG, k=64)))
    text = " ".join(chunks)
    return " ".join(text.split()[:n_words])


def make_uuid_niah(
    context_len_words: int,
    rng: Optional[random.Random] = None,
    needle_depth_frac: Optional[float] = None,
) -> NIAHExample:
    """Build a UUID-style NIAH example."""
    rng = rng or random.Random()
    needle_value = random_uuid(rng)
    needle_text = f"The secret UUID is {needle_value}."
    needle_words = needle_text.split()

    if needle_depth_frac is None:
        needle_depth_frac = rng.uniform(0.05, 0.95)

    filler_words_total = max(0, context_len_words - len(needle_words))
    insert_at = int(needle_depth_frac * filler_words_total)

    filler = build_filler_text(rng, filler_words_total).split()
    prefix = " ".join(filler[:insert_at])
    suffix = " ".join(filler[insert_at:])
    context = f"{prefix} {needle_text} {suffix}"

    prompt = (
        f"<|im_start|>user\n"
        f"{context}\n\n"
        f"What is the secret UUID mentioned in the text above? "
        f"Reply with just the UUID, nothing else.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return NIAHExample(
        context_len_target=context_len_words,
        actual_len=len(context.split()),
        needle=needle_value,
        needle_position=insert_at,
        needle_depth_frac=needle_depth_frac,
        prompt=prompt,
        expected_answer=needle_value,
        task_kind="uuid",
    )


def make_passkey_niah(
    context_len_words: int,
    rng: Optional[random.Random] = None,
    needle_depth_frac: Optional[float] = None,
) -> NIAHExample:
    rng = rng or random.Random()
    pk = random_passkey(rng)
    needle_text = f"The passkey is {pk}. Remember it well."
    if needle_depth_frac is None:
        needle_depth_frac = rng.uniform(0.05, 0.95)
    filler_words_total = max(0, context_len_words - len(needle_text.split()))
    insert_at = int(needle_depth_frac * filler_words_total)

    filler = build_filler_text(rng, filler_words_total).split()
    prefix = " ".join(filler[:insert_at])
    suffix = " ".join(filler[insert_at:])
    context = f"{prefix} {needle_text} {suffix}"

    prompt = (
        f"<|im_start|>user\n"
        f"{context}\n\n"
        f"What was the passkey mentioned earlier? "
        f"Reply with just the 5-digit number.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return NIAHExample(
        context_len_target=context_len_words,
        actual_len=len(context.split()),
        needle=pk,
        needle_position=insert_at,
        needle_depth_frac=needle_depth_frac,
        prompt=prompt,
        expected_answer=pk,
        task_kind="passkey",
    )


def score_answer(model_output: str, expected: str, task_kind: str) -> bool:
    """Exact-match scorer."""
    output = model_output.strip().lower()
    expected = expected.strip().lower()
    if task_kind == "uuid":
        m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", output)
        if m:
            return m.group(0) == expected
        return expected in output
    if task_kind == "passkey":
        m = re.search(r"\b\d{5}\b", output)
        if m:
            return m.group(0) == expected
        return expected in output
    return expected in output


def run_niah(
    generate_fn: Callable[[str], str],
    context_lens: List[int],
    n_per_length: int = 30,
    task_kind: str = "uuid",
    seed: int = 42,
    out_path: Optional[str | Path] = None,
) -> dict:
    """Run NIAH at multiple context lengths, return summary dict.

    generate_fn: takes prompt str, returns model's completion str.
    """
    rng = random.Random(seed)
    results: List[NIAHResult] = []
    summary: dict = {"per_length": {}, "overall_pass_rate": 0.0}

    for ctx_len in context_lens:
        per_len_pass = 0
        per_len_total = 0
        per_len_lat = []
        for i in range(n_per_length):
            if task_kind == "uuid":
                ex = make_uuid_niah(ctx_len, rng=rng)
            elif task_kind == "passkey":
                ex = make_passkey_niah(ctx_len, rng=rng)
            else:
                raise ValueError(f"unknown task {task_kind}")

            t0 = time.time()
            try:
                model_answer = generate_fn(ex.prompt)
            except Exception as e:
                model_answer = f"<error: {e!r}>"
            latency = time.time() - t0

            ok = score_answer(model_answer, ex.expected_answer, task_kind)
            results.append(NIAHResult(
                example=ex, model_answer=model_answer, pass_=ok, latency_s=latency,
            ))
            per_len_pass += int(ok)
            per_len_total += 1
            per_len_lat.append(latency)

        summary["per_length"][ctx_len] = {
            "pass_rate": per_len_pass / max(per_len_total, 1),
            "n": per_len_total,
            "n_pass": per_len_pass,
            "avg_latency_s": sum(per_len_lat) / max(len(per_len_lat), 1),
        }

    total_pass = sum(1 for r in results if r.pass_)
    summary["overall_pass_rate"] = total_pass / max(len(results), 1)
    summary["total_n"] = len(results)
    summary["task_kind"] = task_kind
    summary["context_lens"] = context_lens

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "results": [
                    {
                        "actual_len": r.example.actual_len,
                        "needle_depth_frac": r.example.needle_depth_frac,
                        "expected": r.example.expected_answer,
                        "got": r.model_answer[:100],
                        "pass": r.pass_,
                        "latency_s": r.latency_s,
                    }
                    for r in results
                ],
            }, f, ensure_ascii=False, indent=2)

    return summary


def smoke() -> None:
    """No-model smoke test: build examples + score against expected."""
    rng = random.Random(42)
    for ctx in [100, 1000, 10000]:
        ex = make_uuid_niah(ctx, rng=rng)
        assert score_answer(f"The UUID is {ex.expected_answer}", ex.expected_answer, "uuid"), \
            f"smoke failed at ctx={ctx}"
        print(f"OK ctx={ctx} actual={ex.actual_len} depth={ex.needle_depth_frac:.2f} needle={ex.needle[:8]}...")
    print("NIAH smoke OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="No-model smoke test")
    p.add_argument("--lens", type=str, default="1000,10000,100000",
                   help="Comma-separated context lengths in words")
    p.add_argument("--n-per-len", type=int, default=30)
    p.add_argument("--task", type=str, default="uuid", choices=["uuid", "passkey"])
    p.add_argument("--out", type=str, default="/workspace/runs/niah_results.json")
    args = p.parse_args()

    if args.smoke:
        smoke()
        return

    print("Use as a library: pass generate_fn=lambda prompt: model.generate(prompt)")
    print(f"Args: lens={args.lens} n={args.n_per_len} task={args.task}")


if __name__ == "__main__":
    main()
