"""synapforge.bench — paper-grade evaluation harness.

English benchmarks for parity-vs-baseline reporting:
    humaneval    — Python code generation, pass@1, 164 problems
    mbpp         — Mostly Basic Python Problems, pass@1, 974 problems
    mmlu         — 57-subject multi-choice exam, accuracy
    gsm8k        — grade-school math word problems, exact-match
    hellaswag    — commonsense continuation, 4-choice accuracy
    lambada      — last-word prediction accuracy

Chinese benchmarks (multilingual coverage):
    cmmlu        — 67 Chinese subjects multi-choice, accuracy
    ceval        — 52 Chinese exam subjects (STEM/Humanities/Social/Other), accuracy
    agieval_zh   — AGIEval Chinese subset (Gaokao + civil service + LogiQA-zh)
    gsm8k_zh     — translated GSM8K, regex-extract Chinese-aware numeric answer

Use the registry to dispatch by name:

    from synapforge.bench import run_bench, run_all
    out = run_bench("cmmlu", ckpt="runs/sf_100m/best.pt", tokenizer="Qwen/Qwen2.5-0.5B")
    summary = run_all(ckpt=..., tokenizer=..., names=["mmlu", "cmmlu", "ceval"])
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable, Optional

# Lazy registry — modules import torch / datasets only when first invoked.
BENCH_REGISTRY: Dict[str, str] = {
    # English suite.
    "humaneval":  "synapforge.bench.humaneval",
    "mbpp":       "synapforge.bench.mbpp",
    "mmlu":       "synapforge.bench.mmlu",
    "gsm8k":      "synapforge.bench.gsm8k",
    "hellaswag":  "synapforge.bench.hellaswag",
    "lambada":    "synapforge.bench.lambada",
    # Chinese suite (added 2026-05).
    "cmmlu":      "synapforge.bench.cmmlu",
    "ceval":      "synapforge.bench.ceval",
    "agieval_zh": "synapforge.bench.agieval_zh",
    "gsm8k_zh":   "synapforge.bench.gsm8k_zh",
}

# Bench groups for orchestrator convenience.
EN_BENCHES = ("humaneval", "mbpp", "mmlu", "gsm8k", "hellaswag", "lambada")
ZH_BENCHES = ("cmmlu", "ceval", "agieval_zh", "gsm8k_zh")


def run_bench(name: str, ckpt: Optional[str] = None, tokenizer: Optional[str] = None,
              model: Any = None, tok: Any = None, **kw) -> Dict[str, Any]:
    """Dispatch a single benchmark by name. Returns its summary dict."""
    if name not in BENCH_REGISTRY:
        raise KeyError(f"unknown bench {name!r}; registry={sorted(BENCH_REGISTRY)}")
    mod = import_module(BENCH_REGISTRY[name])
    return mod.run_bench(ckpt=ckpt, tokenizer=tokenizer, model=model, tok=tok, **kw)


def run_all(ckpt: Optional[str] = None, tokenizer: Optional[str] = None,
            names: Optional[Iterable[str]] = None, **kw) -> Dict[str, Dict[str, Any]]:
    """Run multiple benchmarks (default: all). Loads the model once and shares it."""
    names = list(names) if names else list(BENCH_REGISTRY)
    # Lazy-load model+tokenizer once and pass to each bench.
    from synapforge.eval.generate import load_synapforge, load_tokenizer
    model, _src = load_synapforge(ckpt) if ckpt else (None, "no-ckpt")
    tok = load_tokenizer(tokenizer) if tokenizer else None
    out: Dict[str, Dict[str, Any]] = {}
    for n in names:
        try:
            out[n] = run_bench(n, model=model, tok=tok, **kw)
        except Exception as e:  # pragma: no cover  — benches are noisy
            out[n] = {"error": f"{type(e).__name__}: {e}"}
    return out


__all__ = ["BENCH_REGISTRY", "EN_BENCHES", "ZH_BENCHES", "run_bench", "run_all"]
