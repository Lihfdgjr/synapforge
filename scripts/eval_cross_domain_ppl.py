"""T9.6 -- Cross-domain perplexity eval harness.

Goal
----
Anti-overfit verifier. When the trainer reports `val ppl 10` we want to be
confident the number is **not** just memorisation of the alpaca-zh + synth-zh
training mix. This harness measures perplexity on **held-out** splits of
domains the model has never been trained on (default: wikitext-103 sample,
C4-en, GSM8K reasoning, HumanEval code, plus a Chinese news split for
balance), then emits a per-domain table + a weighted cross-domain average.

Read alongside `docs/BASELINE_COMPARISON.md` and the
`## ppl 10 verification protocol` section in `docs/SCALING_RATIONALE.md`.

Usage
-----
Smoke (CI-safe -- no HF download, no GPU)::

    python scripts/eval_cross_domain_ppl.py --smoke \\
        --out runs/eval_smoke.json

Real run (requires an actual ckpt + the HF datasets cache pre-warmed)::

    python scripts/eval_cross_domain_ppl.py \\
        --ckpt runs/v24h_qwen3/best_step_007500.pt \\
        --out runs/v24h_qwen3/cross_domain.json \\
        --domains wikitext,c4,gsm8k,humaneval,zh_news \\
        --max-tokens-per-domain 100000

Output
------
JSON shape::

    {
        "ckpt": "...",
        "ts": "2026-05-02T12:34:56Z",
        "domains": {
            "wikitext":   {"ppl": 18.5, "tokens": 100000, "status": "ok"},
            "c4":         {"ppl": 22.1, "tokens": 100000, "status": "ok"},
            "gsm8k":      {"ppl": 41.0, "tokens":  85000, "status": "ok"},
            "humaneval":  {"ppl": 35.2, "tokens":  20000, "status": "ok"},
            "zh_news":    {"ppl": 28.7, "tokens": 100000, "status": "ok"},
        },
        "weighted_avg_ppl": 26.9,
        "n_domains_evaluated": 5,
        "n_domains_skipped": 0,
        "smoke": false
    }

Constraints
-----------
* DO NOT include any training data here -- strict holdout.
* DO NOT call HF `datasets` in CI tests -- use ``--smoke`` for those.
* CPU-friendly: `--device cpu --max-tokens-per-domain 1000` runs in <1s on
  a vanilla laptop with the smoke fixtures.
* OOM-resilient: a domain that hits OOM during forward is **skipped** with
  ``status: "oom"`` so the rest of the eval still completes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Domain registry. Each entry maps to a held-out source the model has *not*
# seen during pretraining or SFT (alpaca-zh + synth-zh + wikitext-103-train).
# We deliberately include wikitext-103 *test* split which is the standard
# held-out chunk -- not the train slice that the trainer ingests.
# ---------------------------------------------------------------------------
DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wikitext": {
        "hf_path": ("wikitext", "wikitext-103-raw-v1"),
        "split": "test",
        "text_field": "text",
        "weight": 1.0,
        "description": "WikiText-103 test split -- canonical held-out LM eval.",
    },
    "c4": {
        "hf_path": ("allenai/c4", "en"),
        "split": "validation",
        "text_field": "text",
        "weight": 1.0,
        "description": "C4 English validation slice -- generalist web text.",
    },
    "gsm8k": {
        "hf_path": ("gsm8k", "main"),
        "split": "test",
        "text_field": "question",
        "weight": 0.5,
        "description": "GSM8K test questions -- math reasoning domain.",
    },
    "humaneval": {
        "hf_path": ("openai_humaneval",),
        "split": "test",
        "text_field": "prompt",
        "weight": 0.3,
        "description": "HumanEval prompts -- code-generation domain.",
    },
    "zh_news": {
        "hf_path": ("RUCAIBox/Chinese-Generation-DOC",),
        "split": "test",
        "text_field": "text",
        "weight": 0.7,
        "description": "Chinese news -- bilingual generalisation check.",
    },
}


# ---------------------------------------------------------------------------
# Smoke fixtures: deterministic synthetic per-domain "tokens" so the test
# harness can verify the math without any HF download. Each fixture returns
# a list[list[int]] -- the same shape ParquetTokenStream would emit.
# ---------------------------------------------------------------------------
def _smoke_token_stream(domain: str, max_tokens: int, seed: int = 0) -> List[List[int]]:
    """Deterministic token chunks per domain. Same seed -> same stream."""
    rng = _SmallRNG(seed=hash((domain, seed)) & 0xFFFFFFFF)
    chunks: List[List[int]] = []
    used = 0
    seq_len = 64  # smoke uses short sequences for speed
    vocab = 256
    while used < max_tokens:
        n = min(seq_len, max_tokens - used)
        if n < 8:
            break
        chunks.append([rng.next() % vocab for _ in range(n)])
        used += n
    return chunks


class _SmallRNG:
    """Stateless hash-step PRNG; avoids importing numpy in CI."""

    def __init__(self, seed: int) -> None:
        self.state = (seed | 1) & 0xFFFFFFFF

    def next(self) -> int:
        # Linear congruential -- enough for deterministic test fixtures.
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state


# ---------------------------------------------------------------------------
# Tiny mock model for smoke tests. Returns logits of fixed shape so the
# CE/exp(loss) path exercises end-to-end without torch installed.
# ---------------------------------------------------------------------------
class _MockModel:
    """Pure-Python mock LM for smoke tests.

    Computes a *deterministic* ppl per token chunk by hashing the chunk's
    bytes -- no torch import, no GPU, runs in microseconds.
    """

    vocab_size = 256

    def perplexity(self, chunks: Sequence[Sequence[int]]) -> Tuple[float, int]:
        """Return (ppl, n_tokens) over the chunks, deterministically."""
        total_loss = 0.0
        total_tokens = 0
        for c in chunks:
            if len(c) < 2:
                continue
            # Hash-based fake CE in [0.5, 4.0] -- a band that gives ppl in
            # ~exp(0.5)..exp(4.0) = 1.6..55, plausible for an LM.
            h = 0
            for x in c:
                h = (h * 31 + int(x)) & 0xFFFFFFFF
            fake_ce = 0.5 + (h % 350) / 100.0  # 0.5..4.0
            n_tok = len(c) - 1
            total_loss += fake_ce * n_tok
            total_tokens += n_tok
        ppl = math.exp(total_loss / max(total_tokens, 1))
        return ppl, total_tokens


# ---------------------------------------------------------------------------
# Real loader hook (lazy). Returns a callable model.perplexity(chunks)->(ppl,n).
# ---------------------------------------------------------------------------
def _load_real_model(ckpt_path: str, device: str = "cpu") -> Any:  # pragma: no cover
    """Load a Synap-1 checkpoint via the existing factory + return a wrapper.

    Lazy-imports torch + synapforge so the smoke path has zero heavy deps.
    Real evaluation goes through this; CI tests stub it with --smoke.
    """
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    from synapforge.eval.generate import load_synapforge  # type: ignore

    model, src = load_synapforge(ckpt_path=ckpt_path, device=device)
    print(f"[eval-cd] loaded model: {src}", flush=True)

    class _Real:
        def perplexity(self, chunks: Sequence[Sequence[int]]) -> Tuple[float, int]:
            total_loss = 0.0
            total_tokens = 0
            with torch.no_grad():
                for c in chunks:
                    if len(c) < 2:
                        continue
                    ids = torch.tensor([list(c)], dtype=torch.long, device=device)
                    try:
                        out = model(ids)
                        logits = out.logits if hasattr(out, "logits") else out
                    except Exception:
                        # Fall through to caller's OOM handling.
                        raise
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    # next-token CE
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = ids[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="sum",
                    )
                    n_tok = shift_labels.numel()
                    total_loss += float(loss)
                    total_tokens += n_tok
            ppl = math.exp(total_loss / max(total_tokens, 1))
            return ppl, total_tokens

    return _Real()


# ---------------------------------------------------------------------------
# HF dataset loader (lazy, smoke-bypassed). Returns list[list[int]] chunks.
# ---------------------------------------------------------------------------
def _load_hf_domain_tokens(  # pragma: no cover
    domain: str, max_tokens: int, tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
) -> List[List[int]]:
    """Pull a held-out split for `domain` and tokenize it.

    NEVER called in CI tests -- those go through ``--smoke``.
    """
    try:
        from datasets import load_dataset  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"Real eval needs `datasets` + `transformers`; install or use --smoke. "
            f"({e})"
        )

    spec = DOMAIN_REGISTRY[domain]
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset(*spec["hf_path"], split=spec["split"])
    chunks: List[List[int]] = []
    used = 0
    seq_len = 512
    for row in ds:
        text = row.get(spec["text_field"], "")
        if not text or not text.strip():
            continue
        ids = tok.encode(text, add_special_tokens=False)
        # split into seq_len chunks
        for start in range(0, len(ids), seq_len):
            seg = ids[start : start + seq_len]
            if len(seg) < 8:
                continue
            chunks.append(seg)
            used += len(seg)
            if used >= max_tokens:
                return chunks
    return chunks


# ---------------------------------------------------------------------------
# Core eval driver. Per-domain try/except so OOM on one domain doesn't
# nuke the whole eval -- we record status="oom" and continue.
# ---------------------------------------------------------------------------
def evaluate_cross_domain(
    model: Any,
    domains: Sequence[str],
    max_tokens_per_domain: int,
    smoke: bool = False,
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
) -> Dict[str, Any]:
    per_domain: Dict[str, Dict[str, Any]] = {}
    n_skipped = 0

    for d in domains:
        if d not in DOMAIN_REGISTRY:
            per_domain[d] = {
                "ppl": None,
                "tokens": 0,
                "status": "unknown_domain",
                "weight": 0.0,
            }
            n_skipped += 1
            continue

        spec = DOMAIN_REGISTRY[d]
        try:
            if smoke:
                chunks = _smoke_token_stream(d, max_tokens_per_domain)
            else:
                chunks = _load_hf_domain_tokens(d, max_tokens_per_domain, tokenizer_name)
            ppl, n_tok = model.perplexity(chunks)
            per_domain[d] = {
                "ppl": float(ppl),
                "tokens": int(n_tok),
                "status": "ok",
                "weight": float(spec["weight"]),
            }
        except (MemoryError, RuntimeError) as e:
            # CUDA OOM raises RuntimeError("CUDA out of memory."); we treat
            # any RuntimeError mentioning oom/cuda as a per-domain skip.
            msg = f"{type(e).__name__}: {e}"
            is_oom = "oom" in msg.lower() or "out of memory" in msg.lower()
            per_domain[d] = {
                "ppl": None,
                "tokens": 0,
                "status": "oom" if is_oom else "error",
                "error": msg,
                "weight": float(spec["weight"]),
            }
            n_skipped += 1
            print(f"[eval-cd] domain={d} SKIPPED ({per_domain[d]['status']}): {msg}",
                  file=sys.stderr, flush=True)
            continue
        except Exception as e:  # any other error -> domain-skip, keep going
            per_domain[d] = {
                "ppl": None,
                "tokens": 0,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "weight": float(spec["weight"]),
                "trace": traceback.format_exc(limit=3) if not smoke else None,
            }
            n_skipped += 1
            print(f"[eval-cd] domain={d} SKIPPED (error): {e}", file=sys.stderr,
                  flush=True)
            continue

    # Weighted cross-domain average. Skip domains with status != ok.
    num = 0.0
    den = 0.0
    n_ok = 0
    for d, rec in per_domain.items():
        if rec.get("status") != "ok" or rec.get("ppl") is None:
            continue
        w = float(rec.get("weight", 0.0))
        num += w * float(rec["ppl"])
        den += w
        n_ok += 1
    weighted_avg = (num / den) if den > 0 else None

    return {
        "domains": per_domain,
        "weighted_avg_ppl": weighted_avg,
        "n_domains_evaluated": n_ok,
        "n_domains_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def _parse_domains(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default=None,
                    help="Path to Synap-1 .pt checkpoint (omit with --smoke).")
    ap.add_argument("--out", required=True,
                    help="Output JSON path.")
    ap.add_argument("--domains",
                    default="wikitext,c4,gsm8k,humaneval,zh_news",
                    help="Comma-separated domain keys (see DOMAIN_REGISTRY).")
    ap.add_argument("--max-tokens-per-domain", type=int, default=100_000,
                    help="Cap per-domain token budget (default 100K).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device for the real model. Smoke ignores this.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B",
                    help="HF tokenizer for the real path.")
    ap.add_argument("--smoke", action="store_true",
                    help="Use synthetic fixtures + mock model (CI-safe).")
    args = ap.parse_args(argv)

    domains = _parse_domains(args.domains)
    if not domains:
        print("[eval-cd] FATAL: --domains was empty", file=sys.stderr)
        return 2

    if args.smoke:
        model = _MockModel()
        ckpt_label = "<smoke>"
    else:
        if not args.ckpt:
            print("[eval-cd] FATAL: --ckpt is required without --smoke",
                  file=sys.stderr)
            return 2
        if not Path(args.ckpt).exists():
            print(f"[eval-cd] FATAL: ckpt not found: {args.ckpt}",
                  file=sys.stderr)
            return 2
        model = _load_real_model(args.ckpt, device=args.device)  # pragma: no cover
        ckpt_label = args.ckpt

    t0 = time.time()
    result = evaluate_cross_domain(
        model=model,
        domains=domains,
        max_tokens_per_domain=int(args.max_tokens_per_domain),
        smoke=bool(args.smoke),
        tokenizer_name=args.tokenizer,
    )
    elapsed = time.time() - t0

    payload = {
        "ckpt": ckpt_label,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed, 3),
        "max_tokens_per_domain": int(args.max_tokens_per_domain),
        "smoke": bool(args.smoke),
        **result,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # Console summary -- one line per domain plus the headline.
    print(f"[eval-cd] wrote {out_path}", flush=True)
    for d, rec in payload["domains"].items():
        if rec.get("status") == "ok":
            print(f"  {d:>10}: ppl={rec['ppl']:7.2f}  tokens={rec['tokens']}",
                  flush=True)
        else:
            print(f"  {d:>10}: SKIPPED ({rec['status']})", flush=True)
    print(f"[eval-cd] weighted_avg_ppl={payload['weighted_avg_ppl']}  "
          f"evaluated={payload['n_domains_evaluated']}  "
          f"skipped={payload['n_domains_skipped']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
