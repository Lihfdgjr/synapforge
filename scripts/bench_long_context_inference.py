"""Demonstrate constant-time per-token decode at long contexts.

This is the **investor demo proof artifact** for the
"100M+ context, constant-time inference" claim. It shows that the per-
token decode latency of the SynapForge architecture is independent of
the prefix length, because the recurrent state (LiquidCell + PLIF) is
the only carrier of context — there is no growing KV cache.

Method
------
For each ``seq_len`` in ``--lens``:

1. Build a fresh ``InferenceState`` and run ``incremental_step`` over a
   warmup prompt of length ``seq_len`` (this seeds the state).
2. Time the next ``--decode-tokens`` calls to ``incremental_step``.
3. Report per-token latency, total wall-time, and tokens/s.

The decode-phase latency should stay flat across ``seq_len`` (modulo
small allocator / cache effects), because every step is the same
constant-cost block-stack pass over a single token.

Usage
-----
::

    python scripts/bench_long_context_inference.py
    python scripts/bench_long_context_inference.py --d 256 --n-layers 4
    python scripts/bench_long_context_inference.py \
        --lens 128 1024 8192 65536 524288 1048576

Note on max_seq
---------------
The model carries a learnable ``pos_embed`` of length ``max_seq``;
positions beyond that fall off a cliff (currently raises ValueError in
encode/incremental_step). For the bench we lift ``max_seq`` to cover
the largest tested sequence — that's a positional-embedding budget,
not a per-step compute budget. The headline claim is about the latter.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

# Allow running directly: ``python scripts/bench_long_context_inference.py``
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from synapforge.inference import InferenceState, incremental_step  # noqa: E402
from synapforge.model_100m import build_synapforge_100m  # noqa: E402


def _build_model(d: int, n_layers: int, max_seq: int, vocab: int):
    """Tiny demo model — same architecture as production, sized for CPU bench."""
    torch.manual_seed(20260502)
    return build_synapforge_100m(
        vocab=vocab,
        d=d,
        n_layers=n_layers,
        loop_depth=1,
        max_seq=max_seq,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
        use_grad_checkpoint=False,
        freeze_vocab_tail=False,
        live_vocab=vocab,
        lm_head_spectral_norm=False,
        weight_quant_cfc="none",
        latent_k=0,
    ).eval()


def _warmup_state(model, seq_len: int, vocab: int) -> InferenceState:
    """Seed an InferenceState by running incremental_step over a random prompt."""
    state = InferenceState()
    g = torch.Generator().manual_seed(seq_len)
    # Stream tokens through one at a time; for 1M+ this takes a while
    # but it's a ONE-TIME cost outside the timed window.
    chunk = 4096
    pos = 0
    while pos < seq_len:
        n = min(chunk, seq_len - pos)
        ids = torch.randint(0, vocab, (1, n), generator=g, dtype=torch.long)
        for t in range(n):
            _, state = incremental_step(model, ids[:, t], state)
        pos += n
        # Periodic checkpoint print for very long warmups so the user sees
        # progress instead of thinking the script hung.
        if seq_len >= 1024 and pos % 16384 == 0:
            print(f"  warmup {pos}/{seq_len}", flush=True)
    return state


def bench_decode_phase(
    model, state: InferenceState, n_tokens: int, vocab: int
) -> dict:
    """Time the decode phase: incremental_step calls over n_tokens."""
    g = torch.Generator().manual_seed(42)
    ids = torch.randint(0, vocab, (1, n_tokens), generator=g, dtype=torch.long)
    # Warm one call so any lazy-init (cudnn etc) isn't measured.
    _, state = incremental_step(model, ids[:, 0], state)
    t0 = time.perf_counter()
    for t in range(1, n_tokens):
        _, state = incremental_step(model, ids[:, t], state)
    dt = time.perf_counter() - t0
    n_timed = n_tokens - 1
    return {
        "total_ms": dt * 1000.0,
        "per_token_ms": dt / n_timed * 1000.0,
        "tokens_per_s": n_timed / dt,
        "n_timed": n_timed,
    }


def run_bench(
    seq_lens: List[int],
    decode_tokens: int,
    d: int,
    n_layers: int,
    vocab: int,
    skip_above: int = 0,
) -> dict:
    """Run the (seq_len -> latency) sweep.

    ``skip_above``: lengths > this are reported as ``skipped`` rather
    than executed. Lets the demo print the FULL claim table even when
    we can't physically warm up to 1M on a laptop.
    """
    max_seq = max(seq_lens) + decode_tokens + 64
    model = _build_model(d=d, n_layers=n_layers, max_seq=max_seq, vocab=vocab)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: d={d} n_layers={n_layers} loop_depth={model.loop_depth} "
          f"params={n_params:,}")
    print(f"  device: cpu (CUDA: {torch.cuda.is_available()})")
    print()
    print(f"  {'seq_len':>10}  {'per-tok ms':>12}  {'total ms':>10}  {'tok/s':>8}")
    print("  " + "-" * 50)
    rows = []
    for seq_len in seq_lens:
        if skip_above and seq_len > skip_above:
            row = {
                "seq_len": seq_len,
                "skipped": True,
                "reason": f"seq_len > skip_above={skip_above}",
            }
            rows.append(row)
            print(f"  {seq_len:>10}  {'(skipped)':>12}")
            continue

        # Warmup state to seq_len (untimed).
        t_warm = time.perf_counter()
        state = _warmup_state(model, seq_len, vocab=vocab)
        warm_dt = time.perf_counter() - t_warm

        result = bench_decode_phase(
            model, state, n_tokens=decode_tokens, vocab=vocab
        )
        row = {
            "seq_len": seq_len,
            "warmup_s": warm_dt,
            **result,
        }
        rows.append(row)
        print(
            f"  {seq_len:>10}  {result['per_token_ms']:>10.3f}ms  "
            f"{result['total_ms']:>8.1f}ms  {result['tokens_per_s']:>6.1f}"
        )

    # Summary metric: max / min per-token latency across executed rows.
    timed = [r for r in rows if not r.get("skipped")]
    if len(timed) >= 2:
        per_tok = [r["per_token_ms"] for r in timed]
        ratio = max(per_tok) / min(per_tok)
        print()
        print(f"  per-token max/min ratio across {len(timed)} contexts: {ratio:.2f}x")
        print(f"  (constant-time claim: ratio should approach 1.0; <2.0 is honest)")

    return {
        "model": {"d": d, "n_layers": n_layers, "params": n_params},
        "rows": rows,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench_long_context_inference")
    p.add_argument(
        "--lens", type=int, nargs="+",
        default=[128, 1024, 8192, 65536, 524288, 1048576],
        help="sequence lengths to test (default: 128, 1K, 8K, 64K, 512K, 1M)",
    )
    p.add_argument(
        "--decode-tokens", type=int, default=32,
        help="how many post-prefill tokens to time at each seq_len",
    )
    p.add_argument("--d", type=int, default=64, help="hidden dim (small for CPU bench)")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument(
        "--skip-above", type=int, default=8192,
        help=("skip warmup-to-seq_len for lens > this on CPU. The 1M warmup "
              "takes minutes on CPU even at d=64. The decode-phase latency "
              "is the headline claim; we'll print 'skipped' for the rest. "
              "On A100 with --d 64 you can bump this to 1048576."),
    )
    p.add_argument(
        "--save", type=str, default=None,
        help="optional path to dump JSON results",
    )
    args = p.parse_args(argv)

    print("=" * 60)
    print("SynapForge long-context inference bench")
    print("R-fold incremental decode: per-token cost is constant in seq_len")
    print("=" * 60)
    out = run_bench(
        seq_lens=args.lens,
        decode_tokens=args.decode_tokens,
        d=args.d,
        n_layers=args.n_layers,
        vocab=args.vocab,
        skip_above=args.skip_above,
    )
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  saved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
