#!/usr/bin/env python3
"""scripts/bench_torch_compile.py — A/B torch.compile speedup on SynapForge100M.

Goal (T2.11 / DEEP_MAINT_QUEUE.md): the trainer ships a
``--torch-compile reduce-overhead`` flag claiming a 5-15% speedup, but the
number was never measured. This bench builds a full SynapForge100M
(d=512, n_layers=10, vocab=151936, ``backend='gpu_dense'`` so any GPU works,
not just A800), runs N forward+backward passes WITHOUT compile, then
re-builds the model with ``torch.compile(model, mode='reduce-overhead')``
and re-runs the same N passes. We report tok/s with vs without, the
ratio, and the % speedup.

Usage
-----
    python3 scripts/bench_torch_compile.py \\
        --steps 100 --batch-size 8 --seq-len 256 \\
        --device cuda

Output: a JSON record at ``bench_results/torch_compile_HHMMSS.json``::

    {
      "device": "cuda",
      "torch_version": "2.4.0",
      "batch_size": 8,
      "seq_len": 256,
      "vocab": 151936,
      "steps": 100,
      "no_compile_tok_s": 12345.6,
      "no_compile_step_ms": 165.7,
      "compile_tok_s": 13987.3,
      "compile_step_ms": 146.3,
      "speedup_ratio": 1.133,
      "pct_speedup": 13.3,
      "compile_supported": true,
      "compile_skip_reason": null
    }

Constraints
-----------
* Works on CPU too. Some torch / Windows combos refuse to compile on CPU
  (RuntimeError "Windows not yet supported for torch.compile"); we catch
  that and emit ``compile_supported=False`` + ``compile_skip_reason``,
  letting the JSON still be written so the harness is hermetic.
* Realistic shapes: bs=8, seq=256, vocab=151936. Avoids OOM on smaller
  GPUs (the live trainer runs bs=80 on A800-80GB; bs=8 leaves ~10x
  headroom and still saturates the kernel).
* Does NOT modify ``train_100m_kd.py``. T2.11 is a measurement task;
  the perf knob already exists in the trainer and the trainer keeps
  shipping with ``--torch-compile off`` as the default.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make the repo root importable so ``import synapforge`` works no matter
# where pytest / a shell invokes us from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_model(
    *,
    vocab: int,
    d: int,
    n_layers: int,
    seq_len: int,
    device: str,
    seed: int,
):
    """Build a fresh SynapForge100M and put it on ``device``.

    We always rebuild from scratch (not just deepcopy) so the no-compile
    and compile measurements start from identical-init weights — the
    seed is fixed so back-to-back calls produce bitwise-equal Param
    tensors. This guards against any "warm cache" artefact where the
    second build happens to land in a friendlier kernel.
    """
    import torch
    from synapforge.model_100m import SynapForge100M

    torch.manual_seed(seed)
    # max_seq must be >= seq_len; use seq_len directly so pos_embed is
    # exactly the right size (no slack memory).
    model = SynapForge100M(
        vocab=vocab,
        d=d,
        n_layers=n_layers,
        loop_depth=1,  # bench measures one loop per block, matches RDT default
        max_seq=seq_len,
        ffn_ratio=8.0,
        sparsity=0.95,
        # Default flags ON — this matches the live trainer config.
        freeze_vocab_tail=True,
        weight_quant_cfc="none",
    ).to(device)
    model.train()
    return model


def _make_batch(*, batch_size: int, seq_len: int, vocab: int, device: str):
    import torch
    # Token ids in [0, live_vocab) so freeze_vocab_tail's grad-zero hook
    # has nothing to do (zero rows beyond live_vocab keeps the bench
    # focused on the actual compute path, not the hook).
    live_vocab = min(vocab, 151643)  # qwen-2.5 live vocab
    x = torch.randint(
        low=0, high=live_vocab,
        size=(batch_size, seq_len),
        device=device, dtype=torch.long,
    )
    # next-token target = shift by 1; pad last col with arbitrary id.
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


def _time_steps(model, *, steps: int, batch_size: int, seq_len: int,
                vocab: int, device: str, warmup: int = 5) -> tuple[float, float]:
    """Run ``warmup + steps`` forward+backward passes and return
    ``(tok_per_s, mean_step_ms)`` measured on the last ``steps``.

    Uses ``torch.cuda.synchronize()`` on CUDA so we measure wall time
    of the kernel, not just dispatch.
    """
    import torch
    import torch.nn.functional as F

    is_cuda = device.startswith("cuda")
    # Build a fresh optimizer per call so we don't accumulate state
    # across the (no-compile, compile) pair.
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Warmup: not timed. On CUDA the first step pays the kernel
    # autotune; on torch.compile the first call pays the compile cost.
    for _ in range(warmup):
        x, y = _make_batch(
            batch_size=batch_size, seq_len=seq_len, vocab=vocab, device=device,
        )
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        opt.step()

    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = _make_batch(
            batch_size=batch_size, seq_len=seq_len, vocab=vocab, device=device,
        )
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        opt.step()
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = batch_size * seq_len * steps
    tok_per_s = tokens / elapsed if elapsed > 0 else 0.0
    mean_step_ms = (elapsed / steps) * 1000.0 if steps > 0 else 0.0
    return tok_per_s, mean_step_ms


def _try_compile(model, mode: str):
    """Wrap ``model`` with ``torch.compile``; return (wrapped, error or None).

    Returns ``(model, None)`` on success or ``(None, error_str)`` on
    failure (Windows torch 2.0, missing triton, OS quirks). Caller
    must treat ``None`` as "skip the compile arm".
    """
    import torch
    if not hasattr(torch, "compile"):
        return None, "torch.compile not available (torch < 2.0)"
    try:
        # ``dynamic=True`` matches the trainer's setting (eval batches
        # may have a different shape than train batches, so static
        # would force a recompile).
        return torch.compile(model, mode=mode, dynamic=True), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _run(args) -> dict:
    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[bench] cuda requested but not available; falling back to cpu",
              flush=True)
        device = "cpu"

    print(f"[bench] device={device} torch={torch.__version__}", flush=True)
    print(f"[bench] shapes: bs={args.batch_size} seq={args.seq_len} "
          f"vocab={args.vocab} steps={args.steps}", flush=True)

    # ---- arm 1: no compile ----
    print("[bench] arm 1/2 -- no torch.compile", flush=True)
    m_nc = _build_model(
        vocab=args.vocab, d=args.d, n_layers=args.n_layers,
        seq_len=args.seq_len, device=device, seed=args.seed,
    )
    nc_tok_s, nc_step_ms = _time_steps(
        m_nc, steps=args.steps, batch_size=args.batch_size,
        seq_len=args.seq_len, vocab=args.vocab, device=device,
    )
    print(f"[bench]   no_compile: {nc_tok_s:>9,.1f} tok/s  "
          f"({nc_step_ms:.1f} ms/step)", flush=True)

    # Free the no-compile model before building the second; on small
    # GPUs the resident state would otherwise pin ~600 MiB of weights.
    del m_nc
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- arm 2: compile ----
    print(f"[bench] arm 2/2 -- torch.compile mode={args.compile_mode!r}",
          flush=True)
    m_c = _build_model(
        vocab=args.vocab, d=args.d, n_layers=args.n_layers,
        seq_len=args.seq_len, device=device, seed=args.seed,
    )
    m_compiled, compile_err = _try_compile(m_c, mode=args.compile_mode)
    if m_compiled is None:
        # Skip the compile arm cleanly. Common reasons:
        #   * Windows + torch 2.0.x ("Windows not yet supported for
        #     torch.compile")
        #   * missing triton on a fresh CPU box
        # We still emit a valid JSON so callers can see we ran.
        msg = ("CPU compile not supported on this PyTorch version: "
               + (compile_err or "unknown"))
        print(f"[bench]   compile SKIPPED -- {msg}", flush=True)
        c_tok_s, c_step_ms = 0.0, 0.0
        compile_supported = False
        compile_skip_reason = msg
    else:
        # The inductor backend can fail at *first call* (not at
        # ``torch.compile`` construction): e.g. Windows runners that lack
        # MSVC ``cl.exe``, or boxes without a working C compiler. Trap
        # those here and degrade to a clean skip rather than crashing
        # the bench.
        try:
            c_tok_s, c_step_ms = _time_steps(
                m_compiled, steps=args.steps, batch_size=args.batch_size,
                seq_len=args.seq_len, vocab=args.vocab, device=device,
            )
            print(f"[bench]   compile:    {c_tok_s:>9,.1f} tok/s  "
                  f"({c_step_ms:.1f} ms/step)", flush=True)
            compile_supported = True
            compile_skip_reason = None
        except Exception as exc:
            msg = (f"compile backend failed at first call: "
                   f"{type(exc).__name__}: {exc}")
            print(f"[bench]   compile SKIPPED -- {msg}", flush=True)
            c_tok_s, c_step_ms = 0.0, 0.0
            compile_supported = False
            compile_skip_reason = msg

    if compile_supported and nc_tok_s > 0:
        ratio = c_tok_s / nc_tok_s
        pct = (ratio - 1.0) * 100.0
    else:
        ratio = 0.0
        pct = 0.0

    record = {
        "device": device,
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "vocab": args.vocab,
        "d": args.d,
        "n_layers": args.n_layers,
        "steps": args.steps,
        "compile_mode": args.compile_mode,
        "no_compile_tok_s": round(nc_tok_s, 2),
        "no_compile_step_ms": round(nc_step_ms, 3),
        "compile_tok_s": round(c_tok_s, 2),
        "compile_step_ms": round(c_step_ms, 3),
        "speedup_ratio": round(ratio, 4),
        "pct_speedup": round(pct, 2),
        "compile_supported": compile_supported,
        "compile_skip_reason": compile_skip_reason,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return record


def _save_json(record: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%H%M%S")
    out = out_dir / f"torch_compile_{stamp}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench_torch_compile",
        description="A/B torch.compile speedup on SynapForge100M.",
    )
    p.add_argument("--steps", type=int, default=100,
                   help="timed steps per arm (default 100)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="bench batch size (default 8 -- bs=80 is OOM on "
                        "anything smaller than A800-80GB)")
    p.add_argument("--seq-len", type=int, default=256,
                   help="sequence length (default 256, matches live trainer)")
    p.add_argument("--vocab", type=int, default=151936,
                   help="vocab size (default 151936, matches Qwen 2.5 trainer)")
    p.add_argument("--d", type=int, default=512,
                   help="hidden dim (default 512, SynapForge100M canonical)")
    p.add_argument("--n-layers", type=int, default=10,
                   help="HybridBlock count (default 10, SynapForge100M)")
    p.add_argument("--device", default="cuda",
                   help="cuda or cpu (default cuda; falls back to cpu if "
                        "cuda is unavailable)")
    p.add_argument("--compile-mode", default="reduce-overhead",
                   choices=["reduce-overhead", "max-autotune", "default"],
                   help="torch.compile mode (default reduce-overhead -- "
                        "the safer choice for the PLIFCell sequence loop)")
    p.add_argument("--seed", type=int, default=42,
                   help="rng seed for fresh model build (default 42)")
    p.add_argument("--out-dir", default=str(ROOT / "bench_results"),
                   help="where to write the JSON record")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    record = _run(args)
    out = _save_json(record, Path(args.out_dir))
    print(f"[bench] wrote {out}", flush=True)
    if record["compile_supported"]:
        print(f"[bench] speedup: {record['pct_speedup']:+.2f}% "
              f"(ratio {record['speedup_ratio']:.3f}x)", flush=True)
    else:
        print(f"[bench] compile arm skipped: {record['compile_skip_reason']}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
