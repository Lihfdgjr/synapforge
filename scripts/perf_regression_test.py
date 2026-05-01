#!/usr/bin/env python3
"""Single-shot perf regression test.

Builds a tiny synapforge replica, runs `--steps` training iterations on
synthetic data, and asserts:

    tok_per_s >= --baseline-tok-per-s
    no stage > 1.2× its persisted baseline (regression tolerance: 20%)

Persists baseline numbers to `tests/perf_baseline.json` (separate per device
key: `cpu` / `cuda`). On first invocation just records a baseline; on
subsequent runs compares.

CI usage:
    python scripts/perf_regression_test.py        # uses default 5000 tok/s
    python scripts/perf_regression_test.py --device cuda --baseline-tok-per-s 25000

Exit codes:
    0   all stages within tolerance
    1   regression detected (one or more stages > 1.2× baseline OR tok/s low)

Probes covered:
    * model forward (student)
    * optimizer step
    * dataloader collate (synthetic — checks Python-side allocator stalls)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASELINE = ROOT / "tests" / "perf_baseline.json"
TOLERANCE = 1.20  # 20% slowdown is the regression threshold


def _build_tiny_model(vocab: int, d: int, device: str):
    import torch
    import torch.nn as nn
    torch.manual_seed(0)

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.gru = nn.GRU(d, d, num_layers=2, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab, bias=False)

        def forward(self, ids):
            x = self.emb(ids)
            h, _ = self.gru(x)
            return self.head(self.norm(h))

    m = _Tiny().to(device)
    return m


def _gen_collate_batch(batch_size: int, seq_len: int, vocab: int, device: str):
    """Collate-like step run on CPU then moved to device."""
    import torch
    cpu = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)
    y_cpu = torch.cat([cpu[:, 1:], torch.randint(0, vocab, (batch_size, 1), dtype=torch.long)], dim=1)
    return cpu.to(device, non_blocking=True), y_cpu.to(device, non_blocking=True)


def _run_probes(args) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F

    device = args.device
    model = _build_tiny_model(args.vocab, args.hidden, device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    for _ in range(20):
        x, y = _gen_collate_batch(args.batch_size, args.seq_len, args.vocab, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    if device == "cuda":
        torch.cuda.synchronize()

    fwd_times: List[float] = []
    bwd_times: List[float] = []
    opt_times: List[float] = []
    coll_times: List[float] = []

    t_start = time.perf_counter()
    cum_tok = 0
    for _ in range(args.steps):
        # dataloader/collate stage
        t0 = time.perf_counter()
        x, y = _gen_collate_batch(args.batch_size, args.seq_len, args.vocab, device)
        if device == "cuda":
            torch.cuda.synchronize()
        coll_times.append(time.perf_counter() - t0)

        # forward
        t0 = time.perf_counter()
        logits = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)

        # backward
        t0 = time.perf_counter()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        bwd_times.append(time.perf_counter() - t0)

        # optimizer
        t0 = time.perf_counter()
        optim.step()
        if device == "cuda":
            torch.cuda.synchronize()
        opt_times.append(time.perf_counter() - t0)

        cum_tok += args.batch_size * args.seq_len

    total_s = time.perf_counter() - t_start
    return {
        "tok_per_s":  cum_tok / max(total_s, 1e-6),
        "forward_ms": (sum(fwd_times) / len(fwd_times)) * 1000.0,
        "backward_ms": (sum(bwd_times) / len(bwd_times)) * 1000.0,
        "opt_ms":      (sum(opt_times) / len(opt_times)) * 1000.0,
        "collate_ms":  (sum(coll_times) / len(coll_times)) * 1000.0,
        "total_s":     total_s,
        "n_steps":     args.steps,
    }


def _load_baseline(path: Path, device: str) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(device, {})
    except Exception:
        return {}


def _save_baseline(path: Path, device: str, results: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: Dict[str, Any] = {}
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            blob = {}
    blob[device] = {**results, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def _compare(results: Dict[str, float], baseline: Dict[str, float],
             min_tok_per_s: float) -> Tuple[bool, List[str]]:
    """Return (ok, reasons). Tok/s is checked at floor. Per-stage at TOLERANCE×."""
    reasons: List[str] = []
    if results["tok_per_s"] < min_tok_per_s:
        reasons.append(f"tok/s {results['tok_per_s']:.0f} < floor {min_tok_per_s:.0f}")
    if not baseline:
        return (not reasons, reasons + ["[no baseline yet — saving]"])
    for stage in ("forward_ms", "backward_ms", "opt_ms", "collate_ms"):
        cur = results[stage]
        ref = baseline.get(stage)
        if ref is None or ref <= 0:
            continue
        ratio = cur / ref
        if ratio > TOLERANCE:
            reasons.append(f"{stage}: {cur:.2f}ms vs baseline {ref:.2f}ms "
                           f"(ratio {ratio:.2f}× > {TOLERANCE:.2f}×)")
    # Also flag a tok/s drop > 20% vs baseline.
    base_tok = baseline.get("tok_per_s")
    if base_tok and base_tok > 0:
        if results["tok_per_s"] < base_tok / TOLERANCE:
            reasons.append(f"tok/s: {results['tok_per_s']:.0f} vs baseline {base_tok:.0f} "
                           f"(drop > {(TOLERANCE-1)*100:.0f}%)")
    return (len(reasons) == 0, reasons)


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--device", default=None,
                    help="cpu|cuda. Auto-detect by default.")
    ap.add_argument("--baseline-tok-per-s", type=float, default=None,
                    help="Floor tok/s. Default: 5000 (cpu) / 25000 (cuda).")
    ap.add_argument("--baseline-path", default=str(DEFAULT_BASELINE))
    ap.add_argument("--update-baseline", action="store_true",
                    help="Force overwrite of saved baseline with current run.")
    args = ap.parse_args()

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.baseline_tok_per_s is None:
        args.baseline_tok_per_s = 25000.0 if args.device == "cuda" else 5000.0

    baseline_path = Path(args.baseline_path)
    print(f"[perf-regress] device={args.device} steps={args.steps} "
          f"batch={args.batch_size} seq={args.seq_len} "
          f"floor={args.baseline_tok_per_s:.0f} tok/s", flush=True)

    results = _run_probes(args)
    print(f"[perf-regress] tok/s={results['tok_per_s']:.0f}  "
          f"fwd={results['forward_ms']:.2f}ms  "
          f"bwd={results['backward_ms']:.2f}ms  "
          f"opt={results['opt_ms']:.2f}ms  "
          f"coll={results['collate_ms']:.2f}ms")

    baseline = _load_baseline(baseline_path, args.device)
    ok, reasons = _compare(results, baseline, args.baseline_tok_per_s)

    if not baseline or args.update_baseline:
        _save_baseline(baseline_path, args.device, results)
        print(f"[perf-regress] saved baseline -> {baseline_path} ({args.device})")

    if ok:
        print("[perf-regress] PASS")
        return 0
    print("[perf-regress] FAIL")
    for r in reasons:
        print(f"  - {r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
