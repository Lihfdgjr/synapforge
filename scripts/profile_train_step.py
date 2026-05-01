#!/usr/bin/env python3
"""Per-stage profiler for the synapforge KD training step.

Wraps a (CPU-or-GPU) replica of the `train_100m_kd.py` step in
`torch.profiler` with both CPU and CUDA activities, runs `--warmup` warm-up
iterations followed by `--steps` measured iterations, and emits:

    1. Per-op time table (top-N slowest, sorted by self_cuda_time_total or
       self_cpu_time_total).
    2. Per-stage time: dataloader / forward (student) / KD-teacher-forward /
       backward / optimizer.step.
    3. GPU memory peak per stage (reset_peak_memory_stats between stages).
    4. Chrome / Perfetto-compatible JSON trace at `--output PATH`.

CLI:
    python scripts/profile_train_step.py \\
        --steps 100 --warmup 100 \\
        --output runs/profile/step.json --visualize ascii

The script reuses the actual student/teacher classes when available; on CPU
or when the model can't be built (no CUDA) it falls back to a dummy
SynapforgeStub so the script always smoke-runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Each stage is timed individually so we have a separate "phase" budget that
# can be cross-checked against the full chrome:tracing JSON.
STAGES = ("dataloader", "forward_student", "forward_teacher",
          "kd_loss", "backward", "optimizer_step")


# ---------------------------------------------------------------- model factory


class _SynapforgeStub:
    """Minimal stand-in when the real synapforge model can't be built (CPU or
    missing deps). Keeps the script self-smoke-runnable."""

    def __init__(self, vocab: int, d: int, max_seq: int, device: str) -> None:
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        self.vocab = vocab
        self.d = d
        self.max_seq = max_seq
        self.device = device
        self.tok_embed = nn.Embedding(vocab, d).to(device)
        self.lstm = nn.LSTM(d, d, num_layers=2, batch_first=True).to(device)
        self.norm = nn.LayerNorm(d).to(device)
        self.lm_head = nn.Linear(d, vocab, bias=False).to(device)
        self.tie_lm_head = False
        self.parameters_list = list(self.tok_embed.parameters()) + \
            list(self.lstm.parameters()) + list(self.norm.parameters()) + \
            list(self.lm_head.parameters())

    def parameters(self):
        return iter(self.parameters_list)

    def to(self, device):
        self.device = device
        return self

    def encode(self, ids):
        x = self.tok_embed(ids)
        h, _ = self.lstm(x)
        return self.norm(h)

    def __call__(self, ids):
        return self.lm_head(self.encode(ids))


def _build_student(vocab: int, d: int, max_seq: int, device: str) -> Any:
    """Try to build the real SynapForge100M; fall back to stub on any error."""
    try:
        from synapforge.model_100m import build_synapforge_100m
        m = build_synapforge_100m(vocab=vocab, d=d, n_layers=2,
                                  loop_depth=1, max_seq=max_seq,
                                  ffn_ratio=2.0, sparsity=0.95)
        m.to(device)
        return m
    except Exception as exc:
        print(f"[profile] real model build failed ({type(exc).__name__}: {exc}); "
              f"using stub.", flush=True)
        return _SynapforgeStub(vocab, d, max_seq, device)


def _build_teacher(vocab: int, d: int, max_seq: int, device: str) -> Any:
    """A small frozen LSTM as a stand-in teacher; fast & deterministic for
    smoke runs. Real GPT-2 is loaded only if `--teacher hf` is set."""
    import torch
    import torch.nn as nn
    torch.manual_seed(1)

    class _Teacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.rnn = nn.LSTM(d, d, num_layers=1, batch_first=True)
            self.head = nn.Linear(d, vocab, bias=False)

        def forward(self, ids):
            x = self.emb(ids)
            h, _ = self.rnn(x)
            return self.head(h)

    t = _Teacher().to(device).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


# ---------------------------------------------------------------- step pieces


@contextmanager
def _timer(stage: str, store: Dict[str, List[float]],
           mem_store: Optional[Dict[str, List[int]]] = None,
           device: str = "cpu") -> None:
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
        if mem_store is not None:
            torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    yield
    if device == "cuda":
        torch.cuda.synchronize()
    store.setdefault(stage, []).append(time.perf_counter() - t0)
    if mem_store is not None and device == "cuda":
        mem_store.setdefault(stage, []).append(torch.cuda.max_memory_allocated())


def _kd_loss(student_logits, teacher_logits, T: float = 4.0):
    """Chunked KL — same shape as train_100m_kd._kd_loss but tiny."""
    import torch
    import torch.nn.functional as F
    V = student_logits.size(-1)
    Vt = teacher_logits.size(-1)
    if Vt > V:
        teacher_logits = teacher_logits[..., :V]
    elif V > Vt:
        # left-pad teacher with -inf so its softmax is zero on the extra cols
        pad = torch.full(teacher_logits.shape[:-1] + (V - Vt,), float("-inf"),
                         device=teacher_logits.device, dtype=teacher_logits.dtype)
        teacher_logits = torch.cat([teacher_logits, pad], dim=-1)
    sl = (student_logits / T).float()
    tl = (teacher_logits / T).float().detach()
    log_p = F.log_softmax(sl, dim=-1)
    p_t = F.softmax(tl, dim=-1)
    kl = (p_t * (p_t.add(1e-12).log() - log_p)).sum(-1).mean()
    return kl * (T * T)


def _make_dataloader(batch_size: int, seq_len: int, vocab: int, device: str):
    """Synthetic in-memory dataloader so smoke runs don't depend on parquet."""
    import torch

    def gen():
        while True:
            x = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long)
            y = torch.cat([x[:, 1:], torch.randint(0, vocab, (batch_size, 1), dtype=torch.long)], dim=1)
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    return gen()


# ---------------------------------------------------------------- main runner


def _ascii_bar(value: float, max_value: float, width: int = 30) -> str:
    if max_value <= 0:
        return ""
    frac = max(0.0, min(1.0, value / max_value))
    return "#" * int(frac * width) + "-" * (width - int(frac * width))


def _print_stage_summary(stage_times: Dict[str, List[float]],
                         stage_mem: Dict[str, List[int]],
                         visualize: str) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    print("\n=== per-stage timing (mean ± std over measured steps) ===")
    print(f"{'stage':<20} {'mean_ms':>10} {'std_ms':>8} {'p95_ms':>8} {'pct':>6} {'mem_MB':>10}")
    print("-" * 72)
    total_mean = sum((sum(v) / max(len(v), 1)) for v in stage_times.values())
    for stage in STAGES:
        ts = stage_times.get(stage, [])
        if not ts:
            continue
        mean = sum(ts) / len(ts)
        std = (sum((t - mean) ** 2 for t in ts) / max(len(ts) - 1, 1)) ** 0.5
        p95 = sorted(ts)[int(0.95 * (len(ts) - 1))]
        mem_kb = stage_mem.get(stage, [0])
        mem_mb = max(mem_kb) / (1024 * 1024) if mem_kb else 0.0
        pct = mean / total_mean * 100 if total_mean > 0 else 0
        summary[stage] = {
            "mean_ms": mean * 1000.0, "std_ms": std * 1000.0,
            "p95_ms": p95 * 1000.0, "pct": pct, "mem_mb": mem_mb,
        }
        bar = ""
        if visualize == "ascii":
            bar = "  " + _ascii_bar(mean, max((sum(v) / len(v)) for v in stage_times.values()))
        print(f"{stage:<20} {mean*1000:10.2f} {std*1000:8.2f} {p95*1000:8.2f} "
              f"{pct:5.1f}% {mem_mb:10.1f}{bar}")
    return summary


def _print_op_table(prof, top_n: int = 30) -> None:
    """Use torch.profiler key averages and print the slowest top-N ops."""
    try:
        keys = prof.key_averages()
    except Exception as exc:
        print(f"[profile] could not get key averages: {exc}")
        return
    rows = []
    for ev in keys:
        # Prefer cuda time when CUDA was active; otherwise CPU.
        cuda_time = getattr(ev, "self_device_time_total",
                            getattr(ev, "self_cuda_time_total", 0.0))
        rows.append((ev.key, ev.count, ev.cpu_time_total / 1000.0,
                     (cuda_time or 0.0) / 1000.0))
    # Pick column to sort by based on what's non-zero.
    has_cuda = any(r[3] > 0 for r in rows)
    rows.sort(key=lambda r: r[3] if has_cuda else r[2], reverse=True)
    rows = rows[:top_n]
    print(f"\n=== top {top_n} slowest ops (by "
          f"{'CUDA' if has_cuda else 'CPU'} self time, ms) ===")
    print(f"{'op':<48} {'calls':>6} {'cpu_ms':>10} {'cuda_ms':>10}")
    print("-" * 78)
    for name, calls, cpu_ms, cuda_ms in rows:
        if len(name) > 47:
            name = name[:44] + "..."
        print(f"{name:<48} {calls:>6} {cpu_ms:>10.2f} {cuda_ms:>10.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=4096,
                    help="Smoke vocab. Use 151643 for real Qwen.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--kd-every", type=int, default=1,
                    help="Run teacher forward every N steps (matches trainer).")
    ap.add_argument("--device", default=None,
                    help="cpu|cuda. Auto-detect by default.")
    ap.add_argument("--output", default="runs/profile/step.json",
                    help="Chrome:tracing / perfetto JSON trace path.")
    ap.add_argument("--visualize", default="ascii", choices=["ascii", "none"])
    ap.add_argument("--top-ops", type=int, default=30)
    ap.add_argument("--no-trace", action="store_true",
                    help="Skip writing the (large) chrome trace JSON.")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[profile] device={device} batch={args.batch_size} seq={args.seq_len} "
          f"hidden={args.hidden} vocab={args.vocab}", flush=True)

    student = _build_student(args.vocab, args.hidden, args.seq_len, device)
    teacher = _build_teacher(args.vocab, args.hidden, args.seq_len, device)

    optim = torch.optim.AdamW(list(student.parameters()), lr=3e-4)
    loader = _make_dataloader(args.batch_size, args.seq_len, args.vocab, device)

    # ---- warmup ----
    print(f"[profile] warmup {args.warmup} steps...", flush=True)
    for s in range(args.warmup):
        x, y = next(loader)
        logits = student(x)
        with torch.no_grad():
            t_logits = teacher(x)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
        )
        kd = _kd_loss(logits, t_logits)
        loss = 0.5 * ce + 0.5 * kd
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    if device == "cuda":
        torch.cuda.synchronize()

    # ---- profiled measured steps ----
    stage_times: Dict[str, List[float]] = {}
    stage_mem: Dict[str, List[int]] = {}
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for s in range(args.steps):
            with _timer("dataloader", stage_times, stage_mem, device):
                x, y = next(loader)

            with _timer("forward_student", stage_times, stage_mem, device):
                logits = student(x)

            t_logits = None
            if s % args.kd_every == 0:
                with _timer("forward_teacher", stage_times, stage_mem, device):
                    with torch.no_grad():
                        t_logits = teacher(x)

            with _timer("kd_loss", stage_times, stage_mem, device):
                ce = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                )
                if t_logits is not None:
                    kd = _kd_loss(logits, t_logits)
                    loss = 0.5 * ce + 0.5 * kd
                else:
                    loss = ce

            with _timer("backward", stage_times, stage_mem, device):
                optim.zero_grad(set_to_none=True)
                loss.backward()

            with _timer("optimizer_step", stage_times, stage_mem, device):
                optim.step()

            prof.step() if hasattr(prof, "step") else None

    # ---- print / save ----
    summary = _print_stage_summary(stage_times, stage_mem, args.visualize)
    _print_op_table(prof, top_n=args.top_ops)

    tok_per_step = args.batch_size * args.seq_len
    total_mean_s = sum(s["mean_ms"] for s in summary.values()) / 1000.0
    tok_per_s = tok_per_step / max(total_mean_s, 1e-6)
    print(f"\n[profile] tok/s = {tok_per_s:.0f}  "
          f"(batch={args.batch_size} seq={args.seq_len} mean_step={total_mean_s*1000:.1f}ms)")

    if not args.no_trace:
        try:
            prof.export_chrome_trace(args.output)
            print(f"[profile] wrote chrome trace -> {args.output}")
        except Exception as exc:
            print(f"[profile] chrome trace export failed: {exc}")

    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "device":       device,
            "batch_size":   args.batch_size,
            "seq_len":      args.seq_len,
            "hidden":       args.hidden,
            "vocab":        args.vocab,
            "warmup":       args.warmup,
            "steps":        args.steps,
            "tok_per_s":    tok_per_s,
            "stage":        summary,
            "stage_raw_ms": {
                s: [t * 1000.0 for t in stage_times.get(s, [])] for s in STAGES
            },
        }, f, indent=2)
    print(f"[profile] wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
