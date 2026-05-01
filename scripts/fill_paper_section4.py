#!/usr/bin/env python3
"""scripts/fill_paper_section4.py -- auto-fill paper Section 4 (Results)
from a live train log + a bench-results JSON (T6.2).

Goal (T6.2 / DEEP_MAINT_QUEUE.md): paper draft Section 4 (Results) is
re-rendered every fire from the actual rental train log + the most
recent bench JSON, so the LaTeX numbers never drift behind reality.
The script emits ``paper/section4_auto.tex``, a self-contained
``\\input``able snippet referenced from ``paper/draft.tex``.

What we render:

    Table 4.1 -- Train CE / Val PPL (TTT + holdout) at key steps.
                 Default key steps: 1k, 5k, 10k, plus the "best"
                 (lowest val_ppl_holdout). Missing rows -> \\textbf{TBD}.

    Table 4.2 -- Throughput tok/s (mean), GPU memory peak (mem_GB max),
                 training wall-clock (estimated from step_ms mean *
                 last_step). Bench JSON adds compile-mode tok/s and
                 speedup_ratio if available.

    Figure 4.1 -- caption only (the actual PNG is rendered separately
                  by ``scripts/plot_train_curves.py``). Caption text
                  reads "Train/val curves for Run 3<letter>" derived
                  from the log filename when it matches ``train_run3?.log``.

CLI::

    python scripts/fill_paper_section4.py \\
        --log /workspace/runs/v24h_qwen3/train_run3l.log \\
        --bench-json bench_results/torch_compile_023700.json \\
        --ckpt-step 10000 \\
        --out paper/section4_auto.tex

Constraints
-----------
* Pure stdlib parsing; we re-use the regex helpers from
  ``scripts/plot_train_curves.py`` (T5.5) so log changes only need to
  be patched once.
* If ``paper/draft.tex`` doesn't exist we ship a minimal skeleton with
  a single ``\\input{section4_auto}`` placeholder. (We never overwrite
  an existing draft.)
* Booktabs LaTeX. ``\\toprule / \\midrule / \\bottomrule``.
* If a metric is missing in the log (e.g., we picked ``--ckpt-step
  10000`` but the log only has data up to step 8500), we emit
  ``\\textbf{TBD}`` rather than silently fudging.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Re-use the parser from T5.5 so we don't drift on log format.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import plot_train_curves as ptc  # noqa: E402

TBD = r"\textbf{TBD}"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@dataclass
class Section4Data:
    """All the numbers Section 4 needs, normalised to floats / strings."""
    rows: List[Dict[str, object]] = field(default_factory=list)
    throughput_tok_s_mean: Optional[float] = None
    mem_gb_peak: Optional[float] = None
    wall_clock_hours: Optional[float] = None
    last_step: Optional[int] = None
    compile_supported: Optional[bool] = None
    compile_pct_speedup: Optional[float] = None
    bench_device: Optional[str] = None
    figure_caption: str = "Train/val curves."


def _nearest_step_event(events: List[Dict[str, float]],
                        target_step: int,
                        tolerance: int = 250) -> Optional[Dict[str, float]]:
    """Return the event whose step is closest to ``target_step``, but only
    if it's within ``tolerance`` (so step=10000 doesn't snap to step=4000
    just because that's the only data we have)."""
    if not events:
        return None
    best = min(events, key=lambda e: abs(int(e["step"]) - target_step))
    return best if abs(int(best["step"]) - target_step) <= tolerance else None


def _best_val_event(val_events: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Find the val event with the lowest val_ppl_holdout."""
    if not val_events:
        return None
    valid = [e for e in val_events if e.get("ppl_holdout", float("inf")) > 0]
    return min(valid, key=lambda e: e["ppl_holdout"]) if valid else None


def _fmt(v: Optional[float], fmt: str = "{:.3f}") -> str:
    """Format ``v`` with ``fmt``, or return TBD when v is None / NaN."""
    if v is None:
        return TBD
    try:
        f = float(v)
    except (TypeError, ValueError):
        return TBD
    if f != f:  # NaN
        return TBD
    return fmt.format(f)


def _fmt_int(v: Optional[float]) -> str:
    if v is None:
        return TBD
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return TBD


# ---------------------------------------------------------------------------
# Aggregation: log + bench JSON -> Section4Data
# ---------------------------------------------------------------------------
def _figure_caption_from_log(log_path: Path) -> str:
    """`train_run3l.log` -> 'Train/val curves for Run 3l.'"""
    m = re.match(r"train_(run\d+[a-z]?)", log_path.stem)
    if m:
        run_name = m.group(1)
        # 'run3l' -> 'Run 3l'
        pretty = re.sub(r"^run", "Run ", run_name)
        return f"Train/val curves for {pretty}."
    return f"Train/val curves for {log_path.stem}."


def _wall_clock_hours(step_events: List[Dict[str, float]]) -> Optional[float]:
    """Estimate wall-clock from step_ms mean * last_step.

    Why mean instead of cumulative real time? The log timestamps are
    HH:MM:SS but no date, so wall-clock arithmetic across day-rollovers
    is fragile. ``step_ms * total_steps`` is a stable lower-bound that
    matches what we already report in PROGRESS.md.
    """
    ms_vals = [e["step_ms"] for e in step_events if "step_ms" in e]
    if not ms_vals or not step_events:
        return None
    last_step = int(step_events[-1]["step"])
    if last_step <= 0:
        return None
    mean_ms = sum(ms_vals) / len(ms_vals)
    return (mean_ms * last_step) / (1000.0 * 60.0 * 60.0)


def aggregate(parsed: ptc.ParsedLog,
              bench: Optional[dict],
              ckpt_step: int,
              key_steps: Tuple[int, ...] = (1000, 5000, 10000)) -> Section4Data:
    """Walk parsed log + bench JSON, populate Section4Data with formatted
    values. Missing inputs leave None fields (rendered as TBD downstream)."""
    data = Section4Data()

    # ---- Table 4.1 rows ---------------------------------------------------
    # One row per key step + an explicit "ckpt_step" row + a "best" row.
    target_steps = sorted(set(list(key_steps) + [ckpt_step]))
    for s in target_steps:
        ev_step = _nearest_step_event(parsed.step_events, s)
        ev_val = _nearest_step_event(parsed.val_events, s)
        label = f"{s}"
        if s == ckpt_step:
            label = f"{s} (snapshot)"
        data.rows.append({
            "label": label,
            "step": int(ev_step["step"]) if ev_step else None,
            "ce": ev_step.get("ce") if ev_step else None,
            "val_ppl_ttt": ev_val["ppl_ttt"] if ev_val else None,
            "val_ppl_holdout": ev_val["ppl_holdout"] if ev_val else None,
        })

    best_val = _best_val_event(parsed.val_events)
    if best_val is not None:
        ev_step = _nearest_step_event(parsed.step_events, int(best_val["step"]))
        data.rows.append({
            "label": f"{int(best_val['step'])} (best)",
            "step": int(best_val["step"]),
            "ce": ev_step.get("ce") if ev_step else None,
            "val_ppl_ttt": best_val.get("ppl_ttt"),
            "val_ppl_holdout": best_val.get("ppl_holdout"),
        })
    else:
        data.rows.append({
            "label": "best",
            "step": None,
            "ce": None,
            "val_ppl_ttt": None,
            "val_ppl_holdout": None,
        })

    # ---- Table 4.2 throughput / memory / wall-clock ----------------------
    if parsed.step_events:
        data.last_step = int(parsed.step_events[-1]["step"])
        tok_s = [e["tok_s"] for e in parsed.step_events if "tok_s" in e]
        if tok_s:
            data.throughput_tok_s_mean = sum(tok_s) / len(tok_s)
        mem = [e["mem_gb"] for e in parsed.step_events if "mem_gb" in e]
        if mem:
            data.mem_gb_peak = max(mem)
        data.wall_clock_hours = _wall_clock_hours(parsed.step_events)

    # ---- Bench JSON overlay ----------------------------------------------
    if bench:
        data.compile_supported = bool(bench.get("compile_supported", False))
        if data.compile_supported:
            data.compile_pct_speedup = float(bench.get("pct_speedup", 0.0))
        data.bench_device = bench.get("device")

    return data


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------
_HEADER = r"""% AUTO-GENERATED by scripts/fill_paper_section4.py -- DO NOT EDIT BY HAND.
% Re-run after every train log / bench JSON refresh:
%   python scripts/fill_paper_section4.py --log <log> --bench-json <bench>
%       --ckpt-step <N> --out paper/section4_auto.tex
"""


def _render_table_4_1(data: Section4Data) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Train CE and validation perplexity (TTT vs holdout) at key training steps. "
        r"\textit{TTT} = test-time-trained val (with inference-time STDP); "
        r"\textit{holdout} = static val with STDP frozen.}",
        r"\label{tab:train_progression}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Step & Train CE & Val PPL (TTT) & Val PPL (holdout) & Notes \\",
        r"\midrule",
    ]
    for row in data.rows:
        step_cell = _fmt_int(row.get("step"))
        ce_cell = _fmt(row.get("ce"), "{:.3f}")
        ttt_cell = _fmt(row.get("val_ppl_ttt"), "{:.2f}")
        ho_cell = _fmt(row.get("val_ppl_holdout"), "{:.2f}")
        # Pull the parenthetical tag out of the label so the Step column
        # stays purely numeric.
        label = str(row.get("label", ""))
        m = re.match(r"^\s*\d+\s*\((.+)\)\s*$", label)
        notes = m.group(1) if m else "--"
        lines.append(
            f"{step_cell} & {ce_cell} & {ttt_cell} & {ho_cell} & {notes} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _render_table_4_2(data: Section4Data) -> str:
    tok_s = _fmt(data.throughput_tok_s_mean, "{:,.1f}")
    mem = _fmt(data.mem_gb_peak, "{:.2f}")
    wall = _fmt(data.wall_clock_hours, "{:.2f}")
    if data.compile_supported and data.compile_pct_speedup is not None:
        compile_cell = (
            f"{data.compile_pct_speedup:+.2f}\\% on {data.bench_device or 'unknown'}"
        )
    elif data.compile_supported is False:
        compile_cell = r"skipped (CPU/Windows)"
    else:
        compile_cell = TBD

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Throughput, peak GPU memory, training wall-clock, and "
        r"\texttt{torch.compile} speedup over the same A800 backbone. "
        r"Throughput is the mean over all \texttt{step\_ms}-stamped log lines.}",
        r"\label{tab:perf}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Throughput (tok/s mean) & {tok_s} \\\\",
        f"GPU memory peak (GB) & {mem} \\\\",
        f"Training wall-clock (h, est.) & {wall} \\\\",
        f"\\texttt{{torch.compile}} speedup & {compile_cell} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _render_figure_caption(data: Section4Data) -> str:
    return (
        r"\begin{figure}[h]" + "\n"
        r"\centering" + "\n"
        r"% Place rendered PNG via scripts/plot_train_curves.py here:" + "\n"
        r"% \includegraphics[width=0.95\linewidth]{figures/curves.png}" + "\n"
        f"\\caption{{{data.figure_caption}}}" + "\n"
        r"\label{fig:train_val_curves}" + "\n"
        r"\end{figure}"
    )


def render_section4(data: Section4Data) -> str:
    """Compose the whole ``section4_auto.tex`` payload."""
    parts = [
        _HEADER,
        r"\section{Results}",
        r"\label{sec:results}",
        "",
        _render_table_4_1(data),
        "",
        _render_table_4_2(data),
        "",
        _render_figure_caption(data),
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Skeleton draft.tex creation
# ---------------------------------------------------------------------------
DRAFT_SKELETON = r"""% paper/draft.tex -- SynapForge paper draft skeleton.
%
% Section 4 (Results) is auto-generated by:
%   python scripts/fill_paper_section4.py --log <train.log> \
%       --bench-json bench_results/torch_compile_*.json \
%       --ckpt-step <N> --out paper/section4_auto.tex
%
% Re-run after every rental log scp.
\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}

\title{SynapForge: A 100M-Parameter Liquid-Spiking Hybrid for Long-Context Inference}
\author{SynapForge Team}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
SynapForge is a 100M-parameter hybrid recurrent network combining
multiplicative Closed-form Continuous-time (CfC) cells with
Parametric LIF spiking neurons, targeting test-time STDP plasticity
and matmul-free inference. We present training progression, validation
perplexity, and throughput on A800-80GB.
\end{abstract}

\section{Introduction}
\label{sec:intro}
% TODO: motivation, contributions, prior art.

\section{Architecture}
\label{sec:arch}
% TODO: HybridBlock, CfC + PLIF, NeuroMCP, R-fold.

\section{Training}
\label{sec:training}
% TODO: KD setup, data, hyperparameters, phase manager.

\input{section4_auto}

\section{Discussion}
\label{sec:discussion}
% TODO: limitations, future work.

\bibliographystyle{plain}
% \bibliography{refs}

\end{document}
"""


def maybe_write_draft_skeleton(draft_path: Path) -> bool:
    """Create ``paper/draft.tex`` if it doesn't already exist. Returns
    True iff we wrote a new file (so the caller can log it)."""
    if draft_path.exists():
        return False
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(DRAFT_SKELETON, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_bench_json(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    if not path.exists():
        print(f"[fill] WARN: --bench-json {path} not found; skipping",
              file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[fill] WARN: cannot parse {path}: {exc}", file=sys.stderr)
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fill_paper_section4",
        description="Auto-fill paper Section 4 (Results) from train log + bench JSON.",
    )
    ap.add_argument("--log", required=True, type=Path,
                    help="path to train_run3*.log (rental-side trainer capture)")
    ap.add_argument("--bench-json", type=Path, default=None,
                    help="optional bench_results/torch_compile_*.json with throughput numbers")
    ap.add_argument("--ckpt-step", type=int, default=10000,
                    help="which step's snapshot to highlight in Table 4.1 (default 10000)")
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "paper" / "section4_auto.tex",
                    help="output .tex path (default paper/section4_auto.tex)")
    ap.add_argument("--draft", type=Path,
                    default=_REPO_ROOT / "paper" / "draft.tex",
                    help="draft.tex; created from skeleton if absent (default paper/draft.tex)")
    args = ap.parse_args(argv)

    log_path = args.log.resolve()
    if not log_path.exists():
        print(f"[fill] ERROR: --log {log_path} does not exist", file=sys.stderr)
        return 2

    parsed = ptc.parse_log_file(log_path)
    bench = _load_bench_json(args.bench_json.resolve() if args.bench_json else None)
    data = aggregate(parsed, bench, ckpt_step=args.ckpt_step)
    data.figure_caption = _figure_caption_from_log(log_path)

    text = render_section4(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    wrote_draft = maybe_write_draft_skeleton(args.draft.resolve())
    print(f"[fill] log={log_path}")
    print(
        f"[fill] parsed: step={parsed.n_step} val={parsed.n_val} "
        f"spike={parsed.n_spike}"
    )
    print(f"[fill] rows in Table 4.1: {len(data.rows)}")
    print(
        f"[fill] tok/s mean={_fmt(data.throughput_tok_s_mean, '{:,.1f}')} "
        f"mem_GB peak={_fmt(data.mem_gb_peak, '{:.2f}')} "
        f"wall_clock_h={_fmt(data.wall_clock_hours, '{:.2f}')}"
    )
    if wrote_draft:
        print(f"[fill] wrote NEW draft skeleton -> {args.draft}")
    print(f"[fill] wrote {args.out}  ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
