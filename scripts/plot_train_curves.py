"""plot_train_curves -- matplotlib renderer for live train.log on rental.

Reads a `train_run3*.log` file (rental-side trainer stdout/stderr capture)
and emits a 4-panel PNG:

    Panel 1: Train CE  vs step
    Panel 2: Val PPL   vs step  (TTT vs holdout overlaid)
    Panel 3: Spike rate mean + dead/total layers vs step  (twin-axes)
    Panel 4: Throughput tok/s vs step

Pure-regex parser (no trainer imports). Handles missing data gracefully
(e.g. logs that pre-date the `[stl=...]` field, or truncated streams).
matplotlib only -- no seaborn/plotly.

Patterns (loose -- tolerate optional fields and surrounding noise):

    [HH:MM:SS] step N loss=L ce=C kd=K z=Z lr=LR step_ms=MS tok/s=T mem_GB=M [stl=ST]
    VAL step N: val_ppl_ttt=PT val_ppl_holdout=PH (honest)
    spike: mean=M range=[a, b] dead=D/T sat=S/T

CLI::

    python scripts/plot_train_curves.py --log /workspace/runs/v24h_qwen3/train_run3l.log
    python scripts/plot_train_curves.py --log run3m.log --out docs/CURVES_run3m.png

Run as a script *or* import the regex helpers and `parse_log()` in tests.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
# Loose -- allow optional whitespace, optional bracketed timestamp prefix,
# and optional trailing `[stl=...]` field. Keys are space-separated `key=value`
# pairs. We don't anchor to start-of-line because some captures have ANSI
# resets / log-prefixes upstream of the metric block.
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_NUMI = r"-?\d+"

STEP_RE = re.compile(
    r"\bstep\s+(?P<step>\d+)\b"
    r".*?\bce=(?P<ce>" + _NUM + r")"
    r"(?:.*?\bkd=(?P<kd>" + _NUM + r"))?"
    r"(?:.*?\bz=(?P<z>" + _NUM + r"))?"
    r"(?:.*?\blr=(?P<lr>" + _NUM + r"))?"
    r"(?:.*?\bstep_ms=(?P<step_ms>" + _NUM + r"))?"
    r"(?:.*?\btok/s=(?P<tok_s>" + _NUM + r"))?"
    r"(?:.*?\bmem_GB=(?P<mem_gb>" + _NUM + r"))?"
    r"(?:.*?\bstl=(?P<stl>" + _NUM + r"))?"
)

VAL_RE = re.compile(
    r"\bVAL\s+step\s+(?P<step>\d+)\s*:\s*"
    r"val_ppl_ttt=(?P<ppl_ttt>" + _NUM + r")"
    r"\s+val_ppl_holdout=(?P<ppl_holdout>" + _NUM + r")"
)

SPIKE_RE = re.compile(
    r"\bspike\s*:\s*mean=(?P<mean>" + _NUM + r")"
    r"\s+range=\[\s*(?P<lo>" + _NUM + r")\s*,\s*(?P<hi>" + _NUM + r")\s*\]"
    r"\s+dead=(?P<dead>" + _NUMI + r")\s*/\s*(?P<total>" + _NUMI + r")"
    r"\s+sat=(?P<sat>" + _NUMI + r")\s*/\s*\d+"
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ParsedLog:
    """Per-event time series. Each list is independently sortable on `step`."""

    step_events: List[Dict[str, float]] = field(default_factory=list)
    val_events: List[Dict[str, float]] = field(default_factory=list)
    spike_events: List[Dict[str, float]] = field(default_factory=list)

    @property
    def n_step(self) -> int:
        return len(self.step_events)

    @property
    def n_val(self) -> int:
        return len(self.val_events)

    @property
    def n_spike(self) -> int:
        return len(self.spike_events)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_step_line(line: str) -> Optional[Dict[str, float]]:
    """Return the dict of fields for one `step N ce=...` line, or None."""
    m = STEP_RE.search(line)
    if not m:
        return None
    gd = m.groupdict()
    out: Dict[str, float] = {"step": float(gd["step"]), "ce": float(gd["ce"])}
    for k in ("kd", "z", "lr", "step_ms", "tok_s", "mem_gb", "stl"):
        v = _to_float(gd.get(k))
        if v is not None:
            out[k] = v
    return out


def parse_val_line(line: str) -> Optional[Dict[str, float]]:
    m = VAL_RE.search(line)
    if not m:
        return None
    gd = m.groupdict()
    return {
        "step": float(gd["step"]),
        "ppl_ttt": float(gd["ppl_ttt"]),
        "ppl_holdout": float(gd["ppl_holdout"]),
    }


def parse_spike_line(line: str, step_hint: Optional[int] = None) -> Optional[Dict[str, float]]:
    """Spike lines don't carry their own step number -- caller passes the
    most-recent step from a preceding `step N` line via ``step_hint``.
    """
    m = SPIKE_RE.search(line)
    if not m:
        return None
    gd = m.groupdict()
    out: Dict[str, float] = {
        "mean": float(gd["mean"]),
        "range_lo": float(gd["lo"]),
        "range_hi": float(gd["hi"]),
        "dead": float(gd["dead"]),
        "total": float(gd["total"]),
        "sat": float(gd["sat"]),
    }
    if step_hint is not None:
        out["step"] = float(step_hint)
    return out


def parse_log(lines: Iterable[str]) -> ParsedLog:
    """Walk `lines` once, accumulate all 3 event streams.

    We track the most-recent ``step N`` so spike lines (which do not embed
    their own step number) are aligned to the closest preceding step.
    """
    out = ParsedLog()
    last_step: Optional[int] = None
    for line in lines:
        # Order matters: step lines are most common. Try them first.
        step_ev = parse_step_line(line)
        if step_ev is not None:
            out.step_events.append(step_ev)
            last_step = int(step_ev["step"])
            continue
        val_ev = parse_val_line(line)
        if val_ev is not None:
            out.val_events.append(val_ev)
            continue
        spike_ev = parse_spike_line(line, step_hint=last_step)
        if spike_ev is not None and "step" in spike_ev:
            out.spike_events.append(spike_ev)
            continue
    # Sort defensively: rental nohup logs sometimes interleave on relaunch.
    out.step_events.sort(key=lambda d: d["step"])
    out.val_events.sort(key=lambda d: d["step"])
    out.spike_events.sort(key=lambda d: d["step"])
    return out


def parse_log_file(path: Path) -> ParsedLog:
    """Read `path` (UTF-8, errors ignored -- some rental logs have stray bytes)."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return parse_log(f)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _import_matplotlib():
    """Import matplotlib in non-interactive Agg backend (CI-safe)."""
    import matplotlib  # noqa: WPS433
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433
    return plt


def render_png(parsed: ParsedLog, out_path: Path, title: str,
               figsize: Tuple[float, float] = (12.0, 8.0), dpi: int = 150) -> Path:
    """Render the 4-panel figure and write it to ``out_path``.

    Panels with no data plot a "(no data)" annotation rather than crash.
    """
    plt = _import_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=13)

    ax_ce, ax_val, ax_spike, ax_thr = (
        axes[0][0], axes[0][1], axes[1][0], axes[1][1],
    )

    # Panel 1 -- Train CE -------------------------------------------------
    ax_ce.set_title("Train CE vs step")
    ax_ce.set_xlabel("step")
    ax_ce.set_ylabel("ce")
    ax_ce.grid(True, alpha=0.3)
    if parsed.step_events:
        xs = [e["step"] for e in parsed.step_events]
        ys = [e["ce"] for e in parsed.step_events]
        ax_ce.plot(xs, ys, color="C0", linewidth=1.0, label=f"ce (n={len(xs)})")
        ax_ce.legend(loc="upper right", fontsize=8)
    else:
        ax_ce.text(0.5, 0.5, "(no step events)", ha="center", va="center",
                   transform=ax_ce.transAxes, color="gray")

    # Panel 2 -- Val PPL --------------------------------------------------
    ax_val.set_title("Val PPL vs step (TTT vs holdout)")
    ax_val.set_xlabel("step")
    ax_val.set_ylabel("ppl")
    ax_val.grid(True, alpha=0.3)
    if parsed.val_events:
        xs = [e["step"] for e in parsed.val_events]
        ys_ttt = [e["ppl_ttt"] for e in parsed.val_events]
        ys_ho = [e["ppl_holdout"] for e in parsed.val_events]
        ax_val.plot(xs, ys_ttt, color="C2", marker="o", linewidth=1.0, label="ttt")
        ax_val.plot(xs, ys_ho, color="C3", marker="s", linewidth=1.0, label="holdout")
        ax_val.legend(loc="upper right", fontsize=8)
        # Log scale if dynamic range is large (early-training ppl bounces 1e2-1e5).
        finite = [v for v in ys_ttt + ys_ho if v and v > 0]
        if finite and max(finite) / max(min(finite), 1e-9) > 50.0:
            ax_val.set_yscale("log")
    else:
        ax_val.text(0.5, 0.5, "(no VAL events)", ha="center", va="center",
                    transform=ax_val.transAxes, color="gray")

    # Panel 3 -- Spike rate + dead/total ---------------------------------
    ax_spike.set_title("Spike rate (mean) + dead/total layers")
    ax_spike.set_xlabel("step")
    ax_spike.set_ylabel("mean spike rate", color="C1")
    ax_spike.grid(True, alpha=0.3)
    ax_spike.tick_params(axis="y", labelcolor="C1")
    ax_spike2 = ax_spike.twinx()
    ax_spike2.set_ylabel("dead / total", color="C4")
    ax_spike2.tick_params(axis="y", labelcolor="C4")
    if parsed.spike_events:
        xs = [e["step"] for e in parsed.spike_events]
        ys_mean = [e["mean"] for e in parsed.spike_events]
        ys_dead = [
            (e["dead"] / e["total"]) if e["total"] > 0 else 0.0
            for e in parsed.spike_events
        ]
        ax_spike.plot(xs, ys_mean, color="C1", linewidth=1.0, label="mean")
        ax_spike2.plot(xs, ys_dead, color="C4", linewidth=1.0,
                       linestyle="--", label="dead/total")
        # Build a combined legend.
        lines1, labels1 = ax_spike.get_legend_handles_labels()
        lines2, labels2 = ax_spike2.get_legend_handles_labels()
        ax_spike.legend(lines1 + lines2, labels1 + labels2,
                        loc="upper right", fontsize=8)
    else:
        ax_spike.text(0.5, 0.5, "(no spike events)", ha="center", va="center",
                      transform=ax_spike.transAxes, color="gray")

    # Panel 4 -- Throughput tok/s ----------------------------------------
    ax_thr.set_title("Throughput tok/s vs step")
    ax_thr.set_xlabel("step")
    ax_thr.set_ylabel("tok/s")
    ax_thr.grid(True, alpha=0.3)
    thr_xs = [e["step"] for e in parsed.step_events if "tok_s" in e]
    thr_ys = [e["tok_s"] for e in parsed.step_events if "tok_s" in e]
    if thr_xs:
        ax_thr.plot(thr_xs, thr_ys, color="C5", linewidth=1.0,
                    label=f"tok/s (n={len(thr_xs)})")
        # Rolling mean overlay (window=20 if we have >=40 points).
        if len(thr_ys) >= 40:
            w = max(20, len(thr_ys) // 20)
            roll = []
            for i in range(len(thr_ys)):
                lo = max(0, i - w + 1)
                roll.append(sum(thr_ys[lo:i + 1]) / (i - lo + 1))
            ax_thr.plot(thr_xs, roll, color="black", linewidth=1.2,
                        linestyle=":", label=f"rolling-{w}")
        ax_thr.legend(loc="lower right", fontsize=8)
    else:
        ax_thr.text(0.5, 0.5, "(no tok/s field)", ha="center", va="center",
                    transform=ax_thr.transAxes, color="gray")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_out_for(log: Path) -> Path:
    """`docs/CURVES_<basename-without-ext>.png` next to repo `docs/`."""
    repo_root = Path(__file__).resolve().parent.parent
    stem = log.stem  # e.g. "train_run3l"
    # Prefer "run-name" suffix. `train_run3l.log` -> "run3l".
    if stem.startswith("train_"):
        stem = stem[len("train_"):]
    return repo_root / "docs" / f"CURVES_{stem}.png"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--log", required=True, type=Path,
                    help="path to train.log (rental-side trainer capture)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG path (default: docs/CURVES_<run>.png)")
    ap.add_argument("--title", type=str, default=None,
                    help="figure title (default: log filename basename)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--figsize", type=str, default="12,8",
                    help='comma-separated "W,H" inches (default 12,8)')
    args = ap.parse_args(argv)

    log_path = args.log.resolve()
    if not log_path.exists():
        print(f"[plot] ERROR: --log {log_path} does not exist", file=sys.stderr)
        return 2

    out_path = args.out.resolve() if args.out else _default_out_for(log_path)
    title = args.title or log_path.name
    try:
        w, h = [float(x) for x in args.figsize.split(",")]
    except (ValueError, AttributeError):
        print(f"[plot] WARN: bad --figsize {args.figsize!r}; using 12,8", file=sys.stderr)
        w, h = 12.0, 8.0

    parsed = parse_log_file(log_path)
    print(f"[plot] log={log_path}")
    print(
        f"[plot] parsed: step={parsed.n_step} val={parsed.n_val} "
        f"spike={parsed.n_spike}"
    )
    if parsed.n_step == 0 and parsed.n_val == 0 and parsed.n_spike == 0:
        print("[plot] WARN: no events found -- emitting empty PNG anyway")

    written = render_png(
        parsed, out_path, title=title,
        figsize=(w, h), dpi=args.dpi,
    )
    size = os.path.getsize(written)
    print(f"[plot] wrote {written}  ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
