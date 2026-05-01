"""plot_eval_curves -- ASCII (+ optional PNG) renderer for auto_eval/index.json.

Reads `<out_dir>/auto_eval/index.json` (produced by auto_eval_daemon.py) and
emits compact text-mode line plots for every available metric:

    ce loss vs step
    ppl vs step              (derived from ce when missing: exp(ce))
    mmlu acc vs step
    hellaswag acc vs step
    lambada acc vs step
    humaneval pass@1 vs step
    mbpp pass@1 vs step
    gsm8k acc vs step

No matplotlib required. If matplotlib is installed and `--png OUTDIR` is
passed, we additionally write per-metric PNGs to OUTDIR.

CLI:
    python scripts/plot_eval_curves.py --index runs/sf_100m/auto_eval/index.json
    python scripts/plot_eval_curves.py --index ... --png runs/sf_100m/auto_eval/figs/

Investor demo: this runs anywhere (no GPU, no extra deps in the ASCII path).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ASCII_W = 56
ASCII_H = 14


def _load_index(idx_path: Path) -> Dict[str, Any]:
    if not idx_path.exists():
        return {}
    # Tolerate UTF-8 BOM (PowerShell `Set-Content -Encoding utf8` adds one).
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(idx_path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            print(f"[plot] failed to parse {idx_path}: {exc}", file=sys.stderr)
            return {}
    print(f"[plot] failed to read {idx_path} as utf-8/utf-8-sig", file=sys.stderr)
    return {}


def _series_from(idx: Dict[str, Any]) -> Dict[str, List[Tuple[int, float]]]:
    """Pivot index.json {bucket: {step, bench:{...}, ce, ppl, ...}} into per-metric lists."""
    series: Dict[str, List[Tuple[int, float]]] = {
        "ce": [], "ppl": [],
        "mmlu": [], "hellaswag": [], "lambada": [],
        "humaneval": [], "mbpp": [], "gsm8k": [],
    }
    for _, entry in idx.items():
        step = entry.get("step")
        if step is None:
            continue
        ce = entry.get("ce")
        ppl = entry.get("ppl")
        if ce is None and isinstance(ppl, (int, float)):
            try:
                ce = math.log(float(ppl))
            except (ValueError, OverflowError):
                ce = None
        if isinstance(ce, (int, float)):
            series["ce"].append((int(step), float(ce)))
        if isinstance(ppl, (int, float)):
            series["ppl"].append((int(step), float(ppl)))
        bench = entry.get("bench") or {}
        for k, v in bench.items():
            if k in series and isinstance(v, (int, float)):
                series[k].append((int(step), float(v)))
    for k in list(series):
        series[k].sort()
    return series


def _render_ascii(name: str, points: List[Tuple[int, float]],
                  width: int = ASCII_W, height: int = ASCII_H) -> str:
    """Return a multi-line string: title + plot + axis labels + summary."""
    out: List[str] = []
    out.append(f"--- {name} ---")
    if not points:
        out.append("  (no data)")
        return "\n".join(out)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = max(x1 - x0, 1)
    yspan = max(y1 - y0, 1e-9)

    grid = [[" "] * width for _ in range(height)]
    for x, y in points:
        cx = int((x - x0) / span * (width - 1))
        # Single-point or flat series: pin to vertical mid for visibility.
        if yspan <= 1e-9:
            cy = height // 2
        else:
            cy = int((y - y0) / yspan * (height - 1))
            cy = (height - 1) - cy
        grid[cy][cx] = "*"

    # axis ruler on the leftmost column
    for r in range(height):
        if r == 0:
            grid[r][0] = "|"
        elif r == height - 1:
            grid[r][0] = "|"
        else:
            if grid[r][0] == " ":
                grid[r][0] = ":"

    # render
    out.append(f"  y in [{y0:.4g}, {y1:.4g}]   x in [{x0}, {x1}]")
    for r, row in enumerate(grid):
        prefix = ""
        if r == 0:
            prefix = f"{y1:>8.4g} "
        elif r == height - 1:
            prefix = f"{y0:>8.4g} "
        else:
            prefix = " " * 9
        out.append(prefix + "".join(row))
    out.append(" " * 9 + "+" + "-" * (width - 1))
    out.append(" " * 9 + f"{x0:<{width // 2}}{x1:>{width - width // 2 - 1}}")
    out.append(f"  N={len(points)}  first=({xs[0]}, {ys[0]:.4g})  last=({xs[-1]}, {ys[-1]:.4g})")
    return "\n".join(out)


def _summary_table(series: Dict[str, List[Tuple[int, float]]]) -> str:
    rows = []
    cols = ["metric", "n", "first", "last", "best", "delta"]
    for name, pts in series.items():
        if not pts:
            rows.append([name, "0", "-", "-", "-", "-"])
            continue
        ys = [y for _, y in pts]
        # For ce/ppl: best = min; for accuracy benches: best = max.
        is_loss = name in ("ce", "ppl")
        best = min(ys) if is_loss else max(ys)
        delta = ys[-1] - ys[0]
        rows.append([
            name, str(len(pts)),
            f"{ys[0]:.4g}", f"{ys[-1]:.4g}",
            f"{best:.4g}",
            f"{delta:+.4g}",
        ])
    widths = [max(len(c), max(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    sep = "  ".join("-" * w for w in widths)
    out: List[str] = []
    out.append("\n=== Summary ===")
    out.append("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    out.append(sep)
    for r in rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def _maybe_png(series: Dict[str, List[Tuple[int, float]]], out_dir: Path) -> Optional[List[Path]]:
    """If matplotlib is installed, dump per-metric PNGs into out_dir."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[plot] matplotlib not available; skipping --png", file=sys.stderr)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name, pts in series.items():
        if not pts:
            continue
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, marker=".", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs step  (n={len(pts)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="path to auto_eval/index.json")
    ap.add_argument("--png", default=None,
                    help="optional dir to dump matplotlib PNGs (graceful fallback)")
    ap.add_argument("--metrics", default=None,
                    help="csv subset of {ce,ppl,mmlu,hellaswag,lambada,humaneval,mbpp,gsm8k}")
    args = ap.parse_args()
    idx_path = Path(args.index).resolve()
    idx = _load_index(idx_path)
    if not idx:
        print(f"[plot] empty/no index at {idx_path}; nothing to plot")
        return
    series = _series_from(idx)
    if args.metrics:
        wanted = {m.strip() for m in args.metrics.split(",") if m.strip()}
        series = {k: v for k, v in series.items() if k in wanted}
    print(f"[plot] index={idx_path}")
    print(f"[plot] buckets={len(idx)}  metrics={sum(1 for s in series.values() if s)} non-empty")
    for name, pts in series.items():
        print(_render_ascii(name, pts))
    print(_summary_table(series))
    if args.png:
        out = _maybe_png(series, Path(args.png).resolve())
        if out:
            print(f"[plot] wrote {len(out)} PNG(s) to {Path(args.png).resolve()}")


if __name__ == "__main__":
    main()
