r"""Smoke tests for ``scripts/fill_paper_section4.py`` (T6.2).

Covers:
  - test_smoke_with_mock_log: end-to-end run on a synthetic log + bench
    JSON -- verifies ``section4_auto.tex`` is written, contains both
    expected tables, and the draft skeleton lands at ``paper/draft.tex``
    when absent.
  - test_table_format_booktabs: emitted LaTeX uses booktabs rules
    (``\toprule``/``\midrule``/``\bottomrule``) for both Table 4.1 and 4.2.
  - test_tbd_placeholder_when_metric_missing: when the log doesn't
    contain a target step (or the bench JSON is missing), the cells
    fall back to ``\textbf{TBD}`` rather than silently fabricating.

Pure stdlib / regex parsing -- no matplotlib, no torch, no GPU.
Tests run on CPU in <2s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# scripts/ is added to sys.path by tests/integration/conftest.py.
import fill_paper_section4 as fps  # noqa: E402


# ---------------------------------------------------------------------------
# Mock log + bench fixture builders
# ---------------------------------------------------------------------------
def _mock_train_log(tmp_path: Path,
                    n_steps: int = 60,
                    start_step: int = 1000,
                    step_stride: int = 200,
                    val_every: int = 5,
                    include_target: bool = True) -> Path:
    """Synthesize a ``train_run3z.log`` covering steps 1000..1000+n*200.

    With defaults: 60 events covering [1000, 12800] -- step 1k / 5k /
    10k all hit. ``include_target=False`` truncates before step 5k so
    the missing-data test can exercise the TBD fallback.
    """
    lines: list[str] = []
    last = n_steps if include_target else min(n_steps, 8)
    for i in range(last):
        step = start_step + i * step_stride
        ce = round(9.5 - (i / max(1, n_steps)) * 4.0, 4)
        kd = round(0.05 + (i / max(1, n_steps)) * 0.5, 4)
        z = round(94.0 + (i % 7) * 0.1, 3)
        lr = "3.0e-04"
        step_ms = round(50.0 + (i % 5), 2)
        tok_s = round(8000 + (i % 11) * 50, 1)
        mem_gb = round(60.0 + (i % 3) * 0.5, 2)
        lines.append(
            f"[01:35:{i % 60:02d}] step {step} loss={ce + 0.01:.4f} "
            f"ce={ce} kd={kd:.4f} z={z} lr={lr} "
            f"step_ms={step_ms} tok/s={tok_s} mem_GB={mem_gb}"
        )
        # Sprinkle a VAL line every val_every step events.
        if i % val_every == 0 and i > 0:
            ttt = round(500.0 - (i / max(1, n_steps)) * 350.0, 2)
            ho = round(ttt * 1.05, 2)  # holdout slightly worse
            lines.append(f"[01:35:{i % 60:02d}] VAL step {step}: "
                         f"val_ppl_ttt={ttt} val_ppl_holdout={ho} (honest)")

    log = tmp_path / "train_run3z.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


def _mock_bench_json(tmp_path: Path, *, supported: bool = True) -> Path:
    if supported:
        rec = {
            "device": "cuda",
            "torch_version": "2.4.0",
            "batch_size": 8,
            "seq_len": 256,
            "vocab": 151936,
            "d": 512,
            "n_layers": 10,
            "steps": 100,
            "compile_mode": "reduce-overhead",
            "no_compile_tok_s": 12345.6,
            "no_compile_step_ms": 165.7,
            "compile_tok_s": 13987.3,
            "compile_step_ms": 146.3,
            "speedup_ratio": 1.133,
            "pct_speedup": 13.3,
            "compile_supported": True,
            "compile_skip_reason": None,
            "timestamp": "2026-05-02T03:00:00Z",
        }
    else:
        rec = {
            "device": "cpu",
            "torch_version": "2.4.0",
            "batch_size": 2,
            "seq_len": 16,
            "vocab": 1024,
            "d": 64,
            "n_layers": 2,
            "steps": 2,
            "compile_mode": "reduce-overhead",
            "no_compile_tok_s": 100.0,
            "no_compile_step_ms": 5.0,
            "compile_tok_s": 0.0,
            "compile_step_ms": 0.0,
            "speedup_ratio": 0.0,
            "pct_speedup": 0.0,
            "compile_supported": False,
            "compile_skip_reason": "Windows torch.compile not supported",
            "timestamp": "2026-05-02T03:00:00Z",
        }
    bj = tmp_path / "torch_compile_030000.json"
    bj.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return bj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_smoke_with_mock_log(tmp_path: Path) -> None:
    """End-to-end: run the script against a synthetic log + bench JSON.

    Asserts:
      * exit code 0
      * ``section4_auto.tex`` written and non-empty
      * file contains \\section{Results}, both Table 4.1 and 4.2 captions,
        and the figure 4.1 caption with the run name resolved
      * ``paper/draft.tex`` is created with an \\input{section4_auto}
        line when it doesn't already exist
    """
    log = _mock_train_log(tmp_path)
    bench = _mock_bench_json(tmp_path)
    out = tmp_path / "paper" / "section4_auto.tex"
    draft = tmp_path / "paper" / "draft.tex"

    rc = fps.main([
        "--log", str(log),
        "--bench-json", str(bench),
        "--ckpt-step", "10000",
        "--out", str(out),
        "--draft", str(draft),
    ])
    assert rc == 0, f"main() returned non-zero rc={rc}"

    assert out.exists(), f"expected {out} to exist"
    text = out.read_text(encoding="utf-8")
    assert len(text) > 200, f"section4 output suspiciously short: {len(text)}B"

    # Section header + table captions present.
    assert r"\section{Results}" in text
    assert "Train CE and validation perplexity" in text
    assert "Throughput, peak GPU memory" in text
    # Run-name resolved ('train_run3z.log' -> 'Run 3z').
    assert "Train/val curves for Run 3z." in text

    # Draft skeleton was created (was absent before this run).
    assert draft.exists()
    draft_text = draft.read_text(encoding="utf-8")
    assert r"\input{section4_auto}" in draft_text
    assert r"\documentclass" in draft_text


def test_table_format_booktabs(tmp_path: Path) -> None:
    """Both tables MUST use booktabs rules (\\toprule/\\midrule/\\bottomrule).

    We render via the public ``render_section4`` helper directly (no
    file I/O) so the assert is on the LaTeX string itself.
    """
    log = _mock_train_log(tmp_path)
    bench = _mock_bench_json(tmp_path)
    parsed = fps.ptc.parse_log_file(log)
    bench_rec = json.loads(bench.read_text(encoding="utf-8"))
    data = fps.aggregate(parsed, bench_rec, ckpt_step=10000)
    text = fps.render_section4(data)

    # Booktabs rules appear at least once each in the file.
    assert text.count(r"\toprule") >= 2, "expected \\toprule in both tables"
    assert text.count(r"\midrule") >= 2, "expected \\midrule in both tables"
    assert text.count(r"\bottomrule") >= 2, (
        "expected \\bottomrule in both tables"
    )
    # And the booktabs-mandatory environment is `tabular`, not the
    # legacy `array`/`tabularx`.
    assert text.count(r"\begin{tabular}") >= 2
    assert text.count(r"\end{tabular}") >= 2

    # Table labels for cross-references.
    assert r"\label{tab:train_progression}" in text
    assert r"\label{tab:perf}" in text
    assert r"\label{fig:train_val_curves}" in text


def test_tbd_placeholder_when_metric_missing(tmp_path: Path) -> None:
    """Missing metrics -> ``\\textbf{TBD}``, never silent zeros.

    Two ways we trigger TBD:
      1. ``--ckpt-step 10000`` against a log truncated at step ~2400
         -> the snapshot row's CE / val PPL cells are TBD.
      2. No bench JSON passed -> torch.compile speedup cell is TBD.
    """
    # Truncated log: only ~8 step events, last step ~2400 (well below
    # 5k and 10k so the 5k/10k rows fall back to TBD).
    log = _mock_train_log(tmp_path, include_target=False)
    out = tmp_path / "paper" / "section4_auto.tex"
    draft = tmp_path / "paper" / "draft.tex"

    # No --bench-json on this run -- compile speedup cell should be TBD.
    rc = fps.main([
        "--log", str(log),
        "--ckpt-step", "10000",
        "--out", str(out),
        "--draft", str(draft),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")

    # Some TBDs MUST appear; otherwise we silently fabricated numbers.
    n_tbd = text.count(r"\textbf{TBD}")
    assert n_tbd >= 2, (
        f"expected at least 2 TBD placeholders (5k row, 10k row, "
        f"and/or compile speedup), got {n_tbd} in:\n{text}"
    )

    # In particular the compile speedup cell must read TBD when there's
    # no bench JSON.
    # (We grep for the literal row header string.)
    assert r"\texttt{torch.compile}" in text
    # The "+13.3% on cuda" string from a populated bench would only
    # appear when bench JSON was provided -- it must NOT be here.
    assert "+13.3" not in text


def test_compile_skip_renders_explicit_label(tmp_path: Path) -> None:
    """When compile_supported=False the perf table must say so
    explicitly (``skipped (CPU/Windows)``), not show a 0%.

    This guards against the silent-zero failure mode where compile
    gets reported as "+0.00% speedup" instead of "we couldn't measure
    it on this device."
    """
    log = _mock_train_log(tmp_path)
    bench = _mock_bench_json(tmp_path, supported=False)
    out = tmp_path / "paper" / "section4_auto.tex"
    draft = tmp_path / "paper" / "draft.tex"

    rc = fps.main([
        "--log", str(log),
        "--bench-json", str(bench),
        "--ckpt-step", "10000",
        "--out", str(out),
        "--draft", str(draft),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "skipped (CPU/Windows)" in text
    # And we did not pretend 0% was a real measurement.
    assert "+0.00" not in text


def test_aggregate_picks_best_val(tmp_path: Path) -> None:
    """The 'best' row in Table 4.1 is the val event with the lowest
    val_ppl_holdout, even if it's not the most recent."""
    log = _mock_train_log(tmp_path, n_steps=40)
    parsed = fps.ptc.parse_log_file(log)
    data = fps.aggregate(parsed, bench=None, ckpt_step=5000)
    # The "best" row is the one whose label string ends with " (best)".
    best = [r for r in data.rows if str(r["label"]).endswith("(best)")]
    assert len(best) == 1, f"expected exactly 1 best row, got {best}"
    best_row = best[0]
    # Compare to manual best.
    val_holdouts = [e["ppl_holdout"] for e in parsed.val_events]
    assert val_holdouts, "fixture should emit at least one VAL event"
    assert abs(best_row["val_ppl_holdout"] - min(val_holdouts)) < 1e-6
