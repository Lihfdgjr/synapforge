"""tests/integration/test_bench_torch_compile.py — smoke + JSON-format
asserts for the T2.11 torch.compile bench harness
(``scripts/bench_torch_compile.py``).

These tests run on CPU so CI can verify the harness's correctness
without a GPU. The compile arm is allowed to skip on CPU/Windows
because torch 2.0.x explicitly refuses to compile on Windows; the
script must handle that path gracefully (record ``compile_supported
== False`` + a ``compile_skip_reason`` string).

What we check:
    * test_bench_smoke -- the script runs end-to-end with --steps 2,
      writes a JSON file, and reports a positive no-compile latency.
      The compile arm is allowed to skip (we only require the script
      not to crash).
    * test_bench_json_format -- the JSON record contains the four
      headline keys required by docs/PERF_KNOBS.md (no_compile_tok_s,
      compile_tok_s, speedup_ratio, pct_speedup) plus the meta fields.

We invoke the bench through ``main()`` (in-process) rather than
``subprocess.run`` so the test inherits pytest's tmp dir + the
already-loaded torch module (saves ~5s of import overhead on CI).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _import_bench():
    """Import the bench harness module; importorskip if torch missing."""
    pytest.importorskip("torch")
    if "bench_torch_compile" in sys.modules:
        return importlib.reload(sys.modules["bench_torch_compile"])
    return importlib.import_module("bench_torch_compile")


def _tiny_argv(tmp_out: Path, *, steps: int = 2):
    """Tiny shapes so the test runs in seconds on CPU.

    bs=2 seq=16 vocab=1024 d=64 n_layers=2 keeps the model under
    ~1 MiB of weights (the canonical d=512 / n_layers=10 / vocab=151936
    config builds 100M params and is way too heavy for a CPU-only
    test runner). The shape sanity is identical, just smaller numbers.
    """
    return [
        "--steps", str(steps),
        "--batch-size", "2",
        "--seq-len", "16",
        "--vocab", "1024",
        "--d", "64",
        "--n-layers", "2",
        "--device", "cpu",
        "--out-dir", str(tmp_out),
    ]


def test_bench_smoke(tmp_path: Path) -> None:
    """End-to-end: bench runs, writes a JSON, and the no-compile
    latency is positive (i.e. arm 1 actually executed forward+backward).

    The compile arm is allowed to skip on CPU/Windows -- the script
    must NOT crash when ``torch.compile`` raises (Windows torch 2.0,
    missing triton, etc.). We only assert the JSON is written and the
    no-compile measurement is non-degenerate.
    """
    bench = _import_bench()
    out_dir = tmp_path / "bench"
    rc = bench.main(_tiny_argv(out_dir, steps=2))
    assert rc == 0, f"bench main() returned non-zero rc={rc}"

    # JSON file written
    files = sorted(out_dir.glob("torch_compile_*.json"))
    assert len(files) == 1, f"expected 1 JSON, got {[f.name for f in files]}"

    rec = json.loads(files[0].read_text(encoding="utf-8"))

    # No-compile arm must have produced a positive latency. If this is
    # 0 then arm 1 silently failed and the rest of the bench is
    # meaningless.
    assert rec["no_compile_tok_s"] > 0, (
        f"no_compile_tok_s should be positive, got {rec['no_compile_tok_s']}"
    )
    assert rec["no_compile_step_ms"] > 0, (
        f"no_compile_step_ms should be positive, "
        f"got {rec['no_compile_step_ms']}"
    )

    # Compile arm: either it ran (compile_supported=True + tok_s>0) OR
    # it skipped cleanly (compile_supported=False + a non-empty reason).
    if rec["compile_supported"]:
        assert rec["compile_tok_s"] > 0, (
            "compile_supported=True but compile_tok_s is 0"
        )
        # Speedup is allowed to be negative (compile sometimes hurts on
        # tiny models because the fixed-cost overhead dominates), but
        # the ratio must be a finite positive number.
        assert rec["speedup_ratio"] > 0
    else:
        assert rec["compile_skip_reason"], (
            "compile_supported=False but compile_skip_reason is empty -- "
            "should explain WHY (Windows? missing triton? old torch?)"
        )
        assert rec["compile_tok_s"] == 0
        assert rec["speedup_ratio"] == 0
        assert rec["pct_speedup"] == 0


def test_bench_json_format(tmp_path: Path) -> None:
    """Pin the JSON schema so downstream consumers (rental agents,
    PERF_KNOBS doc updates) can rely on a stable key set.

    Required headline keys: no_compile_tok_s, compile_tok_s,
    speedup_ratio, pct_speedup.
    Required meta keys: device, torch_version, batch_size, seq_len,
    vocab, d, n_layers, steps, compile_mode, compile_supported,
    timestamp.
    """
    bench = _import_bench()
    out_dir = tmp_path / "bench"
    bench.main(_tiny_argv(out_dir, steps=2))
    files = sorted(out_dir.glob("torch_compile_*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text(encoding="utf-8"))

    headline = {"no_compile_tok_s", "compile_tok_s",
                "speedup_ratio", "pct_speedup"}
    missing_headline = headline - rec.keys()
    assert not missing_headline, (
        f"JSON record missing headline keys: {sorted(missing_headline)}; "
        f"got {sorted(rec.keys())}"
    )

    meta = {"device", "torch_version", "batch_size", "seq_len", "vocab",
            "d", "n_layers", "steps", "compile_mode", "compile_supported",
            "timestamp"}
    missing_meta = meta - rec.keys()
    assert not missing_meta, (
        f"JSON record missing meta keys: {sorted(missing_meta)}"
    )

    # Type / range checks. Catching a string leak here is cheaper than
    # discovering it on the rental's first real run.
    assert isinstance(rec["no_compile_tok_s"], (int, float))
    assert isinstance(rec["compile_tok_s"], (int, float))
    assert isinstance(rec["speedup_ratio"], (int, float))
    assert isinstance(rec["pct_speedup"], (int, float))
    assert rec["device"] == "cpu"
    assert rec["batch_size"] == 2
    assert rec["seq_len"] == 16
    assert rec["vocab"] == 1024
    assert rec["d"] == 64
    assert rec["n_layers"] == 2
    assert rec["steps"] == 2
    assert rec["compile_mode"] == "reduce-overhead"
    assert isinstance(rec["compile_supported"], bool)
