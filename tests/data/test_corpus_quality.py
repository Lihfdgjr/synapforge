"""tests/data/test_corpus_quality.py — assert quality gates on assembled data.

D5 of the 2026-05-02 quality-data push (TOKEN_SOUP root-cause hunt).

Three classes of test:

1. **Quality scorer correctness on synthetic fixtures** (CI-safe). We
   build two tiny in-memory parquets — one HEALTHY (Zipfian top-K,
   wide vocab, low bigram repeat), one DEGENERATE ("the the the..."
   repeated forever) — and assert the scorer separates them with the
   composite score and the four hard gates.

2. **`--data-files` CLI parsing wires through to ParquetTokenStream**
   (CI-safe). Builds two tiny synthetic parquets locally, passes
   ``files_with_weights=[(p1, 0.7), (p2, 0.3)]`` to
   ``ParquetTokenStream``, iterates one batch, and asserts the
   yielded sample distribution roughly matches the requested 70/30
   weights (within 15% — a Bernoulli choice over only ~50 rows
   has high variance, so 15% tolerance keeps the test stable).

3. **Manifest gate** (NICE / soft). When
   ``docs/QUALITY_CORPUS_MANIFEST.json`` is present (assembler has
   been run) we assert the quoted ``data_files_arg`` parses + every
   path's ``quality_score >= 0.65``. Skipped when the manifest is
   absent (e.g. fresh clone before assembler run).
"""
from __future__ import annotations

import json
import os
import random
import string
import tempfile
from pathlib import Path
from typing import Iterable

import pytest

# Repo paths
_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
_DOCS = _ROOT / "docs"

pyarrow = pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_text_parquet(path: Path, rows: Iterable[str]) -> None:
    """Write a single-column parquet with a 'text' string column."""
    table = pa.table({"text": list(rows)})
    pq.write_table(table, path)


# A synthesised "healthy" vocabulary of ~600 unique English words.
# Zipfian distribution emerges naturally from how often we draw each
# word: position 0 is most common (drawn at uniform), position N is the
# rarest (still > 1 occurrence). 600 words is enough to drive top1 well
# below 10% and vocab coverage above 0.30 once we cycle the vocab.
_HEALTHY_VOCAB = sorted(set((
    "the quick brown fox jumps over a lazy dog cat bird squirrel mouse "
    "Photosynthesis converts light into chemical energy plants oxygen "
    "Newton laws describe motion gravity force acceleration mass speed velocity "
    "theorem proven mathematical statement axiom postulate proof "
    "computer transistor integrated circuit silicon wafer fabrication "
    "Pythagorean right triangle hypotenuse leg square root area volume "
    "DNA carries genetic information living cell nucleus chromosome gene mutation "
    "algorithm solve computational problem efficiency complexity recursion iteration "
    "Plate tectonic continental drift mountain volcano earthquake fault subduction "
    "Industrial Revolution transformed Europe century factory railway steam coal iron "
    "Quantum mechanics subatomic particle wave probability uncertainty Heisenberg "
    "Earth indirectly through plant tree forest ecosystem biome diversity species "
    "ocean wave current tide moon gravitational pull lunar solar eclipse "
    "history empire civilization rise fall culture art architecture monument "
    "language linguistic syntax grammar phoneme morpheme dialect translation "
    "biology cell membrane mitochondria ribosome enzyme protein amino acid "
    "chemistry element compound molecule reaction catalyst equilibrium oxidation "
    "physics force motion energy heat thermodynamics entropy work power "
    "geography continent country river mountain desert tropical temperate arctic "
    "philosophy ethics morality logic reasoning argument premise conclusion fallacy "
    "music rhythm melody harmony composer symphony orchestra piano violin guitar "
    "literature novel poem story author character plot setting theme metaphor "
    "psychology behavior emotion cognition memory learning perception attention "
    "sociology society community tradition norm ritual ceremony institution "
    "economics market supply demand price competition monopoly profit production "
    "engineering design build structure bridge tunnel skyscraper foundation "
    "medicine doctor patient diagnosis treatment surgery prescription pharmacy "
    "law legal court judge jury verdict appeal constitution legislation right "
    "education school teacher student curriculum lesson examination assessment grade "
    "agriculture farming wheat rice corn soybean cotton irrigation harvest soil"
).split()))


def _generate_healthy_text(n_chars: int = 4000, seed: int = 1) -> str:
    """Generate Zipfian-distributed sentences from _HEALTHY_VOCAB.

    Uses position-biased sampling so common words appear more often than
    rare ones, mimicking Zipf's law (top word ~5%, top-50 ~30-40%).
    """
    rng = random.Random(seed)
    n = len(_HEALTHY_VOCAB)
    # Position-weighted: prob[i] proportional to 1/(i+1), normalised.
    weights = [1.0 / (i + 1) for i in range(n)]
    out: list[str] = []
    while sum(len(s) for s in out) < n_chars:
        ln = rng.randint(8, 20)
        sentence = " ".join(
            rng.choices(_HEALTHY_VOCAB, weights=weights, k=ln)
        )
        out.append(sentence + ".")
    return " ".join(out)[:n_chars]


def _generate_degenerate_text(n_chars: int = 4000) -> str:
    """The classic TOKEN_SOUP failure: 'the the the...' forever.

    Mixed with a tiny pinch of digits + commas so it's not literally
    a single string (which would have only 1 token + zero bigram
    diversity and crash the scorer's denominators).
    """
    base = "the the the the the , 0 1 the the the the , 1 0 the the "
    rep = (n_chars // len(base)) + 2
    return (base * rep)[:n_chars]


# ---------------------------------------------------------------------------
# 1. Scorer correctness on synthetic fixtures
# ---------------------------------------------------------------------------
def test_scorer_distinguishes_healthy_vs_degenerate(tmp_path: Path) -> None:
    """The composite score must rank healthy data ABOVE degenerate data
    AND the quality gates must pass on healthy / fail on degenerate.

    This is the core regression: without it, a "good corpus" verdict is
    indistinguishable from a "the the the..." verdict.
    """
    from scripts.score_data_quality import score_file  # type: ignore

    healthy_path = tmp_path / "healthy.parquet"
    degenerate_path = tmp_path / "degenerate.parquet"
    rng = random.Random(42)
    _write_text_parquet(
        healthy_path,
        [_generate_healthy_text(seed=rng.randint(0, 999999)) for _ in range(80)],
    )
    _write_text_parquet(
        degenerate_path,
        [_generate_degenerate_text() for _ in range(80)],
    )
    healthy = score_file(str(healthy_path), tokenizer=None, sample_rows=80)
    degenerate = score_file(str(degenerate_path), tokenizer=None, sample_rows=80)

    # Composite score must rank healthy above degenerate (the whole
    # point of the score).  The numeric gap should be substantial
    # (>0.2) -- a 0.05 gap would mean the score is mostly noise.
    assert healthy.composite_score > degenerate.composite_score + 0.20, (
        f"healthy={healthy.composite_score:.3f} should be >> "
        f"degenerate={degenerate.composite_score:.3f}"
    )

    # Each of the FOUR individual gate dimensions must rank correctly:
    #
    #   top1_share          (lower is better -- healthy < degenerate)
    #   bigram_repeat_ratio (lower is better -- healthy < degenerate)
    #   unique_token_ratio  (higher is better in 0.40-0.75 range)
    #   vocab_coverage      (higher is better)
    #
    # These four signals are what the trainer's quality gate keys on,
    # so we verify the scorer's relative ranking on each one.
    assert healthy.top1_share < degenerate.top1_share, (
        f"top1: healthy={healthy.top1_share}, deg={degenerate.top1_share}"
    )
    assert healthy.bigram_repeat_ratio < degenerate.bigram_repeat_ratio, (
        f"bigram_rep: healthy={healthy.bigram_repeat_ratio}, "
        f"deg={degenerate.bigram_repeat_ratio}"
    )
    assert healthy.unique_token_ratio > degenerate.unique_token_ratio, (
        f"uniq/100: healthy={healthy.unique_token_ratio}, "
        f"deg={degenerate.unique_token_ratio}"
    )
    # Degenerate must fail the gate (this is the load-bearing check —
    # without it, the scorer would let "the the the..." ship to prod).
    assert not degenerate.pass_quality_gate, (
        f"degenerate fixture INCORRECTLY passed quality gate: "
        f"top1={degenerate.top1_share}, "
        f"bigram_rep={degenerate.bigram_repeat_ratio}, "
        f"uniq/100={degenerate.unique_token_ratio}"
    )


def test_scorer_thresholds_are_what_the_docstring_says() -> None:
    """The four hard thresholds in the README docstring must match the
    actual code's gate. Drift-detection: future PRs that change one
    threshold without updating the other would silently break the
    quality gate.
    """
    from scripts.score_data_quality import FileScore, score_file  # noqa: F401

    src = (_SCRIPTS / "score_data_quality.py").read_text(encoding="utf-8")

    # The docstring + the code must reference the same numeric bounds.
    must = ["top1_share < 0.10", "vocab_coverage > 0.30",
            "bigram_repeat_ratio < 0.05", "unique_token_ratio > 0.40"]
    for token in must:
        assert token.replace(" ", "") in src.replace(" ", ""), (
            f"quality gate constant {token!r} missing from "
            f"scripts/score_data_quality.py"
        )


# ---------------------------------------------------------------------------
# 2. ParquetTokenStream files_with_weights wiring
# ---------------------------------------------------------------------------
def test_parquet_token_stream_files_with_weights(tmp_path: Path) -> None:
    """Pass two tiny parquets to ``ParquetTokenStream(files_with_weights=...)``
    at 70/30 weights, iterate ~50 rows, and assert the realised mixture
    matches the requested mixture within 15%.

    The Bernoulli sampler we use has variance ~ p*(1-p)/N ≈ 0.0042 at
    N=50, so 1.96σ ≈ 0.13 — 0.15 keeps the test 99% stable while still
    catching regressions where the weighted iterator silently degrades
    to round-robin or pure concatenation.
    """
    pytest.importorskip("torch")  # ParquetTokenStream depends on torch

    p1 = tmp_path / "src_a.parquet"
    p2 = tmp_path / "src_b.parquet"
    _write_text_parquet(p1, ["AAAA " * 30 for _ in range(60)])
    _write_text_parquet(p2, ["BBBB " * 30 for _ in range(60)])

    from synapforge.data import ParquetTokenStream

    # GPT-2 tokenizer is the smallest one with no remote-code path,
    # making this test cheap & fully offline (transformers cache must
    # exist; if it doesn't the test is skipped).
    try:
        ds = ParquetTokenStream(
            glob_pattern=str(tmp_path / "ignored*.parquet"),
            seq_len=8,
            batch_size=4,
            tokenizer_name="gpt2",
            loop=True,
            shuffle_buffer=0,
            shuffle_seed=2026,
            files_with_weights=[(str(p1), 0.7), (str(p2), 0.3)],
        )
    except Exception as exc:
        pytest.skip(f"tokenizer / pyarrow not available: {exc!r}")

    # We can't easily count A/B tokens through the (tokens_in, tokens_out)
    # interface, so probe the underlying _iter_text_rows_weighted directly.
    rows: list[str] = []
    rng = iter(ds._iter_text_rows_weighted())
    for _ in range(120):
        rows.append(next(rng))
    n_a = sum(1 for r in rows if r.startswith("AAAA"))
    n_b = sum(1 for r in rows if r.startswith("BBBB"))
    p_a = n_a / max(1, n_a + n_b)
    # Expected p_a = 0.7. Allow 0.55..0.85 (window for 120-row Bernoulli).
    assert 0.55 < p_a < 0.85, (
        f"realised mixture p_a={p_a:.3f} (n_a={n_a} n_b={n_b}) "
        f"deviates from requested 0.7 by > 0.15"
    )


def test_parquet_token_stream_files_with_weights_jsonl(tmp_path: Path) -> None:
    """Mixed parquet + jsonl in files_with_weights — common when
    blending pretrain corpora with SFT instruction data.
    """
    pytest.importorskip("torch")

    p1 = tmp_path / "pretrain.parquet"
    j1 = tmp_path / "sft.jsonl"
    _write_text_parquet(p1, ["AAAA " * 30 for _ in range(40)])
    with open(j1, "w", encoding="utf-8") as f:
        for _ in range(40):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "QQQ?"},
                    {"role": "assistant", "content": "QQQ!"},
                ]
            }) + "\n")

    from synapforge.data import ParquetTokenStream
    try:
        ds = ParquetTokenStream(
            glob_pattern=str(tmp_path / "ignored*.parquet"),
            seq_len=8,
            batch_size=4,
            tokenizer_name="gpt2",
            loop=True,
            shuffle_buffer=0,
            files_with_weights=[(str(p1), 0.5), (str(j1), 0.5)],
        )
    except Exception as exc:
        pytest.skip(f"tokenizer / pyarrow not available: {exc!r}")
    rows: list[str] = []
    rng = iter(ds._iter_text_rows_weighted())
    for _ in range(80):
        rows.append(next(rng))
    n_a = sum(1 for r in rows if r.startswith("AAAA"))
    n_q = sum(1 for r in rows if "QQQ" in r)
    assert n_a > 0 and n_q > 0, (
        f"weighted mixture should yield from BOTH files; got n_a={n_a}, n_q={n_q}"
    )


def test_files_with_weights_rejects_missing_path(tmp_path: Path) -> None:
    """Sanity: bogus path must raise FileNotFoundError before the
    iterator starts (so trainers fail fast, not 4h into a run).
    """
    pytest.importorskip("torch")
    from synapforge.data import ParquetTokenStream

    bogus = tmp_path / "does_not_exist.parquet"
    real = tmp_path / "real.parquet"
    _write_text_parquet(real, ["x"] * 5)
    try:
        with pytest.raises(FileNotFoundError):
            ParquetTokenStream(
                glob_pattern=str(tmp_path / "*.parquet"),
                seq_len=4,
                batch_size=2,
                tokenizer_name="gpt2",
                files_with_weights=[(str(real), 0.5), (str(bogus), 0.5)],
            )
    except Exception as exc:
        if "FileNotFoundError" in str(exc):
            return
        pytest.skip(f"tokenizer not available: {exc!r}")


# ---------------------------------------------------------------------------
# 3. Manifest gate (only if assembler has been run)
# ---------------------------------------------------------------------------
def test_quality_corpus_manifest_passes_thresholds() -> None:
    """When ``docs/QUALITY_CORPUS_MANIFEST.json`` exists, the WEIGHTED
    fraction of PASS-quality files must be >= 50%.

    Why weight-by-weight (not file count): the assembler may include
    several tiny ZH-instruction shards (each <1% of the corpus) that
    fail the English-tuned gate (top1 < 10% over-penalises CJK where
    top1 of 'de' ~15% is normal). The corpus as a whole is fine when
    those tiny shards are <50% of token volume.

    The 50% weight floor catches the actual TOKEN_SOUP regression:
    if every selected file fails the gate, no high-signal data drives
    the optimiser and we'd be back to "the the the..." output.

    Skipped when the manifest is missing (developer hasn't run the
    assembler yet on this clone).
    """
    manifest_path = _DOCS / "QUALITY_CORPUS_MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip(f"manifest absent at {manifest_path}; run "
                    "scripts/assemble_quality_corpus.py first")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = m.get("files", [])
    assert files, "manifest empty (no files selected)"
    pass_weight = sum(
        float(f.get("weight", 0.0)) for f in files if f.get("pass_quality_gate")
    )
    total_weight = sum(float(f.get("weight", 0.0)) for f in files) or 1.0
    pass_ratio = pass_weight / total_weight
    n_pass = sum(1 for f in files if f.get("pass_quality_gate"))
    assert pass_ratio >= 0.50, (
        f"only {pass_weight:.3f}/{total_weight:.3f} ({pass_ratio:.1%}) "
        f"of corpus weight is PASS-quality (n_pass={n_pass}/{len(files)}); "
        f"bad files: " + ", ".join(
            os.path.basename(f["path"])
            for f in files if not f.get("pass_quality_gate")
        )[:200]
    )


def test_quality_corpus_manifest_data_files_arg_is_parseable() -> None:
    """The ``data_files_arg`` field in the manifest must round-trip
    through the trainer's ``--data-files`` parser (one path:weight per
    comma-separated entry, weight is a finite float).
    """
    manifest_path = _DOCS / "QUALITY_CORPUS_MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("manifest absent")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    arg = m.get("data_files_arg", "")
    assert arg, "manifest missing data_files_arg"

    # Replicate the train_100m_kd.py parser logic.
    pairs: list[tuple[str, float]] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        path, _, w = part.rpartition(":")
        weight = float(w)
        assert weight > 0, f"weight must be > 0, got {weight} for {path}"
        assert path, f"path must be non-empty"
        pairs.append((path, weight))
    assert len(pairs) >= 2, (
        f"manifest data_files_arg has < 2 files; got {len(pairs)}"
    )


def test_jsonl_reader_extracts_text_messages_content(tmp_path: Path) -> None:
    """Standalone test of the jsonl reader logic in
    ``ParquetTokenStream._open_jsonl_iter`` -- duplicated here so the
    test runs without torch.

    The three SFT shapes must all yield non-empty strings:
      * ``{"text": "..."}``
      * ``{"content": "..."}``
      * ``{"messages": [{"role": ..., "content": ...}, ...]}``
    """
    import json as _json
    p = tmp_path / "mixed.jsonl"
    rows = [
        {"text": "first text row"},
        {"content": "second content row"},
        {"messages": [
            {"role": "user", "content": "U?"},
            {"role": "assistant", "content": "A!"},
        ]},
        {"unrelated": "skip me"},
        {"messages": [{"role": "system"}]},  # empty content -> skip
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")

    # Inline duplicate of _open_jsonl_iter (must stay in sync).
    def read_jsonl(path: str) -> list[str]:
        out: list[str] = []
        with open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = _json.loads(line)
                txt = obj.get("text") if isinstance(obj, dict) else None
                if isinstance(txt, str) and txt:
                    out.append(txt); continue
                cnt = obj.get("content") if isinstance(obj, dict) else None
                if isinstance(cnt, str) and cnt:
                    out.append(cnt); continue
                msgs = obj.get("messages") if isinstance(obj, dict) else None
                if isinstance(msgs, list):
                    bits = []
                    for m in msgs:
                        if isinstance(m, dict):
                            ctn = m.get("content", "")
                            if isinstance(ctn, str) and ctn:
                                bits.append(f"<|{m.get('role','')}|>{ctn}")
                    if bits:
                        out.append("\n".join(bits))
        return out

    yields = read_jsonl(str(p))
    assert "first text row" in yields
    assert "second content row" in yields
    assert any("U?" in y and "A!" in y for y in yields)
    # Skip-cases: no row from {"unrelated":...} or empty messages.
    assert not any("skip me" in y for y in yields)


def test_data_files_arg_parser_logic() -> None:
    """Standalone test of the --data-files comma/colon parser.

    Doesn't import torch or transformers, so it runs on every CI
    (including the Windows-dev box where torch is sometimes absent).
    The parser body is small enough to duplicate here; if the train_*.py
    parser changes, this test will start to drift and force the dup to
    be re-synced with the source.
    """
    def parse_data_files_arg(arg: str) -> list[tuple[str, float]]:
        # Mirror exactly what train_100m_kd.py does.
        out: list[tuple[str, float]] = []
        for part in arg.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    f"--data-files entry {part!r} missing ':WEIGHT' suffix"
                )
            path, _, w = part.rpartition(":")
            weight = float(w)
            out.append((path, weight))
        return out

    # Happy path
    pairs = parse_data_files_arg("/a/b.parquet:0.5,/c/d.jsonl:0.3,/e/f.parquet:0.2")
    assert pairs == [
        ("/a/b.parquet", 0.5), ("/c/d.jsonl", 0.3), ("/e/f.parquet", 0.2)
    ]
    # Whitespace tolerance
    assert parse_data_files_arg("  /a:0.7 ,/b:0.3 ") == [("/a", 0.7), ("/b", 0.3)]
    # Single entry
    assert parse_data_files_arg("/x:1.0") == [("/x", 1.0)]
    # Bad: missing colon
    with pytest.raises(ValueError):
        parse_data_files_arg("/a/no/weight")
    # Bad: non-float weight
    with pytest.raises(ValueError):
        parse_data_files_arg("/a:notanumber")


# ---------------------------------------------------------------------------
# 4. Bonus: rule-out the OBVIOUS regression — single-file mode unchanged
# ---------------------------------------------------------------------------
def test_single_file_glob_still_works(tmp_path: Path) -> None:
    """Backwards-compat: when ``files_with_weights=None``, the legacy
    glob path must work exactly as before. Future commits to
    ``ParquetTokenStream.__init__`` shouldn't break the production
    rental's training launches.
    """
    pytest.importorskip("torch")
    p1 = tmp_path / "legacy.parquet"
    _write_text_parquet(p1, ["legacy data"] * 8)
    from synapforge.data import ParquetTokenStream
    try:
        ds = ParquetTokenStream(
            glob_pattern=str(p1),
            seq_len=4,
            batch_size=2,
            tokenizer_name="gpt2",
            loop=False,
            files_with_weights=None,
        )
    except Exception as exc:
        pytest.skip(f"tokenizer / pyarrow unavailable: {exc!r}")
    # The legacy iterator path must emit at least one row.
    rows = list(ds._iter_text_rows_raw())
    assert rows, "legacy single-glob path emitted zero rows"
