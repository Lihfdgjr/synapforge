"""Statistical tests for ``synapforge.native.data.WeightedMixedStream``.

Goals
-----
1. **Weight ratios respected over 10K samples.**

   With sources at weights ``(0.7, 0.3)`` and tokenizers/parquets that
   emit a recognisably distinct first-token (``"alpha"`` vs ``"beta"``),
   the realised proportion of "alpha" tokens vs "beta" tokens in the
   first 10K row-level chunks must lie within +/- 0.03 of the requested
   weights. (Sampling noise on N=10K, p=0.7 is sigma=0.0046, so a
   3-sigma window is 0.014; 0.03 leaves headroom for tokenizer
   id-aliasing skew.)

2. **All sources contribute** -- no source's pick-rate is zero (the
   ``random.choices`` weighting must hit every weight > 0 source).

3. **``parse_data_files_arg`` round-trip** matches what
   ``train_100m_kd.py`` builds via ``rsplit(":", 1)``.

4. **Mixed parquet+jsonl** -- a 50/50 mix with one parquet and one
   jsonl produces shape-correct batches.

5. **Pre-built sources mode** -- caller supplies pre-built streams,
   weights at the constructor.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# package shim -- see test_native_parquet_stream.py for explanation
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import types

if "synapforge" not in sys.modules:
    sf = types.ModuleType("synapforge")
    sf.__path__ = [str(_REPO_ROOT / "synapforge")]
    sys.modules["synapforge"] = sf
if "synapforge.native" not in sys.modules:
    sn = types.ModuleType("synapforge.native")
    sn.__path__ = [str(_REPO_ROOT / "synapforge" / "native")]
    sys.modules["synapforge.native"] = sn
if "synapforge.native.data" not in sys.modules:
    snd = types.ModuleType("synapforge.native.data")
    snd.__path__ = [str(_REPO_ROOT / "synapforge" / "native" / "data")]
    sys.modules["synapforge.native.data"] = snd

from synapforge.native.data.jsonl_stream import NativeJsonlStream  # noqa: E402
from synapforge.native.data.mixed_stream import (  # noqa: E402
    WeightedMixedStream,
    parse_data_files_arg,
)
from synapforge.native.data.parquet_stream import NativeParquetStream  # noqa: E402
from synapforge.native.data.tokenizer import NativeTokenizer  # noqa: E402

_PRUNER_TOK_DIR = "C:/Users/26979/.synapide/pruner-model"


def _have_qwen() -> bool:
    return os.path.isfile(os.path.join(_PRUNER_TOK_DIR, "tokenizer.json"))


@pytest.fixture(scope="module")
def tokenizer() -> NativeTokenizer:
    if not _have_qwen():
        pytest.skip("Qwen tokenizer.json not found")
    return NativeTokenizer(_PRUNER_TOK_DIR, backend="json")


# ---------------------------------------------------------------------------
# parse_data_files_arg
# ---------------------------------------------------------------------------


def test_parse_data_files_arg_basic(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.jsonl"
    a.write_text("")
    b.write_text("")
    s = f"{a}:0.7,{b}:0.3"
    parsed = parse_data_files_arg(s)
    assert parsed == [(str(a), 0.7), (str(b), 0.3)]


def test_parse_data_files_arg_missing_path():
    with pytest.raises(FileNotFoundError):
        parse_data_files_arg("nope.parquet:1.0")


def test_parse_data_files_arg_missing_colon(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_text("")
    with pytest.raises(ValueError):
        parse_data_files_arg(str(a))


def test_parse_data_files_arg_negative_weight(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_text("")
    with pytest.raises(ValueError):
        parse_data_files_arg(f"{a}:-1.0")


# ---------------------------------------------------------------------------
# weight ratios
# ---------------------------------------------------------------------------


def test_weight_ratio_respected_over_10k(tokenizer, tmp_path):
    """Sample 10K row-level chunks; assert per-source frequency within +/- 3%.

    We construct two parquets each containing a unique single-word text
    (``"alpha"`` and ``"beta"``). Encoding either one produces a
    recognisable ID. After 10K chunks, the ratio of "alpha" tokens
    among the first token of every chunk should match the requested
    weight (the per-row reservoir is pure-content per file, so the
    first-token of every chunk reflects which source produced the
    chunk).
    """
    p_alpha = tmp_path / "alpha.parquet"
    p_beta = tmp_path / "beta.parquet"
    # Each row is one short text; the streamer concatenates with eos
    # and slides a length seq_len+1 window. Pick text long enough
    # that one row produces multiple chunks (~50 chars after BPE = ~10
    # tokens; with seq_len=8 that's 1-2 chunks per row).
    pq.write_table(pa.table({"text": ["alpha"] * 5000}), str(p_alpha))
    pq.write_table(pa.table({"text": ["beta"] * 5000}), str(p_beta))

    weight_alpha = 0.7
    weight_beta = 0.3
    mix = WeightedMixedStream.from_paths_with_weights(
        [(str(p_alpha), weight_alpha), (str(p_beta), weight_beta)],
        seq_len=4, batch_size=1, tokenizer=tokenizer,
    )
    # Pull 10K row-level chunks via the internal hook.
    chunks = list(itertools.islice(mix._iter_chunks(), 10_000))
    assert len(chunks) == 10_000

    # ID of "alpha" / "beta" -- look up via tokenizer encode.
    alpha_id = tokenizer.encode("alpha")[0]
    beta_id = tokenizer.encode("beta")[0]
    assert alpha_id != beta_id

    # Count: a chunk is "alpha-source" if its first token == alpha_id
    # (the streamer concatenates row-tokens + eos before windowing, so
    # the very first token of every row's first chunk is whatever the
    # row text begins with).
    n_alpha = sum(1 for c in chunks if c[0] == alpha_id)
    n_beta = sum(1 for c in chunks if c[0] == beta_id)
    # eos chunks (where the window straddles a doc boundary) are also
    # possible; ignore them in the ratio computation.
    n_total = n_alpha + n_beta
    assert n_total > 5_000, f"too few classifiable chunks: {n_total}"
    realised_alpha = n_alpha / n_total
    realised_beta = n_beta / n_total
    # 3% tolerance is comfortable on N=5K-10K classifiable chunks.
    assert abs(realised_alpha - weight_alpha) < 0.03, (
        f"weighted-alpha drift: requested {weight_alpha}, got "
        f"{realised_alpha:.3f}"
    )
    assert abs(realised_beta - weight_beta) < 0.03, (
        f"weighted-beta drift: requested {weight_beta}, got "
        f"{realised_beta:.3f}"
    )


# ---------------------------------------------------------------------------
# heterogeneous mixing
# ---------------------------------------------------------------------------


def test_mixed_parquet_jsonl(tokenizer, tmp_path):
    """A parquet + a jsonl mix produces shape-correct batches."""
    p = tmp_path / "data.parquet"
    j = tmp_path / "data.jsonl"
    pq.write_table(pa.table({"text": ["one parquet line"] * 200}), str(p))
    with open(j, "w", encoding="utf-8") as f:
        for _ in range(200):
            f.write(json.dumps({"text": "one jsonl line"}) + "\n")
    mix = WeightedMixedStream.from_paths_with_weights(
        [(str(p), 0.5), (str(j), 0.5)],
        seq_len=8, batch_size=4, tokenizer=tokenizer,
    )
    batches = list(itertools.islice(iter(mix), 5))
    assert len(batches) == 5
    for x, y in batches:
        assert x.shape == (4, 8)
        assert np.array_equal(x[:, 1:], y[:, :-1])


def test_prebuilt_sources_mode(tokenizer, tmp_path):
    """Mode B: caller passes pre-built ``NativeParquetStream`` instances."""
    p = tmp_path / "a.parquet"
    pq.write_table(pa.table({"text": ["short text " * 5] * 200}), str(p))
    ds = NativeParquetStream(
        str(p), seq_len=8, batch_size=1, tokenizer=tokenizer, loop=True,
    )
    mix = WeightedMixedStream(
        [(ds, 1.0)], seq_len=8, batch_size=2, tokenizer=tokenizer,
    )
    batches = list(itertools.islice(iter(mix), 3))
    assert len(batches) == 3
    for x, _ in batches:
        assert x.shape == (2, 8)


def test_zero_weight_skipped(tokenizer, tmp_path):
    """Sources with zero weight don't contribute and don't crash."""
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    pq.write_table(pa.table({"text": ["alpha"] * 50}), str(a))
    pq.write_table(pa.table({"text": ["beta"] * 50}), str(b))
    mix = WeightedMixedStream.from_paths_with_weights(
        [(str(a), 1.0), (str(b), 0.0)],
        seq_len=4, batch_size=2, tokenizer=tokenizer,
    )
    # Should not raise; only "a" contributes.
    batches = list(itertools.islice(iter(mix), 3))
    assert len(batches) == 3


# ---------------------------------------------------------------------------
# all sources contribute
# ---------------------------------------------------------------------------


def test_all_sources_contribute(tokenizer, tmp_path):
    """With three sources at non-zero weight, each must yield at least one chunk."""
    paths = []
    surfaces = ["alpha", "beta", "gamma"]
    for surf in surfaces:
        p = tmp_path / f"{surf}.parquet"
        pq.write_table(pa.table({"text": [surf] * 1000}), str(p))
        paths.append(str(p))
    mix = WeightedMixedStream.from_paths_with_weights(
        [(paths[0], 0.5), (paths[1], 0.3), (paths[2], 0.2)],
        seq_len=4, batch_size=1, tokenizer=tokenizer,
    )
    chunks = list(itertools.islice(mix._iter_chunks(), 5_000))
    surface_ids = [tokenizer.encode(s)[0] for s in surfaces]
    counts = {sid: 0 for sid in surface_ids}
    for c in chunks:
        if c[0] in counts:
            counts[c[0]] += 1
    for sid, c in counts.items():
        assert c > 0, f"source with id {sid} contributed 0 chunks"
