"""Integration tests for the audio synthetic data generator (T3.3).

Covers
------
1. ``test_smoke_n_10``                    -- ``--smoke`` writes 10 rows of
   the documented schema (parquet columns + types).
2. ``test_deterministic``                 -- same ``--seed`` produces
   byte-identical mel_bytes (no hidden non-determinism).
3. ``test_mel_byte_count``                -- every row's ``mel_bytes`` is
   exactly 8000 bytes (80 mel bins * 100 frames).
4. ``test_caption_mentions_audio_property`` -- every caption contains one
   of {``Hz``, ``noise``, ``chirp``}.

All tests run on CPU with only ``numpy`` + ``pyarrow``. They never require
``librosa``, ``torch``, or the rental.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


# Local sys.path injection mirrors test_collect_kd_data.py so the test
# module is collectable in isolation (no conftest dependence).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def synth_audio():
    pytest.importorskip("numpy")
    if "synth_audio_pretrain" in sys.modules:
        return importlib.reload(sys.modules["synth_audio_pretrain"])
    return importlib.import_module("synth_audio_pretrain")


# --------------------------------------------------------------- 1: smoke
def test_smoke_n_10(tmp_path, synth_audio):
    """``--smoke`` writes exactly 10 rows with the documented schema."""
    pytest.importorskip("pyarrow")
    out = tmp_path / "smoke.parquet"
    argv = ["--smoke", "--seed", "42", "--out", str(out)]
    rc = synth_audio.main(argv)
    assert rc == 0
    assert out.exists()

    import pyarrow.parquet as pq
    table = pq.read_table(str(out))
    assert table.num_rows == 10
    cols = set(table.column_names)
    expected = {"text", "caption", "mel_bytes", "freq_hz", "klass"}
    assert expected.issubset(cols), f"missing cols: {expected - cols}"

    # Manifest sanity.
    import json
    with open(str(out) + ".manifest.json") as f:
        manifest = json.load(f)
    assert manifest["kind"] == "synth_audio_mel"
    assert manifest["rows"] == 10
    assert manifest["seed"] == 42
    assert manifest["mel_bytes_per_row"] == 8000
    assert manifest["sample_rate"] == 16000
    assert set(manifest["classes"]) == {"sine", "noise", "chirp"}


# --------------------------------------------------------- 2: determinism
def test_deterministic(tmp_path, synth_audio):
    """Re-running with the same seed produces identical mel_bytes per row."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    c = tmp_path / "c.parquet"

    assert synth_audio.main(["--smoke", "--seed", "11", "--out", str(a)]) == 0
    assert synth_audio.main(["--smoke", "--seed", "11", "--out", str(b)]) == 0
    assert synth_audio.main(["--smoke", "--seed", "99", "--out", str(c)]) == 0

    ta = pq.read_table(str(a)).to_pylist()
    tb = pq.read_table(str(b)).to_pylist()
    tc = pq.read_table(str(c)).to_pylist()

    # Same seed -> identical bytes per row + identical captions.
    for ra, rb in zip(ta, tb):
        assert bytes(ra["mel_bytes"]) == bytes(rb["mel_bytes"]), \
            "mel_bytes differ across runs with same seed"
        assert ra["caption"] == rb["caption"], \
            "caption differs across runs with same seed"
        assert ra["klass"] == rb["klass"]
        assert ra["freq_hz"] == rb["freq_hz"]

    # Different seed -> at least *some* rows must differ (else seed is dead).
    diffs = sum(
        1 for ra, rc in zip(ta, tc)
        if bytes(ra["mel_bytes"]) != bytes(rc["mel_bytes"])
    )
    assert diffs >= 5, (
        f"different seeds should differ on >=5/10 rows; got {diffs}/10"
    )


# ------------------------------------------------------- 3: mel byte count
def test_mel_byte_count(tmp_path, synth_audio):
    """Each ``mel_bytes`` cell is exactly 80 * 100 = 8000 bytes."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    out = tmp_path / "mel.parquet"
    assert synth_audio.main(
        ["--smoke", "--seed", "7", "--out", str(out)]
    ) == 0
    table = pq.read_table(str(out))
    mel_col = table.column("mel_bytes").to_pylist()
    assert len(mel_col) == 10
    for i, mb in enumerate(mel_col):
        assert len(mb) == 8000, (
            f"row {i}: mel_bytes len {len(mb)} != 8000 (expected 80*100)"
        )
        # All values must be valid uint8 (0..255).
        # ``mb`` is bytes; iter -> int 0..255 by definition, but we still
        # spot check they're not all zero (bug guard) for non-noise rows.
        if any(v != 0 for v in mb[:32]):
            break
    else:
        pytest.fail("all rows have leading zeros in mel_bytes -- quant broken")


# ------------------------------------ 4: caption mentions audio property
def test_caption_mentions_audio_property(tmp_path, synth_audio):
    """Every caption contains at least one of {Hz, noise, chirp}."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    out = tmp_path / "captions.parquet"
    # Use 30 rows so we very likely hit all three classes (CLASS_LIST has 3).
    assert synth_audio.main(
        ["--n", "30", "--seed", "3", "--out", str(out)]
    ) == 0
    table = pq.read_table(str(out))
    captions = table.column("caption").to_pylist()
    assert len(captions) == 30
    keywords = ("Hz", "noise", "chirp")
    for cap in captions:
        assert any(kw in cap for kw in keywords), (
            f"caption {cap!r} mentions none of {keywords}"
        )

    # Sanity: across 30 rows we should see >=2 of the 3 classes (not the
    # full set strictly, but the seed should diversify).
    klasses = set(table.column("klass").to_pylist())
    assert len(klasses) >= 2, f"only saw klasses {klasses} in 30 rows"


# ----------------------------- 5: helper-level math (extra robustness) -----
def test_mel_filterbank_shape(synth_audio):
    fb = synth_audio.build_mel_filterbank()
    assert fb.shape == (synth_audio.N_MELS,
                        synth_audio.N_FFT // 2 + 1)
    # Each filter must have non-zero energy (no dead bins).
    sums = fb.sum(axis=1)
    assert (sums > 0).all(), "dead mel filter found"


def test_quantize_uint8_range(synth_audio):
    """Quantizer maps any log-mel range to [0, 255] uint8."""
    import numpy as np
    rng = np.random.default_rng(0)
    log_mel = rng.normal(-3.0, 1.5, (80, 100)).astype("float32")
    q = synth_audio.quantize_mel_to_uint8(log_mel)
    assert q.dtype.name == "uint8"
    assert q.shape == (80, 100)
    assert int(q.min()) >= 0
    assert int(q.max()) <= 255


def test_mel_spectrogram_shape(synth_audio):
    """mel_spectrogram of a 16k-sample wav yields (80, 100) frames."""
    import numpy as np
    wav = np.sin(2 * np.pi * 440.0 *
                 np.arange(synth_audio.N_SAMPLES) /
                 synth_audio.SAMPLE_RATE).astype("float32")
    log_mel = synth_audio.mel_spectrogram(wav)
    assert log_mel.shape == (synth_audio.N_MELS, synth_audio.N_FRAMES)
    assert log_mel.shape == (80, 100)


def test_generate_row_returns_correct_keys(synth_audio):
    row = synth_audio.generate_row(seed_int=12345)
    expected = {"text", "caption", "mel_bytes", "freq_hz", "klass"}
    assert set(row.keys()) >= expected
    assert isinstance(row["mel_bytes"], (bytes, bytearray))
    assert len(row["mel_bytes"]) == synth_audio.MEL_BYTES_PER_ROW
    assert row["klass"] in synth_audio.CLASS_LIST
