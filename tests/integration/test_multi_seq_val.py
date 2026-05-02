"""T9.3 (DEEP_MAINT_QUEUE.md) -- multi-seq-len VAL + monotonic-quality test.

The user 铁律 (memory: feedback_50m_context_monotonic_quality.md and
feedback_long_context_quality.md) is that an inference-STDP model should
get LOWER per-token ppl as context grows. Existing T1.7/T1.8 tests cover
the eval-side claim with mocked tensors; this test covers the TRAINER
side -- that ``--val-seq-lens`` actually adds per-length val passes,
emits the per-seq log format, and computes the
``quality_grows_with_seq`` flag correctly.

Tests
-----
* ``test_default_single_seq``        -- no flag => ``_parse_val_seq_lens``
                                        returns ``[]`` (legacy behaviour).
* ``test_multi_seq_logs_per_seq``    -- 3 lengths => 3 ppl entries in
                                        the per-seq dict + 3 ``seq=...``
                                        columns in the log line.
* ``test_monotonic_check_emits_flag``-- a ppl dict that strictly drops
                                        with seq_len yields True; one
                                        with a rise yields False.
* ``test_auto_scale_batch``          -- empty ``--val-seq-lens-bs`` +
                                        auto-scale => bs scales as
                                        ``int(base_bs * train_seq /
                                        seq_len)`` floor 1.

All tests CPU-friendly: helpers are pure stdlib + dict; the eval-loop
plumbing is exercised via ``unittest.mock`` so no real torch model /
parquet shard is needed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_trainer() -> ModuleType:
    """Load ``train_100m_kd.py`` as a module so we can call its helpers.

    Skips the test when torch / synapforge transitive imports are
    unavailable -- the trainer's top-level imports pull in torch +
    synapforge, so any environment that lacks them genuinely cannot
    exercise the trainer surface. The pure-stdlib helpers
    (``_parse_val_seq_lens``, ``_monotonic_quality_grows``,
    ``_format_per_seq_log``) live INSIDE that module too, so we accept
    the heavy import as the cost of testing them in-place.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not installed; trainer module cannot be imported")
    try:
        import synapforge  # noqa: F401
    except Exception:
        pytest.skip("synapforge not importable in this env")

    src = _REPO_ROOT / "train_100m_kd.py"
    spec = importlib.util.spec_from_file_location("train_100m_kd_iso", src)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not build spec for {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Default single-seq behaviour: empty --val-seq-lens => no extra val.
# ---------------------------------------------------------------------------


def test_default_single_seq():
    """Empty CSV => empty pair list => caller falls through to legacy path."""
    mod = _load_trainer()

    # The contract: when --val-seq-lens is empty, _parse_val_seq_lens
    # returns []. The trainer's main val loop short-circuits the
    # multi-seq branch on `if _vsl_pairs:`, so behaviour is identical
    # to the historical single-seq val flow.
    assert mod._parse_val_seq_lens(
        val_seq_lens_csv="",
        val_seq_lens_bs_csv="",
        train_seq_len=256,
        train_batch_size=32,
        auto_scale=True,
    ) == []

    # Whitespace-only must also be treated as empty (CSV parsing).
    assert mod._parse_val_seq_lens(
        val_seq_lens_csv="   ",
        val_seq_lens_bs_csv="",
        train_seq_len=256,
        train_batch_size=32,
        auto_scale=True,
    ) == []

    # The monotonic check on an empty dict (legacy path, never written)
    # is True trivially -- there are no adjacent pairs to violate.
    assert mod._monotonic_quality_grows({}) is True


# ---------------------------------------------------------------------------
# 2. Multi-seq: each seq gets a ppl log line.
# ---------------------------------------------------------------------------


def test_multi_seq_logs_per_seq():
    """3 requested lengths => 3 ppl entries + the rendered log line shows
    each ``seq=N ppl=X.XX`` token, sorted by seq_len ascending.
    """
    mod = _load_trainer()

    pairs = mod._parse_val_seq_lens(
        val_seq_lens_csv="256,512,1024",
        val_seq_lens_bs_csv="",
        train_seq_len=256,
        train_batch_size=32,
        auto_scale=True,
    )
    assert len(pairs) == 3, pairs
    assert [p[0] for p in pairs] == [256, 512, 1024]

    # Simulate a per-seq ppl dict the way the trainer would produce it.
    per_seq_ppl = {256: 120.0, 512: 100.0, 1024: 80.0}
    line = mod._format_per_seq_log(
        step=100, per_seq_ppl=per_seq_ppl,
        monotonic=mod._monotonic_quality_grows(per_seq_ppl),
    )
    # The header is the canonical "VAL step <N>" prefix that downstream
    # log-parsers (phase_manager, dashboards) already match on.
    assert line.startswith("VAL step 100 ")
    # Each requested length gets its own ``seq=N ppl=X.XX`` token.
    assert "seq=256 ppl=120.00" in line
    assert "seq=512 ppl=100.00" in line
    assert "seq=1024 ppl=80.00" in line
    # Monotonic flag is on its own line below for grep-friendliness.
    assert "quality_grows_with_seq=True" in line
    # The seq tokens are sorted ASC -- 256 must appear before 1024.
    assert line.index("seq=256 ") < line.index("seq=1024 ")

    # End-to-end shape: build a fake ParquetTokenStream + fake evaluate
    # and assert _eval_at_seq_lens returns one entry per length.
    fake_stream_calls = []

    class _FakeStream:
        def __init__(self, *a, **kw):
            fake_stream_calls.append(kw)

        def __iter__(self):
            return iter(())

    def _fake_evaluate(model, val_iter, n_batches=16, plif_cells=None):
        # Return a deterministic ppl that grows ~inversely with the call
        # count, so the resulting per-seq dict is monotonically decreasing
        # in the order the helper iterates pairs (256 -> 512 -> 1024).
        idx = len(fake_stream_calls)  # 1 after first call
        return float(200.0 / max(1, idx))

    with mock.patch.object(mod, "ParquetTokenStream", _FakeStream), \
         mock.patch.object(mod, "evaluate", _fake_evaluate):
        out = mod._eval_at_seq_lens(
            model=None,
            pairs=pairs,
            val_glob="dummy/*.parquet",
            data_glob="dummy/*.parquet",
            tokenizer_name="gpt2",
            n_batches=4,
            remote_warehouse=None,
        )
    # One entry per (seq_len, batch_size) pair.
    assert set(out.keys()) == {256, 512, 1024}
    # And each ParquetTokenStream construction used the requested seq_len.
    used_seq_lens = sorted(int(k.get("seq_len")) for k in fake_stream_calls)
    assert used_seq_lens == [256, 512, 1024]


# ---------------------------------------------------------------------------
# 3. Monotonic-quality flag: True iff strict DECREASE across seq_len asc.
# ---------------------------------------------------------------------------


def test_monotonic_check_emits_flag():
    """``quality_grows_with_seq`` is the user 铁律. ppl strictly drops
    as context grows => True; otherwise False.

    The flag is what surfaces a regression to phase_manager /
    dashboards, so the contract has to be exact: no off-by-one
    (single-entry => True), NaN/Inf failures, and equality counts as
    a regression (model not USING the extra context).
    """
    mod = _load_trainer()

    # Strict-decrease => True (the inference-STDP claim).
    assert mod._monotonic_quality_grows({256: 120.0, 512: 100.0, 1024: 80.0}) is True

    # Any rise => False.
    assert mod._monotonic_quality_grows({256: 80.0, 512: 100.0, 1024: 120.0}) is False
    assert mod._monotonic_quality_grows({256: 120.0, 512: 130.0, 1024: 80.0}) is False

    # Equality at any pair => False (model not using extra context).
    assert mod._monotonic_quality_grows({256: 100.0, 512: 100.0, 1024: 80.0}) is False

    # NaN / Inf at any seat => False.
    assert mod._monotonic_quality_grows({256: 120.0, 512: float("nan"), 1024: 80.0}) is False
    assert mod._monotonic_quality_grows({256: 120.0, 512: float("inf"), 1024: 80.0}) is False

    # Trivial cases.
    assert mod._monotonic_quality_grows({}) is True
    assert mod._monotonic_quality_grows({256: 120.0}) is True

    # The renderer surfaces both True and False explicitly so log parsers
    # can grep either way.
    on = mod._format_per_seq_log(7, {256: 120.0, 1024: 80.0}, True)
    off = mod._format_per_seq_log(7, {256: 120.0, 1024: 130.0}, False)
    assert "quality_grows_with_seq=True" in on
    assert "quality_grows_with_seq=False" in off


# ---------------------------------------------------------------------------
# 4. Auto-scale batch size: bigger seq => smaller batch.
# ---------------------------------------------------------------------------


def test_auto_scale_batch():
    """Empty --val-seq-lens-bs + auto-scale => bs = int(base_bs * train_seq / seq_len).

    The activation memory cost grows linearly with seq_len at fixed
    batch (lots of intermediates: hidden, attention scores, KV cache).
    Doubling seq_len at fixed bs roughly doubles the activation
    footprint, so to stay in the same VRAM band we halve the batch.
    The auto-scale formula does exactly that.
    """
    mod = _load_trainer()

    pairs = mod._parse_val_seq_lens(
        val_seq_lens_csv="256,512,1024,4096",
        val_seq_lens_bs_csv="",
        train_seq_len=256,
        train_batch_size=32,
        auto_scale=True,
    )
    # Same seq_len => same bs (divisor 1).
    assert pairs[0] == (256, 32)
    # 2x seq_len => half bs.
    assert pairs[1] == (512, 16)
    # 4x seq_len => quarter bs.
    assert pairs[2] == (1024, 8)
    # 16x seq_len => 1/16 bs == 2.
    assert pairs[3] == (4096, 2)

    # Bigger seq always yields smaller-or-equal batch.
    bs_seq_pairs = sorted(pairs, key=lambda x: x[0])
    for i in range(len(bs_seq_pairs) - 1):
        s1, b1 = bs_seq_pairs[i]
        s2, b2 = bs_seq_pairs[i + 1]
        assert b2 <= b1, (
            f"auto-scale must yield non-increasing bs as seq grows; "
            f"saw seq={s1} bs={b1} then seq={s2} bs={b2}"
        )

    # Floor at 1 -- never produce bs=0 even if seq_len is huge.
    pairs_huge = mod._parse_val_seq_lens(
        val_seq_lens_csv="1000000",
        val_seq_lens_bs_csv="",
        train_seq_len=256,
        train_batch_size=4,
        auto_scale=True,
    )
    assert pairs_huge == [(1000000, 1)]

    # auto_scale=False => fall back to base_bs at every length (caller's
    # OOM problem). This covers the explicit opt-out path.
    pairs_no_scale = mod._parse_val_seq_lens(
        val_seq_lens_csv="256,1024,4096",
        val_seq_lens_bs_csv="",
        train_seq_len=256,
        train_batch_size=8,
        auto_scale=False,
    )
    assert [b for (_, b) in pairs_no_scale] == [8, 8, 8]

    # Explicit --val-seq-lens-bs overrides auto-scale entirely.
    pairs_explicit = mod._parse_val_seq_lens(
        val_seq_lens_csv="256,512,1024",
        val_seq_lens_bs_csv="24,12,4",
        train_seq_len=256,
        train_batch_size=32,
        auto_scale=True,  # ignored when bs CSV non-empty
    )
    assert pairs_explicit == [(256, 24), (512, 12), (1024, 4)]


# ---------------------------------------------------------------------------
# 5. Bad CSV / mismatched lengths => ValueError so the trainer logs and
#    falls through to the legacy single-seq path.
# ---------------------------------------------------------------------------


def test_bad_csv_raises_value_error():
    """The trainer wraps the helper in try/except so it logs and continues;
    that contract is only sound if the helper raises ValueError on
    malformed input rather than crashing or silently truncating."""
    mod = _load_trainer()

    with pytest.raises(ValueError):
        mod._parse_val_seq_lens(
            val_seq_lens_csv="abc,def",
            val_seq_lens_bs_csv="",
            train_seq_len=256,
            train_batch_size=32,
            auto_scale=True,
        )

    with pytest.raises(ValueError):
        mod._parse_val_seq_lens(
            val_seq_lens_csv="256,-512",
            val_seq_lens_bs_csv="",
            train_seq_len=256,
            train_batch_size=32,
            auto_scale=True,
        )

    with pytest.raises(ValueError):
        mod._parse_val_seq_lens(
            val_seq_lens_csv="256,512",
            val_seq_lens_bs_csv="24,12,4",  # length mismatch
            train_seq_len=256,
            train_batch_size=32,
            auto_scale=True,
        )
