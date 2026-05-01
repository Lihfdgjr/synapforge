"""Integration tests for the byte-patch image synth pipeline (T3.2).

Covers
------
1. ``test_smoke_n_10``           -- ``main(['--smoke', ...])`` runs end-to-end
   on CPU with a stubbed tokenizer; produces a 10-row parquet with the
   contract columns and a manifest sidecar.
2. ``test_deterministic``        -- same ``--seed`` -> identical bytes
   (image_patches column hash matches across two runs).
3. ``test_patch_byte_count``     -- every ``image_patches`` row is exactly
   768 bytes (16 patches * 48 bytes/patch).
4. ``test_caption_describes_image`` -- caption mentions a shape OR color
   that is also recorded in the meta columns of the same row (sanity
   that the caption isn't decoupled from the rendered scene).

All tests pin the offline ``_StubTokenizer`` so CI never needs to load
the real Qwen tokenizer or hit HuggingFace network.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


# Ensure the script dir is on sys.path even when this module runs without
# tests/integration/conftest.py being collected.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def synth_img_module(monkeypatch):
    """Reload module fresh; force ``_load_tokenizer`` to return the stub.

    Critical for hermeticism: the real ``_load_tokenizer`` would try
    ``transformers`` then HuggingFace network; we patch that out so tests
    pass on a torch-less, network-less CI box.
    """
    pytest.importorskip("pyarrow")
    pytest.importorskip("PIL")
    if "synth_image_pretrain" in sys.modules:
        mod = importlib.reload(sys.modules["synth_image_pretrain"])
    else:
        mod = importlib.import_module("synth_image_pretrain")

    def _stub_loader(name="Qwen/Qwen2.5-0.5B"):
        return mod._StubTokenizer()

    monkeypatch.setattr(mod, "_load_tokenizer", _stub_loader)
    return mod


def _run_main(mod, **kwargs) -> Path:
    """Invoke ``mod.main(argv)`` and return the output path."""
    out = Path(kwargs["output"])
    argv = ["--output", str(out)]
    if "n" in kwargs:
        argv += ["--n", str(kwargs["n"])]
    if "seed" in kwargs:
        argv += ["--seed", str(kwargs["seed"])]
    if kwargs.get("smoke"):
        argv += ["--smoke"]
    rc = mod.main(argv)
    assert rc == 0, f"main exited {rc}"
    return out


# -------------------- 1. smoke n=10 ---------------------------------------
def test_smoke_n_10(tmp_path, synth_img_module):
    """``--smoke`` produces a 10-row parquet with the expected schema."""
    import pyarrow.parquet as pq

    out = tmp_path / "synth_image_smoke.parquet"
    _run_main(synth_img_module, output=out, smoke=True, seed=42)

    assert out.exists(), "parquet not written"
    assert (out.with_name(out.name + ".manifest.json")).exists(), \
        "manifest sidecar missing"

    table = pq.read_table(str(out))
    assert table.num_rows == 10, f"want 10 rows; got {table.num_rows}"
    cols = set(table.column_names)
    assert {"text", "image_patches", "caption", "shape", "color", "bg"}.issubset(cols), \
        f"missing required columns; got {cols}"

    # Stub tokenizer always returns [1, 2, 3].
    first_text = list(table.column("text")[0].as_py())
    assert first_text == [1, 2, 3], f"stub tok contract; got {first_text}"

    # Manifest sanity.
    with open(str(out) + ".manifest.json") as f:
        m = json.load(f)
    assert m["kind"] == "synth_image_pretrain"
    assert m["rows"] == 10
    assert m["patch_bytes_per_image"] == 768
    assert m["image_size"] == 32


# -------------------- 2. determinism --------------------------------------
def test_deterministic(tmp_path, synth_img_module):
    """Same seed -> identical image_patches across two independent runs."""
    import pyarrow.parquet as pq

    def _hash(out: Path) -> str:
        df = pq.read_table(str(out)).to_pandas()
        # Hash the whole image_patches column to detect ANY drift.
        h = hashlib.sha256()
        for row in df["image_patches"]:
            h.update(bytes(list(row)))
        return h.hexdigest()

    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"
    out_c = tmp_path / "c.parquet"
    _run_main(synth_img_module, output=out_a, n=8, seed=11)
    _run_main(synth_img_module, output=out_b, n=8, seed=11)
    _run_main(synth_img_module, output=out_c, n=8, seed=99)

    h_a, h_b, h_c = _hash(out_a), _hash(out_b), _hash(out_c)
    assert h_a == h_b, "same seed must produce identical image_patches"
    assert h_a != h_c, "different seed must produce different patches"


# -------------------- 3. byte count invariant -----------------------------
def test_patch_byte_count(tmp_path, synth_img_module):
    """Every row's image_patches list is exactly 768 bytes (uint8)."""
    import pyarrow.parquet as pq

    out = tmp_path / "bytes.parquet"
    _run_main(synth_img_module, output=out, smoke=True, seed=7)

    df = pq.read_table(str(out)).to_pandas()
    assert len(df) == 10
    for i, row in enumerate(df["image_patches"]):
        bs = list(row)
        assert len(bs) == 768, f"row {i}: want 768 bytes; got {len(bs)}"
        # uint8 range 0-255 (parquet stores ints, not bytes objects).
        assert min(bs) >= 0 and max(bs) <= 255, \
            f"row {i}: byte values out of range [{min(bs)}, {max(bs)}]"

    # Module-level invariant should match.
    assert synth_img_module.PATCH_SEQ_BYTES == 768
    assert synth_img_module.N_PATCHES == 16
    assert synth_img_module.PATCH_BYTES == 48


def test_image_to_patches_invariant(synth_img_module):
    """``image_to_patches`` is a pure function; smoke a known input."""
    import numpy as np

    # All-grey 32x32 image.
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    raw = synth_img_module.image_to_patches(img)
    assert len(raw) == 768
    # Mean-pool of constant input -> all bytes identical.
    arr = np.frombuffer(raw, dtype=np.uint8)
    assert (arr == 128).all(), "constant-image patches must be constant"

    # Mismatched shape raises.
    with pytest.raises(ValueError):
        synth_img_module.image_to_patches(np.zeros((16, 16, 3), dtype=np.uint8))


# -------------------- 4. caption sanity -----------------------------------
def test_caption_describes_image(tmp_path, synth_img_module):
    """Caption text mentions the row's shape OR color label."""
    import pyarrow.parquet as pq

    out = tmp_path / "captions.parquet"
    _run_main(synth_img_module, output=out, n=20, seed=42)

    df = pq.read_table(str(out)).to_pandas()
    assert len(df) == 20
    matched = 0
    for _, row in df.iterrows():
        cap = str(row["caption"]).lower()
        # Shape OR color must appear verbatim in caption (current
        # template embeds both, so this is a strict per-row sanity).
        if str(row["shape"]) in cap or str(row["color"]) in cap:
            matched += 1
    assert matched == len(df), (
        f"only {matched}/{len(df)} captions referenced their shape/color; "
        f"sample row 0 caption={df.iloc[0]['caption']!r} "
        f"shape={df.iloc[0]['shape']!r} color={df.iloc[0]['color']!r}"
    )


# -------------------- bonus coverage --------------------------------------
def test_arg_parse_smoke(synth_img_module):
    """``_parse_args`` round-trips required flags."""
    ns = synth_img_module._parse_args([
        "--output", "/tmp/x.parquet", "--smoke", "--seed", "13",
    ])
    assert ns.smoke is True
    assert ns.seed == 13
    assert ns.output == "/tmp/x.parquet"


def test_stub_tokenizer_contract(synth_img_module):
    """``_StubTokenizer.encode`` returns [1,2,3] for any text -- contract."""
    tok = synth_img_module._StubTokenizer()
    assert tok.encode("any string at all") == [1, 2, 3]
    assert tok.encode("") == [1, 2, 3]
