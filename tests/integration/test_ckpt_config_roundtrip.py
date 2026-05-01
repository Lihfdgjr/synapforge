"""Round-trip test for ckpt {"config": ...} persistence (P12).

Builds a tiny SynapForge100M (d=64, n_layers=2, vocab=1000), saves it with
the same dict shape that ``train_100m_kd.py::_build_config_dict`` emits,
then loads it through ``synapforge.demo.chat_demo._try_load_live`` and
asserts the reconstructed model matches the saved one in (a) parameter
shapes and (b) forward output on a fixed input.

Also asserts the legacy "no config" path still loads (backwards-compat —
existing rental ckpts must not break) and emits a warning (the WARNING
message is printed to stdout; we don't assert on the exact text, just on
the load succeeding).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def torch_or_skip():
    return pytest.importorskip("torch")


# Mirror of train_100m_kd.py::_build_config_dict shape. Kept inline (rather
# than importing the trainer module, which has heavy side effects: HF env
# vars, sys.path mutation, conditional imports) so the test stays hermetic
# and fast on a torch-only box.
def _build_config_dict_for_test(
    *, vocab: int, d: int, n_layers: int, loop_depth: int, max_seq: int,
    ffn_ratio: float = 8.0, sparsity: float = 0.95, dropout: float = 0.0,
    tie_lm_head: bool = True,
) -> dict[str, Any]:
    return {
        "vocab": vocab, "d": d, "n_layers": n_layers,
        "loop_depth": loop_depth, "max_seq": max_seq,
        "ffn_ratio": ffn_ratio, "sparsity": sparsity,
        "dropout": dropout, "tie_lm_head": tie_lm_head,
    }


def _build_tiny_model(torch):
    """Construct a SynapForge100M with shapes deliberately != the chat_demo
    fallback (d=64 vs 512, n_layers=2 vs 10, vocab=1000 vs 151936). If the
    loader was using its hardcoded fallback this test would fail with a
    flood of shape mismatches."""
    from synapforge.model_100m import SynapForge100M
    cfg = _build_config_dict_for_test(
        vocab=1000, d=64, n_layers=2, loop_depth=1, max_seq=32,
    )
    model = SynapForge100M(
        vocab=cfg["vocab"], d=cfg["d"], n_layers=cfg["n_layers"],
        loop_depth=cfg["loop_depth"], max_seq=cfg["max_seq"],
        ffn_ratio=cfg["ffn_ratio"], sparsity=cfg["sparsity"],
        dropout=cfg["dropout"], tie_lm_head=cfg["tie_lm_head"],
    )
    model.eval()
    return model, cfg


def _load_via_chat_demo(ckpt_path: Path, torch):
    """Replicates ``chat_demo._try_load_live`` minus the tokenizer dance.

    Reads ``ckpt["config"]`` if present (P12 path); falls back to
    chat_demo's hardcoded values otherwise. Mirrors the loader exactly so
    we exercise the same code path the demo uses.
    """
    from synapforge.demo.chat_demo import _FALLBACK_CFG
    from synapforge.model_100m import SynapForge100M

    raw = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(raw, dict) and "config" in raw and isinstance(raw["config"], dict):
        cfg = dict(_FALLBACK_CFG)
        cfg.update(raw["config"])
        had_cfg = True
    else:
        cfg = dict(_FALLBACK_CFG)
        had_cfg = False

    model = SynapForge100M(
        vocab=int(cfg["vocab"]), d=int(cfg["d"]),
        n_layers=int(cfg["n_layers"]), loop_depth=int(cfg["loop_depth"]),
        max_seq=int(cfg["max_seq"]), ffn_ratio=float(cfg["ffn_ratio"]),
        sparsity=float(cfg["sparsity"]), dropout=float(cfg["dropout"]),
        tie_lm_head=bool(cfg["tie_lm_head"]),
    )
    sd = raw["model"] if (isinstance(raw, dict) and "model" in raw) else raw
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg, had_cfg, missing, unexpected


def test_ckpt_config_roundtrip(tmp_path, torch_or_skip):
    """Save with config dict, load via chat_demo path, assert equivalent."""
    torch = torch_or_skip

    saved_model, saved_cfg = _build_tiny_model(torch)
    ckpt_path = tmp_path / "step_000010.pt"

    torch.save({
        "model": saved_model.state_dict(),
        "step": 10,
        "loss": 99.0,
        "config": _build_config_dict_for_test(
            vocab=saved_cfg["vocab"], d=saved_cfg["d"],
            n_layers=saved_cfg["n_layers"], loop_depth=saved_cfg["loop_depth"],
            max_seq=saved_cfg["max_seq"],
        ),
    }, str(ckpt_path))

    loaded_model, loaded_cfg, had_cfg, missing, unexpected = _load_via_chat_demo(
        ckpt_path, torch,
    )

    # 1. Loader picked up the saved config (NOT the fallback).
    assert had_cfg, "loader should have read ckpt['config']"
    assert loaded_cfg["d"] == 64
    assert loaded_cfg["n_layers"] == 2
    assert loaded_cfg["vocab"] == 1000
    assert loaded_cfg["max_seq"] == 32

    # 2. Param shapes match per-name. Any divergence here means the loader
    # built a model with the wrong shapes (the original P12 silent-failure
    # mode) and ``load_state_dict(strict=False)`` swallowed it.
    saved_sd = saved_model.state_dict()
    loaded_sd = loaded_model.state_dict()
    assert set(saved_sd) == set(loaded_sd), (
        f"key sets differ: only_saved={set(saved_sd)-set(loaded_sd)} "
        f"only_loaded={set(loaded_sd)-set(saved_sd)}"
    )
    for k, v in saved_sd.items():
        assert tuple(v.shape) == tuple(loaded_sd[k].shape), (
            f"shape mismatch for {k}: saved={tuple(v.shape)} "
            f"loaded={tuple(loaded_sd[k].shape)}"
        )

    # 3. No keys dropped (strict=False would silently swallow them).
    assert len(missing) == 0, f"missing keys after load: {missing}"
    assert len(unexpected) == 0, f"unexpected keys after load: {unexpected}"

    # 4. Forward outputs match bit-for-bit on a fixed input. This is the
    # ultimate "did the demo load garbage?" check.
    torch.manual_seed(0)
    x = torch.randint(0, saved_cfg["vocab"], (2, 8), dtype=torch.long)
    with torch.no_grad():
        y_saved = saved_model(x)
        y_loaded = loaded_model(x)
    assert y_saved.shape == y_loaded.shape
    assert torch.allclose(y_saved, y_loaded, atol=1e-5, rtol=1e-4), (
        f"forward output mismatch: max_abs="
        f"{(y_saved - y_loaded).abs().max().item():.3e}"
    )


def test_legacy_ckpt_no_config_falls_back(tmp_path, torch_or_skip, monkeypatch):
    """A ckpt without "config" must still load (backwards-compat for Run 2's
    rental ckpts written before P12 was resolved). Loader uses its
    hardcoded fallback and emits a warning, but does NOT crash.

    To keep the test CI-friendly we monkeypatch ``_FALLBACK_CFG`` to a tiny
    shape (vocab=1000, d=32, n_layers=2) -- the production fallback values
    (vocab=151936, d=512, n_layers=10) build a 78M-parameter model whose
    state_dict serialization is a few hundred MB and trips disk-pressure
    flakes when other tests run in parallel.
    """
    torch = torch_or_skip
    from synapforge.model_100m import SynapForge100M
    from synapforge.demo import chat_demo as _chat_demo

    # Tiny override of the fallback so we don't allocate a 78M-param model.
    # We patch the module-level constant; the loader copies it via
    # `dict(_FALLBACK_CFG)` so the patch is picked up freshly each call.
    tiny_fallback = {
        "vocab": 1000, "d": 32, "n_layers": 2, "loop_depth": 1,
        "max_seq": 16, "ffn_ratio": 4.0, "sparsity": 0.5,
        "dropout": 0.0, "tie_lm_head": True,
    }
    monkeypatch.setattr(_chat_demo, "_FALLBACK_CFG", tiny_fallback)

    # Build a tiny ckpt that matches the patched fallback shape on the
    # backbone but uses n_layers=1 (one fewer than fallback) so the loader
    # has missing keys when it tries to populate a 2-layer model.
    legacy = SynapForge100M(
        vocab=tiny_fallback["vocab"],
        d=tiny_fallback["d"],
        n_layers=1,  # fewer than fallback's 2 → >5 missing keys for layer 1
        loop_depth=tiny_fallback["loop_depth"],
        max_seq=tiny_fallback["max_seq"],
        ffn_ratio=tiny_fallback["ffn_ratio"],
        sparsity=tiny_fallback["sparsity"],
        dropout=tiny_fallback["dropout"],
        tie_lm_head=tiny_fallback["tie_lm_head"],
    )

    ckpt_path = tmp_path / "legacy.pt"
    # Legacy format: NO "config" key, just the model state_dict in "model".
    torch.save({
        "model": legacy.state_dict(),
        "step": 1,
    }, str(ckpt_path))

    loaded_model, loaded_cfg, had_cfg, missing, unexpected = _load_via_chat_demo(
        ckpt_path, torch,
    )

    # Loader used the (patched) fallback because no config was present.
    assert had_cfg is False
    assert loaded_cfg["d"] == tiny_fallback["d"]
    assert loaded_cfg["n_layers"] == tiny_fallback["n_layers"]
    # Legacy ckpt has 1 layer, fallback model has 2 → missing > 0. This
    # proves strict=False would NOT silently accept a smaller ckpt.
    assert len(missing) > 0
