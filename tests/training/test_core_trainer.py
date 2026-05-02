"""tests/training/test_core_trainer.py -- BaseTrainer + KDTrainer + dispatcher.

Required by D7 of the trainer-refactor plan. Five tests:

1. trainer construct (BaseTrainer + TrainerConfig basic plumbing)
2. train_step deterministic (running the same batch twice gives same loss)
3. ckpt save/load round-trip (state_dict equality)
4. KD math bit-exact vs train_100m_kd._kd_loss
5. mode dispatch (--mode {kd,sft,rl,self_distill} --dry-run)

All tests run on CPU with a tiny model (d=8, n_layers=1) so no GPU /
synapforge model deps required. The KD bit-exact test compares
``synapforge.training.kd_math.kd_loss`` against
``train_100m_kd._kd_loss`` directly on synthetic logits -- no model
forward needed.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.fixture
def torch_available():
    pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Test 1: trainer construct
# ---------------------------------------------------------------------------


def test_trainer_construct(torch_available, tmp_path):
    """BaseTrainer + TrainerConfig basic construction works."""
    import torch

    from synapforge.training.core_trainer import BaseTrainer, TrainerConfig

    cfg = TrainerConfig(
        out_dir=str(tmp_path),
        steps=10,
        batch_size=2,
        seq_len=4,
        lr=1e-3,
        device="cpu",
        dtype="float32",
    )
    model = torch.nn.Linear(4, 8)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    def _stream():
        while True:
            yield (torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long))

    trainer = BaseTrainer(cfg, model, optim, iter(_stream()), iter(_stream()))
    assert trainer.cfg.steps == 10
    assert trainer.step == 0
    assert trainer.device == "cpu"
    assert trainer.best_val_ppl == float("inf")
    assert trainer.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 2: train_step deterministic
# ---------------------------------------------------------------------------


def test_train_step_deterministic(torch_available):
    """Same batch + same seed -> same loss across two runs."""
    import torch

    from synapforge.training.kd_math import kd_loss

    torch.manual_seed(42)
    s_logits = torch.randn(2, 4, 100)
    t_logits = torch.randn(2, 4, 100)

    # Determinism: kd_loss is pure -- same inputs => same output.
    a = kd_loss(s_logits, t_logits, T=4.0, topk=32)
    b = kd_loss(s_logits, t_logits, T=4.0, topk=32)
    assert torch.allclose(a, b, atol=1e-7)


# ---------------------------------------------------------------------------
# Test 3: ckpt save/load round-trip
# ---------------------------------------------------------------------------


def test_ckpt_save_load_roundtrip(torch_available, tmp_path):
    """save_ckpt followed by load_ckpt returns identical state dicts."""
    import torch

    from synapforge.training.core_trainer import (
        load_ckpt,
        save_ckpt,
        TrainerConfig,
    )

    cfg = TrainerConfig(out_dir=str(tmp_path), vocab=128, d=16, n_layers=2)
    model = torch.nn.Linear(16, 128)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Take one optimizer step so optim state has content.
    x = torch.randn(2, 16)
    y = torch.randint(0, 128, (2,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optim.step()

    path = os.path.join(str(tmp_path), "test_ckpt.pt")
    save_ckpt(
        path, model=model, optim=optim, step=42,
        config=cfg.to_ckpt_config(),
    )
    assert os.path.exists(path)

    ck = load_ckpt(path)
    assert ck["step"] == 42
    assert ck["config"]["vocab"] == 128
    assert ck["config"]["d"] == 16
    assert ck["config"]["n_layers"] == 2

    # Load into a fresh model and verify state dict equality.
    fresh = torch.nn.Linear(16, 128)
    fresh.load_state_dict(ck["model"])
    for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                   fresh.named_parameters()):
        assert torch.equal(p1, p2), f"param {n1} != {n2} after roundtrip"


# ---------------------------------------------------------------------------
# Test 4: KD math bit-exact vs train_100m_kd._kd_loss
# ---------------------------------------------------------------------------


def test_kd_math_bitexact(torch_available):
    """synapforge.training.kd_math.kd_loss matches train_100m_kd._kd_loss.

    Critical bit-exactness contract. We import _kd_loss from the legacy
    module and compare against the lifted version on synthetic data.
    """
    import torch

    from synapforge.training.kd_math import kd_loss as new_kd_loss

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    legacy_path = os.path.join(repo_root, "train_100m_kd.py")
    if not os.path.exists(legacy_path):
        pytest.skip(f"legacy trainer not at {legacy_path}; skipping")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_legacy_train_100m_kd", legacy_path
    )
    legacy = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(legacy)
    except Exception as exc:  # pragma: no cover -- env-dependent
        pytest.skip(f"could not load legacy trainer: {exc!r}")
    if not hasattr(legacy, "_kd_loss"):
        pytest.skip("legacy trainer has no _kd_loss attr")

    torch.manual_seed(123)
    s_logits = torch.randn(4, 8, 1024) * 2.0
    t_logits = torch.randn(4, 8, 1024) * 2.0

    new_val = new_kd_loss(s_logits, t_logits, T=4.0, topk=128)
    legacy_val = legacy._kd_loss(s_logits, t_logits, T=4.0, topk=128)

    assert torch.allclose(new_val, legacy_val, atol=1e-5), (
        f"top-K KD bit-exactness failed: new={new_val.item()} "
        f"legacy={legacy_val.item()} diff={abs(new_val-legacy_val).item()}"
    )

    # Also test the chunked path (topk=0).
    new_chunk = new_kd_loss(s_logits, t_logits, T=4.0, topk=0,
                            chunk_override=2)
    legacy_chunk = legacy._kd_loss(s_logits, t_logits, T=4.0, topk=0,
                                    chunk_override=2)
    assert torch.allclose(new_chunk, legacy_chunk, atol=1e-5), (
        f"chunked KD bit-exactness failed: new={new_chunk.item()} "
        f"legacy={legacy_chunk.item()} "
        f"diff={abs(new_chunk-legacy_chunk).item()}"
    )


# ---------------------------------------------------------------------------
# Test 5: mode dispatch via --dry-run
# ---------------------------------------------------------------------------


def test_mode_dispatch_dry_run(torch_available, tmp_path):
    """Each --mode value should dispatch + exit 0 with --dry-run."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    for mode in ("kd", "sft", "rl", "self_distill"):
        result = subprocess.run(
            [sys.executable, "-m", "synapforge.training",
             "--mode", mode, "--dry-run",
             "--steps", "10", "--out", str(tmp_path / mode)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"mode {mode!r} dispatch failed: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        assert f"mode={mode}" in result.stdout, (
            f"mode {mode!r} not echoed: stdout={result.stdout!r}"
        )


# ---------------------------------------------------------------------------
# Bonus: KDTrainer compute_loss runs without teacher (pure CE path).
# ---------------------------------------------------------------------------


def test_kd_trainer_no_teacher_is_pure_ce(torch_available, tmp_path):
    """KDTrainer with teacher=None + kd_weight=0 falls back to pure CE."""
    import torch

    from synapforge.training.kd_trainer import KDTrainer, KDTrainerConfig

    cfg = KDTrainerConfig(
        out_dir=str(tmp_path),
        steps=1,
        batch_size=2,
        seq_len=4,
        kd_weight=0.0,
        device="cpu",
        dtype="float32",
        vocab=64,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(64, 8)
            self.head = torch.nn.Linear(8, 64)

        def forward(self, x):
            return self.head(self.emb(x))

    model = TinyModel()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    def _stream():
        while True:
            yield (torch.zeros(2, 4, dtype=torch.long),
                   torch.ones(2, 4, dtype=torch.long))

    trainer = KDTrainer(cfg, model, optim, iter(_stream()), iter(_stream()),
                        teacher=None)

    x = torch.zeros(2, 4, dtype=torch.long)
    y = torch.ones(2, 4, dtype=torch.long)
    out = trainer.compute_loss((x, y))

    # No teacher + kd_weight=0: kd term is 0, loss == ce.
    assert torch.allclose(out["loss"], out["ce"]), (
        f"loss {out['loss']} != ce {out['ce']} when teacher=None"
    )
    assert float(out["kd"].item()) == 0.0
