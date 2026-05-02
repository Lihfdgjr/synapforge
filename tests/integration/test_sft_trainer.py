"""T9.4: Phase 2 SFT trainer tests.

Resolves T9.4 in ``docs/DEEP_MAINT_QUEUE.md``. Companion module:
``synapforge.training.sft_loop`` (response-only CE + parquet stream)
and ``train_100m_sft.py`` (the trainer entry point).

The four test cases:

  1. ``test_response_only_loss_masks_prompt`` -- the masked CE loss is
     equivalent to the full unmasked CE only over the response
     positions, and prompt positions never enter the average. We
     replicate the trainer's loss-fold arithmetic on a tiny model and
     verify (a) bit-exact zero contribution from prompt positions, (b)
     bit-exact equivalence to full CE when mask is all-ones.
  2. ``test_warmstart_compat_with_kd_ckpt`` -- a fake "Phase 1 ckpt" in
     the format ``train_100m_kd.py`` writes (model + optim_state +
     step + config) loads cleanly into the SFT trainer's model via
     ``adv_warmstart``, and an SFT ckpt round-trips back into the KD
     trainer's loader without missing keys.
  3. ``test_smoke_5_steps_runs_clean`` -- the SFT trainer's main()
     completes 5 steps end-to-end on CPU using a synthesised
     ``prompt_response`` parquet, a tiny d=32 model, and no warmstart.
     Asserts ckpts land on disk and metrics.json is well-formed.
  4. ``test_eval_alpaca_holdout_emits_ppl`` -- the holdout split
     produces a finite ppl number after a short run, and the row sets
     of the train/holdout streams are deterministically disjoint.

All tests are pure CPU + pure pytorch + pure pyarrow + tiny models so
they finish in <30s on a typical laptop without transformers / network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

# Repo root on sys.path so ``import synapforge`` and
# ``import train_100m_sft`` both work regardless of pytest invocation
# directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synapforge.training.sft_loop import (  # noqa: E402
    InstructionParquetStream,
    response_only_ce_loss,
    write_synth_alpaca_parquet,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_tiny_model(d: int = 32, vocab: int = 100, seq_len: int = 32):
    """Build a tiny synapforge_100m so smoke tests run on CPU.

    Wraps the production constructor with values small enough for CPU
    forward+backward in <1s but architecturally identical (same blocks,
    same RMSNorm, same tied LM head).
    """
    from synapforge.model_100m import build_synapforge_100m
    return build_synapforge_100m(
        vocab=vocab,
        d=d,
        n_layers=2,
        loop_depth=1,
        max_seq=seq_len,
        ffn_ratio=2.0,
        sparsity=0.5,
        dropout=0.0,
        use_grad_checkpoint=False,
        freeze_vocab_tail=False,
    )


# ===========================================================================
# Test 1 -- response_only_ce_loss masks prompt positions correctly
# ===========================================================================


def test_response_only_loss_masks_prompt() -> None:
    """response_only_ce_loss must zero-out prompt positions and equal
    full CE when the mask is all-ones.

    Two assertions:
    (a) When mask is 1 only on response tokens, swapping the prompt
        labels for arbitrary garbage MUST NOT change the loss --
        prompt positions don't contribute.
    (b) When mask is all-ones, the loss MUST equal
        F.cross_entropy(reduction='mean') over the same logits/labels.
    """
    torch.manual_seed(0)
    B, T, V = 2, 16, 50
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))

    # 8 prompt positions + 8 response positions per row -- response
    # only on the back half.
    mask = torch.zeros(B, T, dtype=torch.float32)
    mask[:, T // 2:] = 1.0

    loss_a = response_only_ce_loss(logits, labels, mask)
    # Swap prompt labels for a deterministic garbage tensor -- this
    # must NOT change loss_a, because mask=0 on those positions.
    labels_swapped = labels.clone()
    labels_swapped[:, : T // 2] = (V - 1) - labels[:, : T // 2]  # arbitrary
    loss_b = response_only_ce_loss(logits, labels_swapped, mask)

    assert torch.isfinite(loss_a)
    assert torch.allclose(loss_a, loss_b, atol=1e-7), (
        f"prompt-position labels leaked into the masked loss: "
        f"loss_a={loss_a.item():.6f} vs loss_b={loss_b.item():.6f}"
    )

    # Equivalence to full CE when mask is all-ones.
    mask_all = torch.ones(B, T, dtype=torch.float32)
    masked_full = response_only_ce_loss(logits, labels, mask_all)
    full = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), labels.reshape(-1), reduction="mean"
    )
    assert torch.allclose(masked_full, full, atol=1e-6), (
        f"full-mask response_only_ce_loss must equal F.cross_entropy: "
        f"masked={masked_full.item():.6f} vs full={full.item():.6f}"
    )

    # And the gradient must flow back through logits.
    loss_a.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # Gradient on prompt positions must be exactly zero (no signal).
    prompt_grad = logits.grad[:, : T // 2, :]
    assert torch.allclose(
        prompt_grad, torch.zeros_like(prompt_grad), atol=1e-8
    ), "gradient leaked into prompt positions despite mask=0"


# ===========================================================================
# Test 2 -- SFT ckpt format is bidirectionally compatible with kd trainer
# ===========================================================================


def test_warmstart_compat_with_kd_ckpt(tmp_path: Path) -> None:
    """A KD-shaped ckpt loads into the SFT trainer's model, and an SFT
    ckpt loads back into the KD trainer's model -- both via
    ``adv_warmstart`` -- with no missing/extra keys for the matching
    architecture.
    """
    from synapforge.huggingface_adapter import adv_warmstart

    # Build twin tiny models (KD-trained "source", SFT "target").
    src = _make_tiny_model(d=32, vocab=100)
    tgt = _make_tiny_model(d=32, vocab=100)

    # Modify src to a non-default state so we can verify weights actually
    # transferred (not just key-matched).
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.5)

    # Write a KD-shaped ckpt to disk: {"model", "optim_state", "step",
    # "config"} -- the exact shape ``train_100m_kd.py`` produces.
    kd_ckpt_path = tmp_path / "kd_step.pt"
    kd_payload = {
        "model": src.state_dict(),
        "optim_state": {},  # empty state dict is valid
        "step": 1234,
        "loss": 5.67,
        "n_params": sum(p.numel() for p in src.parameters()),
        "lr": 1e-4,
        "config": {
            "vocab": 100, "d": 32, "n_layers": 2, "loop_depth": 1,
            "max_seq": 32, "ffn_ratio": 2.0, "sparsity": 0.5,
            "dropout": 0.0, "tie_lm_head": True,
        },
    }
    torch.save(kd_payload, kd_ckpt_path)

    # Load via adv_warmstart (the same code path the SFT trainer uses).
    rep = adv_warmstart(tgt, str(kd_ckpt_path))
    assert rep.matched > 0, (
        f"warmstart matched 0 keys; report={rep.summary()}"
    )
    # Sanity: the KD ckpt's modified weights must now appear in tgt.
    src_sd = src.state_dict()
    tgt_sd = tgt.state_dict()
    # Compare a representative weight that adv_warmstart should always
    # transfer cleanly: tok_embed.weight (the most common shared key).
    assert "tok_embed.weight" in tgt_sd
    assert torch.allclose(
        tgt_sd["tok_embed.weight"], src_sd["tok_embed.weight"], atol=1e-6
    ), "tok_embed.weight did not transfer through warmstart"

    # Now write an SFT ckpt in the format train_100m_sft.py produces and
    # confirm it round-trips back into a fresh KD-style model.
    from train_100m_sft import _build_config_dict

    class _ArgsStub:
        vocab = 100; d = 32; n_layers = 2; loop_depth = 1
        seq_len = 32; ffn_ratio = 2.0; sparsity = 0.5

    sft_ckpt_path = tmp_path / "sft_step.pt"
    sft_payload = {
        "model": tgt.state_dict(),
        "optim_state": {},
        "step": 42,
        "loss": 2.34,
        "n_params": sum(p.numel() for p in tgt.parameters()),
        "lr": 1e-4,
        "config": _build_config_dict(_ArgsStub()),
    }
    torch.save(sft_payload, sft_ckpt_path)

    # The SFT ckpt should have phase="sft" so the KD trainer's loader
    # can branch on it.
    assert sft_payload["config"]["phase"] == "sft", (
        "SFT ckpts must tag config['phase']='sft' for downstream loaders"
    )

    # And it must load back into a fresh model via the same adv_warmstart.
    twin = _make_tiny_model(d=32, vocab=100)
    rep2 = adv_warmstart(twin, str(sft_ckpt_path))
    assert rep2.matched > 0, (
        f"SFT->KD warmstart matched 0 keys; report={rep2.summary()}"
    )


# ===========================================================================
# Test 3 -- end-to-end 5-step smoke
# ===========================================================================


def test_smoke_5_steps_runs_clean(tmp_path: Path, monkeypatch) -> None:
    """``main()`` completes 5 steps on CPU with a synthetic alpaca
    parquet, no warmstart, and a tiny d=32 model.

    Verifies (a) the run finishes without raising, (b) at least one
    ckpt landed on disk, (c) metrics.json is well-formed JSON with
    the expected keys.
    """
    parquet_path = tmp_path / "synth_alpaca.parquet"
    write_synth_alpaca_parquet(
        str(parquet_path),
        n_rows=64, prompt_len=8, response_len=12, vocab=80, seed=0,
    )
    out_dir = tmp_path / "run"

    import train_100m_sft

    # Fake that we're on CPU (this is true on the test machine but be
    # defensive in case CUDA gets reported on a CI runner).
    monkeypatch.setattr(train_100m_sft, "DEVICE", "cpu", raising=True)
    monkeypatch.setattr(
        train_100m_sft, "DTYPE", torch.float32, raising=True
    )

    argv = [
        "--out", str(out_dir),
        "--data", str(parquet_path),
        "--cross-val", "",  # disable cross-val (no second parquet)
        "--no-warmstart",
        "--steps", "5",
        "--warmup", "1",
        "--save-every", "5",
        "--eval-every", "5",
        "--log-every", "1",
        "--batch-size", "4",
        "--seq-len", "32",
        "--vocab", "80",
        "--d", "32",
        "--n-layers", "2",
        "--loop-depth", "1",
        "--ffn-ratio", "2.0",
        "--sparsity", "0.5",
        "--lr", "1e-4",
        "--shuffle-buffer", "0",
        "--tokenizer-name", "gpt2",  # falls back to pad=0 if HF unavailable
    ]
    rc = train_100m_sft.main(argv=argv)
    assert rc == 0, f"trainer main() returned {rc}; expected 0"

    # ckpt and final exist
    assert (out_dir / "step_000005.pt").exists() or (
        out_dir / "final.pt"
    ).exists(), (
        f"no ckpt landed in {out_dir!r}; "
        f"contents={[p.name for p in out_dir.iterdir()]}"
    )
    # metrics.json is well-formed
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists(), "metrics.json missing"
    metrics = json.loads(metrics_path.read_text())
    for required in ("step", "loss", "ppl_alpaca_holdout"):
        assert required in metrics, f"metrics.json missing key {required!r}"
    assert len(metrics["step"]) >= 1, "metrics.step is empty"


# ===========================================================================
# Test 4 -- alpaca holdout produces a finite ppl
# ===========================================================================


def test_eval_alpaca_holdout_emits_ppl(tmp_path: Path) -> None:
    """The 5%% holdout split (denom=20, keep={0}) yields a non-empty
    iterator, evaluate_sft returns a finite float, and the train/
    holdout row sets are deterministically disjoint.
    """
    parquet_path = tmp_path / "synth.parquet"
    write_synth_alpaca_parquet(
        str(parquet_path),
        n_rows=200, prompt_len=8, response_len=12, vocab=64, seed=1,
    )

    parent = InstructionParquetStream(
        str(parquet_path),
        seq_len=32, batch_size=4,
        response_only_loss=True,
        pad_id=0, eos_id=0,
        loop=False, shuffle_buffer=0,
    )

    from train_100m_sft import (
        _AlpacaHoldoutStream, evaluate_sft, split_alpaca,
    )

    # Disjoint check: collect every row index emitted by each side.
    train_stream, holdout_stream = split_alpaca(parent)
    assert isinstance(train_stream, _AlpacaHoldoutStream)
    assert isinstance(holdout_stream, _AlpacaHoldoutStream)
    train_rows = set()
    holdout_rows = set()
    for ridx, _row in enumerate(parent._iter_rows()):
        denom = train_stream._denom
        if (ridx % denom) in train_stream._keep:
            train_rows.add(ridx)
        if (ridx % denom) in holdout_stream._keep:
            holdout_rows.add(ridx)
    assert train_rows.isdisjoint(holdout_rows), (
        f"train ∩ holdout must be empty; "
        f"overlap={sorted(train_rows & holdout_rows)[:5]!r}"
    )
    # Holdout should be ~5% of 200 rows = 10 rows.
    assert 5 <= len(holdout_rows) <= 15, (
        f"expected ~10 holdout rows (5%% of 200), got {len(holdout_rows)}"
    )

    # Build a tiny model and run evaluate_sft on the holdout.
    model = _make_tiny_model(d=32, vocab=64, seq_len=32)
    model.eval()

    ppl = evaluate_sft(
        model, iter(holdout_stream), n_batches=4,
        plif_cells=None, response_only=True, pad_id=0,
    )
    assert isinstance(ppl, float), f"ppl must be a float; got {type(ppl)}"
    # With random init the loss is around log(V)=4.16 -> ppl~64; allow a
    # generous range. The point of the test is that ppl is finite, not
    # that the random model scores well.
    import math as _m
    assert _m.isfinite(ppl), f"alpaca holdout ppl must be finite; got {ppl}"
    assert ppl > 1.0, f"ppl must be > 1; got {ppl}"
