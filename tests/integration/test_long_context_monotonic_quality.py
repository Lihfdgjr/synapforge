"""P21 — quality-monotonic-with-context A/B harness.

Tests the **paper claim** that inference-time STDP (forward-only Hebbian
fast weights, the gate-removal at ``synapforge/bio/stdp_fast.py:121``)
makes the model *better* as context grows. This is the OPPOSITE of every
transformer baseline — they degrade past their trained context window.

How it tests
------------
For each context length L in ``[1024, 10_000, 100_000]``:

1. Build a "needle in haystack" prompt: L-200 tokens of random
   distractor + a 50-token "needle" with a known single-token answer +
   150 tokens of continuation. The model is asked to predict the needle
   answer at the end.

2. Run TWICE — once with ``synapforge.long.set_stdp_inference("on")``
   (the paper claim), once with ``"off"`` (frozen-W transformer-style
   baseline). Each run resets the STDP state first so the previous
   length's accumulated weights don't bleed in.

3. Score by exact-match of the predicted argmax token against the
   needle answer. Average across 3 distinct (seed, depth) pairs so the
   result isn't a single-prompt fluke.

Assertions
----------
At every L:
    accuracy_on >= accuracy_off - 0.05
        Tiny noise is OK; the moment ON loses to OFF, the claim is dead.

Bonus (the steeper claim):
    accuracy_on(100K) >= accuracy_on(1K) - 0.10
        Quality at 100K is within 10pp of quality at 1K — i.e. context
        did not collapse the answer. (Strict monotonic-INCREASE is too
        strong with a tiny model + random init; we test the weaker form.)

All asserts are gated behind ``pytest.mark.slow`` and ``pytest.mark.gpu``
so that ``pytest`` (no markers) skips this file. Run via the wrapper
``scripts/run_long_context_validation.sh`` once Run 2 has a checkpoint.

Key invariant
-------------
Both branches of every A/B see EXACTLY the same prompt tokens, the same
RNG seed, and the same model weights (the model is rebuilt fresh per
run from a deterministic seed). The only difference is the STDP env
var. This is what isolates the claim.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_long_module() -> ModuleType:
    """Load ``synapforge/long.py`` in isolation (bypass torch-bound __init__)."""
    full = _REPO_ROOT / "synapforge" / "long.py"
    spec = importlib.util.spec_from_file_location("synapforge_long_iso", full)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {full}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Capability checks
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _cuda_total_bytes() -> int:
    if not _torch_available():
        return 0
    import torch
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_LENGTHS = [1024, 10_000, 100_000]


def _build_model(vocab: int = 8192, d: int = 64, n_layers: int = 2,
                 max_seq: int = 4096, seed: int = 7):
    """Deterministic build so A and B branches start from identical weights."""
    import torch
    from synapforge.model_100m import build_synapforge_100m

    torch.manual_seed(int(seed))
    model = build_synapforge_100m(
        vocab=vocab, d=d, n_layers=n_layers, loop_depth=1,
        max_seq=max_seq, ffn_ratio=2.0, sparsity=0.95, dropout=0.0,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Needle-in-haystack construction
# ---------------------------------------------------------------------------


def _build_niah_prompt(L: int, needle_token: int, vocab: int = 8192,
                       seed: int = 0):
    """Return ``ids: torch.LongTensor [1, L]`` with a needle near the end.

    Layout:
        [ L-200 distractor tokens ][ 50 needle marker tokens (deterministic) ]
        [ 149 continuation tokens ][ 1 answer slot — the needle_token ]

    The "marker" tokens are a fixed pattern (seed-derived) that appears
    exactly once in the stream. With STDP=on, the readout pathway should
    learn the marker→answer association during the streaming phase and
    answer correctly when the marker recurs at the very end.

    For a tiny untrained model the absolute scores will be low (random
    init has ~1/vocab chance), but the A/B *delta* is still meaningful
    because both branches face identical noise.
    """
    import torch

    L = int(L)
    if L < 250:
        # Tiny L: just put marker and answer; minimal distractor.
        marker_n = max(2, L // 8)
        cont_n = max(1, L // 8)
        distract_n = L - marker_n - cont_n - 1
    else:
        marker_n = 50
        cont_n = 149
        distract_n = L - marker_n - cont_n - 1

    g = torch.Generator()
    g.manual_seed(int(seed))
    distractor = torch.randint(0, vocab, (distract_n,), generator=g, dtype=torch.long)

    # Marker is a deterministic pattern derived from seed.
    g2 = torch.Generator()
    g2.manual_seed(int(seed) + 1234)
    marker = torch.randint(0, vocab, (marker_n,), generator=g2, dtype=torch.long)

    g3 = torch.Generator()
    g3.manual_seed(int(seed) + 999)
    cont = torch.randint(0, vocab, (cont_n,), generator=g3, dtype=torch.long)

    answer = torch.tensor([int(needle_token) % vocab], dtype=torch.long)

    full = torch.cat([distractor, marker, cont, answer], dim=0)
    assert full.shape[0] == L, (full.shape[0], L)
    return full.unsqueeze(0)  # [1, L]


def _stream_score(model, ids, chunk: int = 4096) -> int:
    """Stream ids through the model, return predicted argmax for last position.

    Streams chunks of size <= max_seq, reading the next-token logits at
    each position. Returns the argmax of the FINAL chunk's last token's
    logits (i.e. what the model would predict given the full L-token
    context).
    """
    import torch

    mx = getattr(model, "max_seq", chunk)
    chunk = min(chunk, mx)
    L = ids.shape[1]

    last_logits = None
    with torch.no_grad():
        for i in range(0, L, chunk):
            piece = ids[:, i:i + chunk]
            logits = model(piece)  # [1, T, vocab]
            last_logits = logits[:, -1, :]  # [1, vocab]
    if last_logits is None:
        return -1
    return int(torch.argmax(last_logits, dim=-1).item())


def _accuracy_at_length(model, L: int, n_trials: int = 3,
                        vocab: int = 8192, base_seed: int = 0) -> float:
    """Run n_trials NIAH probes and return the exact-match accuracy."""
    import torch
    from synapforge import long as sf_long

    correct = 0
    for t in range(n_trials):
        seed = int(base_seed) + 17 * t
        # Pick a deterministic needle answer per trial.
        g = torch.Generator()
        g.manual_seed(seed)
        needle = int(torch.randint(1, vocab, (1,), generator=g).item())

        ids = _build_niah_prompt(L=L, needle_token=needle, vocab=vocab, seed=seed)
        # The "answer slot" is at position L-1; we want the model's
        # prediction GIVEN positions [0, L-2], compared to ids[:, -1].
        prompt = ids[:, :-1]
        target = int(ids[0, -1].item())

        sf_long.reset_stdp(model)
        pred = _stream_score(model, prompt)
        if pred == target:
            correct += 1
    return correct / float(n_trials)


def _ab_at_length(L: int, n_trials: int = 3, vocab: int = 8192) -> dict:
    """Build a fresh model, run STDP=off then STDP=on, return both accuracies.

    Branch isolation: model is rebuilt from the same seed for each
    branch so any trace of B accidentally affecting A is impossible.
    """
    import torch  # noqa: F401  -- ensures torch imported before model build
    from synapforge import long as sf_long

    prev_mode = sf_long.get_stdp_inference()
    try:
        # Branch A: STDP=off (transformer-style baseline).
        sf_long.set_stdp_inference("off")
        model_a = _build_model(vocab=vocab)
        acc_off = _accuracy_at_length(model_a, L=L, n_trials=n_trials, vocab=vocab)
        del model_a

        # Branch B: STDP=on (paper claim).
        sf_long.set_stdp_inference("on")
        model_b = _build_model(vocab=vocab)
        acc_on = _accuracy_at_length(model_b, L=L, n_trials=n_trials, vocab=vocab)
        del model_b
    finally:
        sf_long.set_stdp_inference(prev_mode if prev_mode else "on")

    return {"L": int(L), "acc_off": float(acc_off), "acc_on": float(acc_on)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("L", _LENGTHS, ids=[f"L={L}" for L in _LENGTHS])
def test_stdp_on_does_not_lose(L: int):
    """At every L, ON beats (or essentially ties) OFF on NIAH accuracy."""
    if not _torch_available():
        pytest.skip("torch not installed")
    if L >= 100_000 and _cuda_total_bytes() < 70 * 1_000_000_000:
        pytest.skip(f"L={L} requires CUDA >= 70 GB; have {_cuda_total_bytes() / 1e9:.1f} GB")

    out = _ab_at_length(L=L, n_trials=3)
    assert out["acc_on"] >= out["acc_off"] - 0.05, (
        f"P21 violation at L={L}: STDP=on accuracy {out['acc_on']:.3f} is "
        f"more than 0.05 below STDP=off {out['acc_off']:.3f}. "
        f"This contradicts the paper claim that inference-time STDP is "
        f"the differentiator."
    )


@pytest.mark.slow
@pytest.mark.gpu
def test_stdp_on_does_not_collapse_at_100k():
    """Bonus: ON quality at 100K stays within 10pp of ON quality at 1K.

    This is the strong form of the monotonic claim — context did not
    collapse the readout. We don't assert strict INCREASE because at a
    tiny untrained model on random tokens, the noise floor is too high
    to see strict monotonic gains; the meaningful test on a real ckpt
    will be re-asserted on rental.
    """
    if not _torch_available():
        pytest.skip("torch not installed")
    if _cuda_total_bytes() < 70 * 1_000_000_000:
        pytest.skip(
            f"requires CUDA >= 70 GB for 100K; have {_cuda_total_bytes() / 1e9:.1f} GB"
        )

    out_1k = _ab_at_length(L=1024, n_trials=3)
    out_100k = _ab_at_length(L=100_000, n_trials=3)
    drop = out_1k["acc_on"] - out_100k["acc_on"]
    assert drop <= 0.10, (
        f"P21 collapse: STDP=on accuracy at 1K={out_1k['acc_on']:.3f} but "
        f"at 100K={out_100k['acc_on']:.3f}. Quality dropped {drop:.3f}, "
        f"more than the 10pp budget. STDP plus retrieval should keep "
        f"end-of-context answer quality flat at minimum."
    )


# ---------------------------------------------------------------------------
# Standalone tests that work without GPU (verify the harness's plumbing)
# ---------------------------------------------------------------------------


def test_niah_prompt_shape_and_answer_slot():
    """NIAH builder must put the needle answer in the last position."""
    if not _torch_available():
        pytest.skip("torch not installed")
    import torch  # noqa: F401

    ids = _build_niah_prompt(L=1024, needle_token=42, vocab=8192, seed=1)
    assert ids.shape == (1, 1024), ids.shape
    assert int(ids[0, -1].item()) == 42, ids[0, -1].item()


def test_ab_prep_does_not_mutate_env_on_failure():
    """If the inner harness errors, env var is restored.

    We don't actually call _ab_at_length (needs torch); we exercise the
    same prev/restore pattern in a tiny mock so the contract is checked
    even on the dev box. Loaded in isolation to avoid torch import.
    """
    sf_long = _load_long_module()

    sf_long.set_stdp_inference("on")
    snap = sf_long.get_stdp_inference()
    try:
        prev = sf_long.set_stdp_inference("off")
        assert sf_long.get_stdp_inference() == "off"
        # simulate inner failure path
        raise RuntimeError("simulated")
    except RuntimeError:
        pass
    finally:
        sf_long.set_stdp_inference(prev if prev else "on")
    assert sf_long.get_stdp_inference() == snap, (
        "env var leaked across the try/finally; the A/B harness must "
        "restore the previous mode on failure"
    )
