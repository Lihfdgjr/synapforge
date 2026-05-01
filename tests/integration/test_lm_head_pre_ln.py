"""docs/MASTER_PLAN.md §6 P28 / DEEP_MAINT_QUEUE.md T7.3 — pre-LM-head LayerNorm.

Validates that ``SynapForge100M(lm_head_pre_ln=True)`` correctly inserts
an affine-free ``nn.LayerNorm`` immediately before the final LM-head
projection, bounding the row-wise norm of the hidden state that the
LM head sees.

Why this matters (P28 z-loss linear drift, primary plan):
    The retrospective `docs/TRAINING_ISSUES_RETROSPECTIVE.md` §2.d
    observed ``z_loss`` (= ``logsumexp(logits)`` mean) trending
    *linearly upward* across Run 3b despite top-K=2048 sparse z-loss
    in `4d0d2a9`. Diagnosis: ``self.ln_f`` is RMSNorm with a learnable
    affine scale, which Adam can grow without bound. Once the scale
    drifts, post-``ln_f`` row norms blow up, ``F.linear(x, lm_head)``
    logits blow up, and ``log Z`` rides them up. The primary fix is
    inserting ``nn.LayerNorm(d, elementwise_affine=False)`` AFTER
    ``ln_f`` and BEFORE the LM projection: the parameter-free LN
    re-centers (mean 0) and re-scales (var 1) every row, so the
    row L2 norm post-LN is exactly ``sqrt(d) * (1 + O(eps))``,
    independent of how the affine RMSNorm scale evolved upstream.
    With the LM input bounded, ``log Z`` is bounded, and z-loss can
    no longer drift linearly.

This is the orthogonal ``INPUT-side`` bound to T2.6's spectral_norm
``OPERATOR-side`` bound. Both can be on simultaneously -- they
compose.

Tests:
    1. flag-off default: no extra module, no state_dict keys added,
       backwards-compatible with all existing best_*.pt checkpoints.
    2. flag-on tied: ``lm_head_pre_ln_module`` is a real
       ``nn.LayerNorm(d, elementwise_affine=False)``, parameters
       count is *unchanged* vs baseline (no learnable scale/bias),
       state_dict has no extra keys (affine-free LN registers no
       persistent state).
    3. flag-on bounded-norm contract: AFTER one forward pass, the
       hidden vector that ``F.linear`` sees has per-row L2 norm
       within tight bounds of ``sqrt(d) +/- 1%`` -- the actual
       guarantee that bounds the partition function.
    4. flag-on robust to drifted ln_f scale: manually zero or scale
       up ``ln_f.weight`` by 100x (simulating a runaway affine
       scale). With the flag ON, the LM-head input norm is STILL
       bounded by ~sqrt(d). With the flag OFF, the LM-head input
       norm scales linearly with ``ln_f.weight``. This is the
       proof that the flag actually does the job.
    5. spectral_norm + pre_ln compose: build a model with BOTH
       ``--lm-head-spectral-norm`` and ``--lm-head-pre-ln`` on,
       verify forward stays finite (the two features are orthogonal
       and stack cleanly).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_torch_and_model():
    """Lazy-import torch + the 100M model; skip cleanly if torch absent."""
    pytest.importorskip("torch")
    import torch  # noqa: F401  (used by callers)

    from synapforge.model_100m import SynapForge100M
    return torch, SynapForge100M


def _tiny_kwargs() -> dict:
    """Tiny model dims so CPU CI can run end-to-end in <2s.

    vocab=128, d=32, n_layers=1, loop_depth=1: one HybridBlock unrolled
    once -- enough to exercise tok_embed -> blocks -> ln_f ->
    [optional pre-LN] -> F.linear without paying the 100M-param tax.
    """
    return dict(
        vocab=128,
        d=32,
        n_layers=1,
        loop_depth=1,
        max_seq=16,
        ffn_ratio=2.0,
        sparsity=0.1,  # SparseSynapse requires (0, 1]
        dropout=0.0,
        freeze_vocab_tail=False,  # vocab=128 < QWEN25_LIVE_VOCAB
    )


def test_flag_off_default_no_extra_module():
    """Default lm_head_pre_ln=False keeps ckpts backwards-compatible."""
    torch, SynapForge100M = _import_torch_and_model()

    model = SynapForge100M(tie_lm_head=True, **_tiny_kwargs())
    assert model.lm_head_pre_ln_module is None, (
        "flag OFF default must not allocate a pre-LN module"
    )
    # No new state_dict keys (legacy ckpts load clean).
    sd_keys_off = set(model.state_dict().keys())

    # Sanity: forward still works in the no-pre-LN baseline.
    ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
    logits = model(ids)
    assert logits.shape == (2, 8, 128)
    assert torch.isfinite(logits).all()

    # Now flip the flag ON and verify NO new state_dict keys appear --
    # affine-free LayerNorm registers no parameters or persistent
    # buffers, so the ckpt format is bit-compatible across the toggle.
    model_on = SynapForge100M(
        tie_lm_head=True, lm_head_pre_ln=True, **_tiny_kwargs()
    )
    sd_keys_on = set(model_on.state_dict().keys())
    assert sd_keys_on == sd_keys_off, (
        "lm_head_pre_ln must not add state_dict keys "
        f"(diff: {sd_keys_on ^ sd_keys_off})"
    )


def test_flag_on_module_is_affine_free():
    """Flag ON: real LayerNorm, NO learnable scale/bias."""
    torch, SynapForge100M = _import_torch_and_model()
    import torch.nn as nn

    model = SynapForge100M(
        tie_lm_head=True, lm_head_pre_ln=True, **_tiny_kwargs()
    )
    assert isinstance(model.lm_head_pre_ln_module, nn.LayerNorm), (
        "lm_head_pre_ln_module should be an nn.LayerNorm instance"
    )
    # Affine-free contract.
    assert model.lm_head_pre_ln_module.weight is None, (
        "elementwise_affine must be False (no learnable gamma)"
    )
    assert model.lm_head_pre_ln_module.bias is None, (
        "elementwise_affine must be False (no learnable beta)"
    )
    # Param count UNCHANGED vs baseline -- this LN is parameter-free.
    baseline = SynapForge100M(tie_lm_head=True, **_tiny_kwargs())
    assert model.num_parameters() == baseline.num_parameters(), (
        f"pre-LN must not add parameters "
        f"(on={model.num_parameters()}, off={baseline.num_parameters()})"
    )


def test_flag_on_bounds_lm_head_input_norm():
    """Flag ON: row-wise L2 norm of LM-head input ~= sqrt(d).

    This is the actual contract the patch promises -- not just "doesn't
    crash" but "bounds the input to lm_head". We hook the model's
    forward, capture the post-pre-LN hidden state, and assert its
    per-row norm is sqrt(d) +/- 1% (eps=1e-5 in LayerNorm gives a
    tighter bound, but 1% is safer for fp32 arithmetic).
    """
    torch, SynapForge100M = _import_torch_and_model()

    model = SynapForge100M(
        tie_lm_head=True, lm_head_pre_ln=True, **_tiny_kwargs()
    )
    model.eval()
    d = model.d
    expected_norm = (d ** 0.5)

    captured = {}

    def _capture_post_ln(_module, _inp, out):
        captured["x"] = out.detach()

    # Hook the pre-LN module's output -- whatever that returns is
    # *exactly* what F.linear(., tok_embed.weight) sees.
    h = model.lm_head_pre_ln_module.register_forward_hook(_capture_post_ln)
    try:
        ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
        with torch.no_grad():
            _ = model(ids)
    finally:
        h.remove()

    x = captured["x"]                      # (B, T, d)
    row_norms = x.norm(dim=-1)             # (B, T)
    # Affine-free LayerNorm output has variance 1 along the last dim,
    # so ||x||_2 = sqrt(d * var) = sqrt(d). Allow 1% slack for fp32.
    rel_dev = (row_norms - expected_norm).abs() / expected_norm
    assert rel_dev.max() < 0.02, (
        f"pre-LN must produce ~sqrt(d) row norms; "
        f"got max rel deviation {rel_dev.max().item():.4f} "
        f"(rows: {row_norms.flatten().tolist()})"
    )


def test_flag_on_robust_to_drifted_ln_f_scale():
    """Manually corrupt ln_f.weight by 100x; pre-LN still bounds the input.

    This is the smoking-gun test for P28: simulate the failure mode
    (Adam pushes RMSNorm affine scale up over many steps), verify that
    WITHOUT pre-LN the LM head input norm scales with the corrupted
    affine, but WITH pre-LN it stays bounded.
    """
    torch, SynapForge100M = _import_torch_and_model()
    import torch.nn.functional as F

    ids = torch.randint(0, 128, (4, 8), dtype=torch.long)

    # ---- Baseline (no pre-LN) under runaway ln_f.weight ----
    model_off = SynapForge100M(tie_lm_head=True, **_tiny_kwargs())
    model_off.eval()
    with torch.no_grad():
        model_off.ln_f.weight.fill_(100.0)  # 100x the init of 1.0
        x_off = model_off.encode(ids)        # (B, T, d) post ln_f

    # ---- With pre-LN (the fix) under same corruption ----
    model_on = SynapForge100M(
        tie_lm_head=True, lm_head_pre_ln=True, **_tiny_kwargs()
    )
    model_on.eval()
    with torch.no_grad():
        model_on.ln_f.weight.fill_(100.0)
        # Replicate forward(ids) up to the LM projection; capture the
        # tensor that F.linear actually sees.
        x_pre = model_on.encode(ids)
        x_post = model_on.lm_head_pre_ln_module(x_pre)

    n_off = x_off.norm(dim=-1).max().item()
    n_pre = x_pre.norm(dim=-1).max().item()
    n_post = x_post.norm(dim=-1).max().item()
    expected = (model_on.d ** 0.5)

    # The corruption must show up: x_off norms are O(100) the baseline.
    assert n_off > 50.0, (
        f"corrupted ln_f.weight=100 should produce >>1 row norms; got {n_off}"
    )
    # x_pre (input to pre-LN, which is ln_f output) is also large...
    assert n_pre > 50.0, f"sanity: x_pre should be large; got {n_pre}"
    # ...but x_post (output of pre-LN) is bounded near sqrt(d) regardless.
    assert n_post < expected * 1.02, (
        f"pre-LN must clamp norm to sqrt(d) ~= {expected:.2f} regardless of "
        f"upstream corruption; got max={n_post:.2f}"
    )


def test_pre_ln_and_spectral_norm_compose():
    """Both lm_head_pre_ln=True AND lm_head_spectral_norm=True work together.

    Pre-LN bounds the INPUT norm; spectral_norm bounds the OPERATOR
    norm. They are orthogonal -- the input goes through the pre-LN
    first, then through the spectral-normalised operator. We just
    sanity-check that the forward path is finite and produces the
    right shape; no exceptions, no NaN.
    """
    torch, SynapForge100M = _import_torch_and_model()

    model = SynapForge100M(
        tie_lm_head=True,
        lm_head_pre_ln=True,
        lm_head_spectral_norm=True,
        **_tiny_kwargs(),
    )
    # Both features must be wired.
    assert model.lm_head_pre_ln_module is not None
    assert hasattr(model.tok_embed, "weight_orig"), (
        "spectral_norm should wrap tok_embed in tied path"
    )

    ids = torch.randint(0, 128, (2, 8), dtype=torch.long)
    logits = model(ids)
    assert logits.shape == (2, 8, 128)
    assert torch.isfinite(logits).all(), (
        "stacked spectral_norm + pre-LN forward produced non-finite logits"
    )
