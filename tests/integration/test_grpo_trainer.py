"""DEEP_MAINT_QUEUE T9.5 -- GRPO RL trainer + verifiers tests.

Pinned contracts (each test = one assertion class):

  test_grpo_loss_computation_correct
      Manually-derived gradient must match what ``grpo_loss`` produces
      for a fixed (rewards, log_pi, log_pi_ref) tuple. This is the
      single load-bearing correctness test for the policy gradient.

  test_sympy_verifier_correct_for_simple_math
      The sympy verifier must give reward=1.0 for "2+2 = 4" and
      reward=0.0 for "2+2 = 5". Skipped if sympy isn't installed.

  test_ast_verifier_safe_no_exec
      The AST verifier MUST refuse to score any rollout containing
      ``exec``, ``eval``, ``os.system``, ``__import__``, etc. (returns
      reward=0.0). This is the *security boundary* test -- the verifier
      must NEVER execute model output.

  test_kl_penalty_prevents_drift
      After 10 GRPO steps with kl_beta=0.5, the policy's parameters
      must stay within a bounded L2 distance of the reference. With
      kl_beta=0.0 the same setup MUST drift further (sanity check that
      the KL term is doing real work).

All tests are CPU-only and finish in < 5 s on a typical laptop. None
depend on the 100M model weights, the parquet data, or a GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Whole module needs torch (verifiers + GRPO loss).
torch = pytest.importorskip("torch")

import torch.nn as nn

# Repo root on path so ``import synapforge`` works regardless of pytest cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synapforge.training.grpo import (  # noqa: E402
    ast_verifier,
    compute_advantages,
    extract_final_number,
    freeze_reference_policy,
    get_verifier,
    grpo_loss,
    sample_rollouts_mock,
    sympy_verifier,
)


# ===========================================================================
# Test 1 -- GRPO loss matches manually-derived gradient
# ===========================================================================


def test_grpo_loss_computation_correct() -> None:
    """The autograd gradient of ``grpo_loss`` must match a hand-derived value.

    For ``log_pi`` 1D with N rollouts (on-policy, ratio == 1):
        advantage_i = (r_i - mean(r)) / (std(r) + eps)     [detached]
        pol_loss    = -mean(advantage * log_pi)
        kl_per_i    = exp(-(log_pi_i - log_pi_ref_i)) + (log_pi_i - log_pi_ref_i) - 1
        kl_loss     = mean(kl_per_i)
        loss        = pol_loss + beta * kl_loss

    Hand derivation of d loss / d log_pi_i:
        from pol_loss   : -advantage_i / N
        from kl_loss    : beta * (1 - exp(-(log_pi_i - log_pi_ref_i))) / N

    We pin this exact derivation because policy-gradient sign errors
    are the most common silent bug in RL implementations.
    """
    torch.manual_seed(0)

    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    log_pi = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    log_pi_ref = torch.tensor([-1.5, -2.5, -3.5, -4.5])
    kl_beta = 0.1

    loss, stats = grpo_loss(
        log_pi=log_pi,
        log_pi_old=None,
        log_pi_ref=log_pi_ref,
        rewards=rewards,
        clip_eps=0.2,
        kl_beta=kl_beta,
    )
    loss.backward()

    # Hand derivation (Schulman k3 KL):
    advantages = compute_advantages(rewards).detach()
    n = rewards.numel()
    r = (log_pi.detach() - log_pi_ref.detach())
    kl_grad = (1.0 - torch.exp(-r)) / n
    expected_grad = (-advantages / n) + kl_beta * kl_grad
    actual_grad = log_pi.grad

    assert actual_grad is not None, "log_pi must have gradient"
    assert torch.allclose(actual_grad, expected_grad, atol=1e-6), (
        f"GRPO gradient mismatch:\n"
        f"  expected: {expected_grad.tolist()}\n"
        f"  actual:   {actual_grad.tolist()}"
    )

    # Stats sanity.
    assert stats.n_rollouts == 4
    assert 0.0 <= stats.reward_mean <= 1.0
    assert stats.advantage_abs_mean > 0  # advantages are non-trivial.
    # Schulman k3 estimator is non-negative.
    assert stats.kl_loss >= -1e-6, f"KL must be non-negative, got {stats.kl_loss}"


def test_grpo_loss_zero_when_uniform_rewards() -> None:
    """When all rewards are equal, advantages are zero so policy loss is zero.

    Only the KL term should drive the gradient -- this is the contract
    that lets early-training "all wrong" or "all right" batches not
    cause runaway updates.
    """
    rewards = torch.tensor([0.5] * 4, dtype=torch.float32)
    log_pi = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    log_pi_ref = log_pi.detach().clone()
    loss, stats = grpo_loss(
        log_pi=log_pi,
        log_pi_old=None,
        log_pi_ref=log_pi_ref,
        rewards=rewards,
        kl_beta=0.0,
    )
    assert abs(stats.pol_loss) < 1e-6
    assert abs(stats.advantage_abs_mean) < 1e-6


def test_compute_advantages_normalises_correctly() -> None:
    """Advantage = (r - mean) / std, sanity-check the formula on known input."""
    r = torch.tensor([0.0, 1.0, 0.0, 1.0])
    a = compute_advantages(r)
    # mean=0.5, std=0.5 (population), so advantages are [-1, 1, -1, 1].
    assert torch.allclose(a, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-5)


# ===========================================================================
# Test 2 -- sympy verifier rewards correctly
# ===========================================================================


def test_sympy_verifier_correct_for_simple_math() -> None:
    """The sympy verifier must give the right reward for trivial math.

    Pinned cases:
      * "2+2 = 4" with gt=4 -> reward 1.0
      * "2+2 = 5" with gt=4 -> reward 0.0
      * "Answer: 42" with gt=42 -> reward 1.0
      * "#### 7" with gt=7 -> reward 1.0 (canonical GSM8K format)
      * "no number here" with gt=4 -> reward 0.0 (no extractable answer)
    """
    pytest.importorskip("sympy")

    # Canonical GSM8K format
    assert sympy_verifier("Hmm let me think.\n#### 7", 7) == 1.0
    assert sympy_verifier("Hmm let me think.\n#### 8", 7) == 0.0

    # Inline equation
    assert sympy_verifier("So 2+2 = 4 hence the answer.\n#### 4", 4) == 1.0
    assert sympy_verifier("So 2+2 = 5\n#### 5", 4) == 0.0

    # Word-form
    assert sympy_verifier("Answer: 42", 42) == 1.0

    # No extractable numeric answer
    assert sympy_verifier("I don't know.", 4) == 0.0

    # Empty rollout
    assert sympy_verifier("", 4) == 0.0


def test_sympy_verifier_handles_negatives_and_decimals() -> None:
    """Edge cases: negative integers, decimal answers."""
    pytest.importorskip("sympy")
    assert sympy_verifier("...so the answer is\n#### -7", -7) == 1.0
    assert sympy_verifier("...so the answer is\n#### -7", 7) == 0.0
    # Decimal final answer
    assert sympy_verifier("answer: 3.14", 3.14) == 1.0


def test_sympy_verifier_via_registry() -> None:
    """``get_verifier('sympy')`` must round-trip to the same callable."""
    pytest.importorskip("sympy")
    fn = get_verifier("sympy")
    assert fn("Answer: 9", 9) == 1.0


def test_extract_final_number_patterns() -> None:
    """Extractor must handle GSM8K, answer:, \\boxed{}, and end-of-line ``=``."""
    assert extract_final_number("blah\n#### 42") == 42.0
    assert extract_final_number("answer: 42") == 42.0
    assert extract_final_number("ANSWER = 42") == 42.0
    assert extract_final_number("\\boxed{42}") == 42.0
    assert extract_final_number("the result is = 42.") == 42.0
    assert extract_final_number("foo bar 42") == 42.0  # last-resort path
    assert extract_final_number("") is None


# ===========================================================================
# Test 3 -- AST verifier rejects unsafe rollouts WITHOUT executing them
# ===========================================================================


def test_ast_verifier_safe_no_exec() -> None:
    """The AST verifier MUST refuse to score any rollout with unsafe code.

    Security boundary: the verifier MUST NOT call ``exec``/``eval`` on
    model output. This test pins that contract by feeding it a series
    of *malicious* rollouts and asserting reward=0.0 for every one.

    A failure here means a model could emit ``__import__('os').system(
    'rm -rf')`` and get rewarded for it -- catastrophic.
    """
    # Forbidden bare names
    for code in [
        "exec('print(1)')",
        "eval('1+1')",
        "compile('x', '<x>', 'exec')",
        "open('/etc/passwd')",
        "input()",
        "__import__('os')",
    ]:
        assert ast_verifier(code, 4) == 0.0, f"unsafe accepted: {code!r}"

    # Forbidden module imports
    for code in [
        "import os",
        "import subprocess",
        "from os import system",
        "from subprocess import call",
        "import shutil; shutil.rmtree('/')",
        "import socket",
        "import ctypes",
        "import pickle",
    ]:
        assert ast_verifier(code, 4) == 0.0, f"unsafe import accepted: {code!r}"

    # Forbidden attribute access (sandbox-escape patterns)
    for code in [
        "().__class__.__bases__[0].__subclasses__()",
        "x = ().__class__\nprint(x)",
        "y = open  # alias",
    ]:
        assert ast_verifier(code, 4) == 0.0, f"sandbox-escape accepted: {code!r}"

    # getattr with forbidden string
    assert ast_verifier("getattr(x, 'exec')(...)", 4) == 0.0

    # Syntactically-broken code -> reward 0 (don't crash the trainer either)
    assert ast_verifier("def f(:", 4) == 0.0
    assert ast_verifier("...nonsense...", 4) == 0.0

    # Safe code with correct numeric answer in a comment -> reward 1.0
    safe_correct = "x = 2 + 2\n# answer: 4"
    assert ast_verifier(safe_correct, 4) == 1.0

    # Safe code with WRONG answer -> reward 0.0
    safe_wrong = "x = 2 + 2\n# answer: 5"
    assert ast_verifier(safe_wrong, 4) == 0.0


def test_ast_verifier_strips_code_fences() -> None:
    """Realistic LLM rollouts wrap code in `````python ... `````; verifier must handle that."""
    rollout = "```python\nx = 1 + 1\n# answer: 2\n```"
    assert ast_verifier(rollout, 2) == 1.0


def test_ast_verifier_via_registry() -> None:
    """``get_verifier('ast')`` must round-trip and refuse exec-tagged rollouts."""
    fn = get_verifier("ast")
    assert fn("# answer: 7\nx = 7", 7) == 1.0
    assert fn("exec('boom')", 7) == 0.0


def test_get_verifier_unknown_raises() -> None:
    """Unknown verifier name -> ValueError so trainers fail fast at startup."""
    with pytest.raises(ValueError):
        get_verifier("not-a-real-verifier")


# ===========================================================================
# Test 4 -- KL penalty prevents catastrophic drift
# ===========================================================================


class _ToyPolicy(nn.Module):
    """Tiny LM-style "policy" that emits a per-rollout log-prob.

    We don't need a real LM here. The test runs GRPO updates on this
    toy and measures L2 drift of its parameters from a frozen-copy
    reference -- with kl_beta high, drift is bounded; with kl_beta=0,
    drift grows without bound.

    Architecture: a single linear ``W: dim -> vocab`` projecting
    hidden states to vocab logits, followed by gather of the
    log-softmax of the chosen target token. This gives a properly
    *normalised* log-probability per rollout (between -log(vocab) and
    0) -- matching how a real LM produces its action log-probs and
    keeping the toy test's numerical regime close to the real trainer.
    """

    def __init__(self, dim: int = 8, vocab: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, vocab, bias=False)
        self.vocab = vocab

    def log_prob_seq(
        self, hidden: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return ``(B,)`` log-prob of ``targets`` given ``hidden``.

        ``hidden``  shape ``(B, dim)``.
        ``targets`` shape ``(B,)`` of integer token IDs.
        """
        logits = self.linear(hidden)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _l2_param_drift(model_a: nn.Module, model_b: nn.Module) -> float:
    """L2 distance between two models' flattened parameter vectors."""
    sa = torch.cat([p.detach().flatten() for p in model_a.parameters()])
    sb = torch.cat([p.detach().flatten() for p in model_b.parameters()])
    return float((sa - sb).norm().item())


def _run_n_grpo_steps(
    model: nn.Module, ref_model: nn.Module, *, kl_beta: float, n_steps: int
) -> None:
    """Run ``n_steps`` of toy GRPO with fixed rewards + random hidden inputs.

    LR is small (0.05) to keep the regime in the linear-update zone where
    KL has its expected stabilising effect (large LR + per-step KL gradient
    can over-correct and blow up either way -- not what we're testing).
    """
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    torch.manual_seed(7)
    for _ in range(n_steps):
        hidden = torch.randn(8, 8)
        targets = torch.randint(0, model.vocab, (8,))
        # Fake rewards that have non-trivial advantage variance.
        rewards = torch.tensor(
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32
        )
        log_pi = model.log_prob_seq(hidden, targets)
        with torch.no_grad():
            log_pi_ref = ref_model.log_prob_seq(hidden, targets)
        loss, _ = grpo_loss(
            log_pi=log_pi,
            log_pi_old=None,
            log_pi_ref=log_pi_ref,
            rewards=rewards,
            kl_beta=kl_beta,
        )
        optim.zero_grad()
        loss.backward()
        optim.step()


def test_kl_penalty_prevents_drift() -> None:
    """High kl_beta must keep the policy near the reference; kl_beta=0 must not.

    Sanity contract:
      drift(kl_beta=0.5) < drift(kl_beta=0.0)
    after 10 RL steps from the same init. With KL=0 the policy is free
    to chase advantages anywhere in parameter space; with KL>0 it pays
    a per-step penalty proportional to ``log_pi - log_pi_ref``, so
    drift stays bounded.

    Without this contract the RL trainer can erase chat ability
    within a few hundred steps -- pinning it via test prevents
    accidentally dropping the KL term in a future refactor.
    """
    torch.manual_seed(11)
    model_kl = _ToyPolicy()
    ref_kl = freeze_reference_policy(model_kl)
    _run_n_grpo_steps(model_kl, ref_kl, kl_beta=0.5, n_steps=10)
    drift_with_kl = _l2_param_drift(model_kl, ref_kl)

    torch.manual_seed(11)
    model_no = _ToyPolicy()
    ref_no = freeze_reference_policy(model_no)
    _run_n_grpo_steps(model_no, ref_no, kl_beta=0.0, n_steps=10)
    drift_without_kl = _l2_param_drift(model_no, ref_no)

    # KL-on must drift *less* than KL-off. Add a small slack to avoid a
    # flaky test from numerical noise; the gap is large in practice.
    assert drift_with_kl < drift_without_kl, (
        f"KL penalty failed to constrain drift: "
        f"with-KL={drift_with_kl:.4f} >= no-KL={drift_without_kl:.4f}"
    )
    # Also: with KL on, drift must be finite + bounded (no NaN / inf).
    assert drift_with_kl < 50.0, f"with-KL drift suspiciously huge: {drift_with_kl}"


def test_freeze_reference_policy_is_independent() -> None:
    """Modifying the live model must NOT change the frozen reference.

    We deepcopy the model in ``freeze_reference_policy``; this test
    pins that contract because a *shared-tensor* bug here would mean
    the KL term silently degenerates to 0 after the first step.
    """
    model = _ToyPolicy()
    ref = freeze_reference_policy(model)
    init = _l2_param_drift(model, ref)
    assert init == 0.0  # Same weights at snapshot time.

    # Now mutate the live model.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    after = _l2_param_drift(model, ref)
    assert after > 0.0, "freeze_reference_policy must deepcopy, not alias"

    # Reference must NOT have grad-enabled params.
    for p in ref.parameters():
        assert not p.requires_grad, "ref params must be requires_grad=False"


# ===========================================================================
# Sanity: the mock-rollout sampler returns shape-correct data
# ===========================================================================


def test_sample_rollouts_mock_shape_and_determinism() -> None:
    """Mock rollouts must be deterministic given a seed and shape (P, N, 2)."""
    rolls_a = sample_rollouts_mock(["q1", "q2"], [4, 7], n_rollouts=8, seed=42)
    rolls_b = sample_rollouts_mock(["q1", "q2"], [4, 7], n_rollouts=8, seed=42)
    assert rolls_a == rolls_b, "mock rollouts must be deterministic for a seed"
    assert len(rolls_a) == 2
    for group in rolls_a:
        assert len(group) == 8
        for text, gt in group:
            assert isinstance(text, str)
            assert gt in (4, 7, 5, 8)  # gt or off-by-one


# ===========================================================================
# Smoke: the train_100m_rl --smoke entrypoint runs end-to-end on CPU
# ===========================================================================


def test_train_100m_rl_smoke_runs() -> None:
    """``train_100m_rl --smoke`` must run 3 mock GRPO steps without error.

    This is the *entrypoint* test -- if the trainer's smoke path
    breaks, the GRPO loss arithmetic + verifier choice + step loop
    are all silently broken. Real GPU training is verified separately
    by the queue's H5 phase-trigger in DEEP_MAINT_QUEUE.md.
    """
    # Local import so the test file imports cleanly even if torch is
    # missing on a CI runner that has importorskip-skipped this module.
    train_rl = pytest.importorskip("train_100m_rl")
    rc = train_rl.main(["--smoke", "--rl-rollouts", "4", "--rl-verifier", "ast"])
    assert rc == 0


def test_train_100m_rl_dry_run() -> None:
    """``--dry-run`` prints config and exits with rc=0; no GPU touched."""
    train_rl = pytest.importorskip("train_100m_rl")
    rc = train_rl.main(
        ["--dry-run", "--rl-rollouts", "8", "--rl-verifier", "sympy"]
    )
    assert rc == 0
