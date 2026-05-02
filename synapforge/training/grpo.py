"""DEEP_MAINT_QUEUE T9.5 — Group Relative Policy Optimization (GRPO) loss + verifiers.

GRPO (DeepSeek-R1, GSM8K) is a critic-free RL algorithm that trains a
policy LLM with a verifier-derived reward. For each prompt, sample
``N`` independent rollouts, score them with a verifier function
(reward ∈ ℝ), then push up the log-probability of high-reward rollouts
relative to the group mean.

Why GRPO and not PPO
--------------------
PPO needs a *value head* — a learned critic that predicts expected
return per token. That doubles model parameters at training time and
introduces a regression loss term whose target is itself a moving
network output (hard to stabilize on a 100M policy). GRPO drops the
critic by **using the per-prompt group mean as the baseline**: the
advantage estimator is

    A_i = (r_i - mean(r)) / (std(r) + ε)

across the ``N`` rollouts of the same prompt. This is unbiased when
``N`` is reasonable (8 in DeepSeek-R1; we default to 8 too) and saves
both compute and code-paths.

Loss formula (one prompt, N rollouts)
-------------------------------------
    advantage_i = (r_i - mean(r)) / (std(r) + ε)
    pol_loss    = - advantage_i * log_prob_pi(rollout_i)
    kl_loss     =   β * KL( pi(rollout_i) || pi_ref(rollout_i) )
    loss        = pol_loss.mean() + kl_loss.mean()

The KL penalty against the *frozen reference* (= the SFT checkpoint
right before RL kicks in) is **load-bearing**: GRPO with no KL term
catastrophically forgets after ~50 steps because the verifier reward
is sparse and the policy collapses to short-but-correct outputs that
no longer chat. β=0.01 is the DeepSeek default; we expose ``--rl-kl-beta``.

The (optional) PPO-style ratio clipping (``--rl-clip 0.2``) is wired
through a ``ratio = exp(log_pi - log_pi_old)`` clamp; in pure on-policy
GRPO ``log_pi_old == log_pi`` so the clip is a no-op, but it activates
once you do *multi-epoch* updates over the same rollouts (memory-buffer
RL). Default 0.2 matches PPO/RLHF/DeepSeek convention.

Verifiers
---------
A *verifier* takes ``(rollout_text: str, ground_truth: str | int) ->
float`` reward in [0.0, 1.0]. We ship two:

* **sympy** — parse a final-answer line ("####  N" or "answer: N" or
  "= N" or last numeric token) and compare to ground truth via
  ``sympy.simplify(a - b).is_zero``. Skipped at import time when sympy is
  missing — the verifier explicitly raises ``RuntimeError`` so tests
  use ``pytest.importorskip("sympy")``.
* **ast** — *safe* code-rollout verifier that **NEVER executes** model
  output. Instead:
    1. Parse the rollout as Python via ``ast.parse`` (must be
       syntactically valid Python).
    2. Reject any unsafe identifier (``exec``, ``eval``, ``__import__``,
       ``os.system``, ``subprocess``, ``open``, ``input``, ``compile``,
       ``__builtins__``, ``globals``, ``locals``).
    3. Compare the source-text equality (or a regex-extracted final
       expression) against ground truth.
  The ground-truth-pattern match is enough for the GSM8K /
  HumanEval-style numerical-answer prompts; full *code-execution* RL
  belongs in a sandboxed worker (out of scope here). The contract is:
  **the verifier MUST refuse to run model output** — see
  ``test_ast_verifier_safe_no_exec`` for the test that pins this.

Reference-policy KL
-------------------
We compute KL on the rollout sequence's token-level log-probs using
the Schulman 2020 ``k3`` non-negative estimator:

    r            = log_pi - log_pi_ref
    kl_per_token = exp(-r) + r - 1                 # ≥ 0 ∀r
    kl_seq       = kl_per_token.sum(dim=time)      # per-rollout
    kl_loss      = kl_seq.mean()                    # batch reduction

The ``k3`` form is used instead of the naive ``E[log_pi - log_pi_ref]``
because the naive estimator is *signed* — minimising it pushes the
policy *below* the reference (assigning lower probability to its own
samples), the *opposite* of what we want. ``k3`` is identically
non-negative and minimised at zero exactly when the policies match.

CPU-safe public surface
-----------------------
This module imports ``torch`` lazily inside callables so unit tests can
``importorskip('torch')`` and still cover the verifier-only paths
(those don't need torch at all).
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Verifier interface
# --------------------------------------------------------------------------


# (rollout_text, ground_truth) -> reward in [0.0, 1.0]
VerifierFn = Callable[[str, Any], float]


# Banned identifiers in code rollouts. Keep this list canonical — any
# match anywhere in the AST (including nested attribute access) rejects
# the rollout outright with reward=0.0. The contract pinned by
# ``test_ast_verifier_safe_no_exec`` is: **no model-emitted code is ever
# executed by the verifier**. We rely on AST-walk + identifier-blocklist
# rather than sandboxing because that's a much smaller trust boundary.
_AST_FORBIDDEN_NAMES = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "__builtins__",
        "__class__",
        "__bases__",
        "__subclasses__",
        # Common dangerous module attrs
        "system",
        "popen",
        "spawn",
        "fork",
        "kill",
        "remove",
        "unlink",
        "rmdir",
        # File / network
        "socket",
        "subprocess",
        "shutil",
    }
)

# Forbidden module imports (top-level dotted names). Same rationale.
_AST_FORBIDDEN_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "ctypes",
        "pickle",
        "marshal",
        "importlib",
        "pty",
        "telnetlib",
    }
)


# Final-answer extraction patterns, tried in order. The first match wins.
_FINAL_ANSWER_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)\s*$", re.MULTILINE),
    re.compile(r"(?:answer|ans|result)\s*[:=]\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}"),
    re.compile(r"=\s*(-?[\d,]+(?:\.\d+)?)\s*\.?\s*$", re.MULTILINE),
)


def extract_final_number(text: str) -> Optional[float]:
    """Pull the canonical final numeric answer from a CoT rollout.

    Tries (in priority order): ``#### N`` (GSM8K), ``answer: N``,
    ``\\boxed{N}`` (MATH), then ``= N`` at end-of-line. Strips comma
    thousand-separators. Returns ``None`` on no match.
    """
    if not text:
        return None
    for pat in _FINAL_ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(1).replace(",", "").strip()
            try:
                return float(raw)
            except ValueError:
                continue
    # Last-resort: last numeric token in the text.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def sympy_verifier(rollout_text: str, ground_truth: Any) -> float:
    """Sympy-based numeric verifier for math chain-of-thought rollouts.

    Reward 1.0 if extracted-final-answer ≡ ground_truth (after
    ``sympy.simplify(a - b).is_zero``). Reward 0.0 otherwise — including
    the no-final-answer case.

    Raises ``RuntimeError`` if sympy is unavailable so callers can
    ``pytest.importorskip("sympy")`` and trainer code can detect a
    missing dependency at startup, not silently fall through to 0.0.
    """
    try:
        import sympy  # type: ignore
    except ImportError as exc:  # pragma: no cover -- exercised via tests
        raise RuntimeError(
            "sympy_verifier requires the 'sympy' package; install with "
            "`pip install sympy` or pass --rl-verifier ast instead."
        ) from exc

    pred = extract_final_number(rollout_text)
    if pred is None:
        return 0.0
    try:
        gt = float(ground_truth)
    except (TypeError, ValueError):
        return 0.0
    try:
        # Note: ``sympy.Float(0.0) == 0`` is *False* in sympy, but
        # ``.is_zero`` correctly recognises it. We use ``is_zero``
        # because it's the canonical zero-test on a sympy expression.
        diff = sympy.simplify(pred - gt)
        return 1.0 if bool(diff.is_zero) else 0.0
    except Exception:
        # Sympy can raise on weirdly-shaped numbers (NaN, inf). Treat
        # as wrong rather than crashing the whole rollout batch.
        return 0.0


def _ast_contains_forbidden(tree: ast.AST) -> Optional[str]:
    """Walk an AST and return the first forbidden token, or None.

    Catches:
      * ``Name`` nodes (bare identifiers like ``exec``)
      * ``Attribute`` chains (``os.system``, ``__class__.__bases__``)
      * ``Import`` / ``ImportFrom`` from the forbidden-modules set
      * ``Call`` to a forbidden name
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in _AST_FORBIDDEN_NAMES:
                return node.id
        elif isinstance(node, ast.Attribute):
            if node.attr in _AST_FORBIDDEN_NAMES:
                return node.attr
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _AST_FORBIDDEN_IMPORTS:
                    return f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in _AST_FORBIDDEN_IMPORTS:
                return f"from {node.module}"
        elif isinstance(node, ast.Call):
            # Reject indirect calls of the form getattr(..., 'exec')
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in _AST_FORBIDDEN_NAMES:
                        return f"getattr-string {arg.value!r}"
    return None


def ast_verifier(rollout_text: str, ground_truth: Any) -> float:
    """Safe AST-based code verifier — NEVER executes the rollout.

    Contract (pinned by ``test_ast_verifier_safe_no_exec``):
      1. Rollout MUST parse as syntactically-valid Python.
      2. Rollout MUST NOT reference forbidden names / imports
         (``exec``, ``eval``, ``os.system``, etc.).
      3. Reward 1.0 if the extracted final number matches ``ground_truth``.
         (Same extractor as ``sympy_verifier`` so the harness behaves
         predictably across both backends.)
      4. Reward 0.0 otherwise — including unsafe code, parse errors,
         and missing-final-answer cases.

    Out-of-scope: full code execution to compare program output against
    expected stdout. That requires a sandbox (subprocess + seccomp +
    nsjail) and is the job of a downstream worker. Keeping this verifier
    *non-executing* is a security boundary.
    """
    if not rollout_text:
        return 0.0
    # Strip a possible leading code-fence so the parser sees just code.
    text = rollout_text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError:
        return 0.0
    bad = _ast_contains_forbidden(tree)
    if bad is not None:
        return 0.0
    # Reuse the numeric extractor on the *original* rollout text (so we
    # also accept "# answer: 42" comments etc.).
    pred = extract_final_number(rollout_text)
    if pred is None:
        return 0.0
    try:
        gt = float(ground_truth)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if math.isclose(pred, gt, abs_tol=1e-6) else 0.0


# Verifier registry — keyed by the ``--rl-verifier`` CLI flag name.
VERIFIERS: Dict[str, VerifierFn] = {
    "sympy": sympy_verifier,
    "ast": ast_verifier,
}


def get_verifier(name: str) -> VerifierFn:
    """Resolve a verifier-name CLI flag to its callable."""
    if name not in VERIFIERS:
        raise ValueError(
            f"unknown --rl-verifier {name!r}; choose from {sorted(VERIFIERS)}"
        )
    return VERIFIERS[name]


# --------------------------------------------------------------------------
# GRPO loss (torch-only)
# --------------------------------------------------------------------------


@dataclass
class GRPOStats:
    """Diagnostic snapshot returned by ``grpo_loss``.

    Fields are plain Python floats / ints — safe to log / json-dump
    every step. No tensors held to avoid leaking GPU memory into the
    metrics buffer.
    """

    loss: float
    pol_loss: float
    kl_loss: float
    reward_mean: float
    reward_std: float
    advantage_abs_mean: float
    n_rollouts: int


def compute_advantages(
    rewards: "torch.Tensor",
    eps: float = 1e-8,
) -> "torch.Tensor":
    """Group-normalised advantages: ``(r - mean) / (std + eps)``.

    Operates on a 1D tensor of size ``N`` (rollouts of the same prompt)
    or a 2D tensor of shape ``(B, N)`` for ``B`` independent prompts.

    For ``N == 1`` we fall back to ``r - r`` = 0 — single-rollout GRPO
    has no signal, this prevents a div-by-zero NaN.
    """
    import torch

    if rewards.numel() == 0:
        return rewards.clone()
    if rewards.ndim == 1:
        if rewards.numel() == 1:
            return torch.zeros_like(rewards)
        m = rewards.mean()
        s = rewards.std(unbiased=False)
        return (rewards - m) / (s + eps)
    # 2D batched form.
    m = rewards.mean(dim=-1, keepdim=True)
    s = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return (rewards - m) / (s + eps)


def kl_divergence_per_token(
    log_pi: "torch.Tensor",
    log_pi_ref: "torch.Tensor",
    mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Per-rollout *non-negative* KL estimator (Schulman 2020 ``k3``).

    ``log_pi`` and ``log_pi_ref`` should be the *log-probabilities of
    the actually-sampled tokens* under each policy (one scalar per
    token), shape ``(N, T)`` or ``(N,)`` if already summed.

    ``mask`` (optional) is 1 over response tokens, 0 over prompt /
    pad. When provided, KL only accumulates over response tokens — KL
    on prompt tokens is meaningless because both policies must
    reproduce the prompt verbatim.

    The naive estimator ``E[log π - log π_ref]`` is unbiased but can
    go negative on a finite batch — minimising it then *encourages*
    the policy to push log_pi below the reference (assigning lower
    probability to its own samples), which is the opposite of what
    KL is supposed to do. We use the Schulman 2020 ``k3`` estimator:

        KL ≈ exp(log_ref - log_pi) - (log_ref - log_pi) - 1
            = exp(-r) + r - 1   where r = log_pi - log_pi_ref

    which is identically non-negative (Bregman divergence of exp at
    0) and has the same expectation as KL while having much lower
    variance. This is the load-bearing reason ``test_kl_penalty_prevents_drift``
    passes.
    """
    import torch

    if log_pi.shape != log_pi_ref.shape:
        raise ValueError(
            f"log_pi {tuple(log_pi.shape)} and log_pi_ref "
            f"{tuple(log_pi_ref.shape)} must have identical shape"
        )
    r = log_pi - log_pi_ref
    # Schulman k3: exp(-r) + r - 1 — non-negative for all r, zero iff r==0.
    # We *clamp* ``-r`` before the exp so a runaway policy (log_pi much
    # less than log_pi_ref, i.e. r ≪ 0) doesn't overflow exp() to inf
    # and NaN out the gradient. Clamp at 20 -> exp(20) ≈ 5e8, still
    # large enough to drive the policy back towards ref but safely finite
    # in fp32 / bf16.
    neg_r = (-r).clamp(max=20.0)
    kl = torch.exp(neg_r) + r - 1.0
    if mask is not None:
        if mask.shape != log_pi.shape:
            raise ValueError(
                f"mask {tuple(mask.shape)} must match log_pi "
                f"{tuple(log_pi.shape)}"
            )
        kl = kl * mask.to(kl.dtype)
    if kl.ndim == 1:
        return kl
    # Sum over time -> per-rollout scalar
    return kl.sum(dim=-1)


def grpo_loss(
    log_pi: "torch.Tensor",
    log_pi_old: Optional["torch.Tensor"],
    log_pi_ref: "torch.Tensor",
    rewards: "torch.Tensor",
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
    response_mask: Optional["torch.Tensor"] = None,
    advantages_eps: float = 1e-8,
) -> Tuple["torch.Tensor", GRPOStats]:
    """Group Relative Policy Optimization loss.

    Parameters
    ----------
    log_pi : ``(N, T)`` or ``(N,)``
        Token-level log-probs of the *current* policy on the sampled
        rollouts (autograd-attached).
    log_pi_old : ``(N, T)`` or ``(N,)`` or None
        Log-probs at *rollout time* (detached). When None, we set
        ``log_pi_old = log_pi.detach()`` — pure on-policy, ratio == 1
        and the clip is a no-op. Required when doing multi-epoch
        rollout-buffer updates.
    log_pi_ref : ``(N, T)`` or ``(N,)``
        Log-probs of the *frozen reference* (SFT-snapshot) policy on
        the same rollouts (detached, no grad).
    rewards : ``(N,)``
        Verifier-derived reward, one per rollout. Float in [0, 1] (or
        any reals — they're normalised via group mean/std).
    clip_eps : float
        PPO-style ratio clamp range (1-eps, 1+eps). Default 0.2.
    kl_beta : float
        KL-penalty coefficient. β=0 kills the regulariser; default 0.01
        per DeepSeek-R1.
    response_mask : ``(N, T)``, optional
        1 over response tokens, 0 over prompt/pad. Only matters when
        ``log_pi`` is 2D.
    advantages_eps : float
        Numerical-stability term in the advantage normaliser.

    Returns
    -------
    loss : 0-d tensor (autograd-attached)
    stats : GRPOStats
    """
    import torch

    if rewards.numel() == 0:
        raise ValueError("grpo_loss received zero rollouts (rewards.numel() == 0)")
    if log_pi.shape != log_pi_ref.shape:
        raise ValueError(
            f"log_pi {tuple(log_pi.shape)} != log_pi_ref "
            f"{tuple(log_pi_ref.shape)}"
        )
    if log_pi_old is None:
        log_pi_old = log_pi.detach()
    elif log_pi_old.shape != log_pi.shape:
        raise ValueError(
            f"log_pi_old {tuple(log_pi_old.shape)} != log_pi "
            f"{tuple(log_pi.shape)}"
        )

    # Sum log-probs over time -> per-rollout scalar.
    if log_pi.ndim == 1:
        sum_log_pi = log_pi
        sum_log_pi_old = log_pi_old
    else:
        if response_mask is not None:
            mask = response_mask.to(log_pi.dtype)
            sum_log_pi = (log_pi * mask).sum(dim=-1)
            sum_log_pi_old = (log_pi_old * mask).sum(dim=-1)
        else:
            sum_log_pi = log_pi.sum(dim=-1)
            sum_log_pi_old = log_pi_old.sum(dim=-1)

    advantages = compute_advantages(rewards, eps=advantages_eps).detach()

    # PPO-style ratio (always 1.0 on-policy).
    ratio = torch.exp(sum_log_pi - sum_log_pi_old.detach())
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pol_per = -torch.minimum(ratio * advantages, ratio_clipped * advantages)
    pol_loss = pol_per.mean()

    # KL against frozen reference, per rollout.
    kl_per = kl_divergence_per_token(log_pi, log_pi_ref, response_mask)
    kl_loss = kl_per.mean()

    total = pol_loss + kl_beta * kl_loss

    stats = GRPOStats(
        loss=float(total.detach().item()),
        pol_loss=float(pol_loss.detach().item()),
        kl_loss=float(kl_loss.detach().item()),
        reward_mean=float(rewards.detach().mean().item()),
        reward_std=float(rewards.detach().std(unbiased=False).item()),
        advantage_abs_mean=float(advantages.detach().abs().mean().item()),
        n_rollouts=int(rewards.numel()),
    )
    return total, stats


# --------------------------------------------------------------------------
# Reference-policy snapshot helper
# --------------------------------------------------------------------------


def freeze_reference_policy(model: "torch.nn.Module") -> "torch.nn.Module":
    """Return a *deep-copied, eval-mode, requires_grad=False* mirror.

    Used by the GRPO trainer at startup: snapshot the SFT model right
    before RL begins so the KL term keeps the policy from drifting
    catastrophically. The snapshot lives on the same device as the
    live model — caller can ``.to('cpu')`` if VRAM is tight.

    Important: this is a **structural** copy via ``copy.deepcopy``; the
    state-dict is bit-exact. We do *not* try to share buffers — that
    would risk both models writing to RoPE caches simultaneously.
    """
    import copy

    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    for b in ref.buffers():
        # Freeze running stats too -- the ref must be exactly the SFT model.
        b.requires_grad_(False)
    return ref


# --------------------------------------------------------------------------
# Rollout sampling helper (deterministic-mock-friendly)
# --------------------------------------------------------------------------


def sample_rollouts_mock(
    prompts: Sequence[str],
    ground_truths: Sequence[Any],
    *,
    n_rollouts: int = 8,
    template: str = "Q: {prompt}\nA: The answer is #### {gt}",
    seed: int = 0,
) -> List[List[Tuple[str, Any]]]:
    """Deterministic mock rollout sampler for CPU tests.

    Returns ``[[(rollout_text, gt) × N] × len(prompts)]``. Each rollout
    is a templated string that the verifier scores trivially — half are
    "correct" (final answer == gt), half are off-by-one — so the test
    can verify advantage normalisation produces non-degenerate signal.

    Production rollout sampling (with the real model) lives in
    ``train_100m_rl.py`` because it needs the LM forward + tokenizer.
    """
    rng = _mock_rng(seed)
    out: List[List[Tuple[str, Any]]] = []
    for prompt, gt in zip(prompts, ground_truths):
        rollouts: List[Tuple[str, Any]] = []
        for i in range(n_rollouts):
            # Half correct, half off-by-one — keeps tests deterministic
            # while producing non-zero reward variance per group.
            if rng() % 2 == 0:
                text = template.format(prompt=prompt, gt=gt)
            else:
                try:
                    bad = float(gt) + 1
                except (TypeError, ValueError):
                    bad = "WRONG"
                text = template.format(prompt=prompt, gt=bad)
            rollouts.append((text, gt))
        out.append(rollouts)
    return out


def _mock_rng(seed: int):
    """Tiny LCG so tests don't need numpy / random.seed contamination."""
    state = [seed & 0xFFFFFFFF]

    def next_int() -> int:
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0]

    return next_int


__all__ = [
    "VerifierFn",
    "VERIFIERS",
    "GRPOStats",
    "ast_verifier",
    "compute_advantages",
    "extract_final_number",
    "freeze_reference_policy",
    "get_verifier",
    "grpo_loss",
    "kl_divergence_per_token",
    "sample_rollouts_mock",
    "sympy_verifier",
]
