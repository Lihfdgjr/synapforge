"""train_100m_rl -- GRPO RL post-training for synapforge_100m on top of SFT (T9.5).

Phase-4 trainer in the ppl-gated training pipeline (see
``docs/MASTER_PLAN.md`` §3 and ``docs/DEEP_MAINT_QUEUE.md`` H5):

    Phase 0  -> warm KD pretrain        ce ≈ 9.6 -> ppl ≤ 250
    Phase 1  -> + self-learn / curiosity
    Phase 2  -> + multimodal aux        ppl ≤ 100
    Phase 3  -> SFT (alpaca / gsm8k)    ppl ≤ 60   <-- T9.4 (sister agent)
    Phase 4  -> GRPO RL on math         chat eval >= 0.6  <-- THIS FILE

Goal: push ppl from 80 (post-SFT) to 20-30 by training on chain-of-thought
math rollouts where the verifier (sympy) gives reward = 1.0 only for
arithmetically-correct final answers.

Algorithm (DeepSeek-R1-style GRPO)
-----------------------------------
1. Snapshot the SFT checkpoint as a *frozen reference policy*.
2. For each prompt batch (size B):
   a. Sample N rollouts per prompt (default 8) with temperature 0.7.
   b. Score each rollout via the verifier -> reward in [0, 1].
   c. Group-normalise rewards -> advantages.
   d. Compute log-probs of rollouts under live + reference policies.
   e. GRPO loss = -A * log_pi(rollout)  +  β * KL(pi || pi_ref).
3. Backprop, step, repeat.

The KL term against the frozen reference is **load-bearing**: without
it the policy collapses to short-but-correct outputs and chat ability
disappears within ~50 steps. β=0.01 is the published DeepSeek default.

CLI flags (added on top of the SFT trainer set)
------------------------------------------------
* ``--rl-rollouts 8``         number of rollouts sampled per prompt
* ``--rl-verifier sympy|ast`` which verifier to score with
* ``--rl-temperature 0.7``    sampling temperature for rollouts
* ``--rl-data gsm8k|humaneval``
                              prompt source (gsm8k = math, humaneval = code)
* ``--rl-clip 0.2``           PPO-style ratio clip (default GRPO value)
* ``--rl-kl-beta 0.01``       KL-penalty coefficient
* ``--rl-max-new 128``        max new tokens per rollout (response only)

CPU-safe testing
----------------
The trainer's *correctness* is unit-tested via ``synapforge.training.grpo``
(``tests/integration/test_grpo_trainer.py``). The full launcher in this
file requires a real GPU + tokenizer + parquet — those tests use a
deterministic mock rollout function so the GRPO loss / verifier paths
can run on CPU without a model forward.

See ``docs/MASTER_PLAN.md`` §3 (phase 4) and the queue entry T9.5 for
the wider context.
"""
from __future__ import annotations

import os as _os_early

_os_early.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


def _early_path_setup() -> None:
    """Make ``synapforge`` importable regardless of cwd, like the SFT trainer."""
    sys.path[:] = [
        p for p in sys.path if p not in ("/workspace/synapforge", "")
    ]
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_early_path_setup()


# Architecture constants (match the SFT and KD trainers so warmstart loads cleanly).
SEQ_LEN = 1024
MODEL_FFN_RATIO = 8.0
MODEL_SPARSITY = 0.95
MODEL_DROPOUT = 0.0
MODEL_TIE_LM_HEAD = True


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO RL trainer for synapforge_100m")
    # --- I/O ---
    p.add_argument("--out", default="/workspace/runs/v24h_qwen_rl",
                   help="checkpoint output dir")
    p.add_argument("--warmstart", required=False,
                   help="SFT checkpoint to load + freeze as reference policy")
    p.add_argument("--rl-data", default="gsm8k",
                   choices=("gsm8k", "humaneval"),
                   help="prompt source")
    p.add_argument("--rl-data-parquet", default=None,
                   help="explicit parquet path; overrides --rl-data default")
    p.add_argument("--tokenizer-path", default=None,
                   help="HF tokenizer for rollout sampling (required if not --dry-run)")
    # --- model architecture (must match SFT for warmstart) ---
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--loop-depth", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=SEQ_LEN)
    # --- RL hyperparameters ---
    p.add_argument("--rl-rollouts", type=int, default=8,
                   help="rollouts sampled per prompt (group size N)")
    p.add_argument("--rl-verifier", default="sympy", choices=("sympy", "ast"),
                   help="reward function for scoring rollouts")
    p.add_argument("--rl-temperature", type=float, default=0.7,
                   help="rollout sampling temperature")
    p.add_argument("--rl-clip", type=float, default=0.2,
                   help="PPO-style ratio clip range")
    p.add_argument("--rl-kl-beta", type=float, default=0.01,
                   help="KL penalty coefficient against frozen ref policy")
    p.add_argument("--rl-max-new", type=int, default=128,
                   help="max new tokens per rollout (response length)")
    # --- Optimiser ---
    p.add_argument("--batch-size", type=int, default=4,
                   help="prompts per training step (each spawns rl-rollouts rollouts)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-6,
                   help="GRPO LR is 10-100x smaller than SFT to avoid policy collapse")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    # --- Backend / runtime ---
    p.add_argument("--backend", default="triton_block",
                   choices=("gpu_dense", "triton_block"))
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    # --- Smoke / dry-run paths ---
    p.add_argument("--dry-run", action="store_true",
                   help="print resolved config + exit; no GPU work")
    p.add_argument("--smoke", action="store_true",
                   help="run 3 GRPO steps on mock rollouts (no real model); for CI")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Prompt loaders -- tiny adapters around the parquet schemas the
# tokenize_gsm8k.py / tokenize_humaneval_mbpp.py shippers produce.
# ---------------------------------------------------------------------------


@dataclass
class RLPrompt:
    """One prompt + its verifier ground-truth.

    For GSM8K math: ``prompt`` = "Q: ...\nA:" and ``ground_truth`` = int
    final answer. For HumanEval-style code: ``prompt`` = problem
    statement and ``ground_truth`` = expected numeric output (we keep
    the verifier purely numeric to avoid sandboxed exec).
    """

    prompt: str
    ground_truth: Any
    prompt_ids: Optional[List[int]] = None


def _load_gsm8k_prompts(parquet_path: str, n: int) -> List[RLPrompt]:
    """Read the gsm8k parquet that ``tokenize_gsm8k.py`` writes.

    Schema: question (str), cot (str), answer (str), input_ids
    (list<int32>), final_answer (int64). For RL we only need the
    question + final_answer (the model's own CoT comes from sampling).
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    questions = table["question"].to_pylist()
    finals = table["final_answer"].to_pylist()
    out: List[RLPrompt] = []
    for q, fa in list(zip(questions, finals))[:n]:
        out.append(RLPrompt(prompt=f"Q: {q}\nA:", ground_truth=int(fa)))
    return out


def _load_humaneval_prompts(parquet_path: str, n: int) -> List[RLPrompt]:
    """Best-effort HumanEval/MBPP loader; falls back to question-only.

    The shipped tokenize_humaneval_mbpp.py output schema is best-effort
    here — we only require ``question`` + ``final_answer`` columns. If
    the parquet has only string-answer columns the AST verifier still
    works because it compares numeric tail tokens.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    cols = {c: table[c].to_pylist() for c in table.column_names}
    qcol = next((c for c in ("question", "prompt", "task") if c in cols), None)
    acol = next(
        (c for c in ("final_answer", "answer", "expected") if c in cols), None
    )
    if qcol is None or acol is None:
        raise RuntimeError(
            f"humaneval parquet {parquet_path!r} missing question/answer cols"
        )
    out: List[RLPrompt] = []
    for q, a in list(zip(cols[qcol], cols[acol]))[:n]:
        out.append(RLPrompt(prompt=str(q), ground_truth=a))
    return out


def _resolve_prompt_set(args: argparse.Namespace) -> List[RLPrompt]:
    """Pick the parquet to load based on --rl-data + --rl-data-parquet."""
    if args.rl_data_parquet:
        path = args.rl_data_parquet
    elif args.rl_data == "gsm8k":
        path = "/workspace/data/math/gsm8k_qwen.parquet"
    elif args.rl_data == "humaneval":
        path = "/workspace/data/code/humaneval_mbpp_qwen.parquet"
    else:
        raise ValueError(f"unknown --rl-data {args.rl_data!r}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RL prompt parquet not found: {path}. Run scripts/tokenize_gsm8k.py first."
        )
    if args.rl_data == "gsm8k":
        return _load_gsm8k_prompts(path, n=args.steps * args.batch_size)
    return _load_humaneval_prompts(path, n=args.steps * args.batch_size)


# ---------------------------------------------------------------------------
# Rollout sampling (live model, GPU path)
# ---------------------------------------------------------------------------


def _sample_rollout_live(
    model: Any,
    tokenizer: Any,
    prompt: str,
    n_rollouts: int,
    max_new: int,
    temperature: float,
    device: str,
    seed_offset: int,
) -> List[Tuple[List[int], List[int]]]:
    """Sample ``n_rollouts`` independent rollouts from the live model.

    Returns a list of ``(prompt_ids + response_ids, response_ids)``
    pairs; the second tuple element is the response slice the verifier
    sees.

    We use simple per-step softmax sampling (no top-p) because GRPO
    benefits from full-distribution exploration — top-p collapses the
    high-reward tail too aggressively for RL-from-verifier signals.

    Production-grade speedups (KV-cache reuse, batch-of-N sampling)
    belong here in a follow-up — current path is per-rollout sequential
    so it's correct and reproducible. Each rollout uses an independent
    torch.Generator seeded from ``seed_offset + i`` for determinism.
    """
    import torch

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out: List[Tuple[List[int], List[int]]] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for i in range(n_rollouts):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed_offset * 10_000 + i)
        ids = prompt_t.clone()
        response: List[int] = []
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(ids)  # (1, T, V)
                next_logits = logits[0, -1, :].float() / max(temperature, 1e-6)
                probs = torch.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1, generator=gen).item())
                response.append(next_id)
                ids = torch.cat(
                    [ids, torch.tensor([[next_id]], dtype=ids.dtype, device=device)],
                    dim=1,
                )
                if eos_id is not None and next_id == eos_id:
                    break
        out.append((prompt_ids + response, response))
    return out


def _logprob_of_sequence(
    model: Any,
    full_ids: List[int],
    response_len: int,
    device: str,
) -> "torch.Tensor":
    """Compute summed log-prob of the *response* tokens under ``model``.

    The response is the last ``response_len`` tokens of ``full_ids``;
    everything before them is the prompt and we don't include its
    log-prob (the policy doesn't get credit for reproducing the prompt).
    """
    import torch

    if response_len <= 0:
        return torch.zeros((), device=device)
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(ids[:, :-1])  # (1, T-1, V)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[:, 1:]  # (1, T-1)
    # Slice the last ``response_len`` positions — those are the response tokens.
    targets_resp = targets[:, -response_len:]
    log_probs_resp = log_probs[:, -response_len:, :]
    selected = log_probs_resp.gather(-1, targets_resp.unsqueeze(-1)).squeeze(-1)
    return selected.sum()  # scalar


# ---------------------------------------------------------------------------
# Single-step GRPO update
# ---------------------------------------------------------------------------


def _one_grpo_step(
    *,
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    optim: Any,
    prompts: List[RLPrompt],
    verifier_fn: Any,
    device: str,
    args: argparse.Namespace,
    step_idx: int,
) -> Any:
    """Run one full GRPO step over ``len(prompts)`` prompts.

    For each prompt:
      1. Sample ``args.rl_rollouts`` rollouts.
      2. Score each rollout with ``verifier_fn`` -> reward.
      3. Compute live + ref log-probs of the response tokens.
      4. Build the GRPO loss + KL term and backprop.

    Returns aggregated stats dict.
    """
    import torch

    from synapforge.training.grpo import (
        compute_advantages,
        grpo_loss,
    )

    model.train()
    optim.zero_grad(set_to_none=True)

    all_log_pi: List["torch.Tensor"] = []
    all_log_pi_ref: List["torch.Tensor"] = []
    all_rewards: List[float] = []
    all_group_rewards: List[List[float]] = []

    for p_idx, prompt in enumerate(prompts):
        rollouts = _sample_rollout_live(
            model,
            tokenizer,
            prompt.prompt,
            n_rollouts=args.rl_rollouts,
            max_new=args.rl_max_new,
            temperature=args.rl_temperature,
            device=device,
            seed_offset=step_idx * len(prompts) + p_idx,
        )

        group_rewards: List[float] = []
        log_pi_grp: List["torch.Tensor"] = []
        log_pi_ref_grp: List["torch.Tensor"] = []
        for full_ids, resp_ids in rollouts:
            text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            r = float(verifier_fn(text, prompt.ground_truth))
            group_rewards.append(r)
            log_pi_grp.append(
                _logprob_of_sequence(model, full_ids, len(resp_ids), device)
            )
            with torch.no_grad():
                log_pi_ref_grp.append(
                    _logprob_of_sequence(ref_model, full_ids, len(resp_ids), device)
                )
        # Stack -> (N,) tensors per prompt, advantage-normalise within group.
        log_pi_t = torch.stack(log_pi_grp)
        log_pi_ref_t = torch.stack(log_pi_ref_grp).detach()
        rewards_t = torch.tensor(group_rewards, dtype=torch.float32, device=device)

        loss_p, stats = grpo_loss(
            log_pi=log_pi_t,
            log_pi_old=None,
            log_pi_ref=log_pi_ref_t,
            rewards=rewards_t,
            clip_eps=args.rl_clip,
            kl_beta=args.rl_kl_beta,
        )
        loss_p.backward()

        all_log_pi.append(log_pi_t.detach())
        all_log_pi_ref.append(log_pi_ref_t.detach())
        all_rewards.extend(group_rewards)
        all_group_rewards.append(group_rewards)

    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=args.grad_clip,
        )
    optim.step()

    flat_rewards = torch.tensor(all_rewards)
    return {
        "reward_mean": float(flat_rewards.mean().item()) if len(all_rewards) else 0.0,
        "reward_std": (
            float(flat_rewards.std(unbiased=False).item())
            if len(all_rewards) > 1
            else 0.0
        ),
        "n_rollouts": len(all_rewards),
        "n_prompts": len(prompts),
        "group_rewards": all_group_rewards,
    }


def _lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    """Cosine-decay schedule with linear warmup -- same as SFT trainer."""
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Smoke (CI-callable, no GPU needed)
# ---------------------------------------------------------------------------


def _smoke(args: argparse.Namespace) -> int:
    """Run 3 mock GRPO steps on CPU with a deterministic toy "model"."""
    import torch

    from synapforge.training.grpo import (
        get_verifier,
        grpo_loss,
        sample_rollouts_mock,
    )

    _log("[smoke] mock rollouts + verifier reward + GRPO loss step")
    verifier = get_verifier(args.rl_verifier)
    prompts = ["Q: 2+2", "Q: 5*3", "Q: 10-7"]
    gts = [4, 15, 3]
    # AST verifier needs syntactically-valid Python rollouts; sympy is
    # happy with prose. We escape the prompt as a string-literal so any
    # punctuation in it doesn't break ast.parse().
    if args.rl_verifier == "ast":
        template = "prompt = {prompt!r}\n# answer: {gt}"
    else:
        template = "Q: {prompt}\nA: The answer is #### {gt}"
    for step in range(1, 4):
        groups = sample_rollouts_mock(
            prompts, gts, n_rollouts=args.rl_rollouts, seed=step,
            template=template,
        )
        for prompt, gt, rolls in zip(prompts, gts, groups):
            rewards = torch.tensor(
                [verifier(text, ground_truth) for text, ground_truth in rolls],
                dtype=torch.float32,
            )
            # Toy log_pi: random-but-fixed per (step, rollout); ref = +noise.
            torch.manual_seed(step * 13 + len(prompt))
            log_pi = torch.randn(args.rl_rollouts, requires_grad=True)
            log_pi_ref = log_pi.detach().clone() + 0.01
            loss, stats = grpo_loss(
                log_pi=log_pi,
                log_pi_old=None,
                log_pi_ref=log_pi_ref,
                rewards=rewards,
                clip_eps=args.rl_clip,
                kl_beta=args.rl_kl_beta,
            )
            loss.backward()
            _log(
                f"[smoke] step {step} {prompt!r}: r_mean={stats.reward_mean:.2f} "
                f"loss={stats.loss:.4f} pol={stats.pol_loss:.4f} "
                f"kl={stats.kl_loss:.4f}"
            )
    _log("[smoke] OK -- 3 GRPO steps + sympy/ast verifiers + KL all live")
    return 0


# ---------------------------------------------------------------------------
# Main GPU launcher
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.dry_run:
        _log("[rl] dry-run; resolved config:")
        for k, v in sorted(vars(args).items()):
            _log(f"  {k}={v}")
        return 0

    if args.smoke:
        return _smoke(args)

    # ------------------------------------------------------------------ GPU
    # Real training path. We import torch + the model + tokenizer here
    # rather than at module-load time so unit tests that ``importorskip``
    # torch can still import this module's CLI surface.
    import torch
    import torch.nn.functional as F  # noqa: F401 -- reserved for future logging

    from synapforge.huggingface_adapter import adv_warmstart  # noqa: F401
    from synapforge.model_100m import build_synapforge_100m
    from synapforge.optim import build_optimizer
    from synapforge.training.grpo import (
        freeze_reference_policy,
        get_verifier,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    _log(f"[rl] device={device} dtype={dtype} out={args.out}")
    _log(
        f"[rl] N_rollouts={args.rl_rollouts} verifier={args.rl_verifier} "
        f"temp={args.rl_temperature} clip={args.rl_clip} "
        f"kl_beta={args.rl_kl_beta} max_new={args.rl_max_new}"
    )

    # --- model ---
    model = build_synapforge_100m(
        vocab=args.vocab,
        d=args.d,
        n_layers=args.n_layers,
        loop_depth=args.loop_depth,
        max_seq=args.max_seq,
        ffn_ratio=MODEL_FFN_RATIO,
        sparsity=MODEL_SPARSITY,
        dropout=MODEL_DROPOUT,
        use_grad_checkpoint=args.grad_checkpoint,
    )
    n_params = model.num_parameters()
    _log(f"[rl] model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # --- warmstart from SFT (REQUIRED) ---
    if not args.warmstart:
        _log(
            "[rl] FATAL: --warmstart is REQUIRED for GRPO; cannot RL a "
            "from-scratch model."
        )
        return 2
    if not os.path.exists(args.warmstart):
        _log(f"[rl] FATAL: --warmstart {args.warmstart!r} not found")
        return 2
    rep = adv_warmstart(model, args.warmstart)
    _log(
        f"[rl] warmstart matched={rep.matched}/{rep.total_target} "
        f"missing={len(rep.missing)} extra={len(rep.extra)}"
    )
    model = model.to(device)
    model.train()

    # --- frozen reference policy snapshot ---
    ref_model = freeze_reference_policy(model)
    _log("[rl] reference policy snapshot frozen (KL anchor)")

    # --- backend ---
    if args.backend == "triton_block":
        try:
            from synapforge.backends.triton_block import TritonBlockBackend
            from synapforge.backends.triton_block_kernel import _HAS_TRITON

            be = TritonBlockBackend()
            stats = be.compile(model)
            _log(
                f"[rl] backend triton_block: avail={_HAS_TRITON} "
                f"pairs={stats.get('n_pairs_fused', 0)}"
            )
        except Exception as e:
            _log(f"[rl] triton_block FAILED: {e}; using gpu_dense")
            args.backend = "gpu_dense"

    # --- tokenizer ---
    if args.tokenizer_path is None:
        _log("[rl] FATAL: --tokenizer-path required for rollout sampling")
        return 2
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- optimiser ---
    optim = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # --- prompts ---
    prompts_all = _resolve_prompt_set(args)
    _log(f"[rl] loaded {len(prompts_all):,} prompts from --rl-data {args.rl_data}")
    verifier_fn = get_verifier(args.rl_verifier)

    # --- run ---
    t0 = time.time()
    best_reward = -float("inf")
    cursor = 0
    for step in range(1, args.steps + 1):
        cur_lr = _lr_at(step, args.lr, args.warmup, args.steps)
        for pg in optim.param_groups:
            pg["lr"] = cur_lr

        # Slice next batch of prompts (cycle when exhausted).
        batch: List[RLPrompt] = []
        for _ in range(args.batch_size):
            batch.append(prompts_all[cursor % len(prompts_all)])
            cursor += 1
        t_step = time.time()
        stats = _one_grpo_step(
            model=model,
            ref_model=ref_model,
            tokenizer=tok,
            optim=optim,
            prompts=batch,
            verifier_fn=verifier_fn,
            device=device,
            args=args,
            step_idx=step,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.time() - t_step) * 1000.0

        if step % args.log_every == 0 or step == 1:
            _log(
                f"[rl] step {step:5d} r_mean={stats['reward_mean']:.3f} "
                f"r_std={stats['reward_std']:.3f} "
                f"lr={cur_lr:.2e} step_ms={step_ms:.0f} "
                f"n_rollouts={stats['n_rollouts']}"
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "model": model.state_dict(),
                "step": step,
                "reward_mean": stats["reward_mean"],
                "reward_std": stats["reward_std"],
                "config": {
                    "vocab": args.vocab,
                    "d": args.d,
                    "n_layers": args.n_layers,
                    "loop_depth": args.loop_depth,
                    "max_seq": args.max_seq,
                    "ffn_ratio": MODEL_FFN_RATIO,
                    "sparsity": MODEL_SPARSITY,
                    "dropout": MODEL_DROPOUT,
                    "tie_lm_head": MODEL_TIE_LM_HEAD,
                },
                "rl_meta": {
                    "rollouts": args.rl_rollouts,
                    "verifier": args.rl_verifier,
                    "temperature": args.rl_temperature,
                    "kl_beta": args.rl_kl_beta,
                    "rl_data": args.rl_data,
                },
            }
            if stats["reward_mean"] > best_reward:
                best_reward = stats["reward_mean"]
                best_path = os.path.join(args.out, f"best_step_{step:06d}.pt")
                torch.save(ckpt, best_path)
                _log(
                    f"[rl] saved BEST ckpt {best_path} "
                    f"(r_mean={stats['reward_mean']:.3f})"
                )
            torch.save(ckpt, os.path.join(args.out, f"step_{step:06d}.pt"))

    _log(f"[rl] done in {time.time()-t0:.0f}s; best r_mean={best_reward:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
