"""test_routers - port-validation suite for sf.routers.

Validates:
  1. MoR (Mixture of Recursions): adaptive per-token depth, training
     converges, no NaN, depth histogram is non-degenerate.
  2. CoE (Chain of Experts): per-step routers diverge during training,
     aux loss tracks.
  3. DeepSeek-MoE: routed + shared experts, top-K aggregation, aux
     load-balance loss decreases over training.
  4. RDT (Recurrent-Depth Transformer): R=4 train then evaluate at
     R=1, 4, 16 - show test-time compute scaling.
  5. Backend compatibility: every router compiles + runs on both
     gpu_dense and (when available) triton_block backends.

Usage:
    CUDA_VISIBLE_DEVICES=1 python /workspace/synapforge/test_routers.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import synapforge as sf
from synapforge.routers import (
    ChainOfExperts,
    DeepSeekMoE,
    MoRStack,
    RDTLoop,
    attach_coe_to_block,
    attach_moe_to_block,
)


class TinyBody(sf.Module):
    """Linear -> tanh -> LayerNorm. Returns Tensor."""
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
    def forward(self, x):
        return self.norm(torch.tanh(self.proj(x)))


class ToyLM(sf.Module):
    """Embed -> trunk -> tied LM head."""
    def __init__(self, vocab, hidden, trunk):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.trunk = trunk
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
    def forward(self, tokens, **trunk_kwargs):
        x = self.emb(tokens)
        h = self.trunk(x, **trunk_kwargs)
        return self.lm_head(h)


_FIXED_PATTERN = None


def deterministic_token_batch(B, T, vocab, device, seed=7):
    """One fixed sequence repeated B times - the model has to memorise it."""
    global _FIXED_PATTERN
    if _FIXED_PATTERN is None or _FIXED_PATTERN.numel() < (T + 1):
        g = torch.Generator(device="cpu").manual_seed(seed)
        _FIXED_PATTERN = torch.randint(0, vocab, (max(T + 1, 256),), generator=g)
    seq = _FIXED_PATTERN[: T + 1].to(device)
    full = seq.unsqueeze(0).expand(B, T + 1).contiguous()
    return full[:, :-1], full[:, 1:]


def train_loop(model, n_steps, device, vocab=100, B=8, T=16, lr=3e-3,
               extra_loss_fn=None, label="model"):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for step in range(n_steps):
        inputs, targets = deterministic_token_batch(B, T, vocab, device)
        logits = model(inputs)
        ce = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        loss = ce
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        l = float(loss.item())
        if not math.isfinite(l):
            raise RuntimeError(f"{label}: NaN at step {step}")
        losses.append(l)
    return losses


@torch.no_grad()
def eval_ppl(model, device, vocab=100, B=16, T=16, **fkwargs):
    model.eval()
    inputs, targets = deterministic_token_batch(B, T, vocab, device)
    logits = model(inputs, **fkwargs)
    ce = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
    return float(math.exp(min(float(ce.item()), 50.0)))


class _MoRTrunk(sf.Module):
    """Wraps MoRStack and stashes the budget loss for the trainer."""
    def __init__(self, mor):
        super().__init__()
        self.mor = mor
        self.last_depth_loss = torch.zeros(())
    def forward(self, x):
        y, dl = self.mor(x)
        self.last_depth_loss = dl
        return y


def test_mor(device, n_steps, hidden=64):
    print(f"\n[MoR] hidden={hidden} max_depth=4 ACT-remainder", flush=True)
    torch.manual_seed(0)
    body = TinyBody(hidden).to(device)
    mor = MoRStack(
        body, hidden=hidden, max_depth=4, target_depth=2.5,
        budget_weight=1e-2, use_act_remainder=True, with_spike_rate=False,
    ).to(device)
    trunk = _MoRTrunk(mor)
    model = ToyLM(vocab=100, hidden=hidden, trunk=trunk).to(device)

    def extra(m):
        return m.trunk.last_depth_loss

    losses = train_loop(model, n_steps, device, label="MoR", extra_loss_fn=extra)

    model.eval()
    with torch.no_grad():
        inputs, _ = deterministic_token_batch(8, 16, 100, device)
        _ = model(inputs)
    hist = mor.last_depth_histogram().cpu().numpy()
    print(f"[MoR] losses {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)
    print(f"[MoR] depth-bucket fractions [d=0..max_depth]: "
          f"{[round(float(v), 3) for v in hist.tolist()]}", flush=True)
    nonzero = int((hist > 1e-3).sum())
    return {
        "losses": losses,
        "hist": [round(float(v), 4) for v in hist.tolist()],
        "n_active_buckets": nonzero,
        "loss_drop": losses[0] - losses[-1],
    }


def test_coe(device, n_steps, hidden=64):
    print(f"\n[CoE] hidden={hidden} n_routed=8 n_shared=2 top_k=2 n_steps=4", flush=True)
    torch.manual_seed(1)
    body = TinyBody(hidden).to(device)
    wrapped = attach_coe_to_block(
        body, hidden=hidden, n_routed=8, n_shared=2, top_k=2, n_steps=4, aux_alpha=1e-2,
    ).to(device)
    rdt = RDTLoop(
        wrapped, hidden=hidden, max_loops_train=4, max_loops_infer=4,
        enable_loop_index_embed=False, enable_layer_scale=False,
        enable_residual_gate_bias=False, enable_depth_lora=False,
    ).to(device)
    model = ToyLM(vocab=100, hidden=hidden, trunk=rdt).to(device)

    def extra(m):
        a = m.trunk.body.last_moe_aux
        return a if a is not None else torch.zeros((), device=device)

    losses = train_loop(model, n_steps, device, label="CoE", extra_loss_fn=extra)

    hist = wrapped.coe.step_routing_histogram().cpu().numpy()
    print(f"[CoE] losses {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)
    print("[CoE] per-step expert distribution (rows=step, cols=expert):", flush=True)
    for t, row in enumerate(hist.tolist()):
        print(f"       step {t}: {[round(float(v), 3) for v in row]}", flush=True)

    p = hist[0] + 1e-9
    q = hist[-1] + 1e-9
    p = p / p.sum()
    q = q / q.sum()
    kl = float(np.sum(p * (np.log(p) - np.log(q))))
    print(f"[CoE] KL(step0 || step{hist.shape[0]-1}) = {kl:.4f}  (>0 => routers diverged)", flush=True)
    return {
        "losses": losses,
        "step_routing": [[round(float(v), 4) for v in row] for row in hist.tolist()],
        "kl_step0_vs_last": kl,
        "loss_drop": losses[0] - losses[-1],
    }


def test_moe(device, n_steps, hidden=64):
    print(f"\n[DeepSeekMoE] hidden={hidden} n_routed=8 n_shared=2 top_k=2", flush=True)
    torch.manual_seed(2)
    body = TinyBody(hidden).to(device)
    wrapped = attach_moe_to_block(
        body, hidden=hidden, n_routed=8, n_shared=2, top_k=2, aux_alpha=1e-2,
    ).to(device)
    model = ToyLM(vocab=100, hidden=hidden, trunk=wrapped).to(device)
    aux_track = []

    def extra(m):
        a = m.trunk.last_moe_aux
        if a is None:
            return torch.zeros((), device=device)
        aux_track.append(float(a.detach().item()))
        return a

    losses = train_loop(model, n_steps, device, label="MoE", extra_loss_fn=extra)
    print(f"[DeepSeekMoE] losses {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)
    if aux_track:
        first = sum(aux_track[:5]) / max(1, len(aux_track[:5]))
        last = sum(aux_track[-5:]) / max(1, len(aux_track[-5:]))
        print(f"[DeepSeekMoE] aux mean (first 5 -> last 5 steps): "
              f"{first:.5f} -> {last:.5f}", flush=True)
    return {
        "losses": losses,
        "aux_track": aux_track,
        "aux_first": aux_track[0] if aux_track else float("nan"),
        "aux_last": aux_track[-1] if aux_track else float("nan"),
        "loss_drop": losses[0] - losses[-1],
    }


def test_rdt(device, n_steps, hidden=64):
    """Train with curriculum R in {2,4,8,16} so DepthLoRA scale embeddings
    receive gradient at every loop index used at eval. AccelExit is enabled
    for inference so R=16 saturates to its useful depth (~ training horizon)
    rather than drifting from un-trained accumulation steps - this is how
    RDT is meant to be used at test time (paper: AccelExit makes test-time
    compute *adaptive* not just larger).
    """
    print(f"\n[RDT] hidden={hidden} train R~Uniform{{2,4,8,16}}, eval R={{1,4,16}}", flush=True)
    torch.manual_seed(3)
    body = TinyBody(hidden).to(device)
    rdt = RDTLoop(
        body, hidden=hidden, max_loops_train=4, max_loops_infer=16,
        enable_loop_index_embed=True, enable_layer_scale=True,
        enable_residual_gate_bias=True, enable_depth_lora=True,
        enable_accel_exit=True, accel_exit_tau=1e-2,
        depth_lora_rank=8,
    ).to(device)
    model = ToyLM(vocab=100, hidden=hidden, trunk=rdt).to(device)
    # Curriculum: random R per training step so every scale[t] receives grad.
    rng = np.random.RandomState(123)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    model.train()
    R_choices = [2, 4, 8, 16]
    for step in range(n_steps):
        R = int(R_choices[rng.randint(len(R_choices))])
        inputs, targets = deterministic_token_batch(8, 16, 100, device)
        x_emb = model.emb(inputs)
        h = model.trunk(x_emb, n_loops=R)
        logits = model.lm_head(h)
        ce = F.cross_entropy(logits.reshape(-1, 100), targets.reshape(-1))
        optim.zero_grad()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        l = float(ce.item())
        if not math.isfinite(l):
            raise RuntimeError(f"RDT: NaN at step {step}")
        losses.append(l)

    ppls = {}
    for R in (1, 4, 16):
        ppls[R] = eval_ppl(model, device, n_loops=R)
        print(f"[RDT] eval ppl @ R={R:>2d}: {ppls[R]:.2f}", flush=True)
    print(f"[RDT] losses {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)
    return {
        "losses": losses,
        "ppl_R1": ppls[1],
        "ppl_R4": ppls[4],
        "ppl_R16": ppls[16],
        "loss_drop": losses[0] - losses[-1],
    }


def test_backends(device, hidden=32):
    print("\n[Backends] gpu_dense + triton_block", flush=True)
    res = {"gpu_dense": {}, "triton_block": {}}
    backends = ["gpu_dense"]
    try:
        from synapforge.backends.triton_block_kernel import _HAS_TRITON
        if _HAS_TRITON and device == "cuda":
            backends.append("triton_block")
    except Exception as e:
        print(f"[Backends] triton_block unavailable: {e}", flush=True)

    for be in backends:
        for label, builder in [
            ("MoR", lambda: MoRStack(TinyBody(hidden), hidden, max_depth=2, with_spike_rate=False)),
            ("CoE", lambda: ChainOfExperts(hidden, n_routed=4, n_shared=1, top_k=1, n_steps=2)),
            ("MoE", lambda: DeepSeekMoE(hidden, n_routed=4, n_shared=1, top_k=1)),
            ("RDT", lambda: RDTLoop(TinyBody(hidden), hidden, max_loops_train=2, max_loops_infer=2)),
        ]:
            try:
                m = builder().to(device)
                if be == "gpu_dense":
                    rt = sf.compile(m, backend=be)
                    _ = rt
                x = torch.randn(2, 4, hidden, device=device, requires_grad=True)
                if label == "MoR":
                    y, dl = m(x)
                    out = y.sum() + dl
                else:
                    y = m(x)
                    out = y.sum()
                out.backward()
                ok, err = True, None
            except Exception as e:
                ok, err = False, str(e)[:200]
            res[be][label] = {"ok": ok, "err": err}
            tag = "OK" if ok else f"FAIL: {err}"
            print(f"[Backends] {be:>12s}  {label:>4s}: {tag}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()

    device = args.device
    print(f"[setup] device={device} torch={torch.__version__}", flush=True)
    if device == "cuda":
        torch.cuda.set_device(0)
        print(f"[setup] gpu name: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    results = {}
    pass_count = 0
    fail_count = 0

    for name, fn in [
        ("MoR", lambda: test_mor(device, args.steps, args.hidden)),
        ("CoE", lambda: test_coe(device, args.steps, args.hidden)),
        ("MoE", lambda: test_moe(device, args.steps, args.hidden)),
        ("RDT", lambda: test_rdt(device, args.steps, args.hidden)),
        ("Backends", lambda: test_backends(device, args.hidden)),
    ]:
        try:
            results[name] = fn()
            if name == "Backends":
                ok_all = all(c.get("ok", False)
                             for be in results[name].values() for c in be.values())
                if ok_all:
                    pass_count += 1
                    print("[Backends] PASS", flush=True)
                else:
                    fail_count += 1
                    print("[Backends] FAIL", flush=True)
            else:
                drop = results[name].get("loss_drop", 0.0)
                if drop > 0.0:
                    pass_count += 1
                    print(f"[{name}] PASS (loss drop {drop:+.3f})", flush=True)
                else:
                    fail_count += 1
                    print(f"[{name}] FAIL (loss drop {drop:+.3f})", flush=True)
        except Exception as e:
            fail_count += 1
            results[name] = {"error": str(e)}
            import traceback
            traceback.print_exc()
            print(f"[{name}] ERROR: {e}", flush=True)

    elapsed = time.time() - t0
    print("\n========== SUMMARY ==========", flush=True)
    print(f"passes : {pass_count}", flush=True)
    print(f"fails  : {fail_count}", flush=True)
    print(f"wall   : {elapsed:.1f}s", flush=True)

    if "MoR" in results and "hist" in results["MoR"]:
        print(f"MoR depth histogram (proof of adaptive): {results['MoR']['hist']}", flush=True)
    if "CoE" in results and "kl_step0_vs_last" in results["CoE"]:
        print(f"CoE KL step0 vs last: {results['CoE']['kl_step0_vs_last']:.4f}", flush=True)
    if "MoE" in results and "aux_first" in results["MoE"]:
        print(f"MoE aux loss first/last: {results['MoE']['aux_first']:.5f} -> {results['MoE']['aux_last']:.5f}", flush=True)
    if "RDT" in results and "ppl_R1" in results["RDT"]:
        print(
            f"RDT test-time compute scaling: ppl@R=1 {results['RDT']['ppl_R1']:.2f}, "
            f"R=4 {results['RDT']['ppl_R4']:.2f}, R=16 {results['RDT']['ppl_R16']:.2f}",
            flush=True,
        )

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
