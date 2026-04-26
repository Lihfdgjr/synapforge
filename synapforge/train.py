"""sf.train — minimal training loop using synapforge primitives.

Wires together:
    * a sf.Module (or vanilla nn.Module — same interface) model
    * sf.optim.PlasticityAwareAdamW (multi-source gradient merging)
    * sf.plasticity.PlasticityEngine (Hebbian/STDP/BCM rules)
    * sf.data.ParquetTokenStream (token-pair iterator)

This is the *proof we replaced PyTorch* claim's training-loop side: while
``loss.backward()`` is still ``torch.autograd``, the optimizer step, the
parameter update, the plasticity injection, and the learning-rate / state
machinery all live in synapforge.

API
---

    >>> from synapforge.train import train
    >>> from synapforge.optim import build_optimizer
    >>> from synapforge.plasticity import PlasticityEngine
    >>> data = ParquetTokenStream("...", batch_size=32, seq_len=256)
    >>> metrics = train(model, iter(data), n_steps=100, lr=3e-4,
    ...                 plasticity_engine=engine,
    ...                 out_dir="/workspace/runs/sf_smoke")

Returns ``metrics`` dict::

    {"step": [...], "loss": [...], "step_ms": [...], "tok_per_s": [...]}

If a ``PlasticityEngine`` is passed, the loop uses it in two stages:
  1. **Inside the model's forward**, plasticity rules call
     ``rule.observe(pre=..., post=..., t=step)`` to update eligibility
     traces — the model is responsible for hooking observe in.
  2. **After loss.backward()**, ``engine.step(t, weight_dict)`` returns
     ``{name: dW}``. Each delta is registered on the ``MultiSourceParam``
     wrapper inside the optimizer, so the next ``optim.step()`` sees a
     fused gradient and updates W exactly once.

This split avoids the autograd "version mismatch" trap (mutating a tensor
read by autograd) — W is mutated by ONE owner (the optimizer) per step.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterator, Optional

import torch

from .optim import build_optimizer, PlasticityAwareAdamW

try:
    from .plasticity import PlasticityEngine
except ImportError:  # pragma: no cover — sibling-agent guard
    PlasticityEngine = None  # type: ignore[misc,assignment]


def _grad_norm(model: torch.nn.Module) -> float:
    """Return total L2 grad norm across all parameters (for monitoring)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def _attach_deltas_to_optim(
    deltas: dict[str, torch.Tensor],
    optim: PlasticityAwareAdamW,
    named_params: dict[str, torch.nn.Parameter],
    source_name: str,
) -> int:
    """Push engine-emitted deltas onto each MultiSourceParam under `source`.

    Skips deltas whose target parameter does not declare ``source_name`` as
    one of its grad sources (so we don't accidentally widen sources mid-run).
    Returns count of deltas attached.
    """
    n = 0
    for name, delta in deltas.items():
        param = named_params.get(name)
        if param is None:
            continue
        msp = optim.get_ms_param(param)
        if msp is None:
            continue
        if source_name not in msp.sources:
            continue
        # Auto-fix shape: we can broadcast a (out, in) delta onto a (out, in)
        # weight; if the engine emits a transposed shape, transpose it once.
        d = delta
        if d.shape != param.shape:
            if d.shape == param.shape[::-1]:
                d = d.t().contiguous()
            else:
                continue  # silently skip mismatched shapes
        msp.attach_plast_delta(source_name, d)
        n += 1
    return n


def train(
    model: torch.nn.Module,
    data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    n_steps: int,
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    grad_clip: float = 1.0,
    log_every: int = 10,
    save_every: int = 0,
    device: str = "cuda",
    out_dir: Optional[str] = None,
    plasticity_engine: Any = None,
    plasticity_source: str = "hebb",
    optimizer: Optional[PlasticityAwareAdamW] = None,
) -> dict[str, list]:
    """Train ``model`` for ``n_steps`` using synapforge optim + plasticity.

    Args:
        model: the network (sf.Module or any nn.Module).
        data_iter: iterator yielding ``(tokens_in [B,T], tokens_out [B,T])``
            int64 tensors on cpu. ``train`` moves them to ``device``.
        n_steps: number of optimizer steps.
        lr / weight_decay: AdamW hyperparameters.
        grad_clip: max L2 norm; use 0 to disable.
        log_every: print + record metrics every K steps.
        save_every: save checkpoint every K steps. ``0`` disables.
        device: cuda or cpu.
        out_dir: where to save checkpoints + final metrics. ``None`` skips.
        plasticity_engine: optional ``sf.plasticity.PlasticityEngine``.
        plasticity_source: name of the source the engine pushes deltas under
            (must match ``MultiSourceParam.sources`` declarations on params).
        optimizer: optional pre-built optimizer; ``None`` builds a default
            ``FusedAdamW`` from the model.

    Returns:
        dict of per-step metrics (subsampled by ``log_every``).
    """
    model = model.to(device)
    model.train()
    if optimizer is None:
        optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    named_params = dict(model.named_parameters())

    metrics: dict[str, list] = {
        "step": [], "loss": [], "step_ms": [],
        "tok_per_s": [], "grad_norm": [], "plast_n": [],
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    t_train_start = time.time()
    last_log_t = t_train_start

    for step in range(n_steps):
        t_step_start = time.time()
        try:
            x, y = next(data_iter)
        except StopIteration:
            print(f"[step {step}] data iterator exhausted; stopping early",
                  flush=True)
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ---------- forward ----------
        out = model(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        # ---------- backward ----------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = _grad_norm(model) if grad_clip > 0 else 0.0
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )

        # ---------- plasticity ----------
        plast_n = 0
        if plasticity_engine is not None:
            try:
                deltas = plasticity_engine.step(t=step, weight_dict=named_params)
            except RuntimeError as exc:
                print(f"[step {step}] plasticity error: {exc}", flush=True)
                deltas = {}
            plast_n = _attach_deltas_to_optim(
                deltas, optimizer, named_params, plasticity_source
            )

        # ---------- optimizer step (merges bp + plast) ----------
        optimizer.step()

        # ---------- metrics ----------
        if device == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.time() - t_step_start) * 1000.0
        tok_per_s = float(x.numel()) / max(step_ms / 1000.0, 1e-6)

        if step % log_every == 0:
            metrics["step"].append(step)
            metrics["loss"].append(float(loss.item()))
            metrics["step_ms"].append(step_ms)
            metrics["tok_per_s"].append(tok_per_s)
            metrics["grad_norm"].append(gnorm)
            metrics["plast_n"].append(plast_n)
            wall = time.time() - t_train_start
            since = time.time() - last_log_t
            last_log_t = time.time()
            print(
                f"step={step:5d}  loss={loss.item():7.4f}  "
                f"step_ms={step_ms:6.1f}  tok/s={tok_per_s:7.0f}  "
                f"|g|={gnorm:6.3f}  plast={plast_n}  "
                f"wall={wall:6.1f}s  dt={since:5.2f}s",
                flush=True,
            )

        if save_every and step > 0 and step % save_every == 0 and out_dir:
            ck_path = os.path.join(out_dir, f"step_{step:06d}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                ck_path,
            )
            print(f"[checkpoint] saved {ck_path}", flush=True)

    # ---------- summary ----------
    elapsed = time.time() - t_train_start
    print(
        f"\nTRAIN DONE  steps={n_steps}  elapsed={elapsed:.1f}s "
        f"({elapsed/60.0:.2f}min)",
        flush=True,
    )
    if metrics["loss"]:
        print(
            f"  initial_loss={metrics['loss'][0]:.4f}  "
            f"final_loss={metrics['loss'][-1]:.4f}  "
            f"avg_tok/s={sum(metrics['tok_per_s'])/len(metrics['tok_per_s']):.0f}",
            flush=True,
        )

    if out_dir is not None:
        final_path = os.path.join(out_dir, "metrics.pt")
        torch.save(metrics, final_path)
        print(f"[metrics] saved {final_path}", flush=True)

    return metrics


__all__ = ["train"]
