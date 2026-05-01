"""trainer_mixins -- opt-in advanced training components for train_100m_kd.

Three mixins, each default-off and isolated so a missing dependency or
data file makes the mixin a graceful no-op (training continues unchanged):

  * MultimodalMixin  -- run image/audio/etc encoders, add InfoNCE
                        contrastive aux loss between modal-hidden and
                        text-hidden (alpha=0.05 default).
  * SelfLearnMixin   -- at val time pull top-K high-CE examples, run a
                        single-step TestTimeTraining update, re-eval and
                        report the lift.
  * CuriosityMixin   -- compute the 6-signal curiosity reward (ICM
                        free-energy + ||delta W_STDP|| + HNSW gap +
                        spike-rate variance + novelty - noise) and
                        expose ``curiosity_loss(state)`` for the trainer.

Anchors:
  - synapforge.modal.UnifiedEmbed            (image/audio/.. byte-patch)
  - synapforge.intrinsic.FreeEnergySurprise  (ICM 1705.05363)
  - synapforge.intrinsic.NoveltyDrive        (running-EMA novelty)
  - synapforge.curiosity.CuriosityScorer     (6-signal STDP-driven reward)
  - synapforge.self_learn.TestTimeTraining   (TTT 1909.13231)
  - synapforge.self_learn.ExperienceReplayBuffer (DER++ 2004.07211)

Each mixin has a ``smoke()`` classmethod that runs on dummy tensors with
no real model required, so they can be unit-tested in isolation. The
``__main__`` block runs all three smokes in sequence and prints a summary.

Constraints:
  * Default off. ``train_100m_kd`` only constructs a mixin when its
    matching CLI flag is non-default.
  * Failures must not kill training. Each forward call wraps the body in
    a try/except and returns a zero contribution on error.
  * No new heavy dependencies -- everything ships with synapforge already.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. MultimodalMixin -- byte-patch encoders + InfoNCE contrastive aux
# =============================================================================


class MultimodalMixin:
    """Run modality encoders and add an InfoNCE contrastive auxiliary loss.

    The mixin is constructed with a list of modalities (subset of
    ``["image", "audio"]`` by default) and a directory of pre-tokenised
    modal samples. At each training step the trainer calls
    ``contrastive_loss(text_hidden)`` which:

      1) Pulls one sample per modality from the data dir (cycled).
      2) Runs the matching ``synapforge.modal.*PatchEmbed``.
      3) Mean-pools the modal hidden over time -> ``z_modal`` (B, d).
      4) Mean-pools text_hidden -> ``z_text`` (B, d).
      5) Returns ``alpha * InfoNCE(z_text, z_modal)``.

    On any failure (missing data, encoder not built, shape mismatch) it
    returns a zero loss with grad-attached so ``.backward()`` is safe.

    Default ``alpha=0.05`` is small enough that the contrastive signal
    nudges the backbone toward modality-aware reps without dominating
    the LM CE+KD loss.
    """

    SUPPORTED = ("image", "audio")

    def __init__(
        self,
        model: nn.Module,
        modal_list: Sequence[str] = (),
        modal_data_dir: str = "",
        hidden: int = 512,
        alpha: float = 0.05,
        temperature: float = 0.1,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.modal_list = tuple(m for m in modal_list if m in self.SUPPORTED)
        self.modal_data_dir = str(modal_data_dir) if modal_data_dir else ""
        self.hidden = int(hidden)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.verbose = bool(verbose)
        self.encoders: dict[str, nn.Module] = {}
        self._sample_idx = 0
        self._cached_samples: dict[str, list[torch.Tensor]] = {}
        self.enabled = bool(self.modal_list) and bool(self.modal_data_dir)

        if self.enabled:
            self._build_encoders()
            self._scan_data_dir()
        elif self.verbose and modal_list:
            warnings.warn(
                "MultimodalMixin: modal-list given but modal-data-dir empty; "
                "mixin disabled (no-op).",
                RuntimeWarning,
                stacklevel=2,
            )

    # ----------------------------------------------------------- internal
    def _build_encoders(self) -> None:
        try:
            from .modal.image import ImagePatchEmbed
            from .modal.audio import AudioPatchEmbed
        except Exception as exc:  # pragma: no cover -- defensive
            warnings.warn(f"MultimodalMixin: cannot import modal encoders: {exc}",
                          RuntimeWarning, stacklevel=2)
            self.enabled = False
            return
        if "image" in self.modal_list:
            self.encoders["image"] = ImagePatchEmbed(hidden=self.hidden, patch=8)
        if "audio" in self.modal_list:
            self.encoders["audio"] = AudioPatchEmbed(
                hidden=self.hidden, sample_rate=16000, chunk_ms=20, mode="raw",
            )
        # Move to model device when possible.
        try:
            dev = next(self.model.parameters()).device
            for m in self.encoders.values():
                m.to(dev)
        except StopIteration:
            pass

    def _scan_data_dir(self) -> None:
        if not self.modal_data_dir or not os.path.isdir(self.modal_data_dir):
            if self.verbose:
                warnings.warn(
                    f"MultimodalMixin: data dir {self.modal_data_dir!r} missing; "
                    "no-op.",
                    RuntimeWarning, stacklevel=2,
                )
            self.enabled = False
            return
        for m in self.modal_list:
            sub = os.path.join(self.modal_data_dir, m)
            if not os.path.isdir(sub):
                self._cached_samples[m] = []
                continue
            files = sorted(
                os.path.join(sub, f) for f in os.listdir(sub)
                if f.endswith(".pt")
            )[:64]
            self._cached_samples[m] = files

    def _next_sample(self, modality: str, batch_size: int) -> Optional[torch.Tensor]:
        files = self._cached_samples.get(modality, [])
        if not files:
            return None
        path = files[self._sample_idx % len(files)]
        self._sample_idx += 1
        try:
            t = torch.load(path, map_location="cpu")
        except Exception:
            return None
        if t.dim() == 3 and modality == "image":
            t = t.unsqueeze(0)
        if t.dim() == 1 and modality == "audio":
            t = t.unsqueeze(0)
        # Tile to batch_size if needed.
        if t.shape[0] < batch_size:
            reps = (batch_size + t.shape[0] - 1) // t.shape[0]
            t = t.repeat(reps, *([1] * (t.dim() - 1)))[:batch_size]
        else:
            t = t[:batch_size]
        return t

    # ----------------------------------------------------------- public
    def parameters(self):
        for m in self.encoders.values():
            yield from m.parameters()

    def contrastive_loss(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """InfoNCE between mean-pooled text and modal hidden.

        text_hidden: (B, T, d). Returns scalar tensor (alpha-scaled).
        On any error returns a zero scalar with grad still attached to
        text_hidden so .backward() is safe.
        """
        zero = (text_hidden.float().sum() * 0.0)
        if not self.enabled:
            return zero
        try:
            B = text_hidden.shape[0]
            d = text_hidden.shape[-1]
            if d != self.hidden:
                return zero
            z_text = text_hidden.mean(dim=1)  # (B, d)
            losses = []
            dev = text_hidden.device
            dtype = text_hidden.dtype
            for m in self.modal_list:
                enc = self.encoders.get(m)
                if enc is None:
                    continue
                sample = self._next_sample(m, B)
                if sample is None:
                    continue
                sample = sample.to(dev)
                z_mod = enc(sample)              # (B, T_m, d)
                if z_mod.dim() != 3 or z_mod.shape[-1] != d:
                    continue
                z_mod = z_mod.mean(dim=1).to(dtype)
                # InfoNCE: positives on diagonal, negatives across batch.
                z_t = F.normalize(z_text.float(), dim=-1)
                z_m = F.normalize(z_mod.float(), dim=-1)
                logits = z_t @ z_m.t() / max(self.temperature, 1e-3)
                labels = torch.arange(B, device=dev)
                losses.append(F.cross_entropy(logits, labels))
            if not losses:
                return zero
            return self.alpha * torch.stack(losses).mean()
        except Exception as exc:  # pragma: no cover -- defensive
            if self.verbose:
                warnings.warn(f"MultimodalMixin.contrastive_loss failed: {exc}",
                              RuntimeWarning, stacklevel=2)
            return zero

    @classmethod
    def smoke(cls) -> dict[str, Any]:
        """Run a self-contained sanity check on dummy tensors.

        Builds a tiny ImagePatchEmbed, fakes a text_hidden, and verifies
        contrastive_loss returns a finite scalar. No real model needed.
        """
        torch.manual_seed(0)
        hidden = 64
        # Construct a dummy "model" parameter so .parameters() works.
        dummy = nn.Linear(hidden, hidden)
        mix = cls(dummy, modal_list=(), modal_data_dir="", hidden=hidden,
                  verbose=False)
        text_h = torch.randn(2, 8, hidden, requires_grad=True)
        # Disabled path: must return zero with grad-attached.
        z = mix.contrastive_loss(text_h)
        assert torch.isfinite(z), "disabled path returned non-finite"
        assert float(z.detach()) == 0.0, "disabled path must be zero"
        z.backward()  # should not raise
        # Enabled path with a fake encoder bypassing data dir.
        mix.enabled = True
        from .modal.image import ImagePatchEmbed
        enc = ImagePatchEmbed(hidden=hidden, patch=8)
        mix.encoders["image"] = enc
        # Bypass file IO -- inject one in-memory sample.
        sample = torch.randn(2, 3, 16, 16)
        orig = mix._next_sample
        mix._next_sample = lambda mod, B: sample[:B]  # type: ignore
        mix.modal_list = ("image",)
        text_h2 = torch.randn(2, 8, hidden, requires_grad=True)
        z2 = mix.contrastive_loss(text_h2)
        assert torch.isfinite(z2), "enabled path non-finite"
        z2.backward()
        mix._next_sample = orig
        return {
            "ok": True,
            "disabled_loss": 0.0,
            "enabled_loss": float(z2.detach()),
            "alpha": mix.alpha,
        }


# =============================================================================
# 2. SelfLearnMixin -- TTT lift on hardest val examples
# =============================================================================


class SelfLearnMixin:
    """At eval time, take top-K high-CE examples and run a 1-step TTT update.

    Workflow each call:
      1) Caller provides ``(x, y, per_example_ce)`` over the val set.
      2) Mixin picks the top-K (default 8) highest-CE examples.
      3) Snapshots the model weights matched by ``ttt.params_filter``.
      4) Runs ``ttt.adapt(x_topk)`` -- a single SGD step on those filtered
         params using a self-supervised reconstruction loss.
      5) Re-runs forward, reports CE-after, restores the snapshot so
         training continues from the original weights.
      6) Returns lift = mean(CE_before) - mean(CE_after).

    Designed to be cheap (8 examples, 1 SGD step, weights restored) and
    informative (positive lift = self-learn is helping; near-zero or
    negative = not yet ready).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        k_failures: int = 8,
        ttt_lr: float = 1e-3,
        ttt_steps: int = 1,
        params_filter: Sequence[str] = ("shared_tau", "fast.w_", "ttt_",
                                        "tau", "threshold"),
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer  # may be unused (TTT builds its own SGD)
        self.k_failures = int(k_failures)
        self.ttt_lr = float(ttt_lr)
        self.ttt_steps = int(ttt_steps)
        self.params_filter = tuple(params_filter)
        self.verbose = bool(verbose)
        self._ttt = None
        self.last_lift: float = 0.0
        self.history: list[dict[str, float]] = []
        try:
            from .self_learn import TestTimeTraining
            self._ttt = TestTimeTraining(
                model, inner_lr=self.ttt_lr, steps=self.ttt_steps,
                params_filter=self.params_filter,
            )
        except Exception as exc:  # pragma: no cover -- defensive
            warnings.warn(f"SelfLearnMixin: cannot build TTT ({exc}); no-op.",
                          RuntimeWarning, stacklevel=2)
            self._ttt = None

    @torch.no_grad()
    def _per_example_ce(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 3:
            V = out.shape[-1]
            flat_logits = out.reshape(-1, V).float()
            flat_y = y.reshape(-1)
            ce = F.cross_entropy(flat_logits, flat_y, reduction="none")
            # average per-sequence
            ce = ce.reshape(y.shape).mean(dim=-1)
        else:
            ce = (out - y).pow(2).reshape(out.shape[0], -1).mean(dim=-1)
        return ce.detach()

    def adapt_on_failures(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Pick top-K high-CE samples, TTT-adapt, re-eval, return lift.

        Returns a dict with ``before / after / lift / k`` (all 0.0 if the
        mixin is no-op or the call raises).
        """
        if self._ttt is None:
            return {"before": 0.0, "after": 0.0, "lift": 0.0, "k": 0}
        try:
            ce = self._per_example_ce(x, y)
            B = ce.shape[0]
            k = min(self.k_failures, B)
            topk = torch.topk(ce, k=k).indices
            x_top = x[topk]
            y_top = y[topk]
            before = float(ce[topk].mean().item())
            # TTT inner loop adapts filtered params; we restore after.
            try:
                self._ttt.adapt(x_top, restore_after=False)
            except Exception as exc:
                if self.verbose:
                    warnings.warn(f"SelfLearnMixin.adapt inner failed: {exc}",
                                  RuntimeWarning, stacklevel=2)
                self._ttt.restore()
                return {"before": before, "after": before, "lift": 0.0, "k": k}
            ce_after = self._per_example_ce(x_top, y_top)
            after = float(ce_after.mean().item())
            self._ttt.restore()
            lift = before - after
            self.last_lift = lift
            rec = {"before": before, "after": after, "lift": lift, "k": k}
            self.history.append(rec)
            return rec
        except Exception as exc:  # pragma: no cover -- defensive
            if self.verbose:
                warnings.warn(f"SelfLearnMixin.adapt_on_failures failed: {exc}",
                              RuntimeWarning, stacklevel=2)
            return {"before": 0.0, "after": 0.0, "lift": 0.0, "k": 0}

    @classmethod
    def smoke(cls) -> dict[str, Any]:
        """Train a tiny LM on dummy tokens, verify TTT lift call works."""
        torch.manual_seed(0)
        V, T, d = 32, 8, 16

        class TinyLM(nn.Module):
            def __init__(self, V: int, d: int) -> None:
                super().__init__()
                self.emb = nn.Embedding(V, d)
                self.lin = nn.Linear(d, V)
                self.tau = nn.Parameter(torch.ones(1))  # matches "tau" filter

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(self.emb(x) * self.tau)

        m = TinyLM(V, d)
        opt = torch.optim.SGD(m.parameters(), lr=1e-2)
        mix = cls(m, optimizer=opt, k_failures=2, params_filter=("tau",),
                  verbose=False)
        x = torch.randint(0, V, (4, T))
        y = torch.randint(0, V, (4, T))
        rec = mix.adapt_on_failures(x, y)
        assert "lift" in rec, "missing lift"
        assert rec["k"] == 2, f"expected k=2, got {rec['k']}"
        # tau should be restored to 1.0 (snapshot/restore).
        assert abs(float(m.tau.detach()) - 1.0) < 1e-5, (
            f"tau not restored: {float(m.tau.detach())}"
        )
        return {"ok": True, **rec}


# =============================================================================
# 3. CuriosityMixin -- 6-signal STDP-driven curiosity reward
# =============================================================================


class CuriosityMixin:
    """Compute the 6-signal STDP-driven curiosity reward.

    Formula (per ``synapforge.curiosity``):

        C = 0.40 * delta_F             # ICM free-energy reduction
          + 0.25 * ||delta W_STDP||    # synaptic change (the load-bearing
                                       #   signal that beats noisy-TV)
          + 0.15 * G_HNSW              # retrieval coverage gap
          + 0.10 * H[spike]            # spike-rate variance (engagement)
          + 0.05 * N                   # NoveltyDrive EMA
          - 0.05 * V_noise             # noise-variance penalty

    The mixin owns:
      * ``FreeEnergySurprise`` (a small forward-model predictor, the only
        signal in the formula that produces an autograd-visible loss).
      * ``NoveltyDrive`` (running EMA over hidden states; novelty signal).

    ``curiosity_loss(state)`` returns a torch scalar shaped for the
    trainer to multiply by ``--curiosity-weight`` and add to the LM loss
    (so the predictor learns to model state transitions). The other 5
    signals are bookkeeping: they're returned in ``state["curiosity_score"]``
    for logging, but only ``delta_F`` flows gradient to the backbone.

    ``state`` is a dict the trainer builds. Keys we use (all optional):
      h_prev, h_next       : (..., d) hidden tensors for ICM
      stdp_module          : module exposing ``.W`` for ||delta W||
      hnsw_index, query    : optional (HNSW retrieval gap)
      plif_modules         : list of PLIF cells with .last_spike_rate
      noise_variance       : float (e.g., grad-norm variance estimator)

    Missing keys reduce that signal to 0; mixin always returns a finite
    scalar.
    """

    def __init__(
        self,
        model: nn.Module,
        hidden: int = 512,
        ema: float = 0.99,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.hidden = int(hidden)
        self.verbose = bool(verbose)
        self._fe = None
        self._novelty = None
        self._scorer = None
        self.last_score = None
        try:
            from .intrinsic import FreeEnergySurprise, NoveltyDrive
            from .curiosity import (
                CuriosityScorer, stdp_delta_norm_from_module,
                spike_rate_variance_from_modules, retrieval_gap_from_hnsw,
            )
            self._fe = FreeEnergySurprise(hidden_size=self.hidden)
            self._novelty = NoveltyDrive(hidden_size=self.hidden, ema=ema)
            self._scorer = CuriosityScorer()
            self._stdp_norm_fn = stdp_delta_norm_from_module
            self._spike_var_fn = spike_rate_variance_from_modules
            self._hnsw_fn = retrieval_gap_from_hnsw
            try:
                dev = next(model.parameters()).device
                self._fe.to(dev)
                self._novelty.to(dev)
            except StopIteration:
                pass
        except Exception as exc:  # pragma: no cover -- defensive
            warnings.warn(
                f"CuriosityMixin: cannot wire intrinsic/curiosity modules "
                f"({exc}); no-op.",
                RuntimeWarning, stacklevel=2,
            )
            self._fe = None
            self._novelty = None
            self._scorer = None

    def parameters(self):
        if self._fe is not None:
            yield from self._fe.parameters()
        if self._novelty is not None:
            yield from self._novelty.parameters()

    def curiosity_loss(self, state: dict) -> torch.Tensor:
        """Return scalar curiosity loss (autograd-attached via free-energy).

        On any error returns a zero scalar with grad attached to whichever
        hidden tensor is in ``state`` (or a fresh zero if none).
        """
        # Pick a fallback tensor we can attach a zero gradient to. We avoid
        # ``a or b`` because that evaluates a torch.Tensor as bool, which
        # raises for any multi-element tensor.
        h_any = state.get("h_next")
        if not isinstance(h_any, torch.Tensor):
            h_any = state.get("h_prev")
        if isinstance(h_any, torch.Tensor):
            zero = (h_any.float().sum() * 0.0)
        else:
            zero = torch.zeros((), requires_grad=False)
        if self._fe is None or self._scorer is None:
            return zero
        try:
            h_prev = state.get("h_prev")
            h_next = state.get("h_next")
            # ---- 1. delta_F (ICM surprise, autograd-visible) ----
            if (isinstance(h_prev, torch.Tensor) and
                    isinstance(h_next, torch.Tensor)):
                # Flatten leading dims to (N, d) so FreeEnergySurprise's
                # MLP sees the right shape.
                d = self.hidden
                hp = h_prev.reshape(-1, h_prev.shape[-1])
                hn = h_next.reshape(-1, h_next.shape[-1])
                if hp.shape[-1] != d:
                    return zero
                delta_f_loss = self._fe.surprise(hp, hn)
                delta_f_val = float(delta_f_loss.detach().item())
            else:
                delta_f_loss = zero
                delta_f_val = 0.0

            # ---- 2. ||delta W_STDP|| (bookkeeping only) ----
            stdp_module = state.get("stdp_module")
            stdp_val = (float(self._stdp_norm_fn(stdp_module))
                        if stdp_module is not None else 0.0)

            # ---- 3. HNSW retrieval gap (bookkeeping) ----
            hnsw = state.get("hnsw_index")
            query = state.get("query_embedding")
            if hnsw is not None and query is not None:
                gap = float(self._hnsw_fn(hnsw, query))
            else:
                gap = 0.0

            # ---- 4. spike-rate variance (bookkeeping) ----
            plif = state.get("plif_modules") or []
            spike_var = float(self._spike_var_fn(plif))

            # ---- 5. NoveltyDrive (bookkeeping; updates EMA in place) ----
            novelty_val = 0.0
            if isinstance(h_next, torch.Tensor) and h_next.shape[-1] == self.hidden:
                with torch.no_grad():
                    novelty_val = float(self._novelty.novelty(h_next).item())

            # ---- 6. noise variance (passed in by trainer) ----
            noise_var = float(state.get("noise_variance", 0.0))

            score = self._scorer.score(
                free_energy_reduction=delta_f_val,
                stdp_delta_norm=stdp_val,
                retrieval_gap=gap,
                spike_rate_variance=spike_var,
                novelty_ema=novelty_val,
                noise_variance=noise_var,
            )
            self.last_score = score
            state["curiosity_score"] = score
            # Only delta_F's autograd path remains; others were detached.
            return delta_f_loss
        except Exception as exc:  # pragma: no cover -- defensive
            if self.verbose:
                warnings.warn(f"CuriosityMixin.curiosity_loss failed: {exc}",
                              RuntimeWarning, stacklevel=2)
            return zero

    @classmethod
    def smoke(cls) -> dict[str, Any]:
        """Verify curiosity_loss runs end-to-end on dummy tensors."""
        torch.manual_seed(0)
        d = 32
        dummy = nn.Linear(d, d)
        mix = cls(dummy, hidden=d, verbose=False)
        if mix._fe is None:
            return {"ok": False, "reason": "intrinsic/curiosity import failed"}
        h_prev = torch.randn(2, 4, d, requires_grad=True)
        h_next = torch.randn(2, 4, d, requires_grad=True)
        state = {
            "h_prev": h_prev,
            "h_next": h_next,
            "noise_variance": 0.05,
            "plif_modules": [],
            "stdp_module": None,
        }
        loss = mix.curiosity_loss(state)
        assert torch.isfinite(loss), "curiosity loss non-finite"
        loss.backward()
        score = state.get("curiosity_score")
        return {
            "ok": True,
            "loss": float(loss.detach()),
            "score_total": (float(score.total) if score else 0.0),
            "delta_f": (score.delta_f if score else 0.0),
        }


# =============================================================================
# Smoke runner
# =============================================================================


def _run_all_smokes() -> int:
    """Run all 3 mixin smokes in sequence; returns 0 on success."""
    import json
    results = {}
    for name, mixin in [
        ("MultimodalMixin", MultimodalMixin),
        ("SelfLearnMixin", SelfLearnMixin),
        ("CuriosityMixin", CuriosityMixin),
    ]:
        try:
            results[name] = mixin.smoke()
        except Exception as exc:
            results[name] = {"ok": False, "error": repr(exc)}
    print(json.dumps(results, indent=2, default=str))
    all_ok = all(r.get("ok") for r in results.values())
    return 0 if all_ok else 1


__all__ = [
    "MultimodalMixin",
    "SelfLearnMixin",
    "CuriosityMixin",
]


if __name__ == "__main__":
    import sys
    sys.exit(_run_all_smokes())
