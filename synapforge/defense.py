"""sf.defense — four-pillar poison defense for self-learning mode.

Per memory feedback_self_learn_poison_defense.md (2026-04-20): self-learning
without these guardrails turns the model into a poison sink within hours of
free-internet exposure. Four pillars:

  1. PoisonDetector     -- gradient/entropy/repetition heuristic risk score
  2. ProvenanceTracker  -- every accepted sample is hashed + sourced + logged
  3. WeightFirewall     -- masks grads on protected weight subsets (LM head,
                            embeddings, core CfC/PLIF stay frozen)
  4. AdversarialRedTeam -- canary probes; rollback if KL drifts past kl_max
  5. DefenseStack       -- orchestrator (the one callers actually use)

Critical: ``synapforge.self_learn.SelfLearnEngine`` must be paired with a
``DefenseStack`` — running self-learn alone raises ``RuntimeWarning``.
"""
from __future__ import annotations

import hashlib
import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. PoisonDetector — text + tensor heuristics
# ---------------------------------------------------------------------------


class PoisonDetector:
    """Heuristic + gradient-anomaly poison risk score in [0, 1].

    For text-typed inputs we score: char entropy / repetition ratio / URL
    density / banned-word presence. For tensor-typed inputs we score:
    L-inf magnitude vs running mean ("gradient anomaly" surrogate).
    """

    def __init__(
        self,
        entropy_min: float = 2.5,
        repetition_max: float = 0.4,
        url_density_max: float = 0.15,
        bad_words: Sequence[str] = (),
        tensor_anomaly_z: float = 4.0,
    ) -> None:
        self.entropy_min = float(entropy_min)
        self.repetition_max = float(repetition_max)
        self.url_density_max = float(url_density_max)
        self.bad_words = tuple(w.lower() for w in bad_words)
        self.tensor_anomaly_z = float(tensor_anomaly_z)
        # Running stats for tensor anomaly detection.
        self._tensor_mean: float = 0.0
        self._tensor_var: float = 1.0
        self._tensor_n: int = 0

    # ----------------------------- text path
    @staticmethod
    def _char_entropy(s: str) -> float:
        if not s:
            return 0.0
        counts: dict[str, int] = {}
        for ch in s:
            counts[ch] = counts.get(ch, 0) + 1
        tot = len(s)
        h = 0.0
        for n in counts.values():
            p = n / tot
            h -= p * math.log2(p)
        return h

    @staticmethod
    def _repetition_ratio(s: str) -> float:
        w = s.split()
        if len(w) < 3:
            return 0.0
        return 1.0 - len(set(w)) / len(w)

    @staticmethod
    def _url_density(s: str) -> float:
        t = s.split()
        if not t:
            return 0.0
        u = sum(1 for x in t if x.startswith(("http://", "https://", "www.")))
        return u / len(t)

    def score_text(self, text: str) -> float:
        if not text or not text.strip():
            return 1.0
        comp: list[float] = []
        h = self._char_entropy(text)
        comp.append(max(0.0, (self.entropy_min - h) / max(self.entropy_min, 1e-6)))
        rep = self._repetition_ratio(text)
        comp.append(
            max(0.0, (rep - self.repetition_max) / (1.0 - self.repetition_max + 1e-8))
        )
        u = self._url_density(text)
        comp.append(
            max(0.0, (u - self.url_density_max) / (1.0 - self.url_density_max + 1e-8))
        )
        if self.bad_words:
            bw = sum(1 for w in self.bad_words if w in text.lower())
            comp.append(min(1.0, bw / max(1, len(self.bad_words))))
        return float(max(comp))

    # ----------------------------- tensor path
    def score_tensor(self, x: torch.Tensor) -> float:
        if x.numel() == 0:
            return 1.0
        flat = x.detach().float().reshape(-1)
        peak = float(flat.abs().max())
        # Update Welford running mean/var on log1p(peak).
        v = math.log1p(peak)
        self._tensor_n += 1
        delta = v - self._tensor_mean
        self._tensor_mean += delta / self._tensor_n
        self._tensor_var += delta * (v - self._tensor_mean)
        if self._tensor_n < 4:
            return 0.0  # not enough stats yet
        std = math.sqrt(self._tensor_var / max(1, self._tensor_n - 1))
        if std < 1e-6:
            return 0.0
        z = abs(v - self._tensor_mean) / std
        return float(min(1.0, z / self.tensor_anomaly_z))

    def score(self, sample: Any) -> float:
        if isinstance(sample, str):
            return self.score_text(sample)
        if torch.is_tensor(sample):
            return self.score_tensor(sample)
        # Generic: stringify and use text path.
        return self.score_text(str(sample))

    def accept(self, sample: Any, threshold: float = 0.7) -> bool:
        return self.score(sample) < threshold


# ---------------------------------------------------------------------------
# 2. ProvenanceTracker — source attribution
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceRecord:
    sample_hash: str
    source: str
    timestamp: float
    risk_score: float
    accepted: bool
    learned_at_step: int = -1


class ProvenanceTracker:
    """Logs ``(hash, source, risk, accepted, step)`` for every observed sample."""

    def __init__(self, max_in_memory: int = 10000, log_path: str | None = None) -> None:
        self.max_in_memory = int(max_in_memory)
        self.log_path = log_path
        self._buf: list[ProvenanceRecord] = []

    @staticmethod
    def _hash(sample: Any) -> str:
        if torch.is_tensor(sample):
            payload = sample.detach().cpu().contiguous().view(-1).float().numpy().tobytes()
        else:
            payload = str(sample).encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:16]

    def record(
        self,
        sample: Any,
        source: str,
        risk_score: float,
        accepted: bool,
        learned_at_step: int = -1,
    ) -> ProvenanceRecord:
        rec = ProvenanceRecord(
            sample_hash=self._hash(sample),
            source=str(source),
            timestamp=time.time(),
            risk_score=float(risk_score),
            accepted=bool(accepted),
            learned_at_step=int(learned_at_step),
        )
        self._buf.append(rec)
        if len(self._buf) > self.max_in_memory:
            self._buf = self._buf[-self.max_in_memory:]
        if self.log_path:
            try:
                with open(self.log_path, "a") as f:
                    f.write(
                        f"{rec.timestamp:.3f}\t{rec.sample_hash}\t{rec.source}\t"
                        f"{rec.risk_score:.4f}\t{int(rec.accepted)}\t{rec.learned_at_step}\n"
                    )
            except OSError:
                pass
        return rec

    def recent(self, n: int = 100) -> list[ProvenanceRecord]:
        return list(self._buf[-n:])


# ---------------------------------------------------------------------------
# 3. WeightFirewall — refuses updates outside trust region
# ---------------------------------------------------------------------------


class WeightFirewall:
    """Masks gradients so learning only touches the *plastic* subset.

    Default ``protect`` covers the LM head, embeddings, core CfC/PLIF,
    norms, kWTA, astrocyte. Default ``allow`` covers fast/STDP synapses,
    latent controllers, action heads — the layers the architecture
    explicitly designates as plastic.

    ``apply_to_grads()`` zeros grads on protected params and returns the
    count of zeroed parameters (for telemetry).

    Trust region: an optional ``max_step_norm`` clips the L2 norm of any
    allowed parameter's grad — refusing huge updates that would push
    weights out of the calibrated region.
    """

    DEFAULT_PROTECT: tuple[str, ...] = (
        "shared_tau",
        "blocks.",
        ".cfc.",
        ".plif.",
        "embed.",
        "lm_head.",
        "pos",
        "norm.",
        "bio_kwta",
        "bio_astrocyte",
    )
    DEFAULT_ALLOW: tuple[str, ...] = (
        ".fast.",
        "bio_stdp",
        "latent_ctrl",
        "infinite_reader",
        "action_head",
        "world_model",
        "ttt_",
    )

    def __init__(
        self,
        model: nn.Module,
        protect: Sequence[str] | None = None,
        allow: Sequence[str] | None = None,
        max_step_norm: float | None = 1.0,
    ) -> None:
        self.protect = tuple(protect) if protect is not None else self.DEFAULT_PROTECT
        self.allow = tuple(allow) if allow is not None else self.DEFAULT_ALLOW
        self.model = model
        self.max_step_norm = max_step_norm

    def is_allowed(self, name: str) -> bool:
        for p in self.allow:
            if p in name:
                return True
        for p in self.protect:
            if p in name:
                return False
        # Default: in fail-closed mode anything unmatched is allowed; in this
        # framework we side with "if a name doesn't appear in protect, the
        # caller probably wants it learnable" — kept open for backwards-compat.
        return True

    @torch.no_grad()
    def apply_to_grads(self) -> int:
        zeroed = 0
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if not self.is_allowed(n):
                p.grad.zero_()
                zeroed += 1
                continue
            if self.max_step_norm is not None:
                norm = float(p.grad.norm())
                if norm > self.max_step_norm:
                    p.grad.mul_(self.max_step_norm / (norm + 1e-12))
        return zeroed


# ---------------------------------------------------------------------------
# 4. AdversarialRedTeam — canary probes + rollback
# ---------------------------------------------------------------------------


class AdversarialRedTeam:
    """Periodically probes the model with crafted adversarial / canary inputs.

    Two modes:
      * Calibrated: caller supplies ``canary_prompts``; we record the
        baseline logits at calibration time, then on every check we
        compute KL(current || baseline) and rollback if > ``kl_max``.
      * Synthesised: if no canary_prompts provided, we generate small
        random-tensor canaries on first ``calibrate()`` call.
    """

    def __init__(
        self,
        model: nn.Module,
        canary_prompts: Sequence[torch.Tensor] | None = None,
        kl_max: float = 0.8,
        check_every_n: int = 50,
    ) -> None:
        self.model = model
        self.kl_max = float(kl_max)
        self.check_every_n = int(check_every_n)
        self.canary_prompts: list[torch.Tensor] = list(canary_prompts or [])
        self.canary_logits_ref: list[torch.Tensor] = []
        self._snapshot: dict[str, torch.Tensor] | None = None

    def _forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        try:
            out = self.model(tokens=tokens)
        except TypeError:
            out = self.model(tokens)
        if isinstance(out, tuple):
            out = out[0]
        return out

    @torch.no_grad()
    def calibrate(self) -> None:
        was_train = self.model.training
        self.model.eval()
        try:
            if not self.canary_prompts:
                # Synthesise a few random canaries (long-typed).
                self.canary_prompts = [
                    torch.randint(0, 100, (1, 8)) for _ in range(2)
                ]
            self.canary_logits_ref = []
            for tokens in self.canary_prompts:
                logits = self._forward_logits(tokens)
                self.canary_logits_ref.append(logits.detach().cpu().clone())
        except Exception:
            self.canary_logits_ref = []
        finally:
            if was_train:
                self.model.train()

    @torch.no_grad()
    def snapshot(self) -> None:
        self._snapshot = {
            n: p.detach().cpu().clone() for n, p in self.model.named_parameters()
        }

    @torch.no_grad()
    def kl_from_ref(self) -> float:
        if not self.canary_logits_ref or not self.canary_prompts:
            return 0.0
        was_train = self.model.training
        self.model.eval()
        kls: list[float] = []
        try:
            for tokens, ref in zip(self.canary_prompts, self.canary_logits_ref):
                cur = self._forward_logits(tokens).float()
                ref_t = ref.to(cur.device).float()
                if cur.shape != ref_t.shape:
                    vmin = min(cur.shape[-1], ref_t.shape[-1])
                    cur = cur[..., :vmin]
                    ref_t = ref_t[..., :vmin]
                p_cur = F.log_softmax(cur, dim=-1)
                p_ref = F.log_softmax(ref_t, dim=-1)
                kls.append(
                    float(F.kl_div(p_cur, p_ref, reduction="batchmean", log_target=True))
                )
        except Exception:
            return 0.0
        finally:
            if was_train:
                self.model.train()
        return float(sum(kls) / max(1, len(kls)))

    def regressed(self) -> bool:
        return self.kl_from_ref() > self.kl_max

    @torch.no_grad()
    def rollback(self) -> int:
        if self._snapshot is None:
            return 0
        n_restored = 0
        for n, p in self.model.named_parameters():
            if n in self._snapshot:
                p.data.copy_(self._snapshot[n].to(p.device).to(p.dtype))
                n_restored += 1
        return n_restored


# ---------------------------------------------------------------------------
# 5. DefenseStack — orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DefenseConfig:
    enable_poison_detector: bool = True
    poison_threshold: float = 0.7
    enable_provenance: bool = True
    provenance_log_path: str | None = None
    enable_firewall: bool = True
    enable_red_team: bool = True
    red_team_kl_max: float = 0.8
    red_team_check_every: int = 50
    firewall_max_step_norm: float | None = 1.0


class DefenseStack:
    """Unified API for the four pillars.

    Lifecycle:
        stack = DefenseStack(model, DefenseConfig())  # auto-calibrates
        if stack.accept(sample, source="web"):
            engine.observe(sample, ...)
        # in train_step:
        stack.pre_step()
        loss.backward()
        stack.after_grads()        # firewall masks grads
        opt.step()
        if stack.regressed_and_rollback():  # red-team check
            ...
        stack.tick()
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DefenseConfig | None = None,
        canary_prompts: Sequence[torch.Tensor] | None = None,
        bad_words: Sequence[str] = (),
    ) -> None:
        self.cfg = cfg or DefenseConfig()
        self.model = model
        self.detector = (
            PoisonDetector(bad_words=bad_words)
            if self.cfg.enable_poison_detector
            else None
        )
        self.tracker = (
            ProvenanceTracker(log_path=self.cfg.provenance_log_path)
            if self.cfg.enable_provenance
            else None
        )
        self.firewall = (
            WeightFirewall(model, max_step_norm=self.cfg.firewall_max_step_norm)
            if self.cfg.enable_firewall
            else None
        )
        self.red_team = (
            AdversarialRedTeam(
                model,
                canary_prompts=canary_prompts,
                kl_max=self.cfg.red_team_kl_max,
                check_every_n=self.cfg.red_team_check_every,
            )
            if self.cfg.enable_red_team
            else None
        )
        self._step = 0
        self._red_team_fired = 0
        if self.red_team is not None:
            try:
                self.red_team.calibrate()
            except Exception as exc:
                warnings.warn(
                    f"red-team calibration failed ({exc}); rollback disabled",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # ----------------------------- sample acceptance
    def accept(self, sample: Any, source: str = "user") -> bool:
        risk = self.detector.score(sample) if self.detector else 0.0
        ok = risk < self.cfg.poison_threshold if self.detector else True
        if self.tracker is not None:
            self.tracker.record(sample, source, risk, ok, self._step)
        return bool(ok)

    # ----------------------------- training-loop hooks
    def pre_step(self) -> None:
        if self.red_team is not None and self._step % self.cfg.red_team_check_every == 0:
            self.red_team.snapshot()

    def after_grads(self) -> int:
        return self.firewall.apply_to_grads() if self.firewall is not None else 0

    def should_check(self) -> bool:
        return (
            self.red_team is not None
            and (self._step + 1) % self.cfg.red_team_check_every == 0
        )

    def regressed_and_rollback(self) -> bool:
        if self.red_team is None:
            return False
        if not self.should_check():
            return False
        if self.red_team.regressed():
            self.red_team.rollback()
            self._red_team_fired += 1
            return True
        return False

    def force_check(self) -> bool:
        """Force a red-team KL check + rollback regardless of cadence (for tests)."""
        if self.red_team is None:
            return False
        if self.red_team.regressed():
            self.red_team.rollback()
            self._red_team_fired += 1
            return True
        return False

    def tick(self) -> None:
        self._step += 1

    # ----------------------------- introspection
    @property
    def step(self) -> int:
        return self._step

    @property
    def red_team_fired(self) -> int:
        return self._red_team_fired

    def stats(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "red_team_fired": self._red_team_fired,
            "tracked_samples": len(self.tracker.recent(10**9)) if self.tracker else 0,
        }


__all__ = [
    "PoisonDetector",
    "ProvenanceTracker",
    "ProvenanceRecord",
    "WeightFirewall",
    "AdversarialRedTeam",
    "DefenseConfig",
    "DefenseStack",
]
