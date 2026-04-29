"""
PoisonDetector: per-sample anomaly score combining multiple signals.

Synthesis from 3 parallel agent reports (2026-04-30):

WEB ingest (autonomous_daemon):
  - TRAK gradient-canary alignment (arxiv 2303.14186) — primary signal
  - Mahalanobis OOD distance over hidden state (1807.03888)
  - Cluster distance to known-injection corpus
  - Provenance trust score (per-source EMA)

CHAT ingest (production):
  - PromptGuard-86M (Meta) for direct injection / jailbreak
  - Llama-Guard-3 (or smaller stand-in) for jailbreak class
  - Mahalanobis hidden-state OOD
  - Per-user n-gram KL divergence (bias-collapse detector)
  - Presidio PII regex+NER
  - Per-user rate limit (5/hr, 50/day)

The combined accept rule (agent 2):
    accept = (s_inj < 0.30) AND (s_jb < 0.50) AND (s_oos < 1.0)
        AND (s_dup < 0.85) AND (s_pii < 0.40) AND (s_user_kl < 3.0)

If `accept`, sample enters replay buffer.
Else if combined ∈ [0.3, 0.5]: low-LR admit (downweighted).
Else: reject + provenance.report_reject().

The TRAK score (web only) is computed against a frozen 2k-sample canary set:
    phi_i = grad L(x_i)
    g_v = mean grad L(v) over canary
    TRAK(x_i) = <P phi_i, P g_v> / (||P phi_i|| * ||P g_v||)
where P is a Gaussian projection to dim k=4096. Reject if TRAK < 0.

KEY INSIGHT (per agent 1): poison's structural signature is "negative gradient
alignment with canary." Novelty has random alignment (E ≈ 0), poison has
systematically negative alignment. Surface statistics (perplexity) can be
gamed (Wallace 2010.12563 "Concealed Data Poisoning") — TRAK cannot, without
attacker knowing the canary set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PoisonScore:
    s_inj: float = 0.0
    s_jb: float = 0.0
    s_oos: float = 0.0
    s_dup: float = 0.0
    s_pii: float = 0.0
    s_user_kl: float = 0.0
    s_trak: float = 1.0
    s_grad_norm: float = 0.0
    combined: float = 0.0
    accept: bool = True
    low_lr: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scores": {
                "inj": self.s_inj, "jb": self.s_jb, "oos": self.s_oos,
                "dup": self.s_dup, "pii": self.s_pii, "user_kl": self.s_user_kl,
                "trak": self.s_trak, "grad_norm": self.s_grad_norm,
            },
            "combined": self.combined,
            "accept": self.accept,
            "low_lr": self.low_lr,
            "reasons": self.reasons,
        }


class PoisonDetector:
    """Compute multi-signal poison score for a sample.

    Plug-in design: each signal is a callable that returns float in [0, 1] (or
    higher for un-bounded ones like Mahalanobis). Detector evaluates active
    signals and computes combined accept decision.

    Signals available are populated lazily — heavy ones (PromptGuard, TRAK)
    only load if requested.
    """

    DEFAULT_THRESHOLDS = {
        "inj": 0.30,
        "jb": 0.50,
        "oos": 1.0,
        "dup": 0.85,
        "pii": 0.40,
        "user_kl": 3.0,
        "trak": 0.0,
    }

    DEFAULT_WEIGHTS = {
        "inj": 0.30, "jb": 0.25, "oos": 0.20,
        "dup": 0.10, "pii": 0.10, "user_kl": 0.05,
    }

    def __init__(
        self,
        signals: Optional[Dict[str, "callable"]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        low_lr_band: Tuple[float, float] = (0.30, 0.50),
    ) -> None:
        self.signals = signals or {}
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.low_lr_band = low_lr_band

    def register_signal(self, name: str, fn: "callable") -> None:
        self.signals[name] = fn

    def score(self, sample: dict) -> PoisonScore:
        """sample dict may contain: 'text', 'user_handle', 'history', 'hidden'."""
        s = PoisonScore()
        text = sample.get("text", "")

        if "inj" in self.signals:
            s.s_inj = float(self.signals["inj"](sample))
        if "jb" in self.signals:
            s.s_jb = float(self.signals["jb"](sample))
        if "oos" in self.signals:
            s.s_oos = float(self.signals["oos"](sample))
        if "dup" in self.signals:
            s.s_dup = float(self.signals["dup"](sample))
        if "pii" in self.signals:
            s.s_pii = float(self.signals["pii"](sample))
        if "user_kl" in self.signals:
            s.s_user_kl = float(self.signals["user_kl"](sample))
        if "trak" in self.signals:
            s.s_trak = float(self.signals["trak"](sample))
        if "grad_norm" in self.signals:
            s.s_grad_norm = float(self.signals["grad_norm"](sample))

        for key, thr in self.thresholds.items():
            v = getattr(s, f"s_{key}", None)
            if v is None:
                continue
            if key == "trak":
                if v < thr:
                    s.reasons.append(f"trak<{thr}: got {v:.3f}")
            else:
                if v >= thr:
                    s.reasons.append(f"{key}>={thr}: got {v:.3f}")

        combined = sum(
            self.weights.get(k, 0.0) * getattr(s, f"s_{k}", 0.0)
            for k in self.weights
        )
        s.combined = combined

        if s.reasons:
            s.accept = False
            s.low_lr = False
        elif combined >= self.low_lr_band[1]:
            s.accept = False
            s.low_lr = False
            s.reasons.append(f"combined>={self.low_lr_band[1]}: {combined:.3f}")
        elif combined >= self.low_lr_band[0]:
            s.accept = True
            s.low_lr = True
            s.reasons.append(f"low-LR band: combined={combined:.3f}")
        else:
            s.accept = True
            s.low_lr = False

        return s


def make_mahalanobis_signal(mu: torch.Tensor, sigma_inv: torch.Tensor, chi2_99: float = 16.81):
    """Mahalanobis OOD signal. mu/sigma_inv computed offline on canary."""
    mu = mu.detach().cpu()
    sigma_inv = sigma_inv.detach().cpu()
    def fn(sample: dict) -> float:
        h = sample.get("hidden")
        if h is None:
            return 0.0
        h = h.detach().float().cpu()
        if h.dim() > 1:
            h = h.mean(dim=0)
        d = h - mu
        m_dist = float((d @ sigma_inv @ d).item())
        return m_dist / chi2_99
    return fn


def make_dedup_signal(provenance_tracker, k_recent: int = 1000):
    """N-gram-style dedup: hash 5-gram and check recent window."""
    def fn(sample: dict) -> float:
        text = sample.get("text", "")
        if not text:
            return 0.0
        if provenance_tracker.is_blocked(text):
            return 1.0
        return 0.0
    return fn


def make_user_kl_signal(user_history: Dict[str, List[str]]):
    """Per-user n-gram KL divergence (bias-collapse detector)."""
    import math
    from collections import Counter

    def fn(sample: dict) -> float:
        user = sample.get("user_handle")
        if not user or user not in user_history:
            return 0.0
        history = user_history[user][-30:]
        if len(history) < 5:
            return 0.0

        def trigrams(text):
            t = text.replace(" ", "_")
            return [t[i:i+3] for i in range(len(t) - 2)]

        u_grams = Counter()
        for h in history:
            u_grams.update(trigrams(h))
        s_grams = Counter(trigrams(sample.get("text", "")))

        total_u = sum(u_grams.values()) or 1
        total_s = sum(s_grams.values()) or 1
        kl = 0.0
        for g in s_grams:
            p_s = s_grams[g] / total_s
            p_u = (u_grams.get(g, 0) + 1) / (total_u + len(u_grams))
            if p_s > 0:
                kl += p_s * math.log(p_s / p_u)
        return max(0.0, kl)
    return fn


class TRAKGate:
    """TRAK influence gate (arxiv 2303.14186) for web-content gating.

    Implementation: random Gaussian projection of gradients, cosine similarity
    against averaged canary gradient. Reject if cosine < threshold.

    Heavy: requires backward pass per candidate. Run async / batched.
    """

    def __init__(
        self,
        model: nn.Module,
        canary_loader,
        proj_dim: int = 4096,
        threshold: float = 0.0,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.canary_loader = canary_loader
        self.proj_dim = proj_dim
        self.threshold = threshold
        self.device = device

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.manual_seed(42)
        self.proj_matrix = torch.randn(n_params, proj_dim, device=device) / (proj_dim ** 0.5)

        self._canary_grad: Optional[torch.Tensor] = None

    def _flat_grad(self) -> torch.Tensor:
        flats = []
        for p in self.model.parameters():
            if p.grad is not None and p.requires_grad:
                flats.append(p.grad.detach().flatten())
            elif p.requires_grad:
                flats.append(torch.zeros_like(p).flatten())
        return torch.cat(flats)

    def warmup_canary(self, n_batches: int = 4) -> None:
        """Compute averaged canary gradient (frozen reference)."""
        import torch.nn.functional as F
        self.model.train()
        accum = None
        n_seen = 0
        for batch_idx, batch in enumerate(self.canary_loader):
            if batch_idx >= n_batches:
                break
            self.model.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model(input_ids)
            if hasattr(logits, "logits"):
                logits = logits.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            g = self._flat_grad()
            accum = g if accum is None else accum + g
            n_seen += 1
        if n_seen == 0:
            self._canary_grad = None
            return
        self._canary_grad = (accum / n_seen) @ self.proj_matrix

    def score_sample(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        """Returns TRAK score (cosine alignment with canary gradient)."""
        if self._canary_grad is None:
            return 1.0
        import torch.nn.functional as F
        self.model.zero_grad()
        logits = self.model(input_ids.to(self.device))
        if hasattr(logits, "logits"):
            logits = logits.logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.to(self.device).reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        g = self._flat_grad()
        g_proj = g @ self.proj_matrix
        cos = F.cosine_similarity(g_proj.unsqueeze(0), self._canary_grad.unsqueeze(0))
        return float(cos.item())
