"""
Defense module for continual-learning poison resistance.

Two ingress points to the replay buffer share these primitives:
  - WebPoisonGate    : autonomous_daemon ingests web search results
  - ChatPoisonGate   : production chat ingests user messages

The 4-layer pipeline (per feedback_self_learn_poison_defense.md):
  1. PoisonDetector      : per-sample anomaly score (perplexity / gradient / cluster)
  2. ProvenanceTracker   : URL / user_id / timestamp / hash, persisted
  3. WeightFirewall      : KL-divergence-bounded update, freezes when drift > threshold
  4. AdversarialRedTeam  : periodic injection of known-bad samples to verify defense

This module is intentionally NOT runtime-coupled to the trainer. The contract:
  trainer  -> calls  poison_gate.admit(sample, source) -> True/False/Quarantine
  trainer  -> calls  weight_firewall.allow_step(grad)  -> True/False/Clip
  daemon   -> writes JSONL with provenance fields, gate filters before append
"""

from __future__ import annotations

from .poison_detector import PoisonDetector, PoisonScore
from .provenance import ProvenanceEntry, ProvenanceTracker
from .gates import WebPoisonGate, ChatPoisonGate, GateDecision
from .weight_firewall import WeightFirewall

from .legacy import (
    AdversarialRedTeam,
    DefenseConfig,
    DefenseStack,
    ProvenanceRecord,
)

__all__ = [
    "PoisonDetector",
    "PoisonScore",
    "ProvenanceEntry",
    "ProvenanceTracker",
    "WebPoisonGate",
    "ChatPoisonGate",
    "GateDecision",
    "WeightFirewall",
    "AdversarialRedTeam",
    "DefenseConfig",
    "DefenseStack",
    "ProvenanceRecord",
]
