"""
Output safety training stack — Anthropic-style alignment for SynapForge.

Layered with the input poison defense (synapforge.defense):
  - defense/    : INPUT alignment — block poisoned data from corrupting weights
  - safety/     : OUTPUT alignment — prevent generating harmful content

Modules:
  red_team_corpus.py : Attack categories + seed prompts (ZH + EN)
  red_blue.py        : Self-play loop — same model plays attacker AND defender
                       roles via system-prompt swapping; emits DPO preference pairs
  constitutional.py  : CAI 2212.08073 — self-critique against principles +
                       revise loop (SL-CAI stage)
  judge.py           : RLAIF 2309.00267 — AI judge for which-is-safer ranking
  dpo.py             : DPO 2305.18290 — train on (prompt, chosen, rejected)
                       triples, beta=0.1, LoRA r=32

Pipeline (per agent synthesis 2026-04-30):
  Stage 0  SFT warmup on harm-refusal seed (10k pairs)
  Stage 1  CAI SL-CAI: model self-critiques + revises → SFT on revised
  Stage 2  Red-Blue self-play: red prompts → blue refuses → DPO on outcomes
  Stage 3  RLAIF: AI judge ranks; PPO or DPO on preferences

For 375M LNN+SNN we skip PPO (too unstable at small scale), do DPO only.
"""

from __future__ import annotations

from .red_team_corpus import (
    ATTACK_CATEGORIES,
    RedTeamPrompts,
    sample_attack_prompt,
    sample_refusal_template,
)
from .red_blue import RedBlueSelfPlay, BLUE_SYSTEM_PROMPT, RED_SYSTEM_PROMPT
from .constitutional import (
    CONSTITUTIONAL_PRINCIPLES,
    ConstitutionalRevisor,
)
from .judge import AIJudge
from .dpo import DPOTrainer, DPOPair
from .persona_swap_corpus import (
    PersonaSwapEntry,
    generate_corpus as generate_persona_swap_corpus,
    write_jsonl as write_persona_swap_jsonl,
)
from .eval_harness import (
    SafetyEvalConfig,
    evaluate_safety,
    hh_preference_accuracy,
    mmlu_regression,
    persona_robustness,
)

__all__ = [
    "ATTACK_CATEGORIES",
    "RedTeamPrompts",
    "sample_attack_prompt",
    "sample_refusal_template",
    "RedBlueSelfPlay",
    "BLUE_SYSTEM_PROMPT",
    "RED_SYSTEM_PROMPT",
    "CONSTITUTIONAL_PRINCIPLES",
    "ConstitutionalRevisor",
    "AIJudge",
    "DPOTrainer",
    "DPOPair",
    "PersonaSwapEntry",
    "generate_persona_swap_corpus",
    "write_persona_swap_jsonl",
    "SafetyEvalConfig",
    "evaluate_safety",
    "hh_preference_accuracy",
    "mmlu_regression",
    "persona_robustness",
]
