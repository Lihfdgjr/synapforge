"""Phase manager — auto-enable multimodal / self-learn / curiosity at ppl thresholds.

Per user 2026-05-01: "等达到一定程度开多模态和自学习,还有好奇心"

Watches train.log for step-by-step ppl. When validation ppl crosses
a threshold, fires the next phase by writing a flag file + signalling
the trainer (or simply preparing the next-phase launch script).

Phases (from cold-warm v16 warmstart at ce=9.6, ppl≈15000):

  Phase 0 (now)       ce > 5.5   ppl > 250    LM-only KD distill from GPT-2
  Phase 1 (intrinsic) ppl <= 250              + intrinsic curiosity reward
                                              + self-learn (TTT + replay)
                                              + STDP-driven novelty signal
  Phase 2 (modal)     ppl <= 100              + modal byte-patch encoders
                                              (image / audio / time-series)
                                              + cross-modal contrastive aux
  Phase 3 (chat)      ppl <= 60               + alpaca-zh SFT (response-only)
  Phase 4 (RL)        chat eval > 60% pass    + GRPO verifier-RL

Each phase enables flags via /workspace/runs/<run>/.phase. Trainer
should poll this file every K steps (when integrated) and adjust
training accordingly. For now, phase manager just signals + logs;
human reads and restarts trainer with new flags.

Usage:
    python scripts/phase_manager.py \
        --watch /workspace/runs/v24h_lnn \
        --interval 60
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path


PHASES = [
    {"id": 0, "name": "lm_kd",      "ppl_max": float("inf"), "flags": []},
    # NB: only flags that train_100m_kd.py argparse accepts. Audit 2026-05-01:
    # `--intrinsic-curiosity` and `--stdp-novelty` were imagined; trainer has
    # `--self-learn-ttt` and `--curiosity-weight` only. STDP novelty signal
    # is bookkeeping-only (no autograd path) so it stays off until wired.
    {"id": 1, "name": "intrinsic",  "ppl_max": 250.0,
     "flags": ["--self-learn-ttt", "--self-learn-k 8", "--curiosity-weight 0.05"]},
    {"id": 2, "name": "multimodal", "ppl_max": 100.0,
     "flags": ["--modal-list image,audio,time_series"]},
    {"id": 3, "name": "chat_sft",   "ppl_max": 60.0,
     "flags": ["--sft-data /workspace/data/alpaca_zh/alpaca_zh.json",
               "--response-only-loss", "--lr 1e-4"]},
    {"id": 4, "name": "rl_grpo",    "ppl_max": 0.0,  # gated by chat_eval not ppl
     "flags": ["--rl-grpo", "--rl-verifier sympy", "--rl-rollouts 8"]},
]

PPL_RE = re.compile(r"step\s*(\d+).*?ce=([\d.]+)")
# P3 (MASTER_PLAN.md §6): prefer the leak-free holdout ppl when the
# trainer logs both. The trainer line looks like:
#   VAL step 500: val_ppl_ttt=180.21 val_ppl_holdout=185.03 (honest)
# Old trainers logged just  ``ppl=185.03``; the legacy regex still
# picks that up as a fallback.
VAL_PPL_HOLDOUT_RE = re.compile(
    r"val[_\s]?ppl[_\s]?holdout[=:]?\s*([\d.]+)", re.IGNORECASE,
)
VAL_PPL_RE = re.compile(r"val.*?ppl[=:]?\s*([\d.]+)", re.IGNORECASE)


def parse_log(log_path: Path) -> dict | None:
    """Read latest train step + best val ppl from train.log."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    last_step, last_ce = None, None
    for m in PPL_RE.finditer(text):
        last_step = int(m.group(1))
        last_ce = float(m.group(2))
    # P3: prefer val_ppl_holdout (TTT-leak-free) when the trainer
    # logs it. Falls back to the generic ``val.*?ppl`` regex for
    # legacy logs that pre-date the dual-track split.
    holdout_ppls = [
        float(m.group(1)) for m in VAL_PPL_HOLDOUT_RE.finditer(text)
    ]
    if holdout_ppls:
        val_ppls = holdout_ppls
    else:
        val_ppls = [float(m.group(1)) for m in VAL_PPL_RE.finditer(text)]
    # Guard against last_ce being valid but 0.0 (which is falsy in
    # Python). 0.0 ce => exp(0) = 1.0, which is a meaningful ppl, not None.
    last_train_ppl = math.exp(last_ce) if last_ce is not None else None
    return {
        "last_step": last_step,
        "last_ce": last_ce,
        "last_train_ppl": last_train_ppl,
        "best_val_ppl": min(val_ppls) if val_ppls else None,
        "n_val_evals": len(val_ppls),
    }


def decide_phase(state: dict | None) -> int:
    """Return phase id appropriate for current state."""
    if state is None or state.get("last_train_ppl") is None:
        return 0
    # Prefer the better of (best_val_ppl, last_train_ppl). Use explicit
    # None checks -- `or` on a 0.0 ppl would skip the (legitimate) zero.
    bv = state.get("best_val_ppl")
    ppl = bv if bv is not None else state["last_train_ppl"]
    p = 0
    for ph in PHASES:
        if ppl <= ph["ppl_max"]:
            p = ph["id"]
    return p


def write_phase_signal(out_dir: Path, phase_id: int, state: dict) -> None:
    """Write `.phase` JSON so human / trainer can pick it up."""
    phase = next(p for p in PHASES if p["id"] == phase_id)
    payload = {
        "phase_id": phase_id,
        "phase_name": phase["name"],
        "ts": time.time(),
        "state": state,
        "next_phase_flags": phase["flags"],
        "instructions": (
            "Restart trainer with these flags appended to launch_train.sh "
            "to enable the next phase. Existing run continues; ckpt is reused."
        ),
    }
    (out_dir / ".phase").write_text(json.dumps(payload, indent=2))


def watch_loop(watch_dir: Path, interval: int) -> None:
    print(f"[phase] watching {watch_dir}/train.log every {interval}s")
    last_phase = -1
    last_ts = 0.0
    while True:
        log = watch_dir / "train.log"
        state = parse_log(log)
        phase = decide_phase(state)
        now = time.time()
        if phase != last_phase or now - last_ts > 600:
            ppl = state.get("last_train_ppl") if state else None
            ppl_str = f"{ppl:.1f}" if ppl is not None else "?"
            print(f"[phase {time.strftime('%H:%M:%S')}] step={state and state.get('last_step')} "
                  f"ppl={ppl_str} -> phase {phase} ({PHASES[phase]['name']})")
            if phase != last_phase and phase > 0:
                ph = PHASES[phase]
                print(f"[phase] >>> THRESHOLD CROSSED <<< enable: {ph['flags']}")
                write_phase_signal(watch_dir, phase, state or {})
            last_phase = phase
            last_ts = now
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True)
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    watch_loop(Path(args.watch), args.interval)


if __name__ == "__main__":
    main()
