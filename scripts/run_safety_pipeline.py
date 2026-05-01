"""
Run the 4-phase Anthropic-style safety training pipeline.

Per memory feedback_anthropic_safety_stack.md:
  Phase 1: SFT 拒绝       — supervised refusal on HH-RLHF subset (2h GPU)
  Phase 2: CAI SL-CAI 4 轮 — Constitutional AI critique-revise (4h)
  Phase 3: 红蓝 DPO β=0.1 — Red-Blue self-play DPO (6h)
  Phase 4: Hidden-state 探针 — safety classifier on hidden states (30min)

Honest constraint (docs/SAFETY_PLAN.md): the safety pipeline is meant to
run AFTER Phase 3 (chat_sft) of main training. Running on an under-trained
base will give vacuous refusals because the model can't follow either the
attack or the refusal pattern.

This is the orchestrator — it dispatches to the per-phase APIs in
synapforge.safety.* and saves intermediate checkpoints + metrics.
The trainer is decoupled: this script wires data + judge, but the
gradient updates happen in DPOTrainer / your SFT trainer of choice.

Usage:
    # Run all phases
    python scripts/run_safety_pipeline.py \
        --ckpt /workspace/runs/v42/best.pt \
        --out-dir /workspace/runs/safety_v42 \
        --hh-data /workspace/data/hh_rlhf \
        --phase all

    # Run a single phase (resume / experiment)
    python scripts/run_safety_pipeline.py --ckpt ... --phase dpo
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional


# Ensure the local synapforge package is importable when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import safety submodules without triggering synapforge/__init__.py
# (which eagerly imports torch-dependent siblings).  We synthesize the
# package skeleton so relative imports inside safety/*.py resolve.
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import types  # noqa: E402


def _ensure_pkg(name: str, path: Path) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg
    return pkg


def _load_submodule(pkg_name: str, sub_name: str, file_path: Path):
    full_name = f"{pkg_name}.{sub_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[pkg_name], sub_name, mod)
    return mod


_synap_pkg = _ensure_pkg("synapforge", _REPO_ROOT / "synapforge")
_safety_pkg = _ensure_pkg("synapforge.safety",
                          _REPO_ROOT / "synapforge" / "safety")
setattr(_synap_pkg, "safety", _safety_pkg)

_safety_dir = _REPO_ROOT / "synapforge" / "safety"
_red_corpus = _load_submodule("synapforge.safety", "red_team_corpus",
                              _safety_dir / "red_team_corpus.py")
_red_blue = _load_submodule("synapforge.safety", "red_blue",
                            _safety_dir / "red_blue.py")
_constitutional = _load_submodule("synapforge.safety", "constitutional",
                                  _safety_dir / "constitutional.py")
_judge = _load_submodule("synapforge.safety", "judge",
                         _safety_dir / "judge.py")
_persona = _load_submodule("synapforge.safety", "persona_swap_corpus",
                           _safety_dir / "persona_swap_corpus.py")

ATTACK_CATEGORIES = _red_corpus.ATTACK_CATEGORIES
sample_attack_prompt = _red_corpus.sample_attack_prompt
sample_refusal_template = _red_corpus.sample_refusal_template
BLUE_SYSTEM_PROMPT = _red_blue.BLUE_SYSTEM_PROMPT
RED_SYSTEM_PROMPT = _red_blue.RED_SYSTEM_PROMPT
RedBlueSelfPlay = _red_blue.RedBlueSelfPlay
ConstitutionalRevisor = _constitutional.ConstitutionalRevisor
AIJudge = _judge.AIJudge
generate_corpus = _persona.generate_corpus
write_jsonl = _persona.write_jsonl


# dpo.py imports torch — keep a local fallback so the orchestrator runs
# without torch.  Real training wires in DPOTrainer + the real DPOPair.
try:
    _dpo = _load_submodule("synapforge.safety", "dpo",
                           _safety_dir / "dpo.py")
    DPOPair = _dpo.DPOPair
except (ImportError, ModuleNotFoundError):
    @dc.dataclass
    class DPOPair:  # type: ignore[no-redef]
        prompt: str
        chosen: str
        rejected: str
        category: str = ""
        severity: int = 0
        source: str = ""


# ---- Phase config -------------------------------------------------------

@dataclass
class PhaseConfig:
    name: str
    out_subdir: str
    target_hours: float
    target_pairs: int


PHASES: Dict[str, PhaseConfig] = {
    "sft":   PhaseConfig("sft_refusal", "phase1_sft",   target_hours=2.0,  target_pairs=10000),
    "cai":   PhaseConfig("cai_revise",  "phase2_cai",   target_hours=4.0,  target_pairs=4000),
    "dpo":   PhaseConfig("rb_dpo",      "phase3_dpo",   target_hours=6.0,  target_pairs=3000),
    "probe": PhaseConfig("safety_probe", "phase4_probe", target_hours=0.5, target_pairs=1000),
}


# ---- Stub model interface -----------------------------------------------
# The pipeline is model-agnostic.  Wire your real model by passing in a
# PolicyAdapter — for smoke we use a deterministic stub.

@dataclass
class PolicyAdapter:
    """Tiny adapter: trainers/generators you already own bind into these slots.

    For real training, replace these with calls into:
      - generate_text(prompt) → backbone.generate
      - get_hidden_state(text) → backbone.forward(...).hidden[-1].mean(dim=1)
      - sft_step(pair) → trainer.sft_step
      - dpo_step(pairs) → DPOTrainer.step
    """
    generate_text: Callable[[str], str]
    generate_temp: Callable[[str, float], str]
    get_hidden_state: Optional[Callable[[str], list]] = None
    sft_step: Optional[Callable[[dict], dict]] = None
    dpo_step: Optional[Callable[[List[DPOPair]], dict]] = None


def _stub_generate(prompt: str, _temp: float = 0.7) -> str:
    """Default-refuse stub for smoke runs (no model)."""
    rng = random.Random(hash(prompt) & 0xffff)
    is_zh = any("一" <= c <= "鿿" for c in prompt)
    if rng.random() < 0.7:
        return sample_refusal_template(rng, lang="zh" if is_zh else "en")
    return ("Here is a possible answer. Step 1: gather materials. "
            "Step 2: ... [stub-unsafe completion]")


def make_stub_adapter() -> PolicyAdapter:
    return PolicyAdapter(
        generate_text=lambda p: _stub_generate(p, 0.7),
        generate_temp=_stub_generate,
        get_hidden_state=lambda t: [float(len(t) % 7) / 7.0] * 16,
        sft_step=lambda pair: {"loss": 1.0, "stub": True},
        dpo_step=lambda pairs: {"loss": 0.6, "accuracy": 0.55, "stub": True},
    )


# ---- Phase 1: SFT refusal -----------------------------------------------

def _load_hh_or_synth(hh_data: Optional[Path], target_pairs: int,
                      rng: random.Random) -> List[dict]:
    """Try to load HH-RLHF refusal subset; otherwise synth from red corpus."""
    if hh_data and hh_data.exists():
        pairs: List[dict] = []
        for fname in ("refusal.jsonl", "harmless.jsonl", "train.jsonl"):
            p = hh_data / fname
            if not p.exists():
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pairs.append(d)
                    if len(pairs) >= target_pairs:
                        break
            if pairs:
                break
        if pairs:
            return pairs[:target_pairs]

    # Synth fallback: pair red prompts with refusal templates.
    pairs: List[dict] = []
    for _ in range(target_pairs):
        cat, seed, sev = sample_attack_prompt(rng=rng)
        is_zh = any("一" <= c <= "鿿" for c in seed)
        refusal = sample_refusal_template(rng, lang="zh" if is_zh else "en")
        pairs.append({
            "prompt": seed,
            "response": refusal,
            "category": cat,
            "severity": sev,
            "source": "synth_refusal",
        })
    return pairs


def run_sft(adapter: PolicyAdapter, out_dir: Path,
            hh_data: Optional[Path], target_pairs: int, seed: int) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _load_hh_or_synth(hh_data, target_pairs, rng)
    print(f"[phase1/sft] loaded {len(pairs)} (prompt, refusal) pairs")

    sft_jsonl = out_dir / "sft_pairs.jsonl"
    with open(sft_jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    losses: List[float] = []
    if adapter.sft_step is not None:
        for i, pair in enumerate(pairs):
            m = adapter.sft_step(pair)
            losses.append(float(m.get("loss", 0.0)))
            if (i + 1) % max(1, len(pairs) // 10) == 0:
                avg = sum(losses[-200:]) / max(len(losses[-200:]), 1)
                print(f"[phase1/sft] step {i+1}/{len(pairs)} loss={avg:.4f}")
    metrics = {
        "phase": "sft_refusal",
        "n_pairs": len(pairs),
        "mean_loss": float(sum(losses) / max(len(losses), 1)) if losses else None,
        "src": "hh-rlhf" if hh_data and hh_data.exists() else "synth",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase1/sft] done → {out_dir}")
    return metrics


# ---- Phase 2: CAI SL-CAI critique-revise --------------------------------

def run_cai(adapter: PolicyAdapter, out_dir: Path,
            target_pairs: int, n_iters: int, seed: int) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    revisor = ConstitutionalRevisor(
        generate=adapter.generate_text,
        n_iters=n_iters,
        out_jsonl=out_dir / "cai_revisions.jsonl",
        rng=rng,
    )

    samples: List[dict] = []
    for _ in range(target_pairs):
        cat, seed_prompt, sev = sample_attack_prompt(rng=rng)
        try:
            initial = adapter.generate_text(seed_prompt)
        except Exception as e:
            initial = f"[generation error: {e!r}]"
        samples.append({"red_prompt": seed_prompt, "initial_response": initial})

    print(f"[phase2/cai] running {n_iters} critique-revise iterations on "
          f"{len(samples)} attack prompts")
    records = revisor.batch_revise(samples)

    # SFT pairs from revised (red, final_revised) — feed into your SFT loop.
    sft_pairs_path = out_dir / "cai_sft_pairs.jsonl"
    with open(sft_pairs_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_sft_pair(), ensure_ascii=False) + "\n")

    metrics = {
        "phase": "cai_revise",
        "n_records": len(records),
        "n_iters": n_iters,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase2/cai] done → {sft_pairs_path}")
    return metrics


# ---- Phase 3: Red-Blue DPO self-play ------------------------------------

def run_dpo(adapter: PolicyAdapter, out_dir: Path,
            target_pairs: int, beta: float, seed: int,
            persona_swap_frac: float = 0.30) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = AIJudge(mode="auto")  # rule-only without API key

    rb = RedBlueSelfPlay(
        red_generate=adapter.generate_text,
        blue_generate=adapter.generate_temp,
        judge=lambda a, ra, rb_: judge(a, ra, rb_),
        rng=rng,
        out_jsonl=out_dir / "dpo_pairs.jsonl",
        target_persona_swap_frac=persona_swap_frac,
    )

    print(f"[phase3/dpo] generating {target_pairs} self-play DPO pairs "
          f"(persona-swap target {persona_swap_frac:.0%}, β={beta})")
    pair_dicts = rb.generate_pairs(target_pairs)
    pairs = [
        DPOPair(
            prompt=d["prompt"], chosen=d["chosen"], rejected=d["rejected"],
            category=d.get("category", ""), severity=int(d.get("severity", 0)),
            source="self_play",
        )
        for d in pair_dicts
    ]

    losses: List[float] = []
    accs: List[float] = []
    if adapter.dpo_step is not None:
        BATCH = 8
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i:i + BATCH]
            m = adapter.dpo_step(batch)
            losses.append(float(m.get("loss", 0.0)))
            accs.append(float(m.get("accuracy", 0.0)))
            if (i // BATCH + 1) % 25 == 0:
                avg_l = sum(losses[-25:]) / max(len(losses[-25:]), 1)
                avg_a = sum(accs[-25:]) / max(len(accs[-25:]), 1)
                print(f"[phase3/dpo] batch {i//BATCH+1} "
                      f"loss={avg_l:.4f} acc={avg_a:.3f}")

    stats = rb.stats()
    metrics = {
        "phase": "rb_dpo",
        "n_pairs": len(pairs),
        "beta": beta,
        "persona_swap_frac": stats.get("persona_swap_frac"),
        "category_dist": stats.get("category_dist"),
        "mean_loss": float(sum(losses) / max(len(losses), 1)) if losses else None,
        "mean_acc": float(sum(accs) / max(len(accs), 1)) if accs else None,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase3/dpo] done → {out_dir}")
    return metrics


# ---- Phase 4: Hidden-state safety probe ---------------------------------

def _logistic_fit(X: List[List[float]], y: List[int],
                  n_iters: int = 500, lr: float = 0.5) -> tuple[List[float], float]:
    """Pure-python logistic regression so this script has zero hard deps."""
    if not X:
        return [], 0.0
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    n = len(X)
    for _ in range(n_iters):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(w[j] * xi[j] for j in range(d))
            # numerically stable sigmoid
            if z >= 0:
                ez = pow(2.718281828, -z)
                p = 1.0 / (1.0 + ez)
            else:
                ez = pow(2.718281828, z)
                p = ez / (1.0 + ez)
            err = p - yi
            gb += err
            for j in range(d):
                gw[j] += err * xi[j]
        b -= lr * gb / n
        for j in range(d):
            w[j] -= lr * gw[j] / n
    return w, b


def _logistic_predict(x: List[float], w: List[float], b: float) -> float:
    z = b + sum(w[j] * x[j] for j in range(len(w)))
    if z >= 0:
        ez = pow(2.718281828, -z)
        return 1.0 / (1.0 + ez)
    ez = pow(2.718281828, z)
    return ez / (1.0 + ez)


def run_probe(adapter: PolicyAdapter, out_dir: Path,
              target_pairs: int, seed: int) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    if adapter.get_hidden_state is None:
        print("[phase4/probe] no get_hidden_state — skipping probe phase")
        return {"phase": "safety_probe", "skipped": True}

    # Build labeled set: harmful prompts → 1, benign prompts → 0
    benign_prompts = [
        "What is the boiling point of water?",
        "Recommend a Python web framework for beginners.",
        "解释一下相对论的基本原理。",
        "Help me write a thank-you note to my teacher.",
        "推荐三本经济学入门书。",
        "How does photosynthesis work?",
        "给我一个意大利面的食谱。",
        "Explain the difference between TCP and UDP.",
    ]

    half = target_pairs // 2
    X: List[List[float]] = []
    y: List[int] = []
    for _ in range(half):
        cat, seed_prompt, _sev = sample_attack_prompt(rng=rng)
        try:
            h = adapter.get_hidden_state(seed_prompt)
        except Exception:
            continue
        X.append(list(h))
        y.append(1)  # unsafe
    for _ in range(half):
        prompt = rng.choice(benign_prompts)
        try:
            h = adapter.get_hidden_state(prompt)
        except Exception:
            continue
        X.append(list(h))
        y.append(0)

    if not X:
        return {"phase": "safety_probe", "skipped": True,
                "reason": "no hidden states"}

    # 80/20 train/eval split
    pairs = list(zip(X, y))
    rng.shuffle(pairs)
    cut = int(0.8 * len(pairs))
    train, eval_ = pairs[:cut], pairs[cut:]
    Xt, yt = [p[0] for p in train], [p[1] for p in train]
    Xe, ye = [p[0] for p in eval_], [p[1] for p in eval_]

    w, b = _logistic_fit(Xt, yt)
    tp = fp = tn = fn = 0
    for x, y_true in zip(Xe, ye):
        pred = 1 if _logistic_predict(x, w, b) > 0.5 else 0
        if pred == 1 and y_true == 1:
            tp += 1
        elif pred == 1 and y_true == 0:
            fp += 1
        elif pred == 0 and y_true == 0:
            tn += 1
        else:
            fn += 1
    n_eval = max(tp + fp + tn + fn, 1)
    detection = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    weights_path = out_dir / "probe_weights.json"
    weights_path.write_text(
        json.dumps({"w": w, "b": b, "dim": len(w)}, indent=2), encoding="utf-8")

    metrics = {
        "phase": "safety_probe",
        "n_train": len(Xt),
        "n_eval": n_eval,
        "detection_rate": detection,
        "false_positive_rate": fpr,
        "accuracy": (tp + tn) / n_eval,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase4/probe] detection={detection:.2%} fpr={fpr:.2%} → {out_dir}")
    return metrics


# ---- Top-level orchestrator ---------------------------------------------

def ensure_persona_corpus(safety_dir: Path) -> Path:
    out = safety_dir / "persona_swap_red.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"[bootstrap] generating persona-swap corpus → {out}")
    # 105 unique personas × 50 templates = 5250 prompts (cycles past 105 add
    # variant-suffixed entries so prompts stay byte-distinct).
    entries = generate_corpus(n_personas=105, n_templates=50, seed=42)
    write_jsonl(out, entries)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=False,
                    help="Path to base model checkpoint (informational; "
                         "this orchestrator does not load weights).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/workspace/runs/safety_demo"))
    ap.add_argument("--hh-data", type=Path, default=None,
                    help="Path to HH-RLHF jsonl directory.")
    ap.add_argument("--phase", choices=["sft", "cai", "dpo", "probe", "all"],
                    default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cai-iters", type=int, default=4,
                    help="CAI critique-revise iterations (Anthropic 2212.08073).")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--persona-swap-frac", type=float, default=0.30,
                    help="Fraction of DPO pairs from persona-swap class.")
    ap.add_argument("--smoke-pairs", type=int, default=20,
                    help="If set, reduce all per-phase pair counts to N for "
                         "fast smoke runs.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    safety_dir = Path(__file__).resolve().parents[1] / "synapforge" / "safety"
    ensure_persona_corpus(safety_dir)

    adapter = make_stub_adapter()

    summary: Dict[str, dict] = {}
    phases_to_run = (["sft", "cai", "dpo", "probe"]
                     if args.phase == "all" else [args.phase])

    for ph in phases_to_run:
        cfg = PHASES[ph]
        sub = args.out_dir / cfg.out_subdir
        n_pairs = min(cfg.target_pairs, args.smoke_pairs) \
            if args.smoke_pairs else cfg.target_pairs

        t0 = time.time()
        if ph == "sft":
            m = run_sft(adapter, sub, args.hh_data, n_pairs, args.seed)
        elif ph == "cai":
            m = run_cai(adapter, sub, n_pairs, args.cai_iters, args.seed)
        elif ph == "dpo":
            m = run_dpo(adapter, sub, n_pairs, args.dpo_beta, args.seed,
                        persona_swap_frac=args.persona_swap_frac)
        elif ph == "probe":
            m = run_probe(adapter, sub, n_pairs, args.seed)
        else:
            raise ValueError(f"unknown phase: {ph}")
        m["wall_seconds"] = round(time.time() - t0, 2)
        summary[ph] = m

    summary_path = args.out_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== safety pipeline summary → {summary_path} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
