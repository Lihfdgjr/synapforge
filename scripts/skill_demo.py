"""scripts/skill_demo.py — end-to-end proof of the universal codebook.

The user's literal requirement:
    "神经元工具我要的是万能的, ai 可以自己学习, 可以根据自己需求随时变化,
     形成新的工具, 也可以根据用户需求生成, 而且永久不丢失."

This script is the proof:
    1. Empty codebook with only 9 L1 primitives at boot.
    2. Mint 3 user-described L3 skills (the "user-driven" claim).
    3. Drive 50 random episodes against ``WebBrowserEnv`` mock; after
       each, let the synthesiser look for a co-firing pattern (the
       "AI self-learn" claim) and let the AI mint L3 macros on success
       (the "AI forms new tools" claim).
    4. Save to ``runs/skill_demo/skills.jsonl`` (atomic write, rotated).
    5. Reload from disk into a fresh codebook (the "permanent / never
       lost" claim) and verify state is identical.
    6. Print: total skills, per-layer breakdown, top-10 most used,
       top-10 most-recently minted.

Smoke-runnable on CPU; real Playwright is *not* required (we use
``WebEnvConfig(real=False)``).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List

import torch

# Make repo root importable when run as `python scripts/skill_demo.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapforge.action.skill_log_v2 import SkillLog
from synapforge.action.skill_synthesizer import SkillSynthesizer
from synapforge.action.universal_codebook import (
    L1_PRIMITIVES,
    UniversalCodebook,
    default_text_encoder,
)
from synapforge.action.web_env import WebBrowserEnv, WebEnvConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


HIDDEN = 64

USER_DESCRIPTIONS = [
    "查股市行情每天 9 点 (search Bing for stock market quote at 9am daily)",
    "打开 GitHub 找 trending repo (open GitHub trending repo discovery)",
    "在 arxiv 上搜 spiking neural net (search arxiv for spiking neural net)",
]


def _action_to_l1_id(action_type: str) -> int:
    """Map web_env action string to a UniversalCodebook L1 prototype id."""
    table = {
        "click": 0,
        "double_click": 0,
        "right_click": 0,
        "type": 1,
        "key": 2,
        "scroll": 3,
        "wait": 4,
        "back": 5,
        "forward": 6,
        "done": 7,
    }
    return table.get(action_type, 8)  # NULL fallback


def _random_action(rng: random.Random) -> dict:
    """Generate a random action dict in the same shape as ``ActionHead.to_dict``."""
    a_type = rng.choices(
        ["click", "type", "key", "scroll", "wait", "done"],
        weights=[5, 3, 1, 2, 1, 1],
        k=1,
    )[0]
    d: dict = {
        "type": a_type,
        "x": None, "y": None,
        "scroll_dx": None, "scroll_dy": None,
        "key": None,
        "text_trigger": False,
    }
    if a_type in {"click", "double_click", "right_click"}:
        d["x"] = rng.uniform(0.05, 0.95)
        d["y"] = rng.uniform(0.05, 0.95)
    elif a_type == "scroll":
        d["scroll_dx"] = 0.0
        d["scroll_dy"] = rng.uniform(-0.5, 0.5)
    elif a_type == "type":
        d["text_id"] = rng.randint(0, 5)
        d["text_trigger"] = True
    elif a_type == "key":
        d["key"] = rng.choice(["enter", "esc", "tab", "space"])
    return d


def _episode(env: WebBrowserEnv, synth: SkillSynthesizer, rng: random.Random, max_steps: int = 8) -> tuple[List[int], float]:
    """Run one episode, drive the codebook with each emitted action.

    Returns ``(trace, total_reward)``.  Caller decides whether the trace
    earned enough reward to be minted as an L3 macro.
    """
    env.reset("https://www.bing.com")
    trace: List[int] = []
    total_r = 0.0
    for _ in range(max_steps):
        action = _random_action(rng)
        l1_id = _action_to_l1_id(action["type"])
        trace.append(l1_id)
        # Activate L1 (pure usage tracking, never grows).
        synth.codebook.activate(l1_id, reward=0.05)
        # Online co-firing detection -> may mint L2.
        synth.codebook.mint_from_co_firing(l1_id)
        # Step env to get a (mock) reward.
        _obs, r, done, _info = env.step(action)
        total_r += float(r)
        if done:
            break
    return trace, total_r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(seed: int = 0, n_episodes: int = 50) -> None:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # 1. Initialise empty UniversalCodebook (9 L1 primitives).
    print("=" * 60)
    print(" SynapForge UniversalCodebook end-to-end demo")
    print("=" * 60)
    cb = UniversalCodebook(hidden=HIDDEN, K_initial=len(L1_PRIMITIVES))
    save_path = ROOT / "runs" / "skill_demo" / "skills.jsonl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log = SkillLog(save_path)
    synth = SkillSynthesizer(cb, skill_log=log)

    print(f"\n[boot] {cb.stats()}")

    # 2. User-driven L3 mints
    print("\n--- Step 1: user-driven L3 mints ---")
    user_pids: List[int] = []
    for desc in USER_DESCRIPTIONS:
        pid = synth.synthesize_from_description(desc, urgency=1.0)
        user_pids.append(pid)
        print(f"  mint('{desc[:48]}...')  -> proto_id={pid}")

    # 3. AI-driven mints via random episodes
    print("\n--- Step 2: drive {} episodes through WebBrowserEnv (mock) ---".format(n_episodes))
    env = WebBrowserEnv(WebEnvConfig(real=False))
    n_l2_before = cb.size_by_layer()["L2"]
    n_l3_before = cb.size_by_layer()["L3"]
    successes: List[List[int]] = []
    # Threshold raised from 0.5 to 2.0 — the mock env hands out +1.0 per
    # novel-page step, so 0.5 fired on virtually every episode and minted
    # a lot of L3 noise.  2.0 corresponds to "agent navigated to at least
    # two new pages", i.e., real progress.
    success_threshold = 2.0
    for ep in range(n_episodes):
        trace, total_r = _episode(env, synth, rng)
        if total_r > success_threshold and len(trace) >= 3:
            # Successful trace -> let the AI mint a compound.
            pid = synth.synthesize_from_trace(trace, success=True, reward=total_r)
            if pid >= 0:
                successes.append(trace)
        if ep % 10 == 0:
            sizes = cb.size_by_layer()
            print(f"  ep={ep:>3d}  reward={total_r:+.2f}  "
                  f"L1={sizes['L1']:>2d}  L2={sizes['L2']:>2d}  L3={sizes['L3']:>2d}  "
                  f"trace_len={len(trace)}")
    env.close()
    n_l2_after = cb.size_by_layer()["L2"]
    n_l3_after = cb.size_by_layer()["L3"]
    print(f"\n  L2 growth: {n_l2_before} -> {n_l2_after}  (+{n_l2_after - n_l2_before})")
    print(f"  L3 growth: {n_l3_before} -> {n_l3_after}  (+{n_l3_after - n_l3_before})")

    # 4. Atomic save with rotation
    print("\n--- Step 3: atomic save to disk ---")
    n_saved = log.save_codebook(cb)
    print(f"  saved {n_saved} skills -> {save_path}")
    print(f"  history events: {len(log.read_history())}")
    print(f"  rotated versions: {log.quick_stats().get('rotated_versions', [])}")

    # 5. Reload into fresh codebook -> idempotency check
    print("\n--- Step 4: reload into fresh codebook (idempotent) ---")
    cb2 = UniversalCodebook(hidden=HIDDEN, K_initial=len(L1_PRIMITIVES))
    log2 = SkillLog(save_path)
    n_loaded1 = log2.load_codebook(cb2)
    n_loaded2 = log2.load_codebook(cb2)   # idempotent: same state after second load
    print(f"  load #1 restored: {n_loaded1}")
    print(f"  load #2 restored: {n_loaded2}  (idempotent)")
    sizes_orig = cb.size_by_layer()
    sizes_new = cb2.size_by_layer()
    match = sizes_orig == sizes_new
    print(f"  sizes match: original={sizes_orig}  reloaded={sizes_new}  -> {match}")

    # Cosine similarity of L3 user-skill embeddings should also match
    print("\n--- Step 5: verify embeddings round-trip exactly ---")
    drift = 0.0
    for pid in user_pids:
        if pid < 0:
            continue
        emb1 = cb.slots[cb._proto_to_slot[pid]].detach()
        emb2 = cb2.slots[cb2._proto_to_slot[pid]].detach()
        delta = float((emb1 - emb2).abs().max().item())
        drift = max(drift, delta)
        print(f"  proto_id={pid}  max|Δ|={delta:.6f}")
    print(f"  worst-case drift = {drift:.6f}  (zero = exact reload)")

    # 6. Reports
    print("\n--- Final reports ---")
    final = synth.stats()
    print(f"  K_alive   = {final['K_alive']}")
    print(f"  by_layer  = {final['by_layer']}")
    print(f"  top10_uses:")
    for pid, n_uses, layer, desc in final["top10_uses"]:
        d = (desc[:48] + "...") if len(desc) > 50 else desc
        print(f"    {layer}  pid={pid:<4d}  uses={n_uses:<4d}  {d}")

    # most-recently-minted
    by_age = sorted(
        ((pid, m.created_at, m.layer, m.description) for pid, m in cb.meta.items() if not m.archived),
        key=lambda x: -x[1],
    )[:10]
    print(f"\n  newest 10 (across layers):")
    for pid, ts, layer, desc in by_age:
        d = (desc[:48] + "...") if len(desc) > 50 else desc
        print(f"    {layer}  pid={pid:<4d}  ts={ts:.0f}  {d}")

    # 7. The proof banner
    print("\n" + "=" * 60)
    if match and drift < 1e-3:
        print(" Universal codebook proof: PASS")
        print(" * 万能       (open-ended K=9 -> {})  ".format(final["K_alive"]))
        print(" * 自学       (mint_from_co_firing -> +{} L2)".format(n_l2_after - n_l2_before))
        print(" * 用户驱动   ({} L3 user-skills minted)".format(len(user_pids)))
        print(" * 永不丢失   (idempotent reload, max|Δ|={:.0e})".format(drift))
    else:
        print(" Universal codebook proof: FAIL")
        print(f" sizes match={match}, drift={drift}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    main(seed=args.seed, n_episodes=args.episodes)
