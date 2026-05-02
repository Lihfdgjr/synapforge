"""Tests: 10-step rollout in sandbox completes without exception."""
from __future__ import annotations

import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(name: str, relpath: str):
    if name in sys.modules:
        return sys.modules[name]
    full = os.path.join(REPO_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Order matters: closed_loop imports primitives + os_actuator + sandbox + compound_growth.
prim = _load("synapforge.neuromcp.primitives", "synapforge/neuromcp/primitives.py")
sandbox = _load("synapforge.neuromcp.sandbox", "synapforge/neuromcp/sandbox.py")
oa = _load("synapforge.neuromcp.os_actuator", "synapforge/neuromcp/os_actuator.py")
cg = _load("synapforge.neuromcp.compound_growth", "synapforge/neuromcp/compound_growth.py")
cl = _load("synapforge.neuromcp.closed_loop", "synapforge/neuromcp/closed_loop.py")


def _scripted_policy():
    """Repeating script: click->click->type->press_key with 5x repeats so a
    triplet compound emerges by the end of 10 steps."""
    seq = [
        (0, [0.05, 0.27] + [0.0] * 6, 0.9),
        (8, [0, 0, 0, 0, 1, 0, 0, 0], 0.9),
        (9, [0, 0, 0, 0, 0, 26, 0, 0], 0.9),  # press 'enter' (idx 26)
    ] * 4
    idx = [0]

    def policy(_obs):
        i = idx[0] % len(seq)
        idx[0] += 1
        return seq[i]

    return policy


def test_10_step_rollout_no_exception():
    env = cl.ClosedLoopEnv()
    policy = _scripted_policy()
    results = env.rollout(policy, n_steps=10)
    assert len(results) == 10
    successes = [r for r in results if r.success]
    assert len(successes) >= 1, "at least one step must succeed"


def test_compound_emerges_during_rollout():
    grower = cg.CompoundGrowth(
        num_primitives=prim.NUM_PRIMITIVES,
        reuse_threshold=3,
        n_gram_min=2, n_gram_max=3,
        proposal_window=50,
    )
    env = cl.ClosedLoopEnv(growth=grower)
    policy = _scripted_policy()
    env.rollout(policy, n_steps=12)
    sigs = grower.list_signatures()
    assert sigs, "expected at least one compound to emerge in 12-step rollout"


def test_low_confidence_halts():
    env = cl.ClosedLoopEnv(halt_threshold=0.5)
    res = env.step(0, [0.5] * 8, confidence=0.2)
    assert res.halted is True
    assert res.success is False
    assert res.obs is not None and "halted" in res.obs.text.lower()


def test_reward_signs_match_success():
    env = cl.ClosedLoopEnv(success_reward=1.0, failure_reward=-0.5)
    # Click on the OK button (~ x=100, y=70 in 1024x768 -> 0.097, 0.091)
    hit = env.step(0, [100 / 1024.0, 70 / 768.0] + [0.0] * 6, confidence=0.9)
    assert hit.success is True
    # Reward includes a small novelty bonus on first occurrence.
    assert hit.reward >= 1.0 - 1e-6


def test_actuator_dispatch_matches_primitive_id():
    env = cl.ClosedLoopEnv()
    res = env.step(16, [0.0] * 8, confidence=0.9)  # screenshot
    assert res.obs.primitive_id == 16
    res = env.step(7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.3], confidence=0.9)
    assert res.obs.primitive_id == 7  # scroll
    assert res.obs.success is True


def test_episodic_memory_capacity():
    env = cl.ClosedLoopEnv(episodic_capacity=5)
    env.reset()
    for _ in range(8):
        env.step(16, [0.0] * 8, confidence=0.9)
    # Capacity = 5 -> deque maxlen enforces that limit.
    assert len(env.episodic_memory) == 5


def test_stats_consistent_with_history():
    env = cl.ClosedLoopEnv()
    env.rollout(_scripted_policy(), n_steps=6)
    stats = env.stats()
    assert stats["n_steps"] == 6
    assert 0 <= stats["success_rate"] <= 1
    assert stats["cumulative_reward"] != 0.0
