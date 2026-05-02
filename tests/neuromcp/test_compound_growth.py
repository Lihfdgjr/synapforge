"""Tests: compound emergence via Hebbian wire-together rule."""
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


cg = _load(
    "synapforge.neuromcp.compound_growth",
    "synapforge/neuromcp/compound_growth.py",
)


def test_canned_5x_sequence_emerges_compound():
    """Feed (click_at, type_text, press_key) 5 times in a row -- assert
    a new compound emerges with that exact signature."""
    grower = cg.CompoundGrowth(
        num_primitives=24,
        proposal_window=100,
        reuse_threshold=5,
        n_gram_min=3,
        n_gram_max=3,
        max_compounds=64,
    )
    triplet = (0, 8, 9)  # click_at, type_text, press_key
    new_compound = None
    for repeat in range(5):
        for pid in triplet:
            new_compound = grower.observe(pid) or new_compound
            grower.tick()
    assert new_compound is not None, "no compound emerged after 5x triplet"
    assert tuple(new_compound.primitive_seq) == triplet, (
        f"compound seq mismatch: {new_compound.primitive_seq} != {triplet}"
    )
    assert grower.get(triplet) is new_compound


def test_compound_does_not_emerge_below_threshold():
    grower = cg.CompoundGrowth(
        num_primitives=24,
        reuse_threshold=5,
        n_gram_min=2, n_gram_max=2,
    )
    pair = (0, 8)
    seen = []
    for repeat in range(4):
        for pid in pair:
            res = grower.observe(pid)
            seen.append(res)
            grower.tick()
    assert all(r is None for r in seen), f"premature emergence: {seen}"
    assert grower.get(pair) is None


def test_multiple_compounds_emerge_in_one_session():
    grower = cg.CompoundGrowth(
        num_primitives=24,
        reuse_threshold=3,
        n_gram_min=2, n_gram_max=2,
        proposal_window=200,
    )
    pairs = [(0, 8), (4, 7), (1, 2)]  # click+type, drag+scroll, dbl+right
    emerged = []
    for pair in pairs:
        for _ in range(4):
            for pid in pair:
                got = grower.observe(pid)
                if got is not None:
                    emerged.append(tuple(got.primitive_seq))
                grower.tick()
    # Each pair should have emerged at least once.
    for pair in pairs:
        assert pair in emerged, f"missing compound for {pair}; saw {emerged}"


def test_garbage_collect_idle_compound():
    grower = cg.CompoundGrowth(
        num_primitives=24,
        reuse_threshold=2,
        n_gram_min=2, n_gram_max=2,
        gc_idle_steps=5,
        max_compounds=4,
    )
    pair = (0, 1)
    for _ in range(3):
        for pid in pair:
            grower.observe(pid)
            grower.tick()
    assert grower.get(pair) is not None
    # Now run lots of ticks with NO firings -> idle GC should evict it.
    dead_seen = []
    for _ in range(20):
        dead = grower.tick()
        dead_seen.extend(dead)
    assert grower.get(pair) is None or any(
        tuple(d.primitive_seq) == pair for d in dead_seen
    ), "compound should have been GC'd"


def test_commit_binds_compound_to_codebook_slot():
    grower = cg.CompoundGrowth(
        num_primitives=24,
        reuse_threshold=3,
        n_gram_min=2, n_gram_max=2,
    )
    pair = (0, 9)
    for _ in range(4):
        for pid in pair:
            grower.observe(pid)
            grower.tick()
    proto = grower.get(pair)
    assert proto is not None
    grower.commit(proto, slot_idx=42, embedding=[0.1, 0.2, 0.3])
    assert proto.compound_id == 42
    assert proto.embedding == [0.1, 0.2, 0.3]
