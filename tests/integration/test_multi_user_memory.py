"""Multi-user memory namespace isolation tests.

Verifies that:
  - Two users (alice, bob) have ZERO data crosstalk through any API path.
  - Querying user A never returns user B's entries.
  - Purging user A leaves user B intact.
  - Persistence: dump → restart → reload returns identical data.
  - Storage layout matches docs/MULTI_USER_MEMORY.md
    (root_dir/<user_id>/{log.jsonl, prefs.json, index.bin}).
  - Backward compat: legacy 3-arg add(user_hash, text, sample_id) still works.
  - Path-traversal user_ids ("../foo", "a/b") are rejected.

VRAM verification:
  - All operations are CPU/disk only.  When torch is available we assert
    that torch.cuda.memory_allocated() does not grow across 100 add()
    calls.  When torch is missing or there's no CUDA device, the assertion
    becomes a structural check (no .cuda() / .to('cuda') anywhere in the
    retrieval_memory module).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# The retrieval-memory module is intentionally import-isolated: it has no
# torch dependency, but `synapforge/__init__.py` eagerly imports torch via
# the action subpackage. To keep this test CPU-only AND independent of the
# broader package's optional deps, load `retrieval_memory.py` directly via
# its file path. This is the same pattern docs/MULTI_USER_MEMORY.md
# advertises ("zero-import-overhead, CPU-only").
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RM_PATH = _REPO_ROOT / "synapforge" / "learn" / "retrieval_memory.py"
_spec = importlib.util.spec_from_file_location(
    "synapforge_learn_retrieval_memory_test", _RM_PATH
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
RetrievalMemory = _mod.RetrievalMemory
PerUserMemory = _mod.PerUserMemory


# ---------------------------------------------------------------------------
# Core isolation
# ---------------------------------------------------------------------------

def test_two_users_isolated(tmp_path: Path) -> None:
    mem = RetrievalMemory(root_dir=tmp_path)

    alice_facts = [
        "alice likes cats",
        "alice owns three plants",
        "alice lives in Tokyo",
        "alice studies linguistics",
        "alice prefers tea over coffee",
    ]
    bob_facts = [
        "bob hates cats",
        "bob drives a motorcycle",
        "bob lives in Berlin",
        "bob studies physics",
        "bob prefers coffee over tea",
    ]
    for f in alice_facts:
        mem.add(f, user_id="alice")
    for f in bob_facts:
        mem.add(f, user_id="bob")

    # Each user must have exactly 5 entries.
    assert mem.for_user("alice").count() == 5
    assert mem.for_user("bob").count() == 5

    # Alice queries should NEVER surface bob's data, and vice-versa.
    alice_hits = mem.query("cats", user_id="alice")
    assert alice_hits, "alice expected to recall her own cat fact"
    for h in alice_hits:
        assert "alice" in h["text"].lower()
        assert "bob" not in h["text"].lower()

    bob_hits = mem.query("cats", user_id="bob")
    assert bob_hits, "bob expected to recall his own cat fact"
    for h in bob_hits:
        assert "bob" in h["text"].lower()
        assert "alice" not in h["text"].lower()


def test_lookup_alias_matches_query(tmp_path: Path) -> None:
    mem = RetrievalMemory(root_dir=tmp_path)
    mem.add("alice likes cats", user_id="alice")
    a = mem.query("cats", user_id="alice")
    b = mem.lookup("cats", user_id="alice")
    assert a == b


def test_purge_only_affects_target_user(tmp_path: Path) -> None:
    mem = RetrievalMemory(root_dir=tmp_path)
    for f in ["alice fact 1", "alice fact 2", "alice fact 3"]:
        mem.add(f, user_id="alice")
    for f in ["bob fact 1", "bob fact 2"]:
        mem.add(f, user_id="bob")

    assert mem.purge(user_id="alice") is True
    # Alice gone; bob untouched.
    assert mem.for_user("alice").count() == 0
    assert mem.for_user("bob").count() == 2
    assert mem.query("fact", user_id="bob"), "bob's fact survived alice purge"


def test_purge_idempotent(tmp_path: Path) -> None:
    mem = RetrievalMemory(root_dir=tmp_path)
    mem.add("alice fact 1", user_id="alice")
    assert mem.purge(user_id="alice") is True
    # Second purge of empty namespace returns False, no exception.
    assert mem.purge(user_id="alice") is False


# ---------------------------------------------------------------------------
# Persistence (dump → restart → reload)
# ---------------------------------------------------------------------------

def test_persistence_across_instances(tmp_path: Path) -> None:
    facts = [
        "alice fact 1",
        "alice fact 2",
        "alice fact 3",
        "alice fact 4",
        "alice fact 5",
    ]

    # Instance A: write 5 facts.
    mem_a = RetrievalMemory(root_dir=tmp_path)
    for f in facts:
        mem_a.add(f, user_id="alice")
    mem_a.set_pref("favorite_animal", "cats", user_id="alice")
    del mem_a  # simulate process restart

    # Instance B: open the same root_dir.
    mem_b = RetrievalMemory(root_dir=tmp_path)
    assert mem_b.for_user("alice").count() == 5
    assert mem_b.get_pref("favorite_animal", user_id="alice") == "cats"
    # All five facts are reachable via query.
    hits = mem_b.query("alice fact", user_id="alice", top_k=5)
    assert len(hits) == 5
    texts = {h["text"] for h in hits}
    assert texts == set(facts)


def test_storage_layout(tmp_path: Path) -> None:
    """Layout must match docs/MULTI_USER_MEMORY.md."""
    mem = RetrievalMemory(root_dir=tmp_path)
    mem.add("hello", user_id="alice")
    mem.set_pref("k", "v", user_id="alice")

    user_dir = tmp_path / "alice"
    assert user_dir.is_dir(), "per-user directory missing"
    assert (user_dir / "log.jsonl").exists()
    assert (user_dir / "prefs.json").exists()
    assert (user_dir / "index.bin").exists()  # touched, may be 0 bytes

    # log.jsonl must be valid JSONL.
    with open(user_dir / "log.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            assert "text" in obj


def test_list_users(tmp_path: Path) -> None:
    mem = RetrievalMemory(root_dir=tmp_path)
    mem.add("a", user_id="alice")
    mem.add("b", user_id="bob")
    mem.add("c", user_id="carol")
    users = mem.list_users()
    assert {"alice", "bob", "carol"}.issubset(set(users))


# ---------------------------------------------------------------------------
# Preferences (per-user)
# ---------------------------------------------------------------------------

def test_prefs_per_user(tmp_path: Path) -> None:
    mem = RetrievalMemory(root_dir=tmp_path)
    mem.set_pref("favorite_animal", "cats", user_id="alice")
    mem.set_pref("favorite_animal", "dogs", user_id="bob")

    assert mem.get_pref("favorite_animal", user_id="alice") == "cats"
    assert mem.get_pref("favorite_animal", user_id="bob") == "dogs"

    # alice's prefs dict must NOT contain bob's keys
    assert "favorite_animal" in mem.prefs(user_id="alice")
    # Bob's update doesn't pollute alice (already verified above), but ensure
    # cross-user prefs() returns a strict subset.
    bob_prefs = mem.prefs(user_id="bob")
    assert bob_prefs == {"favorite_animal": "dogs"}


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_legacy_three_arg_add(tmp_path: Path) -> None:
    """Old call site `mem.add(user_hash, text, sample_id)` must still work."""
    mem = RetrievalMemory(cache_dir=tmp_path)  # legacy positional kwarg
    mem.add("alice_hash", "legacy text", 17)
    hits = mem.query("alice_hash", "legacy", 4)
    assert hits and hits[0]["text"] == "legacy text"
    # Legacy delete_user alias still works
    assert mem.delete_user("alice_hash") is True


def test_default_user_backward_compat(tmp_path: Path) -> None:
    """Code that doesn't know about user_id keeps working under 'default'."""
    mem = RetrievalMemory(root_dir=tmp_path)  # implicit user_id="default"
    mem.add("hello world")
    mem.add("foo bar")
    # No user_id passed to query => default namespace
    hits = mem.query("hello")
    assert hits and hits[0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# Security: user_id sanitization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", "..", ".", ""])
def test_path_traversal_user_ids_rejected(tmp_path: Path, bad) -> None:
    mem = RetrievalMemory(root_dir=tmp_path)
    with pytest.raises((ValueError, TypeError)):
        mem.add("text", user_id=bad)


def test_user_id_none_falls_back_to_bound_default(tmp_path: Path) -> None:
    """user_id=None must NOT raise; it should reuse the bound default user."""
    mem = RetrievalMemory(root_dir=tmp_path, user_id="default")
    mem.add("text", user_id=None)  # falls back to "default"
    assert mem.for_user("default").count() == 1


# ---------------------------------------------------------------------------
# VRAM / GPU isolation guarantee
# ---------------------------------------------------------------------------

def test_no_torch_imports_in_module() -> None:
    """retrieval_memory must remain CPU/disk-only — no torch dependency.

    Static check: source must not import torch or call .cuda()/.to('cuda').
    This is the structural part of the zero-VRAM guarantee.
    """
    src = _RM_PATH.read_text(encoding="utf-8")
    assert "import torch" not in src, "memory module must not import torch"
    assert ".cuda(" not in src, "memory module must not move tensors to CUDA"
    assert "torch.cuda" not in src
    assert ".to('cuda'" not in src
    assert '.to("cuda"' not in src


def test_vram_does_not_grow_during_add(tmp_path: Path) -> None:
    """Dynamic check: add() must not allocate any GPU memory.

    Skip when torch is absent or there's no CUDA device — the structural
    test above is sufficient in that case.
    """
    try:
        import torch  # type: ignore
    except Exception:
        pytest.skip("torch not installed; structural check covers VRAM guarantee")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; structural check covers VRAM guarantee")

    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    mem = RetrievalMemory(root_dir=tmp_path)
    for i in range(100):
        mem.add(f"alice fact {i}", user_id="alice")
    for i in range(100):
        mem.query("alice fact", user_id="alice", top_k=4)
    after = torch.cuda.memory_allocated()
    delta = after - before
    assert delta == 0, f"retrieval memory leaked {delta} bytes of VRAM"


# ---------------------------------------------------------------------------
# PerUserMemory direct-use surface
# ---------------------------------------------------------------------------

def test_peruser_memory_object_works_standalone(tmp_path: Path) -> None:
    ns = PerUserMemory(user_id="alice", root_dir=tmp_path)
    ns.add("hello")
    assert ns.count() == 1
    hits = ns.query("hello")
    assert hits and hits[0]["text"] == "hello"
