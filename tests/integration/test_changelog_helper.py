"""T6.5: changelog_helper.append_changelog_phase_transition tests.

Spec coverage (from docs/DEEP_MAINT_QUEUE.md T6.5):
  1. test_create_new_changelog            - file absent -> created with header
  2. test_append_to_existing               - existing file kept verbatim + appended
  3. test_atomic_write                     - no partial / .tmp files left after success
  4. test_phase_transition_format          - parse-back the appended entry

Bonus guards:
  5. test_phase_transition_format_with_optional_fields - n/a fallbacks
  6. test_appending_does_not_modify_existing_entries - byte-for-byte preservation
"""
from __future__ import annotations

import importlib.util
import re
from datetime import datetime
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HELPER_PATH = _REPO_ROOT / "scripts" / "changelog_helper.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "changelog_helper", _HELPER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def helper():
    return _load_helper()


def test_create_new_changelog(tmp_out_dir, helper):
    """File absent -> create with `# SynapForge Changelog\\n\\n` header + entry."""
    cl = tmp_out_dir / "CHANGELOG.md"
    assert not cl.exists()
    out = helper.append_changelog_phase_transition(
        phase_id=1,
        val_ppl=240.5,
        step=4000,
        ckpt_path="/workspace/runs/v24h_qwen3/step_004000_no_optim.pt",
        new_flags=[
            "--self-learn-ttt",
            "--self-learn-k", "8",
            "--curiosity-weight", "0.05",
            "--phase-aware",
        ],
        changelog_path=cl,
        now=datetime(2026, 5, 2, 1, 30),
    )
    assert out == cl.resolve() or out == cl  # _resolve_changelog_path may resolve
    text = cl.read_text(encoding="utf-8")
    assert text.startswith("# SynapForge Changelog\n\n"), repr(text[:60])
    assert "## 2026-05-02 01:30 - Phase 1 reached" in text
    assert "val_ppl_holdout: 240.5" in text
    assert "step: 4000" in text
    assert "step_004000_no_optim.pt" in text
    assert "--self-learn-ttt" in text


def test_append_to_existing(tmp_out_dir, helper):
    """Existing file content is preserved verbatim; new entry is appended."""
    cl = tmp_out_dir / "CHANGELOG.md"
    seed = (
        "# SynapForge Changelog\n\n"
        "## 2026-05-01 12:00 - Some prior entry\n"
        "- bullet a\n"
        "- bullet b\n"
    )
    cl.write_text(seed, encoding="utf-8")
    helper.append_changelog_phase_transition(
        phase_id=2,
        val_ppl=95.2,
        step=8500,
        ckpt_path="/workspace/runs/v24h_qwen3/step_008500_no_optim.pt",
        new_flags=["--modal-list", "image,audio,time_series", "--phase-aware"],
        changelog_path=cl,
        now=datetime(2026, 5, 2, 14, 7),
    )
    text = cl.read_text(encoding="utf-8")
    # Old content untouched (byte-level prefix match):
    assert text.startswith(seed), "existing entry got mutated"
    # New entry appended:
    assert "## 2026-05-02 14:07 - Phase 2 reached" in text
    assert "val_ppl_holdout: 95.2" in text
    assert "--modal-list image,audio,time_series" in text


def test_atomic_write(tmp_out_dir, helper, monkeypatch):
    """If atomic write crashes mid-way, no partial CHANGELOG.md and no
    leftover .tmp files survive."""
    cl = tmp_out_dir / "CHANGELOG.md"
    seed = "# SynapForge Changelog\n\nseed line\n"
    cl.write_text(seed, encoding="utf-8")

    real_replace = helper.os.replace

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-rename failure")

    # Force the helper's os.replace to raise -> _atomic_write should clean up.
    monkeypatch.setattr(helper.os, "replace", boom)

    with pytest.raises(RuntimeError, match="simulated"):
        helper.append_changelog_phase_transition(
            phase_id=1,
            val_ppl=240.0,
            step=4000,
            ckpt_path="/tmp/step.pt",
            changelog_path=cl,
        )

    # Original CHANGELOG.md unchanged.
    assert cl.read_text(encoding="utf-8") == seed
    # No leftover .tmp files in the dir.
    leftovers = list(tmp_out_dir.glob("*.tmp"))
    assert leftovers == [], f"leftover temp files survived: {leftovers}"

    # Restore os.replace and confirm a successful write still works.
    monkeypatch.setattr(helper.os, "replace", real_replace)
    helper.append_changelog_phase_transition(
        phase_id=1,
        val_ppl=240.0,
        step=4000,
        ckpt_path="/tmp/step.pt",
        changelog_path=cl,
    )
    assert "Phase 1 reached" in cl.read_text(encoding="utf-8")


def test_phase_transition_format(tmp_out_dir, helper):
    """Append-and-parse-back. Asserts canonical line layout so downstream
    automation (or a diligent investor) can grep for fields."""
    cl = tmp_out_dir / "CHANGELOG.md"
    helper.append_changelog_phase_transition(
        phase_id=3,
        val_ppl=58.4,
        step=12345,
        ckpt_path="/workspace/runs/v24h_qwen3/step_012345_no_optim.pt",
        new_flags=[
            "--sft-data", "/workspace/data/alpaca_zh/alpaca_zh.json",
            "--response-only-loss",
            "--lr", "1e-4",
            "--phase-aware",
        ],
        changelog_path=cl,
        now=datetime(2026, 5, 3, 9, 41),
    )
    text = cl.read_text(encoding="utf-8")

    # Header line: "## YYYY-MM-DD HH:MM - Phase N reached"
    m = re.search(
        r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) - Phase (\d+) reached$",
        text,
        re.MULTILINE,
    )
    assert m is not None, f"header line missing/malformed in:\n{text}"
    assert m.group(1) == "2026-05-03 09:41"
    assert m.group(2) == "3"

    # Bullet body — assert the four canonical bullets in order:
    expected_bullets = [
        r"- val_ppl_holdout: 58\.4 \(target <= 60",
        r"- step: 12345",
        r"- warmstart ckpt: \S*step_012345_no_optim\.pt",
        # Single-line bullet; .* with DOTALL only matches within the line
        # because the bullet body has no embedded newlines.
        r"- new flags added: --sft-data \S+ --response-only-loss --lr 1e-4 --phase-aware",
    ]
    body = text[m.end():]
    for pat in expected_bullets:
        assert re.search(pat, body), (
            f"bullet missing for pattern: {pat!r}\nbody=\n{body}"
        )


def test_phase_transition_format_with_optional_fields(tmp_out_dir, helper):
    """When val_ppl/step/ckpt/flags are None, the helper writes 'n/a'/'(none)'
    instead of crashing. Manual phase transitions can use this surface."""
    cl = tmp_out_dir / "CHANGELOG.md"
    helper.append_changelog_phase_transition(
        phase_id=0,
        val_ppl=None,
        step=None,
        ckpt_path=None,
        new_flags=None,
        changelog_path=cl,
        now=datetime(2026, 5, 2, 0, 0),
    )
    text = cl.read_text(encoding="utf-8")
    assert "Phase 0 reached" in text
    assert "val_ppl_holdout: n/a" in text
    assert "step: n/a" in text
    assert "warmstart ckpt: n/a" in text
    assert "new flags added: (none)" in text


def test_appending_does_not_modify_existing_entries(tmp_out_dir, helper):
    """Two consecutive appends: the original two prefixes are byte-identical."""
    cl = tmp_out_dir / "CHANGELOG.md"
    helper.append_changelog_phase_transition(
        phase_id=1,
        val_ppl=240.0,
        step=4000,
        ckpt_path="/tmp/a.pt",
        new_flags=["--alpha"],
        changelog_path=cl,
        now=datetime(2026, 5, 2, 1, 0),
    )
    after_first = cl.read_text(encoding="utf-8")

    helper.append_changelog_phase_transition(
        phase_id=2,
        val_ppl=99.0,
        step=8000,
        ckpt_path="/tmp/b.pt",
        new_flags=["--beta"],
        changelog_path=cl,
        now=datetime(2026, 5, 2, 2, 0),
    )
    after_second = cl.read_text(encoding="utf-8")

    # The full first-append text must remain verbatim at the start.
    assert after_second.startswith(after_first), (
        "first entry was mutated by second append"
    )
    # And the second header is present.
    assert "Phase 2 reached" in after_second
