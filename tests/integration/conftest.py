"""Pytest fixtures + collection hooks for integration tests.

Goals:
- Provide hermetic ``tmp_out_dir`` per-test so filesystem state never bleeds.
- Provide a ``fake_ckpt`` builder so SFT/eval pipeline tests don't need a
  real GPU-trained checkpoint.
- Provide a ``fake_tokenizer`` that mimics enough of the HuggingFace
  ``AutoTokenizer`` surface (encode/decode, eos_token_id) without
  requiring `transformers` to be installed.
- Skip tests that require ``playwright`` when it's not installed (the
  web_env tests in particular degrade to a pure-Python mock anyway).

All fixtures are CPU-only and use ``tempfile.TemporaryDirectory()`` so
they're fully isolated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo root importable so `import synapforge` / `import scripts.X`
# both work, regardless of where pytest was invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.fixture
def tmp_out_dir(tmp_path: Path) -> Path:
    """A per-test scratch dir under pytest's tmp_path, already created."""
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def fake_ckpt(tmp_path: Path):
    """Return a callable that writes a tiny ``state_dict``-shaped .pt blob.

    Skips the calling test when torch is unavailable rather than failing
    at fixture-setup time.

    Usage::

        ckpt_path = fake_ckpt("step_000010.pt", weight_dim=8)
    """
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover -- torch optional in CI
        pytest.skip("torch not installed; fake_ckpt unavailable")

    def _build(name: str = "fake.pt", weight_dim: int = 8) -> Path:
        out = tmp_path / name
        sd = {
            "tok_embed.weight": torch.zeros(16, weight_dim),
            "ln_f.weight": torch.ones(weight_dim),
        }
        torch.save({"model": sd, "step": 1, "ppl": 999.0}, out)
        return out

    return _build


class _FakeTokenizer:
    """Minimal HF-compatible tokenizer for tests that don't need real BPE.

    Each ASCII char maps to a unique id; non-ASCII chars share a bucket.
    """

    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = "<pad>"
    pad_token_id = 0

    def encode(self, s, add_special_tokens=False, return_tensors=None):
        ids = [3 + (ord(c) % 200) for c in s]
        if return_tensors == "pt":
            import torch
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) - 3 + 32) if int(i) >= 3 else "" for i in ids)

    def convert_tokens_to_ids(self, token):
        return -1


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``needs_playwright`` when playwright isn't installed."""
    try:
        import playwright  # type: ignore  # noqa: F401
        has_pw = True
    except Exception:
        has_pw = False
    if has_pw:
        return
    skip = pytest.mark.skip(reason="playwright not installed")
    for item in items:
        if "needs_playwright" in item.keywords:
            item.add_marker(skip)
