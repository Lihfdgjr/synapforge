"""End-to-end smoke test for the T1.1 chat sample CLI.

Verifies that ``synapforge.demo.chat_demo``:
- can be imported (no syntax errors at module level)
- exposes the queue-T1.1 prompt sets verbatim (regression guard:
  ``docs/DEEP_MAINT_QUEUE.md`` validation gates check for those exact
  strings -- if anyone renames them, the cron CLI breaks silently)
- builds an argparse parser without errors
- runs ``run_demo`` end-to-end against a tiny fake ckpt + fake tokenizer
  and writes a JSON file shaped for the queue spec
  (``{ckpt, step, samples: [{lang, prompt, response}, ...]}``)
- accepts both ``prompt_set="t1"`` and ``prompt_set="legacy"``
- falls back to recorded-replay (mode='recorded') when no ckpt exists,
  without crashing or printing diagnostic noise (verbose=False)
- strips ``module.`` prefixes from DDP-saved state_dicts
- ``--verbose`` flag is wired (parser accepts it)

These run on CPU only; the live load path is exercised against a tiny
constructed SynapForge100M (vocab=64, d=16, n_layers=1) so the test
finishes in <5s without GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def torch_or_skip():
    return pytest.importorskip("torch")


def test_module_imports():
    """The CLI module imports without ImportError or SyntaxError."""
    import synapforge.demo.chat_demo as mod

    # Public-facing symbols the queue / cli.py / disclose.py rely on.
    assert hasattr(mod, "run_demo")
    assert hasattr(mod, "main")
    assert hasattr(mod, "_FALLBACK_CFG")
    assert hasattr(mod, "_try_load_live")
    assert hasattr(mod, "EN_PROMPTS_T1")
    assert hasattr(mod, "ZH_PROMPTS_T1")
    assert hasattr(mod, "EN_PROMPTS")  # legacy alias
    assert hasattr(mod, "ZH_PROMPTS")  # legacy alias


def test_t1_prompts_are_verbatim_per_queue_spec():
    """Regression guard: queue T1.1 names these exact strings."""
    from synapforge.demo.chat_demo import EN_PROMPTS_T1, ZH_PROMPTS_T1

    # docs/DEEP_MAINT_QUEUE.md T1.1 — DO NOT change without updating
    # the queue validation gates and CHAT_SAMPLES.md schema.
    assert EN_PROMPTS_T1 == [
        "The capital of France is",
        "In the morning, I like to",
        "Photosynthesis is the process",
        "def reverse_string(s):",
        "Once upon a time,",
    ]
    assert ZH_PROMPTS_T1 == [
        "中国的首都是",
        "今天天气",
        "光合作用是",
        "我喜欢吃",
        "从前有一个",
    ]


def test_argparser_builds_no_syntax_errors():
    """``main(['--help'])`` exercises the argparse build."""
    from synapforge.demo import chat_demo

    # argparse calls sys.exit(0) on --help; trap that.
    with pytest.raises(SystemExit) as exc:
        chat_demo.main(["--help"])
    assert exc.value.code == 0


def test_argparser_accepts_all_documented_flags():
    """Every flag the queue + this module's docstring document parses cleanly."""
    from synapforge.demo import chat_demo
    import argparse

    # Re-parse the actual parser by patching out run_demo so we can run
    # main() without doing real work.
    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {"mode": "stub", "samples": [], "pairs": []}

    orig = chat_demo.run_demo
    chat_demo.run_demo = _fake_run
    try:
        rc = chat_demo.main([
            "--ckpt", "/no/such/ckpt.pt",
            "--tokenizer-path", "Qwen/Qwen2.5-0.5B",
            "--max-new", "60",
            "--temperature", "0.7",
            "--save", "/tmp/_chat_smoke.json",
            "--device", "cpu",
            "--verbose",
            "--prompt-set", "t1",
        ])
    finally:
        chat_demo.run_demo = orig
    assert rc == 0
    assert captured["max_new"] == 60
    assert captured["temperature"] == pytest.approx(0.7)
    assert captured["device"] == "cpu"
    assert captured["verbose"] is True
    assert captured["prompt_set"] == "t1"


def test_recorded_fallback_when_no_ckpt(tmp_path, capsys):
    """No ckpt + no tokenizer => recorded replay path, mode='recorded'."""
    from synapforge.demo.chat_demo import run_demo

    save = tmp_path / "out.json"
    out = run_demo(
        ckpt=str(tmp_path / "does-not-exist.pt"),
        tokenizer_path=None,
        max_new=8,
        temperature=0.7,
        save_path=str(save),
        device="cpu",
        verbose=False,  # quiet by default per task spec
        prompt_set="legacy",  # recorded transcript matches legacy prompts
    )
    captured = capsys.readouterr()
    # verbose=False => no stdout chatter
    assert captured.out == "", f"unexpected stdout: {captured.out!r}"

    assert out["mode"] == "recorded"
    assert out["n_prompts"] == 10
    # Save format must carry "samples" with lang/prompt/response triples.
    assert save.is_file()
    j = json.loads(save.read_text(encoding="utf-8"))
    assert j["mode"] == "recorded"
    assert "samples" in j
    assert len(j["samples"]) == 10
    for s in j["samples"]:
        assert s["lang"] in ("EN", "ZH")
        assert "prompt" in s and "response" in s


def test_strip_module_prefix_helper():
    """DDP-saved state_dicts use 'module.' prefix; loader must strip it."""
    from synapforge.demo.chat_demo import _strip_module_prefix

    sd = {
        "module.tok_embed.weight": 1,
        "module.ln_f.weight": 2,
        "ln_other.weight": 3,
    }
    out = _strip_module_prefix(sd)
    assert "tok_embed.weight" in out
    assert "ln_f.weight" in out
    assert "ln_other.weight" in out  # non-module keys preserved
    assert "module.tok_embed.weight" not in out

    # No-op when nothing has the prefix.
    plain = {"a.b": 1}
    assert _strip_module_prefix(plain) == plain


def test_live_load_with_tiny_ckpt(tmp_path, torch_or_skip, monkeypatch, capsys):
    """End-to-end live path with a tiny fake ckpt + fake tokenizer.

    This is the test the task spec asks for: 'mocked tiny ckpt + tokenizer,
    runs end-to-end without errors'. We monkeypatch the
    ``transformers.AutoTokenizer`` lookup so the test doesn't need network /
    HF cache, and build a tiny SynapForge100M (vocab=64, d=16, n_layers=1)
    that fits in a few MB.
    """
    torch = torch_or_skip
    from synapforge.demo import chat_demo
    from synapforge.model_100m import SynapForge100M

    # ---------- 1) build a tiny ckpt with the same shape as production ----------
    cfg = {
        "vocab": 64,
        "d": 16,
        "n_layers": 1,
        "loop_depth": 1,
        "max_seq": 32,
        "ffn_ratio": 4.0,
        "sparsity": 0.5,
        "dropout": 0.0,
        "tie_lm_head": True,
    }
    model = SynapForge100M(
        vocab=cfg["vocab"], d=cfg["d"], n_layers=cfg["n_layers"],
        loop_depth=cfg["loop_depth"], max_seq=cfg["max_seq"],
        ffn_ratio=cfg["ffn_ratio"], sparsity=cfg["sparsity"],
        dropout=cfg["dropout"], tie_lm_head=cfg["tie_lm_head"],
    )

    # Round-trip via 'module.' prefix to exercise the strip helper.
    sd = {f"module.{k}": v for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "tiny.pt"
    torch.save({"model": sd, "step": 42, "loss": 99.0, "config": cfg},
               str(ckpt_path))

    # ---------- 2) fake tokenizer that doesn't need network ----------
    class _FakeTok:
        eos_token = "<eos>"
        eos_token_id = 2
        pad_token = "<pad>"
        pad_token_id = 0

        def encode(self, s, add_special_tokens=False, return_tensors=None):
            # Map each character to a small id within vocab=64.
            ids = [3 + (ord(c) % 60) for c in s] or [3]
            if return_tensors == "pt":
                return torch.tensor([ids], dtype=torch.long)
            return ids

        def decode(self, ids, skip_special_tokens=True):
            # Decode is best-effort; the smoke test only asserts shape.
            return "x" * int(len(ids))

        def convert_tokens_to_ids(self, token):
            return -1

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, trust_remote_code=False):  # noqa: ARG004
            return _FakeTok()

    # Monkeypatch transformers.AutoTokenizer in case it's importable.
    # If not installed, the real module won't be touched.
    fake_transformers = type(sys)("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # ---------- 3) run end-to-end ----------
    save = tmp_path / "live.json"
    out = chat_demo.run_demo(
        ckpt=str(ckpt_path),
        tokenizer_path="fake/tok",
        max_new=4,
        temperature=0.7,
        save_path=str(save),
        device="cpu",
        verbose=False,
        prompt_set="t1",
    )
    captured = capsys.readouterr()
    assert captured.out == "", f"unexpected stdout: {captured.out!r}"

    # Live mode reached, the strip helper accepted the 'module.' prefix.
    assert out["mode"] == "live"
    assert out["prompt_set"] == "t1"
    assert out["n_prompts"] == 10
    # 'step' must round-trip from the ckpt dict.
    assert out["step"] == 42

    # Save format spec.
    assert save.is_file()
    j = json.loads(save.read_text(encoding="utf-8"))
    assert j["ckpt"].endswith("tiny.pt")
    assert j["step"] == 42
    assert isinstance(j["samples"], list)
    assert len(j["samples"]) == 10
    en = [s for s in j["samples"] if s["lang"] == "EN"]
    zh = [s for s in j["samples"] if s["lang"] == "ZH"]
    assert len(en) == 5 and len(zh) == 5
    # Prompts must match T1.1 verbatim.
    assert en[0]["prompt"] == "The capital of France is"
    assert zh[0]["prompt"] == "中国的首都是"
    for s in j["samples"]:
        assert isinstance(s["response"], str)


def test_run_demo_quiet_by_default(tmp_path, capsys):
    """Default invocation is silent unless verbose=True."""
    from synapforge.demo.chat_demo import run_demo

    save = tmp_path / "q.json"
    run_demo(
        ckpt=str(tmp_path / "missing.pt"),
        tokenizer_path=None,
        max_new=4,
        save_path=str(save),
        device="cpu",
        # default verbose=False
        prompt_set="legacy",
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert save.is_file()
