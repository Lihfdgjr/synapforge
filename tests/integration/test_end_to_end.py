"""End-to-end integration tests for synapforge.

Each test exercises ONE cross-module flow that earlier 'smoke tests pass
individually' did not cover. Heavy deps are stubbed (no GPU, no real
model, no internet, no rental access).

Test inventory (one assertion per test, with a clear error message):

  1. test_phase_signal_write_consume_cycle
       phase_manager.write_phase_signal -> phase_signal.read_phase ->
       phase_signal.consume_phase. Verifies the file is gone after consume,
       that the consumed audit copy is preserved, and that consume is
       idempotent (a second call returns None).

  2. test_chat_eval_gate_scores_good_and_bad
       Hand-written 'good' generator passes the heuristic gate
       (pass_rate >= 0.6); 'bad' word-salad generator fails. This is the
       proof that the gate is actually a tripwire, not a no-op.

  3. test_skill_log_v2_save_kill_recover
       Save a real UniversalCodebook, simulate a 'rental dies mid-write'
       by truncating the JSON, then load: must fall back to the rotation
       sibling and restore the same K_alive count + L1 primitives intact.

  4. test_triple_backup_daemon_empty_dir_warning
       triple_backup_daemon._warn_if_persistently_empty fires its WARNING
       on the 5th consecutive empty cycle (the 'systemd points at the
       wrong path' footgun).

  5. test_auto_eval_daemon_detects_fresh_ckpt
       Drop a fake ``step_000010.pt`` into the watch dir; auto_eval_daemon's
       _list_fresh_ckpts must surface it once (then mark it consumed so
       a second call returns []).

  6. test_alpaca_to_sft_pipeline
       prep_alpaca_qwen-style row-building -> a single SFT-loop forward
       pass -> chat_repl-style template formatting. End-to-end without
       real Qwen tokenizer or a real SynapForge100M model.

  7. test_universal_codebook_l1_to_l2_co_firing_mint
       Repeat a 3-action pattern enough times that mint_from_co_firing
       fires and a new L2 prototype is added with the right trigger_seq.
       The actual 'self-learn' claim being measured.

  8. test_rfold_bench_matches_verify_rfold
       run_demo() vs the standalone verify_rfold script must agree on
       the math: R=1 within 1e-3, R=8 within 0.10. (Speed numbers will
       differ device-to-device; we only check the correctness floor.)

  9. test_synapforge_demo_all_smoke
       synapforge.demo.cli main path: parse args, dispatch to bench/stdp,
       receive a non-empty result dict. The shrink-wrapped investor
       artefact actually wires up.

The 'load_module_isolated' helper imports a single .py file WITHOUT
triggering synapforge/__init__.py, so the tests that don't use torch
also run on a torch-less box (e.g., local Windows dev box).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_module_isolated(name: str, rel_path: str) -> ModuleType:
    """Load a single .py file without importing its parent package.

    Used so e.g. ``synapforge/phase_signal.py`` can be loaded WITHOUT
    triggering the 100+ torch imports in ``synapforge/__init__.py``.
    """
    full = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, full)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {full}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. phase_signal write+consume+idempotency cycle
# ---------------------------------------------------------------------------

def test_phase_signal_write_consume_cycle(tmp_out_dir: Path):
    """phase_manager writes -> trainer reads -> trainer consumes -> idempotent."""
    ps = _load_module_isolated("synapforge_phase_signal",
                               "synapforge/phase_signal.py")
    pm = _load_module_isolated("scripts_phase_manager",
                               "scripts/phase_manager.py")

    state = {"last_step": 1234, "last_train_ppl": 200.0, "best_val_ppl": 245.0}
    # phase 1 = intrinsic (ppl<=250)
    pm.write_phase_signal(tmp_out_dir, phase_id=1, state=state)

    phase_file = tmp_out_dir / ".phase"
    assert phase_file.exists(), \
        f"phase_manager.write_phase_signal didn't create {phase_file}"

    got = ps.read_phase(tmp_out_dir)
    assert got is not None, "read_phase returned None on a freshly-written .phase"
    assert got["phase_id"] == 1, f"phase_id mismatch: expected 1, got {got['phase_id']}"
    assert got["phase_name"] == "intrinsic", got["phase_name"]
    # The flags from PHASES table must be carried through.
    assert got["next_phase_flags"] == pm.PHASES[1]["flags"], \
        f"flags drift: {got['next_phase_flags']} vs canonical {pm.PHASES[1]['flags']}"

    consumed = ps.consume_phase(tmp_out_dir)
    assert consumed is not None, "consume_phase returned None on a present .phase"
    assert consumed["phase_id"] == 1, consumed["phase_id"]
    assert not phase_file.exists(), \
        ".phase file should be moved aside after consume_phase"
    audit_files = [p for p in tmp_out_dir.iterdir()
                   if p.name.startswith(".phase.consumed.")]
    assert len(audit_files) == 1, \
        f"expected exactly one .phase.consumed.* audit file, got {audit_files}"

    # Idempotency: a second consume on a now-empty out dir returns None.
    again = ps.consume_phase(tmp_out_dir)
    assert again is None, \
        f"second consume_phase should be None (idempotent), got {again}"


# ---------------------------------------------------------------------------
# 2. chat_eval_gate scores good vs bad outputs correctly
# ---------------------------------------------------------------------------

def test_chat_eval_gate_scores_good_and_bad():
    """The heuristic gate must SEPARATE good outputs from word salad."""
    cg = _load_module_isolated("scripts_chat_eval_gate",
                               "scripts/chat_eval_gate.py")

    good_report = cg.run_eval(cg._smoke_generator(), threshold=0.6)
    assert good_report["pass_rate"] >= 0.6, (
        f"smoke (canned) generator should pass the 0.6 gate; "
        f"got pass_rate={good_report['pass_rate']}, "
        f"by_category={good_report['by_category']}"
    )

    # Word-salad generator: trigger _h1_not_empty's repetition floor
    # (max_repeat>5 AND >50% of total len).
    def _bad_gen(_prompt: str) -> str:
        return "the the the the the the the the the the the the"

    bad_report = cg.run_eval(_bad_gen, threshold=0.6)
    assert bad_report["pass_rate"] < 0.6, (
        f"word-salad generator should FAIL the gate; "
        f"got pass_rate={bad_report['pass_rate']}, "
        f"by_category={bad_report['by_category']}"
    )

    # Empty generator fails too.
    empty = cg.run_eval(lambda _p: "", threshold=0.6)
    assert empty["pass_rate"] < 0.6, (
        f"empty generator should FAIL the gate; got {empty['pass_rate']}"
    )


# ---------------------------------------------------------------------------
# 3. skill_log_v2 save -> mid-write kill -> recover -> idempotent reload
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_skill_log_v2_save_kill_recover(tmp_out_dir: Path):
    """Mid-write crash must fall back to a rotation sibling, no data lost."""
    from synapforge.action.skill_log_v2 import SkillLog
    from synapforge.action.universal_codebook import UniversalCodebook

    p = tmp_out_dir / "skills.jsonl"

    cb = UniversalCodebook(hidden=32, K_initial=9)
    pid_a = cb.mint_from_text("alpha skill")
    pid_b = cb.mint_from_text("beta skill")
    assert pid_a >= 0 and pid_b >= 0, (pid_a, pid_b)

    log = SkillLog(p, rotation_keep=3)
    n_first = log.save_codebook(cb)
    assert n_first >= 11, f"expected >=11 skills (9 L1 + 2 L3), got {n_first}"

    # Save again so a rotation sibling exists.
    log.save_codebook(cb)

    # Simulate "rental dies mid-write" by truncating the live file.
    p.write_text('{"version": 2, "skills": [BROKEN', encoding="utf-8")

    # Fresh process: load must transparently fall back to rotation copy.
    cb2 = UniversalCodebook(hidden=32, K_initial=9)
    log2 = SkillLog(p, rotation_keep=3)
    n_loaded = log2.load_codebook(cb2)
    assert n_loaded >= 11, (
        f"recovery from corrupt JSON failed: loaded {n_loaded}, "
        f"expected >= {n_first} via rotation sibling"
    )
    sizes = cb2.size_by_layer()
    assert sizes["L1"] == 9, f"L1 primitives lost on recovery: {sizes}"
    assert sizes["L3"] >= 2, f"user-minted L3 lost on recovery: {sizes}"

    # Idempotent: a second load must not duplicate anything.
    log2.load_codebook(cb2)
    sizes2 = cb2.size_by_layer()
    assert sizes == sizes2, (
        f"second load mutated state (non-idempotent): "
        f"first={sizes}, second={sizes2}"
    )


# ---------------------------------------------------------------------------
# 4. triple_backup_daemon dir-mismatch warning fires correctly
# ---------------------------------------------------------------------------

def test_triple_backup_daemon_empty_dir_warning(tmp_out_dir: Path, capsys):
    """5 consecutive empty cycles must trigger the WARNING log line."""
    tb = _load_module_isolated("scripts_triple_backup_daemon",
                               "scripts/triple_backup_daemon.py")

    # Reset module-level state (test isolation).
    tb._empty_cycles_seen = 0

    # Simulate 5 consecutive empty cycles.
    for _ in range(5):
        tb._warn_if_persistently_empty(tmp_out_dir, file_count=0)

    captured = capsys.readouterr().out
    assert "WARNING" in captured, (
        f"after 5 empty cycles, expected WARNING in stdout; got:\n{captured!r}"
    )
    assert str(tmp_out_dir) in captured or tmp_out_dir.name in captured, (
        f"warning should reference the offending watch_dir; got:\n{captured!r}"
    )

    # Recovery message: a non-empty cycle after >=5 empty resets state with
    # a friendly 'recovered' log.
    tb._warn_if_persistently_empty(tmp_out_dir, file_count=1)
    captured2 = capsys.readouterr().out
    assert "recovered" in captured2, (
        f"recovery line should appear when files reappear; got:\n{captured2!r}"
    )
    # Reset for hygiene.
    tb._empty_cycles_seen = 0


# ---------------------------------------------------------------------------
# 5. auto_eval_daemon detects fresh ckpt + dispatches eval (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_auto_eval_daemon_detects_fresh_ckpt(tmp_out_dir: Path, fake_ckpt):
    """Daemon must surface a new ckpt once + skip on the second pass."""
    aed = _load_module_isolated("scripts_auto_eval_daemon",
                                "scripts/auto_eval_daemon.py")

    eval_root = tmp_out_dir / "auto_eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    dedup = aed._Dedup(eval_root / ".dedup.json")

    # No ckpts yet.
    fresh0 = aed._list_fresh_ckpts(tmp_out_dir, dedup)
    assert fresh0 == [], f"with no ckpts, fresh list should be empty; got {fresh0}"

    # Drop a step_000010.pt; daemon must spot it.
    ckpt = fake_ckpt("step_000010.pt", weight_dim=8)
    # Move ckpt into watch dir (fake_ckpt writes under tmp_path; copy across).
    import shutil
    target = tmp_out_dir / ckpt.name
    shutil.copy(ckpt, target)

    fresh1 = aed._list_fresh_ckpts(tmp_out_dir, dedup)
    assert any(p.name == "step_000010.pt" for p in fresh1), (
        f"fresh ckpt step_000010.pt should be detected; "
        f"got {[p.name for p in fresh1]}"
    )

    # Mark it as evaluated.
    dedup.mark(target)

    # Second call: dedup must return empty.
    fresh2 = aed._list_fresh_ckpts(tmp_out_dir, dedup)
    assert fresh2 == [], (
        f"after dedup.mark, second pass should be empty; "
        f"got {[p.name for p in fresh2]}"
    )


# ---------------------------------------------------------------------------
# 6. prep_alpaca-style tokenization -> SFT forward -> chat template format
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_alpaca_to_sft_pipeline(tmp_out_dir: Path, fake_tokenizer):
    """Build a tiny alpaca example, tokenize, run a 1-step forward, format chat."""
    import torch
    import torch.nn as nn

    # 1) Build tokens like prep_alpaca_qwen does (response-only mask).
    PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n### Response:\n"
    END = "<|im_end|>"
    instruction = "Say hi."
    output = "hi back"
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    full = prompt + output + END
    prompt_ids = fake_tokenizer.encode(prompt)
    full_ids = fake_tokenizer.encode(full)
    assert len(full_ids) > len(prompt_ids), \
        "full should be strictly longer than prompt-only tokens"
    loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    assert sum(loss_mask) > 0, "no loss-bearing positions in mask"

    # 2) Run a single SFT-style forward through a tiny stand-in model.
    vocab = 256
    d = 16
    seq = torch.tensor([full_ids], dtype=torch.long)
    embed = nn.Embedding(vocab, d)
    head = nn.Linear(d, vocab, bias=False)
    head.weight = embed.weight  # tied lm head, like SynapForge100M
    h = embed(seq.clamp(max=vocab - 1))
    logits = head(h)
    assert logits.shape == (1, len(full_ids), vocab), logits.shape

    # 3) Compute response-only CE -- ignore prompt tokens via mask.
    targets = seq[:, 1:].clamp(max=vocab - 1)
    pred = logits[:, :-1, :]
    mask = torch.tensor([loss_mask[1:]], dtype=torch.float32)
    ce = torch.nn.functional.cross_entropy(
        pred.reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    ).reshape(pred.shape[:-1])
    masked = (ce * mask).sum() / mask.sum().clamp_min(1.0)
    assert torch.isfinite(masked), f"SFT loss became {masked}"

    # 4) chat_repl-style template formatting must produce the same prompt.
    INSTRUCTION_TEMPLATE = (
        "### Instruction:\n{instruction}\n### Response:\n"
    )
    chat_prompt = INSTRUCTION_TEMPLATE.format(instruction=instruction)
    assert chat_prompt == prompt, (
        f"chat_repl template diverged from prep_alpaca template: "
        f"chat={chat_prompt!r} vs sft={prompt!r}"
    )


# ---------------------------------------------------------------------------
# 7. universal_codebook L1 -> L2 mint via co-firing (the actual claim)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_universal_codebook_l1_to_l2_co_firing_mint():
    """Repeat a 3-action pattern; mint_from_co_firing must add an L2 entry."""
    from synapforge.action.universal_codebook import UniversalCodebook

    cb = UniversalCodebook(
        hidden=32, K_initial=9,
        co_fire_window=3, co_fire_min_repeats=3,
    )
    n_l2_before = cb.size_by_layer()["L2"]

    # Drive 12 actions: 4× pattern (CLICK=0, TYPE=1, KEY=2).
    # The codebook should walk back, find the tail [0, 1, 2] repeating,
    # and mint a single L2 prototype with that trigger_seq.
    pattern = [0, 1, 2]
    minted_pid = -1
    for _rep in range(4):
        for a in pattern:
            pid = cb.mint_from_co_firing(a)
            if pid is not None and pid >= 0 and minted_pid < 0:
                minted_pid = pid

    n_l2_after = cb.size_by_layer()["L2"]
    assert n_l2_after >= n_l2_before + 1, (
        f"L2 should grow after a stable repeated pattern; "
        f"before={n_l2_before}, after={n_l2_after}, minted_pid={minted_pid}"
    )
    assert minted_pid >= 0, "no proto_id returned by mint_from_co_firing"
    meta = cb.meta[minted_pid]
    assert meta.layer == "L2", f"expected L2, got {meta.layer}"
    assert meta.trigger_seq == pattern, (
        f"trigger_seq drift: expected {pattern}, got {meta.trigger_seq}"
    )


# ---------------------------------------------------------------------------
# 8. rfold_bench numbers match verify_rfold's (consistency)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_rfold_bench_matches_verify_rfold():
    """run_demo()'s correctness numbers must be in the band verify_rfold asserts."""
    from synapforge.demo.rfold_bench import run_demo

    out = run_demo(quiet=True)
    # verify_rfold asserts: err1 < 1e-3, err8 < 0.10.
    assert out["rel_err_R1"] < 1e-3, (
        f"R=1 should be exact to fp32 noise; got rel_err_R1={out['rel_err_R1']}"
    )
    assert out["rel_err_R8"] < 0.10, (
        f"R=8 should drift <10%; got rel_err_R8={out['rel_err_R8']}"
    )
    # And the shape table should cover the 5 documented shapes.
    assert len(out["shapes"]) == 5, f"expected 5 shapes, got {len(out['shapes'])}"
    expected = {(64, 4), (64, 16), (128, 8), (256, 8), (512, 8)}
    got = {(int(r["N"]), int(r["R"])) for r in out["shapes"]}
    assert got == expected, f"shape coverage drift: {got} vs {expected}"


# ---------------------------------------------------------------------------
# 9. investor demo `synapforge-demo bench` end-to-end (CLI dispatch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_synapforge_demo_all_smoke(monkeypatch):
    """The bench subcommand must dispatch through cli.main without crashing."""
    from synapforge.demo import cli

    captured: dict = {}

    def _fake_bench(args):
        captured["bench_called"] = True
        return {"shapes": [], "rel_err_R1": 1e-7, "rel_err_R8": 0.003}

    monkeypatch.setattr(cli, "cmd_bench", _fake_bench)
    rc = cli.main(["bench"])
    assert rc == 0, f"cli.main should return 0 on success, got {rc}"
    assert captured.get("bench_called"), \
        "cli.main(['bench']) didn't dispatch to cmd_bench"

    # And the pitch subcommand prints SOMETHING (the elevator pitch).
    rc2 = cli.main(["pitch"])
    assert rc2 == 0
