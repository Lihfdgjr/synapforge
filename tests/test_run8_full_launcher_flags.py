"""Verify scripts/launch_synap1_ultra_run8_full.sh activates the
'decorative -> real' flags expected by feature/activate-decorative.

These tests do NOT execute the launcher (it requires a CUDA box +
warmstart ckpt). They lint the static contents of the script:

* Multimodal: --modal-list, --modal-data-dir, --modal-alpha
* Web daemon corpus glob: web_self_learn / web_*.parquet at weight 0.10
* R-fold ON in training (--rfold + --rfold-chunk)
* Phase-aware mode is on (so phase_manager gates modal at val_ppl <= 100)
* Sparse-spike synapse (the PLIF revival path) is on
* Async data pipeline (so the daemon-driven parquet glob doesn't stall
  the main GPU loop)

Also: bash -n syntax pass on the script (catches typos/quoting errors).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "scripts" / "launch_synap1_ultra_run8_full.sh"


def _read() -> str:
    assert LAUNCHER.is_file(), f"missing launcher at {LAUNCHER}"
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_exists() -> None:
    assert LAUNCHER.is_file(), str(LAUNCHER)


def test_launcher_bash_n_passes() -> None:
    """``bash -n`` syntactic check; only meaningful when bash is on PATH."""
    bash = shutil.which("bash")
    if bash is None:
        # Windows-dev box w/o WSL bash; skip silently.
        return
    out = subprocess.run(
        [bash, "-n", str(LAUNCHER)],
        capture_output=True, text=True, timeout=10,
    )
    assert out.returncode == 0, (
        f"bash -n failed: {out.returncode} stderr=\n{out.stderr}"
    )


def test_modal_flags_present() -> None:
    body = _read()
    # The flags are constructed via $MODAL_ARGS so we look for the inner
    # template strings.
    assert "--modal-list" in body, "missing --modal-list"
    assert "--modal-data-dir" in body, "missing --modal-data-dir"
    assert "--modal-alpha" in body, "missing --modal-alpha"


def test_web_daemon_glob_present() -> None:
    body = _read()
    assert "web_self_learn" in body, (
        "launcher must reference the web_self_learn rolling parquet glob"
    )
    assert "web_*.parquet" in body, "missing rolling parquet pattern"
    # Weight should be small (default 0.10).
    assert "WEB_DAEMON_WEIGHT" in body, (
        "missing tunable WEB_DAEMON_WEIGHT env var"
    )


def test_rfold_in_training() -> None:
    body = _read()
    assert "--rfold" in body, "must include --rfold for training-time R-fold"
    assert "--rfold-chunk" in body, "must include --rfold-chunk"


def test_phase_aware_enabled() -> None:
    body = _read()
    assert "--phase-aware" in body, (
        "phase-aware mode must be on so phase_manager auto-gates modal "
        "at val_ppl <= 100"
    )


def test_sparse_spike_synapse_on() -> None:
    body = _read()
    assert "--sparse-spike-synapse" in body, (
        "sparse-spike-synapse path is the PLIF revival lever for Run 8"
    )


def test_async_data_pipeline_on() -> None:
    body = _read()
    assert "--async-data-pipeline" in body, (
        "async data pipeline is required for the daemon-driven glob to "
        "not stall the main GPU loop"
    )


def test_no_blacklisted_flags() -> None:
    """Black-list reminder lives in echos / comments. The actual command
    must NOT include these flags (per Run 7 charter).

    We extract only the python invocation block (the heredoc inside
    ``setsid bash -c "cd '...' && exec python3 -u train_100m_kd.py ..."``)
    so the warning banner doesn't false-trip the substring scan.
    """
    body = _read()
    start = body.find("exec python3 -u train_100m_kd.py")
    assert start > 0, "could not locate trainer invocation"
    # Stop at the closing ``" </dev/null &``
    end = body.find('" </dev/null', start)
    assert end > start, "could not locate trainer invocation end"
    py_invocation = body[start:end]
    assert "--kwta-k" not in py_invocation, "kWTA was black-listed in Run 7+"
    assert "--cuda-graphs" not in py_invocation, (
        "CUDA Graphs auto-disable when mixins are live; do NOT pass"
    )
    assert "--weight-quant ternary" not in py_invocation, (
        "Ternary BitNet QAT is deferred (LM-head reset risk)"
    )


def test_kd_distill_data_present() -> None:
    """Run 8 base must keep the KD distill data file."""
    body = _read()
    assert "kd_distill_v1_text.parquet" in body or "kd_distill" in body, (
        "missing kd_distill data input"
    )


def test_neuromcp_present_so_stdp_engages() -> None:
    """The STDP-only tag on SparseSynapticLayer.weight is wired in by
    enabling NeuroMCP in the trainer (--neuromcp-weight > 0). Without
    NeuroMCP enabled the head is not constructed and the tag is moot."""
    body = _read()
    assert "--neuromcp-weight" in body, "must enable NeuroMCP for STDP wire-in"


def test_self_learn_ttt_on() -> None:
    body = _read()
    assert "--self-learn-ttt" in body


def test_data_files_includes_glob() -> None:
    body = _read()
    # The launcher composes ``--data-files`` from a base concatenation.
    assert "--data-files" in body
    # We construct DATA_FILES with the rolling glob so the variable is the
    # source of truth.
    assert "DATA_FILES_BASE" in body, "must define DATA_FILES_BASE"
    assert "DATA_FILES=" in body, "must build DATA_FILES variable"


if __name__ == "__main__":
    test_launcher_exists(); print("OK exists")
    test_launcher_bash_n_passes(); print("OK bash -n")
    test_modal_flags_present(); print("OK modal flags")
    test_web_daemon_glob_present(); print("OK web daemon glob")
    test_rfold_in_training(); print("OK rfold in training")
    test_phase_aware_enabled(); print("OK phase-aware")
    test_sparse_spike_synapse_on(); print("OK sparse-spike-synapse")
    test_async_data_pipeline_on(); print("OK async data pipeline")
    test_no_blacklisted_flags(); print("OK no black-listed flags")
    test_kd_distill_data_present(); print("OK kd distill data")
    test_neuromcp_present_so_stdp_engages(); print("OK neuromcp flag")
    test_self_learn_ttt_on(); print("OK self-learn-ttt")
    test_data_files_includes_glob(); print("OK data-files variable")
