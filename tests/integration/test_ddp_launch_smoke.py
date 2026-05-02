"""Smoke test for `scripts/ddp_launch.py` (T8.7).

We do NOT actually run distributed (no second rental yet — see
`docs/PARALLELISM.md`). The launcher is intentionally dry-run-by-default
so this test just verifies:

  1. Single-rank local mode parses cleanly + builds a torchrun command
     containing `--standalone --nproc-per-node=N`.
  2. Multi-node mode without `--master-addr` is rejected (otherwise a
     typo would silently fall through to a single-rank run that wastes
     a rental hour).

Both tests are CPU-only and skip the actual `os.execvp`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LAUNCHER = _REPO_ROOT / "scripts" / "ddp_launch.py"


def _load_launcher():
    """Import scripts/ddp_launch.py as a module (it's a script, not a package)."""
    spec = importlib.util.spec_from_file_location("ddp_launch", str(_LAUNCHER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ddp_launch"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_args_parse_singlerank():
    """Local mode with --nproc-per-node=1 builds a sane torchrun command.

    Asserts:
      - `validate` does not raise
      - resolved backend is `nccl` (local default)
      - command contains `--standalone` and `--nproc-per-node=1`
      - the train script and its args appear in the command
    """
    mod = _load_launcher()
    args = mod.build_parser().parse_args([
        "--mode", "local",
        "--nproc-per-node", "1",
        "--", "train_100m_kd.py", "--backend", "triton_block",
    ])
    mod.validate(args)
    assert mod.resolve_backend(args) == "nccl"
    cmd = mod.build_torchrun_cmd(args)
    assert cmd[0] == "torchrun"
    assert "--standalone" in cmd
    assert "--nproc-per-node=1" in cmd
    assert "train_100m_kd.py" in cmd
    assert "--backend" in cmd and "triton_block" in cmd
    # `--` separator should be stripped before forwarding
    assert "--" not in cmd


def test_master_addr_required_for_multinode():
    """Multi-node mode without --master-addr must SystemExit, not silently
    fall through. A typo here on the master rental would otherwise burn
    cash on a stalled torchrun rendezvous.
    """
    mod = _load_launcher()
    args = mod.build_parser().parse_args([
        "--mode", "multi",
        "--nnodes", "2",
        "--node-rank", "0",
        "--", "train_100m_kd.py",
    ])
    with pytest.raises(SystemExit, match="master-addr"):
        mod.validate(args)


def test_multinode_with_master_addr_resolves_gloo_backend():
    """Sanity check the happy path for multi-rental: backend defaults to
    gloo, torchrun command carries the right rendezvous args.
    """
    mod = _load_launcher()
    args = mod.build_parser().parse_args([
        "--mode", "multi",
        "--nnodes", "2",
        "--node-rank", "1",
        "--master-addr", "117.74.66.77",
        "--master-port", "29500",
        "--", "train_100m_kd.py", "--batch-size", "128",
    ])
    mod.validate(args)
    assert mod.resolve_backend(args) == "gloo"
    cmd = mod.build_torchrun_cmd(args)
    assert "--nnodes=2" in cmd
    assert "--node-rank=1" in cmd
    assert "--master-addr=117.74.66.77" in cmd
    assert "--master-port=29500" in cmd
