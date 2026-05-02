"""DDP launcher skeleton (T8.7) — multi-rental / dual-GPU placeholder.

This is a thin wrapper around `torchrun` that pre-fills the right
arguments for the two paths documented in `docs/PARALLELISM.md`:

  Option A — single-rental dual-GPU (NCCL):
      python scripts/ddp_launch.py --mode local --nproc-per-node 2 \
          -- train_100m_kd.py --backend triton_block --batch-size 256

  Option B — multi-rental DDP (gloo, NOT RECOMMENDED, see PARALLELISM.md):
      python scripts/ddp_launch.py --mode multi --nnodes 2 --node-rank 0 \
          --master-addr 117.74.66.77 --master-port 29500 \
          -- train_100m_kd.py --backend triton_block --batch-size 128

Design intent: the script parses + validates args and prints the
`torchrun` command it would launch. By default it does NOT exec; pass
`--exec` to actually run it. This makes it safe to dry-run / smoke-test
on any box without spinning up distributed.

Cross-host args (--master-addr, --master-port, --node-rank) are required
when --mode=multi, so a typo can never silently fall through to a
single-rank run that wastes a rental hour.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ddp_launch", description=__doc__)
    p.add_argument("--mode", choices=("local", "multi"), default="local",
                   help="local = single rental; multi = cross-rental gloo")
    p.add_argument("--nproc-per-node", type=int, default=1)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--master-addr", default=None,
                   help="required when --mode=multi")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--backend", choices=("nccl", "gloo", "auto"), default="auto",
                   help="forwarded as env var; auto = nccl for local, gloo for multi")
    p.add_argument("--exec", action="store_true",
                   help="actually exec torchrun (default: dry-run prints cmd)")
    p.add_argument("script_and_args", nargs=argparse.REMAINDER,
                   help="train script + its args, e.g. -- train_100m_kd.py --backend triton_block")
    return p


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    return "gloo" if args.mode == "multi" else "nccl"


def validate(args: argparse.Namespace) -> None:
    if args.mode == "multi":
        if args.nnodes < 2:
            raise SystemExit("--mode=multi requires --nnodes>=2")
        if args.master_addr is None:
            raise SystemExit("--mode=multi requires --master-addr")
        if not (0 <= args.node_rank < args.nnodes):
            raise SystemExit(f"--node-rank {args.node_rank} out of [0,{args.nnodes})")
    if not args.script_and_args:
        raise SystemExit("missing train script (use ' -- train_100m_kd.py ...')")


def build_torchrun_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["torchrun"]
    if args.mode == "local":
        cmd += ["--standalone", f"--nproc-per-node={args.nproc_per_node}"]
    else:
        cmd += [
            f"--nnodes={args.nnodes}",
            f"--node-rank={args.node_rank}",
            f"--nproc-per-node={args.nproc_per_node}",
            f"--master-addr={args.master_addr}",
            f"--master-port={args.master_port}",
        ]
    rest = list(args.script_and_args)
    if rest and rest[0] == "--":
        rest = rest[1:]
    cmd += rest
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate(args)
    backend = resolve_backend(args)
    os.environ.setdefault("SYNAPFORGE_DIST_BACKEND", backend)
    cmd = build_torchrun_cmd(args)
    print("# DDP launch (mode=%s backend=%s)" % (args.mode, backend))
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    if args.exec:
        os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
