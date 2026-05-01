"""Investor demo CLI: synapforge-demo {button|bench|all}.

  $ pip install -e .
  $ synapforge-demo all
"""

from __future__ import annotations

import argparse
import json
import sys


PITCH = """
SynapForge — 30 second pitch

  GPT-class transformers cost millions to train. We built a 375-million-
  parameter LNN+SNN hybrid: continuous-time CfC + spiking PLIF + Hebbian
  STDP plasticity. No transformer, no KV cache.

  Three differentiated claims, each with a CPU-runnable demo:

    1.  NeuroMCP — agents use tools without ever emitting a tool-call
        token. Synapses literally grow into the action space (5% -> 28%
        density on the 4-button validation env).

    2.  R-fold latent thinking — k=8 closed-form algebraic CfC fold,
        gives the model "8 reasoning steps" worth of compute per token.
        Math verified: R=1 exact, R=8 drift 0.3%.

    3.  Inference-time STDP — forward-only Hebbian rule active at
        inference, so the network keeps learning from the live context
        at the *weight* level. (Paper claim; gated on v4.1 ckpt.)

  Same architecture scales to the 375M flagship at ppl 44.2 on
  multilingual chat (zh + en + math).
"""


def cmd_button(args) -> dict:
    from .four_button import run_demo
    print("=== NeuroMCP 4-button demo ===")
    return run_demo(n_trials=args.trials, batch=args.batch, quiet=False)


def cmd_bench(args) -> dict:
    from .rfold_bench import run_demo
    print("=== R-fold bench ===")
    return run_demo(quiet=False)


def cmd_pitch(args) -> None:
    print(PITCH)


def cmd_all(args) -> dict:
    cmd_pitch(args)
    print()
    print("-" * 60)
    out = {}
    out["button"] = cmd_button(args)
    print()
    print("-" * 60)
    out["bench"] = cmd_bench(args)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="synapforge-demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("button", help="NeuroMCP 4-button live demo (~1s)")
    pb.add_argument("--trials", type=int, default=80)
    pb.add_argument("--batch", type=int, default=16)
    pb.set_defaults(fn=cmd_button)

    pr = sub.add_parser("bench", help="R-fold vs sequential CfC bench")
    pr.set_defaults(fn=cmd_bench)

    pp = sub.add_parser("pitch", help="Print 30-second investor pitch")
    pp.set_defaults(fn=cmd_pitch)

    pa = sub.add_parser("all", help="Run pitch + button + bench")
    pa.add_argument("--trials", type=int, default=80)
    pa.add_argument("--batch", type=int, default=16)
    pa.set_defaults(fn=cmd_all)

    pj = sub.add_parser("json", help="Run all demos and dump JSON to stdout")
    pj.add_argument("--trials", type=int, default=40)
    pj.add_argument("--batch", type=int, default=16)
    pj.set_defaults(fn=lambda a: print(json.dumps(cmd_all(a), indent=2, default=str)))

    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
