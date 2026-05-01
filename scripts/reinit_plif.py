"""Reinit PLIF cell parameters in a checkpoint -- attempts to break dead-PLIF lockup.

Usage:
    python scripts/reinit_plif.py <src.pt> <dst.pt>

What it does:
    Loads ckpt, scans state_dict for keys ending in `.plif.*.log_tau`,
    `.plif.*.threshold`, `.plif.*.V`, overwrites with patched defaults
    (tau_init=2.5, threshold_init=0.05, V=0). Saves to <dst.pt>.
    Embed/CfC/FFN/lm_head are untouched.

CAVEAT (discovered 2026-05-01):
    PLIFCell parameter naming changed during the v4.x refactor.
    Old ckpts (e.g., step_002250.pt from v24h_qwen) use Mamba/SSM-style
    `A_log` and `threshold` under `blocks.{i}.plif.shared.block.*`.
    New PLIFCell (synapforge/surrogate.py:351-352) uses `log_tau` and
    `threshold` directly. So this script's `log_tau` reinit is a NO-OP
    on those old ckpts -- the PLIF keys don't even exist under the new
    name. The new model loads with PLIFCell *random init* anyway, which
    is what we want post-reinit. The script is still useful when ckpt
    and code use the same naming.

If your ckpt has `A_log` not `log_tau`, just discard the script's output:
    PLIF will be random-init at the patched defaults via build_synapforge_100m,
    which is functionally equivalent to a successful reinit.
"""
import math
import sys

import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    n_reinit = 0
    for k in list(sd.keys()):
        if ".plif." not in k:
            continue
        v = sd[k]
        if k.endswith(".log_tau"):
            sd[k] = torch.full_like(v, math.log(2.5))
            n_reinit += 1
        elif k.endswith(".threshold"):
            sd[k] = torch.full_like(v, 0.05)
            n_reinit += 1
        elif k.endswith(".V"):
            sd[k] = torch.zeros_like(v)
            n_reinit += 1
    print(f"reinit {n_reinit} PLIF tensors")
    if n_reinit == 0:
        print("  WARN: no log_tau/threshold/V keys matched. ckpt may use old A_log "
              "naming; new model will random-init PLIF anyway.")
    if isinstance(ck, dict) and "model" in ck:
        ck["model"] = sd
        torch.save(ck, dst)
    else:
        torch.save(sd, dst)
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
