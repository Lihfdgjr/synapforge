"""Round 7 smoke test — verify every new mscfc port surfaces correctly.

Tries the 9 symbols requested by the Round-7 task brief.  On failure
prints the error and continues; never aborts on first fail.  Prints
``round7-imports: N/9 OK`` at the end so the harness can grep one line.

Usage::

    /opt/conda/bin/python /workspace/synapforge/test_round7_imports.py

Exits 0 if all 9 imports succeed, 1 otherwise.  Designed to be safe to
run as part of CI or as a one-shot detached job.
"""

from __future__ import annotations

import sys
import traceback


def _try(name: str, getter):
    try:
        obj = getter()
        return True, repr(obj)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    sys.path.insert(0, "/workspace")
    try:
        import synapforge as sf  # noqa: F401
    except Exception:
        print("round7-imports: 0/9 OK -- top-level synapforge import failed:")
        traceback.print_exc()
        return 1

    checks = [
        ("sf.WaveFormer1D",            lambda: sf.WaveFormer1D),
        ("sf.WorldModelHead",          lambda: sf.WorldModelHead),
        ("sf.LatentLoopController",    lambda: sf.LatentLoopController),
        ("sf.NoveltyDrive",            lambda: sf.NoveltyDrive),
        ("sf.HierarchicalMemory",      lambda: sf.HierarchicalMemory),
        ("sf.routers.ChainOfExpertsMoE", lambda: sf.routers.ChainOfExpertsMoE),
        ("sf.routers.EnhancedLoopStack", lambda: sf.routers.EnhancedLoopStack),
        ("sf.bio.MultiBandTau",        lambda: sf.bio.MultiBandTau),
        ("sf.bio.AstrocyteGate",       lambda: sf.bio.AstrocyteGate),
    ]
    n_ok = 0
    for name, fn in checks:
        ok, info = _try(name, fn)
        flag = "OK  " if ok else "FAIL"
        print(f"  {flag}  {name:36s} -> {info}")
        if ok:
            n_ok += 1
    total = len(checks)
    print(f"round7-imports: {n_ok}/{total} OK")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
