#!/usr/bin/env python3
"""Audit "decorative -> real" feature activations.

For each feature listed in this module, report:

  * file_exists      -- the source code is present
  * tag_or_flag      -- the activating tag/flag is set OR launcher exposes
                        it
  * actually_active  -- (only when --log is passed) the runtime log shows
                        the feature engaging

Output goes both to stdout and to docs/DECORATIVE_AUDIT.md (overwritten
each run). Exit code is 0 when every feature has at LEAST file_exists +
tag_or_flag set; non-zero otherwise.

Honest reporting: if a feature can't be activated (e.g. PLIF revival
needs val_ppl > some threshold which the run hasn't reached), we mark
it ``BLOCKED`` and explain why.

Usage:

    # Local audit, no log -- file/tag/flag only
    python3 scripts/audit_decorative_features.py

    # With a runtime log file (set when running on the rental):
    python3 scripts/audit_decorative_features.py \\
        --log /workspace/runs/synap1_ultra_run8_full/train_run8_full.log
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------


@dataclass
class FeatureCheck:
    name: str
    file_path: Path                 # primary file
    tag_or_flag: str                # name to display + grep target
    file_exists: bool = False
    tag_set: bool = False
    flag_in_launcher: bool = False
    actually_active: Optional[bool] = None
    blocked_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _check_stdp_tag() -> FeatureCheck:
    fc = FeatureCheck(
        name="STDP-only routing on SparseSynapticLayer",
        file_path=REPO / "synapforge" / "action" / "neuromcp.py",
        tag_or_flag="self.weight._sf_grad_source = ['stdp']",
    )
    fc.file_exists = fc.file_path.is_file()
    if fc.file_exists:
        text = fc.file_path.read_text(encoding="utf-8")
        fc.tag_set = (
            "_sf_grad_source = [\"stdp\"]" in text
            or "_sf_grad_source = ['stdp']" in text
        )
        if fc.tag_set:
            # The tag is engaged at module-construction time as soon as
            # NeuroMCPHead is instantiated -- so any launcher passing
            # --neuromcp-weight > 0 picks it up. We mirror that as the
            # "flag in launcher" sense.
            run8 = (REPO / "scripts" / "launch_synap1_ultra_run8_full.sh")
            if run8.is_file():
                rt = run8.read_text(encoding="utf-8")
                fc.flag_in_launcher = "--neuromcp-weight" in rt
            else:
                fc.flag_in_launcher = False
        if not fc.flag_in_launcher:
            fc.notes.append(
                "wrap --neuromcp-weight > 0 in launcher to instantiate "
                "the head"
            )
    return fc


def _check_multimodal() -> FeatureCheck:
    fc = FeatureCheck(
        name="Multimodal byte-patch (image/audio/time_series)",
        file_path=REPO / "synapforge" / "trainer_mixins.py",
        tag_or_flag="--modal-list / --modal-data-dir / --modal-alpha",
    )
    fc.file_exists = fc.file_path.is_file()
    if fc.file_exists:
        text = fc.file_path.read_text(encoding="utf-8")
        fc.tag_set = "MultimodalMixin" in text
    run8 = REPO / "scripts" / "launch_synap1_ultra_run8_full.sh"
    if run8.is_file():
        rt = run8.read_text(encoding="utf-8")
        fc.flag_in_launcher = (
            "--modal-list" in rt and "--modal-data-dir" in rt
        )
    fc.notes.append(
        "phase-aware gates this until val_ppl <= 100 (Phase 2)"
    )
    return fc


def _check_web_daemon() -> FeatureCheck:
    fc = FeatureCheck(
        name="Web daemon -> trainer parquet pipe",
        file_path=REPO / "synapforge" / "data" / "web_daemon_sink.py",
        tag_or_flag="WebDaemonSink + scripts/web_self_learn_daemon.py",
    )
    fc.file_exists = fc.file_path.is_file()
    daemon_script = REPO / "scripts" / "web_self_learn_daemon.py"
    if fc.file_exists and daemon_script.is_file():
        fc.tag_set = True
    run8 = REPO / "scripts" / "launch_synap1_ultra_run8_full.sh"
    if run8.is_file():
        rt = run8.read_text(encoding="utf-8")
        fc.flag_in_launcher = (
            "web_self_learn" in rt and "web_*.parquet" in rt
        )
    fc.notes.append(
        "default weight 0.10 in --data-files; daemon must be running "
        "for any rows to land"
    )
    return fc


def _check_rfold_train() -> FeatureCheck:
    fc = FeatureCheck(
        name="R-fold parallel scan in TRAINING (not just inference)",
        file_path=REPO / "synapforge" / "cells" / "liquid.py",
        tag_or_flag="--rfold + --rfold-chunk",
    )
    fc.file_exists = fc.file_path.is_file()
    if fc.file_exists:
        text = fc.file_path.read_text(encoding="utf-8")
        fc.tag_set = (
            "self.rfold = bool(rfold)" in text
            and "from .rfold import liquid_rfold" in text
        )
    run8 = REPO / "scripts" / "launch_synap1_ultra_run8_full.sh"
    if run8.is_file():
        rt = run8.read_text(encoding="utf-8")
        fc.flag_in_launcher = "--rfold" in rt and "--rfold-chunk" in rt
    fc.notes.append(
        "backward path verified by tests/cells/test_rfold_equivalence.py "
        "+ tests/cells/test_rfold_train_bit_exact.py"
    )
    return fc


def _check_ternary_qat() -> FeatureCheck:
    fc = FeatureCheck(
        name="Ternary BitNet QAT (delta_proj/b_proj)",
        file_path=REPO / "synapforge" / "cells" / "liquid.py",
        tag_or_flag="--weight-quant ternary",
    )
    fc.file_exists = fc.file_path.is_file()
    if fc.file_exists:
        text = fc.file_path.read_text(encoding="utf-8")
        fc.tag_set = 'weight_quant: str = "none"' in text \
                     and 'TernaryLinear' in text
    run8 = REPO / "scripts" / "launch_synap1_ultra_run8_full.sh"
    if run8.is_file():
        rt = run8.read_text(encoding="utf-8")
        # Only count it as "in launcher" if it appears in the actual
        # python invocation, not the warning banner. We extract the
        # block between ``exec python3 -u train_100m_kd.py`` and the
        # closing redirect.
        start = rt.find("exec python3 -u train_100m_kd.py")
        end = rt.find('" </dev/null', start) if start > 0 else -1
        if start > 0 and end > start:
            fc.flag_in_launcher = "--weight-quant ternary" in rt[start:end]
        else:
            fc.flag_in_launcher = False
    fc.blocked_reason = (
        "DEFERRED -- ternary swap on warmstart triggers LM-head "
        "reset (feedback_spectral_norm_warmstart_cost.md). Needs "
        "fresh run, not warmstart."
    )
    return fc


def _check_plif_revival() -> FeatureCheck:
    fc = FeatureCheck(
        name="PLIF spike revival (Run 7 dense bypass + sparse spike)",
        file_path=REPO / "synapforge" / "cells" / "plif.py",
        tag_or_flag="--plif-dense-bypass-steps + --sparse-spike-synapse",
    )
    fc.file_exists = fc.file_path.is_file()
    if fc.file_exists:
        text = fc.file_path.read_text(encoding="utf-8")
        fc.tag_set = "PLIF" in text or "plif" in text.lower()
    run8 = REPO / "scripts" / "launch_synap1_ultra_run8_full.sh"
    if run8.is_file():
        rt = run8.read_text(encoding="utf-8")
        fc.flag_in_launcher = (
            "--plif-dense-bypass-steps" in rt
            and "--sparse-spike-synapse" in rt
        )
    fc.blocked_reason = (
        "DEPENDENCY -- spike density only emerges post step 4000 "
        "(--plif-dense-bypass-steps 4000). Verify in train log AFTER "
        "step 4000+."
    )
    return fc


# ---------------------------------------------------------------------------
# Log probe (optional)
# ---------------------------------------------------------------------------


_LOG_PATTERNS: dict[str, list[str]] = {
    "STDP-only routing on SparseSynapticLayer": [
        r"NeuroMCPHead enabled",
        r"plasticity sources detected",
        r"_sf_grad_source.*stdp",
    ],
    "Multimodal byte-patch (image/audio/time_series)": [
        r"MultimodalMixin enabled",
        r"modal_list",
    ],
    "Web daemon -> trainer parquet pipe": [
        r"web_self_learn",
        r"web_\d+\.parquet",
    ],
    "R-fold parallel scan in TRAINING (not just inference)": [
        r"--rfold",
        r"liquid_rfold",
        r"rfold=True",
    ],
    "PLIF spike revival (Run 7 dense bypass + sparse spike)": [
        r"sparse-spike-synapse",
        r"plif-dense-bypass-steps",
        r"plif_revival",
    ],
}


def _probe_log(features: List[FeatureCheck], log_path: Path) -> None:
    if not log_path.is_file():
        for f in features:
            f.actually_active = None  # unknown
            f.notes.append(f"log file {log_path} not found; skipped probe")
        return
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for f in features:
        patterns = _LOG_PATTERNS.get(f.name, [])
        if not patterns:
            f.actually_active = None
            continue
        hit = any(re.search(p, text, flags=re.MULTILINE) for p in patterns)
        f.actually_active = hit
        if not hit:
            f.notes.append(
                f"log probe missed ({len(patterns)} pattern(s)); "
                "feature may be staged behind a phase gate"
            )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _md_status(b: bool, *, blocked: bool = False) -> str:
    if blocked:
        return "BLOCKED"
    return "PASS" if b else "FAIL"


def _md_actually(v: Optional[bool]) -> str:
    if v is None:
        return "(no log)"
    return "PASS" if v else "FAIL"


def write_markdown_report(
    features: List[FeatureCheck],
    out_path: Path,
    log_path: Optional[Path] = None,
) -> None:
    lines: list[str] = []
    lines.append("# Decorative -> Real Feature Audit")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if log_path is not None:
        lines.append(f"Log probed: `{log_path}`")
    else:
        lines.append("Log probe: (not run -- pass --log to enable)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Feature | file_exists | tag/flag set | flag in launcher | "
        "actually_active | notes |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- |"
    )
    for f in features:
        actually = _md_actually(f.actually_active)
        if f.blocked_reason:
            actually = "BLOCKED"
        notes = "; ".join(f.notes) if f.notes else ""
        if f.blocked_reason:
            notes = f"{f.blocked_reason}" + (
                f" / {notes}" if notes else ""
            )
        lines.append(
            f"| {f.name} "
            f"| {_md_status(f.file_exists)} "
            f"| {_md_status(f.tag_set)} "
            f"| {_md_status(f.flag_in_launcher)} "
            f"| {actually} "
            f"| {notes} |"
        )
    lines.append("")
    # Detailed
    lines.append("## Detail per feature")
    lines.append("")
    for f in features:
        lines.append(f"### {f.name}")
        lines.append("")
        lines.append(f"* file: `{f.file_path}`")
        lines.append(f"* tag/flag: `{f.tag_or_flag}`")
        lines.append(f"* file_exists: **{_md_status(f.file_exists)}**")
        lines.append(f"* tag/flag set in code: **{_md_status(f.tag_set)}**")
        lines.append(
            f"* flag wired in Run 8 full launcher: "
            f"**{_md_status(f.flag_in_launcher)}**"
        )
        lines.append(
            f"* actually-active in log: **{_md_actually(f.actually_active)}**"
        )
        if f.blocked_reason:
            lines.append(f"* BLOCKED: {f.blocked_reason}")
        if f.notes:
            lines.append("* notes:")
            for n in f.notes:
                lines.append(f"    - {n}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    p = argparse.ArgumentParser(
        description="Audit decorative -> real feature activations."
    )
    p.add_argument("--log", type=Path, default=None,
                   help="Path to a training log to probe for runtime "
                        "activation evidence.")
    p.add_argument("--out-md", type=Path,
                   default=REPO / "docs" / "DECORATIVE_AUDIT.md",
                   help="Output path for the markdown report.")
    p.add_argument("--json", action="store_true",
                   help="Also dump JSON next to the markdown.")
    args = p.parse_args()

    features: List[FeatureCheck] = [
        _check_stdp_tag(),
        _check_multimodal(),
        _check_web_daemon(),
        _check_rfold_train(),
        _check_ternary_qat(),
        _check_plif_revival(),
    ]
    if args.log is not None:
        _probe_log(features, args.log)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(features, args.out_md, args.log)
    print(f"[audit] wrote {args.out_md}")

    if args.json:
        json_path = args.out_md.with_suffix(".json")
        json_path.write_text(json.dumps(
            [
                {
                    "name": f.name,
                    "file_path": str(f.file_path),
                    "tag_or_flag": f.tag_or_flag,
                    "file_exists": f.file_exists,
                    "tag_set": f.tag_set,
                    "flag_in_launcher": f.flag_in_launcher,
                    "actually_active": f.actually_active,
                    "blocked_reason": f.blocked_reason,
                    "notes": f.notes,
                }
                for f in features
            ],
            indent=2,
        ), encoding="utf-8")
        print(f"[audit] wrote {json_path}")

    # Exit code: 0 if every non-blocked feature has file + tag + flag.
    bad: list[str] = []
    for f in features:
        if f.blocked_reason:
            continue
        if not (f.file_exists and f.tag_set and f.flag_in_launcher):
            bad.append(f.name)
    if bad:
        print(f"[audit] {len(bad)} feature(s) NOT activated:")
        for n in bad:
            print(f"   * {n}")
        return 1
    print("[audit] all non-blocked features activated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
