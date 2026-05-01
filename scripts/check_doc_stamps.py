"""Doc-stamp checker -- flag stale docs without rewriting them.

This is **instrumentation, not content cleanup**. It answers:
"has any code referenced by this doc changed since the doc was last verified?"

Usage::

    python scripts/check_doc_stamps.py          # report only, exit 0/1
    python scripts/check_doc_stamps.py --update-stamps  # bump auto-fresh stamps

A doc is one of:
- ``fresh``       -- file unchanged since stamped sha AND no referenced code
                     touched after the stamp date.
- ``auto-fresh``  -- file itself has been edited since the stamp; we trust the
                     edit and bump the stamp (only when ``--update-stamps``).
- ``MAYBE STALE`` -- a referenced code file was modified after
                     ``last_verified_date``. Human must verify content.
- ``STALE``       -- referenced code file no longer exists in the repo.

Refs are detected by simple regex over the doc text:

- backtick-quoted file paths ``\\`scripts/foo.py\\```
- ``synapforge/foo/bar.py`` style paths
- explicit ``synapforge/foo.py:42`` line refs
- ``:func:\\`bar\\``` Sphinx role
- markdown links ``[label](path/to/file.py)``

Resolves P10 in docs/MASTER_PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
STAMP_FILE = DOCS_DIR / "_stamp.json"

# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

# Match `path/to/file.ext` inside backticks. Allow .py, .sh, .md, .json, .toml.
_BACKTICK_PATH = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:py|sh|md|json|toml|yml|yaml|html|cfg))(?::\d+)?`"
)
# Match `synapforge/foo/bar.py` style outside backticks too.
_BARE_PATH = re.compile(
    r"\b((?:synapforge|scripts|tests|docs)/[A-Za-z0-9_./-]+\.(?:py|sh|md|json|toml|yml|yaml))\b"
)
# Sphinx :func:`name` -- bookkeeping only; we can't resolve these to files
# without an import, so we record the symbol but won't flag based on it.
_SPHINX_FUNC = re.compile(r":func:`([A-Za-z0-9_.]+)`")
# Markdown link with file path target.
_MD_LINK = re.compile(
    r"\]\(([A-Za-z0-9_./-]+\.(?:py|sh|md|json|toml|yml|yaml))\)"
)


def extract_refs(doc_path: Path) -> set[str]:
    """Return the set of repo-relative file paths referenced inside ``doc_path``.

    Only paths that resolve to existing files in the repo are returned;
    paths that don't resolve are tracked separately via ``extract_dead_refs``.
    """
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    candidates: set[str] = set()
    for rx in (_BACKTICK_PATH, _BARE_PATH, _MD_LINK):
        for m in rx.finditer(text):
            candidates.add(m.group(1))
    # Filter to existing files inside repo. Strip line-number suffix already
    # handled by the regex.
    refs: set[str] = set()
    for c in candidates:
        # Skip self-references and obvious URL fragments.
        if c.startswith("http://") or c.startswith("https://"):
            continue
        rel = c.lstrip("./")
        target = (REPO_ROOT / rel).resolve()
        try:
            target.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if target.exists() and target.is_file():
            refs.add(rel.replace("\\", "/"))
    return refs


def extract_dead_refs(doc_path: Path) -> set[str]:
    """Return paths the doc claims to reference but that no longer exist."""
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    candidates: set[str] = set()
    for rx in (_BACKTICK_PATH, _BARE_PATH, _MD_LINK):
        for m in rx.finditer(text):
            candidates.add(m.group(1))
    dead: set[str] = set()
    for c in candidates:
        if c.startswith("http://") or c.startswith("https://"):
            continue
        # Heuristic: only treat as "claimed code ref" if it looks like a real
        # source file under the repo (synapforge/ scripts/ tests/) and ends
        # with .py or .sh. Doc-to-doc links missing aren't a "STALE" signal.
        rel = c.lstrip("./").replace("\\", "/")
        if not rel.endswith((".py", ".sh")):
            continue
        if not rel.startswith(("synapforge/", "scripts/", "tests/")):
            continue
        target = (REPO_ROOT / rel).resolve()
        try:
            target.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if not target.exists():
            dead.add(rel)
    return dead


# ---------------------------------------------------------------------------
# Git helpers (subprocess shell-out -- pure Python on the caller side)
# ---------------------------------------------------------------------------


def _git(args: list[str]) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def head_sha() -> str:
    return _git(["rev-parse", "HEAD"])


def last_commit_sha_for(rel_path: str) -> str:
    """Return the short-or-full sha of the last commit that touched ``rel_path``.

    Empty string if the file isn't tracked.
    """
    return _git(["log", "--format=%H", "-1", "--", rel_path])


def last_commit_date_for(rel_path: str) -> str | None:
    """Return ISO date (YYYY-MM-DD) of the last commit touching ``rel_path``.

    None if the file isn't tracked.
    """
    out = _git(["log", "--format=%cs", "-1", "--", rel_path])
    return out or None


# ---------------------------------------------------------------------------
# Stamp file io
# ---------------------------------------------------------------------------


def load_stamps() -> dict[str, dict]:
    if not STAMP_FILE.exists():
        return {}
    return json.loads(STAMP_FILE.read_text(encoding="utf-8"))


def save_stamps(stamps: dict[str, dict]) -> None:
    payload = json.dumps(stamps, indent=2, sort_keys=True) + "\n"
    STAMP_FILE.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Status calculation
# ---------------------------------------------------------------------------


def _date_after(iso_a: str, iso_b: str) -> bool:
    """Return True if iso_a > iso_b (both YYYY-MM-DD)."""
    try:
        a = datetime.strptime(iso_a, "%Y-%m-%d")
        b = datetime.strptime(iso_b, "%Y-%m-%d")
    except ValueError:
        return False
    return a > b


def classify(
    doc_rel: str,
    stamp: dict,
) -> tuple[str, str]:
    """Return (status, reason).

    Status ∈ {"fresh", "auto-fresh", "MAYBE STALE", "STALE"}.
    """
    doc_abs = REPO_ROOT / doc_rel
    if not doc_abs.exists():
        return "STALE", f"doc file missing: {doc_rel}"

    # 1) Any referenced code path that no longer exists -> STALE.
    #    Checked BEFORE auto-fresh: a doc that references deleted files
    #    is content-outdated regardless of how recently it was edited.
    dead = extract_dead_refs(doc_abs)
    if dead:
        joined = ", ".join(sorted(dead))
        return "STALE", f"refs missing files: {joined}"

    # 2) Has the doc itself been modified since the stamp sha?
    last_sha = last_commit_sha_for(doc_rel)
    stamped_sha = stamp.get("last_verified_sha", "")
    if last_sha and stamped_sha and last_sha != stamped_sha:
        return "auto-fresh", (
            f"doc edited since stamp ({stamped_sha[:8]} -> {last_sha[:8]})"
        )

    # 3) Any referenced code modified after stamp date -> MAYBE STALE.
    refs = extract_refs(doc_abs)
    stamped_date = stamp.get("last_verified_date", "1970-01-01")
    bumped: list[str] = []
    for rel in sorted(refs):
        last_date = last_commit_date_for(rel)
        if last_date and _date_after(last_date, stamped_date):
            bumped.append(f"{rel}({last_date})")
    if bumped:
        # Cap reason length so the markdown table stays readable.
        head = ", ".join(bumped[:3])
        more = f" +{len(bumped) - 3} more" if len(bumped) > 3 else ""
        return "MAYBE STALE", f"refs changed after {stamped_date}: {head}{more}"

    return "fresh", "all refs unchanged since stamp"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def collect_docs() -> list[str]:
    """Return sorted repo-relative paths of `docs/*.md` (excluding _stamp.json)."""
    return sorted(
        f"docs/{p.name}" for p in DOCS_DIR.glob("*.md") if p.is_file()
    )


def render_table(rows: Iterable[tuple[str, str, str]]) -> str:
    """Render the report as a 3-column GitHub-flavored markdown table."""
    rows = list(rows)
    out = ["| doc | status | why |", "|-----|--------|-----|"]
    for doc, status, reason in rows:
        # Pipe in reason would break the table; replace.
        safe_reason = reason.replace("|", "\\|")
        out.append(f"| {doc} | {status} | {safe_reason} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-stamps",
        action="store_true",
        help=(
            "Bump 'last_verified_sha' for docs whose status is auto-fresh. "
            "Never auto-flips MAYBE STALE -> fresh."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the markdown table.",
    )
    args = parser.parse_args(argv)

    stamps = load_stamps()
    docs = collect_docs()
    head = head_sha()

    rows: list[tuple[str, str, str]] = []
    any_stale = False

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for doc in docs:
        stamp = stamps.get(doc)
        if stamp is None:
            # Missing stamp = newly added doc. With --update-stamps,
            # initialize at the doc's own last-touched sha (so the
            # stamp tracks the file, not HEAD; matches classify() logic).
            if args.update_stamps:
                init_sha = last_commit_sha_for(doc) or head
                stamps[doc] = {
                    "last_verified_sha": init_sha,
                    "last_verified_date": today,
                    "verifier": "agent",
                }
                rows.append((doc, "auto-fresh", "stamp initialized (was missing)"))
            else:
                rows.append(
                    (doc, "MAYBE STALE", "no stamp -- run --update-stamps")
                )
            continue
        status, reason = classify(doc, stamp)
        rows.append((doc, status, reason))
        if status == "STALE":
            any_stale = True
        if args.update_stamps and status == "auto-fresh":
            # Pin to the doc's own last-touched sha so subsequent runs
            # report 'fresh' rather than re-firing 'auto-fresh' against HEAD.
            doc_sha = last_commit_sha_for(doc) or head
            stamps[doc] = {
                "last_verified_sha": doc_sha,
                "last_verified_date": today,
                "verifier": stamp.get("verifier", "agent"),
            }

    if args.update_stamps:
        save_stamps(stamps)

    if args.json:
        payload = [{"doc": d, "status": s, "reason": r} for d, s, r in rows]
        print(json.dumps(payload, indent=2))
    else:
        print(render_table(rows))

    return 1 if any_stale else 0


if __name__ == "__main__":
    sys.exit(main())
