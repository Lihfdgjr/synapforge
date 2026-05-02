"""audit_rental_data.py — survey parquet/jsonl/txt corpora on a remote host.

D1 of the 2026-05-02 quality-data push (TOKEN_SOUP root cause hunt).

The trainer has been replaying ONE parquet file (``files=1``) for 21000+
steps and the model has degenerated into "the the the..." word-salad
output. PLIF-dead is part of it; the OTHER part we now suspect is the
data itself — a single shard of likely raw-web crawl.

Before we can fix the corpus we need to know what the project already
has on disk. Two hosts of interest:

  * **Rental** (117.74.66.77:41614) — the H100/A800 training box. As of
    2026-05-01 it's mid-relocation and **offline**; the script falls
    through gracefully when SSH fails.
  * **Mohuanfang** (mohuanfang.com via ``~/.ssh/config`` host
    ``myserver``) — 1.5 TB private warehouse, KNOWN to hold
    ``/home/liu/lnn-train/data/`` with FineWeb-Edu, Wiki-zh, WikiText-103,
    cosmopedia, and a partial bitnet baseline.

For each *.parquet / *.jsonl / *.txt under the configured roots we
record:

    {
      "path": "/home/liu/lnn-train/data/fineweb_edu/000_00000.parquet",
      "size_bytes": 2_152_819_114,
      "kind": "parquet",
      "rows": 726000,
      "schema": [["text","string"], ["score","double"], ...],
      "text_column": "text",
      "samples": ["The Independent Jane...", "...", "..."],
      "non_empty_ratio": 0.92,
      "host": "myserver",
      "root": "/home/liu/lnn-train/data",
    }

Output: one JSON manifest at ``docs/RENTAL_DATA_AUDIT.json`` (the user
asked for that exact path even though the rental itself is offline; the
file's purpose is "what the project has, anywhere").

Usage:

    python scripts/audit_rental_data.py
    python scripts/audit_rental_data.py --hosts myserver --roots /home/liu/lnn-train/data
    python scripts/audit_rental_data.py --include-rental    # try the rental too

Implementation note: we shell out to ``ssh`` rather than depending on
paramiko (the project already uses ssh-config aliases everywhere; see
``scripts/sync_to_mohuanfang.sh``). Remote inspection runs a small
inline Python snippet via the ``lnn-train`` venv (which has pyarrow);
local fallback uses pyarrow when the host is the local machine.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Defaults — keep aligned with rental + mohuanfang reality 2026-05-02.
# ---------------------------------------------------------------------------
DEFAULT_HOSTS: list[dict[str, Any]] = [
    {
        "name": "myserver",  # ~/.ssh/config alias for mohuanfang.com
        "ssh": "myserver",
        "roots": [
            "/home/liu/lnn-train/data",
            "/home/liu/synapforge_backup",  # only sub-dirs likely small
        ],
        # Pre-installed venv with pyarrow:
        "venv_python": "/home/liu/lnn-train/.venv/bin/python3",
    },
    {
        # Rental box. As of 2026-05-01 it's mid-relocation and SSH times
        # out; we mark it as opt-in via --include-rental so the default
        # audit doesn't burn a 30-second timeout when the box is dead.
        "name": "rental",
        "ssh": "-o ConnectTimeout=8 -o BatchMode=yes -p 41614 root@117.74.66.77",
        "roots": [
            "/workspace/data",
            "/workspace/datasets",
            "/workspace/teachers",
            "/workspace/runs",
        ],
        "venv_python": "/usr/bin/python3",
        "opt_in": True,
    },
]

# Output JSON manifest path (relative to repo root).
DEFAULT_OUT = "docs/RENTAL_DATA_AUDIT.json"


# Inline remote python script. Streamed in via ssh's stdin so we don't
# need to upload anything. Reads up to 3 sample texts per parquet.
REMOTE_INSPECT_PY = r"""
import json, os, sys
try:
    import pyarrow.parquet as pq
    HAVE_ARROW = True
except Exception as e:
    HAVE_ARROW = False
    PA_ERR = repr(e)

def inspect_parquet(path):
    out = {"path": path, "kind": "parquet", "size_bytes": os.path.getsize(path)}
    if not HAVE_ARROW:
        out["error"] = "pyarrow not available: " + PA_ERR
        return out
    try:
        pf = pq.ParquetFile(path)
        out["rows"] = int(pf.metadata.num_rows)
        out["schema"] = [(f.name, str(f.type)) for f in pf.schema_arrow]
        # Find text column
        text_col = None
        for f in pf.schema_arrow:
            if str(f.type) == "string" and f.name in ("text","content","raw_content","document","code"):
                text_col = f.name
                break
        if text_col is None:
            for f in pf.schema_arrow:
                if str(f.type) == "string":
                    text_col = f.name
                    break
        out["text_column"] = text_col
        # Sample first ~200 rows for non-empty ratio + 3 samples
        if text_col is not None:
            n = 0
            n_empty = 0
            samples = []
            for batch in pf.iter_batches(batch_size=200, columns=[text_col]):
                vs = batch.column(text_col).to_pylist()
                for v in vs:
                    n += 1
                    if not v or not str(v).strip():
                        n_empty += 1
                    elif len(samples) < 3:
                        samples.append(str(v)[:280])
                break
            out["non_empty_ratio"] = round((n - n_empty) / max(1, n), 3)
            out["samples"] = samples
        else:
            out["non_empty_ratio"] = 0.0
            out["samples"] = []
    except Exception as e:
        out["error"] = repr(e)[:200]
    return out

def inspect_jsonl(path):
    out = {"path": path, "kind": "jsonl", "size_bytes": os.path.getsize(path)}
    n = 0
    samples = []
    try:
        with open(path, "rb") as f:
            for line in f:
                n += 1
                if len(samples) < 3:
                    try:
                        samples.append(line.decode("utf-8", errors="ignore")[:280])
                    except Exception:
                        pass
                if n >= 1000:
                    # don't count whole file beyond 1000 -- mark approx
                    out["row_estimate"] = "1000+ (sampled)"
                    break
        out.setdefault("rows", n)
        out["samples"] = samples
    except Exception as e:
        out["error"] = repr(e)[:200]
    return out

def inspect_txt(path):
    out = {"path": path, "kind": "txt", "size_bytes": os.path.getsize(path)}
    samples = []
    try:
        with open(path, "rb") as f:
            chunk = f.read(2048)
        out["samples"] = [chunk.decode("utf-8", errors="ignore")[:280]]
    except Exception as e:
        out["error"] = repr(e)[:200]
    return out

def walk_root(root):
    entries = []
    if not os.path.exists(root):
        return entries
    for dirpath, dirs, files in os.walk(root):
        # Skip noisy caches inside data dirs
        for d in list(dirs):
            if d in (".git","__pycache__","node_modules",".cache"):
                dirs.remove(d)
        for fname in files:
            full = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size < 256:
                continue  # skip empty / tiny placeholder files
            low = fname.lower()
            if low.endswith(".parquet"):
                entries.append(inspect_parquet(full))
            elif low.endswith((".jsonl", ".jsonl.gz")):
                entries.append(inspect_jsonl(full))
            elif low.endswith((".txt", ".text")) and size < 200_000_000:
                entries.append(inspect_txt(full))
    return entries

if __name__ == "__main__":
    roots = sys.argv[1:]
    out = {"roots": roots, "entries": []}
    for r in roots:
        out["entries"].extend(walk_root(r))
    print("__JSON_BEGIN__")
    print(json.dumps(out, ensure_ascii=False))
    print("__JSON_END__")
"""


def _run_ssh(host_cfg: dict[str, Any], roots: Sequence[str]) -> dict[str, Any]:
    """Stream the inspect script to the remote host and parse the JSON.

    Returns ``{"ok": False, "error": str}`` on connection / parse failures
    (a host being down is a soft failure — we still emit the rest of the
    manifest).
    """
    ssh_cmd = host_cfg["ssh"]
    # Build the remote command: cat | python3 SCRIPT --roots ...
    py = host_cfg.get("venv_python", "python3")
    # Quote roots safely
    roots_args = " ".join(f"'{r}'" for r in roots)
    remote = f"{py} - {roots_args}"
    # ssh + -- arg separation. ssh_cmd may contain options ("-p 41614 root@..."),
    # so we shlex-split it.
    import shlex
    parts = shlex.split(ssh_cmd) if ssh_cmd != host_cfg["name"] else [ssh_cmd]
    cmd = ["ssh"] + parts + [remote] if parts and parts != [host_cfg["name"]] else \
          ["ssh", host_cfg["ssh"], remote]
    # Simpler: always invoke as "ssh <ssh_target_string> <remote>"
    if ssh_cmd == host_cfg["name"]:
        cmd = ["ssh", ssh_cmd, remote]
    else:
        # split because ssh_cmd contains -p/-o flags
        cmd = ["ssh"] + parts + [remote]

    try:
        proc = subprocess.run(
            cmd,
            input=REMOTE_INSPECT_PY,
            capture_output=True,
            # Pin encoding to utf-8 -- the default on Windows is cp936/GBK
            # which crashes on Chinese / emoji text in Wikipedia samples.
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ssh timeout"}
    except FileNotFoundError:
        return {"ok": False, "error": "ssh binary not found"}
    if proc.returncode != 0:
        # Log the first non-empty stderr line for diagnosis.
        err_head = (proc.stderr or "").splitlines()[:6]
        return {"ok": False, "error": "ssh failed: " + " | ".join(err_head)}

    out = proc.stdout
    # Find delimiter
    if "__JSON_BEGIN__" not in out or "__JSON_END__" not in out:
        return {"ok": False, "error": "delimiter missing in remote output",
                "raw_tail": out[-400:]}
    body = out.split("__JSON_BEGIN__", 1)[1].split("__JSON_END__", 1)[0].strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"json decode: {e}", "raw_head": body[:400]}
    parsed["ok"] = True
    return parsed


def _run_local(roots: Sequence[str]) -> dict[str, Any]:
    """Local-host fallback: inspect roots on the current machine via pyarrow.

    Used when ``--hosts local`` is passed. Mirrors ``REMOTE_INSPECT_PY``
    behaviour, just executed in-process.
    """
    g: dict[str, Any] = {}
    exec(REMOTE_INSPECT_PY, g, g)  # noqa: S102 (trusted constant)
    walk = g["walk_root"]
    out: dict[str, Any] = {"roots": list(roots), "entries": []}
    for r in roots:
        out["entries"].extend(walk(r))
    out["ok"] = True
    return out


def _summarise(host_name: str, host_result: dict[str, Any]) -> dict[str, Any]:
    """Compact human-readable summary of one host's findings."""
    if not host_result.get("ok"):
        return {"host": host_name, "ok": False, "error": host_result.get("error", "?")}
    entries = host_result.get("entries", [])
    by_kind: dict[str, int] = {}
    total_size = 0
    parquet_rows = 0
    quality_files: list[dict[str, Any]] = []
    for e in entries:
        k = e.get("kind", "?")
        by_kind[k] = by_kind.get(k, 0) + 1
        total_size += int(e.get("size_bytes", 0))
        if k == "parquet":
            parquet_rows += int(e.get("rows", 0) or 0)
            # Heuristic "quality candidate": has text col + >1k rows + >50% non-empty.
            if (e.get("text_column") and (e.get("rows") or 0) > 1000
                    and float(e.get("non_empty_ratio", 0)) > 0.5):
                quality_files.append({
                    "path": e["path"],
                    "rows": e.get("rows"),
                    "size_mb": round(e["size_bytes"] / 1e6, 1),
                    "non_empty": e.get("non_empty_ratio"),
                    "text_column": e.get("text_column"),
                })
    return {
        "host": host_name,
        "ok": True,
        "n_files": len(entries),
        "by_kind": by_kind,
        "total_size_gb": round(total_size / 1e9, 2),
        "parquet_total_rows": parquet_rows,
        "n_quality_candidates": len(quality_files),
        "quality_candidates_top10": sorted(
            quality_files, key=lambda x: x["rows"], reverse=True
        )[:10],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--hosts",
        default="myserver",
        help="Comma-separated host names from DEFAULT_HOSTS. "
             "Use 'local' to scan the current machine. "
             "Default: 'myserver' (mohuanfang.com).",
    )
    ap.add_argument(
        "--roots",
        default=None,
        help="Override host roots (comma-separated). Applies to ALL --hosts.",
    )
    ap.add_argument(
        "--include-rental",
        action="store_true",
        help="Include the rental host even if it's marked opt-in (offline). "
             "Will still soft-fail with timeout if the box is down.",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="JSON manifest output path.",
    )
    args = ap.parse_args(argv)

    requested = [h.strip() for h in args.hosts.split(",") if h.strip()]
    host_cfgs: list[dict[str, Any]] = []
    for r in requested:
        if r == "local":
            host_cfgs.append({"name": "local", "ssh": None, "roots": ["."]})
            continue
        match = next((h for h in DEFAULT_HOSTS if h["name"] == r), None)
        if match is None:
            print(f"[audit] unknown host {r!r}; skipping", file=sys.stderr)
            continue
        if match.get("opt_in") and not args.include_rental and r == "rental":
            print(f"[audit] {r}: opt-in only, pass --include-rental to scan",
                  file=sys.stderr)
            continue
        host_cfgs.append(match)

    override_roots: list[str] | None = None
    if args.roots:
        override_roots = [r.strip() for r in args.roots.split(",") if r.strip()]

    manifest: dict[str, Any] = {
        "kind": "rental_data_audit",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hosts": [],
        "summary": [],
    }

    for cfg in host_cfgs:
        roots = override_roots if override_roots is not None else cfg["roots"]
        print(f"[audit] {cfg['name']}: scanning {roots} ...", file=sys.stderr)
        t0 = time.time()
        if cfg["name"] == "local":
            result = _run_local(roots)
        else:
            result = _run_ssh(cfg, roots)
        dt = time.time() - t0
        print(f"[audit] {cfg['name']}: done in {dt:.1f}s ok={result.get('ok')}",
              file=sys.stderr)
        result["host"] = cfg["name"]
        result["scan_seconds"] = round(dt, 1)
        manifest["hosts"].append(result)
        manifest["summary"].append(_summarise(cfg["name"], result))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Console summary
    print()
    print("=" * 70)
    for s in manifest["summary"]:
        if not s.get("ok"):
            print(f"[{s['host']:10s}] FAIL: {s.get('error', '?')}")
            continue
        print(f"[{s['host']:10s}] {s['n_files']:>4} files, "
              f"{s['total_size_gb']:>6.2f} GB, "
              f"by_kind={s['by_kind']}, "
              f"parquet_rows={s['parquet_total_rows']:,}, "
              f"quality_candidates={s['n_quality_candidates']}")
        for q in s["quality_candidates_top10"][:5]:
            print(f"    {q['path']}  rows={q['rows']:,} size={q['size_mb']:.0f}MB "
                  f"non_empty={q['non_empty']}")
    print("=" * 70)
    print(f"[audit] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
