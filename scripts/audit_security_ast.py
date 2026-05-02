"""AST-based security scan — bare-bones substitute when bandit/ruff are
unavailable. Used by docs/SECURITY_AUDIT_2026-05-02.md.

Usage:
    python scripts/audit_security_ast.py
    python scripts/audit_security_ast.py synapforge/ scripts/
"""
from __future__ import annotations

import ast
import os
import sys
from collections import defaultdict
from typing import List, Tuple

issues: List[Tuple[str, str, str, int, str]] = []  # severity, category, path, line, msg


class Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path

    def _name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
            return parts[0] if parts else ""
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        name = self._name(node)
        is_bare = isinstance(node.func, ast.Name)

        if is_bare and name in ("eval", "exec"):
            issues.append(("HIGH", "unsafe-exec", self.path, node.lineno,
                           f"use of {name}()"))
        if name == "pickle.loads":
            issues.append(("MED", "pickle-loads", self.path, node.lineno,
                           "pickle.loads on untrusted input is RCE"))
        if name == "pickle.load":
            issues.append(("MED", "pickle-load", self.path, node.lineno,
                           "pickle.load on untrusted file is RCE"))
        if name == "os.system":
            issues.append(("HIGH", "os-system", self.path, node.lineno,
                           "os.system shell injection"))
        if name in ("subprocess.run", "subprocess.Popen", "subprocess.call",
                   "subprocess.check_output"):
            for kw in node.keywords:
                if (kw.arg == "shell" and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    issues.append(("HIGH", "shell-true", self.path, node.lineno,
                                   f"{name}(shell=True)"))
        if name == "urllib.request.urlopen":
            has_timeout = any(kw.arg == "timeout" for kw in node.keywords)
            if not has_timeout:
                issues.append(("LOW", "no-timeout", self.path, node.lineno,
                               "urlopen without timeout"))
        if name == "yaml.load":
            unsafe = True
            for kw in node.keywords:
                if kw.arg == "Loader" and isinstance(kw.value, ast.Attribute):
                    if "Safe" in (kw.value.attr or ""):
                        unsafe = False
            if unsafe:
                issues.append(("MED", "yaml-load", self.path, node.lineno,
                               "yaml.load — use safe_load or Loader=SafeLoader"))
        if name == "torch.load":
            wo = None
            for kw in node.keywords:
                if kw.arg == "weights_only" and isinstance(kw.value, ast.Constant):
                    wo = kw.value.value
            if wo is False:
                issues.append(("LOW", "torch-load-pickle", self.path, node.lineno,
                               "torch.load(weights_only=False) — pickle RCE on attacker ckpt"))
        if name == "hashlib.md5":
            issues.append(("LOW", "weak-hash", self.path, node.lineno,
                           "hashlib.md5 — weak (OK for dedup keys, NOT for security)"))
        if name == "hashlib.sha1":
            issues.append(("LOW", "weak-hash", self.path, node.lineno,
                           "hashlib.sha1 — weak"))
        if name == "tempfile.mktemp":
            issues.append(("MED", "mktemp-race", self.path, node.lineno,
                           "tempfile.mktemp — TOCTOU race (use NamedTemporaryFile)"))

        self.generic_visit(node)


def scan_root(root: str) -> int:
    n = 0
    for dp, dn, fn in os.walk(root):
        if "__pycache__" in dp:
            continue
        for f in fn:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f).replace(os.sep, "/")
            try:
                src = open(p, "r", encoding="utf-8").read()
                tree = ast.parse(src, filename=p)
            except SyntaxError as e:
                issues.append(("HIGH", "syntax-error", p,
                               getattr(e, "lineno", 0) or 0,
                               f"parse failed: {e.msg}"))
                continue
            except Exception as e:
                issues.append(("HIGH", "parse-error", p, 0,
                               f"parse failed: {e!r}"))
                continue
            n += 1
            Visitor(p).visit(tree)
    return n


def main(argv: List[str]) -> int:
    if argv:
        roots = argv
    else:
        # Default: scan synapforge/ and scripts/ relative to repo root.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        roots = [
            os.path.join(repo_root, "synapforge"),
            os.path.join(repo_root, "scripts"),
        ]
    total = 0
    for r in roots:
        total += scan_root(r)
    print(f"Files scanned: {total}")
    print(f"Total findings: {len(issues)}")
    sev = defaultdict(int)
    cat = defaultdict(int)
    for s, c, _, _, _ in issues:
        sev[s] += 1
        cat[c] += 1
    print()
    print("By severity:")
    for s in ("HIGH", "MED", "LOW", "INFO"):
        print(f"  {s}: {sev[s]}")
    print()
    print("By category:")
    for c, n in sorted(cat.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    for level in ("HIGH", "MED", "LOW"):
        sub = [i for i in issues if i[0] == level]
        if not sub:
            continue
        print()
        print(f"{level} severity:")
        for _, c, p, ln, msg in sorted(sub):
            print(f"  [{c}] {p}:{ln}: {msg}")
    return 0 if sev["HIGH"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
