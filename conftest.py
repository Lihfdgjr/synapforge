# Repo-root conftest: keep pytest from collecting in-package smoke scripts.
#
# `synapforge/test_*.py` are internal smoke scripts (require torch / GPU /
# rental box) — not part of the unit test suite. They are still runnable
# manually via `python -m synapforge.test_xyz`. The canonical unit tests
# live in `tests/` and `synapforge/tests/`.
#
# Resolves P15 in docs/MASTER_PLAN.md.
collect_ignore_glob = ["synapforge/test_*.py"]
