"""T9.6 -- tests for ``scripts/eval_cross_domain_ppl.py``.

Four required tests (per queue spec):

1. ``test_smoke_writes_json``         -- ``--smoke`` writes a parseable JSON
   with the documented top-level keys.
2. ``test_per_domain_ppl_emitted``    -- the JSON's ``domains`` sub-dict has
   one entry per requested domain, each with ``ppl`` + ``status: "ok"``.
3. ``test_weighted_average_correct``  -- the cross-domain weighted average
   matches the formula `sum(w_i * ppl_i) / sum(w_i)`.
4. ``test_handles_oom_per_domain``    -- a per-domain OOM raised inside the
   model's ``perplexity`` call gets swallowed for that domain only; the
   remaining domains still report status="ok".

All tests run on CPU, no HF datasets, no torch import. Total runtime <1s.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# Make repo + scripts importable so ``import eval_cross_domain_ppl`` works
# regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.fixture
def eval_module():
    if "eval_cross_domain_ppl" in sys.modules:
        return importlib.reload(sys.modules["eval_cross_domain_ppl"])
    return importlib.import_module("eval_cross_domain_ppl")


# ---------------------------------------------------------------------------
# 1. --smoke writes a JSON we can parse with the documented top-level keys.
# ---------------------------------------------------------------------------
def test_smoke_writes_json(tmp_path, eval_module):
    """`main(--smoke)` writes a parseable JSON with the headline keys."""
    out_json = tmp_path / "eval.json"
    rc = eval_module.main([
        "--smoke",
        "--out", str(out_json),
        "--domains", "wikitext,c4",
        "--max-tokens-per-domain", "1024",
    ])
    assert rc == 0, f"expected exit 0; got {rc}"
    assert out_json.exists(), f"expected JSON at {out_json}"

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    # Required top-level keys.
    for k in ("ckpt", "ts", "elapsed_s", "smoke", "domains",
              "weighted_avg_ppl", "n_domains_evaluated",
              "n_domains_skipped"):
        assert k in payload, f"missing top-level key: {k}"
    assert payload["smoke"] is True
    assert payload["ckpt"] == "<smoke>"
    # `domains` is a dict keyed by name.
    assert isinstance(payload["domains"], dict)
    assert set(payload["domains"].keys()) == {"wikitext", "c4"}


# ---------------------------------------------------------------------------
# 2. Per-domain ppl emitted with status="ok" for every requested domain.
# ---------------------------------------------------------------------------
def test_per_domain_ppl_emitted(tmp_path, eval_module):
    """Each requested domain shows up with a numeric ppl and status="ok"."""
    out_json = tmp_path / "eval.json"
    rc = eval_module.main([
        "--smoke",
        "--out", str(out_json),
        "--domains", "wikitext,c4,gsm8k,humaneval,zh_news",
        "--max-tokens-per-domain", "2048",
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    requested = ["wikitext", "c4", "gsm8k", "humaneval", "zh_news"]
    for d in requested:
        assert d in payload["domains"], f"missing domain {d}"
        rec = payload["domains"][d]
        assert rec["status"] == "ok", f"domain {d} not ok: {rec}"
        # ppl is a positive finite number.
        assert isinstance(rec["ppl"], float), f"{d} ppl not float: {rec['ppl']}"
        assert rec["ppl"] > 0, f"{d} ppl non-positive: {rec['ppl']}"
        assert rec["ppl"] < 1e6, f"{d} ppl absurd: {rec['ppl']}"
        # tokens > 0
        assert rec["tokens"] > 0, f"{d} reported zero tokens"
        # weight is non-negative float.
        assert rec["weight"] >= 0.0
    assert payload["n_domains_evaluated"] == len(requested)
    assert payload["n_domains_skipped"] == 0


# ---------------------------------------------------------------------------
# 3. Weighted average matches sum(w * ppl) / sum(w) over status=ok domains.
# ---------------------------------------------------------------------------
def test_weighted_average_correct(tmp_path, eval_module):
    """`weighted_avg_ppl` exactly matches the documented formula."""
    out_json = tmp_path / "eval.json"
    rc = eval_module.main([
        "--smoke",
        "--out", str(out_json),
        "--domains", "wikitext,c4,gsm8k",
        "--max-tokens-per-domain", "2048",
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    # Recompute the weighted average from the per-domain entries.
    num, den = 0.0, 0.0
    for d, rec in payload["domains"].items():
        if rec.get("status") != "ok":
            continue
        num += float(rec["weight"]) * float(rec["ppl"])
        den += float(rec["weight"])
    assert den > 0, "all weights are zero -- can't compute average"
    expected = num / den
    got = payload["weighted_avg_ppl"]
    assert got is not None, "weighted_avg_ppl is None despite ok domains"
    assert abs(got - expected) < 1e-6, (
        f"weighted_avg_ppl={got} does not match recomputed={expected}"
    )


# ---------------------------------------------------------------------------
# 4. OOM on one domain is contained -- other domains still produce status=ok.
# ---------------------------------------------------------------------------
def test_handles_oom_per_domain(eval_module, tmp_path):
    """A model that raises CUDA OOM on one chunk does NOT abort the run.

    We build a stateful mock model that tracks how many times `perplexity`
    has been called and raises a canonical CUDA-OOM RuntimeError on the
    second call (= the second domain in the requested order). The first
    and third domains succeed with the deterministic mock formula.

    Verifies:
      * the failing domain has status="oom"
      * the surviving domains have status="ok" + numeric ppl
      * `weighted_avg_ppl` is computed off only the surviving domains
    """

    class _OOMOnSecondCall:
        def __init__(self) -> None:
            self.call_idx = 0
            self._fallback = eval_module._MockModel()

        def perplexity(self, chunks):
            idx = self.call_idx
            self.call_idx += 1
            if idx == 1:
                # Canonical CUDA OOM message; harness must classify as "oom".
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")
            return self._fallback.perplexity(chunks)

    out_json = tmp_path / "eval.json"
    requested = ["wikitext", "c4", "gsm8k"]
    result = eval_module.evaluate_cross_domain(
        model=_OOMOnSecondCall(),
        domains=requested,
        max_tokens_per_domain=1024,
        smoke=True,  # deterministic synthetic chunks
    )
    payload = {
        "ckpt": "<test>",
        "ts": "2026-01-01T00:00:00Z",
        "smoke": True,
        **result,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    # `c4` failed with OOM; status should reflect that.
    assert payload["domains"]["c4"]["status"] == "oom", (
        f"expected oom; got {payload['domains']['c4']}"
    )
    assert payload["domains"]["c4"]["ppl"] is None
    # The remaining two domains succeeded.
    for d in ("wikitext", "gsm8k"):
        rec = payload["domains"][d]
        assert rec["status"] == "ok", f"{d} should still be ok: {rec}"
        assert isinstance(rec["ppl"], float) and rec["ppl"] > 0
    assert payload["n_domains_evaluated"] == 2
    assert payload["n_domains_skipped"] == 1

    # Weighted avg uses only ok domains -> matches formula.
    num, den = 0.0, 0.0
    for d in ("wikitext", "gsm8k"):
        rec = payload["domains"][d]
        num += float(rec["weight"]) * float(rec["ppl"])
        den += float(rec["weight"])
    expected = num / den
    assert abs(payload["weighted_avg_ppl"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Bonus helper-level tests so coverage doesn't depend solely on main().
# ---------------------------------------------------------------------------
def test_unknown_domain_is_skipped_not_fatal(tmp_path, eval_module):
    """An unknown domain key is reported as ``unknown_domain``; doesn't crash."""
    out_json = tmp_path / "eval.json"
    rc = eval_module.main([
        "--smoke",
        "--out", str(out_json),
        "--domains", "wikitext,zztop_invalid",
        "--max-tokens-per-domain", "1024",
    ])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["domains"]["wikitext"]["status"] == "ok"
    assert payload["domains"]["zztop_invalid"]["status"] == "unknown_domain"
    assert payload["n_domains_evaluated"] == 1
    assert payload["n_domains_skipped"] == 1
