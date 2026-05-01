"""auto_eval_daemon — automatic per-checkpoint eval pipeline.

Watches an out-dir for new `step_*.pt` and `best*.pt` checkpoints, and for
every fresh ckpt runs:

  1. honest_eval (5 EN + 5 ZH chat samples) -> auto_eval/<step>/chat.json
  2. lightweight bench subset (mmlu, hellaswag, lambada -- logits-only,
     no code generation, ~1-3 min each on a small model) -> auto_eval/<step>/bench.json
  3. (optional, only on `best*.pt`) heavyweight bench (humaneval, mbpp,
     gsm8k -- generation-bound, 10-30 min each)

Key properties:

* Streaming-aware. We don't block training. Evals run on a single-worker
  ThreadPoolExecutor, so multiple new ckpts queue up and serialize.
* GPU-share aware. If `nvidia-smi` shows another process holding the GPU
  (the trainer), we default to CPU eval; otherwise we use CUDA. Operator
  can force with `--device cuda` / `--device cpu`.
* Checksum-deduped. Each ckpt's path + sha256[:16] is recorded in
  `<out_dir>/auto_eval/.dedup.json` so we never re-evaluate the same
  bytes (training process may overwrite step files mid-flush).
* Index-friendly. After every eval, `<out_dir>/auto_eval/index.json`
  is rewritten with the merged step -> metrics map, ready for
  `scripts/plot_eval_curves.py` to plot.
* Standalone smoke. With no ckpts in the watch dir, we just idle and
  log "waiting" every cycle.

CLI:
    python scripts/auto_eval_daemon.py \
        --watch /workspace/runs/synapforge_100m \
        --interval 60 \
        --tokenizer Qwen/Qwen2.5-0.5B \
        --bench-light mmlu,hellaswag,lambada \
        --bench-heavy humaneval,mbpp,gsm8k \
        --heavy-only-best \
        [--device auto|cuda|cpu] [--n-light 50] [--n-heavy 20]
        [--once]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _log(msg: str) -> None:
    print(f"[auto-eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError as exc:
        return f"err:{exc.__class__.__name__}"


def _step_from_name(p: Path) -> Optional[int]:
    """Extract integer step from filenames like step_001250.pt / best_step_046350.pt.

    Falls back to None for plain best.pt / final.pt; the caller may bucket
    those under the file's mtime instead.
    """
    name = p.stem
    for token in name.replace("-", "_").split("_"):
        if token.isdigit():
            return int(token)
    return None


def _bucket_for(p: Path) -> str:
    """Pick a stable bucket key for a ckpt filename. Used in auto_eval/<bucket>/.

    Step ints win for `step_*.pt`; for `best*.pt` we use the embedded step
    if any, else `best_<mtime>` so each best becomes its own bucket.
    """
    s = _step_from_name(p)
    if s is not None:
        return f"{s:06d}"
    if p.stem.startswith("best"):
        # No step encoded -- bucket by mtime (seconds since epoch).
        return f"best_{int(p.stat().st_mtime)}"
    return p.stem


# ---------------------------------------------------------------------------
# GPU contention probe
# ---------------------------------------------------------------------------

def _gpu_busy() -> bool:
    """True if nvidia-smi reports another process using the GPU.

    Heuristic: any compute process found means GPU is in use. If `nvidia-smi`
    is missing or fails, return False (assume GPU free / will fall back inside torch).
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    pids = [
        line.strip() for line in proc.stdout.decode(errors="replace").splitlines()
        if line.strip()
    ]
    self_pid = str(os.getpid())
    busy = [pid for pid in pids if pid != self_pid]
    return len(busy) > 0


def _resolve_device(device_pref: str) -> str:
    if device_pref in ("cpu", "cuda"):
        return device_pref
    if _gpu_busy():
        _log("GPU busy (training?) -> running eval on CPU")
        return "cpu"
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# eval helpers (lazy imports so smoke runs even without torch installed)
# ---------------------------------------------------------------------------

def _lazy_load_model_and_tok(ckpt: Path, tokenizer: str) -> Optional[tuple]:
    """Return (model, tok) or None if loaders are unavailable.

    We import inside the function so the daemon can start without torch.
    """
    try:
        from synapforge.eval.generate import load_synapforge, load_tokenizer  # type: ignore
    except Exception as exc:
        _log(f"  load: synapforge.eval.generate unavailable ({exc!r}); skipping")
        return None
    try:
        model, src = load_synapforge(str(ckpt))
        tok = load_tokenizer(tokenizer)
        _log(f"  load OK: src={src}")
        return model, tok
    except Exception as exc:
        _log(f"  load FAIL: {type(exc).__name__}: {exc}")
        return None


def run_honest_eval(model, tok, out_path: Path, device: str) -> Dict[str, Any]:
    """5 EN + 5 ZH greedy generations. Mirrors honest_eval_hook for parity."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from honest_eval_hook import HonestEvalHook, TEST_PROMPTS_EN, TEST_PROMPTS_ZH
    except Exception:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from honest_eval_hook import HonestEvalHook, TEST_PROMPTS_EN, TEST_PROMPTS_ZH  # type: ignore

    hook = HonestEvalHook(
        model, tok, out_dir=str(out_path.parent),
        every_steps=1, max_new_tokens=40, device=device,
        prompts=TEST_PROMPTS_EN + TEST_PROMPTS_ZH,
    )
    record = hook.maybe_eval(step=1, current_ppl=None) or {}
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def run_bench_subset(
    model,
    tok,
    bench_names: List[str],
    n_cap: Optional[int],
    out_path: Path,
) -> Dict[str, Any]:
    """Run a subset of `synapforge.bench` benches with shared model/tok."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from synapforge.bench import BENCH_REGISTRY, run_bench  # type: ignore
    except Exception as exc:
        _log(f"  bench: synapforge.bench unavailable ({exc!r}); skipping")
        return {"error": f"bench-import: {exc!r}"}

    results: Dict[str, Any] = {}
    t0 = time.time()
    for name in bench_names:
        if name not in BENCH_REGISTRY:
            results[name] = {"error": f"unknown-bench: {name}"}
            continue
        kw: Dict[str, Any] = {}
        if n_cap is not None:
            kw["n_per_subject" if name == "mmlu" else "n"] = n_cap
        bench_t0 = time.time()
        try:
            results[name] = run_bench(name, model=model, tok=tok, **kw)
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        _log(f"  bench {name}: {time.time() - bench_t0:.1f}s")
    bundle = {"results": results, "wall_s": time.time() - t0}
    out_path.write_text(json.dumps(bundle, indent=2))
    return bundle


# ---------------------------------------------------------------------------
# index aggregation
# ---------------------------------------------------------------------------

def _extract_metric(summary: Dict[str, Any], bench_name: str) -> Optional[float]:
    """Pull the primary metric out of a bench summary dict."""
    if not isinstance(summary, dict) or "error" in summary:
        return None
    metric_key = {
        "humaneval": "pass@1", "mbpp": "pass@1",
        "mmlu": "acc", "gsm8k": "acc",
        "hellaswag": "acc", "lambada": "acc",
    }.get(bench_name, "acc")
    val = summary.get(metric_key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def update_index(eval_root: Path) -> Dict[str, Any]:
    """Walk <eval_root>/<bucket>/ and merge all metrics into index.json."""
    eval_root.mkdir(parents=True, exist_ok=True)
    idx: Dict[str, Any] = {}
    for sub in sorted(eval_root.iterdir()):
        if not sub.is_dir():
            continue
        bucket = sub.name
        entry: Dict[str, Any] = {"step": None, "verdict": None, "bench": {}}
        # Try to parse step from bucket name
        try:
            entry["step"] = int(bucket)
        except ValueError:
            entry["step"] = None
        chat_path = sub / "chat.json"
        if chat_path.exists():
            try:
                chat = json.loads(chat_path.read_text(encoding="utf-8"))
                entry["verdict"] = chat.get("verdict_heuristic")
                entry["chat_n"] = len(chat.get("samples") or [])
            except Exception:
                pass
        bench_path = sub / "bench.json"
        if bench_path.exists():
            try:
                bench = json.loads(bench_path.read_text(encoding="utf-8"))
                results = bench.get("results", {})
                for name, summary in results.items():
                    m = _extract_metric(summary, name)
                    if m is not None:
                        entry["bench"][name] = m
                    if name == "lambada" and "ppl" in (summary or {}):
                        entry.setdefault("ppl", summary["ppl"])
            except Exception:
                pass
        # Surface any extras we want to plot
        idx[bucket] = entry
    out_file = eval_root / "index.json"
    out_file.write_text(json.dumps(idx, indent=2))
    return idx


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

class _Dedup:
    """Persistent {ckpt_name: sha16} so we don't re-eval the same bytes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.cache: Dict[str, str] = json.loads(self.path.read_text())
        except Exception:
            self.cache = {}

    def is_fresh(self, ckpt: Path) -> bool:
        digest = _sha16(ckpt)
        return self.cache.get(ckpt.name) != digest

    def mark(self, ckpt: Path) -> None:
        self.cache[ckpt.name] = _sha16(ckpt)
        self.path.write_text(json.dumps(self.cache, indent=2))


def _list_fresh_ckpts(watch: Path, dedup: _Dedup) -> List[Path]:
    cands: List[Path] = []
    for pat in ("step_*.pt", "best*.pt"):
        cands.extend(watch.glob(pat))
    # Sort step files by step ascending so backfill goes oldest -> newest.
    cands = sorted(set(cands), key=lambda p: (_step_from_name(p) or 0, p.stat().st_mtime))
    fresh = [c for c in cands if dedup.is_fresh(c)]
    return fresh


def _eval_one(
    ckpt: Path,
    *,
    eval_root: Path,
    tokenizer: str,
    bench_light: List[str],
    bench_heavy: List[str],
    n_light: Optional[int],
    n_heavy: Optional[int],
    heavy_only_best: bool,
    device: str,
) -> None:
    """Run honest_eval + bench subsets for a single ckpt. Best-effort -- never throws."""
    bucket = _bucket_for(ckpt)
    bdir = eval_root / bucket
    bdir.mkdir(parents=True, exist_ok=True)
    _log(f"eval ckpt {ckpt.name} -> {bdir}")
    is_best = ckpt.stem.startswith("best")
    pair = _lazy_load_model_and_tok(ckpt, tokenizer)
    if pair is None:
        # Record an error stub so the index has something to show
        (bdir / "error.json").write_text(json.dumps({
            "ckpt": str(ckpt), "error": "model-load-failed", "ts": time.time(),
        }, indent=2))
        return
    model, tok = pair

    # 1) honest eval
    chat_t0 = time.time()
    try:
        run_honest_eval(model, tok, bdir / "chat.json", device=device)
    except Exception as exc:
        _log(f"  honest_eval FAIL: {type(exc).__name__}: {exc}")
    _log(f"  honest_eval: {time.time() - chat_t0:.1f}s")

    # 2) light bench subset
    light_t0 = time.time()
    try:
        run_bench_subset(
            model, tok, bench_light, n_cap=n_light,
            out_path=bdir / "bench.json",
        )
    except Exception as exc:
        _log(f"  bench-light FAIL: {type(exc).__name__}: {exc}")
    _log(f"  bench-light: {time.time() - light_t0:.1f}s")

    # 3) heavy bench subset (best ckpts only by default)
    if bench_heavy and (is_best or not heavy_only_best):
        heavy_t0 = time.time()
        try:
            run_bench_subset(
                model, tok, bench_heavy, n_cap=n_heavy,
                out_path=bdir / "bench_heavy.json",
            )
        except Exception as exc:
            _log(f"  bench-heavy FAIL: {type(exc).__name__}: {exc}")
        _log(f"  bench-heavy: {time.time() - heavy_t0:.1f}s")


def cycle_once(
    watch: Path,
    *,
    eval_root: Path,
    dedup: _Dedup,
    pool: ThreadPoolExecutor,
    pending: List[Future],
    tokenizer: str,
    bench_light: List[str],
    bench_heavy: List[str],
    n_light: Optional[int],
    n_heavy: Optional[int],
    heavy_only_best: bool,
    device_pref: str,
) -> int:
    """Discover fresh ckpts, queue them, harvest finished futures. Returns n queued."""
    # Harvest any finished futures so dedup can advance
    still_pending: List[Future] = []
    for f in pending:
        if f.done():
            try:
                f.result()
            except Exception as exc:
                _log(f"  worker raised: {type(exc).__name__}: {exc}")
        else:
            still_pending.append(f)
    pending[:] = still_pending

    fresh = _list_fresh_ckpts(watch, dedup)
    if not fresh:
        return 0
    n_q = 0
    for ckpt in fresh:
        device = _resolve_device(device_pref)
        # Mark first so we don't re-queue if the worker is slow (workers
        # serialize; running future is implicitly the source of truth).
        dedup.mark(ckpt)
        fut = pool.submit(
            _eval_one, ckpt,
            eval_root=eval_root,
            tokenizer=tokenizer,
            bench_light=bench_light,
            bench_heavy=bench_heavy,
            n_light=n_light,
            n_heavy=n_heavy,
            heavy_only_best=heavy_only_best,
            device=device,
        )
        # When worker finishes, refresh the index. Done as a callback so a
        # long-running heavy bench still keeps the index live for plotting.
        fut.add_done_callback(lambda _f, root=eval_root: update_index(root))
        pending.append(fut)
        n_q += 1
    return n_q


def watch_loop(
    watch: Path,
    interval: int,
    tokenizer: str,
    bench_light: List[str],
    bench_heavy: List[str],
    n_light: Optional[int],
    n_heavy: Optional[int],
    heavy_only_best: bool,
    device_pref: str,
    once: bool,
) -> None:
    eval_root = watch / "auto_eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    dedup = _Dedup(eval_root / ".dedup.json")
    pool = ThreadPoolExecutor(max_workers=1)
    pending: List[Future] = []

    def _term(sig, _frame):
        _log(f"got signal {sig}; draining {len(pending)} pending eval(s) and exiting")
        pool.shutdown(wait=True, cancel_futures=False)
        update_index(eval_root)
        sys.exit(0)
    signal.signal(signal.SIGINT, _term)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _term)

    _log(f"watching {watch}, every {interval}s")
    _log(f"bench-light={bench_light} bench-heavy={bench_heavy} "
         f"heavy_only_best={heavy_only_best} device={device_pref}")
    n = 0
    while True:
        n += 1
        try:
            n_q = cycle_once(
                watch,
                eval_root=eval_root,
                dedup=dedup,
                pool=pool,
                pending=pending,
                tokenizer=tokenizer,
                bench_light=bench_light,
                bench_heavy=bench_heavy,
                n_light=n_light,
                n_heavy=n_heavy,
                heavy_only_best=heavy_only_best,
                device_pref=device_pref,
            )
            if n_q == 0:
                _log(f"cycle {n}: waiting (no new ckpts; "
                     f"pending={len(pending)})")
            else:
                _log(f"cycle {n}: queued {n_q} eval(s); pending={len(pending)}")
        except Exception as exc:
            _log(f"cycle {n} EXC: {type(exc).__name__}: {exc}")
        if once:
            break
        time.sleep(interval)
    pool.shutdown(wait=True)
    update_index(eval_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="dir of step_*.pt / best*.pt")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--bench-light", default="mmlu,hellaswag,lambada",
                    help="csv of fast bench names (default: mmlu,hellaswag,lambada)")
    ap.add_argument("--bench-heavy", default="humaneval,mbpp,gsm8k",
                    help="csv of slow bench names; gated by --heavy-only-best by default")
    ap.add_argument("--n-light", type=int, default=50,
                    help="cap per light bench (smoke). Drop with --n-light 0 for full eval.")
    ap.add_argument("--n-heavy", type=int, default=20)
    ap.add_argument("--heavy-only-best", action="store_true", default=True,
                    help="only run heavy benches on best*.pt files")
    ap.add_argument("--no-heavy-only-best", dest="heavy_only_best", action="store_false")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--once", action="store_true",
                    help="run a single pass (smoke). Daemon by default.")
    args = ap.parse_args()
    watch = Path(args.watch).resolve()
    watch.mkdir(parents=True, exist_ok=True)
    n_light = None if args.n_light <= 0 else args.n_light
    n_heavy = None if args.n_heavy <= 0 else args.n_heavy
    bench_light = [b.strip() for b in args.bench_light.split(",") if b.strip()]
    bench_heavy = [b.strip() for b in args.bench_heavy.split(",") if b.strip()]
    watch_loop(
        watch, args.interval, args.tokenizer,
        bench_light, bench_heavy,
        n_light, n_heavy, args.heavy_only_best, args.device,
        args.once,
    )


if __name__ == "__main__":
    main()
