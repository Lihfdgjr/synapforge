#!/usr/bin/env python3
"""web_self_learn_daemon.py — minimal Wikipedia/Gutenberg pulled curated
parquet sink, with auto-restart wrapper.

This daemon ties together:

  * synapforge.learn.continual_daemon.WebContentLearner   (7-gate filter)
  * synapforge.data.web_daemon_sink.WebDaemonSink         (rolling parquet)

so the trainer can consume:

    /workspace/data/web_self_learn/web_*.parquet

via ``--data-files <glob>:<weight>`` (typically weight=0.10 — see
scripts/launch_synap1_ultra_run8_full.sh).

Sources (curated, low-noise):

  * Wikipedia (zh, en) -- via the random-article API
  * Project Gutenberg -- via random-text top-100 catalog

NOT SUPPORTED on purpose:

  * Arbitrary scrape -- needs anti-abuse rules. The 7-gate filter is for
    *content quality*, not for *crawler etiquette*.
  * SOCKS / Tor egress -- out of scope for this daemon.

Auto-restart:

  When invoked with ``--supervisor``, the daemon spawns its child loop
  via subprocess and on crash sleeps the configured backoff before
  respawning. Three failures in a 60s window halts (so we don't spam
  logs on a permanent config error).

Usage:

    # one-shot smoke (3 cycles synthesised, no real fetch)
    python3 scripts/web_self_learn_daemon.py --smoke

    # supervised long-run with real fetch (default sources)
    python3 scripts/web_self_learn_daemon.py \\
        --out-dir /workspace/data/web_self_learn \\
        --interval 30 --enable-real-fetch --supervisor

The --enable-real-fetch path requires the `requests` package; without
it we fall back to synthesised + offline curated samples so the daemon
runs in dev environments without network.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_module(name: str, path: Path):
    """Light-weight loader: avoids triggering the heavy synapforge.__init__
    side-effects (which import torch + everything)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_cd = _load_module(
    "synap_continual_daemon",
    REPO / "synapforge" / "learn" / "continual_daemon.py",
)
_wds = _load_module(
    "synap_web_daemon_sink",
    REPO / "synapforge" / "data" / "web_daemon_sink.py",
)
WebContentLearner = _cd.WebContentLearner
WebDaemonSink = _wds.WebDaemonSink
append_decision = _wds.append_decision


# ---------------------------------------------------------------------------
# Curated low-noise sources
# ---------------------------------------------------------------------------


_OFFLINE_SAMPLES: List[Tuple[str, str, str]] = [
    # (source_id, url, text)
    ("web:wikipedia.zh",
     "https://zh.wikipedia.org/wiki/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C",
     "神经网络是一种模仿生物神经系统结构和功能的计算模型。它由大量的人工神经元相互连接组成，"
     "通过调整连接权重来学习数据中的模式。深度学习是神经网络在深层结构上的扩展，已在图像识别、"
     "语音处理和自然语言理解等领域取得显著成功。反向传播算法和梯度下降是训练神经网络的核心方法。"),
    ("web:wikipedia.en",
     "https://en.wikipedia.org/wiki/Neural_network",
     "A neural network is a computational model inspired by the way biological nervous systems "
     "process information. The fundamental unit is the artificial neuron, which receives inputs, "
     "applies a non-linear activation function and emits a signal to downstream neurons. "
     "Modern deep learning extends this idea with many stacked layers and the backpropagation "
     "algorithm to fit free parameters by gradient descent."),
    ("web:gutenberg.org",
     "https://www.gutenberg.org/files/1342/1342-0.txt",
     "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
     "must be in want of a wife. However little known the feelings or views of such a man may be "
     "on his first entering a neighbourhood, this truth is so well fixed in the minds of the "
     "surrounding families that he is considered as the rightful property of some one or other "
     "of their daughters."),
    ("web:arxiv.org",
     "https://arxiv.org/abs/1706.03762",
     "We propose a new simple network architecture, the Transformer, based solely on attention "
     "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two "
     "machine translation tasks show these models to be superior in quality while being more "
     "parallelizable and requiring significantly less time to train."),
    ("web:wikipedia.zh",
     "https://zh.wikipedia.org/wiki/%E5%BE%AE%E7%A7%AF%E5%88%86",
     "微积分是数学的一个分支，主要研究极限、连续、导数和积分等概念。它由牛顿和莱布尼茨在十七世纪"
     "末独立发明，至今仍是物理学、工程学、经济学等学科的基础工具。导数描述函数局部变化率，积分则"
     "处理函数累积量；微积分基本定理建立了二者之间的联系。"),
]


def _random_offline_doc(rng: random.Random) -> dict:
    src, url, txt = rng.choice(_OFFLINE_SAMPLES)
    return {"source_id": src, "url": url, "text": txt}


def _try_real_fetch_wikipedia(
    rng: random.Random, lang: str = "en",
) -> Optional[dict]:
    """Best-effort Wikipedia random fetch via REST API.

    Falls back to None on any error so the caller can use offline samples.
    Per Wikipedia API etiquette: User-Agent must identify our use, we
    bound timeout to 8s, and we never retry within the same cycle.
    """
    try:
        import requests  # type: ignore
    except ImportError:
        return None
    api = (
        f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
    )
    headers = {
        "User-Agent": "synapforge-self-learn-daemon/1.0 "
                      "(content harvest; +https://github.com/Lihfdgjr/synapforge)",
    }
    try:
        r = requests.get(api, headers=headers, timeout=8)
        if r.status_code != 200:
            return None
        d = r.json()
        text = (d.get("extract") or "").strip()
        if len(text) < 100:
            return None
        return {
            "source_id": f"web:wikipedia.{lang}",
            "url": d.get("content_urls", {}).get(
                "desktop", {}).get("page", "")
                   or f"https://{lang}.wikipedia.org",
            "text": text,
        }
    except Exception:  # noqa: BLE001
        return None


def _produce_cycle(
    rng: random.Random, enable_real: bool, n: int = 4,
) -> List[dict]:
    """Yield ``n`` candidate documents for this cycle.

    Strategy: try real Wikipedia first (~50/50 zh and en split when
    real fetch is enabled), fall back to the offline curated set.
    """
    out: List[dict] = []
    for i in range(n):
        doc = None
        if enable_real:
            lang = "zh" if (i + rng.randint(0, 1)) % 2 == 0 else "en"
            doc = _try_real_fetch_wikipedia(rng, lang=lang)
        if doc is None:
            doc = _random_offline_doc(rng)
        out.append(doc)
    return out


# ---------------------------------------------------------------------------
# Main daemon body (single process, no supervisor)
# ---------------------------------------------------------------------------


def _run_inner(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = out_dir / "parquet" if args.parquet_subdir \
                  else out_dir
    parquet_dir.mkdir(parents=True, exist_ok=True)

    learner = WebContentLearner(
        per_source_cap_7d=args.per_source_cap_7d,
        gate_log=out_dir / "gate_log.jsonl",
        blocklist_path=out_dir / "blocked_hashes.txt",
    )
    sink = WebDaemonSink(
        out_dir=parquet_dir,
        rotate_seconds=args.rotate_seconds,
        rotate_max_rows=args.rotate_max_rows,
        rotate_max_bytes=args.rotate_max_bytes,
        write_batch_size=args.write_batch_size,
        quality_score_floor=args.quality_score_floor,
    )
    cycle_log_path = out_dir / "cycle_log.jsonl"

    rng = random.Random(args.rng_seed)
    cycles_done = 0
    accepted_total = 0
    rejected_total = 0

    def _log(rec: dict) -> None:
        try:
            with open(cycle_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _shutdown(_sig=None, _frm=None) -> None:
        try:
            sink.close()
        except Exception:  # noqa: BLE001
            pass
        _log({
            "event": "shutdown",
            "ts": time.time(),
            "cycles": cycles_done,
            "accepted_total": accepted_total,
            "rejected_total": rejected_total,
        })

    if os.name == "posix":
        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

    _log({
        "event": "start",
        "ts": time.time(),
        "out_dir": str(out_dir),
        "parquet_dir": str(parquet_dir),
        "interval_s": args.interval,
        "rotate_seconds": args.rotate_seconds,
        "enable_real_fetch": bool(args.enable_real_fetch),
    })

    try:
        while True:
            cycle_started = time.time()
            cycles_done += 1
            try:
                docs = _produce_cycle(
                    rng, args.enable_real_fetch, n=args.docs_per_cycle,
                )
            except Exception as e:  # noqa: BLE001
                _log({"event": "produce_error", "err": repr(e)})
                docs = []

            cycle_accepted = 0
            for doc in docs:
                try:
                    decision = learner.admit(
                        text=doc["text"],
                        source_id=doc["source_id"],
                        url=doc.get("url"),
                    )
                except Exception as e:  # noqa: BLE001
                    _log({"event": "admit_error", "err": repr(e)})
                    rejected_total += 1
                    continue
                if decision.accept:
                    written = append_decision(
                        sink, decision,
                        text=doc["text"],
                        source_id=doc["source_id"],
                        url=doc.get("url"),
                    )
                    if written:
                        cycle_accepted += 1
                        accepted_total += 1
                    else:
                        rejected_total += 1
                else:
                    rejected_total += 1

            # Always flush at end of cycle so the rolling shard has fresh
            # data; the trainer can pick up partial shards before
            # rotation completes.
            try:
                sink.flush()
            except Exception as e:  # noqa: BLE001
                _log({"event": "flush_error", "err": repr(e)})

            _log({
                "event": "cycle",
                "cycle": cycles_done,
                "produced": len(docs),
                "accepted": cycle_accepted,
                "elapsed_s": round(time.time() - cycle_started, 3),
                "sink_stats": sink.stats(),
            })

            if args.max_cycles and cycles_done >= args.max_cycles:
                _log({
                    "event": "max_cycles_reached",
                    "cycles": cycles_done,
                })
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()

    return 0


# ---------------------------------------------------------------------------
# Supervisor: auto-restart on crash with backoff + circuit breaker
# ---------------------------------------------------------------------------


def _run_supervisor(args: argparse.Namespace) -> int:
    """Spawn the inner daemon as a subprocess and respawn on crash.

    Circuit breaker: if 3 crashes happen within 60s the supervisor stops
    so a misconfig doesn't spam logs forever.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sup_log_path = out_dir / "supervisor_log.jsonl"

    def _log(rec: dict) -> None:
        try:
            with open(sup_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    crash_window: List[float] = []
    iteration = 0

    inner_argv = [
        sys.executable, str(Path(__file__).resolve()),
        "--out-dir", str(args.out_dir),
        "--interval", str(args.interval),
        "--per-source-cap-7d", str(args.per_source_cap_7d),
        "--rotate-seconds", str(args.rotate_seconds),
        "--rotate-max-rows", str(args.rotate_max_rows),
        "--rotate-max-bytes", str(args.rotate_max_bytes),
        "--write-batch-size", str(args.write_batch_size),
        "--quality-score-floor", str(args.quality_score_floor),
        "--rng-seed", str(args.rng_seed),
        "--docs-per-cycle", str(args.docs_per_cycle),
    ]
    if args.enable_real_fetch:
        inner_argv.append("--enable-real-fetch")
    if args.parquet_subdir:
        inner_argv.append("--parquet-subdir")
    if args.max_cycles:
        inner_argv.extend(["--max-cycles", str(args.max_cycles)])

    while True:
        iteration += 1
        start = time.time()
        _log({"event": "spawn", "iter": iteration, "ts": start})
        try:
            rc = subprocess.call(inner_argv)
        except Exception as e:  # noqa: BLE001
            _log({"event": "spawn_error", "err": repr(e)})
            rc = -1
        elapsed = time.time() - start
        _log({"event": "exit", "iter": iteration, "rc": rc,
              "elapsed_s": round(elapsed, 1)})
        if rc == 0 or args.max_cycles:
            return rc

        # Crash window check
        crash_window.append(time.time())
        crash_window = [t for t in crash_window if t > time.time() - 60]
        if len(crash_window) >= 3:
            _log({
                "event": "circuit_breaker_open",
                "n_crashes_60s": len(crash_window),
                "halt": True,
            })
            return 1
        backoff = min(60.0, 2 ** (len(crash_window) - 1))
        _log({"event": "backoff", "sleep_s": backoff})
        time.sleep(backoff)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", type=Path,
                   default=Path("./runs/web_self_learn"))
    p.add_argument("--parquet-subdir", action="store_true",
                   help="Place rolling parquet shards under <out_dir>/parquet/ "
                        "instead of <out_dir>/. Default off so the trainer's "
                        "default glob /workspace/data/web_self_learn/web_*.parquet "
                        "matches without an extra subdir.")
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--per-source-cap-7d", type=int, default=125)
    p.add_argument("--rotate-seconds", type=float, default=3600.0)
    p.add_argument("--rotate-max-rows", type=int, default=50_000)
    p.add_argument("--rotate-max-bytes", type=int, default=200 * 1024 * 1024)
    p.add_argument("--write-batch-size", type=int, default=64)
    p.add_argument("--quality-score-floor", type=float, default=0.0)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--docs-per-cycle", type=int, default=4)
    p.add_argument("--enable-real-fetch", action="store_true")
    p.add_argument("--max-cycles", type=int, default=None,
                   help="Exit after N cycles (smoke / CI).")
    p.add_argument("--smoke", action="store_true",
                   help="Run 3 cycles, fast interval, exit.")
    p.add_argument("--supervisor", action="store_true",
                   help="Spawn child + auto-restart on crash.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.smoke:
        args.interval = 0.05
        args.max_cycles = 3
        args.docs_per_cycle = 4
    if args.supervisor:
        return _run_supervisor(args)
    return _run_inner(args)


if __name__ == "__main__":
    raise SystemExit(main())
