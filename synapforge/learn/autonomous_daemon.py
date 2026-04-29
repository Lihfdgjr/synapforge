"""
Autonomous web learning daemon (replaces hardcoded bilibili daemon).

Loop:
  1. SelfGoalProposer picks N topics where the model is uncertain
     (low coverage in web_cache, high topic-diversity gap)
  2. For each topic: WebSearch across multiple sources (bilibili / arxiv /
     wikipedia / duckduckgo) → fetch content
  3. WebPoisonGate.admit() filters via 7-gate pipeline; survivors append
     to web_cache.jsonl with full provenance
  4. Sleep, repeat

Run as systemd-run user service (per memory feedback_mcp_nohup_hangs_use_systemd_run.md).

CLI:
  python -m synapforge.learn.autonomous_daemon \
    --out /workspace/data/web_cache.jsonl \
    --interval-min 30 --topics-per-cycle 6 \
    --enable-trak-gate False  # heavy, optional
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from synapforge.defense import (
    ChatPoisonGate,
    PoisonDetector,
    ProvenanceTracker,
    WebPoisonGate,
)
from synapforge.defense.gates import SourceBudget
from synapforge.defense.poison_detector import (
    make_dedup_signal,
    make_user_kl_signal,
)


SEED_TOPICS = {
    "math": [
        "高等数学微积分",
        "线性代数特征值",
        "概率论贝叶斯定理",
        "calculus integration techniques",
        "linear algebra eigenvectors",
        "probability theory bayes rule",
        "数学归纳法证明",
        "组合数学排列组合",
    ],
    "code": [
        "Python 多线程编程",
        "PyTorch 自定义 layer",
        "C++ 模板元编程",
        "rust ownership borrow",
        "algorithm sorting comparison",
        "data structure tree traversal",
        "递归动态规划",
        "异步 IO async await",
    ],
    "science": [
        "量子力学薛定谔方程",
        "热力学第二定律",
        "蛋白质折叠机制",
        "general relativity geodesic",
        "neural network plasticity",
        "thermodynamics entropy",
        "化学键共价键离子键",
        "细胞分裂减数分裂",
    ],
    "language": [
        "古文翻译方法",
        "英语语法虚拟语气",
        "汉字部首结构",
        "english grammar conditional",
        "rhetoric techniques metaphor",
        "诗词鉴赏意象",
    ],
    "humanities": [
        "中国历史朝代",
        "世界历史工业革命",
        "西方哲学康德",
        "philosophy phenomenology",
        "经济学供需平衡",
        "心理学认知偏差",
    ],
}

ALL_TOPICS = [(cat, t) for cat, ts in SEED_TOPICS.items() for t in ts]


class SelfGoalProposer:
    """Proposes topics where model is most uncertain.

    Lite version (no full model load): uses coverage gap in web_cache.jsonl.
    Topics with fewest cached entries get priority. This is a stand-in for
    real FreeEnergy-driven goal proposal until v4.2 trainer plugs in.

    Future hook: load latest ckpt, run probe inputs, compute FreeEnergy
    per topic via cross-entropy on a topic-conditioned probe → pick
    highest-FE topics.
    """

    def __init__(self, web_cache_path: str | Path) -> None:
        self.web_cache_path = Path(web_cache_path)

    def _topic_counts(self) -> Counter:
        counts: Counter = Counter()
        if not self.web_cache_path.exists():
            return counts
        with open(self.web_cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                topic = e.get("topic", "unknown")
                counts[topic] += 1
        return counts

    def propose(self, n: int = 6) -> List[tuple[str, str]]:
        """Pick N topics where coverage is lowest, weighted by category balance."""
        counts = self._topic_counts()
        scored: List[tuple[float, str, str]] = []
        for cat, topic in ALL_TOPICS:
            c = counts.get(topic, 0) + counts.get(cat, 0) * 0.1
            score = -c + random.uniform(-0.5, 0.5)
            scored.append((score, cat, topic))
        scored.sort(reverse=True)
        return [(cat, topic) for _, cat, topic in scored[:n]]


def _http_get(url: str, timeout: int = 8) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 SynapForge-Learn/4.2",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("utf-8", errors="ignore")
    except Exception:
        return None


def search_bilibili(query: str, n: int = 3) -> List[Dict]:
    """Search bilibili via web-interface API."""
    url = (
        "https://api.bilibili.com/x/web-interface/search/all/v2"
        f"?keyword={urllib.parse.quote(query)}"
    )
    raw = _http_get(url)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    items = data.get("data", {}).get("result", [])
    out = []
    for group in items:
        if group.get("result_type") != "video":
            continue
        for vid in group.get("data", [])[:n]:
            title = re.sub(r"<[^>]+>", "", vid.get("title", ""))
            desc = vid.get("description", "")
            bvid = vid.get("bvid")
            if not title or not bvid:
                continue
            out.append({
                "url": f"https://www.bilibili.com/video/{bvid}",
                "title": title,
                "text": f"{title}\n\n{desc}",
                "source_id": "web:bilibili.com",
                "bvid": bvid,
            })
    return out


def search_arxiv(query: str, n: int = 3) -> List[Dict]:
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{urllib.parse.quote(query)}"
        f"&start=0&max_results={n}"
    )
    raw = _http_get(url)
    if not raw:
        return []
    out = []
    for m in re.finditer(
        r"<entry>(.+?)</entry>", raw, flags=re.DOTALL
    ):
        block = m.group(1)
        title_m = re.search(r"<title>(.+?)</title>", block, re.DOTALL)
        summary_m = re.search(r"<summary>(.+?)</summary>", block, re.DOTALL)
        link_m = re.search(r'<link href="([^"]+)"', block)
        if not title_m or not summary_m:
            continue
        title = title_m.group(1).strip()
        summary = summary_m.group(1).strip()
        link = link_m.group(1) if link_m else ""
        out.append({
            "url": link,
            "title": title,
            "text": f"{title}\n\n{summary}",
            "source_id": "web:arxiv.org",
        })
    return out


def search_wikipedia(query: str, lang: str = "zh") -> List[Dict]:
    url = (
        f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
        f"{urllib.parse.quote(query)}"
    )
    raw = _http_get(url)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    title = data.get("title", "")
    extract = data.get("extract", "")
    if not extract:
        return []
    return [{
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "title": title,
        "text": f"{title}\n\n{extract}",
        "source_id": f"web:{lang}.wikipedia.org",
    }]


def search_all(query: str) -> List[Dict]:
    out: List[Dict] = []
    for fn in (search_bilibili, search_arxiv):
        try:
            out.extend(fn(query, n=2))
        except Exception:
            pass
    try:
        is_zh = any('一' <= c <= '鿿' for c in query)
        out.extend(search_wikipedia(query, lang="zh" if is_zh else "en"))
    except Exception:
        pass
    return out


class AutonomousLearnDaemon:
    def __init__(
        self,
        web_cache: str | Path,
        provenance_log: str | Path = "/workspace/runs/provenance.jsonl",
        blocklist: str | Path = "/workspace/runs/blocked_hashes.txt",
        interval_min: int = 30,
        topics_per_cycle: int = 6,
    ) -> None:
        self.proposer = SelfGoalProposer(web_cache)

        self.provenance = ProvenanceTracker(
            log_path=provenance_log,
            blocklist_path=blocklist,
        )
        detector = PoisonDetector()
        detector.register_signal("dup", make_dedup_signal(self.provenance))
        detector.register_signal("user_kl", make_user_kl_signal({}))
        self.detector = detector

        self.budget = SourceBudget(
            per_hour=50, per_day=1000, per_7d_per_source=125,
        )
        self.gate = WebPoisonGate(
            detector=self.detector,
            provenance=self.provenance,
            budget=self.budget,
            out_jsonl=web_cache,
        )

        self.interval_s = interval_min * 60
        self.topics_per_cycle = topics_per_cycle
        self._cycle = 0

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def cycle(self) -> Dict:
        self._cycle += 1
        topics = self.proposer.propose(n=self.topics_per_cycle)
        self._log(f"cycle {self._cycle}: topics = {[t for _, t in topics]}")

        admitted = 0
        rejected = 0
        for cat, topic in topics:
            try:
                results = search_all(topic)
            except Exception as e:
                self._log(f"  search error '{topic}': {e}")
                continue
            for r in results:
                decision = self.gate.admit(
                    text=r["text"],
                    source_id=r["source_id"],
                    url=r.get("url"),
                    topic=cat,
                    question=f"关于 {topic} 你能讲讲吗?",
                    extra={"title": r.get("title", "")},
                )
                if decision.accept:
                    admitted += 1
                    self._log(
                        f"  +1 {r['source_id']} ({cat}) trust={decision.provenance.trust_score:.2f} "
                        f"low_lr={decision.low_lr}"
                    )
                else:
                    rejected += 1
        report = {
            "cycle": self._cycle,
            "admitted": admitted,
            "rejected": rejected,
            "budget": self.budget.stats(),
            "provenance": {
                "total": self.provenance.stats()["total_admitted"],
                "blocked": self.provenance.stats()["total_blocked"],
            },
        }
        self._log(f"  cycle done: +{admitted} -{rejected} | {report['provenance']}")
        return report

    def run(self) -> None:
        self._log(
            f"AutonomousLearnDaemon started: interval={self.interval_s}s, "
            f"topics_per_cycle={self.topics_per_cycle}"
        )
        while True:
            try:
                self.cycle()
            except Exception as e:
                self._log(f"cycle exception: {e!r}")
            time.sleep(self.interval_s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="/workspace/data/web_cache.jsonl")
    p.add_argument("--provenance", type=str, default="/workspace/runs/provenance.jsonl")
    p.add_argument("--blocklist", type=str, default="/workspace/runs/blocked_hashes.txt")
    p.add_argument("--interval-min", type=int, default=30)
    p.add_argument("--topics-per-cycle", type=int, default=6)
    args = p.parse_args()

    daemon = AutonomousLearnDaemon(
        web_cache=args.out,
        provenance_log=args.provenance,
        blocklist=args.blocklist,
        interval_min=args.interval_min,
        topics_per_cycle=args.topics_per_cycle,
    )
    daemon.run()


if __name__ == "__main__":
    main()
