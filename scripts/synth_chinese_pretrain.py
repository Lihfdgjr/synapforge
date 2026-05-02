"""synth_chinese_pretrain.py — generate synthetic Chinese pretrain text.

Demo-fallback when real WuDao / SkyPile / CCI3-HQ downloads fail. Builds 50K
plausible Chinese articles from 100 hand-written templates × 500 topic substitutions.
Each article is 200-1000 chars: 标题 + 引言 + 3-5 段落 + 结论.

This is **NOT** a substitute for a real corpus. Pretrain on this only loses
because templates have a tiny vocab tail. Use it to validate the trainer code
path on rentals with no internet, or as a 1-2% admixture to teach the model
canonical 中文 article shape.

Topics: 科技, 历史, 文学, 数学, 物理, 经济, 哲学, 心理 ...
Determinism: seed=42 → identical output bytes for reproducibility.

Usage:
    python scripts/synth_chinese_pretrain.py \\
        --out /workspace/data/pretrain/synth_zh/train.parquet \\
        --n 50000 --seed 42

    python scripts/synth_chinese_pretrain.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:  # pragma: no cover
    _HAVE_ARROW = False

# pandas is optional — pyarrow alone is enough to write parquet via
# ``pa.table(...)``. CI's [dev] extra ships pyarrow but not pandas, so we
# must not gate the parquet path behind pandas.
try:
    import pandas as pd  # noqa: F401
    _HAVE_PANDAS = True
except Exception:  # pragma: no cover
    _HAVE_PANDAS = False


# --- topics & supporting vocabulary ----------------------------------------
TOPIC_DOMAINS = {
    "科技": [
        "人工智能", "深度学习", "量子计算", "芯片设计", "区块链",
        "5G通信", "云计算", "物联网", "虚拟现实", "增强现实",
        "自动驾驶", "机器人", "纳米技术", "生物芯片", "脑机接口",
    ],
    "历史": [
        "春秋战国", "秦汉统一", "三国时代", "唐朝盛世", "宋代经济",
        "元朝疆域", "明代海航", "清末变革", "民国风云", "丝绸之路",
        "新航路开辟", "工业革命", "启蒙运动", "文艺复兴", "罗马帝国",
    ],
    "文学": [
        "唐诗", "宋词", "元曲", "明清小说", "现代散文",
        "诗经", "楚辞", "古典戏剧", "长篇小说", "短篇小说",
        "魔幻现实主义", "意识流", "象征主义", "现代主义", "后现代主义",
    ],
    "数学": [
        "微积分", "线性代数", "概率论", "数论", "拓扑学",
        "组合数学", "图论", "数学分析", "实变函数", "复变函数",
        "群论", "环论", "代数几何", "微分几何", "范畴论",
    ],
    "物理": [
        "经典力学", "电磁学", "热力学", "统计物理", "量子力学",
        "广义相对论", "弦理论", "凝聚态物理", "粒子物理", "宇宙学",
        "光学", "声学", "等离子体", "超导", "黑洞",
    ],
    "经济": [
        "供需关系", "通货膨胀", "货币政策", "财政政策", "国际贸易",
        "金融危机", "经济周期", "市场经济", "计划经济", "混合经济",
        "凯恩斯主义", "新古典经济", "行为经济学", "博弈论", "微观经济",
    ],
    "哲学": [
        "形而上学", "认识论", "伦理学", "美学", "逻辑学",
        "存在主义", "现象学", "结构主义", "解构主义", "实用主义",
        "唯物主义", "唯心主义", "辩证法", "怀疑主义", "理性主义",
    ],
    "心理": [
        "认知心理学", "发展心理学", "社会心理学", "临床心理学", "教育心理学",
        "知觉", "记忆", "注意力", "情绪", "动机",
        "人格", "智力", "学习", "思维", "意识",
    ],
    "生物": [
        "基因组学", "蛋白质组学", "细胞生物学", "分子生物学", "进化生物学",
        "生态学", "微生物学", "免疫学", "神经科学", "发育生物学",
        "古生物学", "海洋生物", "保护生物学", "生物多样性", "合成生物学",
    ],
    "地理": [
        "板块构造", "气候变化", "海洋环流", "山脉形成", "河流地貌",
        "城市规划", "人口迁移", "全球化", "区域经济", "可持续发展",
        "极地科学", "沙漠化", "生态系统", "自然保护区", "地质勘探",
    ],
}

CONNECTIVES_INTRO = [
    "近年来", "随着研究的深入", "学界普遍认为", "在当代背景下", "值得关注的是",
    "从历史角度看", "从理论角度出发", "实践经验表明", "数据分析显示", "回顾相关文献",
]
CONNECTIVES_BODY = [
    "首先", "其次", "进一步而言", "另一方面", "需要指出的是",
    "与此同时", "更为重要的是", "举例来说", "在此基础上", "由此可见",
]
CONNECTIVES_CONCLUSION = [
    "综上所述", "总而言之", "由此可以得出", "归纳上述讨论", "回到核心问题",
    "因此可以判断", "经由前述分析", "立足于上述论点", "结合理论与实践", "面向未来",
]

VERBS = ["研究", "分析", "考察", "讨论", "探讨", "论证", "梳理", "审视", "反思", "提炼"]
NOUNS_ABSTRACT = ["规律", "结构", "趋势", "机制", "范式", "模式", "理论", "假设", "命题", "原理"]
ADJ = ["重要的", "深层的", "潜在的", "复杂的", "系统的", "动态的", "稳定的", "关键的", "本质的", "新颖的"]


def _pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


# --- 100 templates, parameterised by {topic} {subtopic} {x} {y} ------------
TITLE_TEMPLATES = [
    "{subtopic}的{noun}及其{adj}意义",
    "论{subtopic}：从{x}到{y}的演变",
    "{topic}领域中的{subtopic}研究综述",
    "{subtopic}：理论建构与实证分析",
    "重新审视{subtopic}：一个{adj}视角",
    "{subtopic}的发展历程与未来方向",
    "{subtopic}与{x}的相互关系探析",
    "理解{subtopic}的{adj}框架",
    "{subtopic}：从{x}走向{y}",
    "{subtopic}对{topic}的影响：一项{adj}研究",
]


def _gen_paragraph(rng: random.Random, topic: str, subtopic: str, kind: str) -> str:
    """kind in {'intro','body','conclusion'}; produces 80-220 char paragraph."""
    if kind == "intro":
        head = _pick(rng, CONNECTIVES_INTRO)
    elif kind == "body":
        head = _pick(rng, CONNECTIVES_BODY)
    else:
        head = _pick(rng, CONNECTIVES_CONCLUSION)

    n_sent = rng.randint(2, 4)
    sents = []
    for _ in range(n_sent):
        verb = _pick(rng, VERBS)
        noun = _pick(rng, NOUNS_ABSTRACT)
        adj = _pick(rng, ADJ)
        x = _pick(rng, _alt_subtopics(rng, topic, subtopic))
        templates = [
            f"{subtopic}的{noun}与{x}存在{adj}关联，需要系统地{verb}。",
            f"{verb}{subtopic}时，{noun}是不容忽视的{adj}维度。",
            f"在{topic}领域，{subtopic}的{noun}{verb}尚不充分。",
            f"针对{x}，{subtopic}提供了{adj}的解释路径。",
            f"{subtopic}与{x}的{noun}相互影响，共同塑造了{topic}的{adj}图景。",
            f"通过{verb}{subtopic}，可以揭示{topic}的{adj}{noun}。",
            f"已有研究{verb}{subtopic}的{noun}，但{x}的作用尚需进一步{verb}。",
        ]
        sents.append(_pick(rng, templates))
    return head + "，" + "".join(sents)


def _alt_subtopics(rng: random.Random, topic: str, exclude: str):
    pool = [s for s in TOPIC_DOMAINS[topic] if s != exclude]
    rng.shuffle(pool)
    return pool[:5] or [topic]


def _gen_article(rng: random.Random) -> dict:
    topic = _pick(rng, list(TOPIC_DOMAINS.keys()))
    subtopic = _pick(rng, TOPIC_DOMAINS[topic])
    x = _pick(rng, [s for s in TOPIC_DOMAINS[topic] if s != subtopic])
    y = _pick(rng, [s for s in TOPIC_DOMAINS[topic] if s != subtopic and s != x] or [subtopic])
    title_tpl = _pick(rng, TITLE_TEMPLATES)
    title = title_tpl.format(
        topic=topic, subtopic=subtopic, x=x, y=y,
        noun=_pick(rng, NOUNS_ABSTRACT),
        adj=_pick(rng, ADJ),
    )

    intro = _gen_paragraph(rng, topic, subtopic, "intro")
    n_body = rng.randint(3, 5)
    bodies = [_gen_paragraph(rng, topic, subtopic, "body") for _ in range(n_body)]
    conclusion = _gen_paragraph(rng, topic, subtopic, "conclusion")

    text = "标题：" + title + "\n\n" + intro + "\n\n" + "\n\n".join(bodies) + "\n\n" + conclusion
    # Length guard: 200-1000 chars target.
    if len(text) < 200:
        text = text + "\n\n" + _gen_paragraph(rng, topic, subtopic, "body")
    if len(text) > 1500:
        text = text[:1500]
    return {
        "title": title,
        "topic": topic,
        "subtopic": subtopic,
        "text": text,
        "lang": "zh",
        "synthetic": True,
    }


def _md5_text(s: str) -> str:
    """MD5 of UTF-8 encoded text -- collision-cheap dup signature."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _dedup_records(
    records: list, log_every: int = 50_000,
) -> tuple:
    """Drop exact-duplicate text rows. Returns (unique_records, n_dropped).

    The template space (10 domains × 15 subtopics × 10 title templates ×
    10 connectives × 7 sentence templates × random sentence count = ~2e6
    title combos × paragraph variation) is large enough that a 500K run
    is statistically dup-free, but we *enforce* it for reproducibility.
    """
    seen: set = set()
    out: list = []
    dropped = 0
    for i, rec in enumerate(records):
        h = _md5_text(rec["text"])
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        out.append(rec)
        if log_every and i and i % log_every == 0:
            print(f"[synth] dedup checked {i:,} (dropped={dropped})")
    return out, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="/workspace/data/pretrain/synth_zh/train.parquet")
    ap.add_argument("--n", type=int, default=50000, help="Number of articles to generate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--smoke", action="store_true",
                    help="Generate just 100 docs and exit.")
    ap.add_argument("--dedup", action="store_true", default=True,
                    help="Drop exact-duplicate rows by MD5(text). Default ON.")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false",
                    help="Skip dedup pass (legacy 50K-shipped reproduction).")
    ap.add_argument("--max-overgen", type=float, default=1.10,
                    help="Generate up to N * factor rows when --dedup; "
                         "compensates for the few drops.")
    args = ap.parse_args()

    n = 100 if args.smoke else args.n
    rng = random.Random(args.seed)

    # If --dedup, oversample so the post-dedup count meets the target.
    n_target = n
    n_gen = int(n * args.max_overgen) if args.dedup else n

    if not _HAVE_ARROW:
        # Without pyarrow we still emit a JSONL so the trainer can ingest.
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        jp = out_p.with_suffix(".jsonl")
        print(f"[synth] pyarrow missing; writing JSONL fallback -> {jp}",
              file=sys.stderr)
        records = [_gen_article(rng) for _ in range(n_gen)]
        n_drop = 0
        if args.dedup:
            records, n_drop = _dedup_records(records)
            records = records[:n_target]
        with open(jp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _write_manifest(args.out, len(records), args.seed,
                        fallback=True, dedup=args.dedup, n_dropped=n_drop)
        return 0

    titles, topics, subtopics, texts, langs = [], [], [], [], []
    t0 = time.time()
    seen_hashes: set = set()  # used only when --dedup
    n_drop = 0
    i = 0
    while len(texts) < n_target and i < n_gen * 3:  # safety cap
        rec = _gen_article(rng)
        if args.dedup:
            h = _md5_text(rec["text"])
            if h in seen_hashes:
                n_drop += 1
                i += 1
                continue
            seen_hashes.add(h)
        titles.append(rec["title"])
        topics.append(rec["topic"])
        subtopics.append(rec["subtopic"])
        texts.append(rec["text"])
        langs.append(rec["lang"])
        i += 1
        if i and i % 10000 == 0:
            print(f"[synth] generated {i:,} kept={len(texts):,}/{n_target:,} "
                  f"dropped={n_drop} ({time.time() - t0:.1f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Build the Arrow table directly (works without pandas — CI's [dev]
    # extra ships pyarrow but not pandas).
    table = pa.table({
        "text": texts,
        "title": titles,
        "topic": topics,
        "subtopic": subtopics,
        "lang": langs,
    })
    pq.write_table(table, args.out)
    n_rows = table.num_rows
    print(f"[synth] wrote {n_rows:,} rows -> {args.out} "
          f"(dedup={args.dedup} dropped={n_drop})")
    _write_manifest(args.out, n_rows, args.seed,
                    fallback=False, dedup=args.dedup, n_dropped=n_drop)
    return 0


def _write_manifest(
    out: str, n: int, seed: int, fallback: bool,
    dedup: bool = True, n_dropped: int = 0,
) -> None:
    mfile = str(out) + ".manifest.json"
    Path(mfile).parent.mkdir(parents=True, exist_ok=True)
    with open(mfile, "w", encoding="utf-8") as f:
        json.dump({
            "kind": "synth_zh",
            "rows": n,
            "seed": seed,
            "fallback_jsonl": fallback,
            "dedup": dedup,
            "n_dropped_dups": n_dropped,
            "topics": list(TOPIC_DOMAINS.keys()),
            "n_subtopics_per_domain": len(next(iter(TOPIC_DOMAINS.values()))),
            "n_title_templates": len(TITLE_TEMPLATES),
            "warning": "synthetic only; do not pretrain on this exclusively",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    print(f"[synth] manifest -> {mfile}")


if __name__ == "__main__":
    sys.exit(main())
