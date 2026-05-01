"""sf.learn.web_curriculum — 50-task autocurriculum for neural computer use.

5 levels, 10 tasks each.  Every task is a tuple

    (level, url, instruction_zh, instruction_en, success_text_regex,
     success_url_substr, max_steps)

The agent never reads the instruction string — it's there for human review.
Reward is regex/url match on the final page (see web_env.WebBrowserEnv).
We *gate* progression: agent stays in level L until rolling success rate
over the last 10 episodes >= ``promotion_threshold`` (default 0.6).

Why 5 levels?
  L1  open URL + scroll                       (motor skills)
  L2  search + click first result             (basic 2-step plan)
  L3  navigate, find specific text            (perceptual grounding)
  L4  fill form, submit                       (typing + key chord)
  L5  multi-step research (arxiv -> author    (long-horizon credit assign)
      -> related paper -> summarise)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# Curriculum task spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebTask:
    level: int
    url: str
    instruction_zh: str
    instruction_en: str
    success_text_regex: str | None
    success_url_substr: str | None
    max_steps: int = 24

    @property
    def name(self) -> str:
        prefix = self.url.split("//", 1)[-1].split("/", 1)[0]
        return f"L{self.level}/{prefix}"


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def _build_catalogue() -> tuple[WebTask, ...]:
    L1 = (
        WebTask(1, "https://www.bing.com",
                "打开页面并向下滚动一屏", "open page and scroll one screen",
                None, "bing.com", 12),
        WebTask(1, "https://en.wikipedia.org/wiki/Neural_network",
                "打开维基百科,滚到 references 节",
                "open wikipedia, scroll to references",
                r"references|参考", "Neural_network", 16),
        WebTask(1, "https://arxiv.org/abs/2006.04439",
                "打开 arxiv 页, 滚到 abstract 末尾",
                "open arxiv, scroll past abstract",
                r"abstract|comments", "2006.04439", 12),
        WebTask(1, "https://duckduckgo.com",
                "打开 duckduckgo 主页", "open duckduckgo home",
                None, "duckduckgo", 6),
        WebTask(1, "https://github.com/explore",
                "打开 github explore", "open github explore",
                None, "github.com/explore", 6),
        WebTask(1, "https://news.ycombinator.com",
                "打开 hacker news", "open hacker news",
                None, "ycombinator", 6),
        WebTask(1, "https://www.bilibili.com",
                "打开 bilibili 主页", "open bilibili home",
                None, "bilibili", 6),
        WebTask(1, "https://example.com",
                "打开 example", "open example",
                r"example", "example", 4),
        WebTask(1, "https://www.python.org",
                "打开 python 官网", "open python.org",
                r"python", "python.org", 6),
        WebTask(1, "https://pytorch.org",
                "打开 pytorch 官网", "open pytorch.org",
                r"pytorch", "pytorch.org", 6),
    )
    L2 = (
        WebTask(2, "https://www.bing.com",
                "搜 'neural network' 并点第一个结果",
                "search 'neural network' and click first result",
                r"neural network", "wikipedia.org", 18),
        WebTask(2, "https://www.bing.com",
                "搜 '神经形态计算'", "search '神经形态计算'",
                r"神经形态|neuromorph", "search?q=", 14),
        WebTask(2, "https://www.bing.com",
                "搜 'liquid neural network'",
                "search 'liquid neural network'",
                r"liquid", "search?q=liquid", 14),
        WebTask(2, "https://www.bing.com",
                "搜 'spiking neuron'", "search 'spiking neuron'",
                r"spiking", "search?q=spiking", 14),
        WebTask(2, "https://www.bing.com",
                "搜 'STDP plasticity'", "search 'STDP plasticity'",
                r"STDP|plasticity", "search?q=STDP", 14),
        WebTask(2, "https://duckduckgo.com",
                "ddg 搜 'transformer attention'",
                "ddg search 'transformer attention'",
                r"attention", "duckduckgo", 14),
        WebTask(2, "https://www.bing.com",
                "搜 'computer use agent' 后点第一个 wikipedia 结果",
                "search 'computer use agent' click first wiki",
                r"agent|claude", "wikipedia.org", 18),
        WebTask(2, "https://www.bing.com",
                "搜 'arxiv liquid time' 并点 arxiv 结果",
                "search 'arxiv liquid time' click arxiv result",
                r"liquid time", "arxiv.org", 18),
        WebTask(2, "https://www.bing.com",
                "搜 'browser automation' 并点 playwright 结果",
                "search 'browser automation' click playwright",
                r"playwright", "playwright", 18),
        WebTask(2, "https://www.bing.com",
                "搜 'synapforge'", "search 'synapforge'",
                r"synapforge", "search?q=synapforge", 14),
    )
    L3 = (
        WebTask(3, "https://en.wikipedia.org/wiki/Neural_network",
                "打开 NN 维基, 找到 'biological' 段并点入连接",
                "open NN wiki, find 'biological' section and click",
                r"biological", "wiki", 24),
        WebTask(3, "https://en.wikipedia.org/wiki/Spiking_neural_network",
                "打开 SNN 维基, 找 'STDP' 关键字",
                "open SNN wiki, find 'STDP' keyword",
                r"STDP|spike-timing", "Spiking_neural_network", 22),
        WebTask(3, "https://arxiv.org/abs/2006.04439",
                "arxiv 上打开 Hasani liquid time, 找 author 名单",
                "open Hasani liquid time, find author list",
                r"hasani", "2006.04439", 18),
        WebTask(3, "https://arxiv.org/abs/1705.05363",
                "ICM 论文, 找 'curiosity'",
                "ICM paper, find 'curiosity'",
                r"curiosity", "1705.05363", 18),
        WebTask(3, "https://en.wikipedia.org/wiki/Hebbian_theory",
                "Hebb 原理, 找 'fire together'",
                "Hebbian theory, find 'fire together'",
                r"fire together|wire together", "Hebbian", 16),
        WebTask(3, "https://en.wikipedia.org/wiki/Free_energy_principle",
                "free energy 原则, 找 friston",
                "free energy, find friston",
                r"friston", "Free_energy_principle", 16),
        WebTask(3, "https://en.wikipedia.org/wiki/Intrinsic_motivation_(artificial_intelligence)",
                "intrinsic motivation 维基", "intrinsic motivation wiki",
                r"intrinsic", "Intrinsic_motivation", 14),
        WebTask(3, "https://en.wikipedia.org/wiki/Computer_use",
                "computer use 维基, 找 anthropic",
                "computer use wiki, find anthropic",
                r"anthropic", "Computer_use", 16),
        WebTask(3, "https://en.wikipedia.org/wiki/Reinforcement_learning",
                "RL 维基, 找 'reward'", "RL wiki, find 'reward'",
                r"reward", "Reinforcement_learning", 14),
        WebTask(3, "https://en.wikipedia.org/wiki/Liquid_neural_network",
                "liquid NN 维基", "liquid NN wiki",
                r"liquid", "Liquid_neural_network", 14),
    )
    L4 = (
        WebTask(4, "https://duckduckgo.com",
                "在 ddg 输入框打 'spiking', 按 Enter",
                "duckduckgo: type 'spiking' and press enter",
                r"spiking", "duckduckgo", 22),
        WebTask(4, "https://www.bing.com",
                "bing: 输入并提交 'liquid neural network'",
                "bing: type and submit 'liquid neural network'",
                r"liquid", "search?q=liquid", 22),
        WebTask(4, "https://en.wikipedia.org/wiki/Special:Search",
                "维基搜索框: 输 'STDP' Enter",
                "wiki search: type 'STDP' enter",
                r"STDP", "Special:Search", 22),
        WebTask(4, "https://github.com/search",
                "github 搜 'synapforge'", "github search 'synapforge'",
                r"synapforge", "search", 20),
        WebTask(4, "https://news.ycombinator.com",
                "HN 顶部搜框 'neural'", "HN top search 'neural'",
                r"neural", "ycombinator", 22),
        WebTask(4, "https://www.bilibili.com",
                "B站搜框: '神经网络'", "bilibili search '神经网络'",
                r"神经", "bilibili", 22),
        WebTask(4, "https://duckduckgo.com",
                "ddg: 'computer use agent' Enter",
                "ddg: 'computer use agent' enter",
                r"computer use|agent", "duckduckgo", 22),
        WebTask(4, "https://www.bing.com",
                "bing 搜 'claude computer use' Enter",
                "bing 'claude computer use' enter",
                r"claude", "search?q=claude", 22),
        WebTask(4, "https://www.bing.com",
                "搜并打开 wikipedia 'plasticity'",
                "search and open wikipedia 'plasticity'",
                r"plasticity", "wikipedia.org", 24),
        WebTask(4, "https://www.bing.com",
                "搜 'browser playwright' 并点 playwright.dev",
                "search 'browser playwright' open playwright.dev",
                r"playwright", "playwright.dev", 24),
    )
    L5 = (
        WebTask(5, "https://arxiv.org",
                "arxiv: 找 hasani liquid 论文, 找 PDF",
                "arxiv: find hasani liquid paper, find PDF",
                r"hasani.*liquid|liquid.*hasani", "arxiv.org", 32),
        WebTask(5, "https://arxiv.org",
                "arxiv: 搜 'plasticity rl', 打开第一篇, 滚到 conclusion",
                "arxiv: search 'plasticity rl', open first, scroll to conclusion",
                r"conclusion", "arxiv.org", 32),
        WebTask(5, "https://en.wikipedia.org/wiki/Neural_network",
                "维基 NN, 找 'spiking' 链接, 跳转, 再找 'STDP'",
                "wiki NN, find 'spiking' link, jump, then find 'STDP'",
                r"STDP", "Spiking", 28),
        WebTask(5, "https://news.ycombinator.com",
                "HN: 找含 'neural' 的帖, 打开评论页",
                "HN: find post containing 'neural', open comments",
                r"comments|neural", "ycombinator", 30),
        WebTask(5, "https://github.com/explore",
                "github: 找 'plasticity' repo, 进 README",
                "github: find 'plasticity' repo, enter README",
                r"readme|plasticity", "github.com", 30),
        WebTask(5, "https://arxiv.org",
                "arxiv: 搜 'computer use agent', 找 anthropic 文",
                "arxiv: search 'computer use agent', find anthropic paper",
                r"anthropic", "arxiv.org", 32),
        WebTask(5, "https://www.bing.com",
                "搜 'synapforge github', 进 repo, 看 README, 滚到 license",
                "search 'synapforge github', enter repo, see README, scroll to license",
                r"license|MIT|Apache", "github.com", 32),
        WebTask(5, "https://en.wikipedia.org/wiki/Hebbian_theory",
                "Hebb 维基, 跳到 'STDP' 子页, 滚到 reference",
                "Hebb wiki, jump to STDP subpage, scroll to reference",
                r"reference", "Spike", 32),
        WebTask(5, "https://arxiv.org",
                "arxiv: 'free energy active inference', 进 friston, 摘要",
                "arxiv: 'free energy active inference', open friston, abstract",
                r"friston|abstract", "arxiv.org", 32),
        WebTask(5, "https://www.bilibili.com",
                "B站 搜 '类脑计算', 进首条, 滚到弹幕区",
                "bilibili search '类脑', open first, scroll to comments",
                r"弹幕|comments", "bilibili.com", 32),
    )
    return L1 + L2 + L3 + L4 + L5


CATALOGUE: tuple[WebTask, ...] = _build_catalogue()
assert len(CATALOGUE) == 50, f"expected 50 tasks, got {len(CATALOGUE)}"


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------


class WebCurriculum:
    """Gate progression on rolling success rate.

    Stays in level L until ``rolling_success >= promotion_threshold`` for the
    last ``window`` episodes; then advances to L+1.  Caller must call
    ``record(success: bool)`` after each episode.

    ``current_tasks`` returns just the active level's tasks (10 of them).
    """

    def __init__(
        self,
        catalogue: Iterable[WebTask] = CATALOGUE,
        window: int = 10,
        promotion_threshold: float = 0.6,
        start_level: int = 1,
    ) -> None:
        self._catalogue = tuple(catalogue)
        if not self._catalogue:
            raise ValueError("catalogue is empty")
        self.window = int(window)
        self.promotion_threshold = float(promotion_threshold)
        self.level = int(start_level)
        self._max_level = max(t.level for t in self._catalogue)
        self._rolling: list[int] = []
        self._task_idx = 0

    @property
    def max_level(self) -> int:
        return self._max_level

    def current_tasks(self) -> list[WebTask]:
        return [t for t in self._catalogue if t.level == self.level]

    def next_task(self) -> WebTask:
        tasks = self.current_tasks()
        t = tasks[self._task_idx % len(tasks)]
        self._task_idx += 1
        return t

    def record(self, success: bool) -> dict:
        self._rolling.append(1 if success else 0)
        if len(self._rolling) > self.window:
            self._rolling = self._rolling[-self.window :]
        rate = sum(self._rolling) / max(1, len(self._rolling))
        promoted = False
        if (
            len(self._rolling) >= self.window
            and rate >= self.promotion_threshold
            and self.level < self._max_level
        ):
            self.level += 1
            self._rolling.clear()
            self._task_idx = 0
            promoted = True
        return {
            "level": self.level,
            "rolling_success": rate,
            "window_filled": len(self._rolling),
            "promoted": promoted,
        }


__all__ = ["WebTask", "WebCurriculum", "CATALOGUE"]
