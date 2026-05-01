"""sf.data.tool_use_traces -- synthetic agent tool-use traces for SFT.

Generates training data that backs the NeuroMCP claim: the model picks tools
from a learned codebook (no JSON tool tokens, no MCP wire protocol). Each
trace shows *user goal -> agent thought -> (action, observation) loop -> final
answer*; downstream we tokenise these as ordinary chat data and let the
ActionHead learn to emit ``(name, arg)`` action tuples directly from hidden
state (see ``synapforge/action/neuromcp.py``).

Action vocabulary (matches ``synapforge/action/actuator.py``)::

    search(query: str)        - web search
    fetch(url: str)           - HTTP GET, return main text
    click(x: int, y: int)     - mouse click on screen coordinate
    type(text: str)           - keyboard input
    scroll(dy: int)           - scroll N pixels (negative = up)
    done()                    - end the trace; final answer follows

Templates produce 5-15 actions per trace across 5 task families:

    research      arxiv-style fact lookup
    coding        write a small program
    daily         weather/translation/calendar lookups
    web_nav       multi-page navigation (search -> click -> scroll -> read)
    multi_step    plan + 3-5 sub-goals + done

The synthetic distribution is *deterministic* (seeded RNG) so changing the
training set requires changing the seed, not the code. We never call out to
real services here -- this module is for *bootstrap* traces only; real
tool-use logs collected via ``synapforge.tools.WebSearchTool`` etc. should
be merged in afterwards via ``synapforge.action.skill_log``.

Smoke test::

    python -m synapforge.data.tool_use_traces --n 10
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------- #
# Action / Observation / Trace dataclasses                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Action:
    name: str           # one of ACTION_NAMES
    arg: str            # JSON-serialisable scalar arg (kept as string for uniformity)


@dataclass
class Observation:
    text: str           # what the env returned after the action


@dataclass
class Step:
    thought: str
    action: Action
    observation: Observation


@dataclass
class Trace:
    task_family: str
    user_goal: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""


ACTION_NAMES = ("search", "fetch", "click", "type", "scroll", "done")


# --------------------------------------------------------------------------- #
# Per-family templates                                                         #
# --------------------------------------------------------------------------- #


def _research_trace(rng: random.Random) -> Trace:
    topics = [
        ("transformer attention", "All You Need is Attention (2017) by Vaswani et al.",
         "the scaled dot-product attention mechanism"),
        ("ribosome structure", "the cryo-EM 2.7A structure of the bacterial 70S ribosome",
         "the small (30S) and large (50S) subunits"),
        ("liquid neural networks", "Hasani et al 2020 introduce CfC continuous-time RNNs",
         "ODE-based hidden states"),
        ("photosynthesis", "Calvin cycle (1947) outlined by Calvin and Benson",
         "carbon fixation in chloroplast stroma"),
        ("Liquid State Machine", "Maass 2002 originally proposed the LSM",
         "a recurrent reservoir of spiking neurons"),
    ]
    topic, paper, gist = rng.choice(topics)
    user = f"Find the seminal paper about {topic} and summarise its main contribution."
    steps: list[Step] = []
    steps.append(Step(
        thought=f"I need to search for the original paper on {topic}.",
        action=Action(name="search", arg=f"seminal paper {topic}"),
        observation=Observation(text=f"Top hit: {paper}. Cited 12k+ times."),
    ))
    url = f"https://arxiv.org/abs/{rng.randint(1000,2400)}.{rng.randint(10000,99999)}"
    steps.append(Step(
        thought="Open the abstract page to confirm the contribution.",
        action=Action(name="fetch", arg=url),
        observation=Observation(text=f"Abstract: introduces {gist} and shows SOTA on the benchmark."),
    ))
    if rng.random() < 0.5:
        steps.append(Step(
            thought="Scroll to the contributions section to be sure.",
            action=Action(name="scroll", arg="600"),
            observation=Observation(text="Section 3 lists three contributions matching the abstract."),
        ))
    steps.append(Step(
        thought="I have enough; finalise the answer.",
        action=Action(name="done", arg=""),
        observation=Observation(text=""),
    ))
    final = f"The paper is {paper}. Its main contribution is {gist}."
    return Trace(task_family="research", user_goal=user, steps=steps, final_answer=final)


def _coding_trace(rng: random.Random) -> Trace:
    tasks = [
        ("reverse a string", "def reverse(s):\n    return s[::-1]"),
        ("check if a number is prime", "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"),
        ("flatten a nested list one level", "def flatten(xs):\n    return [y for x in xs for y in x]"),
        ("compute factorial", "def fact(n):\n    return 1 if n <= 1 else n * fact(n-1)"),
        ("count word frequency", "from collections import Counter\ndef wf(text):\n    return Counter(text.split())"),
    ]
    task, snippet = rng.choice(tasks)
    user = f"Write a small Python function to {task}, with a couple of test cases."
    steps: list[Step] = []
    steps.append(Step(
        thought="I'll search for a canonical reference implementation as a sanity check.",
        action=Action(name="search", arg=f"python {task} idiomatic"),
        observation=Observation(text="Stack Overflow shows the standard one-line idiom."),
    ))
    steps.append(Step(
        thought="Now I draft the function in an editor.",
        action=Action(name="type", arg=snippet),
        observation=Observation(text="Editor shows the function with no syntax errors."),
    ))
    steps.append(Step(
        thought="Add two test cases.",
        action=Action(name="type", arg="\nassert reverse('abc') == 'cba'\nassert reverse('') == ''"),
        observation=Observation(text="Both assertions pass when run."),
    ))
    steps.append(Step(
        thought="Done.",
        action=Action(name="done", arg=""),
        observation=Observation(text=""),
    ))
    final = f"Here is the function:\n\n```python\n{snippet}\n```\n\nIt passes the assertions above."
    return Trace(task_family="coding", user_goal=user, steps=steps, final_answer=final)


def _daily_trace(rng: random.Random) -> Trace:
    cities = ["Tokyo", "Paris", "Beijing", "Sao Paulo", "Reykjavik"]
    cards = [
        ("weather", "What is the weather forecast for {city} this weekend?",
         "Saturday: 18C, partly cloudy. Sunday: 21C, clear sky."),
        ("translate", "How do you say 'thank you very much' in the local language of {city}?",
         "Doumo arigatou gozaimasu (in Japanese)."),
        ("currency", "Convert 100 USD to the local currency of {city}.",
         "100 USD ~ 14,800 JPY at today's mid-rate."),
    ]
    kind, q_tmpl, ans = rng.choice(cards)
    city = rng.choice(cities)
    user = q_tmpl.format(city=city)
    steps: list[Step] = []
    if kind == "weather":
        steps.append(Step(
            thought="A web search will surface a forecast service.",
            action=Action(name="search", arg=f"weather forecast {city} weekend"),
            observation=Observation(text="weather.com lists the next 7 days."),
        ))
        steps.append(Step(
            thought="Click the weekend tab.",
            action=Action(name="click", arg="640,420"),
            observation=Observation(text=ans),
        ))
    elif kind == "translate":
        steps.append(Step(
            thought="Pull from a translation site.",
            action=Action(name="fetch", arg="https://translate.example.com/api?q=thank+you+very+much"),
            observation=Observation(text=ans),
        ))
    else:
        steps.append(Step(
            thought="Currency rates change; query a live source.",
            action=Action(name="search", arg=f"USD to local currency {city}"),
            observation=Observation(text=ans),
        ))
    steps.append(Step(
        thought="I have the answer.",
        action=Action(name="done", arg=""),
        observation=Observation(text=""),
    ))
    return Trace(task_family="daily", user_goal=user, steps=steps, final_answer=ans)


def _web_nav_trace(rng: random.Random) -> Trace:
    sites = [
        ("github.com", "starred repository for synapforge",
         "synapforge has 1.2k stars; latest release v1.0."),
        ("hackernews.com", "top story about LLM training cost",
         "Top story: Anthropic publishes paper on small-data scaling laws."),
        ("arxiv.org", "latest neural ODE papers",
         "Three new papers in cs.LG this week on continuous-time RNNs."),
    ]
    site, query, ans = rng.choice(sites)
    user = f"Open {site} and tell me the {query}."
    steps: list[Step] = []
    steps.append(Step(
        thought=f"I'll first navigate to {site}.",
        action=Action(name="fetch", arg=f"https://{site}"),
        observation=Observation(text=f"{site} home page loaded; saw the navigation bar."),
    ))
    steps.append(Step(
        thought="Search within the site for the topic.",
        action=Action(name="type", arg=query),
        observation=Observation(text="Search box now contains the query."),
    ))
    steps.append(Step(
        thought="Submit the search.",
        action=Action(name="click", arg="800,80"),
        observation=Observation(text="Results page shows ~25 hits."),
    ))
    steps.append(Step(
        thought="Scroll to scan the list.",
        action=Action(name="scroll", arg="400"),
        observation=Observation(text="The first result looks most relevant."),
    ))
    steps.append(Step(
        thought="Open the top result.",
        action=Action(name="click", arg="200,260"),
        observation=Observation(text=ans),
    ))
    steps.append(Step(
        thought="Done.",
        action=Action(name="done", arg=""),
        observation=Observation(text=""),
    ))
    return Trace(task_family="web_nav", user_goal=user, steps=steps, final_answer=ans)


def _multi_step_trace(rng: random.Random) -> Trace:
    plans = [
        ("Plan a 3-day Tokyo trip.",
         ["Tokyo travel itinerary 3 days",
          "best ramen Shibuya",
          "open a maps tab to check the route"],
         "3-day plan: Day 1 Asakusa + Skytree, Day 2 Shibuya + Harajuku, Day 3 Akihabara + Imperial Palace."),
        ("Build a reading list of 5 ML papers.",
         ["seminal LLM scaling papers 2020-2024",
          "neural ODE recent advances",
          "spiking neural network with surrogate gradient"],
         "Reading list: GPT-3, Chinchilla, LLaMA, Liquid CfC, SuperSpike."),
        ("Compare 3 open-source vector databases.",
         ["faiss vs hnswlib comparison 2024",
          "qdrant performance benchmark",
          "milvus production deployment"],
         "Faiss is fastest at static scale, Qdrant has the best filtering, Milvus offers cluster mode."),
    ]
    user, queries, ans = rng.choice(plans)
    steps: list[Step] = []
    steps.append(Step(
        thought="Decompose the task into 3 sub-questions; search each in turn.",
        action=Action(name="search", arg=queries[0]),
        observation=Observation(text="Got 5 candidate sources; bookmarking top 3."),
    ))
    steps.append(Step(
        thought="Now the second sub-question.",
        action=Action(name="search", arg=queries[1]),
        observation=Observation(text="Found a recent comparison article."),
    ))
    steps.append(Step(
        thought="Third sub-question.",
        action=Action(name="search", arg=queries[2]),
        observation=Observation(text="Got a deployment guide blog post."),
    ))
    steps.append(Step(
        thought="Synthesise the answers.",
        action=Action(name="done", arg=""),
        observation=Observation(text=""),
    ))
    return Trace(task_family="multi_step", user_goal=user, steps=steps, final_answer=ans)


_FAMILY_FNS: dict[str, Callable[[random.Random], Trace]] = {
    "research":   _research_trace,
    "coding":     _coding_trace,
    "daily":      _daily_trace,
    "web_nav":    _web_nav_trace,
    "multi_step": _multi_step_trace,
}


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def generate_trace(family: str | None = None,
                   rng: random.Random | None = None) -> Trace:
    """Generate one synthetic trace. Family auto-selected if None."""
    rng = rng or random.Random()
    if family is None:
        family = rng.choice(tuple(_FAMILY_FNS))
    if family not in _FAMILY_FNS:
        raise ValueError(f"unknown family {family!r}; "
                         f"choose from {sorted(_FAMILY_FNS)}")
    return _FAMILY_FNS[family](rng)


def generate_dataset(n: int = 200, seed: int = 0) -> list[Trace]:
    """Generate ``n`` traces, balanced across the 5 task families."""
    rng = random.Random(seed)
    families = list(_FAMILY_FNS)
    out: list[Trace] = []
    for i in range(n):
        family = families[i % len(families)]
        out.append(generate_trace(family, rng))
    rng.shuffle(out)
    return out


def trace_to_chat(trace: Trace) -> dict:
    """Render a Trace as the (instruction, output) pair our SFT trainer expects.

    The output text intentionally contains explicit ``Thought:`` / ``Action:`` /
    ``Observation:`` markers so a tokenised stream still teaches the model the
    same structure we want at inference (where the ActionHead emits the action
    directly from hidden state).
    """
    body_lines: list[str] = []
    for step in trace.steps:
        body_lines.append(f"Thought: {step.thought}")
        body_lines.append(
            f"Action: {step.action.name}({json.dumps(step.action.arg, ensure_ascii=False)})"
        )
        if step.action.name != "done":
            body_lines.append(f"Observation: {step.observation.text}")
    body = "\n".join(body_lines)
    answer = f"\nFinal: {trace.final_answer}" if trace.final_answer else ""
    return {
        "instruction": trace.user_goal,
        "input": "",
        "output": f"{body}{answer}",
        "source": "tool_use_traces",
        "language": "en",
        "task_family": trace.task_family,
    }


def save_traces(out_path: str | Path, n: int = 200, seed: int = 0) -> Path:
    """Generate and write a JSON list of chat-format examples."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traces = generate_dataset(n=n, seed=seed)
    payload = [trace_to_chat(t) for t in traces]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return out_path


def _print_sample(trace: Trace) -> None:
    print(f"=== task_family={trace.task_family} ===")
    print(f"USER: {trace.user_goal}")
    for i, step in enumerate(trace.steps):
        print(f"  step {i}: {step.action.name}({step.action.arg!r})")
        if step.thought:
            print(f"    thought: {step.thought}")
        if step.observation.text:
            print(f"    obs:     {step.observation.text}")
    if trace.final_answer:
        print(f"FINAL: {trace.final_answer}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="how many sample traces to print")
    ap.add_argument("--save", type=str, default="",
                    help="if non-empty, also save 200 traces to this path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for _ in range(args.n):
        family = rng.choice(tuple(_FAMILY_FNS))
        trace = generate_trace(family, rng)
        _print_sample(trace)
        print()
    if args.save:
        out = save_traces(args.save, n=200, seed=args.seed)
        print(f"saved 200 traces to {out}")


if __name__ == "__main__":
    main()
