"""Smoke test for sf.self_learn / sf.defense / sf.tools / sf.adversarial / sf.distill.

Run:
    /opt/conda/bin/python /workspace/synapforge/test_round9.py
Each component is exercised in isolation and the full integrated stack is
exercised at the end (SelfLearnEngine + DefenseStack + AdversarialTrainer +
DistillTrainer + ToolCaller). Pure CPU; no network; bf16-friendly.
"""
from __future__ import annotations

import sys
import traceback
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- toy model ------------------------------------------------------


class Toy(nn.Module):
    """A 2-layer MLP that accepts either ``tokens=`` (long) or float tensor.

    Vocab = 16, hidden = 32. For continuous inputs of shape [B, hidden] we
    return [B, hidden]; for tokens [B, T] we return [B, T, vocab].
    """

    def __init__(self, vocab: int = 16, hidden: int = 32, ttt_extra: bool = True):
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # Plastic params named so WeightFirewall.allow matches.
        if ttt_extra:
            self.ttt_fast = nn.Parameter(torch.zeros(hidden))
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x: torch.Tensor | None = None, tokens: torch.Tensor | None = None):
        if tokens is None:
            tokens = x
        if tokens is None:
            raise ValueError("need x or tokens")
        if tokens.dtype in (torch.long, torch.int64):
            h = self.embed(tokens.clamp(min=0, max=self.vocab - 1))
            h = F.gelu(self.fc1(h))
            h = h + getattr(self, "ttt_fast", 0.0)
            h = self.fc2(h)
            return self.lm_head(h)
        # continuous
        h = F.gelu(self.fc1(tokens))
        return self.fc2(h)


# ---------- counters -------------------------------------------------------


PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=4)
        FAILED.append((name, f"{exc}\n{tb}"))
        print(f"  FAIL  {name}: {exc}")


# ---------- tests ----------------------------------------------------------


def t_imports():
    import synapforge as sf
    assert hasattr(sf, "SelfLearnEngine")
    assert hasattr(sf, "DefenseStack")
    assert hasattr(sf, "ToolRegistry")
    assert hasattr(sf, "ToolCaller")
    assert hasattr(sf, "FastGradientSignAttack")
    assert hasattr(sf, "DistillTrainer")


def t_replay_buffer():
    from synapforge.self_learn import ExperienceReplayBuffer
    buf = ExperienceReplayBuffer(capacity=8, alpha=0.5)
    for i in range(12):
        buf.add(torch.randn(4), torch.randn(4), reward=float(i))
    assert len(buf) == 8, f"expected eviction to 8, got {len(buf)}"
    s = buf.sample(4)
    assert len(s["input"]) == 4


def t_ttt_forward():
    from synapforge.self_learn import TestTimeTraining
    model = Toy()
    ttt = TestTimeTraining(model, inner_lr=1e-3, steps=2)
    x = torch.randint(0, 16, (2, 6), dtype=torch.long)
    loss = ttt.adapt(x, restore_after=True)
    assert isinstance(loss, float)


def t_self_learn_engine():
    from synapforge.self_learn import SelfLearnEngine
    model = Toy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        eng = SelfLearnEngine(model, replay_size=64, ttt_steps=1)
        assert any("DefenseStack" in str(x.message) for x in w), "must warn when defense missing"
    # observe a few continuous samples (use float so MSE works cleanly)
    for _ in range(5):
        eng.observe(torch.randn(2, 32), torch.randn(2, 32), reward=1.0)
    assert len(eng.replay) == 5
    # train_step
    for _ in range(10):
        loss = eng.train_step((torch.randn(2, 32), torch.randn(2, 32)))
        assert loss == loss  # not NaN
    # adapt
    eng.adapt(torch.randint(0, 16, (1, 4), dtype=torch.long), restore_after=True)


def t_defense_stack():
    from synapforge.defense import DefenseConfig, DefenseStack
    model = Toy()
    cfg = DefenseConfig(red_team_check_every=1, red_team_kl_max=1e6)
    canaries = [torch.randint(0, 16, (1, 4), dtype=torch.long) for _ in range(2)]
    stack = DefenseStack(model, cfg, canary_prompts=canaries)
    assert stack.accept(torch.randn(2, 32), source="test")
    assert stack.accept("hello world this is normal text", source="test")
    # Highly repetitive -> should reject
    assert not stack.accept("aaaa aaaa aaaa aaaa aaaa aaaa aaaa", source="bad")
    # Lifecycle
    stack.pre_step()
    # Simulate a fake grad
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    z = stack.after_grads()
    assert z >= 0
    stack.regressed_and_rollback()
    stack.tick()


def t_defense_fires_on_bad_update():
    """Inject huge weight perturbation -> KL drift -> rollback."""
    from synapforge.defense import DefenseConfig, DefenseStack
    model = Toy()
    cfg = DefenseConfig(red_team_check_every=1, red_team_kl_max=0.001)
    canaries = [torch.randint(0, 16, (1, 4), dtype=torch.long) for _ in range(2)]
    stack = DefenseStack(model, cfg, canary_prompts=canaries)
    # Take initial snapshot
    stack.red_team.snapshot()
    # Perturb every weight massively
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 5.0)
    # Force-check should detect regression and rollback
    fired = stack.force_check()
    assert fired, "red team should have detected drift"


def t_self_learn_with_defense():
    """Self-learn + defense end-to-end (no warn since defense attached)."""
    from synapforge.defense import DefenseConfig, DefenseStack
    from synapforge.self_learn import SelfLearnEngine
    model = Toy()
    canaries = [torch.randint(0, 16, (1, 4), dtype=torch.long) for _ in range(2)]
    defense = DefenseStack(model, DefenseConfig(red_team_check_every=5), canary_prompts=canaries)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        eng = SelfLearnEngine(model, replay_size=32, defense=defense)
        # MUST NOT warn (defense supplied)
        assert not any("DefenseStack" in str(x.message) for x in w), "should not warn when defense attached"
    for i in range(10):
        eng.observe(torch.randn(2, 32), torch.randn(2, 32), reward=1.0, source="data")
        eng.train_step((torch.randn(2, 32), torch.randn(2, 32)))
    # Provenance tracker recorded all 10 observations
    assert len(defense.tracker.recent(100)) >= 10


def t_tool_registry():
    from synapforge.tools import ToolRegistry
    reg = ToolRegistry()
    reg.add_default_tools(web_search_mock=True, web_fetch_mock=True)
    assert reg.has("web_search")
    assert reg.has("web_fetch")
    assert reg.has("web_scrape")
    out = reg.call("web_search", {"query": "test"})
    assert isinstance(out, list) and out, f"expected mock list, got {out}"
    fetch = reg.call("web_fetch", {"url": "https://example.com"})
    assert "text" in fetch


def t_tool_caller_neural():
    from synapforge.tools import ToolCaller, ToolRegistry
    reg = ToolRegistry().add_default_tools(web_search_mock=True, web_fetch_mock=True)
    caller = ToolCaller(hidden=32, registry=reg, execute=False)
    h = torch.randn(2, 32)
    out = caller(h)
    # Every tool name has a selection scalar
    for name in reg.names():
        assert name in out
        assert out[name].shape == torch.Size([2])
    # _selection sums to <= 1 per row (gated by sigmoid).
    sel = out["_selection"]
    assert sel.shape == (2, len(reg))
    # Test execute mode actually calls the tool.
    caller_exec = ToolCaller(hidden=32, registry=reg, execute=True)
    out2 = caller_exec(h, args_by_tool={"web_search": {"query": "x"}})
    assert "_executed" in out2
    assert "_result_vec" in out2


def t_tool_learning_loop():
    from synapforge.tools import ToolCaller, ToolLearningLoop, ToolRegistry
    reg = ToolRegistry().add_default_tools(web_search_mock=True, web_fetch_mock=True)
    caller = ToolCaller(hidden=32, registry=reg)
    loop = ToolLearningLoop(caller)
    h = torch.randn(2, 32)
    # signal>0 => pushes prototype toward h via -log(sel)
    loss = loop.reinforce_loss(h, "web_search", 0.5)
    loss.backward()
    rec = loop.evaluate("web_search", 0.3, args={"query": "ok"})
    assert rec["outcome"] == "success"


def t_fgsm_pgd():
    from synapforge.adversarial import FastGradientSignAttack, ProjectedGradientDescentAttack
    model = Toy()
    fgsm = FastGradientSignAttack(model, epsilon=0.05)
    x = torch.randn(2, 32)
    y = torch.randn(2, 32)
    x_adv = fgsm.call(x, y)
    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max().item() <= 0.05 + 1e-6
    pgd = ProjectedGradientDescentAttack(model, epsilon=0.05, alpha=0.01, steps=3)
    x_adv2 = pgd.call(x, y)
    assert (x_adv2 - x).abs().max().item() <= 0.05 + 1e-5


def t_adv_trainer():
    from synapforge.adversarial import AdversarialTrainer, AdversarialTrainerCfg
    model = Toy()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    adv = AdversarialTrainer(model, opt, AdversarialTrainerCfg(pgd_steps=2, adv_weight=0.3))
    losses = []
    for _ in range(10):
        l = adv.train_step(torch.randn(2, 32), torch.randn(2, 32))
        losses.append(l)
    assert all(l == l for l in losses)  # no NaN


def t_distill():
    from synapforge.distill import DistillationLoss, DistillConfig, DistillTrainer
    student = Toy()
    teacher = Toy()  # frozen teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    cfg = DistillConfig(use_cache=False)  # in-memory only for the smoke
    dt = DistillTrainer(student, teacher, opt, cfg)
    losses = []
    for _ in range(10):
        x = torch.randint(0, 16, (2, 4), dtype=torch.long)
        y = torch.randint(0, 16, (2, 4), dtype=torch.long)
        losses.append(dt.train_step(x, y))
    assert all(l == l for l in losses)
    # DistillationLoss direct
    loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)
    s = torch.randn(4, 16)
    t = torch.randn(4, 16)
    h = torch.randint(0, 16, (4,), dtype=torch.long)
    out = loss_fn(s, t, hard_target=h)
    assert out.item() > 0


def t_full_stack_integration():
    """SelfLearnEngine + DefenseStack + tool dispatch all together."""
    from synapforge.defense import DefenseConfig, DefenseStack
    from synapforge.self_learn import SelfLearnEngine
    from synapforge.tools import ToolCaller, ToolRegistry

    model = Toy()
    canaries = [torch.randint(0, 16, (1, 4), dtype=torch.long) for _ in range(2)]
    defense = DefenseStack(model, DefenseConfig(red_team_check_every=3), canary_prompts=canaries)
    eng = SelfLearnEngine(model, replay_size=64, defense=defense)

    reg = ToolRegistry().add_default_tools(web_search_mock=True, web_fetch_mock=True)
    caller = ToolCaller(hidden=32, registry=reg)

    # Inject "poisoned" string sample — defense should reject
    accepted = defense.accept("a a a a a a a a a a a a", source="web")
    assert not accepted, "obvious poison must be rejected"

    # Real samples accepted
    eng.observe(torch.randn(2, 32), torch.randn(2, 32), reward=1.0, source="real")
    # End-to-end train step (no NaN)
    for _ in range(5):
        eng.train_step((torch.randn(2, 32), torch.randn(2, 32)))

    # Tool dispatch on a hidden state from the model.
    fake_hidden = torch.randn(2, 32)
    out = caller(fake_hidden)
    assert "_selection" in out


# ---------- driver --------------------------------------------------------


def main() -> int:
    print("synapforge round-9 smoke test")
    print("=" * 50)
    # Run tests in dependency order
    cases = [
        ("imports", t_imports),
        ("replay_buffer", t_replay_buffer),
        ("ttt_forward", t_ttt_forward),
        ("self_learn_engine_warns_no_defense", t_self_learn_engine),
        ("defense_stack_basic", t_defense_stack),
        ("defense_fires_on_bad_update", t_defense_fires_on_bad_update),
        ("self_learn_with_defense", t_self_learn_with_defense),
        ("tool_registry", t_tool_registry),
        ("tool_caller_neural", t_tool_caller_neural),
        ("tool_learning_loop", t_tool_learning_loop),
        ("fgsm_pgd", t_fgsm_pgd),
        ("adv_trainer", t_adv_trainer),
        ("distill", t_distill),
        ("full_stack_integration", t_full_stack_integration),
    ]
    for name, fn in cases:
        check(name, fn)
    print("=" * 50)
    print(f"PASS {len(PASSED)}/{len(cases)}    FAIL {len(FAILED)}")
    if FAILED:
        print("\nFailures:")
        for n, msg in FAILED:
            print(f"  - {n}: {msg.splitlines()[0]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
