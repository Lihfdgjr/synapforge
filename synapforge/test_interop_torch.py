"""Tests for sf.interop_torch — verify bidirectional bridge with vanilla nn.Module.

Run: CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python -m pytest \\
        /workspace/synapforge/test_interop_torch.py -v
or as a script: /opt/conda/bin/python /workspace/synapforge/test_interop_torch.py
"""
from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace")

import synapforge as sf
from synapforge.interop_torch import (
    SFAsTorchModule,
    TorchAsSFModule,
    convert_sparse_to_linear,
    replace_linear_with_sparse,
    replace_relu_with_plif,
)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


# -------- 1. nn.Linear inside an sf.Module ------------------------------------


def test_torch_linear_inside_sf_module_runs():
    """sf.Module containing a vanilla nn.Linear: forward + backward + AdamW."""
    class Hybrid(sf.Module):
        def __init__(self, d):
            super().__init__()
            self.linear = nn.Linear(d, d)
            self.plif = sf.PLIF(d, threshold=0.3)

        def forward(self, x):
            h = self.linear(x)
            spk, mem = self.plif(h)
            return spk

    m = Hybrid(32).to(DEV)
    x = torch.randn(2, 4, 32, device=DEV)
    y = m(x)
    assert y.shape == (2, 4, 32)
    loss = y.float().sum()
    loss.backward()
    assert m.linear.weight.grad is not None
    assert torch.isfinite(loss)


# -------- 2. sf.HybridBlock as nn.Module under torch.optim.AdamW --------------


def _build_hybrid():
    class Hybrid(sf.Module):
        def __init__(self, d):
            super().__init__()
            self.cfc = sf.LiquidCell(d, d)
            self.plif = sf.PLIF(d, threshold=0.3)
            self.proj = nn.Linear(d, d)

        def forward(self, x):
            h = self.cfc(x)
            spk, mem = self.plif(h)
            return self.proj(spk)
    return Hybrid


def test_sf_hybrid_under_torch_adamw():
    """Wrap sf.Module via SFAsTorchModule; verify torch.optim.AdamW iterates."""
    Hybrid = _build_hybrid()
    m = SFAsTorchModule(Hybrid(32)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        x = torch.randn(2, 4, 32, device=DEV)
        y = m(x)
        loss = (y.float() ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    # Module must still expose its sf-specific attributes.
    assert hasattr(m, "plasticity_step")


def test_sf_hybrid_inside_nn_sequential():
    """Stack sf cells inside nn.Sequential; runs forward/backward."""
    seq = nn.Sequential(
        nn.Linear(32, 32),
        sf.SparseSynapse(32, 32, sparsity=0.2),
    ).to(DEV)
    x = torch.randn(2, 32, device=DEV)
    y = seq(x)
    assert y.shape == (2, 32)
    y.sum().backward()
    # Check both layers got a gradient.
    assert seq[0].weight.grad is not None
    assert seq[1].weight.grad is not None


# -------- 3. Vanilla nn.Module wrapped as sf.Module ---------------------------


def test_torch_as_sf_module_basic():
    """Vanilla nn.Linear becomes an sf.Module; can have plasticity registered."""
    inner = nn.Linear(16, 16).to(DEV)
    sf_view = TorchAsSFModule(inner)
    assert isinstance(sf_view, sf.Module)
    x = torch.randn(2, 16, device=DEV)
    y = sf_view(x)
    assert y.shape == (2, 16)


def test_torch_as_sf_module_save_load_roundtrip():
    """torch.save / torch.load works on a wrapped vanilla module."""
    inner = nn.Linear(8, 8)
    sf_view = TorchAsSFModule(inner)
    sd = sf_view.state_dict()
    sf_view2 = TorchAsSFModule(nn.Linear(8, 8))
    incompat = sf_view2.load_state_dict(sd, strict=True)
    # state_dict roundtrip should produce no missing/unexpected keys.
    if isinstance(incompat, tuple):
        assert not incompat[0] and not incompat[1]


# -------- 4. replace_linear_with_sparse ---------------------------------------


def test_replace_linear_with_sparse_basic():
    """Linear -> SparseSynapse; forward shape preserved; weights copied."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(32, 8)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    net = Net().to(DEV)
    orig_w_fc1 = net.fc1.weight.detach().clone()
    replaced = replace_linear_with_sparse(net, density=0.5, copy_weights=True)
    assert set(replaced) == {"fc1", "fc2"}
    assert isinstance(net.fc1, sf.SparseSynapse)
    assert isinstance(net.fc2, sf.SparseSynapse)
    # Weights copied bit-equal.
    assert torch.equal(net.fc1.weight, orig_w_fc1)
    # Forward still has the right shape.
    x = torch.randn(4, 16, device=DEV)
    y = net(x)
    assert y.shape == (4, 8)
    # Backward populates grads on the new SparseSynapse params only.
    y.sum().backward()
    assert net.fc1.weight.grad is not None


def test_replace_linear_with_sparse_skip_module():
    """skip_modules keeps named entries vanilla."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(8, 8)
            self.lm_head = nn.Linear(8, 8)

        def forward(self, x):
            return self.lm_head(self.body(x))

    net = Net().to(DEV)
    replaced = replace_linear_with_sparse(net, density=0.3, skip_modules=["lm_head"])
    assert "lm_head" not in replaced
    assert isinstance(net.body, sf.SparseSynapse)
    assert isinstance(net.lm_head, nn.Linear)


# -------- 5. SparseSynapse round-trip back to Linear --------------------------


def test_convert_sparse_to_linear_preserves_output():
    """SparseSynapse -> Linear roundtrip; forward output bit-exact (after mask)."""
    syn = sf.SparseSynapse(16, 8, sparsity=0.3, bias=True).to(DEV)
    parent = nn.Sequential(syn).to(DEV)
    x = torch.randn(2, 16, device=DEV)
    y_before = parent(x).detach().clone()

    convert_sparse_to_linear(parent)
    assert isinstance(parent[0], nn.Linear)
    y_after = parent(x).detach()
    assert torch.allclose(y_before, y_after, atol=1e-5)


# -------- 6. BertModel-style: replace nn.Linear in HF model -------------------


def test_replace_linear_in_bert_runs():
    """HuggingFace BertModel has many nn.Linear; replace + forward must work."""
    transformers = pytest.importorskip("transformers")
    bert_cfg = transformers.BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    bert = transformers.BertModel(bert_cfg).to(DEV)
    n_linear_before = sum(1 for _, m in bert.named_modules() if isinstance(m, nn.Linear))
    assert n_linear_before > 0
    replaced = replace_linear_with_sparse(bert, density=0.5, copy_weights=True)
    assert len(replaced) == n_linear_before
    # No Linear left, only SparseSynapse.
    n_linear_after = sum(1 for _, m in bert.named_modules() if isinstance(m, nn.Linear))
    n_sparse = sum(1 for _, m in bert.named_modules() if isinstance(m, sf.SparseSynapse))
    assert n_linear_after == 0
    assert n_sparse == n_linear_before
    # Forward still runs.
    input_ids = torch.randint(0, 128, (2, 16), device=DEV)
    out = bert(input_ids=input_ids)
    assert out.last_hidden_state.shape == (2, 16, 32)


# -------- 7. replace_relu_with_plif (warns but works) -------------------------


def test_replace_relu_with_plif_warns_and_runs():
    """ReLU swap emits a warning; resulting module forwards to a tensor."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    net = Net().to(DEV)
    with pytest.warns(UserWarning):
        replaced = replace_relu_with_plif(net, threshold=0.5, verbose=True)
    assert replaced == ["relu"]
    x = torch.randn(2, 8, device=DEV)
    y = net(x)
    assert y.shape == (2, 8)


if __name__ == "__main__":
    # Lightweight runner for environments without pytest installed.
    failures = 0
    for name in [n for n in dir(sys.modules[__name__]) if n.startswith("test_")]:
        try:
            print(f"\n=== {name} ===", flush=True)
            globals()[name]()
            print(f"  PASS", flush=True)
        except Exception as exc:
            print(f"  FAIL: {exc!r}", flush=True)
            import traceback
            traceback.print_exc()
            failures += 1
    print(f"\nSummary: {failures} failure(s)", flush=True)
    sys.exit(1 if failures else 0)
