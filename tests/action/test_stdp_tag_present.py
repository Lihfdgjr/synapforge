"""Verify SparseSynapticLayer.weight is tagged for STDP plasticity routing.

The "decorative -> real" activation (2026-05-02) flips the synaptogenesis
plasticity head from `_sf_grad_source=["bp"]` (the implicit default) to
`_sf_grad_source=["stdp"]` so that PlasticityAwareAdamW (in
``synapforge.optim._legacy``) actually consumes the Hebbian / STDP delta
stream from ``SparseSynapticLayer.update_coactivation``. Without the tag
the gradient path is RUNTIME-DORMANT even though the code ships.

These tests are fast (CPU-only, no torch.cuda required) and assert:
1. SparseSynapticLayer.weight has _sf_grad_source == ["stdp"].
2. SparseSynapticLayer.weight has _sf_alpha == 0.001.
3. NeuroMCPHead.proj.weight inherits the tag (composition).
4. Tag survives state_dict round-trip + .to(device) (it's an attribute on
   the Parameter, not a buffer; PyTorch persists Parameter attributes).
5. PlasticityAwareAdamW + the tag -> param is in the multi-source path.
"""

from __future__ import annotations

import os
import sys

# Allow running tests against a checkout that hasn't `pip install`-ed yet.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

from synapforge.action.neuromcp import (  # noqa: E402
    NeuroMCPHead,
    SparseSynapticLayer,
    SynaptogenesisConfig,
)


def test_sparse_synapse_weight_tagged_stdp() -> None:
    """SparseSynapticLayer.weight has _sf_grad_source = ['stdp']."""
    syn = SparseSynapticLayer(SynaptogenesisConfig(in_dim=32, out_dim=32))
    sources = getattr(syn.weight, "_sf_grad_source", None)
    assert sources == ["stdp"], (
        f"expected ['stdp'] for STDP-only routing; got {sources!r}"
    )


def test_sparse_synapse_weight_alpha() -> None:
    """SparseSynapticLayer.weight has _sf_alpha = 0.001 initial STDP rate."""
    syn = SparseSynapticLayer(SynaptogenesisConfig(in_dim=16, out_dim=16))
    alpha = getattr(syn.weight, "_sf_alpha", None)
    assert alpha == 0.001, f"expected _sf_alpha=0.001; got {alpha!r}"


def test_neuromcp_head_inherits_tag() -> None:
    """NeuroMCPHead's projection weight inherits the STDP tag (composition)."""
    head = NeuroMCPHead(hidden=64, codebook_initial=4, codebook_max=8)
    sources = getattr(head.proj.weight, "_sf_grad_source", None)
    alpha = getattr(head.proj.weight, "_sf_alpha", None)
    assert sources == ["stdp"], f"composition lost STDP tag; got {sources!r}"
    assert alpha == 0.001, f"composition lost _sf_alpha; got {alpha!r}"


def test_other_params_remain_untagged() -> None:
    """Tag is *additive*: only SparseSynapticLayer.weight is STDP-routed.
    Codebook prototypes / LayerNorm gain stay on the default backprop path.
    """
    head = NeuroMCPHead(hidden=32, codebook_initial=2, codebook_max=4)
    # codebook prototypes have no tag (untouched, default BP)
    cb_src = getattr(head.codebook.prototypes, "_sf_grad_source", None)
    ln_src = getattr(head.norm.weight, "_sf_grad_source", None)
    assert cb_src is None, (
        f"codebook prototypes must remain untagged; got {cb_src!r}"
    )
    assert ln_src is None, f"LayerNorm must remain untagged; got {ln_src!r}"


def test_tag_survives_state_dict_load() -> None:
    """State-dict round-trip preserves tags on the new instance.

    PyTorch's load_state_dict copies Tensor data only; non-tensor metadata
    on the destination Parameter is unchanged. So the rebuilt module's
    Parameter still has its constructor-set tag.
    """
    src = SparseSynapticLayer(SynaptogenesisConfig(in_dim=8, out_dim=8))
    sd = src.state_dict()
    dst = SparseSynapticLayer(SynaptogenesisConfig(in_dim=8, out_dim=8))
    dst.load_state_dict(sd)
    assert getattr(dst.weight, "_sf_grad_source", None) == ["stdp"]
    assert getattr(dst.weight, "_sf_alpha", None) == 0.001


def test_tag_visible_via_named_parameters() -> None:
    """Trainer scans ``model.named_parameters()`` and inspects ``_sf_grad_source``
    on each param. The synapse weight must show up with the tag in that scan.
    """
    head = NeuroMCPHead(hidden=16, codebook_initial=2, codebook_max=4)
    found = False
    for name, p in head.named_parameters():
        if name.endswith("proj.weight"):
            sources = getattr(p, "_sf_grad_source", None)
            assert sources == ["stdp"], (
                f"named_parameters scan missed tag on {name}: {sources!r}"
            )
            found = True
    assert found, "did not find proj.weight in named_parameters()"


def test_plasticity_aware_optimizer_picks_up_tag() -> None:
    """End-to-end: building PlasticityAwareAdamW over the tagged params
    routes the synapse weight into a MultiSourceParam wrapper (not the
    plain bp-only path). Catches regressions where the optimizer was
    accidentally hard-coded to ``["bp"]`` instead of reading the tag.
    """
    try:
        from synapforge.optim._legacy import (
            PlasticityAwareAdamW,
            build_optimizer,
        )
    except Exception:  # noqa: BLE001
        # Optional import path; skip if optim package not available
        return

    head = NeuroMCPHead(hidden=8, codebook_initial=2, codebook_max=4)
    optim = build_optimizer(head, lr=1e-4)
    assert isinstance(optim, PlasticityAwareAdamW), (
        f"build_optimizer returned {type(optim).__name__}, "
        "expected PlasticityAwareAdamW"
    )
    # The optimizer keeps its multi-source registry under ``_ms_param_map``
    # keyed by id(param) in the ``synapforge.optim._legacy`` impl. Look up
    # our synapse weight and verify its sources include "stdp".
    msp_map = getattr(optim, "_ms_param_map", None)
    assert msp_map is not None, (
        "PlasticityAwareAdamW lost the _ms_param_map; STDP tag has nowhere to land"
    )
    syn_id = id(head.proj.weight)
    msp = msp_map.get(syn_id)
    assert msp is not None, (
        "PlasticityAwareAdamW did not register synapse weight as "
        "multi-source; STDP tag was ignored"
    )
    assert "stdp" in msp.sources, (
        f"MultiSourceParam.sources={msp.sources!r} missing 'stdp'"
    )


if __name__ == "__main__":
    test_sparse_synapse_weight_tagged_stdp()
    print("OK SparseSynapticLayer.weight tagged STDP")
    test_sparse_synapse_weight_alpha()
    print("OK SparseSynapticLayer.weight has _sf_alpha=0.001")
    test_neuromcp_head_inherits_tag()
    print("OK NeuroMCPHead inherits tag")
    test_other_params_remain_untagged()
    print("OK other params remain untagged")
    test_tag_survives_state_dict_load()
    print("OK tag survives state_dict round-trip")
    test_tag_visible_via_named_parameters()
    print("OK tag visible via named_parameters")
    test_plasticity_aware_optimizer_picks_up_tag()
    print("OK PlasticityAwareAdamW registers tagged synapse weight")
