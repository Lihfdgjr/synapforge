"""tests.module — Phase 2 torch-replacement test suite.

See ``docs/TORCH_REPLACEMENT_PLAN.md`` Phase 2 entry. Tests in this
package enforce the Phase 2 contract: bit-exact forward equivalence
between ``synapforge.module.Module``-based blocks and
``torch.nn.Module`` reference implementations, plus state-dict
round-trip with existing torch ckpts.
"""
