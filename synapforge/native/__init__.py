"""synapforge.native — native LNN+SNN runtime (no torch.optim).

Subpackages provide bypass paths for the production training stack:

* ``stdp`` — Spike-Timing-Dependent Plasticity local update rule.
  Plasticity-tagged weights skip AdamW entirely and update via a local
  Hebbian rule that costs O(active_spikes), not O(weights).
"""
