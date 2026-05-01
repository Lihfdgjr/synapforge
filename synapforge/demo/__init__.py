"""SynapForge investor demo — CPU-runnable artifacts.

Two pieces:
- `four_button`: NeuroMCP learns to click the right colored square WITHOUT
  emitting any tool-call tokens. Synapses grow from 5% to 28% density as
  the network discovers the action space.
- `rfold_bench`: closed-form R-fold vs sequential CfC, with honest
  per-shape table.

Entry point: `synapforge-demo` (see pyproject.toml).
"""
