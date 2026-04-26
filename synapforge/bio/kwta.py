"""kWTA — top-k winner-take-all sparse activation.

Cortex follows a 2-10% active rule (Olshausen & Field 1996; Olshausen
2003).  We enforce it explicitly: keep the top-k values per row, zero
out the rest.  Straight-through gradient lets backprop reach all
positions so the *which-k* choice can change next step.
"""

from __future__ import annotations

import torch

from ..module import Module


class KWTA(Module):
    """Top-k winner-take-all sparsification along the last axis.

    Args:
        fraction:        fraction of units to keep active per row, in (0, 1].
                         Default 0.10 (10% active, cortex-typical).
        straight_through: when True, backward pass passes gradient through
                         the un-masked input (lets unselected units learn
                         to compete).  When False, gradient is zero at
                         masked-out positions.

    Example:
        >>> import synapforge as sf
        >>> kwta = sf.bio.KWTA(fraction=0.05)
        >>> y = kwta(torch.randn(8, 256))   # 5% units survive per row
        >>> (y == 0).float().mean()         # ~0.95
    """

    def __init__(self, fraction: float = 0.10, straight_through: bool = True):
        super().__init__()
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = float(fraction)
        self.straight_through = bool(straight_through)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fraction >= 1.0:
            return x
        last = x.size(-1)
        k = max(1, int(last * self.fraction))
        # topk ignores ties deterministically with sorted=False
        topk = x.topk(k, dim=-1, sorted=False)
        # Mask in same dtype as x so bf16 path stays bf16.
        mask = torch.zeros_like(x).scatter_(-1, topk.indices, 1.0)
        if self.straight_through:
            # Forward: x * mask (sparse).  Backward: grad through (x - x.detach())
            # passes a unit gradient at masked positions while the kept
            # positions still receive their normal gradient via x * mask.
            return x * mask + (x - x.detach()) * (1.0 - mask)
        return x * mask

    def extra_repr(self) -> str:
        return f"fraction={self.fraction}, straight_through={self.straight_through}"


__all__ = ["KWTA"]
