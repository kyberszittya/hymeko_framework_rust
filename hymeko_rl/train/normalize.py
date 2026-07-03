r"""Running reward normalisation for off-policy targets.

The off-policy divergence (NaN/denormal blow-up on the quadruped / large-reward tasks) is a Q-scale explosion:
big rewards (e.g. galambos `in_zone +10`) + the $\gamma$-bootstrap drive the critic's target unbounded, and
gradient clipping bounds the *step* but not the *value*. Normalising the reward by a running RMS keeps the
discounted return $O(1/(1-\gamma))$ instead of unbounded, so targets stay in a numerically safe range. The eval
return is still measured on the raw env reward — only the training target is normalised.
"""
from __future__ import annotations

import torch


class RunningRMS:
    """Online root-mean-square of a scalar reward stream; ``normalize(r) = r / (rms + eps)``.

    # Invariants ``rms >= eps > 0`` always (safe to divide); the estimate is over all rewards seen.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self._n: int = 0
        self._mean_sq: float = 0.0
        self.eps: float = eps

    def update(self, x: torch.Tensor) -> None:
        v = x.detach().reshape(-1)
        n = int(v.numel())
        if n == 0:
            return
        self._mean_sq = (self._mean_sq * self._n + float((v * v).sum())) / (self._n + n)
        self._n += n

    @property
    def rms(self) -> float:
        val: float = max(self._mean_sq ** 0.5, self.eps)
        return val

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Update the running estimate from ``x`` and return ``x / rms`` (the bounded-scale reward)."""
        self.update(x)
        out: torch.Tensor = x / self.rms
        return out
