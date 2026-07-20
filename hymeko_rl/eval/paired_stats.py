"""Canonical paired-bootstrap statistics for matched CONTROL vs TREATMENT experiments.

The repository accumulated ~20 ad-hoc percentile-bootstrap snippets across experiment scripts (a §6.5 #3 duplication).
This is the single reusable primitive: a percentile bootstrap CI over a 1-D sample, and a paired-delta summary
(mean / median / IQR / sign-split / 95% CI) for matched-pair analyses (n-step, F11/F12, …). Pure and deterministic
given the seed — the matched-pair analyzers should import these rather than re-roll their own bootstrap.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def boot_ci(x: Sequence[float], seed: int, b: int = 10_000, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """Percentile bootstrap CI of the **mean** of ``x``.

    # Preconditions ``len(x) >= 1`` and ``0 <= lo < hi <= 100``.
    # Postconditions returns ``(lo_pct, hi_pct)`` of the resampled-mean distribution, each rounded to 3 dp.
    # Invariants deterministic in ``seed``; ``x`` is not mutated.
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("boot_ci needs at least one sample")
    if not 0.0 <= lo < hi <= 100.0:
        raise ValueError(f"require 0 <= lo < hi <= 100; got lo={lo}, hi={hi}")
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(b, a.size))].mean(axis=1)
    return round(float(np.percentile(means, lo)), 3), round(float(np.percentile(means, hi)), 3)


def paired_stats(deltas: Sequence[float], seed: int, b: int = 10_000) -> dict[str, object]:
    """Summary of paired TREATMENT−CONTROL ``deltas``: values, n, mean, median, IQR, +/0/− split, bootstrap 95% CI.

    # Preconditions ``len(deltas) >= 1``. # Postconditions ``boot95`` is ``[lo, hi]``; sign counts partition ``n``.
    """
    d = np.asarray(deltas, dtype=float)
    ci = boot_ci(d, seed, b)
    return dict(values=[round(float(v), 3) for v in d], n=int(d.size), mean=round(float(d.mean()), 3),
                median=round(float(np.median(d)), 3),
                iqr=round(float(np.percentile(d, 75) - np.percentile(d, 25)), 3),
                pos=int((d > 0).sum()), zero=int((d == 0).sum()), neg=int((d < 0).sum()), boot95=[ci[0], ci[1]])
