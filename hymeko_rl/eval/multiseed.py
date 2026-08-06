"""Multi-seed evaluation aggregation + a 2-proportion tie-test.

The option-level run's ft_dom "drop" (0.75→0.667) was a 2-episode delta on a single 24-episode eval — not
distinguishable from noise. This module aggregates per-eval-seed metric dicts into mean ± std and runs a
2-proportion z-test (baseline vs candidate) on pooled delivery counts, so a delivery difference is classified
`better` / `tied` / `worse` with a p-value rather than eyeballed. Verifier-agnostic: the caller supplies the
per-seed metric dicts (from `measure_policy(seed0=…)` + `audit_policy(seed0=…)`)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# metrics that are rates over n_eval episodes (a count can be recovered as round(rate*n_eval)) for the z-test
_RATE_KEYS = ("ft_dom", "raw_delivery")


@dataclass
class MultiSeedStats:
    n_seeds: int
    n_eval: int
    mean: dict[str, float]
    std: dict[str, float]
    per_seed: dict[str, list[float]]
    violation_dist: dict[str, int] = field(default_factory=dict)

    def ftdom_count(self) -> tuple[int, int]:
        """Pooled (delivered, total) for the ft_dom rate across all seed batches (for the proportion test)."""
        delivered = int(round(sum(self.per_seed.get("ft_dom", [])) * self.n_eval))
        return delivered, self.n_seeds * self.n_eval

    def summary(self, keys: tuple[str, ...]) -> dict[str, str]:
        return {k: f"{self.mean.get(k, float('nan')):.4f} ± {self.std.get(k, 0.0):.4f}" for k in keys}


def aggregate(per_seed: list[dict], *, n_eval: int) -> MultiSeedStats:
    """Aggregate a list of per-seed metric dicts into mean ± std (numeric keys) + a violation_reason histogram.

    # Preconditions: ``per_seed`` non-empty; each dict shares the numeric keys. # Postconditions: a
      :class:`MultiSeedStats`; non-numeric keys (e.g. ``violation_reason``) are tallied into ``violation_dist``."""
    if not per_seed:
        raise ValueError("aggregate() needs at least one per-seed metric dict")
    numeric_keys = sorted({k for d in per_seed for k, v in d.items() if isinstance(v, (int, float))})
    per: dict[str, list[float]] = {k: [float(d[k]) for d in per_seed if k in d] for k in numeric_keys}
    mean = {k: float(np.mean(v)) for k, v in per.items()}
    std = {k: float(np.std(v)) for k, v in per.items()}
    viol: dict[str, int] = {}
    for d in per_seed:
        r = d.get("violation_reason")
        if isinstance(r, str):
            viol[r] = viol.get(r, 0) + 1
    return MultiSeedStats(n_seeds=len(per_seed), n_eval=n_eval, mean=mean, std=std, per_seed=per, violation_dist=viol)


def two_proportion_ztest(c1: int, n1: int, c2: int, n2: int) -> tuple[float, float]:
    """Two-sided 2-proportion z-test (pooled). Returns (z, p). Degenerate (n≤0 or identical all-0/all-1) → (0, 1)."""
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0
    p1, p2 = c1 / n1, c2 / n2
    p = (c1 + c2) / (n1 + n2)
    se = float(np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)))
    if se < 1e-12:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


@dataclass
class FtdomDecision:
    decision: str          # "better" | "tied" | "worse"
    z: float
    p: float
    base_mean: float
    cand_mean: float
    delta: float

    def as_dict(self) -> dict:
        return {"decision": self.decision, "z": round(self.z, 3), "p": round(self.p, 4),
                "base_mean": round(self.base_mean, 4), "cand_mean": round(self.cand_mean, 4),
                "delta": round(self.delta, 4)}


def compare_ftdom(base: MultiSeedStats, cand: MultiSeedStats, *, alpha: float = 0.05) -> FtdomDecision:
    """Classify the candidate's ft_dom vs baseline via the pooled 2-proportion z-test. p ≥ alpha → 'tied';
    else the sign of (cand − base) gives 'better' / 'worse'."""
    cb, nb = base.ftdom_count()
    cc, nc = cand.ftdom_count()
    z, p = two_proportion_ztest(cc, nc, cb, nb)
    bm, cm = base.mean.get("ft_dom", float("nan")), cand.mean.get("ft_dom", float("nan"))
    if p >= alpha:
        decision = "tied"
    else:
        decision = "better" if cm > bm else "worse"
    return FtdomDecision(decision=decision, z=z, p=p, base_mean=bm, cand_mean=cm, delta=cm - bm)
