"""Action-space perturbation utilities for LOCAL_ACTION_RANKING_FIDELITY.

Perturb a frozen deterministic policy by a *tiny* action offset and expose it through the same ``action_mean``
interface the evaluator uses, so the physical rollout can measure whether a small step the critic *prefers* (raises Q)
actually *helps physically*. Two perturbation families:

* **actuator-basis** — a fixed ``eps`` along ``+/- e_axis`` (a uniform push on one actuator);
* **critic-gradient** — the per-state, norm-``eps`` step along ``dQ/da`` at ``a = pi_0(obs)`` (the local one-step move
  the critic would induce on the actor).

Plus a scipy-free Spearman rank correlation for the ranking-fidelity read (critic ΔQ ranking vs physical ranking).
"""
from __future__ import annotations

import numpy as np
import torch

ACTION_SCALE = 4.0


class PerturbedActor:
    """Wrap a frozen actor: ``action_mean(obs) = clip(base(obs) + delta, ±scale)``.

    ``delta`` is either a fixed ``(action_dim,)`` tensor (actuator-basis direction) or a callable
    ``obs -> (..., action_dim)`` (state-dependent, e.g. the critic-gradient direction).

    Preconditions:  ``base.action_mean(obs)`` returns ``(..., action_dim)``; a fixed ``delta`` matches ``action_dim``.
    Postconditions: output clamped to ``±scale``; ``delta = 0`` reproduces ``base`` exactly; no base param is mutated.
    """

    def __init__(self, base, delta, scale: float = ACTION_SCALE) -> None:
        self.base = base
        self.delta = delta
        self.scale = float(scale)

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.base.action_mean(obs)
        d = self.delta(obs) if callable(self.delta) else torch.as_tensor(self.delta, dtype=a.dtype)
        return torch.clamp(a + d, -self.scale, self.scale)


def actuator_basis_delta(action_dim: int, axis: int, sign: int, eps: float) -> torch.Tensor:
    """Fixed unit-actuator perturbation: ``eps`` along ``sign * e_axis``. ``||delta|| == eps``."""
    v = torch.zeros(action_dim)
    v[axis] = float(sign) * float(eps)
    return v


def critic_grad_delta(base, critic, eps: float):
    """State-dependent perturbation of norm ``eps`` along the critic action-gradient ``dQ/da`` at ``a = base(obs)``.

    This is the local one-step move the critic would induce on the actor — following it is exactly what a TD3 actor
    update does locally, so its physical outcome is the decision-relevant fidelity signal.
    """

    def f(obs: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            a = base.action_mean(obs).detach().requires_grad_(True)
            q = critic.min_q(obs, a).sum()
            (g,) = torch.autograd.grad(q, a)
        n = g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (eps * g / n).detach()

    return f


# ── scipy-free Spearman (average-rank tie handling) ──────────────────────────────────────────────────────────────
def _rank_avg(a: np.ndarray) -> np.ndarray:
    """Average ranks (scipy ``rankdata('average')`` equivalent)."""
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    is_new = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = is_new.cumsum()[inv]                          # 1-based dense group index per element
    group_start = np.r_[np.nonzero(is_new)[0], len(a)]    # start position of each group + sentinel
    # average of the 0-based positions occupied by the element's tie-group
    return 0.5 * (group_start[dense - 1] + group_start[dense] - 1)


def spearman(x, y):
    """Spearman rank correlation. Returns ``None`` for <2 points or a constant vector (no discriminating signal —
    the caller must exclude these rather than treat a degenerate state as correlation 0)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or len(x) != len(y):
        return None
    rx, ry = _rank_avg(x), _rank_avg(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(values, stat=np.median, n_boot=2000, seed=12345):
    """Percentile bootstrap CI of ``stat`` over a 1-D sample (here: per-state statistics). ``None`` entries are dropped
    (degenerate states). Deterministic given ``seed``. Returns {stat, lo, hi, n}."""
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 0:
        return {"stat": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = stat(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return {"stat": round(float(stat(v)), 3), "lo": round(float(np.percentile(boots, 2.5)), 3),
            "hi": round(float(np.percentile(boots, 97.5)), 3), "n": int(len(v))}


def eps_from_drifts(drift_p95, cap):
    """Pre-registered ε-selection for the local-ranking test: {p50, p90, p99} of the empirical accepted per-step p95
    anchor action-drift + the trust cap (largest safe single-action deviation). Falls back to just the cap when no
    accepted steps were observed. Returns ``(sorted unique epsilons, info)`` — never arbitrary constants."""
    d = np.asarray(drift_p95, float)
    if len(d) == 0:
        return [round(float(cap), 4)], {"n_accepted": 0, "source": "trust-cap-fallback", "trust_cap_step_max": cap}
    p50, p90, p99 = (float(np.percentile(d, q)) for q in (50, 90, 99))
    eps = sorted(set(round(float(e), 4) for e in [p50, p90, p99, cap] if e > 1e-5))
    return eps, {"n_accepted": int(len(d)), "p50": round(p50, 4), "p90": round(p90, 4), "p99": round(p99, 4),
                 "trust_cap_step_max": cap, "source": "empirical-accepted-drift"}


def hierarchical_bootstrap_ci(per_seed_values, stat=np.median, n_boot=3000, seed=777):
    """Two-level bootstrap so no single training seed drives the verdict: resample SEEDS with replacement, then STATES
    within each chosen seed, pool, apply ``stat``. ``per_seed_values`` is a list (one entry per seed) of per-state value
    lists (``None`` entries = degenerate states, dropped). Returns {stat, lo, hi, n_seeds, n_states}."""
    seeds = [np.asarray([x for x in s if x is not None], float) for s in per_seed_values]
    seeds = [s for s in seeds if len(s) > 0]
    if not seeds:
        return {"stat": None, "lo": None, "hi": None, "n_seeds": 0, "n_states": 0}
    pooled = np.concatenate(seeds)
    rng = np.random.default_rng(seed); boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = [seeds[j] for j in rng.integers(0, len(seeds), len(seeds))]
        draw = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in chosen])
        boots[b] = stat(draw)
    return {"stat": round(float(stat(pooled)), 3), "lo": round(float(np.percentile(boots, 2.5)), 3),
            "hi": round(float(np.percentile(boots, 97.5)), 3), "n_seeds": len(seeds), "n_states": int(len(pooled))}
