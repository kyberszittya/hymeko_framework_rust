"""Task-independent physical-evaluation contract for option-RL: checkpoint-wise paired baseline comparison, pre-registered
selection, solved-set tracking, seed-disjoint panels. The task supplies a per-state paired evaluator; the engine supplies
the (frozen, pre-registered) *decision rule* so a lucky checkpoint cannot win by chance.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, seed: int = 12345):
    """Percentile bootstrap CI of ``stat`` over a 1-D sample (drops None). Deterministic given ``seed``. {stat, lo, hi, n}.
    Framework-canonical general helper (task modules may keep their own legacy copies)."""
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 0:
        return {"stat": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return {"stat": float(stat(v)), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "n": len(v)}


def paired_delta(rl_per_state, base_per_state):
    """Per-state paired difference (RL − its own baseline). Both are per-state success fractions on the SAME panel/seeds."""
    return [float(r) - float(b) for r, b in zip(rl_per_state, base_per_state)]


def preregistered_select(candidates, boot_seed: int = 0):
    """PRE-REGISTERED checkpoint selection. Each candidate: {name, rl_dev (per-state), base_dev (per-state), exit_dev (mean)}.
    Decision order (frozen): (1) mean paired ΔK6 vs own baseline → (2) bootstrap-CI lower bound → (3) −any_exit → then stable.
    Returns the winning candidate dict augmented with its dev stats. A lucky point estimate cannot win a CI tie."""
    scored = []
    for c in candidates:
        d = paired_delta(c["rl_dev"], c["base_dev"]); ci = bootstrap_ci(d, seed=boot_seed)
        scored.append({**c, "dev_delta": round(float(np.mean(d)), 4), "dev_delta_lo": round(ci["lo"], 4), "exit_dev": round(float(c.get("exit_dev", 0.0)), 4)})
    scored.sort(key=lambda x: (x["dev_delta"], x["dev_delta_lo"], -x["exit_dev"]), reverse=True)
    return scored[0], scored


def paired_final_score(rl_final, base_final, boot_seed: int = 1):
    """Score the selected checkpoint on the UNTOUCHED final panel: paired ΔK6 with bootstrap CI + solved-sets."""
    d = paired_delta(rl_final, base_final); ci = bootstrap_ci(d, seed=boot_seed)
    return {"final_rl": round(float(np.mean(rl_final)), 4), "final_base": round(float(np.mean(base_final)), 4),
            "final_delta": round(float(np.mean(d)), 4), "final_delta_lo": round(ci["lo"], 4), "final_delta_hi": round(ci["hi"], 4),
            "solved_rl": [i for i, v in enumerate(rl_final) if v > 0.5], "solved_base": [i for i, v in enumerate(base_final) if v > 0.5]}


def across_seed_summary(per_seed_deltas, per_seed_lo):
    """Median/IQR of the selected-checkpoint final ΔK6 across seeds + how many seeds have CI-lower>0 (the reproducibility bar)."""
    d = np.asarray(per_seed_deltas, float)
    return {"n_seeds": len(d), "median": round(float(np.median(d)), 4),
            "iqr": [round(float(np.percentile(d, 25)), 4), round(float(np.percentile(d, 75)), 4)],
            "seeds_ci_lower_gt0": int(sum(x > 0 for x in per_seed_lo)), "per_seed": [round(float(x), 4) for x in d]}
