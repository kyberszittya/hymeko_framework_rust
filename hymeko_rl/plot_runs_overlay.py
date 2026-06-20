"""Overlay the Galambos PPO budget runs to read convergence: training-reward trajectories on one
axis, and the held-out goal-rate (common test on the current env) versus budget with Wilson CIs.

The runs are *independent* PPO runs at increasing iteration budget on an evolving env/reward, NOT
snapshots of one trajectory, so the goal-rate panel is the honest cross-run comparison (same eval
env, same held-out seeds) — and it is not monotonic.

    uv run python -m hymeko_rl.plot_runs_overlay
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_CKPT = Path("checkpoints/galambos")
_OUT = Path("reports/gifs/galambos_convergence.png")


@dataclass(frozen=True)
class Run:
    """One PPO run: its checkpoint stem, display label, and measured held-out goal rate."""
    stem: str
    iters: int
    label: str
    goal_pct: float
    ci_lo: float
    ci_hi: float


# Measured 2026-06-21: eval_planar_grasp -n 100 --seed0 3000 on the current env (fair common test).
_RUNS = (
    Run("ppo_noclash", 150, "150 iters", 20.0, 13.3, 28.9),
    Run("ppo_noclash_long", 400, "400 iters", 13.0, 7.8, 21.0),
    Run("ppo_long1k", 1000, "1000 iters", 27.0, 19.3, 36.4),
)


def _ema(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average for a legible trend line.

    Preconditions: ``0 < alpha <= 1``; ``x`` is 1-D.
    Postconditions: returns an array the same shape as ``x``.
    """
    assert 0.0 < alpha <= 1.0
    out = np.empty_like(x, dtype=float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _returns(stem: str) -> np.ndarray:
    data = json.loads((_CKPT / f"{stem}.json").read_text(encoding="utf-8"))
    return np.asarray(data["returns"], dtype=float)


def build(out: Path = _OUT) -> Path:
    """Render the two-panel convergence figure. Postconditions: writes ``out`` (PNG), returns it."""
    fig, (axr, axg) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(_RUNS)))

    for run, c in zip(_RUNS, colors):
        r = _returns(run.stem)
        it = np.arange(1, len(r) + 1)
        axr.plot(it, r, color=c, alpha=0.18, lw=0.8)
        axr.plot(it, _ema(r), color=c, lw=2.0, label=f"{run.label} (n={len(r)})")
    axr.set_xlabel("PPO iteration")
    axr.set_ylabel("mean episodic return")
    axr.set_title("Training reward (raw + EMA)")
    axr.legend(loc="lower right", fontsize=9)
    axr.grid(alpha=0.25)

    budgets = [run.iters for run in _RUNS]
    pct = [run.goal_pct for run in _RUNS]
    lo = [run.goal_pct - run.ci_lo for run in _RUNS]
    hi = [run.ci_hi - run.goal_pct for run in _RUNS]
    axg.errorbar(budgets, pct, yerr=[lo, hi], fmt="o-", color="#b5179e", lw=2,
                 capsize=5, markersize=8)
    for run in _RUNS:
        axg.annotate(f"{run.goal_pct:.0f}%", (run.iters, run.goal_pct),
                     textcoords="offset points", xytext=(8, 8), fontsize=10)
    axg.set_xscale("log")
    axg.set_xticks(budgets)
    axg.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axg.set_xlabel("iteration budget (log)")
    axg.set_ylabel("held-out goal rate (%)")
    axg.set_title("Goal rate vs budget — common test (100 seeds, 95% Wilson CI)")
    axg.set_ylim(0, 45)
    axg.grid(alpha=0.25)

    fig.suptitle("Galambos planar grasp — PPO convergence (independent budget runs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"wrote {build()}")
