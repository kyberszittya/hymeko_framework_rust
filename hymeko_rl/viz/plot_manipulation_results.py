"""Plot the manipulation-learning results (delivery/placement rates) for the README.

Data are the measured results from the 2026-06-24 runs (see reports/2026-06-24-manipulation-learning-report.tex
and results/RESULTS.md). Pure-matplotlib, no torch. Writes PNGs to reports/figures/.

    python -m hymeko_rl.viz.plot_manipulation_results
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_OUT = Path("reports/figures")
_HS, _ML, _BC = "#2a9a4a", "#5566aa", "#cccccc"   # hsikan, mlp, bc


def _bar_points(ax: Axes, x: float, vals: list[float], color: str, width: float = 0.32) -> None:
    """A bar at the mean + the per-seed points scattered on top (so the spread is visible)."""
    ax.bar(x, sum(vals) / len(vals), width=width, color=color, edgecolor="#333", zorder=2)
    ax.scatter([x] * len(vals), vals, color="#222", s=18, zorder=3)


def galambos_bc_ppo() -> Path:
    """Galambos BC vs BC->PPO, HSiKAN vs params-matched MLP (3 seeds; delivery rate)."""
    hs_bc, hs_ppo = [0.04, 0.08, 0.21], [0.21, 0.25, 0.29]
    ml_bc, ml_ppo = [0.08, 0.08, 0.08], [0.17, 0.21, 0.25]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _bar_points(ax, 0.0, hs_bc, _BC)
    _bar_points(ax, 0.4, hs_ppo, _HS)
    _bar_points(ax, 1.4, ml_bc, _BC)
    _bar_points(ax, 1.8, ml_ppo, _ML)
    ax.set_xticks([0.2, 1.6])
    ax.set_xticklabels(["HSiKAN", "MLP (params-matched)"])
    ax.set_ylabel("delivery rate")
    ax.set_ylim(0, 0.6)
    ax.set_title("Galambos: BC -> PPO refine (3 seeds)")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c) for c in (_BC, _HS, _ML)],
              labels=["BC", "->PPO (HSiKAN)", "->PPO (MLP)"], frameon=False, fontsize=9)
    ax.axhline(0.25, ls="--", lw=0.8, color="#999")
    ax.text(2.05, 0.255, "control ceiling ~0.25", fontsize=8, color="#666")
    fig.tight_layout()
    out = _OUT / "galambos_bc_ppo.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def offpolicy_collapse() -> Path:
    """FANUC: off-policy refine collapses the BC clone (pre-warm-start-fix). BC baseline = 0.50."""
    labels = ["BC\nbaseline", "DDPG\nHSiKAN", "DDPG\nMLP", "TD3\nHSiKAN", "TD3\nMLP"]
    vals = [0.50, 0.25, 0.0, 0.0, 0.0]
    colors = [_BC, _HS, _ML, _HS, _ML]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(range(len(vals)), vals, color=colors, edgecolor="#333", zorder=2)
    ax.axhline(0.50, ls="--", lw=0.9, color="#a33")
    ax.text(3.1, 0.515, "BC baseline", fontsize=8, color="#a33")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("placement rate")
    ax.set_ylim(0, 0.62)
    ax.set_title("FANUC: off-policy refine COLLAPSES the BC clone (pre-fix)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = _OUT / "offpolicy_collapse.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def bc_breaks_wall() -> Path:
    """Pure RL (0) vs BC, both tasks - BC breaks the exploration wall; the ceilings differ by task."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = [0, 1]
    ax.bar([xi - 0.2 for xi in x], [0.0, 0.0], width=0.38, color="#ddd", edgecolor="#333",
           label="pure RL (from scratch)")
    ax.bar([xi + 0.2 for xi in x], [0.50, 0.25], width=0.38, color=_HS, edgecolor="#333",
           label="BC (clone demonstrator)")
    ax.set_xticks(x)
    ax.set_xticklabels(["FANUC pick-and-place", "Galambos coin-grasp"])
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 0.62)
    ax.set_title("Behaviour cloning breaks the exploration wall")
    for xi, v in zip(x, [0.50, 0.25]):
        ax.text(xi + 0.2, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = _OUT / "bc_breaks_wall.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    for fn in (galambos_bc_ppo, offpolicy_collapse, bc_breaks_wall):
        print(f"wrote {fn()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
