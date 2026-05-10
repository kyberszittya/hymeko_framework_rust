#!/usr/bin/env python3
"""Generate the cycle-distribution plot for the Epinions lift-studies
c-sweep.

Data source: /tmp/c_sweep_probe.log produced by
hymeko_graph/examples/probe_adaptive_c_sweep.rs at
c ∈ {0.25, 0.5, 1.0, 2.0, 4.0, 8.0}.

Usage:
    python reports/figures/lift_studies_c_sweep_plot.py

Output:
    reports/figures/lift_studies_c_sweep.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Inline data from the c-sweep probe (CPU-only, Epinions, k=4, m_max=128,
# balance pruner, fraction_negative scorer).
c = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
cycles = np.array([364051, 615931, 978090, 1480447, 2110025, 2793736])
covered_v = np.array([51295, 51295, 51295, 51295, 51295, 51295])  # constant
full_rate = np.array([0.3846, 0.3701, 0.3444, 0.3108, 0.2755, 0.2377])
score_0 = np.array([26353, 52472, 117030, 268007, 549147, 936569])
score_0_5 = np.array([137733, 259485, 449856, 693208, 931199, 1117721])
score_1_0 = np.array([199965, 303974, 411204, 519232, 629679, 739446])
mean_score = np.array([0.7384, 0.7042, 0.6504, 0.5848, 0.5191, 0.4647])

# Smoke-test AUC results from the prior degree-adaptive report
# (abbreviated config: c3+c4, h=4, 20 ep).  These are 5 of the 6
# probe-c values (no smoke at c=0.25 yet).
smoke_c = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
smoke_auc = np.array([0.7088, 0.6950, 0.6606, 0.6648, 0.6632])
smoke_baseline_auc = 0.6416  # per-vertex m=128


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    # ── Panel 1: Mean cycle score vs c (the headline finding) ───────────────
    ax = axes[0, 0]
    ax.plot(c, mean_score, "-o", color="C0", lw=2.0, ms=9, zorder=5,
             label="mean fraction_negative score per cycle")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.6,
                label="score = 0.5 (2-negative balanced)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("c (slope of $m_v$ in degree)")
    ax.set_ylabel("Mean retained cycle score")
    ax.set_title(
        "Cycle signal density vs $c$\n"
        "smaller $c$ → smaller per-vertex caps → only top-fraction-neg cycles kept"
    )
    ax.set_ylim(0.40, 0.80)
    # Annotate
    for i, (ci, ms) in enumerate(zip(c, mean_score)):
        ax.annotate(f"{ms:.3f}", (ci, ms), textcoords="offset points",
                     xytext=(7, 7), fontsize=8.5)
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(loc="lower left", fontsize=9)

    # ── Panel 2: Cycle pool size vs c ───────────────────────────────────────
    ax = axes[0, 1]
    width = 0.10
    xs = np.arange(len(c))
    p0 = ax.bar(xs - width, score_0, width, color="C7", label="score 0.0 (all-pos)")
    p1 = ax.bar(xs, score_0_5, width, color="C1", label="score 0.5 (2-neg)")
    p2 = ax.bar(xs + width, score_1_0, width, color="C3", label="score 1.0 (all-neg)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"c={ci}" for ci in c])
    ax.set_ylabel("Cycle count by score bin")
    ax.set_title("Cycle pool composition vs $c$\n(stacked: balanced score categories)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, ls=":", axis="y")
    # Annotate total
    for i, total in enumerate(cycles):
        ax.annotate(f"Σ={total:,}", (xs[i], score_1_0[i] + 50000),
                     ha="center", fontsize=8)

    # ── Panel 3: Full-heap rate vs c ────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(c, full_rate, "-D", color="C2", lw=2.0, ms=9,
             label="full-heap rate (vertices reaching $m_v[v]$)")
    # Reference: structurally reachable vertices = 51295 / 131828 ≈ 0.389
    structural_max = 51295 / 131828
    ax.axhline(structural_max, color="purple", lw=1.0, ls=":",
                label=f"structural max ({structural_max:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("c (slope)")
    ax.set_ylabel("Fraction of vertices with full heap")
    ax.set_title(
        "Per-vertex heap saturation vs $c$\n"
        "smaller $c$ pushes all reachable vertices to fill"
    )
    ax.set_ylim(0.20, 0.42)
    for i, (ci, fr) in enumerate(zip(c, full_rate)):
        ax.annotate(f"{fr:.3f}", (ci, fr), textcoords="offset points",
                     xytext=(7, -12), fontsize=8.5)
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(loc="lower left", fontsize=9)

    # ── Panel 4: AUC predictor — smoke AUC vs mean cycle score ──────────────
    ax = axes[1, 1]
    # Map smoke-c to mean_score by index alignment (c=1.0,2.0,4.0,8.0 in both)
    smoke_c_in_probe_idx = {1.0: 2, 2.0: 3, 4.0: 4, 8.0: 5}
    smoke_xs = []
    smoke_ys = []
    smoke_labels = []
    for sc, sa in zip(smoke_c, smoke_auc):
        if sc in smoke_c_in_probe_idx:
            ms = mean_score[smoke_c_in_probe_idx[sc]]
            smoke_xs.append(ms)
            smoke_ys.append(sa)
            smoke_labels.append(f"c={sc}")
    ax.scatter(smoke_xs, smoke_ys, color="C0", s=80, zorder=5,
                label="smoke AUC vs probe mean score")
    for x, y, lab in zip(smoke_xs, smoke_ys, smoke_labels):
        ax.annotate(lab, (x, y), textcoords="offset points",
                     xytext=(7, 5), fontsize=9)
    # Regress (linear) and extrapolate to c=0.25 and c=0.5 mean scores
    if len(smoke_xs) >= 2:
        slope, intercept = np.polyfit(smoke_xs, smoke_ys, 1)
        xs_fit = np.linspace(min(mean_score), max(mean_score), 100)
        ys_fit = slope * xs_fit + intercept
        ax.plot(xs_fit, ys_fit, "--", color="C0", alpha=0.6,
                 label=f"fit AUC = {slope:.2f}·mean+{intercept:.2f}")
        # Predict for c=0.25 and c=0.5
        for ci, idx in [(0.25, 0), (0.5, 1)]:
            pred_auc = slope * mean_score[idx] + intercept
            ax.scatter([mean_score[idx]], [pred_auc], color="C3", s=120,
                        marker="*", zorder=6,
                        label=(f"predicted c={ci}: AUC≈{pred_auc:.3f}"
                                if ci == 0.25 else None))
            ax.annotate(f"c={ci}\npred AUC≈{pred_auc:.3f}",
                          (mean_score[idx], pred_auc),
                          textcoords="offset points", xytext=(-65, -25),
                          fontsize=9, color="C3",
                          arrowprops=dict(arrowstyle="->", color="C3", lw=0.9))
    ax.axhline(smoke_baseline_auc, color="black", lw=1.0, ls=":",
                label=f"per-vertex baseline AUC ({smoke_baseline_auc})")
    ax.set_xlabel("Mean retained cycle score (from probe)")
    ax.set_ylabel("Single-seed AUC (abbreviated config)")
    ax.set_title(
        "Predictor: smoke AUC vs cycle signal density\n"
        "linear fit → extrapolated AUC at lower $c$"
    )
    ax.grid(alpha=0.25, ls=":")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out = Path(__file__).parent / "lift_studies_c_sweep.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
