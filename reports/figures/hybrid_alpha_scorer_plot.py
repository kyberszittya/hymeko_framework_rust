#!/usr/bin/env python3
"""Generate the gate-function plot for the hybrid α-sweep report.

Data source: 1-seed Epinions abbreviated-config α-sweep results
recorded in `reports/2026-05-10-hybrid-alpha-scorer.md` §3.

Usage:
    python reports/figures/hybrid_alpha_scorer_plot.py

Output:
    reports/figures/hybrid_alpha_scorer_gates.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Data (from the α-sweep) ──────────────────────────────────────────────────
alpha = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
auc = np.array([0.5424, 0.5794, 0.5206, 0.5335, 0.5755])
f1m = np.array([0.2745, 0.5543, 0.5124, 0.5256, 0.5338])
wall_s = np.array([35, 61, 45, 46, 58])

# Baselines
PER_VERTEX_BASELINE = 0.6416
GATE_BAND = 0.02  # ±0.02 around the per-vertex baseline
GATE_LOWER = PER_VERTEX_BASELINE - GATE_BAND
ABB_AUC = 0.5755          # global ABB single-seed (prior report)
ENTROPY_AUC = 0.5396      # pure entropy single-seed (prior report)
PER_VERTEX_F1 = 0.5518
PER_VERTEX_WALL = 148.6


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # ── Panel 1: AUC vs α with gate band ─────────────────────────────────────
    ax = axes[0]
    # Gate band
    ax.axhspan(
        GATE_LOWER,
        PER_VERTEX_BASELINE + GATE_BAND,
        color="green",
        alpha=0.10,
        label=f"smoke gate band (±{GATE_BAND:.2f})",
    )
    ax.axhline(GATE_LOWER, color="green", lw=1.0, ls="--",
                label=f"gate threshold = {GATE_LOWER:.4f}")
    # Per-vertex baseline reference
    ax.axhline(PER_VERTEX_BASELINE, color="black", lw=1.5, ls=":",
                label=f"per-vertex m=128 baseline ({PER_VERTEX_BASELINE:.4f})")
    # Other reference points (single-α equivalents from prior reports)
    ax.scatter([1.0], [ABB_AUC], color="orange", s=70, zorder=4,
                marker="s", label=f"global ABB (≈α=1, prior) {ABB_AUC:.4f}")
    ax.scatter([0.0], [ENTROPY_AUC], color="purple", s=70, zorder=4,
                marker="^", label=f"pure entropy (α=0, prior) {ENTROPY_AUC:.4f}")
    # The hybrid α-sweep itself
    ax.plot(alpha, auc, "-o", color="C0", lw=2.0, ms=8, zorder=5,
             label="hybrid α-sweep AUC")
    # Annotate the best α
    best_idx = int(np.argmax(auc))
    ax.annotate(
        f"best α={alpha[best_idx]:.2f}\nAUC={auc[best_idx]:.4f}",
        xy=(alpha[best_idx], auc[best_idx]),
        xytext=(alpha[best_idx] + 0.10, auc[best_idx] + 0.030),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    )
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(0.50, 0.66)
    ax.set_xlabel("α (signal weight)")
    ax.set_ylabel("AUC")
    ax.set_title("Smoke gate: AUC vs α\n(Epinions 1-seed, c3+c4, h=4, 20 ep)")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    ax.grid(alpha=0.25, ls=":")

    # ── Panel 2: Macro-F1 vs α ───────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(PER_VERTEX_F1, color="black", lw=1.5, ls=":",
                label=f"per-vertex baseline ({PER_VERTEX_F1:.2f})")
    ax.plot(alpha, f1m, "-D", color="C2", lw=2.0, ms=8,
             label="hybrid α-sweep F1")
    # Highlight the F1 jump from α=0 to α=0.25
    ax.annotate(
        "F1 doubles with\nsmall signal injection",
        xy=(alpha[1], f1m[1]),
        xytext=(0.25, 0.30),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="C2", lw=0.9),
    )
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(0.20, 0.62)
    ax.set_xlabel("α (signal weight)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Decision-boundary calibration: F1 vs α")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25, ls=":")

    # ── Panel 3: Wall time vs α ──────────────────────────────────────────────
    ax = axes[2]
    ax.axhline(PER_VERTEX_WALL, color="black", lw=1.5, ls=":",
                label=f"per-vertex baseline ({PER_VERTEX_WALL:.0f}s)")
    ax.bar(alpha, wall_s, width=0.18, color="C1", alpha=0.85,
            edgecolor="black", lw=0.8, label="hybrid α-sweep wall")
    for i, w in enumerate(wall_s):
        ax.annotate(f"{w}s", (alpha[i], w + 2), ha="center", fontsize=8.5)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0, max(PER_VERTEX_WALL, max(wall_s)) * 1.10)
    ax.set_xlabel("α (signal weight)")
    ax.set_ylabel("End-to-end wall time (s)")
    ax.set_title("Speed: wall time vs α")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25, ls=":", axis="y")

    plt.tight_layout()
    out = Path(__file__).parent / "hybrid_alpha_scorer_gates.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
