#!/usr/bin/env python3
"""Generate the gate-function plot for the degree-adaptive m_v c-sweep
report.

Data source: 1-seed Epinions abbreviated-config c-sweep results
recorded at /tmp/adaptive_mv_sweep_2026_05_10/results.tsv (or
inlined below if the file is unavailable).

Usage:
    python reports/figures/degree_adaptive_mv_plot.py

Output:
    reports/figures/degree_adaptive_mv_gates.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Baselines (from prior reports + the smoke gate)
PER_VERTEX_BASELINE_AUC = 0.6416
PER_VERTEX_BASELINE_WALL = 148.6
GATE_BAND = 0.02
GATE_LOWER = PER_VERTEX_BASELINE_AUC - GATE_BAND
PER_VERTEX_BASELINE_F1 = 0.5518


def load_results():
    """Load c-sweep TSV produced by the orchestration script.
    Falls back to None if the file isn't present."""
    tsv = Path("/tmp/adaptive_mv_sweep_2026_05_10/results.tsv")
    if not tsv.exists():
        return None
    rows = []
    with tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows.append({
                    "c": float(row["c"]),
                    "wall_s": float(row["wall_s"]),
                    "auc": float(row["auc"]),
                    "f1m": float(row["f1m"]),
                })
            except (ValueError, KeyError):
                continue
    if not rows:
        return None
    rows.sort(key=lambda r: r["c"])
    return {
        "c": np.array([r["c"] for r in rows]),
        "wall_s": np.array([r["wall_s"] for r in rows]),
        "auc": np.array([r["auc"] for r in rows]),
        "f1m": np.array([r["f1m"] for r in rows]),
    }


def main() -> None:
    data = load_results()
    if data is None:
        raise SystemExit(
            "no /tmp/adaptive_mv_sweep_2026_05_10/results.tsv found; "
            "run signedkan_wip/experiments/run_adaptive_mv_sweep_2026_05_10.sh first"
        )

    c = data["c"]
    auc = data["auc"]
    f1m = data["f1m"]
    wall_s = data["wall_s"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # ── Panel 1: AUC vs c with gate band ─────────────────────────────────────
    ax = axes[0]
    ax.axhspan(
        GATE_LOWER, PER_VERTEX_BASELINE_AUC + GATE_BAND,
        color="green", alpha=0.10,
        label=f"smoke gate band (±{GATE_BAND:.2f})",
    )
    ax.axhline(GATE_LOWER, color="green", lw=1.0, ls="--",
                label=f"gate threshold = {GATE_LOWER:.4f}")
    ax.axhline(PER_VERTEX_BASELINE_AUC, color="black", lw=1.5, ls=":",
                label=f"per-vertex m=128 baseline ({PER_VERTEX_BASELINE_AUC:.4f})")
    ax.plot(c, auc, "-o", color="C0", lw=2.0, ms=8, zorder=5,
             label="degree-adaptive m_v sweep")
    best_idx = int(np.argmax(auc))
    ax.annotate(
        f"best c={c[best_idx]:.0f}\nAUC={auc[best_idx]:.4f}",
        xy=(c[best_idx], auc[best_idx]),
        xytext=(c[best_idx] * 1.5 + 0.5, auc[best_idx] + 0.03),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("c (slope of $m_v$ in degree)")
    ax.set_ylabel("AUC")
    ax.set_title("Smoke gate: AUC vs $c$\n(Epinions 1-seed, c3+c4, h=4, 20 ep)")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    ax.grid(alpha=0.25, ls=":", which="both")

    # ── Panel 2: F1 vs c ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(PER_VERTEX_BASELINE_F1, color="black", lw=1.5, ls=":",
                label=f"per-vertex baseline ({PER_VERTEX_BASELINE_F1:.2f})")
    ax.plot(c, f1m, "-D", color="C2", lw=2.0, ms=8,
             label="degree-adaptive m_v sweep")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("c (slope)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Decision-boundary calibration: F1 vs $c$")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25, ls=":", which="both")

    # ── Panel 3: Wall time vs c ──────────────────────────────────────────────
    ax = axes[2]
    ax.axhline(PER_VERTEX_BASELINE_WALL, color="black", lw=1.5, ls=":",
                label=f"per-vertex baseline ({PER_VERTEX_BASELINE_WALL:.0f}s)")
    ax.bar(np.arange(len(c)), wall_s, width=0.6, color="C1", alpha=0.85,
            edgecolor="black", lw=0.8, label="degree-adaptive sweep")
    ax.set_xticks(np.arange(len(c)))
    ax.set_xticklabels([f"c={int(ci)}" for ci in c])
    for i, w in enumerate(wall_s):
        ax.annotate(f"{int(w)}s", (i, w + max(wall_s) * 0.02),
                     ha="center", fontsize=8.5)
    ax.set_xlim(-0.6, len(c) - 0.4)
    ax.set_ylim(0, max(PER_VERTEX_BASELINE_WALL, max(wall_s)) * 1.15)
    ax.set_xlabel("c (slope)")
    ax.set_ylabel("End-to-end wall time (s)")
    ax.set_title("Speed: wall time vs $c$")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25, ls=":", axis="y")

    plt.tight_layout()
    out = Path(__file__).parent / "degree_adaptive_mv_gates.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
