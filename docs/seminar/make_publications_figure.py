"""Render the publication-portfolio figure for the seminar.

A "one substrate -> many venues" fan: the HyMeKo hypergraph core at the centre,
spokes to each venue family, node colour = submission status. The TPAMI target
is elevated (larger node + gold emphasis ring). Pure matplotlib/numpy. Writes
docs/seminar/figures/publications.png.

Status/venue data is hand-curated (2026-06-16); the T-SMC journal extensions are
independent submissions, NOT children of the SMC 2026 conference. Confirm before
presenting.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "publications.png"

STATUS_COLOR = {
    "submitted": "#1b6ca8",   # submitted / under review
    "prep": "#e08a1e",        # in preparation
    "planned": "#9aa7b4",     # planned
}

# (venue label, status) — venue families around the core.
VENUES = [
    ("IEEE SMC 2026\n(conference + WIP)", "submitted"),
    ("IEEE TPAMI\nstructural priors", "prep"),
    ("Elsevier journal\nAC-HSiKAN", "prep"),
    ("IEEE SISY 2026\n(1 paper)", "submitted"),
    ("IEEE T-SMC: Systems\n(journal, independent)", "prep"),
    ("IEEE T-SMC: Cybernetics\nspectral-entropy reg.", "prep"),
    ("Nature Communications\nleakage audit", "prep"),
    ("MDPI Technologies\nlive demo (built)", "submitted"),
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")

    core = mpatches.FancyBboxPatch(
        (-0.34, -0.16), 0.68, 0.32,
        boxstyle="round,pad=0.02,rounding_size=0.06", fc="#2d2a4a", ec="none", zorder=3)
    ax.add_patch(core)
    ax.text(0, 0, "HyMeKo\none hypergraph substrate", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold", zorder=4)

    n = len(VENUES)
    for i, (label, status) in enumerate(VENUES):
        ang = math.pi / 2 - 2 * math.pi * i / n  # start at top, clockwise
        x, y = 1.02 * math.cos(ang), 0.86 * math.sin(ang)
        col = STATUS_COLOR[status]
        ax.plot([0.34 * math.cos(ang), 0.78 * x], [0.34 * math.sin(ang), 0.78 * y],
                color=col, lw=2.2, zorder=1, alpha=0.8)
        is_flagship = "TPAMI" in label
        hw, hh = (0.33, 0.118) if is_flagship else (0.30, 0.105)
        if is_flagship:
            ring = mpatches.FancyBboxPatch(
                (x - hw - 0.02, y - hh - 0.02), 2 * hw + 0.04, 2 * hh + 0.04,
                boxstyle="round,pad=0.015,rounding_size=0.05",
                fc="none", ec="#d4a017", lw=2.6, zorder=2)
            ax.add_patch(ring)
        node = mpatches.FancyBboxPatch(
            (x - hw, y - hh), 2 * hw, 2 * hh,
            boxstyle="round,pad=0.015,rounding_size=0.04",
            fc=col, ec="none", alpha=0.92, zorder=2)
        ax.add_patch(node)
        ax.text(x, y, label, ha="center", va="center", color="white",
                fontsize=9.0 if is_flagship else 8.5, fontweight="bold", zorder=3)

    legend = [
        mpatches.Patch(color=STATUS_COLOR["submitted"], label="submitted / under review"),
        mpatches.Patch(color=STATUS_COLOR["prep"], label="in preparation"),
        mpatches.Patch(color=STATUS_COLOR["planned"], label="planned"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, -0.06), fontsize=9)
    ax.set_title("Publication portfolio — one substrate, many venues (2026)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}  ({n} venue families)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
