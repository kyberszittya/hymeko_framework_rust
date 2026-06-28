"""Figure for the cross-view-consistency report (CLAUDE.md 9: every experiment emits a plotted form).

Two panels:
  (left)  per-fixture cross-view verdict -- EXACT square (urdf/sdf/mjcf full numeric invariant) and TOPOLOGICAL
          square (all 5 views), with link/joint counts; one green cell per fixture x layer when it commutes.
  (right) the storage-overhead regime curve rho(d_bar) = 1 + 2/d_bar, marking the binary-joint robotics regime
          (rho~2) and the high-arity limit (rho->1) -- the honest reframe of Proposition 4.

Reads reports/cross_view_consistency.json (produced by cross_view.py). Writes the PNG/SVG next to the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
JSON = REPO / "reports" / "cross_view_consistency.json"
OUT = REPO / "reports" / "figures" / "cross_view_consistency"


def main() -> int:
    results = json.loads(JSON.read_text(encoding="utf-8"))
    results.sort(key=lambda r: (-r["n_actuated_joints"], r["fixture"]))
    names = [r["fixture"].replace(".hymeko", "") for r in results]
    exact = np.array([1 if r["exact_consistent"] else 0 for r in results])
    topo = np.array([1 if r["topo_consistent"] else 0 for r in results])
    grid = np.vstack([exact, topo]).T  # rows=fixtures, cols=[exact, topo]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0), gridspec_kw={"width_ratios": [1.7, 1.0]})

    cmap = matplotlib.colors.ListedColormap(["#d62728", "#2ca02c"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EXACT\n(urdf/sdf/mjcf)", "TOPOLOGICAL\n(all 5 views)"], fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i, r in enumerate(results):
        ax.text(0, i, "OK" if exact[i] else "X", ha="center", va="center", color="white", fontsize=8, weight="bold")
        ax.text(1, i, "OK" if topo[i] else "X", ha="center", va="center", color="white", fontsize=8, weight="bold")
        ax.text(1.62, i, f"L{r['n_links']} J{r['n_actuated_joints']}", va="center", fontsize=7, color="#333")
    ax.set_title(r"Cross-view square $X_f(\varepsilon_f(H)) = X_g(\varepsilon_g(H))$" "\n"
                 f"{int(exact.sum())}/{len(results)} exact, {int(topo.sum())}/{len(results)} topological "
                 "(real CLI emitters)", fontsize=11)
    ax.set_xlim(-0.5, 2.3)

    d = np.linspace(1.5, 30, 400)
    rho = 1 + 2.0 / d
    ax2.plot(d, rho, color="#1f77b4", lw=2)
    for label, dv in [("robotics\n(binary joints)", 2), ("low-arity", 3), ("high-arity", 20)]:
        ax2.scatter([dv], [1 + 2.0 / dv], color="#d62728", zorder=5)
        ax2.annotate(f"{label}\n" r"$\rho$=" f"{1 + 2.0/dv:.2f}", (dv, 1 + 2.0 / dv),
                     textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.text(22, 1.02, r"$\rho\to1$ (high arity)", fontsize=8, color="gray")
    ax2.set_xlabel(r"mean hyperedge arity $\bar d$")
    ax2.set_ylabel(r"storage ratio $\rho = 1 + 2/\bar d$")
    ax2.set_title("Storage overhead is controlled,\nvanishing for high-arity relations (Prop. 4)", fontsize=11)
    ax2.grid(alpha=0.3)

    fig.suptitle("HyMeKo cross-view consistency — machine-verified against the real emitters", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=140)
    fig.savefig(OUT.with_suffix(".svg"))
    print(f"wrote {OUT.with_suffix('.png').relative_to(REPO)} and .svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
