"""Spatial 2-D figures for the robot communication cliques demo.

Two views:

- ``network_figure`` — bare network: robots as dots, edges colour-coded
  by sign (blue = reliable, red dashed = jammed).
- ``cliques_figure`` — same network with balanced cliques shaded as
  convex hulls, so the "stable communication teams" pop visually.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .cliques import Clique, RobotNetworkBundle


def _edge_xy(pos: np.ndarray, u: int, v: int) -> tuple[list[float], list[float]]:
    return [pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]]


def network_figure(bundle: RobotNetworkBundle,
                      figsize: tuple[float, float] = (7.0, 6.0)):
    """Plain spatial layout — no clique overlay."""
    fig, ax = plt.subplots(figsize=figsize)
    pos = bundle.positions
    edges = bundle.graph.edges
    signs = bundle.graph.signs

    for (u, v), s in zip(edges, signs):
        xs, ys = _edge_xy(pos, int(u), int(v))
        if s == 1:
            ax.plot(xs, ys, color="#1f77b4", lw=1.8, alpha=0.75, zorder=1)
        else:
            ax.plot(xs, ys, color="#d62728", lw=1.8, alpha=0.9,
                       linestyle="--", zorder=1)

    ax.scatter(pos[:, 0], pos[:, 1], s=180, c="#f4d35e",
                 edgecolors="black", linewidths=0.9, zorder=3)
    for i, (x, y) in enumerate(pos):
        ax.annotate(f"r{i}", (x, y), ha="center", va="center",
                       fontsize=7, zorder=4)

    s = bundle.area_size
    ax.set_xlim(-0.05 * s, 1.05 * s)
    ax.set_ylim(-0.05 * s, 1.05 * s)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"{bundle.name}  ·  {bundle.n_robots} robots  ·  "
        f"comm_range={bundle.comm_range:.1f}  ·  "
        f"noise_prob={bundle.noise_prob:.2f}  ·  seed={bundle.seed}"
    )
    ax.grid(alpha=0.3)
    legend = [
        mpatches.Patch(color="#1f77b4", label="reliable link (+)"),
        mpatches.Patch(color="#d62728", label="jammed / lost (−)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def cliques_figure(bundle: RobotNetworkBundle,
                      cliques: list[Clique],
                      figsize: tuple[float, float] = (7.0, 6.0),
                      max_overlay: int = 5):
    """Same as ``network_figure`` plus convex-hull overlays for the
    largest balanced cliques (up to ``max_overlay`` distinct colours).
    """
    fig, ax = plt.subplots(figsize=figsize)
    pos = bundle.positions
    edges = bundle.graph.edges
    signs = bundle.graph.signs

    # Underlying network — faded so the clique shading dominates.
    for (u, v), s in zip(edges, signs):
        xs, ys = _edge_xy(pos, int(u), int(v))
        if s == 1:
            ax.plot(xs, ys, color="#1f77b4", lw=1.6, alpha=0.4, zorder=1)
        else:
            ax.plot(xs, ys, color="#d62728", lw=1.6, alpha=0.6,
                       linestyle="--", zorder=1)

    # Overlay shaded hulls. Tab10 palette so up to 10 cliques fit.
    palette = plt.get_cmap("tab10")
    for ci, clique in enumerate(cliques[:max_overlay]):
        members = clique.members
        if len(members) < 3:
            continue
        # Convex hull only well-defined for ≥3 points; for 3 just close
        # the triangle.
        member_pos = pos[list(members)]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(member_pos)
            poly_pts = member_pos[hull.vertices]
        except Exception:
            # Fallback: just close the polygon by angle around centroid.
            centroid = member_pos.mean(axis=0)
            angles = np.arctan2(member_pos[:, 1] - centroid[1],
                                   member_pos[:, 0] - centroid[0])
            order = np.argsort(angles)
            poly_pts = member_pos[order]
        polygon = mpatches.Polygon(
            poly_pts, closed=True, alpha=0.20,
            facecolor=palette(ci % 10), edgecolor=palette(ci % 10),
            linewidth=2.0, zorder=2,
            label=f"clique #{ci+1}  size={clique.size}",
        )
        ax.add_patch(polygon)

    ax.scatter(pos[:, 0], pos[:, 1], s=180, c="#f4d35e",
                 edgecolors="black", linewidths=0.9, zorder=3)
    for i, (x, y) in enumerate(pos):
        ax.annotate(f"r{i}", (x, y), ha="center", va="center",
                       fontsize=7, zorder=4)

    s = bundle.area_size
    ax.set_xlim(-0.05 * s, 1.05 * s)
    ax.set_ylim(-0.05 * s, 1.05 * s)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    n_shown = min(len(cliques), max_overlay)
    ax.set_title(
        f"{bundle.name}  ·  {n_shown}/{len(cliques)} balanced cliques shown"
    )
    ax.grid(alpha=0.3)
    if cliques:
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


__all__ = ["network_figure", "cliques_figure"]
