"""Visualize G-SPHF: a signed-incidence hypergraph and its signed-incidence matrix B.

The existing hg_*.png show *types* of hypergraphs (Fano, sunflower). This shows what makes a
description **G-SPHF** — the *signed* incidence: a small robot-arm kinematic hypergraph (links +
axes as vertices; revolute joints as arity-3 signed hyperedges `(+ parent, - child, - axis)`)
beside its signed-incidence matrix B, the structural component of the GGK 4-tuple K = (B, G, mu, r).

Self-contained (matplotlib only). Run: `uv run python docs/architecture/Diagrams/gen_gsphf_figure.py`
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PUR, BLU, GRN, TEAL, AMB, CORAL, INK, MUT = (
    "#6D28D9", "#2563EB", "#0E9D6E", "#14B8A6", "#E0A21B", "#D8552F", "#201A40", "#6B6690")
POS = "#0E9D6E"   # +1 source
NEG = "#D8552F"   # -1 target

# A 3-joint arm as a signed hypergraph.
# vertices: 4 links + 2 axis references; hyperedges: 3 revolute joints (+ parent, - child, - axis).
VERTS = ["base", "shoulder", "elbow", "wrist", "AXIS_Z", "AXIS_Y"]
EDGES = [
    ("j1", [("base", +1), ("shoulder", -1), ("AXIS_Z", -1)]),
    ("j2", [("shoulder", +1), ("elbow", -1), ("AXIS_Y", -1)]),
    ("j3", [("elbow", +1), ("wrist", -1), ("AXIS_Y", -1)]),
]
EDGE_COL = [PUR, BLU, TEAL]

# planar positions: links along a chain, axes below
POSXY = {
    "base": (0.0, 1.0), "shoulder": (1.3, 1.0), "elbow": (2.6, 1.0), "wrist": (3.9, 1.0),
    "AXIS_Z": (0.65, -0.2), "AXIS_Y": (2.6, -0.4),
}


def _blob(ax, members, col):
    pts = np.array([POSXY[m] for m, _ in members], float)
    c = pts.mean(0)
    th = np.linspace(0, 2 * np.pi, 80)
    # an inflated rounded hull approximation: ellipse around the member centroid sized to span them
    rx = max(0.55, float(np.ptp(pts[:, 0])) / 2 + 0.5)
    ry = max(0.45, float(np.ptp(pts[:, 1])) / 2 + 0.45)
    ax.fill(c[0] + rx * np.cos(th), c[1] + ry * np.sin(th),
            facecolor=col, alpha=0.12, edgecolor=col, lw=2.0, zorder=1)
    return c


def draw_hypergraph(ax):
    ax.set_title("signed hypergraph: a 3-joint arm", fontsize=11, color=INK, pad=8)
    for (name, members), col in zip(EDGES, EDGE_COL):
        c = _blob(ax, members, col)
        ax.text(c[0], c[1] + 0.0, name, fontsize=8, color=col, ha="center", va="center",
                fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=col, lw=1.0))
        for m, sgn in members:
            x, y = POSXY[m]
            ax.text(x + 0.18, y + 0.18, "+" if sgn > 0 else "−",
                    color=POS if sgn > 0 else NEG, fontsize=12, fontweight="bold", zorder=9)
    for nm, (x, y) in POSXY.items():
        is_axis = nm.startswith("AXIS")
        ax.scatter([x], [y], s=240, color="white",
                   edgecolor=AMB if is_axis else INK, linewidth=2.2, zorder=5)
        ax.scatter([x], [y], s=46, color=AMB if is_axis else INK, zorder=6)
        ax.text(x, y - 0.32, nm, fontsize=7.5, color=MUT, ha="center", va="top", zorder=7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.9, 4.8)
    ax.set_ylim(-1.2, 1.9)


def draw_incidence(ax):
    """The signed-incidence matrix B: rows = vertices, cols = hyperedges, entries +1 / -1 / 0."""
    B = np.zeros((len(VERTS), len(EDGES)), int)
    for j, (_n, members) in enumerate(EDGES):
        for m, sgn in members:
            B[VERTS.index(m), j] = sgn
    ax.set_title(r"signed incidence  $B \in \{-1,0,+1\}^{V\times E}$", fontsize=11, color=INK, pad=8)
    nv, ne = B.shape
    for i in range(nv):
        for j in range(ne):
            v = B[i, j]
            col = POS if v > 0 else (NEG if v < 0 else "#EDEBF5")
            ax.add_patch(plt.Rectangle((j, nv - 1 - i), 1, 1, facecolor=col,
                                       edgecolor="white", lw=2.0,
                                       alpha=0.85 if v else 0.6))
            if v:
                ax.text(j + 0.5, nv - 1 - i + 0.5, "+1" if v > 0 else "−1",
                        ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    for j, (n, _m) in enumerate(EDGES):
        ax.text(j + 0.5, nv + 0.18, n, ha="center", va="bottom", fontsize=8.5, color=INK)
    for i, v in enumerate(VERTS):
        ax.text(-0.15, nv - 1 - i + 0.5, v, ha="right", va="center", fontsize=8, color=MUT)
    ax.set_xlim(-1.3, ne + 0.2)
    ax.set_ylim(-0.2, nv + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    fig = plt.figure(figsize=(9.4, 4.2), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.05)
    draw_hypergraph(fig.add_subplot(gs[0]))
    draw_incidence(fig.add_subplot(gs[1]))
    fig.suptitle(
        "G-SPHF — a signed hypergraph; B is the structure of the GGK 4-tuple  K = (B, G, μ, r)",
        fontsize=12.5, color=INK, y=1.02)
    fig.text(0.5, -0.02,
             "+ source  ·  − target/axis   |   B = signed incidence   "
             "G = geometry   μ = measure/weights   r = reference",
             ha="center", fontsize=8.5, color=MUT)
    out = "docs/architecture/Diagrams/gsphf_signed_incidence.png"
    fig.savefig(out, transparent=False, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
