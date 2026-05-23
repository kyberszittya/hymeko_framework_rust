"""Render the canonical fixture as a 3D hypergraph: per-vertex
position from a 3-D Fruchterman-Reingold layout, sign-aware
arc colouring carried over from the 2-D grammar of Section 5.

Output: paper/kepaf_v1/figures/canonical_layout_3d.{pdf,png}

The current GPU kernel is 2-D-only; this figure shows that the
data model and rendering grammar lift to 3-D without modification
(positions are just $(x,y,z)$ in the vertex buffer
of Section 3.3). A 3-D GPU kernel is a journal-version follow-up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kepaf_benchmark import fixture_canonical
import networkx as nx

SIGN_COLOR = {1: "#1b6ca8", -1: "#b02a2a", 0: "#888888"}

OUT_DIR = REPO / "paper" / "kepaf_v1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    G, name = fixture_canonical()
    print(f"[{name}] |V|={len(G)} |E|={G.number_of_edges()}")
    pos = nx.spring_layout(G, dim=3, seed=0, iterations=400, k=0.25)

    # Drop outlier vertices: any node whose distance from the bulk
    # centroid exceeds the 70th percentile of distances. Arcs
    # incident to a dropped node are removed too. We trim
    # aggressively to zoom on the centroid so arcs become legible.
    coords = np.asarray(list(pos.values()))
    centre = np.median(coords, axis=0)
    dists  = np.linalg.norm(coords - centre, axis=1)
    cutoff = np.percentile(dists, 70)
    keep   = {n: dists[i] <= cutoff for i, n in enumerate(pos)}
    n_dropped = sum(1 for v in keep.values() if not v)
    print(f"  dropped {n_dropped} outlier nodes "
          f"(cutoff = {cutoff:.3f})")
    pos = {n: p for n, p in pos.items() if keep[n]}
    G_in = G.subgraph([n for n in G.nodes if keep.get(n, False)])

    G, pos = G_in, pos  # rebind for downstream code

    fig = plt.figure(figsize=(5.4, 4.0), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # Vertices vs. hyperedge nodes (Levi graph).
    v_xyz, e_xyz = [], []
    for n, p in pos.items():
        kind = G.nodes[n].get("kind", "vertex")
        (e_xyz if kind == "hyperedge" else v_xyz).append(p)
    v_xyz = np.asarray(v_xyz) if v_xyz else np.zeros((0, 3))
    e_xyz = np.asarray(e_xyz) if e_xyz else np.zeros((0, 3))

    # Arcs, sign-coloured. Thicker + more opaque so they read
    # clearly when the camera is close to the centroid.
    for u, v, d in G.edges(data=True):
        s = d.get("sign", 1)
        c = SIGN_COLOR.get(s, "#888888")
        pu, pv = pos[u], pos[v]
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], [pu[2], pv[2]],
                color=c, lw=1.6, alpha=0.9, zorder=1)

    if len(v_xyz):
        ax.scatter(v_xyz[:, 0], v_xyz[:, 1], v_xyz[:, 2],
                   s=64, c="#EEF1F5", edgecolors="#3a4a5a",
                   linewidths=0.8, zorder=3, label="vertex")
    if len(e_xyz):
        ax.scatter(e_xyz[:, 0], e_xyz[:, 1], e_xyz[:, 2],
                   s=110, marker="s", c="#D7E4F5",
                   edgecolors="#1b6ca8", linewidths=0.8,
                   zorder=4, label="hyperedge")

    # Tight, uniform cube around the (already-filtered) bulk with
    # a small 3 % margin so the centroid graph fills the frame.
    all_xyz = np.concatenate(
        [v_xyz, e_xyz] if len(v_xyz) and len(e_xyz)
        else ([v_xyz] if len(v_xyz) else [e_xyz])
    )
    lo = all_xyz.min(axis=0)
    hi = all_xyz.max(axis=0)
    span = (hi - lo).max() * 1.03
    ctr = 0.5 * (lo + hi)
    ax.set_xlim(ctr[0] - span / 2, ctr[0] + span / 2)
    ax.set_ylim(ctr[1] - span / 2, ctr[1] + span / 2)
    ax.set_zlim(ctr[2] - span / 2, ctr[2] + span / 2)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # Rotated camera: more side-on, slightly lower elevation, so the
    # arcs are seen edge-rather-than-axially.
    ax.view_init(elev=10, azim=120)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out_pdf = OUT_DIR / "canonical_layout_3d.pdf"
    out_png = OUT_DIR / "canonical_layout_3d.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
