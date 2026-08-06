"""matplotlib + NetworkX figures for the kinematic demo.

The kinematic graph (links = vertices, joints = edges) is small enough
(≤ 100 links for any in-repo URDF) that a 2-D NetworkX layout is fine.
Use spring layout — fast, deterministic with a seed, and gives a
recognisable spatial intuition for tree / chain / loopy mechanisms.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .kinematic import KinematicBundle


def kinematic_graph_figure(bundle: KinematicBundle,
                              figsize: tuple[float, float] = (7.0, 6.0),
                              seed: int = 0):
    """2-D layout of the kinematic graph. Returns a matplotlib Figure.

    Edge colour = joint type (blue = rotational, red = prismatic).
    Node labels are link names; truncated if very long.
    """
    G = nx.Graph()
    for i, name in enumerate(bundle.link_names):
        G.add_node(i, label=name)
    for (u, v), s, j in zip(
        bundle.graph.edges, bundle.graph.signs, bundle.joints,
    ):
        G.add_edge(int(u), int(v), sign=int(s), kind=j.joint_type)

    fig, ax = plt.subplots(figsize=figsize)
    if bundle.n_links == 0:
        ax.text(0.5, 0.5, "(empty kinematic graph)",
                ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    pos = nx.spring_layout(G, seed=seed, k=1.0 / np.sqrt(max(bundle.n_links, 1)))
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] == +1]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] == -1]
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges,
                              edge_color="#1f77b4", width=2.0, ax=ax,
                              label="rotational")
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges,
                              edge_color="#d62728", width=2.0, ax=ax,
                              style="dashed", label="prismatic")
    nx.draw_networkx_nodes(G, pos, node_color="#f4d35e",
                              edgecolors="black", linewidths=0.8,
                              node_size=240, ax=ax)
    # Truncate long names.
    labels = {i: (n[:14] + "…" if len(n) > 15 else n)
              for i, n in enumerate(bundle.link_names)}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    ax.set_title(
        f"{bundle.name}  ·  {bundle.n_links} links  ·  "
        f"{bundle.n_joints} joints  ({bundle.n_revolute} rev, "
        f"{bundle.n_prismatic} pris)"
    )
    ax.set_axis_off()
    if pos_edges or neg_edges:
        ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def cycle_arity_figure(bundle: KinematicBundle,
                         figsize: tuple[float, float] = (5.5, 3.5)):
    """Bar chart of cycle counts per arity (k = 3 .. max).

    Zero-count bars rendered greyed-out so an "open chain → no bars at
    all" mechanism is visually distinct from a 4-bar (k=4 spike) or a
    Stewart platform (k=6 spike).
    """
    fig, ax = plt.subplots(figsize=figsize)
    if not bundle.cycle_counts:
        ax.text(0.5, 0.5, "(no cycle data)", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig
    ks = sorted(bundle.cycle_counts.keys())
    counts = [bundle.cycle_counts[k] for k in ks]
    colours = ["#4472C4" if c > 0 else "#cccccc" for c in counts]
    ax.bar([str(k) for k in ks], counts, color=colours,
              edgecolor="black", linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("cycle arity (k)")
    ax.set_ylabel("# cycles")
    ax.set_title(f"Cycle-arity profile — {bundle.name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


__all__ = ["kinematic_graph_figure", "cycle_arity_figure"]
