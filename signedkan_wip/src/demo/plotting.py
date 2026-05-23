"""Matplotlib + NetworkX helpers for the demo. UI-agnostic — returns
``matplotlib.figure.Figure`` objects so any frontend (Gradio, Streamlit,
Jupyter) can display them.
"""
from __future__ import annotations

from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # headless backend; safe inside a GUI server
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .inference import PredictionResult


# ─── ROC curve ──────────────────────────────────────────────────────


def roc_figure(pred: PredictionResult, title: str = "") -> "plt.Figure":
    fpr, tpr = pred.roc_curve_xy
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {pred.auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, alpha=0.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title or "ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ─── Per-arity αₖ bar chart ─────────────────────────────────────────


def alpha_figure(
    alpha: np.ndarray | None,
    labels: list[str] | None = None,
    title: str = "Learned α_κ over tuple types",
) -> "plt.Figure":
    fig, ax = plt.subplots(figsize=(6, 4))
    if alpha is None:
        ax.text(0.5, 0.5, "no αₖ exposed by this model",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="grey")
        ax.set_axis_off()
        return fig
    n = len(alpha)
    xs = np.arange(n)
    if labels is None or len(labels) != n:
        labels = [f"k{i}" for i in range(n)]
    colours = ["#4472C4" if lbl.startswith("c") else "#ED7D31"
               for lbl in labels]
    bars = ax.bar(xs, alpha, color=colours, edgecolor="black", linewidth=0.5)
    for b, a in zip(bars, alpha):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.6, float(alpha.max()) * 1.2))
    ax.set_ylabel("α (softmax weight)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ─── Subgraph viz around a selected edge ────────────────────────────


def subgraph_figure(
    edges: np.ndarray,           # (E, 2) full graph edges
    signs: np.ndarray,           # (E,)   full graph signs
    focus_u: int,
    focus_v: int,
    pred_prob: float | None = None,
    true_sign: int | None = None,
    radius: int = 1,
    max_nodes: int = 60,
    title: str = "",
) -> "plt.Figure":
    """Render a NetworkX subgraph centred on edge (focus_u, focus_v).

    Includes all vertices within ``radius`` hops of either endpoint
    (capped at ``max_nodes`` by degree-descending pruning to keep the
    figure readable on dense graphs).
    """
    G = nx.Graph()
    for (u, v), s in zip(edges, signs):
        G.add_edge(int(u), int(v), sign=int(s))

    # BFS subgraph around the focus edge.
    seeds = {int(focus_u), int(focus_v)}
    keep: set[int] = set(seeds)
    frontier = set(seeds)
    for _ in range(radius):
        next_front = set()
        for v in frontier:
            if v not in G:
                continue
            for nb in G.neighbors(v):
                if nb not in keep:
                    next_front.add(nb)
        keep |= next_front
        frontier = next_front

    # If too dense, keep highest-degree-in-subgraph nodes.
    if len(keep) > max_nodes:
        deg_in_sub = {n: G.degree(n) for n in keep}
        keep_sorted = sorted(keep, key=lambda n: -deg_in_sub[n])
        keep = set(keep_sorted[:max_nodes]) | seeds

    H = G.subgraph(keep).copy()

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(H, seed=0, k=0.6, iterations=50)

    # Edges coloured by sign; focus edge highlighted thicker.
    pos_edges = [(u, v) for u, v, d in H.edges(data=True) if d["sign"] == 1]
    neg_edges = [(u, v) for u, v, d in H.edges(data=True) if d["sign"] == -1]
    nx.draw_networkx_edges(H, pos, edgelist=pos_edges,
                           edge_color="#4472C4", width=1.2, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(H, pos, edgelist=neg_edges,
                           edge_color="#C00000", width=1.2, alpha=0.7,
                           style="dashed", ax=ax)

    # Highlight the focus edge.
    focus_edges = []
    if H.has_edge(int(focus_u), int(focus_v)):
        focus_edges.append((int(focus_u), int(focus_v)))
    if focus_edges:
        nx.draw_networkx_edges(H, pos, edgelist=focus_edges,
                               edge_color="black", width=3.5, ax=ax)

    # Nodes: focus endpoints highlighted.
    other = [n for n in H.nodes if n not in seeds]
    nx.draw_networkx_nodes(H, pos, nodelist=other,
                           node_size=80, node_color="#FFFFFF",
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=list(seeds),
                           node_size=300, node_color="#FFC000",
                           edgecolors="black", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(H, pos, labels={n: str(n) for n in seeds},
                            font_size=10, font_weight="bold", ax=ax)

    # Title with prediction info.
    bits = [f"edge ({focus_u}, {focus_v})"]
    if true_sign is not None:
        bits.append(f"true={'+' if true_sign == 1 else '−'}")
    if pred_prob is not None:
        bits.append(f"p(+)={pred_prob:.3f}")
        ps = "+" if pred_prob > 0.5 else "−"
        bits.append(f"pred={ps}")
    info = "   ".join(bits)
    ax.set_title(f"{title}\n{info}" if title else info)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


__all__ = ["roc_figure", "alpha_figure", "subgraph_figure"]
