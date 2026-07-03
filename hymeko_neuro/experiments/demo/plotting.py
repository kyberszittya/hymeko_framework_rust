"""Matplotlib + NetworkX helpers for the demo. UI-agnostic — returns
``matplotlib.figure.Figure`` objects so any frontend (Gradio, Streamlit,
Jupyter) can display them.
"""
from __future__ import annotations

from typing import Any, Iterable

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


# ─── Signed-balance / frustrated-cycle view ─────────────────────────


def _cycle_elements(
    cycles: Iterable[Iterable[int]],
) -> tuple[set[int], set[tuple[int, int]]]:
    """Vertices and undirected edges spanned by a set of cycles."""
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for cyc in cycles:
        seq = [int(x) for x in cyc]
        nodes.update(seq)
        for j in range(len(seq)):
            a, b = seq[j], seq[(j + 1) % len(seq)]
            edges.add((min(a, b), max(a, b)))
    return nodes, edges


def _prune_for_legibility(
    G: "nx.Graph", max_nodes: int, must_keep: set[int],
) -> "nx.Graph":
    """Subgraph of the ``max_nodes`` highest-degree vertices plus ``must_keep``."""
    if G.number_of_nodes() <= max_nodes:
        return G
    deg = dict(G.degree())
    ranked = sorted(G.nodes, key=lambda n: -deg[n])[:max_nodes]
    return G.subgraph(set(ranked) | must_keep).copy()


def frustration_figure(
    edges: np.ndarray,                 # (E, 2) graph edges
    signs: np.ndarray,                 # (E,)   signs in {-1, +1}
    frustrated_cycles: Iterable[Iterable[int]],  # vertex sequences
    n_nodes: int,
    balance: float | None = None,
    max_nodes: int = 80,
    title: str = "",
) -> "plt.Figure":
    """Draw the signed graph with the frustrated cycles overlaid in bold.

    Positive edges are solid blue, negative edges dashed red; every edge of a
    frustrated (negative sign-product) cycle is over-drawn thick black and its
    vertices highlighted. Presentation only — the balance statistic itself is
    computed upstream (no algorithm logic here, CLAUDE.md §6.5 #2).

    Preconditions: ``edges`` is (E, 2); ``signs`` aligns with ``edges`` and is
    in {-1, +1}; each cycle's consecutive vertices (with wraparound) are graph
    edges. Postconditions: returns a Figure; if the graph exceeds ``max_nodes``
    it is pruned to the highest-degree nodes plus all frustrated-cycle vertices.
    """
    cyc_nodes, cyc_edges = _cycle_elements(frustrated_cycles)
    G = nx.Graph()
    G.add_nodes_from(range(int(n_nodes)))
    for (u, v), s in zip(edges, signs):
        G.add_edge(int(u), int(v), sign=int(s))
    H = _prune_for_legibility(G, max_nodes, cyc_nodes)

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(H, seed=0, k=0.5, iterations=60)
    _draw_signed_edges(H, pos, ax, cyc_edges)
    _draw_nodes(H, pos, ax, cyc_nodes)

    head = title or "Signed-graph balance"
    sub = f"balance = {balance:.3f}" if balance is not None else ""
    sub2 = f"{len(cyc_edges)} frustrated-cycle edges (bold)" if cyc_edges \
        else "no frustration"
    ax.set_title(f"{head}\n{sub}   {sub2}".rstrip())
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _draw_signed_edges(
    H: Any, pos: Any, ax: Any, bold_edges: set[tuple[int, int]],
) -> None:
    """Solid-blue positive, dashed-red negative, thick-black bold overlay."""
    pos_e = [(u, v) for u, v, d in H.edges(data=True) if d["sign"] == 1]
    neg_e = [(u, v) for u, v, d in H.edges(data=True) if d["sign"] == -1]
    nx.draw_networkx_edges(H, pos, edgelist=pos_e, edge_color="#4472C4",
                           width=1.0, alpha=0.6, ax=ax)
    nx.draw_networkx_edges(H, pos, edgelist=neg_e, edge_color="#C00000",
                           width=1.0, alpha=0.6, style="dashed", ax=ax)
    bold = [(u, v) for (u, v) in H.edges() if (min(u, v), max(u, v)) in bold_edges]
    if bold:
        nx.draw_networkx_edges(H, pos, edgelist=bold, edge_color="black",
                               width=3.0, ax=ax)


def _draw_nodes(H: Any, pos: Any, ax: Any, highlight: set[int]) -> None:
    """White nodes, with the highlighted (frustrated) vertices amber + larger."""
    other = [n for n in H.nodes if n not in highlight]
    nx.draw_networkx_nodes(H, pos, nodelist=other, node_size=60,
                           node_color="#FFFFFF", edgecolors="black",
                           linewidths=0.4, ax=ax)
    inside = sorted(highlight & set(H.nodes))
    if inside:
        nx.draw_networkx_nodes(H, pos, nodelist=inside, node_size=200,
                               node_color="#FFC000", edgecolors="black",
                               linewidths=1.2, ax=ax)


__all__ = ["roc_figure", "alpha_figure", "subgraph_figure", "frustration_figure"]
