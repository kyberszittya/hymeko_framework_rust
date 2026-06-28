"""Render the generated design hypergraphs (§9) + the generator→accuracy result.

Two figures: (1) the star-expanded design hypergraphs (points + block-hubs, signed edges) for the generator
family chain/k-uniform/Fano/Steiner/sunflower; (2) the StructuralActor-vs-HSiKAN-vs-MLP accuracy per generator
(why Steiner wins). Reuses the ``hypergraph_designs`` generators and ``structural_actor`` probe — no new logic.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from hymeko_rl.hypergraph_designs import k_uniform_random, steiner_triple_system, sunflower  # noqa: E402
from hymeko_rl.structural_probe import build_chain_graph  # noqa: E402

_OUT = Path("reports/figures")


def _is_base(label: str) -> bool:
    return label.startswith("n") and label[1:].isdigit()


def _edge_colour(signs: set[int]) -> str:
    if signs == {1}:
        return "#2ca02c"          # + coupling
    if signs == {-1}:
        return "#d62728"          # − coupling
    return "#9aa0a6"              # mixed = point↔block incidence (member +1 / hub −1)


def draw_design(hg, ax, title: str) -> None:
    """Draw a (star-expanded) design hypergraph: base points (circles) + block hubs (squares), signed edges."""
    pair_signs: dict[tuple[int, int], set[int]] = {}
    for (u, v), s in zip(hg.edges.tolist(), hg.signs.tolist()):
        pair_signs.setdefault((min(u, v), max(u, v)), set()).add(int(s))
    g = nx.Graph()
    g.add_nodes_from(range(hg.n_vertices))
    g.add_edges_from(pair_signs)
    pos = nx.spring_layout(g, seed=1, k=0.9)
    base = [i for i in range(hg.n_vertices) if _is_base(hg.vertex_labels[i])]
    hubs = [i for i in range(hg.n_vertices) if not _is_base(hg.vertex_labels[i])]
    for (u, v), signs in pair_signs.items():
        nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], edge_color=_edge_colour(signs), width=1.4,
                               alpha=0.75, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=base, node_color="#4c72b0", node_shape="o", node_size=300, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=hubs, node_color="#dd8452", node_shape="s", node_size=150, ax=ax)
    nx.draw_networkx_labels(g, pos, labels={i: hg.vertex_labels[i][1:] for i in base}, font_size=7,
                            font_color="white", ax=ax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def figure_hypergraphs() -> Path:
    designs = [
        ("chain  (k=2-uniform)\n9 points", build_chain_graph(9, seed=0)),
        ("k-uniform  k=3\n9 points, 8 blocks", k_uniform_random(9, 3, n_blocks=8, seed=0)),
        ("Fano  S(2,3,7) = PG(2,2)\nprojective, 7 points 7 lines", steiner_triple_system(7)),
        ("Steiner  S(2,3,9) = AG(2,3)\naffine, 9 points 12 lines", steiner_triple_system(9)),
        ("sunflower\nshared core + disjoint petals", sunflower(core_size=2, petal_size=2, n_petals=3)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    flat = axes.flatten()
    for ax, (title, hg) in zip(flat, designs):
        draw_design(hg, ax, title)
    flat[5].axis("off")
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c72b0", markersize=11, label="base point"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#dd8452", markersize=10,
               label="block hub (a hyperedge / design line)"),
        Line2D([0], [0], color="#9aa0a6", lw=2, label="point–block incidence"),
        Line2D([0], [0], color="#2ca02c", lw=2, label="+ coupling"),
        Line2D([0], [0], color="#d62728", lw=2, label="− coupling"),
    ]
    flat[5].legend(handles=legend, loc="center", frameon=False, fontsize=11,
                   title="generated design hypergraphs")
    fig.suptitle("Generated design hypergraphs — star-expanded (base points + block hubs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _OUT.mkdir(parents=True, exist_ok=True)
    out = _OUT / "design_hypergraphs.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))          # vector PDF for LaTeX
    plt.close(fig)
    return out


def figure_results(*, seeds: int = 2, epochs: int = 180) -> Path:
    from hymeko_rl.structural_actor import run_structural_actor_probe
    gens = [
        ("chain\n(k=2)", build_chain_graph(9, seed=0)),
        ("k-unif\nk=3", k_uniform_random(9, 3, n_blocks=8, seed=0)),
        ("k-unif\nk=4", k_uniform_random(9, 4, n_blocks=8, seed=0)),
        ("Steiner\nS(2,3,9)", steiner_triple_system(9)),
        ("sun-\nflower", sunflower(core_size=2, petal_size=2, n_petals=3)),
    ]
    actor, hsikan, mlp, labels = [], [], [], []
    for label, hg in gens:
        m = run_structural_actor_probe(hg=hg, seeds=seeds, epochs=epochs)["models"]
        labels.append(label)
        actor.append(m["struct_holonomy"]["mse_median"])
        hsikan.append(m["hsikan"]["mse_median"])
        mlp.append(m["mlp"]["mse_median"])
    import numpy as np
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 0.25, actor, 0.25, label="StructuralActor (walk-holonomy)", color="#2ca02c")
    ax.bar(x, hsikan, 0.25, label="HSiKAN (message-passing)", color="#4c72b0")
    ax.bar(x + 0.25, mlp, 0.25, label="MLP (flat)", color="#9aa0a6")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("structural-target test MSE (log; lower=better)")
    ax.set_title("Which generator is the best structural basis? — Steiner wins for the actor")
    ax.legend()
    fig.tight_layout()
    _OUT.mkdir(parents=True, exist_ok=True)
    out = _OUT / "generator_accuracy.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))          # vector PDF for LaTeX
    plt.close(fig)
    return out


def main() -> int:
    a = figure_hypergraphs()
    b = figure_results()
    print(f"wrote {a}\nwrote {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
