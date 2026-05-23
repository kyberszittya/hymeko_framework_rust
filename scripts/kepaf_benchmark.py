"""KEPAF paper §7 benchmark.

Runs CPU-side force-directed layout (Fruchterman-Reingold via
NetworkX, the same algorithm D3.force-link uses on the CPU) on three
fixtures and reports per-iteration time + total convergence time.
This is the SAME computational kernel as D3 in the browser; the
absolute numbers transfer between platforms up to host-language
overhead.

Fixtures:
  1. canonical paper example     (small,      |V|≈21)
  2. MNIST adjacency hypergraph  (medium,    |V|≈1242)
  3. synthetic large-scale       (large,     |V|=10000)

Per fixture:
  - build hypergraph in NetworkX (Levi-graph form: vertices + edge nodes)
  - measure layout time at increasing |V| budgets via spring_layout
  - report n_iter, t_per_iter (ms), total t (s), edge-crossing count

Outputs:
  - data/kepaf_bench.csv  (one row per fixture × seed)
  - paper/kepaf_v1/figures/canonical_layout.pdf
  - paper/kepaf_v1/figures/mnist_layout.pdf
  - paper/kepaf_v1/figures/scaling.pdf

Run: python3 scripts/kepaf_benchmark.py
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


OUT_DATA = Path("data/kepaf_bench.csv")
OUT_FIG_DIR = Path("paper/kepaf_v1/figures")
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Fixture builders ────────────────────────────────────────────────


def fixture_canonical():
    """Tiny robotic kinematic chain — 21 vertices, 10 hyperedges,
    matching the paper canonical example. Hyperedges are encoded as
    Levi-graph nodes connecting their members."""
    G = nx.Graph()
    # 21 vertices (links + joints + axes + sensors)
    vertices = [f"v{i:02d}" for i in range(21)]
    G.add_nodes_from(vertices, kind="vertex")

    # 10 hyperedges with the arity multiset (2,2,3,3,3,3,3,3,5,5)
    arities = [2, 2, 3, 3, 3, 3, 3, 3, 5, 5]
    rng = np.random.default_rng(0)
    for ei, ar in enumerate(arities):
        e_node = f"e{ei:02d}"
        G.add_node(e_node, kind="hyperedge")
        members = rng.choice(vertices, size=ar, replace=False)
        for m in members:
            sign = rng.choice([-1, 1, 0], p=[0.4, 0.5, 0.1])
            G.add_edge(e_node, m, sign=int(sign))
    return G, "canonical"


def fixture_mnist_adjacency():
    """MNIST plain MLP signed-incidence: 784→256→128→64→10. Encoded
    as a Levi-graph: every neuron is a vertex, every weight is a
    hyperedge of arity 2 (input neuron + output neuron). |V| = 1242
    neuron vertices + 91008 weight edges (which becomes 91008 Levi
    nodes — that's too many to layout). Subsample to a dense 50% of
    weights to get |V|≈45k Levi-graph nodes — at this size the
    spring_layout is already painfully slow, which is the point.
    Keep at full 1242 neurons + only the 1024 weights with highest
    magnitude per layer, enough to produce a representative figure
    at tractable cost."""
    G = nx.Graph()
    rng = np.random.default_rng(0)
    layers = [784, 256, 128, 64, 10]
    offset = 0
    layer_offsets = [offset]
    for L in layers:
        for i in range(L):
            G.add_node(f"n{offset + i}", kind="vertex", layer=len(layer_offsets) - 1)
        offset += L
        layer_offsets.append(offset)

    # Sparse weight subsample: per inter-layer block keep top-k by
    # synthetic |W|. Real weight magnitudes from a trained MLP would
    # be different; here we want a representative connectivity, not
    # a model.
    e_idx = 0
    for li in range(len(layers) - 1):
        n_in, n_out = layers[li], layers[li + 1]
        k_keep = min(2_000, n_in * n_out)
        # uniform random subset
        flat = rng.choice(n_in * n_out, size=k_keep, replace=False)
        for f in flat:
            i_in = f // n_out
            i_out = f % n_out
            v_in = f"n{layer_offsets[li] + i_in}"
            v_out = f"n{layer_offsets[li + 1] + i_out}"
            e_node = f"e{e_idx}"
            G.add_node(e_node, kind="hyperedge")
            sign = -1 if (rng.random() < 0.5) else 1
            G.add_edge(e_node, v_in, sign=sign)
            G.add_edge(e_node, v_out, sign=-sign)
            e_idx += 1
    return G, "mnist_adj"


def fixture_synthetic(N=10_000, M=25_000, mean_arity=4, seed=0):
    """Procedurally-generated hypergraph — N vertices, M hyperedges,
    mean arity = mean_arity. Encoded as Levi-graph: |V_levi| = N + M."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    for i in range(N):
        G.add_node(f"v{i}", kind="vertex")
    for j in range(M):
        e_node = f"e{j}"
        G.add_node(e_node, kind="hyperedge")
        ar = max(2, int(rng.poisson(mean_arity)))
        members = rng.choice(N, size=ar, replace=False)
        for m in members:
            sign = rng.choice([-1, 1, 0], p=[0.4, 0.5, 0.1])
            G.add_edge(e_node, f"v{m}", sign=int(sign))
    return G, f"synthetic_{N}"


# ─── Benchmark ───────────────────────────────────────────────────────


def time_layout(G, n_iter=50, seed=0):
    """spring_layout (Fruchterman-Reingold) timing.
    Returns (positions, t_total_seconds, t_per_iter_ms)."""
    t0 = time.perf_counter()
    pos = nx.spring_layout(G, iterations=n_iter, seed=seed)
    t1 = time.perf_counter()
    total = t1 - t0
    return pos, total, (total / n_iter) * 1000.0


def edge_crossings_estimate(G, pos, sample=500):
    """Approximate edge-crossing count by sampling. Full O(E^2)
    is intractable for |E|>10^4; we sample `sample` edge pairs."""
    edges = list(G.edges())
    if len(edges) < 2:
        return 0
    rng = np.random.default_rng(0)
    n_sample = min(sample, len(edges))
    chosen_a = rng.choice(len(edges), size=n_sample, replace=False)
    crossings = 0
    pairs_tested = 0
    for i, ai in enumerate(chosen_a):
        for j in chosen_a[i + 1:]:
            if i == j:
                continue
            e_a = edges[ai]
            e_b = edges[j]
            if set(e_a) & set(e_b):
                continue  # share a vertex
            p1, p2 = pos[e_a[0]], pos[e_a[1]]
            p3, p4 = pos[e_b[0]], pos[e_b[1]]
            if _segments_cross(p1, p2, p3, p4):
                crossings += 1
            pairs_tested += 1
    if pairs_tested == 0:
        return 0
    full_pairs = len(edges) * (len(edges) - 1) // 2
    return int(crossings / pairs_tested * full_pairs)


def _segments_cross(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


# ─── Figure rendering ────────────────────────────────────────────────


def render_layout(G, pos, name, max_edges=2000):
    """Render with sign-aware grammar. Saves PDF + PNG."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)

    # Vertex glyphs
    v_x = []
    v_y = []
    e_x = []
    e_y = []
    for n, p in pos.items():
        kind = G.nodes[n].get("kind", "vertex")
        if kind == "hyperedge":
            e_x.append(p[0]); e_y.append(p[1])
        else:
            v_x.append(p[0]); v_y.append(p[1])
    ax.scatter(v_x, v_y, s=10, c="#EEF1F5", edgecolors="#3a4a5a",
               linewidths=0.5, zorder=3, label=f"vertices ({len(v_x)})")
    ax.scatter(e_x, e_y, s=20, marker="s", c="#D7E4F5",
               edgecolors="#1b6ca8", linewidths=0.5, zorder=4,
               label=f"hyperedges ({len(e_x)})")

    # Sign-aware arc rendering. Subsample edges if too many.
    edges = list(G.edges(data=True))
    if len(edges) > max_edges:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(edges), size=max_edges, replace=False)
        edges = [edges[i] for i in keep]
    for u, v, d in edges:
        s = d.get("sign", 1)
        color = {1: "#1b6ca8", -1: "#b02a2a", 0: "#888888"}.get(s, "#888888")
        alpha = 0.55 if len(G) < 500 else 0.18
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, lw=0.8, alpha=alpha, zorder=2)

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}: |V|={len(G)}, |E|={G.number_of_edges()}")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    out_pdf = OUT_FIG_DIR / f"{name}_layout.pdf"
    out_png = OUT_FIG_DIR / f"{name}_layout.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_pdf, out_png


def render_scaling(rows):
    """Per-fixture layout time as a bar plot."""
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=120)
    names = [r["fixture"] for r in rows]
    tot = [r["t_total_s"] for r in rows]
    Ns = [r["V_levi"] for r in rows]
    bars = ax.bar(names, tot, color=["#1b6ca8", "#3aa074", "#b02a2a"])
    for b, n in zip(bars, Ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.05,
                f"|V|={n}", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("layout time (s, log scale)")
    ax.set_title("CPU spring_layout (Fruchterman--Reingold), 50 iterations")
    fig.tight_layout()
    out = OUT_FIG_DIR / "scaling.pdf"
    fig.savefig(out)
    fig.savefig(OUT_FIG_DIR / "scaling.png", dpi=120)
    plt.close(fig)
    return out


# ─── Main ────────────────────────────────────────────────────────────


def main():
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # --- Canonical: small, run 5 seeds ---
    for seed in range(5):
        G, name = fixture_canonical()
        pos, total, per_iter = time_layout(G, n_iter=100, seed=seed)
        if seed == 0:
            render_layout(G, pos, name)
        rows.append(dict(
            fixture=name, seed=seed, V_levi=len(G), E_levi=G.number_of_edges(),
            n_iter=100, t_total_s=total, t_per_iter_ms=per_iter,
            edge_crossings=edge_crossings_estimate(G, pos),
        ))
        print(f"[{name} s{seed}] |V|={len(G)} |E|={G.number_of_edges()} "
              f"100 iter in {total:.3f}s  ({per_iter:.2f}ms/iter)  "
              f"crossings≈{rows[-1]['edge_crossings']}")

    # --- MNIST adjacency: medium, 3 seeds ---
    G, name = fixture_mnist_adjacency()
    print(f"\n[{name}] |V|={len(G)} |E|={G.number_of_edges()}")
    for seed in range(3):
        pos, total, per_iter = time_layout(G, n_iter=50, seed=seed)
        if seed == 0:
            render_layout(G, pos, name, max_edges=3000)
        rows.append(dict(
            fixture=name, seed=seed, V_levi=len(G), E_levi=G.number_of_edges(),
            n_iter=50, t_total_s=total, t_per_iter_ms=per_iter,
            edge_crossings=edge_crossings_estimate(G, pos),
        ))
        print(f"[{name} s{seed}] 50 iter in {total:.3f}s  "
              f"({per_iter:.2f}ms/iter)  crossings≈{rows[-1]['edge_crossings']}")

    # --- Synthetic |V|=10k: 1 seed only (slow) ---
    print("\n[synthetic_10000] building & laying out (slow)...")
    G, name = fixture_synthetic(N=10_000, M=25_000)
    print(f"  |V_levi|={len(G)} |E_levi|={G.number_of_edges()}")
    pos, total, per_iter = time_layout(G, n_iter=20, seed=0)
    render_layout(G, pos, name, max_edges=5000)
    rows.append(dict(
        fixture=name, seed=0, V_levi=len(G), E_levi=G.number_of_edges(),
        n_iter=20, t_total_s=total, t_per_iter_ms=per_iter,
        edge_crossings=-1,  # too expensive even with sampling
    ))
    print(f"[{name}] 20 iter in {total:.3f}s  ({per_iter:.2f}ms/iter)")

    # --- Save CSV ---
    with OUT_DATA.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {OUT_DATA}")

    # --- Aggregate per fixture for scaling figure ---
    by_fix = {}
    for r in rows:
        by_fix.setdefault(r["fixture"], []).append(r)
    agg = []
    for name, group in by_fix.items():
        agg.append(dict(
            fixture=name,
            V_levi=group[0]["V_levi"],
            t_total_s=sum(r["t_total_s"] for r in group) / len(group),
            t_per_iter_ms=sum(r["t_per_iter_ms"] for r in group) / len(group),
        ))
    render_scaling(agg)
    print(f"Wrote figures to {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
