"""Render the figure battery for the policy-storage artifact (RL-on-HyMeKo technical line).

Generates PNG/SVG into docs/figures/2026-06-21-policy-storage/:
  1. weight matrix <-> star-expansion identity (the core idea)
  2. cart-pole kinematic hypergraph = the signed incidence (a_pos/a_neg)
  3. HSiKAN actor-critic dataflow hypergraph
  4. trained-weight gallery (every learned tensor)
  5. ablation (HSiKAN vs MLP capacity) + vectorization speedup
  6. round-trip verification (bit-exact)

Run: uv run python scripts/render_policy_storage_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures" / "2026-06-21-policy-storage"
OUT.mkdir(parents=True, exist_ok=True)
SD = torch.load(REPO / "reports" / "cartpole_hsikan_policy.pt", weights_only=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.titlesize": 10})
DIV = "RdBu_r"  # diverging map for signed weights


def _save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.svg")


def fig1_identity() -> None:
    """A weight (sub)matrix IS the weighted star expansion of a bipartite hypergraph."""
    w = SD["actor_backbone.layers.0.w_pos.weight"].numpy()[:6, :2]  # 6 vertices x 2 hyperedges
    fig, (axm, axg) = plt.subplots(1, 2, figsize=(9, 4.2))
    vmax = np.abs(w).max()
    im = axm.imshow(w, cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto")
    axm.set_title("weight matrix  W  (6×2 slice)")
    axm.set_xlabel("hyperedges (cols)"); axm.set_ylabel("vertices (rows)")
    axm.set_xticks([0, 1]); axm.set_yticks(range(6))
    fig.colorbar(im, ax=axm, fraction=0.046, label="weight")

    g = nx.Graph()
    vts = [f"v{i}" for i in range(w.shape[0])]
    hes = [f"e{j}" for j in range(w.shape[1])]
    pos = {v: (0, -i) for i, v in enumerate(vts)}
    pos |= {e: (1.6, -2.5 - 2 * j) for j, e in enumerate(hes)}
    for i, v in enumerate(vts):
        for j, e in enumerate(hes):
            g.add_edge(v, e, w=w[i, j])
    nx.draw_networkx_nodes(g, pos, nodelist=vts, node_color="#9fd6ff", node_size=520, ax=axg)
    nx.draw_networkx_nodes(g, pos, nodelist=hes, node_color="#f6c177", node_shape="s",
                           node_size=620, ax=axg)
    ews = [g[u][v]["w"] for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, ax=axg, width=[1 + 4 * abs(x) / vmax for x in ews],
                           edge_color=ews, edge_cmap=plt.get_cmap(DIV), edge_vmin=-vmax, edge_vmax=vmax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=axg)
    axg.set_title("its star expansion (edge weight = W[i,j])")
    axg.axis("off")
    fig.suptitle("A weight matrix is the star expansion of a weighted hypergraph", fontweight="bold")
    _save(fig, "01-weight-matrix-star-expansion")


def fig2_kinematic_incidence() -> None:
    """The cart-pole's a_pos/a_neg ARE the signed incidence = star expansion of the robot."""
    ap = SD["actor_backbone.a_pos"].numpy()
    an = SD["actor_backbone.a_neg"].numpy()
    fig = plt.figure(figsize=(10, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])
    axg = fig.add_subplot(gs[0])
    g = nx.DiGraph()
    g.add_nodes_from(["cart", "pole"])
    g.add_edge("cart", "pole", label="hinge")   # cart -> pole (the kinematic chain)
    pos = {"cart": (0, 0), "pole": (1, 0.5)}
    nx.draw_networkx_nodes(g, pos, node_color="#7aa2f7", node_size=1500, ax=axg)
    nx.draw_networkx_edges(g, pos, ax=axg, width=2.5, edge_color="#38bdf8",
                           connectionstyle="arc3,rad=0.1", arrowsize=18)
    nx.draw_networkx_labels(g, pos, font_size=9, font_color="white", ax=axg)
    axg.set_title("cart-pole kinematic hypergraph\n(2 vertices, signed joints)")
    axg.axis("off")
    for ax, m, t in ((fig.add_subplot(gs[1]), ap, "a_pos (+)"),
                     (fig.add_subplot(gs[2]), an, "a_neg (−)")):
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=max(ap.max(), an.max(), 1e-6))
        ax.set_title(t); ax.set_xticks([0, 1], ["cart", "pole"]); ax.set_yticks([0, 1], ["cart", "pole"])
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("The robot's signed incidence is the fixed (structural) part of the policy",
                 fontweight="bold")
    _save(fig, "02-kinematic-incidence")


def fig3_dataflow() -> None:
    """The HSiKAN actor-critic as a layered dataflow hypergraph."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    nodes = {
        "obs": (0, 0, "#9fd6ff", "obs\n(2,2)"),
        "ac0": (1.4, 1, "#c4b5fd", "SignedConv₀\nself/pos/neg"),
        "ac1": (2.8, 1, "#c4b5fd", "SignedConv₁"),
        "apool": (4.0, 1, "#86efac", "mean-pool"),
        "amean": (5.2, 1, "#f6c177", "actor_mean"),
        "act": (6.4, 1, "#fca5a5", "action\n(1,)"),
        "cc0": (1.4, -1, "#c4b5fd", "SignedConv₀"),
        "cc1": (2.8, -1, "#c4b5fd", "SignedConv₁"),
        "cpool": (4.0, -1, "#86efac", "mean-pool"),
        "crit": (5.2, -1, "#f6c177", "critic"),
        "val": (6.4, -1, "#fca5a5", "value\n(1,)"),
    }
    edges = [("obs", "ac0"), ("ac0", "ac1"), ("ac1", "apool"), ("apool", "amean"), ("amean", "act"),
             ("obs", "cc0"), ("cc0", "cc1"), ("cc1", "cpool"), ("cpool", "crit"), ("crit", "val")]
    for a, b in edges:
        ax.annotate("", xy=nodes[b][:2], xytext=nodes[a][:2],
                    arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.6))
    for _, (x, y, c, lbl) in nodes.items():
        ax.add_patch(plt.Rectangle((x - 0.42, y - 0.32), 0.84, 0.64, fc=c, ec="#334155", zorder=3))
        ax.text(x, y, lbl, ha="center", va="center", fontsize=7.5, zorder=4)
    ax.text(3.2, 1.9, "ACTOR  (reads the kinematic hypergraph)", fontsize=9, color="#6d28d9")
    ax.text(3.2, -1.95, "CRITIC  (separate backbone)", fontsize=9, color="#6d28d9")
    ax.set_xlim(-0.7, 7.1); ax.set_ylim(-2.4, 2.4); ax.axis("off")
    fig.suptitle("HSiKAN actor-critic — dataflow hypergraph (obs → signed message passing → heads)",
                 fontweight="bold")
    _save(fig, "03-actor-critic-dataflow")


def fig4_weight_gallery() -> None:
    """Every learned weight tensor of the trained policy as a heatmap."""
    keys = [k for k in SD if SD[k].ndim == 2]  # the matrices
    n = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.3))
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, keys):
        w = SD[k].numpy()
        vmax = np.abs(w).max() or 1.0
        ax.imshow(w, cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(k.replace("_backbone", "").replace(".weight", ""), fontsize=6.2)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"The trained cart-pole policy — {n} weight matrices (each a weighted incidence)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "04-weight-gallery")


def fig5_results() -> None:
    """Ablation (HSiKAN vs MLP capacity) + vectorization speedup."""
    def load(p):
        return [json.loads(l) for l in (REPO / "reports" / p).read_text().splitlines() if l.strip()]
    rows = load("2026-06-21-cartpole-multiseed.jsonl") + load("2026-06-21-cartpole-controls.jsonl")
    groups = {"HSiKAN\n26.2k": ("hsikan", 26243), "MLP\n9k": ("mlp", 9091),
              "MLP\n26.7k": ("mlp", 26659), "MLP\n135k": ("mlp", 134659)}
    means, sds, labels, colors = [], [], [], []
    for lbl, (pol, npar) in groups.items():
        v = [r["upright_steps"] for r in rows if r["policy"] == pol and r["n_params"] == npar]
        means.append(np.mean(v)); sds.append(np.std(v)); labels.append(lbl)
        colors.append("#7aa2f7" if pol == "hsikan" else "#f6a")
    fig, (axa, axv) = plt.subplots(1, 2, figsize=(10, 4))
    axa.bar(labels, means, yerr=sds, capsize=5, color=colors, edgecolor="#334155")
    axa.axhline(200, ls="--", c="gray", lw=1); axa.set_ylabel("upright-steps / 200 (5-seed)")
    axa.set_title("Structure vs capacity: a params-matched MLP ties HSiKAN")
    for i, (m, s) in enumerate(zip(means, sds)):
        axa.text(i, m + s + 4, f"{m:.0f}±{s:.0f}", ha="center", fontsize=8)
    # vectorization
    cfg = ["single", "vec N=8", "vec N=16"]; it = [3.08, 1.14, 0.93]
    axv.bar(cfg, it, color=["#cbd5e1", "#86efac", "#34d399"], edgecolor="#334155")
    axv.set_ylabel("wall / PPO iteration (s)"); axv.set_title("Vectorized rollout: 3.1× faster")
    for i, t in enumerate(it):
        axv.text(i, t + 0.05, f"{t:.2f}s", ha="center", fontsize=8)
    fig.suptitle("Measured results behind the baseline", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "05-results")


def fig6_roundtrip() -> None:
    """Round-trip verification: stored .hymeko reconstructs the weights bit-exact."""
    import sys
    sys.path.insert(0, str(REPO))
    from hymeko_rl.agents.policy_store import hymeko_to_policy
    sd2 = hymeko_to_policy(REPO / "data" / "nn" / "cartpole_hsikan_policy.hymeko")
    orig = np.concatenate([SD[k].numpy().ravel() for k in SD])
    recon = np.concatenate([sd2[k].numpy().ravel() for k in SD])
    fig, (axs, axh) = plt.subplots(1, 2, figsize=(9.5, 4))
    axs.scatter(orig, recon, s=4, c="#7aa2f7", alpha=0.5)
    lim = [orig.min(), orig.max()]
    axs.plot(lim, lim, "k--", lw=1)
    axs.set_xlabel("original weight"); axs.set_ylabel("reconstructed from .hymeko")
    axs.set_title(f"bit-exact: max |Δ| = {np.abs(orig - recon).max():.0e}  (n={len(orig)})")
    axh.hist((orig - recon), bins=41, color="#34d399", edgecolor="#334155")
    axh.set_xlabel("weight error (original − reconstructed)"); axh.set_ylabel("count")
    axh.set_title("error distribution (all zero)")
    fig.suptitle("Trained policy ⇄ HyMeKo hypergraph: round-trip is lossless", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "06-roundtrip-verification")


def fig7_learned_incidence() -> None:
    """signedkan: the LEARNED signed incidence (star edges) vs the kinematic init — the deepest form of
    'weights as a hypergraph'. Skipped if the signedkan checkpoint is absent."""
    ckpt = REPO / "reports" / "cartpole_signedkan_policy.pt"
    if not ckpt.is_file():
        print("  (skip fig7: no signedkan checkpoint)")
        return
    import sys
    sys.path.insert(0, str(REPO))
    from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
    sd = torch.load(ckpt, weights_only=True)
    env = InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())
    init_pos, init_neg = (a.numpy() for a in env.hg.dense_signed_adj("cpu"))
    tr_pos, tr_neg = sd["actor_backbone.a_pos"].numpy(), sd["actor_backbone.a_neg"].numpy()
    labels = ["cart", "pole"]

    fig = plt.figure(figsize=(11, 5.2))
    mats = [("a_pos init (kinematic)", init_pos, "Blues"), ("a_pos LEARNED", tr_pos, "Blues"),
            ("a_neg init (kinematic)", init_neg, "Reds"), ("a_neg LEARNED", tr_neg, "Reds")]
    for i, (t, m, cmap) in enumerate(mats):
        ax = fig.add_subplot(2, 3, i + 1 + (i // 2))  # leave col 3 of each row for the graph
        vmax = max(np.abs(m).max(), 1e-6)
        ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([0, 1], labels, fontsize=7); ax.set_yticks([0, 1], labels, fontsize=7)
        for (r, c), v in np.ndenumerate(m):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)

    axg = fig.add_subplot(1, 3, 3)
    g = nx.DiGraph(); g.add_nodes_from(labels)
    pos = {"cart": (0, 0), "pole": (1.2, 0.0)}
    edges = []
    for (r, c), w in np.ndenumerate(tr_pos):
        if abs(w) > 1e-3 and r != c:
            edges.append((labels[c], labels[r], w, "#2563eb"))
    for (r, c), w in np.ndenumerate(tr_neg):
        if abs(w) > 1e-3 and r != c:
            edges.append((labels[c], labels[r], -w, "#db2777"))
    nx.draw_networkx_nodes(g, pos, node_color="#cbd5e1", node_size=2200, ax=axg)
    nx.draw_networkx_labels(g, pos, font_size=9, ax=axg)
    for u, v, w, col in edges:
        axg.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(
            arrowstyle="-|>", color=col, lw=1 + 5 * abs(w) / 2, connectionstyle="arc3,rad=0.25"))
    axg.set_title("learned incidence as signed star edges\n(blue +, pink −)", fontsize=9)
    axg.set_xlim(-0.5, 1.7); axg.set_ylim(-0.8, 0.8); axg.axis("off")
    fig.suptitle("signedkan: the signed incidence is LEARNED — the trained weights are the star edges",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "07-learned-incidence")


def main() -> None:
    print(f"rendering -> {OUT}")
    fig1_identity(); fig2_kinematic_incidence(); fig3_dataflow()
    fig4_weight_gallery(); fig5_results(); fig6_roundtrip(); fig7_learned_incidence()
    print("done.")


if __name__ == "__main__":
    main()
