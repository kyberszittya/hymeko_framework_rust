"""4-panel iteration-progression figure for KEPAF Section 4: the
canonical 21-vertex fixture laid out by the GPU force_directed kernel
at t in {0, 10, 30, 100} iterations.

Output: paper/kepaf_v1/figures/iter_progression.{pdf,png}

Calls the layout binary with `dump_every`=1 so each iteration's
positions are captured, then renders the four chosen iterations via
the existing renderer.
"""

from __future__ import annotations

import json
import os
import subprocess
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

LAYOUT_BIN = REPO / "target" / "release" / "examples" / "layout_from_json"
OUT_DIR = REPO / "paper" / "kepaf_v1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITERS_TO_SHOW = [0, 10, 30, 100]


def run_layout_with_snapshots(G, n_iter: int = 100, dump_every: int = 1,
                              seed: int = 0):
    if not LAYOUT_BIN.exists():
        raise SystemExit(f"missing binary: {LAYOUT_BIN}")
    labels = list(G.nodes())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    edges = [[label_to_id[u], label_to_id[v]] for u, v in G.edges()]
    payload = json.dumps({
        "n_nodes": len(labels),
        "n_iter": n_iter,
        "seed": seed,
        "dump_every": dump_every,
        "edges": edges,
    })
    proc = subprocess.run(
        [str(LAYOUT_BIN)], input=payload,
        capture_output=True, text=True, check=True,
    )
    out = json.loads(proc.stdout)
    return labels, out


def render_panel(ax, G, pos, title):
    v_xy, e_xy = [], []
    for n, p in pos.items():
        kind = G.nodes[n].get("kind", "vertex")
        (e_xy if kind == "hyperedge" else v_xy).append(p)
    if v_xy:
        v_xy = np.asarray(v_xy)
        ax.scatter(v_xy[:, 0], v_xy[:, 1], s=22, c="#EEF1F5",
                   edgecolors="#3a4a5a", linewidths=0.7, zorder=3)
    if e_xy:
        e_xy = np.asarray(e_xy)
        ax.scatter(e_xy[:, 0], e_xy[:, 1], s=42, marker="s",
                   c="#D7E4F5", edgecolors="#1b6ca8",
                   linewidths=0.7, zorder=4)

    for u, v, d in G.edges(data=True):
        s = d.get("sign", 1)
        c = {1: "#1b6ca8", -1: "#b02a2a", 0: "#888888"}.get(s, "#888888")
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=c, lw=1.0, alpha=0.7, zorder=2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    print("== iteration progression panel ==")
    G, name = fixture_canonical()
    print(f"[{name}] |V|={len(G)} |E|={G.number_of_edges()}")
    labels, out = run_layout_with_snapshots(G, n_iter=100,
                                            dump_every=1, seed=0)
    print(f"  device={out['device']}  wall={out['wall_ms']:.1f}ms  "
          f"snaps={len(out['snapshots'])}")
    snaps_by_iter = {s["iter"]: s["positions"] for s in out["snapshots"]}

    fig, axes = plt.subplots(1, 4, figsize=(11.4, 2.9), dpi=120)
    for ax, it in zip(axes, ITERS_TO_SHOW):
        positions = snaps_by_iter.get(it)
        if positions is None:
            ax.set_axis_off()
            continue
        pos = {labels[i]: tuple(xy) for i, xy in enumerate(positions)}
        # Centre + scale per panel for visual stability across iterations.
        arr = np.asarray(positions, dtype=float)
        ax.set_xlim(arr[:, 0].min() - 0.05,
                    arr[:, 0].max() + 0.05)
        ax.set_ylim(arr[:, 1].min() - 0.05,
                    arr[:, 1].max() + 0.05)
        render_panel(ax, G, pos, f"$t={it}$ iterations")

    fig.suptitle("Canonical 21-vertex fixture: GPU layout convergence",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    out_pdf = OUT_DIR / "iter_progression.pdf"
    out_png = OUT_DIR / "iter_progression.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
