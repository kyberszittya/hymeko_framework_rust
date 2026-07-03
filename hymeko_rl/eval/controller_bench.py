"""Controller benchmark — the topology→performance map (Phase 1 of Kato's isomorphic-controllers program).

For a family of topologies (``topology_zoo.TOPOLOGIES``) over matched size N, instantiate each as a *learned*
controller (HSiKAN / ``SignedKANBackbone`` over its signed adjacency) and ask: *which controller topology best
predicts which plant's structural target?* The plant defines a fixed structural regression target (reusing the
``structural_probe`` harness); each controller topology is trained on it; the held-out MSE fills a
``map[plant][controller]`` matrix. The diagonal (controller topology = plant topology) tests whether *matching
the plant's interconnection structure* helps; denser topologies test whether connectivity buys expressiveness.

This is the cheap supervised *rank* of the plan — CPU-light, single process, parallel-safe beside an RL run.
The closed-loop control task (cart-pole / arm) is the machine-bound follow-up, gated on this rank correlating.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from hymeko_rl.experiments.reach_arch_compare import _iqr
from hymeko_rl.experiments.structural_probe import _standardised_split, build_model, train_eval
from hymeko_rl.control.topology_zoo import TOPOLOGIES, permuted


def _median_mse_over_seeds(plant_hg, controller_hg, *, n_train: int, n_test: int, hidden: int, n_layers: int,
                           epochs: int, seeds: int) -> tuple[float, float]:
    """(median, IQR) held-out MSE of a HSiKAN controller on ``controller_hg`` predicting ``plant_hg``'s
    structural target. The split (plant data) is regenerated per seed; the controller seeds its own init."""
    vals: list[float] = []
    for s in range(seeds):
        split = _standardised_split(plant_hg, "structural", n_train=n_train, n_test=n_test, seed=1000 + s)
        torch.manual_seed(s)                                   # controller init varies per seed
        model = build_model("hsikan", controller_hg, hidden, n_layers=n_layers)
        vals.append(train_eval(model, split, epochs=epochs, seed=s))
    return statistics.median(vals), _iqr(vals)


def run_topology_map(*, names: "list[str] | None" = None, n_nodes: int = 9, hidden: int = 24, n_layers: int = 2,
                     n_train: int = 256, n_test: int = 1024, seeds: int = 3, epochs: int = 250,
                     graph_seed: int = 0) -> dict[str, object]:
    """The plant×controller topology→performance map (median test MSE).

    # Preconditions every name in :data:`TOPOLOGIES`; ``n_nodes`` a perfect square if ``grid`` is included.
    # Postconditions ``matrix[plant][controller]`` median MSE; ``best_controller[plant]`` the argmin."""
    names = names or list(TOPOLOGIES)
    hgs = {name: TOPOLOGIES[name](n_nodes, seed=graph_seed) for name in names}
    matrix: dict[str, dict[str, float]] = {}
    iqr: dict[str, dict[str, float]] = {}
    for plant in names:
        matrix[plant], iqr[plant] = {}, {}
        for controller in names:
            med, spread = _median_mse_over_seeds(
                hgs[plant], hgs[controller], n_train=n_train, n_test=n_test, hidden=hidden,
                n_layers=n_layers, epochs=epochs, seeds=seeds)
            matrix[plant][controller] = round(med, 5)
            iqr[plant][controller] = round(spread, 5)
    best = {plant: min(matrix[plant], key=lambda c: matrix[plant][c]) for plant in names}
    diag_is_best = {plant: best[plant] == plant for plant in names}
    return dict(n_nodes=n_nodes, names=names, seeds=seeds, epochs=epochs, matrix=matrix, iqr=iqr,
                best_controller=best, diagonal_is_best=diag_is_best,
                edge_counts={n: int(hgs[n].edges.shape[0]) // 2 for n in names})


def equivariance_check(*, topology: str = "small_world", n_nodes: int = 9, hidden: int = 24, n_layers: int = 2,
                       seed: int = 0, perm_seed: int = 7, batch: int = 64) -> dict[str, float]:
    """The EXACT well-definedness guard (replaces the confounded statistical one): the learned controller must be
    permutation-equivariant, so an isomorphic relabelling is a genuine no-op. With **identical weights**, the
    pooled output on ``(H, x)`` must equal the output on ``(π(H), π(x))`` — relabel the graph and its input by the
    same permutation. A tiny residual proves ``C(H)`` depends only on the topology up to isomorphism (so the
    map's off-diagonal structure is real, not a labelling artefact).

    # Postconditions ``residual`` is the max abs output difference; equivariant iff ``residual < 1e-4``."""
    base = TOPOLOGIES[topology](n_nodes, seed=seed)
    perm = torch.randperm(n_nodes, generator=torch.Generator().manual_seed(perm_seed)).numpy()
    iso = permuted(base, perm)
    inv = torch.from_numpy(perm.argsort())                      # inverse: x_perm[:, perm[i]] = x[:, i]
    x = torch.randn(batch, n_nodes, 1, generator=torch.Generator().manual_seed(seed + 1))
    x_perm = x[:, inv, :]
    torch.manual_seed(seed)
    m_base = build_model("hsikan", base, hidden, n_layers=n_layers)
    torch.manual_seed(seed)                                     # identical weights; only the adjacency differs
    m_iso = build_model("hsikan", iso, hidden, n_layers=n_layers)
    m_base.eval()
    m_iso.eval()
    with torch.no_grad():
        residual = float((m_base(x) - m_iso(x_perm)).abs().max())
    return dict(topology=topology, residual=round(residual, 8), equivariant=residual < 1e-4)


def plot_topology_map(report: dict[str, object], out_path: str | Path) -> Path:
    """Heat-map of ``matrix[plant][controller]`` (rows = plant, cols = controller); annotate the per-row argmin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = report["names"]          # type: ignore[index]
    matrix = report["matrix"]        # type: ignore[index]
    best = report["best_controller"] # type: ignore[index]
    grid = np.array([[matrix[p][c] for c in names] for p in names])
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 1.0 * len(names) + 1.5))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("controller topology")
    ax.set_ylabel("plant topology (target)")
    ax.set_title("Topology→performance: held-out MSE (lower=better); ★ = best controller per plant")
    for i, p in enumerate(names):
        for j, c in enumerate(names):
            star = "★" if best[p] == c else ""
            ax.text(j, i, f"{grid[i, j]:.2f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] > grid.mean() else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="topology->performance map (Phase 1)")
    ap.add_argument("--n-nodes", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out-dir", default="reports/topology_map")
    a = ap.parse_args(argv)
    report = run_topology_map(n_nodes=a.n_nodes, seeds=a.seeds, epochs=a.epochs)
    report["equivariance"] = [equivariance_check(topology=t, n_nodes=a.n_nodes)
                              for t in ("chain", "star", "small_world", "complete")]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "topology_map.json").write_text(json.dumps(report, indent=2))
    plot_topology_map(report, out / "topology_map")
    print(json.dumps({k: report[k] for k in ("best_controller", "diagonal_is_best", "equivariance",
                                             "edge_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
