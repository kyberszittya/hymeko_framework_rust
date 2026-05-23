"""Phase 10 — HSiKAN on kinematic graphs.

Builds a synthetic kinematic dataset by unioning many random
mechanisms (4-bar / Stewart / delta / serial of various sizes) into
a single big disconnected SignedGraph. Each edge is a joint; sign =
+1 (revolute / continuous) or −1 (prismatic).

Task: predict joint type (sign) from graph structure. This is the
same prediction problem as signed link sign prediction — HSiKAN
consumes it without modification. The αₖ pattern that emerges on
this kinematic dataset will tell us whether the model auto-discovers
the dominant cycle arity for kinematic mechanisms (k=4 from
four-bars, k=6 from parallel manipulators).
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import numpy as np

from signedkan_wip.src.datasets import SignedGraph
from signedkan_wip.src.kinematic import write_fixture, _serial_arm_urdf
from signedkan_wip.src.kinematic import urdf_to_signed_graph
from .run_phase2_mixed_arity import run_one_mixed


def union_signed_graphs(graphs: list[SignedGraph]) -> SignedGraph:
    """Disjoint-union of SignedGraphs. Vertex IDs are offset so each
    input graph occupies a distinct ID range."""
    edges_acc = []
    signs_acc = []
    offset = 0
    for g in graphs:
        if g.edges.shape[0] > 0:
            edges_acc.append(g.edges + offset)
            signs_acc.append(g.signs)
        offset += g.n_nodes
    edges = (np.concatenate(edges_acc, axis=0)
             if edges_acc else np.zeros((0, 2), dtype=np.int64))
    signs = (np.concatenate(signs_acc) if signs_acc
             else np.zeros((0,), dtype=np.int8))
    return SignedGraph(edges=edges, signs=signs, n_nodes=offset)


def build_kinematic_dataset(n_each: dict[str, int], seed: int = 0) -> SignedGraph:
    """Build a unioned kinematic dataset from fixtures.

    n_each: dict mapping fixture name → count, e.g.
        {"four_bar": 20, "stewart": 10, "delta_3rrr": 10, "serial_7": 5}
    """
    rng = random.Random(seed)
    graphs = []
    tmp_files = []
    try:
        for name, count in n_each.items():
            for i in range(count):
                # serial_N: vary N for variety
                if name.startswith("serial_"):
                    n_links = int(name.split("_")[1])
                    # Add some randomization
                    n_links_actual = max(2, n_links + rng.randint(-2, 2))
                    from signedkan_wip.src.kinematic import _serial_arm_urdf
                    import tempfile
                    f = tempfile.NamedTemporaryFile(
                        mode="w", suffix=f"_serial_{n_links_actual}.urdf",
                        delete=False,
                    )
                    f.write(_serial_arm_urdf(n_links_actual))
                    f.close()
                    path = Path(f.name)
                else:
                    path = write_fixture(name)
                tmp_files.append(path)
                g, _, _ = urdf_to_signed_graph(path)
                graphs.append(g)
        return union_signed_graphs(graphs)
    finally:
        for p in tmp_files:
            try: p.unlink()
            except FileNotFoundError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_4bar", type=int, default=20)
    ap.add_argument("--n_stewart", type=int, default=10)
    ap.add_argument("--n_delta", type=int, default=10)
    ap.add_argument("--n_serial4", type=int, default=10)
    ap.add_argument("--n_serial7", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--grid", type=int, default=5)
    args = ap.parse_args()

    counts = {
        "four_bar": args.n_4bar,
        "stewart": args.n_stewart,
        "delta_3rrr": args.n_delta,
        "serial_4": args.n_serial4,
        "serial_7": args.n_serial7,
    }
    print(f"Building kinematic dataset with: {counts}", flush=True)
    g = build_kinematic_dataset(counts, seed=0)
    print(f"Combined graph: {g.stats()}", flush=True)

    # Register a custom dataset name in `load()` cache so run_one_mixed
    # picks it up. Easiest path: monkey-patch a closure-based loader.
    from . import datasets as _ds
    DATASET_NAME = "phase10_kinematic"
    _orig_load = _ds.load
    _stash = {DATASET_NAME: g}
    def _load(name: str) -> SignedGraph:
        if name in _stash:
            return _stash[name]
        return _orig_load(name)
    _ds.load = _load
    # Also patch in the run module's namespace.
    import signedkan_wip.experiments.runs.run_phase2_mixed_arity as _rp
    _rp.load = _load

    print(f"\n=== HSiKAN on kinematic dataset, {len(args.seeds)} seeds ===",
          flush=True)
    print(f"{'config':<24s}  {'AUC_med':>8s}  {'std':>6s}  {'F1m_med':>8s}  alpha", flush=True)

    # Kinematic dataset has k=4 (4-bars) and k=6 (Stewart/delta) cycles.
    # k=3 and k=5 are 0 — auto-dropped by run_one_mixed.
    configs = [
        ("k4_only",       (4,),         0.0),
        ("k6_only",       (6,),         0.0),
        ("k46_lambda0",   (4, 6),       0.0),
        ("k46_lambda1",   (4, 6),       1.0),
        ("k4567_lambda0", (4, 5, 6, 7), 0.0),
        ("k4567_lambda1", (4, 5, 6, 7), 1.0),
    ]
    for cell, arities, lam in configs:
        aucs, f1ms, alphas = [], [], []
        for seed in args.seeds:
            try:
                r = run_one_mixed(
                    DATASET_NAME, seed=seed,
                    hidden=args.hidden, n_layers=2, grid=args.grid,
                    n_epochs=args.n_epochs,
                    arities=arities,
                    max_per_arity={k: 30_000 for k in arities},
                    coef_smooth_lam=0.0, participation_lam=0.0,
                    grad_clip=0.0, weight_decay=0.0,
                    early_stopping=False, class_weighted=False,
                    lr_schedule="cosine",
                    feature_edges="all",
                    m_e_mode="edge_in_cycle",
                    balance_lambda=lam,
                )
                aucs.append(r["test_auc"])
                f1ms.append(r["test_f1_macro"])
                alphas.append(r["alpha"])
            except Exception as e:
                print(f"  {cell:<24s} seed={seed} FAILED: {e!r}", flush=True)
        if not aucs:
            continue
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        # Alpha length may have shrunk if some arities had 0 cycles.
        n_alpha = min(len(a) for a in alphas)
        amed = [round(statistics.median([a[i] for a in alphas]), 2)
                for i in range(n_alpha)]
        print(f"{cell:<24s}  {statistics.median(aucs):>8.4f}  "
              f"{std:>6.4f}  {statistics.median(f1ms):>8.4f}  {amed}",
              flush=True)


if __name__ == "__main__":
    main()
