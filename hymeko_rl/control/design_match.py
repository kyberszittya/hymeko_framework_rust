"""Match study — do combinatorial-design hypergraphs (Steiner / k-uniform / sunflower) matter more for control
design than the ad-hoc graph zoo did? (Kato, 2026-06-27.)

Two **size-controlled** sub-studies (star-expansion makes size depend on block count, so we fix it to keep the
comparison fair — different sizes can't share the topology→performance map):

1. **Coupling-order axis** — k-uniform hypergraphs over n=9 base, 8 blocks each, k ∈ {2,3,4}. All star-expand to
   17 vertices, so the *only* difference is the interaction order. Does higher-order coupling matter?
2. **Balance axis** — Steiner ``S(2,3,9)`` (balanced, every pair once) vs a random 3-uniform hypergraph with the
   same block count (12) — both star-expand to 21. Does the balanced design beat random of equal order+count?

Each sub-study runs the Phase-1 **representation** map (reuse ``controller_bench._median_mse_over_seeds``) and the
Phase-2 **structured-control** map (reuse ``structured_control``), as plant×controller, on the *same* design set.
No new benchmark machinery — only the generators are new.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from hymeko_rl.eval.controller_bench import _median_mse_over_seeds
from hymeko_rl.control.hypergraph_designs import k_uniform_random, steiner_triple_system
from hymeko_rl.control.structured_control import (
    make_plant, mask_from_topology, structured_lqr, unconstrained_lqr,
)


def _representation_map(states: dict, *, seeds: int = 3, epochs: int = 200) -> dict[str, dict[str, float]]:
    """Plant×controller held-out MSE (lower=better); matched (diagonal) best ⇒ the design structure is
    load-bearing for representation."""
    out: dict[str, dict[str, float]] = {}
    for plant, p_hg in states.items():
        out[plant] = {}
        for ctrl, c_hg in states.items():
            med, _ = _median_mse_over_seeds(p_hg, c_hg, n_train=256, n_test=1024, hidden=20,
                                            n_layers=2, epochs=epochs, seeds=seeds)
            out[plant][ctrl] = round(med, 5)
    return out


def _control_map(states: dict, *, eps: float = 0.95, iters: int = 800) -> dict[str, dict[str, float]]:
    """Plant×controller structured-LQR suboptimality ρ=J(K_H)/J* (≥1); diagonal lowest ⇒ design coupling
    controls its own plant best."""
    out: dict[str, dict[str, float]] = {}
    masks = {nm: mask_from_topology(hg) for nm, hg in states.items()}
    for plant, p_hg in states.items():
        big_a, big_b = make_plant(p_hg, eps=eps)
        n = big_a.shape[0]
        q, r = np.eye(n), np.eye(n)
        _, j_star = unconstrained_lqr(big_a, big_b, q, r)
        out[plant] = {}
        for ctrl in states:
            _, j_h, _ = structured_lqr(big_a, big_b, masks[ctrl], q, r, iters=iters)
            out[plant][ctrl] = round(j_h / j_star, 4) if math.isfinite(j_h) else math.inf
    return out


def _diagonal_best(matrix: dict[str, dict[str, float]], *, tol: float = 0.0) -> dict[str, bool]:
    return {p: matrix[p][p] <= min(matrix[p].values()) + tol for p in matrix}


def run_design_match(*, seeds: int = 3, epochs: int = 200) -> dict[str, object]:
    """The two size-controlled sub-studies. # Postconditions for each: representation + control maps, the
    matched-is-best verdicts, and the across-set magnitude (does order/balance change the numbers)."""
    order_set = {f"k{k}": k_uniform_random(9, k, n_blocks=8, seed=0) for k in (2, 3, 4)}
    balance_set = {"steiner9": steiner_triple_system(9), "rand3_12": k_uniform_random(9, 3, n_blocks=12, seed=1)}
    result: dict[str, object] = {}
    for label, states in (("coupling_order", order_set), ("balance", balance_set)):
        rep = _representation_map(states, seeds=seeds, epochs=epochs)
        ctl = _control_map(states)
        result[label] = dict(
            sizes={nm: hg.n_vertices for nm, hg in states.items()},
            representation=rep, control=ctl,
            rep_matched_best=_diagonal_best(rep, tol=1e-4),
            ctl_matched_best=_diagonal_best(ctl, tol=2e-3),
            rep_diagonal={nm: rep[nm][nm] for nm in states},      # the matched-representation error per design
        )
    return result


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="combinatorial-design match study")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out-dir", default="reports/design_match")
    a = ap.parse_args(argv)
    report = run_design_match(seeds=a.seeds, epochs=a.epochs)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "design_match.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
