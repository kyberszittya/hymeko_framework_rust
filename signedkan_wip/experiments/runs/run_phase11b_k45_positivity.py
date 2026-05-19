"""Phase 11b — positivity-ratio sweep with k45/k345 variants.

Phase 11 ran the positivity sweep (50%-95%) with hsikan_k34 only.
Phase 9 found that k=4+k=5 (hsikan_k45) is the architectural sweet
spot. This phase adds hsikan_k45 and hsikan_k345 to the positivity
sweep so the regime curve uses the strongest HSiKAN variant.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from signedkan_wip.src.datasets import load
from .run_phase2_mixed_arity import run_one_mixed


POSITIVITIES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase11b_k45_positivity.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()

    for pos in POSITIVITIES:
        for seed in args.seeds:
            dataset = f"sbmsweep_pos{pos}_s{seed}"
            g = load(dataset)
            frac_pos = float((g.signs == 1).mean())
            n_edges = int(g.edges.shape[0])

            for arities, label in [
                ((3, 4, 5), "hsikan_k345"),
                ((4, 5),    "hsikan_k45"),
            ]:
                try:
                    r = run_one_mixed(
                        dataset, seed, hidden=16, n_layers=2, grid=3,
                        n_epochs=args.n_epochs,
                        arities=arities,
                        max_per_arity={3: 30000, 4: 30000, 5: 30000},
                        only_k3=False,
                        coef_smooth_lam=0.0, participation_lam=0.0,
                        grad_clip=0.0, weight_decay=0.0,
                        early_stopping=False, class_weighted=False,
                    )
                    r["arch"] = label
                    r["pos_in"] = pos
                    r["frac_pos"] = frac_pos
                    r["n_edges"] = n_edges
                    print(f"  {label:14s} pos={pos:>3d} (real {frac_pos:.2f}) seed={seed}  "
                          f"AUC={r['test_auc']:.4f}  alpha={[round(a,2) for a in r['alpha']]}  "
                          f"{r['elapsed_s']:.1f}s")
                    runs.append(r)
                except Exception as e:
                    print(f"  {label:14s} FAILED on {dataset}: {e!r}")

    summary = {}
    keys = sorted({(r["arch"], r["pos_in"]) for r in runs})
    for arch, pos in keys:
        cell = [r for r in runs if r["arch"] == arch and r["pos_in"] == pos]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        frac = cell[0]["frac_pos"]
        alphas = [r.get("alpha", []) for r in cell]
        summary[f"{arch}|pos{pos}"] = {
            "auc_mean":  round(float(np.mean(aucs)), 4),
            "auc_std":   round(float(np.std(aucs)), 4),
            "f1m_mean":  round(float(np.mean(f1ms)), 4),
            "n_seeds":   len(cell),
            "frac_pos":  round(frac, 4),
            "auc_seeds": [round(a, 4) for a in aucs],
            "f1m_seeds": [round(f, 4) for f in f1ms],
            "alpha_seeds": [[round(a, 3) for a in alpha] for alpha in alphas],
        }

    out = {
        "runs": runs,
        "summary": summary,
        "wall_clock_s": round(time.time() - t_total, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(runs)} runs in {out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
