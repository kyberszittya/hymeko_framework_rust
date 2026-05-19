"""Run a focused sweep that adds the spectral-entropy Lyapunov-safe
regulariser to SignedKAN training and compares it against the
unregularised SignedKAN baseline.

Sweep axes (kept narrow for tonight's compute budget):
  * λ_0 ∈ {0.001, 0.01, 0.05}   — base regularisation strength
  * H_target ∈ {0.5, 0.7}        — Kolmogorov-Arnold-style floor
  * datasets = bitcoin_alpha (and OTC if time allows)
  * hidden = 32, lr = 5e-2, 100 epochs (the converged baseline)
  * seeds = 3

Reports test AUC, macro-F1, plus the trained-state H_norm and
last λ_eff so the regulariser's behaviour is visible.

Run:
  python -m signedkan_wip.experiments.runs.run_entropy_sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_compare import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--lam0-grid", nargs="+", type=float,
                    default=[0.001, 0.01, 0.05])
    ap.add_argument("--target-grid", nargs="+", type=float,
                    default=[0.5, 0.7])
    ap.add_argument("--out",
                    default="signedkan_wip/experiments/results/entropy_sweep.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        # Unregularised baseline reference (re-runs at this hidden/lr).
        for seed in args.seeds:
            r = run_one("signedkan", dataset, args.hidden, seed,
                         args.n_epochs, lr=args.lr)
            print(f"  baseline       {dataset:14s} h={args.hidden} "
                  f"seed={seed}  AUC={r['test_auc']:.4f}  "
                  f"F1_mac={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:,}  {r['elapsed_s']:.1f}s")
            results.append(r)
        # Entropy-regularised arm × HP grid.
        for lam0 in args.lam0_grid:
            for tgt in args.target_grid:
                for seed in args.seeds:
                    r = run_one("signedkan_entropy", dataset,
                                 args.hidden, seed, args.n_epochs,
                                 lr=args.lr, entropy_lam0=lam0,
                                 entropy_target=tgt)
                    r["entropy_target"] = tgt
                    print(f"  entropy_reg λ0={lam0:.3f} H*={tgt:.2f}  "
                          f"{dataset:14s} seed={seed}  "
                          f"AUC={r['test_auc']:.4f}  "
                          f"F1_mac={r['test_f1_macro']:.4f}  "
                          f"H_n={r['last_h_norm']:.3f}  "
                          f"λ_eff={r['last_lam_eff']:.4f}  "
                          f"{r['elapsed_s']:.1f}s")
                    results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
