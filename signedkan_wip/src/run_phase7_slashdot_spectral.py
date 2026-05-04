"""HSiKAN-mixed on Slashdot — random vs signed-Laplacian spectral init.

Hypothesis: at the cycle-bound 0.770 ceiling, the model is spending
capacity learning community structure from scratch. Initialising the
node embedding from the smallest eigvecs of the signed Laplacian
should give a head start and (if the hypothesis is right) push past
the ceiling.

Same recipe as the sweep: h=16, L=2, jk=concat, share_weights=True,
catmull-rom, lr=5e-2, no regularisers, no early stopping.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .datasets import load
from .run_phase2_mixed_arity import run_one_mixed
from .spectral_init import compute_spectral_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase7_slashdot_spectral.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--max_k4_grid", nargs="+", type=int,
                    default=[200_000, 500_000])
    ap.add_argument("--max_k3", type=int, default=30_000)
    ap.add_argument("--cycle_batch_size", type=int, default=10_000)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute spectral init ONCE per dataset (it's seed-independent
    # at the eigenvector level; we vary only the model seed).
    print("Computing signed-Laplacian spectral init for Slashdot ...")
    g = load("slashdot")
    t0 = time.time()
    spec_init = compute_spectral_init(
        g, hidden_dim=args.hidden, init_scale=0.05, seed=0,
    )
    print(f"  shape={tuple(spec_init.shape)}  "
          f"in {time.time()-t0:.1f}s")

    rows = []
    t_start = time.time()
    for max_k4 in args.max_k4_grid:
        for seed in args.seeds:
            for init_kind in ("random", "spectral"):
                cfg_str = (f"k4={max_k4:>7,} seed={seed} init={init_kind}")
                print(f"\n[{cfg_str}] ...", flush=True)
                t0 = time.time()
                kwargs = dict(
                    dataset="slashdot", seed=seed,
                    hidden=args.hidden, n_layers=args.n_layers, grid=3,
                    n_epochs=args.n_epochs,
                    arities=(3, 4),
                    max_k3=args.max_k3,
                    max_k4=max_k4,
                    coef_smooth_lam=0.0, participation_lam=0.0,
                    grad_clip=0.0, weight_decay=0.0,
                    early_stopping=False, class_weighted=False,
                    cycle_batch_size=args.cycle_batch_size,
                )
                if init_kind == "spectral":
                    kwargs["spectral_init_eigvec"] = spec_init
                try:
                    r = run_one_mixed(**kwargs)
                    r["init_kind"] = init_kind
                    r["max_k4"] = max_k4
                    rows.append(r)
                    with out_path.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"  AUC={r['test_auc']:.4f}  "
                          f"F1m={r['test_f1_macro']:.4f}  "
                          f"alpha={['%.2f' % x for x in r['alpha']]}  "
                          f"{time.time()-t0:.1f}s")
                except Exception as e:
                    print(f"  FAILED: {e!r}")

    print(f"\nTotal: {time.time()-t_start:.1f}s   results → {out_path}")

    # Quick aggregation.
    print("\n=== Median across seeds ===")
    print(f"{'max_k4':>8s}  {'init':>9s}  {'AUC_med':>8s}  "
          f"{'F1m_med':>8s}  {'Δ_AUC':>8s}")
    import statistics
    for max_k4 in args.max_k4_grid:
        cells = {init_kind: [] for init_kind in ("random", "spectral")}
        for r in rows:
            if r.get("max_k4") == max_k4:
                cells[r["init_kind"]].append(r["test_auc"])
        if not cells["random"] or not cells["spectral"]:
            continue
        med_r = statistics.median(cells["random"])
        med_s = statistics.median(cells["spectral"])
        print(f"{max_k4:>8,}  {'random':>9s}  {med_r:>8.4f}  "
              f"{statistics.median([r['test_f1_macro'] for r in rows if r.get('max_k4')==max_k4 and r['init_kind']=='random']):>8.4f}  "
              f"{'':>8s}")
        print(f"{max_k4:>8,}  {'spectral':>9s}  {med_s:>8.4f}  "
              f"{statistics.median([r['test_f1_macro'] for r in rows if r.get('max_k4')==max_k4 and r['init_kind']=='spectral']):>8.4f}  "
              f"{med_s-med_r:>+8.4f}")


if __name__ == "__main__":
    main()
