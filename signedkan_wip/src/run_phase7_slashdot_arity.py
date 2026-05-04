"""HSiKAN-mixed on Slashdot — does adding k=5 break the 0.77 ceiling?

Three configurations at h=16 / L=2 / cycle_batch_size=10k / 3 seeds:
  k34_500k   — baseline (the sweep ceiling)
  k4only_500k — drops k=3 (αₖ already at 0.91 weight on k=4)
  k345       — adds k=5 to the mix
  k34_1M     — pushes max_k4 higher (cycle-saturation check)

Hypothesis: if 0.77 holds across all four, the ceiling is structural
(community structure, not arity gap). If k345 or k34_1M moves it,
the bottleneck is the cycle distribution.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .run_phase2_mixed_arity import run_one_mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase7_slashdot_arity.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=80)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(
        dataset="slashdot",
        hidden=16, n_layers=2, grid=3,
        n_epochs=args.n_epochs,
        coef_smooth_lam=0.0, participation_lam=0.0,
        grad_clip=0.0, weight_decay=0.0,
        early_stopping=False, class_weighted=False,
        cycle_batch_size=10_000,
    )

    cells = [
        ("k34_500k",      dict(arities=(3, 4),
                                 max_k3=30_000, max_k4=500_000)),
        ("k4only_500k",   dict(arities=(4,),
                                 max_k4=500_000)),
        ("k345",          dict(arities=(3, 4, 5),
                                 max_k3=30_000,
                                 max_k4=500_000,
                                 max_k5=500_000)),
        ("k34_1M",        dict(arities=(3, 4),
                                 max_k3=30_000,
                                 max_k4=1_000_000)),
    ]

    rows = []
    t_start = time.time()
    for cell_name, cfg in cells:
        for seed in args.seeds:
            tag = f"{cell_name} seed={seed}"
            print(f"\n[{tag}] ...", flush=True)
            t0 = time.time()
            try:
                r = run_one_mixed(seed=seed, **common, **cfg)
                r["cell"] = cell_name
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
    print("\n=== Median across seeds ===")
    import statistics
    print(f"{'cell':<14s}  {'AUC_med':>8s}  {'F1m_med':>8s}  {'alpha':>15s}")
    for cell_name, _ in cells:
        cell_rows = [r for r in rows if r.get("cell") == cell_name]
        if not cell_rows:
            continue
        aucs = [r["test_auc"] for r in cell_rows]
        f1ms = [r["test_f1_macro"] for r in cell_rows]
        alphas = cell_rows[0].get("alpha", [])
        amed = [round(statistics.median([r["alpha"][i] for r in cell_rows]), 2)
                 for i in range(len(alphas))]
        print(f"{cell_name:<14s}  {statistics.median(aucs):>8.4f}  "
              f"{statistics.median(f1ms):>8.4f}  {str(amed):>15s}")


if __name__ == "__main__":
    main()
