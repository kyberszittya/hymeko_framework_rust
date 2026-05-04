"""SignedKAN — Phase 3.6 hyperparameter sweep.

Per DECISIONS.md / plan §3.6:
  3 seeds × 3 learning rates × 2 hidden dims = 18 runs on Bitcoin Alpha
  (primary), best config replicated on Bitcoin OTC (secondary).

Per-run config: 100 epochs, batch_size = full-batch (no minibatch
required after the vectorisation fix in train.py), grid=5, k=3.

Run:
    python3 -m src.run_phase3_sweep --dataset bitcoin_alpha
    python3 -m src.run_phase3_sweep --dataset bitcoin_otc \
                                     --hidden-grid 32 --lr-grid 5e-2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .train import TrainConfig, train


SWEEP_DIR = Path("signedkan_wip/experiments/results/phase3_sweep")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha",
                    choices=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr-grid", type=float, nargs="+",
                    default=[1e-2, 5e-2, 1e-1])
    ap.add_argument("--hidden-grid", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--n-epochs", type=int, default=100)
    args = ap.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SWEEP_DIR / f"sweep_{args.dataset}.log"
    summary_path = SWEEP_DIR / f"sweep_{args.dataset}.json"
    summary = []
    t_total = time.time()
    n_runs = len(args.seeds) * len(args.lr_grid) * len(args.hidden_grid)
    print(f"Phase-3 sweep: {n_runs} runs on {args.dataset}\n")
    with log_path.open("w") as logf:
        for hidden in args.hidden_grid:
            for lr in args.lr_grid:
                for seed in args.seeds:
                    cfg = TrainConfig(
                        dataset=args.dataset, hidden=hidden,
                        lr=lr, n_epochs=args.n_epochs, seed=seed,
                        log_every=args.n_epochs,    # only end-of-run log
                        out_dir=SWEEP_DIR,
                    )
                    print(f"  hidden={hidden:>3} lr={lr:.0e} seed={seed} ...")
                    t0 = time.time()
                    res = train(cfg)
                    dt = time.time() - t0
                    record = dict(
                        hidden=hidden, lr=lr, seed=seed,
                        test_auc=res["test"]["auc"],
                        test_f1_binary=res["test"]["f1_binary"],
                        test_f1_macro=res["test"]["f1_macro"],
                        n_params=res["n_params"],
                        elapsed_s=dt,
                    )
                    summary.append(record)
                    logf.write(json.dumps(record) + "\n")
                    logf.flush()
                    print(f"    auc={record['test_auc']:.4f}  "
                          f"f1_bin={record['test_f1_binary']:.4f}  "
                          f"f1_mac={record['test_f1_macro']:.4f}  "
                          f"({dt:.1f}s)")

    summary_path.write_text(json.dumps(summary, indent=2))
    elapsed = time.time() - t_total
    print(f"\nSweep finished in {elapsed:.0f}s. Summary at {summary_path}")

    # Best by macro-F1.
    best = max(summary, key=lambda r: r["test_f1_macro"])
    print(f"\nBest by test macro-F1:")
    print(f"  hidden={best['hidden']} lr={best['lr']:.0e} seed={best['seed']}")
    print(f"  test_auc={best['test_auc']:.4f}  "
          f"f1_bin={best['test_f1_binary']:.4f}  "
          f"f1_mac={best['test_f1_macro']:.4f}")

    # Aggregate by (hidden, lr) — mean ± std across seeds.
    print(f"\nMean ± std across {len(args.seeds)} seeds per (hidden, lr):")
    print(f"  {'hidden':>6}  {'lr':>6}  {'AUC':>14}  {'F1bin':>14}  {'F1mac':>14}")
    import statistics as s
    for hidden in args.hidden_grid:
        for lr in args.lr_grid:
            group = [r for r in summary if r["hidden"] == hidden and r["lr"] == lr]
            aucs = [r["test_auc"] for r in group]
            f1bs = [r["test_f1_binary"] for r in group]
            f1ms = [r["test_f1_macro"] for r in group]
            print(f"  {hidden:>6}  {lr:>6.0e}  "
                  f"{s.mean(aucs):.3f} ± {s.stdev(aucs) if len(aucs)>1 else 0:.3f}  "
                  f"{s.mean(f1bs):.3f} ± {s.stdev(f1bs) if len(f1bs)>1 else 0:.3f}  "
                  f"{s.mean(f1ms):.3f} ± {s.stdev(f1ms) if len(f1ms)>1 else 0:.3f}")


if __name__ == "__main__":
    main()
