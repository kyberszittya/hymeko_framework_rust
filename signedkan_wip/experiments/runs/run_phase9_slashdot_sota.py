"""Phase 9 — Slashdot SOTA chase.

Targeted ablation to close the Slashdot AUC gap (HSiKAN 0.77 → SGCN
~0.91). Combines new architectural levers from late session:
  - attention M_e (init-fixed for near-uniform start)
  - direct messaging (SGCN-style sign-conditional path alongside cycles)
  - more cycles per arity (max=100k)
  - larger hidden dim (h=32)

Each cell varies one or more of {attention, direct, max_per, h, λ}
on Slashdot k34 (k=5 dropped — too slow on Slashdot).

Resumable JSONL output. Cells already in the file are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from itertools import product
from pathlib import Path

from .run_phase2_mixed_arity import run_one_mixed


# (cell_name, attention, direct, max_per, hidden, balance_lambda)
CELLS = [
    ("baseline_30k",        False, False, 30_000,  16, 0.0),
    ("baseline_30k_g3_fix", False, False, 30_000,  16, 0.0),  # original recipe
    ("baseline_100k",       False, False, 100_000, 16, 0.0),
    ("balance_30k_l05",     False, False, 30_000,  16, 0.05),
    ("balance_100k_l05",    False, False, 100_000, 16, 0.05),
    ("attn_30k",            True,  False, 30_000,  16, 0.0),
    ("attn_100k",           True,  False, 100_000, 16, 0.0),
    ("direct_30k",          False, True,  30_000,  16, 0.0),
    ("direct_100k",         False, True,  100_000, 16, 0.0),
    ("attn_direct_30k",     True,  True,  30_000,  16, 0.0),
    ("attn_direct_100k",    True,  True,  100_000, 16, 0.0),
    ("h32_direct_100k",     False, True,  100_000, 32, 0.0),
    ("h32_attn_direct_100k", True, True,  100_000, 32, 0.0),
]


def _existing_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    keys = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((r.get("cell"), r.get("seed")))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase9_slashdot_sota.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=80)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_keys(out_path)
    print(f"Resuming from {len(done)} done cells", flush=True)

    rows = []
    t_start = time.time()
    for cell_name, attn, direct, max_per, hidden, lam in CELLS:
        for seed in args.seeds:
            key = (cell_name, seed)
            if key in done:
                continue
            # Pick recipe variant (one cell uses morning's grid=3 + fixed lr).
            if cell_name == "baseline_30k_g3_fix":
                grid, lr_sched = 3, "fixed"
            else:
                grid, lr_sched = 5, "cosine"

            tag = f"{cell_name:<22s} seed={seed}"
            print(f"[{tag}] ...", flush=True, end="")
            t0 = time.time()
            try:
                r = run_one_mixed(
                    "slashdot", seed=seed,
                    hidden=hidden, n_layers=2, grid=grid,
                    n_epochs=args.n_epochs,
                    arities=(3, 4),
                    max_per_arity={3: 30_000, 4: max_per},
                    coef_smooth_lam=0.0, participation_lam=0.0,
                    grad_clip=0.0, weight_decay=0.0,
                    early_stopping=False, class_weighted=False,
                    lr_schedule=lr_sched,
                    feature_edges="all",
                    m_e_mode="edge_in_cycle",
                    balance_lambda=lam,
                    attention_m_e=attn,
                    direct_messaging=direct,
                    cycle_batch_size=10_000 if max_per > 50_000 else None,
                )
                r["cell"] = cell_name
                rows.append(r)
                with out_path.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"  AUC={r['test_auc']:.4f}  "
                      f"alpha={['%.2f' % x for x in r['alpha']]}  "
                      f"{time.time()-t0:.0f}s",
                      flush=True)
            except Exception as e:
                print(f"  FAILED: {e!r}", flush=True)
                with out_path.open("a") as f:
                    f.write(json.dumps(dict(
                        cell=cell_name, seed=seed,
                        status="error", error=repr(e),
                    )) + "\n")

    print(f"\nTotal phase 9: {time.time()-t_start:.0f}s", flush=True)
    print("\n=== Median across seeds ===", flush=True)
    print(f"{'cell':<22s}  {'AUC_med':>8s}  {'std':>6s}  {'F1m_med':>8s}", flush=True)
    rows_all = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    rows_all = [r for r in rows_all if r.get("test_auc") is not None]
    for cell_name, *_ in CELLS:
        cell_rows = [r for r in rows_all if r.get("cell") == cell_name]
        if not cell_rows:
            continue
        aucs = [r["test_auc"] for r in cell_rows]
        f1ms = [r["test_f1_macro"] for r in cell_rows]
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        print(f"{cell_name:<22s}  {statistics.median(aucs):>8.4f}  "
              f"{std:>6.4f}  {statistics.median(f1ms):>8.4f}",
              flush=True)


if __name__ == "__main__":
    main()
