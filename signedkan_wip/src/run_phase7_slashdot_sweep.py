"""Phase-7 Slashdot HSiKAN gap-closing sweep.

Grid:
    hidden     ∈ {16, 32, 64}
    max_k4     ∈ {200_000, 500_000}
    n_layers   ∈ {2, 3}
    seeds      = (0, 1, 2)
    cycle_batch_size = 10_000   (always — needed for ≥200k)

Total cells: 3 · 2 · 2 = 12 architecture configs · 3 seeds = 36 runs.
Plus SGCN+balance and MLP baselines at the same 3 seeds for reference.

Resumability: results stream to a JSONL file as each cell completes.
A crash leaves all earlier results intact; rerun skips cells already
present in the JSONL. Per-cell wall-clock timeout aborts a single
hung cell without killing the sweep.

Memory: the caller is expected to wrap the launch in `ulimit -v`.
The cycle batching keeps per-step activation memory at
O(batch · k · S · d) ≪ total cycle count.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from itertools import product
from pathlib import Path

from .run_phase2_mixed_arity import run_one_mixed
from .run_phase7_slashdot import run_mlp, run_gcn
from .run_sgcn_baseline import run_one_sgcn


def _existing_keys(out_path: Path) -> set[str]:
    """Read JSONL and return the set of (arch, seed, hidden, n_layers, max_k4)
    keys already completed."""
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
        keys.add((
            r.get("arch", ""),
            int(r.get("seed", -1)),
            int(r.get("hidden", -1)),
            int(r.get("n_layers", -1)),
            int(r.get("max_k4", -1)),
        ))
    return keys


def _append_jsonl(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


class CellTimeout(Exception):
    pass


def _with_timeout(fn, seconds: int, *args, **kwargs):
    """SIGALRM-based wall-clock timeout. Linux only."""
    def _h(_signum, _frame):
        raise CellTimeout(f"cell exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase7_slashdot_sweep.jsonl")
    ap.add_argument("--n_epochs", type=int, default=80)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden_grid", nargs="+", type=int,
                    default=[16, 32, 64])
    ap.add_argument("--max_k4_grid", nargs="+", type=int,
                    default=[200_000, 500_000])
    ap.add_argument("--n_layers_grid", nargs="+", type=int,
                    default=[2, 3])
    ap.add_argument("--cycle_batch_size", type=int, default=10_000)
    ap.add_argument("--cell_timeout_s", type=int, default=1800,
                    help="Per-cell wall-clock cap. Default 30 min.")
    ap.add_argument("--max_k3", type=int, default=30_000)
    ap.add_argument("--skip_baselines", action="store_true", default=False)
    args = ap.parse_args()

    out_path = Path(args.out)
    done = _existing_keys(out_path)
    print(f"Sweep → {out_path}")
    print(f"Already-done cells in jsonl: {len(done)}")

    t_start = time.time()

    # --- Baselines (one per seed; cheap) ---
    if not args.skip_baselines:
        for seed in args.seeds:
            for arch_name, fn, kwargs in [
                ("mlp_blind",    run_mlp,
                  dict(dataset="slashdot", seed=seed, hidden=32,
                        n_epochs=args.n_epochs)),
                ("gcn_blind",    run_gcn,
                  dict(dataset="slashdot", seed=seed, hidden=32,
                        n_layers=2, n_epochs=args.n_epochs)),
                ("sgcn_balance", run_one_sgcn,
                  dict(dataset="slashdot", seed=seed, hidden=32,
                        n_layers=2, n_epochs=args.n_epochs, lr=5e-3,
                        balance_alpha=0.5, adj_protocol="full_graph",
                        early_stopping=False, class_weighted=False,
                        weight_decay=0.0)),
            ]:
                key = (arch_name, seed, kwargs.get("hidden", -1),
                        kwargs.get("n_layers", -1), -1)
                if key in done:
                    print(f"  skip {arch_name} seed={seed} (already done)")
                    continue
                try:
                    t0 = time.time()
                    r = _with_timeout(fn, args.cell_timeout_s, **kwargs)
                    r["arch"] = arch_name
                    r["seed"] = seed
                    r["hidden"] = kwargs.get("hidden", -1)
                    r["n_layers"] = kwargs.get("n_layers", -1)
                    r["max_k4"] = -1
                    _append_jsonl(out_path, r)
                    print(f"  ✓ {arch_name:14s} seed={seed} "
                          f"AUC={r['test_auc']:.4f} "
                          f"F1m={r['test_f1_macro']:.4f}  "
                          f"{time.time()-t0:.1f}s")
                except (CellTimeout, Exception) as e:
                    print(f"  ✗ {arch_name} seed={seed} FAILED: {e!r}")

    # --- HSiKAN grid ---
    cells = list(product(args.hidden_grid, args.max_k4_grid,
                          args.n_layers_grid, args.seeds))
    print(f"\nHSiKAN cells: {len(cells)} runs total")
    for i, (hidden, max_k4, n_layers, seed) in enumerate(cells):
        arch_name = "hsikan_k34_mixed"
        key = (arch_name, seed, hidden, n_layers, max_k4)
        if key in done:
            print(f"  [{i+1:2d}/{len(cells)}] skip h={hidden} k4={max_k4} "
                  f"L={n_layers} seed={seed} (done)")
            continue
        cfg_str = (f"h={hidden:>2d} k4={max_k4:>7,} L={n_layers} "
                   f"seed={seed}")
        print(f"  [{i+1:2d}/{len(cells)}] {cfg_str} ...",
              flush=True, end="")
        try:
            t0 = time.time()
            r = _with_timeout(
                run_one_mixed,
                args.cell_timeout_s,
                "slashdot", seed,
                hidden=hidden, n_layers=n_layers, grid=3,
                n_epochs=args.n_epochs,
                arities=(3, 4),
                max_k3=args.max_k3,
                max_k4=max_k4,
                coef_smooth_lam=0.0, participation_lam=0.0,
                grad_clip=0.0, weight_decay=0.0,
                early_stopping=False, class_weighted=False,
                cycle_batch_size=args.cycle_batch_size,
            )
            r["arch"] = arch_name
            r["max_k4"] = max_k4
            _append_jsonl(out_path, r)
            print(f"  AUC={r['test_auc']:.4f} F1m={r['test_f1_macro']:.4f} "
                  f"alpha={['%.2f' % x for x in r['alpha']]} "
                  f"{time.time()-t0:.1f}s")
        except CellTimeout as e:
            print(f"  TIMEOUT ({e})")
            _append_jsonl(out_path, dict(
                arch=arch_name, seed=seed, hidden=hidden, n_layers=n_layers,
                max_k4=max_k4, status="timeout",
                elapsed_s=args.cell_timeout_s,
            ))
        except Exception as e:
            print(f"  FAILED: {e!r}")
            _append_jsonl(out_path, dict(
                arch=arch_name, seed=seed, hidden=hidden, n_layers=n_layers,
                max_k4=max_k4, status="error", error=repr(e),
            ))

    print(f"\nTotal sweep wall: {time.time()-t_start:.1f}s")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
