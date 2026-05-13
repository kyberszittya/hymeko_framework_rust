"""Multi-seed test-AUC gate vs documented HSiKAN headline numbers.

Loads targets from ``sota_reference.json`` (from ``HSIKAN_STATE_2026_05_04.md``)
and runs ``run_final_cell.cell_signed_graph`` for each seed.  Compares the
**empirical mean ± std** of this run's test AUC to:

  * ``self`` (default): pass if mean ≥ ``hsikan_mean - k·hsikan_std``
    (regression guard; k defaults to 2 where std is known).
  * ``table``: pass only if mean ≥ ``hsikan_mean`` (aspirational).
  * ``competitor``: pass if mean ≥ ``best_competitor_mean`` (global-SOTA bar).

Usage (defaults to **CUDA** when available, else CPU)::

    python -m signedkan_wip.src.benchmarks.run_hsikan_sota_gate \\
        --datasets bitcoin_alpha bitcoin_otc --seeds 0 1 2 3 4 \\
        --hidden 16 --n-epochs 80

Force CPU::

    python -m signedkan_wip.src.benchmarks.run_hsikan_sota_gate \\
        --datasets bitcoin_alpha --seeds 0 --device cpu

Refuse to run without a GPU (exit code 2 if no CUDA, or if ``--device`` is
not ``cuda``)::

    python -m signedkan_wip.src.benchmarks.run_hsikan_sota_gate \\
        --datasets bitcoin_alpha bitcoin_otc --seeds 0 1 2 3 4 \\
        --require-cuda

**VRAM (8 GB class):** keep the GPU exclusive to this process. Optionally set
``--cycle-batch 2000`` (chunked forward, matches Optuna) and/or lower
``--max-k4`` / ``--max-k3`` if you still hit OOM while building sparse
``M_e`` incidence — tuple caps dominate peak allocation before training.

With ``--device cuda``, this driver takes the **CUDA job flock** so it does
not overlap ``run_optuna_search`` / ``run_hsikan_optuna_chase`` on the same
host (see ``cuda_job_lock``; disable with ``HYMEKO_CUDA_DISABLE_JOB_LOCK=1``).

Single-seed Optuna search until one trial clears the same competitor bar
(``run_hsikan_optuna_chase``); use this gate afterward for multi-seed proof.

Exit code 0 iff every selected dataset passes under the chosen mode.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def _load_reference() -> dict[str, Any]:
    here = Path(__file__).resolve().parent / "sota_reference.json"
    with here.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets", nargs="+", required=True,
        help="Subset of keys in sota_reference.json targets",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(5)),
        help="Seeds forwarded to cell_signed_graph (split + init)",
    )
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--max-k4", type=int, default=200_000)
    ap.add_argument(
        "--max-k3",
        type=int,
        default=None,
        help="If set, export HSIKAN_MAX_K3 for this process (default env: 30000). "
        "Lower to shrink sparse M_e before forward.",
    )
    ap.add_argument(
        "--cycle-batch",
        type=int,
        default=None,
        metavar="N",
        help="If set, export HSIKAN_CYCLE_BATCH=N (chunked encode_edges). "
        "Unset for Bitcoin matches historical no-batch path; 2000–4000 "
        "helps ~8 GB GPUs when attention is off.",
    )
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda when available (default), else cpu; use --device cpu to force CPU.",
    )
    ap.add_argument(
        "--mode", choices=("self", "table", "competitor"), default="self",
        help="self: mean ≥ published_mean - k·std; table: mean ≥ mean; "
        "competitor: mean ≥ best_competitor_mean",
    )
    ap.add_argument(
        "--k-sigma", type=float, default=2.0,
        help="Slack multiplier for self-mode when hsikan_std is set",
    )
    ap.add_argument(
        "--require-cuda",
        action="store_true",
        help="Require CUDA: exit 2 if unavailable; forbid --device cpu.",
    )
    args = ap.parse_args()

    if args.cycle_batch is not None:
        os.environ["HSIKAN_CYCLE_BATCH"] = str(args.cycle_batch)
    if args.max_k3 is not None:
        os.environ["HSIKAN_MAX_K3"] = str(args.max_k3)

    if args.require_cuda:
        if args.device != "cuda":
            print(
                "[gate] --require-cuda needs --device cuda "
                f"(got {args.device!r})",
                file=sys.stderr,
            )
            sys.exit(2)
        if not torch.cuda.is_available():
            print(
                "[gate] --require-cuda set but CUDA is not available",
                file=sys.stderr,
            )
            sys.exit(2)

    ref = _load_reference()
    device = torch.device(args.device)

    from signedkan_wip.src.benchmarks.cuda_job_lock import cuda_job_lock
    from signedkan_wip.src.run_final_cell import cell_signed_graph

    lock_ctx = cuda_job_lock() if device.type == "cuda" else contextlib.nullcontext()
    with lock_ctx:
        _run_gate_after_lock(args, ref, device, cell_signed_graph)


def _run_gate_after_lock(
    args: argparse.Namespace,
    ref: dict[str, Any],
    device: torch.device,
    cell_signed_graph: Callable[..., Any],
) -> None:
    targets: dict[str, Any] = ref["targets"]
    all_pass = True
    print(json.dumps({"citation": ref["citation"], "mode": args.mode}, indent=2))
    for ds in args.datasets:
        if ds not in targets:
            print(f"[skip] unknown dataset {ds!r}", file=sys.stderr)
            continue
        t = targets[ds]
        aucs: list[float] = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            out = cell_signed_graph(
                ds, "HSiKAN", args.hidden, args.n_epochs, args.max_k4, device,
                seed=seed,
            )
            if out is None:
                raise RuntimeError(f"cell_signed_graph returned None for {ds}")
            aucs.append(float(out["auc"]))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        mean = float(statistics.mean(aucs))
        std_run = float(statistics.pstdev(aucs)) if len(aucs) > 1 else 0.0

        h_mean = float(t["hsikan_mean"])
        h_std = t.get("hsikan_std")
        comp = t.get("best_competitor_mean")

        if args.mode == "table":
            thr = h_mean
            label = "table_mean"
        elif args.mode == "competitor":
            if comp is None:
                print(f"[skip] {ds}: no competitor target", file=sys.stderr)
                continue
            thr = float(comp)
            label = "competitor_best_mean"
        else:
            if h_std is None:
                thr = h_mean
                label = "self_mean_no_std_fallback"
            else:
                thr = h_mean - args.k_sigma * float(h_std)
                label = f"self_mean_minus_{args.k_sigma:g}sigma"

        ok = mean >= thr
        all_pass = all_pass and ok
        row = {
            "dataset": ds,
            "seeds": list(args.seeds),
            "n_seeds": len(aucs),
            "aucs": aucs,
            "mean": mean,
            "std_run": std_run,
            "target_mode": args.mode,
            "threshold": thr,
            "threshold_label": label,
            "published_hsikan_mean": h_mean,
            "published_hsikan_std": h_std,
            "pass": ok,
        }
        if comp is not None:
            row["published_competitor_mean"] = float(comp)
            row["published_competitor_name"] = t.get("best_competitor_name")
        print(json.dumps(row, indent=2))
        if not ok:
            print(
                f"[FAIL] {ds}: mean={mean:.4f} < threshold={thr:.4f} ({label})",
                file=sys.stderr,
            )
        else:
            print(f"[PASS] {ds}: mean={mean:.4f} ≥ {thr:.4f} ({label})", flush=True)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
