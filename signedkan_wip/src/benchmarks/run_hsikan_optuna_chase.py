"""HSiKAN Optuna loop: keep sampling until a trial clears an AUC bar.

Default bar is ``best_competitor_mean`` from ``sota_reference.json`` — the
same headline competitor column used by ``run_hsikan_sota_gate
--mode competitor`` (mean **test** AUC from ``run_final_cell`` JSON).

Each Optuna trial is **one** ``run_final_cell`` subprocess at ``--seed``
(default 0), identical to ``run_optuna_search.objective``.

Drivers take a repo-wide **CUDA job flock** (see ``cuda_job_lock``) so only
one of {gate, Optuna, chase} runs at a time per machine unless
``HYMEKO_CUDA_DISABLE_JOB_LOCK=1``.

This is a **single-seed** search proxy. A breach here does **not** replace
multi-seed confirmation; after a hit, run::

    python -m signedkan_wip.src.benchmarks.run_hsikan_sota_gate \\
        --datasets <dataset> --mode competitor --seeds 0 1 2 3 4

Exit codes: 0 = bar breached; 2 = ``--max-total-trials`` exhausted without
breach; 1 = configuration error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import optuna

from signedkan_wip.experiments.runs.run_optuna_search import objective


def load_sota_reference() -> dict[str, Any]:
    here = Path(__file__).resolve().parent / "sota_reference.json"
    with here.open(encoding="utf-8") as f:
        return json.load(f)


def competitor_target_auc(ref: dict[str, Any], dataset: str) -> tuple[float, str]:
    targets: dict[str, Any] = ref["targets"]
    if dataset not in targets:
        raise KeyError(f"dataset {dataset!r} not in sota_reference.json targets")
    t = targets[dataset]
    raw = t.get("best_competitor_mean")
    if raw is None:
        raise KeyError(f"{dataset}: missing best_competitor_mean")
    name = str(t.get("best_competitor_name", "?"))
    return float(raw), name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        required=True,
        choices=[
            "bitcoin_alpha",
            "bitcoin_otc",
            "slashdot",
            "epinions",
            "sbm_n200",
            "sbm_n400",
        ],
    )
    ap.add_argument(
        "--target-auc",
        type=float,
        default=None,
        help="Override bar (default: best_competitor_mean for --dataset)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--run-timeout-s", type=int, default=7200)
    ap.add_argument("--trials-per-round", type=int, default=10)
    ap.add_argument(
        "--max-total-trials",
        type=int,
        default=500,
        help="Hard cap on Optuna trials attempted (including pruned).",
    )
    ap.add_argument("--study-name", required=True)
    ap.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (required for resume across processes)",
    )
    ap.add_argument("--sampler", default="tpe", choices=["tpe", "gp", "random"])
    args = ap.parse_args()

    from signedkan_wip.src.benchmarks.cuda_job_lock import cuda_job_lock

    with cuda_job_lock():
        if args.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif args.sampler == "gp":
            sampler = optuna.samplers.GPSampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)

        ref = load_sota_reference()
        if args.target_auc is not None:
            target = float(args.target_auc)
            cname = "cli_override"
        else:
            target, cname = competitor_target_auc(ref, args.dataset)

        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "target_auc": target,
                    "competitor_name": cname,
                    "max_total_trials": args.max_total_trials,
                    "trials_per_round": args.trials_per_round,
                    "seed": args.seed,
                    "n_epochs": args.n_epochs,
                },
                indent=2,
            ),
            flush=True,
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            sampler=sampler,
        )

        total_attempts = 0
        while total_attempts < args.max_total_trials:
            batch = min(
                args.trials_per_round, args.max_total_trials - total_attempts
            )
            study.optimize(
                lambda t: objective(
                    t,
                    args.dataset,
                    seed=args.seed,
                    n_epochs=args.n_epochs,
                    run_timeout_s=args.run_timeout_s,
                ),
                n_trials=batch,
            )
            total_attempts += batch

            try:
                best = float(study.best_value)
            except ValueError:
                print(
                    f"[chase] no completed trials yet after {total_attempts} "
                    "attempts",
                    flush=True,
                )
                continue

            print(
                f"[chase] attempts={total_attempts}  best_auc={best:.4f}  "
                f"target={target:.4f}",
                flush=True,
            )
            if best >= target:
                print(
                    json.dumps(
                        {
                            "status": "breach",
                            "best_auc": best,
                            "target_auc": target,
                            "best_trial": study.best_trial.number,
                            "best_params": study.best_params,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                raise SystemExit(0)

        try:
            best_final = float(study.best_value)
        except ValueError:
            best_final = None
        print(
            json.dumps(
                {
                    "status": "budget_exhausted",
                    "attempts": total_attempts,
                    "target_auc": target,
                    "best_auc": best_final,
                },
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
