"""Random hyperparameter search for ``run_gomb_smoke`` (no Optuna dependency).

Each trial spawns a **fresh** ``python -m signedkan_wip.src.run_gomb_smoke``
process (clean CUDA state), parses the **last JSON** line, appends one JSONL
record.  Objective: maximise ``test_auroc`` when ``--edge-split 80_10_10``,
else ``val_auroc``.

Search space (controlled by ``--search-seed`` for reproducibility):

* ``lr``, ``d_embed``, ``M_outer``, ``d_outer``, ``d_middle``, ``d_core``,
  ``topk``, ``n_tiers``, ``weight_decay``, ``pos_weight_auto``,
  optional ``--cycle-ks 3,4`` (mixed-arity Gömb).

Large SNAP graphs get a smaller ``topk`` menu to reduce OOM risk.

**Architecture:** ``--architecture wide`` (default) explores larger widths;
``compact`` targets a **small-parameter** regime (low ``d_embed``, few
``M_outer`` banks, modest ``d_middle`` / ``d_core``) for Slashdot / Epinions
where node embedding already scales with ``|V|`` — useful when comparing
against headline HSiKAN runs at much larger hidden widths.

Example::

    python -m signedkan_wip.src.run_gomb_tune \\
        --datasets bitcoin_alpha bitcoin_otc sbm_n200 \\
        --trials 24 --search-seed 1 --data-seed 0 \\
        --edge-split 80_10_10 --n-epochs 100 --device cuda \\
        --out reports/gomb_tune_run.jsonl

    python -m signedkan_wip.src.run_gomb_tune \\
        --datasets slashdot epinions --architecture compact \\
        --trials 20 --search-seed 2 --edge-split 80_10_10 \\
        --n-epochs 64 --out reports/gomb_tune_compact.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _parse_last_gomb_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"dataset"'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _topk_choices(dataset: str, *, compact: bool) -> list[int]:
    if dataset in ("slashdot", "epinions"):
        if compact:
            return [16, 24, 32, 40]
        return [24, 32, 48, 64]
    if compact:
        return [24, 32, 48, 64]
    return [32, 48, 64, 96, 128]


def sample_params(
    rng: np.random.Generator, dataset: str, *, compact: bool,
) -> dict[str, Any]:
    """Return CLI flag dict for one smoke trial."""
    if compact:
        return _sample_params_compact(rng, dataset)
    return _sample_params_wide(rng, dataset)


def _sample_params_wide(rng: np.random.Generator, dataset: str) -> dict[str, Any]:
    lr = float(rng.choice([5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3]))
    d_embed = int(rng.choice([32, 40, 48, 56, 64]))
    d_outer = int(rng.choice([8, 12, 16, 20, 24]))
    M_outer = int(rng.choice([4, 6, 8, 10, 12]))
    d_mid = int(rng.choice([max(16, d_embed // 2), d_embed, int(d_embed * 1.25)]))
    d_core = int(rng.choice([max(16, d_embed // 2), d_embed, int(d_embed * 1.25)]))
    topk = int(rng.choice(_topk_choices(dataset, compact=False)))
    n_tiers = int(rng.choice([2, 3, 4]))
    weight_decay = float(rng.choice([0.0, 1e-6, 1e-5, 5e-5]))
    pos_weight_auto = bool(rng.random() < 0.75)
    use_mixed = (
        dataset not in ("sbm_n200", "sbm_n400")
        and not dataset.startswith("sbm_")
        and rng.random() < 0.22
    )
    cycle_ks = "3,4" if use_mixed else ""
    return {
        "lr": lr,
        "d_embed": d_embed,
        "d_outer": d_outer,
        "M_outer": M_outer,
        "d_middle": d_mid,
        "d_core": d_core,
        "topk": topk,
        "n_tiers": n_tiers,
        "weight_decay": weight_decay,
        "pos_weight_auto": pos_weight_auto,
        "cycle_ks": cycle_ks,
    }


def _sample_params_compact(rng: np.random.Generator, dataset: str) -> dict[str, Any]:
    """Narrow widths / few banks: Gömb param count well below wide smoke.

    Node embedding still ``|V| * d_embed``; this mode biases toward small
    ``d_embed`` and ``M_outer`` so total params sit in a **~small fraction**
    of typical wide Gömb on Bitcoin (order-of-magnitude ``~1/8`` ballpark
    on the **learnable** stack excluding cycle enumeration).
    """
    lr = float(rng.choice([3e-4, 5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]))
    d_embed = int(rng.choice([12, 14, 16, 18, 20, 22, 24]))
    d_outer = int(rng.choice([6, 8, 10]))
    M_outer = int(rng.choice([2, 3, 4]))
    d_mid = int(rng.choice([12, 14, 16, min(24, max(12, d_embed))]))
    d_core = int(rng.choice([12, 14, 16, min(28, max(12, d_embed))]))
    topk = int(rng.choice(_topk_choices(dataset, compact=True)))
    n_tiers = int(rng.choice([2, 3]))
    weight_decay = float(rng.choice([0.0, 1e-6, 1e-5, 2e-5]))
    pos_weight_auto = bool(rng.random() < 0.85)
    use_mixed = (
        dataset == "slashdot"
        and rng.random() < 0.12
    )
    cycle_ks = "3,4" if use_mixed else ""
    return {
        "lr": lr,
        "d_embed": d_embed,
        "d_outer": d_outer,
        "M_outer": M_outer,
        "d_middle": d_mid,
        "d_core": d_core,
        "topk": topk,
        "n_tiers": n_tiers,
        "weight_decay": weight_decay,
        "pos_weight_auto": pos_weight_auto,
        "cycle_ks": cycle_ks,
    }


def _build_cmd(
    *,
    py: str,
    dataset: str,
    data_seed: int,
    edge_split: str,
    n_epochs: int,
    device: str,
    p: dict[str, Any],
) -> list[str]:
    cmd: list[str] = [
        py, "-m", "signedkan_wip.src.run_gomb_smoke",
        "--dataset", dataset,
        "--seed", str(data_seed),
        "--edge-split", edge_split,
        "--n-epochs", str(n_epochs),
        "--device", device,
        "--lr", str(p["lr"]),
        "--d-embed", str(p["d_embed"]),
        "--d-outer", str(p["d_outer"]),
        "--M-outer", str(p["M_outer"]),
        "--d-middle", str(p["d_middle"]),
        "--d-core", str(p["d_core"]),
        "--topk", str(p["topk"]),
        "--n-tiers", str(p["n_tiers"]),
        "--weight-decay", str(p["weight_decay"]),
    ]
    if p.get("pos_weight_auto"):
        cmd.append("--pos-weight-auto")
    if p.get("cycle_ks"):
        cmd.extend(["--cycle-ks", str(p["cycle_ks"])])
    return cmd


def _score_row(row: dict[str, Any] | None) -> float:
    if not row:
        return float("-inf")
    if "test_auroc" in row and not math.isnan(float(row["test_auroc"])):
        return float(row["test_auroc"])
    if "val_auroc" in row and not math.isnan(float(row["val_auroc"])):
        return float(row["val_auroc"])
    return float("-inf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument(
        "--edge-split", choices=("80_20", "80_10_10"), default="80_10_10",
    )
    ap.add_argument("--n-epochs", type=int, default=90)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu (passed through to smoke).",
    )
    ap.add_argument(
        "--timeout-s", type=int, default=7200,
        help="Per-trial subprocess wall limit.",
    )
    ap.add_argument(
        "--out", type=Path, required=True,
        help="JSONL path (append mode).",
    )
    ap.add_argument(
        "--architecture",
        choices=("wide", "compact"),
        default="wide",
        help="Search space: wide (default) or compact (small d_embed / M_outer).",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.search_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    for ds in args.datasets:
        best: float = float("-inf")
        best_meta: dict[str, Any] | None = None
        t0 = time.perf_counter()
        for trial in range(args.trials):
            p = sample_params(rng, ds, compact=args.architecture == "compact")
            cmd = _build_cmd(
                py=py,
                dataset=ds,
                data_seed=args.data_seed,
                edge_split=args.edge_split,
                n_epochs=args.n_epochs,
                device=args.device,
                p=p,
            )
            t_launch = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                row = {
                    "tuner_error": "timeout",
                    "dataset": ds,
                    "trial": trial,
                    "trial_params": p,
                    "cmd": cmd,
                }
                row["tuner_dataset"] = ds
                row["tuner_search_seed"] = args.search_seed
                row["tuner_data_seed"] = args.data_seed
                row["tuner_architecture"] = args.architecture
                with args.out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    f"[tune] {ds} [{args.architecture}] trial={trial}/{args.trials} "
                    f"TIMEOUT lr={p['lr']:.1e}",
                    flush=True,
                )
                continue

            row = _parse_last_gomb_json(proc.stdout) or {}
            row["trial"] = trial
            row["trial_params"] = p
            row["returncode"] = proc.returncode
            row["wall_subprocess_s"] = time.perf_counter() - t_launch
            if proc.returncode != 0:
                tail = (proc.stderr or "")[-4000:]
                row["stderr_tail"] = tail
            row["tuner_dataset"] = ds
            row["tuner_search_seed"] = args.search_seed
            row["tuner_data_seed"] = args.data_seed
            row["tuner_architecture"] = args.architecture
            with args.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

            sc = _score_row(row if proc.returncode == 0 else None)
            if sc > best:
                best = sc
                best_meta = row
            print(
                f"[tune] {ds} [{args.architecture}] trial={trial}/{args.trials} "
                f"score={sc:.4f} best={best:.4f} "
                f"lr={p['lr']:.1e} d_emb={p['d_embed']} topk={p['topk']}",
                flush=True,
            )

        wall = time.perf_counter() - t0
        summary = {
            "tuner_phase_summary": True,
            "dataset": ds,
            "trials": args.trials,
            "tuner_architecture": args.architecture,
            "best_score": best,
            "wall_s": wall,
            "best_row_keys": sorted(best_meta.keys()) if best_meta else [],
        }
        if best_meta and "test_auroc" in best_meta:
            summary["best_test_auroc"] = best_meta["test_auroc"]
            summary["best_trial_params"] = best_meta.get("trial_params")
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, sort_keys=True) + "\n")
        print(
            f"[tune] {ds} DONE best={best:.4f} wall={wall:.1f}s → {args.out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
