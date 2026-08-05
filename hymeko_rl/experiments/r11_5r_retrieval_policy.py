"""R11.5R — characterize the nearest-robust-basin RETRIEVAL delivery policy (the density-curve-indicated teacher-free
deployment form). Build the table from the robust bank's train demos (with recert survival), then measure closed-loop
strict-K6 coverage per split for each (metric x rule) cell: train is LEAVE-ONE-OUT (self-retrieval is trivially 1.0),
held-out (dev+test) uses the full-train table. Reconstruct each scenario snapshot ONCE; parallelize by scenario shard.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, fresh_rig, reconstruct_capture, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.evaluate import rollout_theta
from hymeko_rl.coin_delivery.delivery_bc.retrieval import (
    RetrievalConfig,
    RetrievalDeliveryPolicy,
    RetrievalDeploymentCertificate,
    SelectRule,
    freeze_table,
    load_frozen,
)
from hymeko_rl.experiments.r11_4b_conditioned_bc import _load_dataset

RECERT = Path("reports/2026-08-05-r11-5r-robust-teacher/merged.json")
B1_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
DEFAULT_OUT = Path("reports/2026-08-06-r11-5r-retrieval")
DEPLOY = "std_nearest"                            # the deployment default (== the density-curve baseline)
_WIDE = "WIDE_RECERTIFIED"

# (metric x rule) ablation cells. nearest is k-independent; widest_basin/dist_weighted use k=3.
CELLS: "list[tuple[str, RetrievalConfig]]" = [
    ("std_nearest", RetrievalConfig(True, 1, SelectRule.NEAREST)),
    ("std_widest3", RetrievalConfig(True, 3, SelectRule.WIDEST_BASIN)),
    ("std_weighted3", RetrievalConfig(True, 3, SelectRule.DIST_WEIGHTED)),
    ("raw_nearest", RetrievalConfig(False, 1, SelectRule.NEAREST)),
    ("raw_widest3", RetrievalConfig(False, 3, SelectRule.WIDEST_BASIN)),
    ("raw_weighted3", RetrievalConfig(False, 3, SelectRule.DIST_WEIGHTED)),
]


def survival_map(recert: Path) -> "dict[str, float]":
    """scenario_id -> local-K6 survival@1% (robust t1 for WIDE, nominal-fallback t0 otherwise)."""
    out: dict[str, float] = {}
    for r in json.loads(recert.read_text())["rows"]:
        if r.get("status") == _WIDE:
            out[r["scenario_id"]] = float(r["t1"]["survival1"])
        elif "t0" in r:
            out[r["scenario_id"]] = float(r["t0"]["survival1"])
    return out


def _train_table(samples: "list[Any]", surv: "dict[str, float]") -> "tuple[np.ndarray, np.ndarray, np.ndarray, dict]":
    train = sorted((s for s in samples if s.split == "train"), key=lambda s: s.scenario_id)   # stable order for LOO idx
    X = np.array([s.x for s in train], np.float64)
    T = np.array([s.theta for s in train], np.float64)
    S = np.array([surv.get(s.scenario_id, 0.0) for s in train], np.float64)
    return X, T, S, {s.scenario_id: i for i, s in enumerate(train)}


def _slice(seq: list, offset: int, limit: int) -> list:
    return seq[offset:(offset + limit if limit else len(seq))]


def _split_rate(rows: "list[dict]", cell: str, split: str) -> float:
    sub = [r for r in rows if r.get("split") == split and cell in r]
    return round(sum(1 for r in sub if r[cell]["k6"]) / len(sub), 3) if sub else 0.0


def _held_rate(rows: "list[dict]", cell: str) -> float:
    sub = [r for r in rows if r.get("split") in ("dev", "test") and cell in r]
    return round(sum(1 for r in sub if r[cell]["k6"]) / len(sub), 3) if sub else 0.0


def _held_dtz(rows: "list[dict]", cell: str) -> float:
    sub = [r[cell]["dtz_mm"] for r in rows if r.get("split") in ("dev", "test") and cell in r]
    return round(float(np.mean(sub)), 2) if sub else 0.0


def _retrieval_verdict(agg: "dict[str, dict]", best: str) -> str:
    """Descriptive: retrieval is a deployable teacher-free form; flag whether its best held-out cell clears the 0.50 bar."""
    return ("R11_5R_RETRIEVAL_DEPLOYABLE_MEETS_050" if agg[best]["held"] >= 0.50
            else "R11_5R_RETRIEVAL_DEPLOYABLE_BELOW_050")


def aggregate(rows: "list[dict]") -> "dict[str, Any]":
    cells = [name for name, _ in CELLS]
    agg = {c: {"train_loo": _split_rate(rows, c, "train"), "dev": _split_rate(rows, c, "dev"),
               "test": _split_rate(rows, c, "test"), "held": _held_rate(rows, c), "dtz_held": _held_dtz(rows, c)}
           for c in cells}
    best = max(cells, key=lambda c: (agg[c]["held"], -agg[c]["dtz_held"]))
    dcfg = dict(CELLS)[DEPLOY]
    cert = RetrievalDeploymentCertificate(True, True, True, dcfg.k, dcfg.select.value, dcfg.standardize,
                                          agg[DEPLOY]["train_loo"], agg[DEPLOY]["dev"], agg[DEPLOY]["test"])
    return {"cells": agg, "best_cell": best, "deploy_cell": DEPLOY, "deploy_beaten_by": best if best != DEPLOY else None,
            "certificate": cert.__dict__, "is_deployable": cert.is_deployable(), "verdict": _retrieval_verdict(agg, best)}


def _run_eval(args: argparse.Namespace) -> None:
    cfg, conf, obj = bc_context()
    samples = _load_dataset(args.dataset_dir)
    X, T, S, idx_of = _train_table(samples, survival_map(args.recert))
    policies = [(name, RetrievalDeliveryPolicy.fit(X, T, S, rc)) for name, rc in CELLS]
    order = sorted(samples, key=lambda s: (s.split, s.scenario_id))
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for smp in _slice(order, args.offset, args.limit):
        rc = reconstruct_capture(fresh_rig(), cfg, conf, obj, scenario_by_id(smp.scenario_id), smp.seed)
        if rc is None:
            rows.append({"scenario_id": smp.scenario_id, "split": smp.split, "error": "reconstruction_failed"})
            continue
        snap = rc.result.outcome.snapshot
        x = np.array(smp.x, np.float64)
        excl = idx_of.get(smp.scenario_id) if smp.split == "train" else None       # leave-one-out for train
        row: dict[str, Any] = {"scenario_id": smp.scenario_id, "split": smp.split}
        for name, pol in policies:
            row[name] = rollout_theta(snap, pol.predict(x, exclude_idx=excl))
        rows.append(row)
        print(f"{smp.scenario_id:26s} {smp.split:5s} " + "  ".join(
            f"{n}={row[n]['k6']:d}" for n, _ in policies if n in row), flush=True)
    with (args.out / f"cells_{args.offset:03d}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _run_merge(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(args.out / "cells_*.jsonl"))):
        rows += [json.loads(line) for line in Path(f).open() if line.strip()]
    result = aggregate([r for r in rows if "error" not in r])
    (args.out / "retrieval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=== R11.5R RETRIEVAL CHARACTERIZATION ===", flush=True)
    print(json.dumps({k: result[k] for k in ("cells", "best_cell", "is_deployable", "verdict")}, indent=2), flush=True)
    print("R11_5R_RETRIEVAL_DONE", flush=True)


# the shipped deployment configs (user: std_weighted3 = best generalizer PRIMARY, std_nearest = in-distribution control)
PRIMARY_CFG = dict(CELLS)["std_weighted3"]
CONTROL_CFG = dict(CELLS)["std_nearest"]


def _run_freeze(args: argparse.Namespace) -> None:
    """Write the SELF-CONTAINED frozen deployment artifact (train robust-theta table + primary/control config +
    characterization + provenance). R11.6C loads this — the retrieval policy is shipped as a frozen baseline."""
    samples = _load_dataset(args.dataset_dir)
    X, T, S, _idx = _train_table(samples, survival_map(args.recert))
    train_ids = sorted(s.scenario_id for s in samples if s.split == "train")
    table = freeze_table(train_ids, X, T, S, PRIMARY_CFG)
    char = json.loads((args.out / "retrieval.json").read_text()) if (args.out / "retrieval.json").exists() else {}
    table_md5 = hashlib.md5(json.dumps(table["X"], sort_keys=True).encode()).hexdigest()   # noqa: S324 (integrity id, not security)
    payload = {"table": table, "primary_config": PRIMARY_CFG.to_json(), "control_config": CONTROL_CFG.to_json(),
               "n_train": len(train_ids), "characterization": char.get("cells", {}),
               "certificate": char.get("certificate", {}),
               "provenance": {"dataset_dir": str(args.dataset_dir), "recert": str(args.recert), "table_md5": table_md5}}
    out = args.out / "frozen_policy.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pol = load_frozen(table)                                    # sanity: the artifact round-trips to a usable policy
    _ = pol.predict(np.asarray(table["X"], np.float64)[0])
    print(f"FROZEN_POLICY_WRITTEN {out} n_train={len(train_ids)} md5={table_md5} primary={PRIMARY_CFG.select.value}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("eval", "merge", "freeze"), required=True)
    ap.add_argument("--dataset-dir", type=Path, default=B1_DATASET)
    ap.add_argument("--recert", type=Path, default=RECERT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"eval": _run_eval, "merge": _run_merge, "freeze": _run_freeze}[args.phase](args)


if __name__ == "__main__":
    main()
