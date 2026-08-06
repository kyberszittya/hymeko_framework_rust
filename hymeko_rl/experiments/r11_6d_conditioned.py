"""R11.6D Phase 4.1 — handoff-conditioned transport predictor: calibrate + evaluate against the baseline.

Fit a ridge predictor of delivered dtz for a (theta, handoff) pair on the train transfer matrix, select the top-1 theta
by minimum predicted dtz. Lambda is chosen by train leave-one-scenario-out (query handoff + own theta removed). Then
deployment / dev top-1 K6 vs descriptor-nearest. Interpretable (linear coefficients). No new rollouts; test sealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer
from hymeko_rl.coin_delivery.transport_predictor import RidgePredictor, select_top1, training_rows
from hymeko_rl.coin_delivery.transport_retrieval import build_signatures, cell_index
from hymeko_rl.experiments.r11_4b_conditioned_bc import _load_dataset
from hymeko_rl.experiments.r11_6d_transport_retrieval import C3_FAR, _handoff_qf, _nearest_baseline

MATRIX = Path("reports/2026-08-06-r11-6d-matrix/matrix.json")
FROZEN = Path("reports/2026-08-06-r11-5r-retrieval/frozen_policy.json")
B1_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
DEFAULT_OUT = Path("reports/2026-08-06-r11-6d-conditioned")
LAMBDAS = (1.0, 5.0, 10.0, 30.0, 100.0)


def _delivers(idx: dict, h: str, t: str) -> bool:
    c = idx.get((h, t))
    return bool(c and c["k6"] and c["safe"])


def _loso_k6(cells: list, idx: dict, qf: dict, loso_sigs: dict, thetas: list, train: list, lam: float) -> float:
    hits = 0
    for sj in train:
        sigs = loso_sigs[sj]
        phi, y = training_rows(cells, sigs, qf, drop_handoffs=frozenset({sj}), drop_theta=sj)
        top1 = select_top1(RidgePredictor.fit(phi, y, lam), qf[sj], sigs, [t for t in thetas if t != sj])
        hits += _delivers(idx, sj, top1)
    return round(hits / len(train), 3)


def _panel_k6(pred: RidgePredictor, idx: dict, qf: dict, sigs: dict, panel: list, thetas: list) -> "dict[str, Any]":
    rows = {h: select_top1(pred, qf[h], sigs, thetas) for h in panel}
    return {"top1_k6": round(float(np.mean([_delivers(idx, h, t) for h, t in rows.items()])), 3),
            "selected": {h: {"top1": t, "k6": _delivers(idx, h, t), "dtz": idx.get((h, t), {}).get("dtz_mm")}
                         for h, t in rows.items()}}


def _verdict(dep: float, dev: float, c3: int, base_dev: float, r7_ok: bool) -> str:
    if dep >= 1.0 and dev >= 6 / 7 and c3 >= 2 and dev > base_dev and r7_ok:
        return "R11_6D_HANDOFF_CONDITIONED_TRANSPORT_PASS"
    if dev > base_dev:
        return "R11_6D_HANDOFF_CONDITIONED_BEATS_BASELINE_BELOW_GATE"
    return "R11_6D_HANDOFF_CONDITIONING_NO_GAIN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=MATRIX)
    ap.add_argument("--frozen", type=Path, default=FROZEN)
    ap.add_argument("--dataset-dir", type=Path, default=B1_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    m = json.loads(args.matrix.read_text())
    cells, thetas = m["cells"], sorted(m["thetas"])
    idx, qf = cell_index(cells), _handoff_qf(cells)
    train = sorted(h for h, v in qf.items() if v["split"] == "train")
    dev = sorted(h for h, v in qf.items() if v["split"] == "dev")
    fp = json.loads(args.frozen.read_text())
    X = np.asarray(fp["table"]["X"], np.float64)
    x_by = {s.scenario_id: np.asarray(s.x, np.float64) for s in _load_dataset(args.dataset_dir)}
    base_dev = _nearest_baseline(idx, dev, thetas, x_by, X, Standardizer.fit(X))

    loso_sigs = {sj: build_signatures(cells, exclude=frozenset({sj})) for sj in train}
    lam, loso = max(((la, _loso_k6(cells, idx, qf, loso_sigs, thetas, train, la)) for la in LAMBDAS),
                    key=lambda z: z[1])
    all_sigs = build_signatures(cells)
    phi, y = training_rows(cells, all_sigs, qf)
    pred = RidgePredictor.fit(phi, y, lam)
    dep = _panel_k6(pred, idx, qf, all_sigs, train, thetas)
    dev_eval = _panel_k6(pred, idx, qf, all_sigs, dev, thetas)
    out = _assemble(lam, loso, dep, dev_eval, base_dev, pred)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "conditioned.json").write_text(json.dumps({**out, "dev_selected": dev_eval["selected"]}, indent=2),
                                               encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    print("R11_6D_CONDITIONED_DONE", flush=True)


def _assemble(lam: float, loso: float, dep: dict, dev_eval: dict, base_dev: float,
              pred: RidgePredictor) -> "dict[str, Any]":
    c3 = {h: dev_eval["selected"][h] for h in C3_FAR if h in dev_eval["selected"]}
    c3_k6 = sum(1 for r in c3.values() if r["k6"])
    r7_ok = bool(c3.get("bank_c3_r7_a+45", {}).get("k6", False))
    return {"lambda": lam, "train_loso_k6": loso, "deployment_train_like": dep["top1_k6"],
            "dev_top1_k6": dev_eval["top1_k6"], "baseline_dev_k6": base_dev,
            "beats_baseline": dev_eval["top1_k6"] > base_dev, "c3_far_k6": f"{c3_k6}/3", "c3_detail": c3,
            "coefficients": pred.coefficients(),
            "verdict": _verdict(dep["top1_k6"], dev_eval["top1_k6"], c3_k6, base_dev, r7_ok)}


if __name__ == "__main__":
    main()
