"""R11.6D Phase 4 — calibrate + evaluate the transportability-aware retrieval on the transfer matrix.

Weights are chosen train-only by leave-one-scenario-out (the query's own theta unusable). Then: deployment train-like
(own theta available), train LOSO (generalization), and dev — each as TOP-1 complete-theta selection (no blending),
against the descriptor-nearest baseline. Reports ranking quality AND top-1 closed-loop K6 (the decisive metric). The
TEST split is sealed. No runtime CEM / oracle / teacher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer
from hymeko_rl.coin_delivery.transport_retrieval import (
    WEIGHT_GRID,
    build_signatures,
    cell_index,
    evaluate_handoff,
    score,
)
from hymeko_rl.experiments.r11_4b_conditioned_bc import _load_dataset

MATRIX = Path("reports/2026-08-06-r11-6d-matrix/matrix.json")
FROZEN = Path("reports/2026-08-06-r11-5r-retrieval/frozen_policy.json")
B1_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
DEFAULT_OUT = Path("reports/2026-08-06-r11-6d-retrieval")
C3_FAR = ["bank_c3_r7_a+45", "bank_c3_r9_a-30", "bank_c3_r9_a-15"]


def _handoff_qf(cells: "list[dict]") -> "dict[str, dict]":
    qf: dict[str, dict] = {}
    for c in cells:
        qf.setdefault(c["handoff"], {"d_required_mm": c["d_required_mm"], "bearing": c["bearing"], "split": c["split"]})
    return qf


def _auroc(labels: "list[bool]", scores: "list[float]") -> "float | None":
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _loso_score(idx: dict, qf: dict, loso_sigs: dict, thetas: list, train: list, w: Any) -> "tuple[float, float]":
    res = [evaluate_handoff(idx, sj, qf[sj], loso_sigs[sj], [t for t in thetas if t != sj], w) for sj in train]
    rate = round(float(np.mean([r["k6"] for r in res])), 3)
    regrets = [r["regret"] for r in res if r["regret"] is not None]
    return rate, round(float(np.mean(regrets)), 2) if regrets else 0.0


def _calibrate(cells: list, idx: dict, qf: dict, thetas: list, train: list) -> "tuple[Any, dict]":
    """Leave-one-scenario-out over train (own theta + own handoff removed); pick weights by (top-1 K6, -mean regret)."""
    loso_sigs = {sj: build_signatures(cells, exclude=frozenset({sj})) for sj in train}
    best: "tuple[tuple, Any, float] | None" = None
    for w in WEIGHT_GRID:
        rate, regret = _loso_score(idx, qf, loso_sigs, thetas, train, w)
        key = (rate, -regret)
        if best is None or key > best[0]:
            best = (key, w, regret)
    assert best is not None
    return best[1], {"loso_k6": best[0][0], "mean_regret": best[2]}


def _nearest_baseline(idx: dict, panel: list, thetas: list, x_by: dict, X: np.ndarray, std: Standardizer) -> float:
    """Descriptor-nearest top-1 K6 on a panel (the R11.6C retrieval control)."""
    Xs = std.transform(X)
    hits = 0
    for hid in panel:
        i = int(np.argmin(np.linalg.norm(Xs - std.transform(x_by[hid]), axis=1)))
        c = idx.get((hid, thetas[i]))
        hits += bool(c and c["k6"] and c["safe"])
    return round(hits / len(panel), 3) if panel else 0.0


def _handoff_auroc(idx: dict, qf: dict, sigs: dict, h: str, thetas: list, w: Any) -> "float | None":
    labels = [bool(idx.get((h, t)) and idx[(h, t)]["k6"] and idx[(h, t)]["safe"]) for t in thetas if t in sigs]
    scores = [score(qf[h], sigs[t], w) for t in thetas if t in sigs]
    return _auroc(labels, scores)


def _panel_eval(idx: dict, qf: dict, sigs: dict, panel: list, thetas: list, w: Any) -> "dict[str, Any]":
    rows = {h: evaluate_handoff(idx, h, qf[h], sigs, thetas, w) for h in panel}
    aurocs = [a for a in (_handoff_auroc(idx, qf, sigs, h, thetas, w) for h in panel) if a is not None]
    return {"top1_k6": round(float(np.mean([r["k6"] for r in rows.values()])), 3),
            "top3_deliverable": round(float(np.mean([r["top3_deliverable"] for r in rows.values()])), 3),
            "mean_auroc": round(float(np.mean(aurocs)), 3) if aurocs else None, "per_scenario": rows}


def _verdict(dep: float, dev: dict, c3_k6: int, base_dev: float, r7_ok: bool) -> str:
    if dep >= 1.0 and dev["top1_k6"] >= 6 / 7 and c3_k6 >= 2 and dev["top1_k6"] > base_dev and r7_ok:
        return "R11_6D_TRANSPORTABILITY_AWARE_RETRIEVAL_PASS"
    if dev["mean_auroc"] and dev["mean_auroc"] >= 0.6:
        return "R11_6D_TRANSPORTABILITY_SIGNAL_POSITIVE_SELECTION_UNCALIBRATED"
    return "R11_6D_TRANSPORT_SIGNATURE_INSUFFICIENT"


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
    X, std = np.asarray(fp["table"]["X"], np.float64), Standardizer.fit(np.asarray(fp["table"]["X"], np.float64))
    x_by = {s.scenario_id: np.asarray(s.x, np.float64) for s in _load_dataset(args.dataset_dir)}

    w, cal = _calibrate(cells, idx, qf, thetas, train)
    all_sigs = build_signatures(cells)
    dep = _panel_eval(idx, qf, all_sigs, train, thetas, w)            # deployment (own theta available)
    dev_eval = _panel_eval(idx, qf, all_sigs, dev, thetas, w)
    out = _assemble(w, cal, dep, dev_eval,
                    _nearest_baseline(idx, dev, thetas, x_by, X, std),
                    _nearest_baseline(idx, train, thetas, x_by, X, std))   # nearest=self on train -> ~1.0 memorization
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "retrieval.json").write_text(json.dumps({**out, "dev_per_scenario": dev_eval["per_scenario"]}, indent=2),
                                             encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    print("R11_6D_RETRIEVAL_DONE", flush=True)


def _assemble(w: Any, cal: dict, dep: dict, dev_eval: dict, base_dev: float, base_train_loso: float) -> "dict[str, Any]":
    c3_rows = {h: dev_eval["per_scenario"][h] for h in C3_FAR if h in dev_eval["per_scenario"]}
    c3_k6 = sum(1 for r in c3_rows.values() if r["k6"])
    r7_ok = bool(c3_rows.get("bank_c3_r7_a+45", {}).get("k6", False))
    return {"weights": w.__dict__, "calibration": cal, "deployment_train_like": dep["top1_k6"],
            "train_loso_k6": cal["loso_k6"], "baseline_train_loso_k6": base_train_loso,
            "dev": {k: dev_eval[k] for k in ("top1_k6", "top3_deliverable", "mean_auroc")},
            "baseline_dev_k6": base_dev, "c3_far_angle_k6": f"{c3_k6}/3",
            "c3_detail": {h: {"top1": r["top1"], "k6": r["k6"], "sel_dtz": r["sel_dtz"], "regret": r["regret"]}
                          for h, r in c3_rows.items()},
            "verdict": _verdict(dep["top1_k6"], dev_eval, c3_k6, base_dev, r7_ok)}


if __name__ == "__main__":
    main()
