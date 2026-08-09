"""R12.2-B ranker test — does a STRUCTURED model exploit orientation better than a flat MLP?

On the orientation-varying dataset (`r12_2a_orientation_dataset` fed by the orientation-aware bank), train each of
MLP / random-sparse / task-HSiKAN / Steiner / degree-matched WITH and WITHOUT the orientation feature ([sin,cos] yaw,
appended to the object_state node for the HSiKANs). The metric that matters is the per-handoff top-1 K6 (rank a
handoff's candidate θ by predicted P, take top-1, read its K6), and the quantity of interest is the INTERACTION

    Δ_arch = top1K6(with orientation) − top1K6(without) ,   then  Δ_task-HSiKAN − Δ_MLP .

Δ_HSiKAN − Δ_MLP > 0 ⇒ the physical structure turns orientation into selection skill better than a flat model — the
R12.2 hypothesis. ≈0 ⇒ orientation is a shared-input lever (helps flat and structured alike). Split: E1 = unseen
scenario (a scenario's all yaw×θ move together). Single family (O5-R), so no cross-family E2 here.

Run:  python -m hymeko_rl.experiments.r12_2b_ranker [family] [epochs] [n_seeds] [holdout_scenario]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery.transportability_critic import MatchedModels, build_input_row, count_params
from hymeko_rl.experiments.r12_hsikan1_ablation import _auroc, _train

_OUT = Path("reports/2026-08-08-r12-2-orientation")
_HOLDOUT = "bank_c3_r7_a-15"        # E1: held-out scenario (unseen at train), all its yaw×θ move together


def _load(fam: str) -> list[dict]:
    return [json.loads(ln) for ln in (_OUT / f"orientation_dataset_{fam}.jsonl").read_text().splitlines()]


def _matrix(rows: list[dict], orientation: bool) -> "tuple[np.ndarray, np.ndarray]":
    X = np.asarray([build_input_row(r["x"], r["theta"], r["handoff_family"],
                                    r["post_grasp_yaw_deg"] if orientation else None) for r in rows], np.float32)
    y = np.asarray([float(r["k6"]) for r in rows], np.float32)
    return X, y


def _split(rows: list[dict], holdout: str) -> "tuple[list[int], list[int]]":
    tr, te = [], []
    for i, r in enumerate(rows):
        (te if r["scenario"] == holdout else tr).append(i)
    return tr, te


def _handoff_key(r: dict) -> tuple:
    return (r["handoff_family"], r["scenario"], r["seed"], r["yaw_deg"])   # a handoff = object at a yaw in a scenario


def _top1_k6(rows: list[dict], idx: list[int], p: np.ndarray) -> "tuple[float, float]":
    by: dict[tuple, list[int]] = {}
    for j, i in enumerate(idx):
        by.setdefault(_handoff_key(rows[i]), []).append(j)
    top1 = oracle = 0
    for js in by.values():
        k6 = np.asarray([rows[idx[j]]["k6"] for j in js], float)
        top1 += int(k6[int(np.argmax(p[js]))])
        oracle += int(k6.max() > 0)
    n = max(1, len(by))
    return top1 / n, oracle / n


def _run_seed(rows: list[dict], holdout: str, seed: int, epochs: int, dev: str) -> dict:
    tr, te = _split(rows, holdout)
    tk = {_handoff_key(rows[i]) for i in tr}
    ek = {_handoff_key(rows[i]) for i in te}
    assert not (tk & ek), f"LEAKAGE: {len(tk & ek)} handoffs in both splits"
    out: dict = {}
    for orientation in (False, True):
        X, y = _matrix(rows, orientation)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        Xtr, ytr = torch.tensor(Xn[tr], device=dev), torch.tensor(y[tr], device=dev)
        pos_w = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
        for name, model in MatchedModels(orientation=orientation).build(seed).items():
            model.to(dev)
            _train(model, Xtr, ytr, epochs, pos_w, dev)
            model.eval()
            with torch.no_grad():
                p_te = model(torch.tensor(Xn[te], device=dev)).cpu().numpy()
            top1, oracle = _top1_k6(rows, te, p_te)
            out.setdefault(name, {})["with" if orientation else "without"] = top1
            out[name]["auroc_" + ("with" if orientation else "without")] = _auroc(y[te], p_te)
            out[name]["params"] = count_params(model)
            out[name]["oracle"] = oracle
    return out


def _ci(v: list[float]) -> float:
    return float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main() -> int:
    fam = sys.argv[1] if len(sys.argv) > 1 else "O5-R"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    holdout = sys.argv[4] if len(sys.argv) > 4 else _HOLDOUT
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = _load(fam)
    names = list(MatchedModels().build(0))
    runs = [_run_seed(rows, holdout, s, epochs, dev) for s in range(n_seeds)]

    agg: dict = {}
    for name in names:
        wi = [r[name]["with"] for r in runs]
        wo = [r[name]["without"] for r in runs]
        dl = [w - o for w, o in zip(wi, wo)]
        agg[name] = {"without": round(float(np.mean(wo)), 3), "with": round(float(np.mean(wi)), 3),
                     "delta": round(float(np.mean(dl)), 3), "delta_ci": round(_ci(dl), 3),
                     "auroc_without": round(float(np.mean([r[name]["auroc_without"] for r in runs])), 3),
                     "auroc_with": round(float(np.mean([r[name]["auroc_with"] for r in runs])), 3),
                     "params": runs[0][name]["params"], "oracle": round(float(np.mean([r[name]["oracle"] for r in runs])), 3)}
    # interaction: task-HSiKAN gains from orientation MINUS MLP gains (the R12.2 hypothesis)
    d_task = [r["A2_task_hsikan"]["with"] - r["A2_task_hsikan"]["without"] for r in runs]
    d_mlp = [r["A0_mlp"]["with"] - r["A0_mlp"]["without"] for r in runs]
    interaction = [dt - dm for dt, dm in zip(d_task, d_mlp)]
    inter_mean, inter_ci = round(float(np.mean(interaction)), 3), round(_ci(interaction), 3)
    verdict = ("STRUCTURE EXPLOITS ORIENTATION — Δ_task-HSiKAN − Δ_MLP > 0 (CI-excl-0)" if inter_mean - inter_ci > 0 else
               "orientation is a SHARED lever — Δ_HSiKAN − Δ_MLP ≈ 0 (structure gains no more than flat)")

    print(f"\n=== R12.2-B ranker [{fam}], {n_seeds} seeds × {epochs}ep, holdout {holdout}, oracle "
          f"{agg['A0_mlp']['oracle']:.2f} ===", flush=True)
    print("model                without  with   Δ(with−without)      AUROC wo→w", flush=True)
    for name, a in agg.items():
        print(f"  {name:20s} {a['without']:.3f}  {a['with']:.3f}  {a['delta']:+.3f}±{a['delta_ci']:.3f}   "
              f"{a['auroc_without']:.3f}→{a['auroc_with']:.3f}", flush=True)
    print(f"\nINTERACTION Δ_task-HSiKAN − Δ_MLP = {inter_mean:+.3f}±{inter_ci:.3f} ⇒ {verdict}", flush=True)

    summary = {"family": fam, "epochs": epochs, "n_seeds": n_seeds, "holdout": holdout, "dev": dev,
               "per_model": agg, "interaction": inter_mean, "interaction_ci": inter_ci, "verdict": verdict}
    (_OUT / "b_ranker.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote b_ranker.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
