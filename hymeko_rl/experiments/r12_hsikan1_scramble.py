"""R12 / HSiKAN-1 closure-check T1 — structure-use (correct vs scrambled incidence).

Trains a structured HSiKAN with its intended incidence, then at INFERENCE evaluates the SAME weights under
degree-preserving SCRAMBLES (same per-node degree + edge sizes, random grouping) and a fully-random incidence. This
separates two very different readings of the Phase-1b null:

  * correct ≈ scrambled  ⇒ the model does not USE the physical structure (topology is inert given the weights);
  * correct ≫ scrambled but ≤ MLP ⇒ the model DOES use the structure, it just yields no deployment advantage here.

The HypergraphNet edge/update functions are shared across edges and node encoders are per-node, so swapping
``model.incidence`` after training is a valid, weight-preserving intervention on the message-passing topology.

Run:  python -m hymeko_rl.experiments.r12_hsikan1_scramble [epochs] [n_seeds] [n_scramble]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery.transportability_critic import (
    HypergraphNet, degree_matched_incidence, random_sparse_incidence, steiner_incidence, task_incidence)
from hymeko_rl.experiments.r12_hsikan1_ablation import (
    _OUT, _auroc, _load, _matrix, _split, _top1_k6, _train)

# the structured incidences whose structure-use we probe (name → intended incidence factory)
_STRUCTURED = {
    "A2_task_hsikan": task_incidence,
    "A3_steiner_hsikan": steiner_incidence,
}


def _eval(model: HypergraphNet, rows: list[dict], te: list[int], Xn: np.ndarray, y: np.ndarray,
          dev: str) -> "tuple[float, float]":
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(Xn[te], device=dev)).cpu().numpy()
    top1, _ = _top1_k6(rows, te, p)
    return _auroc(y[te], p), top1


def _run_mode(rows: list[dict], X: np.ndarray, y: np.ndarray, mode: str, seed: int, epochs: int,
              n_scr: int, dev: str) -> dict:
    tr, te = _split(rows, mode)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xtr, ytr = torch.tensor(Xn[tr], device=dev), torch.tensor(y[tr], device=dev)
    pos_w = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
    rng = np.random.default_rng(1000 + seed)
    out: dict = {}
    for name, factory in _STRUCTURED.items():
        intended = factory()
        torch.manual_seed(seed)
        model = HypergraphNet(intended).to(dev)
        _train(model, Xtr, ytr, epochs, pos_w, dev)                       # trained with the INTENDED structure
        auc_c, t1_c = _eval(model, rows, te, Xn, y, dev)
        deg = [_eval(_swap(model, degree_matched_incidence(rng, intended)), rows, te, Xn, y, dev)
               for _ in range(n_scr)]                                     # degree-preserving scrambles
        ran = [_eval(_swap(model, random_sparse_incidence(rng, len(intended))), rows, te, Xn, y, dev)
               for _ in range(n_scr)]                                     # fully-random incidence
        _swap(model, intended)                                           # restore (hygiene)
        out[name] = {
            "correct": {"auroc": auc_c, "top1": t1_c},
            "scrambled_deg": _ms([d[0] for d in deg], [d[1] for d in deg]),
            "scrambled_rand": _ms([r[0] for r in ran], [r[1] for r in ran]),
        }
    return out


def _swap(model: HypergraphNet, incidence: list[tuple[int, ...]]) -> HypergraphNet:
    model.incidence = incidence
    return model


def _ms(aucs: list[float], t1s: list[float]) -> dict:
    return {"auroc": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs)),
            "top1": float(np.mean(t1s)), "top1_sd": float(np.std(t1s))}


def _ci(v: list[float]) -> float:
    return float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main() -> int:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_scr = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = _load()
    X, y, _ = _matrix(rows)
    summary: dict = {"epochs": epochs, "n_seeds": n_seeds, "n_scramble": n_scr, "dev": dev}
    for mode in ("E1", "E2"):
        per_seed = [_run_mode(rows, X, y, mode, s, epochs, n_scr, dev) for s in range(n_seeds)]
        agg: dict = {}
        for name in _STRUCTURED:
            c_t1 = [ps[name]["correct"]["top1"] for ps in per_seed]
            s_t1 = [ps[name]["scrambled_deg"]["top1"] for ps in per_seed]
            c_au = [ps[name]["correct"]["auroc"] for ps in per_seed]
            s_au = [ps[name]["scrambled_deg"]["auroc"] for ps in per_seed]
            r_au = [ps[name]["scrambled_rand"]["auroc"] for ps in per_seed]
            d_t1 = [c - s for c, s in zip(c_t1, s_t1)]
            d_au = [c - s for c, s in zip(c_au, s_au)]
            agg[name] = {
                "correct_top1": round(float(np.mean(c_t1)), 3), "scr_deg_top1": round(float(np.mean(s_t1)), 3),
                "delta_top1": round(float(np.mean(d_t1)), 3), "delta_top1_ci": round(_ci(d_t1), 3),
                "correct_auroc": round(float(np.mean(c_au)), 3), "scr_deg_auroc": round(float(np.mean(s_au)), 3),
                "scr_rand_auroc": round(float(np.mean(r_au)), 3),
                "delta_auroc": round(float(np.mean(d_au)), 3), "delta_auroc_ci": round(_ci(d_au), 3),
            }
        summary[mode] = agg
        print(f"\n=== {mode} — structure-use (correct vs degree-preserving scramble, {n_scr} scr × {n_seeds} seeds) ===",
              flush=True)
        for name, a in agg.items():
            uses = "USES structure" if a["delta_auroc"] - a["delta_auroc_ci"] > 0 else "IGNORES structure (Δ≈0)"
            print(f"  {name:20s} AUROC correct {a['correct_auroc']:.3f} vs scr {a['scr_deg_auroc']:.3f} "
                  f"(Δ {a['delta_auroc']:+.3f}±{a['delta_auroc_ci']:.3f})  top1 Δ {a['delta_top1']:+.3f}"
                  f"±{a['delta_top1_ci']:.3f}  → {uses}", flush=True)
    Path(_OUT).mkdir(parents=True, exist_ok=True)
    (_OUT / "scramble_test.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote scramble_test.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
