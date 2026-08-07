"""R12 / HSiKAN-1 Phase 1 — architecture ablation (offline ranking).

Trains A0 MLP + A1 random-sparse / A2 task-contact-hypergraph / A3 Steiner + A3c degree-matched HSiKAN at matched
budget on the R11.7B transportability dataset, and evaluates the metric that matters: per-handoff TOP-1 K6 (rank the
handoff's candidate θ by predicted P(K6), take the top-1, read its dataset K6 label) — the offline proxy for the
closed-loop physical gate (Phase 2 does the real rollout). Also AUROC + oracle regret. Scenario-level split with a
leakage assertion; E1 = unseen scenario, E2 = unseen family.

Run:  python -m hymeko_rl.experiments.r12_hsikan1_ablation [epochs] [eval=E1|E2] [seed]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.transportability_critic import (
    MatchedModels, build_input_row, count_params)

_OUT = Path("reports/2026-08-07-r12-hsikan1")
_FAMILIES = ("O0", "O1-L", "O2-M", "O4-S")
_HOLDOUT_SCENARIO = "bank_c1_+0.01_+0.00"     # E1: unseen scenario (last of the 6, held out for all families)
_HOLDOUT_FAMILY = "O4-S"                       # E2: unseen family


def _load() -> list[dict]:
    rows: list[dict] = []
    for fam in _FAMILIES:
        p = _OUT / f"dataset_{fam}.jsonl"
        rows += [json.loads(ln) for ln in p.read_text().splitlines()]
    return rows


def _split(rows: list[dict], mode: str) -> "tuple[list[int], list[int]]":
    tr, te = [], []
    for i, r in enumerate(rows):
        if mode == "E1":
            (te if r["scenario"] == _HOLDOUT_SCENARIO else tr).append(i)
        else:                                  # E2: unseen family
            (te if r["handoff_family"] == _HOLDOUT_FAMILY else tr).append(i)
    return tr, te


def _matrix(rows: list[dict]) -> "tuple[np.ndarray, np.ndarray, list[tuple]]":
    X = np.asarray([build_input_row(r["x"], r["theta"], r["handoff_family"]) for r in rows], np.float32)
    y = np.asarray([float(r["k6"]) for r in rows], np.float32)
    key = [(r["handoff_family"], r["scenario"], r["seed"]) for r in rows]   # handoff identity (for top-1 grouping)
    return X, y, key


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _top1_k6(rows: list[dict], idx: list[int], p: np.ndarray) -> "tuple[float, float]":
    """Per-handoff: rank its θ by predicted p, take top-1, read its K6. Returns (top1_K6_rate, oracle_rate)."""
    by: dict[tuple, list[int]] = {}
    for j, i in enumerate(idx):
        by.setdefault((rows[i]["handoff_family"], rows[i]["scenario"], rows[i]["seed"]), []).append(j)
    top1 = oracle = 0
    for js in by.values():
        pj = p[js]
        k6 = np.asarray([rows[idx[j]]["k6"] for j in js], float)
        top1 += int(k6[int(np.argmax(pj))])
        oracle += int(k6.max() > 0)
    n = len(by)
    return top1 / n, oracle / n


def _train(model: nn.Module, Xtr: torch.Tensor, ytr: torch.Tensor, epochs: int, pos_w: float, dev: str) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))
    model.train()
    n = len(ytr)
    for _ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, 256):
            bi = perm[b:b + 256]
            opt.zero_grad()
            loss = lossf(model(Xtr[bi]), ytr[bi])
            loss.backward()
            opt.step()


def main() -> int:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    mode = sys.argv[2] if len(sys.argv) > 2 else "E1"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    rows = _load()
    X, y, _key = _matrix(rows)
    tr, te = _split(rows, mode)
    # leakage assertion: no handoff (family,scenario,seed) in both splits
    tk = {(rows[i]["handoff_family"], rows[i]["scenario"], rows[i]["seed"]) for i in tr}
    ek = {(rows[i]["handoff_family"], rows[i]["scenario"], rows[i]["seed"]) for i in te}
    assert not (tk & ek), f"LEAKAGE: {len(tk & ek)} handoffs in both splits"

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xtr = torch.tensor(Xn[tr], device=dev)
    ytr = torch.tensor(y[tr], device=dev)
    pos_w = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
    print(f"R12 ablation {mode} seed{seed} dev={dev}: train {len(tr)} / test {len(te)} pairs, "
          f"test handoffs {len(ek)}, pos_weight {pos_w:.2f}, epochs {epochs}", flush=True)

    results = []
    for name, model in MatchedModels().build(seed).items():
        model.to(dev)
        _train(model, Xtr, ytr, epochs, pos_w, dev)
        model.eval()
        with torch.no_grad():
            p_te = model(torch.tensor(Xn[te], device=dev)).cpu().numpy()
        auroc = _auroc(y[te], p_te)
        top1, oracle = _top1_k6(rows, te, p_te)
        results.append({"model": name, "params": count_params(model), "auroc": round(auroc, 3),
                        "top1_k6": round(top1, 3), "oracle_k6": round(oracle, 3),
                        "regret": round(oracle - top1, 3)})
        print(f"  {name:20s} params={count_params(model):6d} AUROC={auroc:.3f} "
              f"top1_K6={top1:.3f} oracle={oracle:.3f} regret={oracle - top1:.3f}", flush=True)

    # budget-match assertion: A0-A3 params within ±20% of the median
    pc = [r["params"] for r in results]
    med = float(np.median(pc))
    band = all(0.8 * med <= p <= 1.2 * med for p in pc)
    res = {"mode": mode, "seed": seed, "epochs": epochs, "n_train": len(tr), "n_test": len(te),
           "budget_matched_within_20pct": band, "param_median": med, "results": results}
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / f"ablation_{mode}_seed{seed}.json").write_text(json.dumps(res, indent=1))
    print(f"\nbudget matched (±20%): {band} (median {med:.0f} params). wrote ablation_{mode}_seed{seed}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
