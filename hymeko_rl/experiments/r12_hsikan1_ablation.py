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


def _run_one(rows: list[dict], X: np.ndarray, y: np.ndarray, mode: str, seed: int, epochs: int, dev: str) -> list[dict]:
    tr, te = _split(rows, mode)
    tk = {(rows[i]["handoff_family"], rows[i]["scenario"], rows[i]["seed"]) for i in tr}
    ek = {(rows[i]["handoff_family"], rows[i]["scenario"], rows[i]["seed"]) for i in te}
    assert not (tk & ek), f"LEAKAGE: {len(tk & ek)} handoffs in both splits"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xtr, ytr = torch.tensor(Xn[tr], device=dev), torch.tensor(y[tr], device=dev)
    pos_w = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
    out = []
    for name, model in MatchedModels().build(seed).items():
        model.to(dev)
        _train(model, Xtr, ytr, epochs, pos_w, dev)
        model.eval()
        with torch.no_grad():
            p_te = model(torch.tensor(Xn[te], device=dev)).cpu().numpy()
        top1, oracle = _top1_k6(rows, te, p_te)
        out.append({"model": name, "params": count_params(model), "auroc": _auroc(y[te], p_te),
                    "top1_k6": top1, "oracle_k6": oracle, "regret": oracle - top1})
    return out


def _agg(runs: list[list[dict]]) -> list[dict]:
    """Mean ± 95% CI across seeds, per model."""
    names = [r["model"] for r in runs[0]]
    agg = []
    for k, name in enumerate(names):
        stat = {}
        for m in ("auroc", "top1_k6", "regret", "oracle_k6"):
            v = np.array([run[k][m] for run in runs], float)
            stat[m] = round(float(v.mean()), 3)
            stat[f"{m}_ci"] = round(float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, 3)
        agg.append({"model": name, "params": runs[0][k]["params"], **stat})
    return agg


def main() -> int:
    if sys.argv[1:2] == ["sweep"]:
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 80
        n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        rows = _load()
        X, y, _ = _matrix(rows)
        summary: dict = {"epochs": epochs, "n_seeds": n_seeds, "dev": dev}
        for mode in ("E1", "E2"):
            runs = [_run_one(rows, X, y, mode, s, epochs, dev) for s in range(n_seeds)]
            agg = _agg(runs)
            summary[mode] = agg
            print(f"\n=== {mode} ({n_seeds} seeds, {epochs} ep) — mean±95%CI ===", flush=True)
            for a in agg:
                print(f"  {a['model']:20s} AUROC {a['auroc']:.3f}±{a['auroc_ci']:.3f}  "
                      f"top1_K6 {a['top1_k6']:.3f}±{a['top1_k6_ci']:.3f}  regret {a['regret']:.3f} "
                      f"(oracle {a['oracle_k6']:.2f})", flush=True)
        _OUT.mkdir(parents=True, exist_ok=True)
        (_OUT / "ablation_sweep.json").write_text(json.dumps(summary, indent=1))
        print("\nwrote ablation_sweep.json", flush=True)
        return 0
    # single run
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    mode = sys.argv[2] if len(sys.argv) > 2 else "E1"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = _load()
    X, y, _ = _matrix(rows)
    res = _run_one(rows, X, y, mode, seed, epochs, dev)
    for r in res:
        print(f"  {r['model']:20s} AUROC={r['auroc']:.3f} top1_K6={r['top1_k6']:.3f} regret={r['regret']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
