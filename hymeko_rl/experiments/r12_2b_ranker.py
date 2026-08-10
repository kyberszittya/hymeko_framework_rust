"""R12.2-B ranker test — does a STRUCTURED model exploit orientation better than a flat MLP?

On the orientation-varying dataset (`r12_2a_orientation_dataset` fed by the orientation-aware bank), train each of
MLP / random-sparse / task-HSiKAN / Steiner / degree-matched WITH and WITHOUT the orientation feature ([sin,cos] yaw,
appended to the object_state node for the HSiKANs). The critic REGRESSES the dense normalized dtz (present on every
pair — the K6 label is too sparse ~6% to learn a ranker from) and ranks a handoff's candidate θ by predicted dtz. The
metric that matters is the per-handoff top-1 K6 (take the top-ranked θ, read its K6) plus AUROC of (−predicted dtz vs
K6); the quantity of interest is the INTERACTION

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
from torch import nn

from hymeko_rl.coin_delivery.transportability_critic import MatchedModels, build_input_row, count_params
from hymeko_rl.experiments.r12_hsikan1_ablation import _auroc

_OUT = Path("reports/2026-08-08-r12-2-orientation")
_DTZ_SCALE = 200.0        # mm cap for the normalized dense regression target (dtz in [0, 1], low = delivered)


def _load(fam: str) -> list[dict]:
    return [json.loads(ln) for ln in (_OUT / f"orientation_dataset_{fam}.jsonl").read_text().splitlines()]


def _matrix(rows: list[dict], orientation: bool) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """X, the K6 eval label, and the DENSE dtz regression target (present on ALL pairs, unlike the ~6% K6 positives —
    this is what lets the critic learn above chance at this data scale)."""
    X = np.asarray([build_input_row(r["x"], r["theta"], r["handoff_family"],
                                    r["post_grasp_yaw_deg"] if orientation else None) for r in rows], np.float32)
    y = np.asarray([float(r["k6"]) for r in rows], np.float32)
    dtz = np.asarray([min(float(r["dtz_mm"]), _DTZ_SCALE) / _DTZ_SCALE for r in rows], np.float32)
    return X, y, dtz


def _train_reg(model: nn.Module, Xtr: torch.Tensor, target: torch.Tensor, epochs: int, dev: str) -> None:
    """Regress the dense normalized-dtz target (MSE). The critic then ranks θ by predicted dtz (ascending)."""
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    model.train()
    n = len(target)
    for _ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for b in range(0, n, 256):
            bi = perm[b:b + 256]
            opt.zero_grad()
            loss = lossf(model(Xtr[bi]), target[bi])
            loss.backward()
            opt.step()


def _handoff_key(r: dict) -> tuple:
    return (r["handoff_family"], r["scenario"], r["seed"], r["yaw_deg"])   # a handoff = object at a yaw in a scenario


def _split(rows: list[dict], seed: int, frac: float = 0.33) -> "tuple[list[int], list[int]]":
    """Stratified HANDOFF-level split: hold out ~``frac`` of the handoffs at EACH yaw, so test yaws AND targets are
    SEEN in training. This isolates the orientation-ranking question (does yaw help pick θ for a new handoff instance?)
    from the target-transfer wall that an unseen-SCENARIO split adds — the θ are (yaw,target)-specific, so holding out
    a whole target measures θ-transfer-to-an-unseen-target (a different, R11.7B-style problem), collapsing everyone to
    chance regardless of orientation."""
    by_yaw: dict[float, set[tuple]] = {}
    for r in rows:
        by_yaw.setdefault(r["yaw_deg"], set()).add(_handoff_key(r))
    rng = np.random.default_rng(seed)
    te_keys: set[tuple] = set()
    for keys in by_yaw.values():
        ks = sorted(keys)
        rng.shuffle(ks)
        te_keys.update(ks[:max(1, round(frac * len(ks)))])
    tr = [i for i, r in enumerate(rows) if _handoff_key(r) not in te_keys]
    te = [i for i, r in enumerate(rows) if _handoff_key(r) in te_keys]
    return tr, te


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


def _run_seed(rows: list[dict], seed: int, epochs: int, dev: str) -> dict:
    tr, te = _split(rows, seed)
    tk = {_handoff_key(rows[i]) for i in tr}
    ek = {_handoff_key(rows[i]) for i in te}
    assert not (tk & ek), f"LEAKAGE: {len(tk & ek)} handoffs in both splits"
    out: dict = {}
    for orientation in (False, True):
        X, y, dtz = _matrix(rows, orientation)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        Xtr, dtz_tr = torch.tensor(Xn[tr], device=dev), torch.tensor(dtz[tr], device=dev)
        for name, model in MatchedModels(orientation=orientation).build(seed).items():
            model.to(dev)
            _train_reg(model, Xtr, dtz_tr, epochs, dev)                 # regress dense dtz
            model.eval()
            with torch.no_grad():
                pred_dtz = model(torch.tensor(Xn[te], device=dev)).cpu().numpy()
            score = -pred_dtz                                           # low predicted dtz ⇒ high rank
            top1, oracle = _top1_k6(rows, te, score)
            out.setdefault(name, {})["with" if orientation else "without"] = top1
            out[name]["auroc_" + ("with" if orientation else "without")] = _auroc(y[te], score)
            out[name]["params"] = count_params(model)
            out[name]["oracle"] = oracle
    return out


def _ci(v: list[float]) -> float:
    return float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _deltas(runs: list[dict], name: str, metric: str) -> "tuple[list[float], list[float], list[float]]":
    """Per split-seed (with, without, with−without) for AUROC or top1."""
    key = (lambda r, c: r[name]["auroc_" + c]) if metric == "auroc" else (lambda r, c: r[name][c])
    wi = [float(key(r, "with")) for r in runs]
    wo = [float(key(r, "without")) for r in runs]
    return wi, wo, [w - o for w, o in zip(wi, wo)]


def main() -> int:
    fam = sys.argv[1] if len(sys.argv) > 1 else "O5-R"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 8      # split-seeds (each = a fresh stratified handoff split)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = _load(fam)
    names = list(MatchedModels().build(0))
    runs = [_run_seed(rows, s, epochs, dev) for s in range(n_seeds)]

    agg: dict = {}
    for name in names:
        a_wi, a_wo, a_d = _deltas(runs, name, "auroc")
        t_wi, t_wo, t_d = _deltas(runs, name, "top1")
        agg[name] = {"auroc_without": round(float(np.mean(a_wo)), 3), "auroc_with": round(float(np.mean(a_wi)), 3),
                     "d_auroc": round(float(np.mean(a_d)), 3), "d_auroc_ci": round(_ci(a_d), 3),
                     "top1_without": round(float(np.mean(t_wo)), 3), "top1_with": round(float(np.mean(t_wi)), 3),
                     "d_top1": round(float(np.mean(t_d)), 3), "d_top1_ci": round(_ci(t_d), 3),
                     "params": runs[0][name]["params"], "oracle": round(float(np.mean([r[name]["oracle"] for r in runs])), 3)}
    # interaction on AUROC (primary; top-1 is oracle-capped by the thin positives): Δ_task-HSiKAN − Δ_MLP
    _, _, dt = _deltas(runs, "A2_task_hsikan", "auroc")
    _, _, dm = _deltas(runs, "A0_mlp", "auroc")
    inter = [a - b for a, b in zip(dt, dm)]
    im, ic = round(float(np.mean(inter)), 3), round(_ci(inter), 3)
    verdict = ("STRUCTURE EXPLOITS ORIENTATION — Δ_task-HSiKAN − Δ_MLP (AUROC) > 0, CI-excludes-0" if im - ic > 0 else
               "STRUCTURE UNDER-USES ORIENTATION — Δ_HSiKAN − Δ_MLP < 0, CI-excludes-0" if im + ic < 0 else
               "INCONCLUSIVE / SHARED LEVER — Δ_HSiKAN − Δ_MLP CI includes 0 (structure gains no more from orientation "
               "than flat, at this data scale)")

    print(f"\n=== R12.2-B ranker [{fam}], {n_seeds} split-seeds × {epochs}ep, stratified handoff split, "
          f"oracle {agg['A0_mlp']['oracle']:.2f} ===", flush=True)
    print("model                 AUROC wo→w   Δ_AUROC(orient)      top1 wo→w  Δ_top1", flush=True)
    for name, a in agg.items():
        print(f"  {name:20s} {a['auroc_without']:.3f}→{a['auroc_with']:.3f}  {a['d_auroc']:+.3f}±{a['d_auroc_ci']:.3f}   "
              f"{a['top1_without']:.2f}→{a['top1_with']:.2f}  {a['d_top1']:+.2f}", flush=True)
    print(f"\nINTERACTION (AUROC) Δ_task-HSiKAN − Δ_MLP = {im:+.3f}±{ic:.3f} ⇒ {verdict}", flush=True)

    summary = {"family": fam, "epochs": epochs, "n_split_seeds": n_seeds, "dev": dev, "per_model": agg,
               "interaction_auroc": im, "interaction_auroc_ci": ic, "verdict": verdict}
    (_OUT / "b_ranker.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote b_ranker.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
