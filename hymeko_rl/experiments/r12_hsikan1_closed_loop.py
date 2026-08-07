"""R12 / HSiKAN-1 closure-check T3 — physical closed-loop top-1 for ALL FOUR models on one frozen panel.

Not just the MLP: MLP / random-sparse / task-HSiKAN / Steiner each rank a handoff's pooled-θ candidates, pick their
own top-1, and that θ is ROLLED OUT physically (`_delivery_signals`, a fresh deterministic sim) on the identical frozen
panel of held-out handoffs. Answers the deployment question the offline ranking cannot settle alone: does the offline
top-1 ordering survive real rollout, and how do the four compare head-to-head under one selection protocol?

The dataset's K6 labels are themselves physical rollouts, so a fresh roll of a pooled θ must reproduce them — we assert
that agreement (a reproduction check), then report per-model physical top-1 K6.

Run:  python -m hymeko_rl.experiments.r12_hsikan1_closed_loop [train_seeds] [mode=E1|E2|both]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import clip_theta
from hymeko_rl.coin_delivery.exact_zero_composition import _delivery_signals, reach_capture_descriptor
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.coin_delivery.transportability_critic import MatchedModels, build_input_row
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r12_hsikan1_dataset import _FAMILIES, _SCENARIOS, _pooled_thetas
from hymeko_rl.experiments.r12_hsikan1_ablation import (
    _HOLDOUT_FAMILY, _HOLDOUT_SCENARIO, _OUT, _load, _matrix, _split, _train)

_HANDOFF_SEEDS = range(5)          # the physical handoff grid the dataset was built on (fixed)


def _panel(mode: str) -> list[tuple[str, str, int]]:
    """Frozen held-out handoff panel for a mode: E1 = the held-out scenario across all families; E2 = the held-out
    family across all scenarios."""
    if mode == "E1":
        return [(f, _HOLDOUT_SCENARIO, s) for f in _FAMILIES for s in _HANDOFF_SEEDS]
    return [(_HOLDOUT_FAMILY, sid, s) for sid in _SCENARIOS for s in _HANDOFF_SEEDS]


def _acquire_panel(panel: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], Any]:
    """Acquire each panel handoff ONCE (deterministic by seed); keep x + snapshot for ranking and physical rollout."""
    cfg, conf, obj = bc_context()
    rigs = {f: _rig(object_spec=variant(f).object_spec) for f in {p[0] for p in panel}}
    got: dict[tuple[str, str, int], Any] = {}
    for fam, sid, seed in panel:
        h = reach_capture_descriptor(rigs[fam], scenario_by_id(sid), seed, cfg, conf, obj)
        if h.record is None:                                   # valid handoff (record None ⇒ certified straddle)
            got[(fam, sid, seed)] = h
    return got


def _cached_k6(rows: list[dict]) -> dict[tuple[str, str, int, int], bool]:
    return {(r["handoff_family"], r["scenario"], r["seed"], r["theta_idx"]): bool(r["k6"]) for r in rows}


def _run(mode: str, thetas: np.ndarray, X: np.ndarray, y: np.ndarray, rows: list[dict], n_seeds: int,
         dev: str) -> dict:
    tr, _ = _split(rows, mode)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr = torch.tensor(((X - mu) / sd)[tr], device=dev)
    ytr = torch.tensor(y[tr], device=dev)
    pos_w = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))

    panel = _panel(mode)
    handoffs = _acquire_panel(panel)
    keys = list(handoffs)                                      # the frozen panel that actually certified
    # candidate input matrix per handoff (normalized): 101 pooled θ, object features from the HANDOFF family
    cand: dict[tuple[str, str, int], np.ndarray] = {}
    for k in keys:
        fam = k[0]
        rows_c = [build_input_row([float(v) for v in np.asarray(handoffs[k].x, np.float64)],
                                  [float(t) for t in th], fam) for th in thetas]   # object features via fam
        cand[k] = ((np.asarray(rows_c, np.float32) - mu) / sd).astype(np.float32)

    cached = _cached_k6(rows)
    roll_memo: dict[tuple[str, str, int, int], bool] = {}
    agree = [0, 0]

    def physical_k6(k: tuple[str, str, int], ti: int) -> bool:
        mk = (k[0], k[1], k[2], ti)
        if mk not in roll_memo:
            s = _delivery_signals(handoffs[k].snap, clip_theta(thetas[ti]))
            roll_memo[mk] = bool(s.k6)
            if mk in cached:                                   # reproduction check vs the dataset's physical label
                agree[0] += int(cached[mk] == roll_memo[mk])
                agree[1] += 1
        return roll_memo[mk]

    model_names = list(MatchedModels().build(0))
    per_seed: dict[str, list[float]] = {m: [] for m in model_names}
    off_phys_match: dict[str, list[int]] = {m: [0, 0] for m in model_names}
    for seed in range(n_seeds):
        models = MatchedModels().build(seed)
        for name, model in models.items():
            model.to(dev)
            _train(model, Xtr, ytr, 80, pos_w, dev)
            model.eval()
            hits = 0
            for k in keys:
                with torch.no_grad():
                    p = model(torch.tensor(cand[k], device=dev)).cpu().numpy()
                ti = int(np.argmax(p))                         # top-1 selection
                phys = physical_k6(k, ti)
                hits += int(phys)
                off = cached.get((k[0], k[1], k[2], ti))       # offline label of the SAME pick
                if off is not None:
                    off_phys_match[name][0] += int(off == phys)
                    off_phys_match[name][1] += 1
            per_seed[name].append(hits / max(1, len(keys)))
    oracle = float(np.mean([max(cached.get((k[0], k[1], k[2], ti), False) for ti in range(len(thetas)))
                            for k in keys]))
    out = {"mode": mode, "n_panel": len(keys), "oracle_top1_k6": round(oracle, 3),
           "reproduction_agreement": round(agree[0] / max(1, agree[1]), 4), "reproduction_n": agree[1], "models": {}}
    for name in model_names:
        v = np.array(per_seed[name], float)
        m = off_phys_match[name]
        out["models"][name] = {
            "phys_top1_k6": round(float(v.mean()), 3),
            "phys_top1_k6_ci": round(float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, 3),
            "offline_physical_pick_agreement": round(m[0] / max(1, m[1]), 4)}
    return out


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    thetas, _ = _pooled_thetas()
    rows = _load()
    X, y, _ = _matrix(rows)
    modes = ("E1", "E2") if which == "both" else (which,)
    t0 = time.perf_counter()
    summary: dict = {"n_train_seeds": n_seeds, "dev": dev, "results": []}
    for mode in modes:
        r = _run(mode, thetas, X, y, rows, n_seeds, dev)
        summary["results"].append(r)
        print(f"\n=== {mode} physical closed-loop (panel {r['n_panel']}, oracle {r['oracle_top1_k6']:.3f}, "
              f"reproduction {r['reproduction_agreement']:.3f} of {r['reproduction_n']}) ===", flush=True)
        for name, m in r["models"].items():
            print(f"  {name:20s} phys top1-K6 {m['phys_top1_k6']:.3f}±{m['phys_top1_k6_ci']:.3f}  "
                  f"(offline↔physical pick agreement {m['offline_physical_pick_agreement']:.3f})", flush=True)
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    Path(_OUT).mkdir(parents=True, exist_ok=True)
    (_OUT / "closed_loop.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote closed_loop.json ({summary['wall_s'] / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
