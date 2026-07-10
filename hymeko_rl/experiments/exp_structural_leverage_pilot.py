"""Stage 1 — the cheap 2-rung supervised pilot for the HSiKAN structural-leverage hypothesis (H1/H2).

Reuses the structural probe (``structural_probe.py``) end-to-end (no new trainer/model, §6.1) and adds the one new
control — the signed-incidence scramble (``incidence_scramble.py``). Two rungs on the fixed 7-vertex signed graph:

* **Rung A (structure-max):** target ``structural`` = ``Σ_v tanh(α·(B²x)_v)`` — a signed 2-hop aggregate, exactly
  what HSiKAN's conv computes. HSiKAN should beat a params-matched MLP here.
* **Rung B (flat control):** target ``bag`` = ``Σ_v tanh(α·x_v)`` — structure-independent (the MLP-tie control from
  the cart-pole finding, in synthetic form). HSiKAN should **not** win.

The **H2 causal test** is the decoupling: the dataset ``(X, y)`` is always generated from the **true** graph's
``B``, but the HSiKAN backbone is built on either the true or a **degree/sign-preserving scrambled** graph. The MLP
is structure-blind (it flattens the per-node obs), so it is one shared baseline across conditions. If HSiKAN's
advantage is *caused* by structure, it must appear on the true graph and collapse under scramble — while the flat
control stays a tie and the scramble fabricates no advantage.

    python -m hymeko_rl.experiments.exp_structural_leverage_pilot --seeds 3

Reported per (target, condition): mean±std held-out MSE (3 seeds), params (matched), the MLP−HSiKAN skill gap, the
MLP/HSiKAN ratio, and the scramble **collapse** (fraction of the HSiKAN advantage the scramble removes).
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.experiments.incidence_scramble import scramble_signed_incidence, scramble_stats
from hymeko_rl.experiments.structural_probe import (
    Backbone,
    Target,
    _standardised_split,
    build_model,
    build_toy_graph,
    match_mlp_hidden,
    train_eval,
)

_TARGETS: tuple[Target, ...] = ("structural", "bag")   # Rung A (structure-max), Rung B (flat control)


def _f(x: object) -> float:
    """Coerce a report cell (typed ``object`` in the JSON-ish dicts, numeric at runtime) to float."""
    assert isinstance(x, (int, float)), f"expected numeric report cell, got {type(x).__name__}"
    return float(x)
# verdict thresholds (stated up front, per the decision rule): a real HSiKAN advantage, ≥half collapse under
# scramble, and a flat control that neither wins nor fabricates a win under scramble.
_ADV_RATIO_MIN = 1.5                         # MLP/HSiKAN ratio that counts as a real structural advantage
_COLLAPSE_FRAC_MIN = 0.5                     # scramble must remove ≥50% of that advantage (H2)
_FLAT_BAND = (0.7, 1.5)                      # ratio band that counts as "≈ tie" on the flat control


@dataclass(frozen=True)
class PilotConfig:
    seeds: int = 3
    hsikan_hidden: int = 32
    n_layers: int = 2
    n_train: int = 256
    n_test: int = 1024
    epochs: int = 300
    scramble_seed: int = 0
    swaps_per_edge: int = 20


def _run_cell(model_graph: HypergraphState, data_graph: HypergraphState, target: Target, kind: Backbone,
              hidden: int, cfg: PilotConfig) -> tuple[list[float], int]:
    """Train ``kind`` (built on ``model_graph``) on data generated from ``data_graph``; return per-seed test MSEs.

    The decoupling is the H2 lever: ``data_graph`` fixes the target (the true structure), ``model_graph`` is what
    the backbone sees (true or scrambled). # Postconditions ``len(mses) == cfg.seeds``; params are seed-invariant."""
    mses: list[float] = []
    n_params = 0
    for s in range(cfg.seeds):
        split = _standardised_split(data_graph, target, n_train=cfg.n_train, n_test=cfg.n_test, seed=1000 + s)
        torch.manual_seed(s)                                  # reproducible init regardless of call history
        model = build_model(kind, model_graph, hidden, n_layers=cfg.n_layers)
        n_params = model.n_params()
        mses.append(train_eval(model, split, epochs=cfg.epochs, seed=s))
    return mses, n_params


def _agg(mses: list[float]) -> dict[str, object]:
    return {"mean": round(statistics.fmean(mses), 5),
            "std": round(statistics.pstdev(mses), 5) if len(mses) > 1 else 0.0,
            "per_seed": [round(m, 5) for m in mses]}


def run_pilot(cfg: PilotConfig | None = None) -> dict[str, object]:
    """The full 2-rung pilot: {structural, bag} × {MLP, HSiKAN-original, HSiKAN-scrambled}, ``cfg.seeds`` seeds.

    # Postconditions returns a report with per-cell aggregates, gap/ratio/collapse per target, the scramble
      preservation stats, and an H1/H2 verdict computed from the thresholds above."""
    cfg = cfg or PilotConfig()
    torch.set_num_threads(1)
    hg = build_toy_graph()
    hg_scr = scramble_signed_incidence(hg, seed=cfg.scramble_seed, swaps_per_edge=cfg.swaps_per_edge)
    mlp_hidden, hk_params, mlp_params = match_mlp_hidden(hg, cfg.hsikan_hidden)

    cells: dict[str, dict[str, object]] = {}
    per_target: dict[str, dict[str, object]] = {}
    for target in _TARGETS:
        mlp_mses, mlp_p = _run_cell(hg, hg, target, "mlp", mlp_hidden, cfg)          # structure-blind baseline
        hk_o_mses, hk_p = _run_cell(hg, hg, target, "hsikan", cfg.hsikan_hidden, cfg)  # true structure
        hk_s_mses, hk_ps = _run_cell(hg_scr, hg, target, "hsikan", cfg.hsikan_hidden, cfg)  # scrambled structure
        assert hk_p == hk_ps, "HSiKAN params must be identical under scramble (adjacency is a buffer, not a weight)"
        cells[f"{target}/mlp"] = {**_agg(mlp_mses), "n_params": mlp_p}
        cells[f"{target}/hsikan_original"] = {**_agg(hk_o_mses), "n_params": hk_p}
        cells[f"{target}/hsikan_scrambled"] = {**_agg(hk_s_mses), "n_params": hk_ps}

        mlp_m = statistics.fmean(mlp_mses)
        hk_o, hk_s = statistics.fmean(hk_o_mses), statistics.fmean(hk_s_mses)
        gap_o, gap_s = mlp_m - hk_o, mlp_m - hk_s                # skill gap (MSE); positive = HSiKAN better
        ratio_o = mlp_m / max(hk_o, 1e-9)
        ratio_s = mlp_m / max(hk_s, 1e-9)
        collapse_frac = (gap_o - gap_s) / gap_o if gap_o > 1e-9 else None
        per_target[target] = {
            "mlp_mse": round(mlp_m, 5), "hsikan_original_mse": round(hk_o, 5),
            "hsikan_scrambled_mse": round(hk_s, 5),
            "gap_original": round(gap_o, 5), "gap_scrambled": round(gap_s, 5),
            "ratio_mlp_over_hsikan_original": round(ratio_o, 3),
            "ratio_mlp_over_hsikan_scrambled": round(ratio_s, 3),
            "collapse_abs": round(gap_o - gap_s, 5),
            "collapse_frac": round(collapse_frac, 3) if collapse_frac is not None else None,
            # architecture-CONTROLLED structure benefit: same HSiKAN, only the structure differs (scrambled/true).
            # >1 ⇒ correct structure helps; ≈1 ⇒ structure irrelevant. Free of the per-node-architecture confound
            # that the HSiKAN-vs-MLP ratio carries (see the report — HSiKAN also wins the structure-free bag target).
            "structure_benefit": round(hk_s / max(hk_o, 1e-9), 3),
        }

    robustness = _scramble_robustness(hg, cfg)
    verdict = _verdict(per_target, robustness)
    st = scramble_stats(hg, hg_scr)
    return {
        "config": asdict(cfg), "hsikan_hidden": cfg.hsikan_hidden, "mlp_hidden": mlp_hidden,
        "hk_params": hk_params, "mlp_params": mlp_params,
        "scramble_stats": asdict(st), "per_target": per_target, "cells": cells,
        "scramble_robustness": robustness, "verdict": verdict,
    }


def _scramble_robustness(hg: HypergraphState, cfg: PilotConfig, n_scramble_seeds: int = 5,
                         ) -> dict[str, object]:
    """H2 robustness: HSiKAN's *structural*-target MSE under several independent degree/sign-preserving scrambles.

    Guards against "one unlucky scramble": if the structure is truly load-bearing, *every* scramble should degrade
    HSiKAN (not just ``scramble_seed``). # Postconditions per-scramble-seed HSiKAN MSE + median/IQR + the
    reference true-structure and MLP MSEs (means over ``cfg.seeds`` training seeds)."""
    hk_true, _ = _run_cell(hg, hg, "structural", "hsikan", cfg.hsikan_hidden, cfg)
    mlp_hidden, _hk, _mk = match_mlp_hidden(hg, cfg.hsikan_hidden)
    mlp, _ = _run_cell(hg, hg, "structural", "mlp", mlp_hidden, cfg)
    per_scramble: list[float] = []
    for ss in range(n_scramble_seeds):
        hg_scr = scramble_signed_incidence(hg, seed=ss, swaps_per_edge=cfg.swaps_per_edge)
        scr, _ = _run_cell(hg_scr, hg, "structural", "hsikan", cfg.hsikan_hidden, cfg)
        per_scramble.append(round(statistics.fmean(scr), 5))
    q1, q3 = (float(x) for x in np.percentile(per_scramble, [25, 75]))
    hk_true_m = statistics.fmean(hk_true)
    return {
        "n_scramble_seeds": n_scramble_seeds,
        "hsikan_true_mse": round(hk_true_m, 5), "mlp_mse": round(statistics.fmean(mlp), 5),
        "hsikan_scrambled_per_seed": per_scramble,
        "hsikan_scrambled_median": round(statistics.median(per_scramble), 5),
        "hsikan_scrambled_iqr": round(q3 - q1, 5),
        # every scramble worse than true structure ⇒ the degradation is robust, not a single unlucky rewiring.
        "all_scrambles_degrade": all(m > hk_true_m for m in per_scramble),
        "worst_case_still_degrades": min(per_scramble) > hk_true_m,
    }


_STRUCT_BENEFIT_MIN = 1.5        # architecture-controlled: correct structure must lower HSiKAN MSE by ≥1.5× (structural)
_STRUCT_BENEFIT_FLAT = 1.25      # ...and must NOT help on the structure-free target (≤1.25× ≈ no effect)


def _mlp_gap_framing(per_target: dict[str, dict[str, object]]) -> dict[str, object]:
    """The PRE-REGISTERED naive framing (HSiKAN vs params-matched MLP). Kept for transparency — it is confounded
    (the MLP comparison mixes per-node architecture-fit with structure; see the scramble framing)."""
    s, b = per_target["structural"], per_target["bag"]
    checks = {
        "structural_advantage": _f(s["ratio_mlp_over_hsikan_original"]) >= _ADV_RATIO_MIN,
        "scramble_collapses_advantage": s["collapse_frac"] is not None
        and _f(s["collapse_frac"]) >= _COLLAPSE_FRAC_MIN,
        "flat_control_tie": _FLAT_BAND[0] <= _f(b["ratio_mlp_over_hsikan_original"]) <= _FLAT_BAND[1],
        "flat_no_fabricated_advantage": _FLAT_BAND[0] <= _f(b["ratio_mlp_over_hsikan_scrambled"]) <= _FLAT_BAND[1],
    }
    label = "SUPPORTED" if all(checks.values()) else (
        "WEAKENED" if checks["structural_advantage"] and
        (checks["scramble_collapses_advantage"] or checks["flat_control_tie"]) else "FALSIFIED")
    return {"label": label, "checks": checks,
            "thresholds": {"adv_ratio_min": _ADV_RATIO_MIN, "collapse_frac_min": _COLLAPSE_FRAC_MIN,
                           "flat_band": list(_FLAT_BAND)}}


def _scramble_framing(per_target: dict[str, dict[str, object]],
                      robustness: dict[str, object]) -> dict[str, object]:
    """The confound-free framing: HSiKAN-true vs HSiKAN-scrambled (same architecture, only structure differs).

    H2 = structure is causally load-bearing on the structural target; H1 (2-rung) = that benefit is present on
    ``structural`` and absent on the structure-free ``bag`` target."""
    s, b = per_target["structural"], per_target["bag"]
    checks = {
        "structure_helps_on_structural": _f(s["structure_benefit"]) >= _STRUCT_BENEFIT_MIN,          # H2
        "structure_irrelevant_on_flat": _f(b["structure_benefit"]) <= _STRUCT_BENEFIT_FLAT,           # H1 differentiation
        "robust_across_scrambles": bool(robustness["all_scrambles_degrade"]),                         # not one unlucky rewiring
    }
    label = "SUPPORTED" if all(checks.values()) else (
        "WEAKENED" if checks["structure_helps_on_structural"] else "FALSIFIED")
    return {"label": label, "checks": checks,
            "thresholds": {"struct_benefit_min": _STRUCT_BENEFIT_MIN, "struct_benefit_flat_max": _STRUCT_BENEFIT_FLAT}}


def _verdict(per_target: dict[str, dict[str, object]], robustness: dict[str, object]) -> dict[str, object]:
    """Both framings + an honest primary. The scramble framing is primary (it isolates structure); the MLP-gap
    framing is reported for transparency and is confounded by HSiKAN's per-node architecture bias."""
    mlp_gap = _mlp_gap_framing(per_target)
    scramble = _scramble_framing(per_target, robustness)
    return {
        "primary": "scramble_framing (architecture-controlled)",
        "primary_label": scramble["label"],
        "mlp_gap_framing": mlp_gap,
        "scramble_framing": scramble,
        "note": ("MLP-gap framing is confounded: HSiKAN also wins the structure-FREE bag target for per-node "
                 "architecture reasons. The scramble framing (same net, only structure differs) isolates the "
                 "causal structural effect."),
    }


def plot_pilot(report: dict[str, object], out_path: str | Path) -> Path:
    """Grouped-bar test MSE (target × {MLP, HSiKAN-orig, HSiKAN-scrambled}) with std error bars (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = cast("dict[str, dict[str, object]]", report["cells"])
    pt = cast("dict[str, dict[str, object]]", report["per_target"])
    verdict = cast("dict[str, object]", report["verdict"])
    conditions = [("mlp", "MLP (flat, matched)", "#7f7f7f"),
                  ("hsikan_original", "HSiKAN · true structure", "#2ca02c"),
                  ("hsikan_scrambled", "HSiKAN · scrambled", "#d62728")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.26
    for i, (key, label, color) in enumerate(conditions):
        means = [_f(cells[f"{t}/{key}"]["mean"]) for t in _TARGETS]
        stds = [_f(cells[f"{t}/{key}"]["std"]) for t in _TARGETS]
        ax.bar(np.arange(len(_TARGETS)) + (i - 1) * width, means, width, yerr=stds, capsize=3,
               label=label, color=color, edgecolor="black", alpha=0.85)
    ax.set_xticks(np.arange(len(_TARGETS)))
    ax.set_xticklabels([f"{t}\n(MLP/HSiKAN={pt[t]['ratio_mlp_over_hsikan_original']}× → "
                        f"{pt[t]['ratio_mlp_over_hsikan_scrambled']}× scrambled)" for t in _TARGETS])
    ax.set_ylabel("held-out test MSE (lower = better)")
    ax.set_title(f"Structural-leverage pilot — H2 (scramble framing): "
                 f"{verdict['primary_label']}  "
                 f"(structure benefit {pt['structural']['structure_benefit']}× vs "
                 f"{pt['bag']['structure_benefit']}× flat)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32, help="HSiKAN width (MLP params-matched)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--scramble-seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/figures/2026_07_10_structural_leverage_pilot")
    a = ap.parse_args(argv)
    cfg = PilotConfig(seeds=a.seeds, hsikan_hidden=a.hidden, epochs=a.epochs, n_train=a.n_train,
                      scramble_seed=a.scramble_seed)
    report = run_pilot(cfg)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "structural_leverage_pilot.json").write_text(json.dumps(report, indent=2, default=float))
    plot_pilot(report, out / "structural_leverage_pilot")
    print(json.dumps(report["per_target"], indent=2))
    print(json.dumps(report["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
