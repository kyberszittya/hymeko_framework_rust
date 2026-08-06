"""Head-to-head: SignedHyperLiNGAM vs DirectLiNGAM structural recovery across SEM regimes.

The measured claim (§3 discipline: measured, not asserted): the two **tie** on additive SEMs (DirectLiNGAM already
recovers support F1 ≈ 1.0 — no headroom), and SignedHyperLiNGAM **wins** on **joint-interaction** mechanisms
(``x_i = x_a·x_b + …``), where a parent's marginal effect vanishes so pairwise LiNGAM structurally misses it.
Reuses ``generate_signed_dag`` / ``sample_sem`` (harness), ``estimate_b`` / ``recovery_metrics`` (robustness), and
``SignedHyperLiNGAM`` / ``sample_interaction_sem`` — no re-implemented SEM, metric, or solver (§6.1). Three-form
output: a JSON table + a grouped-bar recall plot.

    python -m hymeko_rl.experiments.exp_signed_hyper_lingam --seeds 10
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hymeko_rl.eval.causal.lingam_estimated_b_robustness import estimate_b, recovery_metrics
from hymeko_rl.eval.causal.lingam_operator_harness import generate_signed_dag, sample_sem
from hymeko_rl.eval.causal.signed_hyper_lingam import SignedHyperLiNGAM, sample_interaction_sem

_REGIMES = ("linear", "nonlinear", "joint")
_METRICS = ("support_recall", "support_f1", "sign_agreement")


def _median_iqr(values: "Sequence[float]") -> "dict[str, float]":
    a = np.asarray([v for v in values if v == v], dtype=float)          # drop NaNs
    if a.size == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    q1, med, q3 = (float(np.percentile(a, p)) for p in (25, 50, 75))
    return {"median": round(med, 3), "q1": round(q1, 3), "q3": round(q3, 3)}


def _sample_regime(regime: str, b: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """One dataset for a regime: joint uses the interaction SEM; the others the harness's additive samplers."""
    if regime == "joint":
        return sample_interaction_sem(b, n_samples, seed=seed)
    return sample_sem(b, n_samples, kind=regime, seed=seed)


def _collect_regime(regime: str, n_vars: int, density: float, n_samples: int, seeds: "Sequence[int]",
                    ) -> "dict[str, dict[str, list[float]]]":
    """Per-seed recovery metrics for both methods on one regime."""
    out: "dict[str, dict[str, list[float]]]" = {
        "DirectLiNGAM": {m: [] for m in _METRICS}, "SignedHyperLiNGAM": {m: [] for m in _METRICS}}
    for seed in seeds:
        b = generate_signed_dag(n_vars, density=density, seed=seed)
        x = _sample_regime(regime, b, n_samples, 100 + seed)
        results = {"DirectLiNGAM": recovery_metrics(b, estimate_b(x)),
                   "SignedHyperLiNGAM": recovery_metrics(b, SignedHyperLiNGAM().fit(x).adjacency)}
        for method, rec in results.items():
            for m in _METRICS:
                out[method][m].append(rec[m])
    return out


def run_head_to_head(*, n_vars: int = 6, density: float = 0.4, n_samples: int = 3500,
                     seeds: "Sequence[int]" = tuple(range(10)), out_dir: "Path | None" = None) -> "dict[str, Any]":
    """Recovery head-to-head over the three SEM regimes; writes JSON + a recall plot. Returns the summary."""
    from hymeko_rl.eval.evaluate import experiment_dir, now_stamp
    out = out_dir or experiment_dir("reports/figures", "signed_hyper_lingam")
    out.mkdir(parents=True, exist_ok=True)
    per = {r: _collect_regime(r, n_vars, density, n_samples, seeds) for r in _REGIMES}
    agg = {regime: {method: {m: _median_iqr(per[regime][method][m]) for m in _METRICS}
                    for method in ("DirectLiNGAM", "SignedHyperLiNGAM")} for regime in _REGIMES}
    verdicts = {regime: _verdict(agg[regime]) for regime in _REGIMES}
    summary: "dict[str, Any]" = {"kind": "SignedHyperLiNGAM vs DirectLiNGAM structural recovery", "stamp": now_stamp(),
                                 "n_vars": n_vars, "density": density, "n_samples": n_samples, "seeds": list(seeds),
                                 "aggregate": agg, "verdicts": verdicts}
    (out / "signed_hyper_lingam.json").write_text(json.dumps(summary, indent=2, default=float))
    summary["plot"] = str(_plot(agg, out / "signed_hyper_lingam.png") or "")
    for regime in _REGIMES:
        a = agg[regime]
        print(f"[shl] {regime:10s} recall: DirectLiNGAM {a['DirectLiNGAM']['support_recall']['median']:.3f}  "
              f"SignedHyperLiNGAM {a['SignedHyperLiNGAM']['support_recall']['median']:.3f}  -> {verdicts[regime]}",
              flush=True)
    return summary


def _verdict(regime_agg: "dict[str, Any]") -> str:
    dl = regime_agg["DirectLiNGAM"]["support_recall"]["median"]
    sh = regime_agg["SignedHyperLiNGAM"]["support_recall"]["median"]
    if sh > dl + 0.05:
        return "SignedHyperLiNGAM_WINS"
    if dl > sh + 0.05:
        return "DirectLiNGAM_WINS"
    return "TIE"


def _plot(agg: "dict[str, Any]", out_path: Path) -> "Path | None":
    """Grouped-bar support-recall by regime x method (median, IQR) — the tie/tie/win shape."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                       # noqa: BLE001 - viz best-effort (§9)
        return None
    methods = ("DirectLiNGAM", "SignedHyperLiNGAM")
    x = np.arange(len(_REGIMES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, method in enumerate(methods):
        med = [agg[r][method]["support_recall"]["median"] for r in _REGIMES]
        lo = [agg[r][method]["support_recall"]["median"] - agg[r][method]["support_recall"]["q1"] for r in _REGIMES]
        hi = [agg[r][method]["support_recall"]["q3"] - agg[r][method]["support_recall"]["median"] for r in _REGIMES]
        ax.bar(x + j * width, med, width, yerr=[lo, hi], capsize=4, label=method,
               color=("#7f7f7f" if method == "DirectLiNGAM" else "#2C7FB8"))
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([r + ("\n(joint-interaction)" if r == "joint" else "") for r in _REGIMES])
    ax.set_ylabel("support recall (median, IQR)")
    ax.set_ylim(0, 1.08)
    ax.set_title("SignedHyperLiNGAM vs DirectLiNGAM: tie on additive, WIN on joint mechanisms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _joint_sink_b(n_vars: int) -> "np.ndarray":
    """A SEM support whose sink (``n_vars-1``) has a joint mechanism over two independent roots {0,1}."""
    b = np.zeros((n_vars, n_vars), dtype=float)
    b[n_vars - 1, 0], b[n_vars - 1, 1] = 0.9, 0.8          # sink <- {0,1}  (made a joint product downstream)
    b[2, 0] = 0.7                                          # an additive edge for structure
    if n_vars > 3:
        b[3, 1] = -0.6
    return b


def run_hsikan_modeling(*, n_vars: int = 5, n_split: int = 1200, seeds: "Sequence[int]" = tuple(range(6)),
                        hidden: int = 24, n_layers: int = 2, epochs: int = 200, out_dir: "Path | None" = None,
                        ) -> "dict[str, Any]":
    """Discovery → modeling: on a joint-sink SEM, does HSiKAN over **SignedHyperLiNGAM's** discovered operator model
    the joint mechanism better than over **DirectLiNGAM's** (which misses the joint edges) or a linear predictor?

    The ``hsikan_mechanism`` doctrine (LiNGAM proposes, HSiKAN models) with a *better proposal*: SHL recovers the
    joint hyperedge ``{0,1}→sink`` so HSiKAN can model ``x_sink = x_0·x_1``; DirectLiNGAM's operator lacks those
    edges, so HSiKAN over it — like the linear baseline — cannot."""
    from hymeko_rl.eval.causal.hsikan_mechanism import signed_adjacency_split
    from hymeko_rl.eval.causal.lingam_estimated_b_robustness import estimate_b
    from hymeko_rl.eval.causal.lingam_operator_harness import (
        _fit_linear,
        _score,
        _standardise_splits,
        build_hsikan_operator,
    )
    from hymeko_rl.eval.evaluate import experiment_dir, now_stamp
    out = out_dir or experiment_dir("reports/figures", "signed_hyper_lingam_hsikan")
    out.mkdir(parents=True, exist_ok=True)
    sink = n_vars - 1
    b = _joint_sink_b(n_vars)
    rows: "dict[str, list[float]]" = {"hsikan_over_shl": [], "hsikan_over_directlingam": [], "linear": []}
    for seed in seeds:
        x = sample_interaction_sem(b, n_split * 3, seed=100 + seed, joint_frac=1.0)
        xs = [x[:n_split], x[n_split:2 * n_split], x[2 * n_split:]]
        prep = _standardise_splits(xs, [s[:, sink] for s in xs], sink)
        operators = {"hsikan_over_directlingam": estimate_b(xs[0]),
                     "hsikan_over_shl": SignedHyperLiNGAM().fit(xs[0]).adjacency}
        for name, badj in operators.items():
            a_pos, a_neg = signed_adjacency_split(badj)
            builder = functools.partial(build_hsikan_operator, a_pos, a_neg, hidden, n_layers, sink)
            rows[name].append(_score(builder, prep[0], prep[1], prep[2], epochs=epochs, seed=seed)["r2_test"])
        rows["linear"].append(_fit_linear(prep[0], prep[2])["r2_test"])
    agg = {name: _median_iqr(vals) for name, vals in rows.items()}
    shl_wins = (agg["hsikan_over_shl"]["median"] > agg["hsikan_over_directlingam"]["median"] + 0.1
                and agg["hsikan_over_shl"]["median"] > agg["linear"]["median"] + 0.1)
    summary: "dict[str, Any]" = {"kind": "HSiKAN over discovered operator (joint-sink SEM)", "stamp": now_stamp(),
                                 "n_vars": n_vars, "sink": sink, "seeds": list(seeds), "sink_r2": agg,
                                 "shl_operator_lets_hsikan_model_the_joint": shl_wins}
    (out / "signed_hyper_lingam_hsikan.json").write_text(json.dumps(summary, indent=2, default=float))
    for name in ("hsikan_over_shl", "hsikan_over_directlingam", "linear"):
        a = agg[name]
        print(f"[shl-hsikan] {name:26s} sink R2 median={a['median']:.3f} [{a['q1']:.3f},{a['q3']:.3f}]", flush=True)
    print(f"[shl-hsikan] SHL-operator lets HSiKAN model the joint mechanism = {shl_wins} -> {out}", flush=True)
    return summary


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-vars", type=int, default=6)
    ap.add_argument("--density", type=float, default=0.4)
    ap.add_argument("--n-samples", type=int, default=3500)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--hsikan", action="store_true", help="run the HSiKAN-over-discovered-operator modeling study")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else None
    if a.hsikan:
        run_hsikan_modeling(seeds=tuple(range(a.seeds)), out_dir=out)
        return 0
    run_head_to_head(n_vars=a.n_vars, density=a.density, n_samples=a.n_samples, seeds=tuple(range(a.seeds)),
                     out_dir=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
