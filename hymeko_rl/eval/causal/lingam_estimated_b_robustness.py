"""DirectLiNGAM-estimated-``B`` robustness layer for the operator harness — does the H1/H2 result survive when the
signed operator is **estimated** rather than provided?

The ground-truth-``B`` harness (``lingam_operator_harness.py``) showed HSiKAN benefits from the correct signed
operator on nonlinear-structured SEMs, and the scramble collapses it. Here the operator comes from
``DirectLiNGAM.fit`` instead of the oracle, so the six-model comparison adds *HSiKAN over estimated ``B_hat``* and
*HSiKAN over scrambled estimated ``B_hat``*, plus structural-recovery metrics of ``B_hat`` vs ``B``.

**Two claims kept separate (do not merge):**
* **A — operator usefulness:** HSiKAN benefits from a meaningful signed operator (established under oracle ``B``).
* **B — causal discovery:** DirectLiNGAM can recover a *useful* signed operator from the data (this test).

Reuses the harness end-to-end (SEM, models, scramble); adds only the estimated-operator source. No change to the
bridge or the ground-truth harness claims. Non-claims: no real-MetaWorld validation, no RL, no claim that
DirectLiNGAM is sufficient unless this test supports it.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from hymeko_rl.eval.causal.hsikan_mechanism import signed_adjacency_split
from hymeko_rl.eval.causal.lingam import DirectLiNGAM, LingamConfig
from hymeko_rl.eval.causal.lingam_operator_harness import (
    HarnessConfig,
    _f,
    _fit_linear,
    _match_hidden,
    _sample_condition,
    _score,
    _standardise_splits,
    build_deepsets_model,
    build_hsikan_operator,
    build_mlp,
    generate_signed_dag,
)
from hymeko_rl.experiments.incidence_scramble import scramble_signed_operator

_MODELS = ("linear", "mlp", "deepsets", "hsikan_gt", "hsikan_est", "hsikan_est_scrambled")
_ADV_FRAC = 0.05        # HSiKAN must beat a baseline's MSE by ≥5% to count as an advantage
_COLLAPSE_FRAC = 0.20   # a scramble must raise HSiKAN MSE by ≥20% to count as a collapse
_CLOSE_TO_GT = 0.50     # estimated-HSiKAN within 50% MSE of ground-truth-HSiKAN = "close"


# ── estimation + structural recovery ──────────────────────────────────────────────────────────────────────
def estimate_b(x_train: np.ndarray, *, threshold: float = 0.0) -> np.ndarray:
    """Estimate ``B_hat[effect, cause]`` from raw training samples via DirectLiNGAM (significance + magnitude
    pruning are built in). ``threshold`` optionally zeroes additionally-small coefficients before use."""
    b_hat = DirectLiNGAM(LingamConfig()).fit(np.asarray(x_train, dtype=np.float64)).adjacency.copy()
    if threshold > 0.0:
        b_hat[np.abs(b_hat) <= threshold] = 0.0
    return b_hat


def _rank(v: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(v)).astype(np.float64)


def recovery_metrics(b: np.ndarray, b_hat: np.ndarray, *, top_k: "int | None" = None) -> "dict[str, float]":
    """Structural recovery of ``b_hat`` vs the ground-truth ``b``: support precision/recall/F1, sign agreement on
    the shared support, Pearson/Spearman on shared-support weights, and top-``k`` edge overlap."""
    st, sh = b != 0.0, b_hat != 0.0
    tp, fp, fn = int((st & sh).sum()), int((~st & sh).sum()), int((st & ~sh).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    both = st & sh
    sign = float((np.sign(b[both]) == np.sign(b_hat[both])).mean()) if both.any() else float("nan")
    if both.sum() >= 2:
        u, w = b[both], b_hat[both]
        pear = float(np.corrcoef(u, w)[0, 1]) if u.std() > 0 and w.std() > 0 else float("nan")
        spear = float(np.corrcoef(_rank(u), _rank(w))[0, 1]) if u.std() > 0 and w.std() > 0 else float("nan")
    else:
        pear = spear = float("nan")
    k = top_k or int(st.sum())
    if k > 0:                                             # flat indices are a valid edge identity (same flattening)
        top_t = set(np.argsort(-np.abs(b).ravel())[:k].tolist())
        top_h = set(np.argsort(-np.abs(b_hat).ravel())[:k].tolist())
        overlap = len(top_t & top_h) / k
    else:
        overlap = float("nan")
    return {"support_precision": round(prec, 3), "support_recall": round(rec, 3),
            "support_f1": round(2 * prec * rec / max(1e-9, prec + rec), 3),
            "sign_agreement": round(sign, 3), "pearson": round(pear, 3), "spearman": round(spear, 3),
            "topk_overlap": round(overlap, 3), "n_edges_true": int(st.sum()), "n_edges_hat": int(sh.sum())}


# ── run ───────────────────────────────────────────────────────────────────────────────────────────────────
def _agg(per_seed: "list[dict[str, float]]") -> "dict[str, object]":
    tests = [d["test"] for d in per_seed]
    return {"test_median": round(statistics.median(tests), 5), "r2_median": round(statistics.median(
        d["r2_test"] for d in per_seed), 4), "n_params": int(per_seed[0]["n_params"]),
        "per_seed_test": [round(t, 5) for t in tests]}


def run_condition_estimated(condition: str, cfg: HarnessConfig, *, threshold: float = 0.0) -> "dict[str, object]":
    """One condition across ``cfg.seeds`` replicates with the six-model comparison (adds estimated + scrambled-
    estimated operators) and the structural-recovery metrics of ``B_hat`` vs ``B``."""
    collected: "dict[str, list[dict[str, float]]]" = {m: [] for m in _MODELS}
    recoveries: "list[dict[str, float]]" = []
    for seed in range(cfg.seeds):
        b = generate_signed_dag(cfg.n, density=cfg.density, sign_balance=cfg.sign_balance, seed=seed)
        sink = cfg.n - 1
        xs, ys = zip(*(_sample_condition(condition, b, sink, cfg.samples, sample_seed=seed * 10 + k)
                       for k in range(3)))
        b_hat = estimate_b(xs[0], threshold=threshold)             # estimate from RAW train X only
        recoveries.append(recovery_metrics(b, b_hat))
        tr, va, te = _standardise_splits(list(xs), list(ys), sink)
        gt = signed_adjacency_split(b)
        est = signed_adjacency_split(b_hat)
        est_scr = scramble_signed_operator(est[0], est[1], seed=seed)
        target = build_hsikan_operator(gt[0], gt[1], cfg.hsikan_hidden, cfg.n_layers, sink).n_params()
        mlp_h = _match_hidden(lambda h: build_mlp(cfg.n, h, cfg.n_layers), target)
        ds_h = _match_hidden(lambda h: build_deepsets_model(h, cfg.n_layers), target)
        collected["linear"].append({**_fit_linear(tr, te), "n_params": 0.0})
        collected["mlp"].append(_score(lambda: build_mlp(cfg.n, mlp_h, cfg.n_layers), tr, va, te,
                                        epochs=cfg.epochs, seed=seed))
        collected["deepsets"].append(_score(lambda: build_deepsets_model(ds_h, cfg.n_layers), tr, va, te,
                                            epochs=cfg.epochs, seed=seed))
        collected["hsikan_gt"].append(_score(
            lambda: build_hsikan_operator(gt[0], gt[1], cfg.hsikan_hidden, cfg.n_layers, sink),
            tr, va, te, epochs=cfg.epochs, seed=seed))
        collected["hsikan_est"].append(_score(
            lambda: build_hsikan_operator(est[0], est[1], cfg.hsikan_hidden, cfg.n_layers, sink),
            tr, va, te, epochs=cfg.epochs, seed=seed))
        collected["hsikan_est_scrambled"].append(_score(
            lambda: build_hsikan_operator(est_scr[0], est_scr[1], cfg.hsikan_hidden, cfg.n_layers, sink),
            tr, va, te, epochs=cfg.epochs, seed=seed))
    agg = {m: _agg(collected[m]) for m in _MODELS}
    med = {m: _f(agg[m]["test_median"]) for m in _MODELS}
    recovery = {k: round(statistics.median(r[k] for r in recoveries), 3) for k in recoveries[0]}
    return {
        "condition": condition, "models": agg, "recovery": recovery,
        "gap_est_vs_mlp": round(med["mlp"] - med["hsikan_est"], 5),
        "gap_est_vs_linear": round(med["linear"] - med["hsikan_est"], 5),
        "collapse_est_frac": round((med["hsikan_est_scrambled"] - med["hsikan_est"]) / med["hsikan_est"], 3)
        if med["hsikan_est"] > 1e-9 else None,
        "degradation_est_vs_gt_frac": round((med["hsikan_est"] - med["hsikan_gt"]) / med["hsikan_gt"], 3)
        if med["hsikan_gt"] > 1e-9 else None,
    }


def _verdict(rows: "dict[str, dict[str, object]]") -> "dict[str, object]":
    """Classify (case A/B/C/D) and evaluate the coffee-push decision gate, separating operator-usefulness (A) from
    discovery quality (B)."""
    nl, flat = rows.get("nonlinear"), rows.get("flat")
    checks: "dict[str, bool]" = {}
    case = "UNKNOWN"
    gate = False
    if nl is not None:
        est_beats_mlp = _f(nl["gap_est_vs_mlp"]) > _ADV_FRAC * _f(
            cast("dict[str, dict[str, object]]", nl["models"])["hsikan_est"]["test_median"])
        collapse = nl["collapse_est_frac"] is not None and _f(nl["collapse_est_frac"]) >= _COLLAPSE_FRAC
        close_gt = nl["degradation_est_vs_gt_frac"] is not None and _f(nl["degradation_est_vs_gt_frac"]) < _CLOSE_TO_GT
        checks = {"estimated_beats_mlp_on_nonlinear": est_beats_mlp,
                  "scramble_of_estimated_collapses": collapse,
                  "estimated_close_to_ground_truth": close_gt}
        flat_fp = flat is not None and flat["collapse_est_frac"] is not None and abs(
            _f(flat["collapse_est_frac"])) >= _COLLAPSE_FRAC
        checks["no_flat_false_positive"] = not flat_fp
        if flat_fp:
            case = "D_flat_false_positive"
        elif close_gt and collapse:
            case = "A_strong"
        elif est_beats_mlp:
            case = "B_partial"
        else:
            case = "C_negative"
        gate = (not flat_fp) and est_beats_mlp and collapse
    return {"case": case, "coffee_push_gate": "PASS" if gate else "HOLD", "checks": checks,
            "thresholds": {"advantage_frac": _ADV_FRAC, "collapse_frac": _COLLAPSE_FRAC, "close_to_gt": _CLOSE_TO_GT},
            "note": "Case A/B ⇒ discovery yields a useful operator; C ⇒ oracle-only (discovery insufficient here); "
                    "D ⇒ estimation injects spurious structure. Claim A (operator usefulness) is independent of B "
                    "(discovery quality)."}


def run_robustness(cfg: HarnessConfig | None = None, *, threshold: float = 0.0) -> "dict[str, object]":
    cfg = cfg or HarnessConfig()
    import torch
    torch.set_num_threads(1)
    rows = {c: run_condition_estimated(c, cfg, threshold=threshold) for c in cfg.conditions}
    return {"config": asdict(cfg), "threshold": threshold, "conditions": rows, "verdict": _verdict(rows)}


def plot_robustness(report: "dict[str, object]", out_path: "str | Path") -> Path:
    """Grouped-bar test MSE per condition × the six models (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = cast("dict[str, dict[str, object]]", report["conditions"])
    conds = list(rows)
    colors = ("#7f7f7f", "#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.14
    for i, (m, c) in enumerate(zip(_MODELS, colors)):
        vals = [_f(cast("dict[str, object]", rows[cd]["models"])[m]["test_median"]) for cd in conds]  # type: ignore[index]
        ax.bar(np.arange(len(conds)) + (i - 2.5) * width, vals, width, label=m, color=c, edgecolor="black", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([f"{cd}\n(est collapse {rows[cd]['collapse_est_frac']})" for cd in conds])
    ax.set_ylabel("held-out test MSE (log; lower = better)")
    ax.set_title(f"Estimated-B robustness — {report['verdict']['case']} / gate {report['verdict']['coffee_push_gate']}")  # type: ignore[index]
    ax.legend(fontsize=8, ncol=6)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.0, help="extra magnitude threshold on B_hat before split")
    ap.add_argument("--out-dir", default="reports/figures/2026_07_10_hsikan_lingam_estimated_b")
    a = ap.parse_args(argv)
    cfg = HarnessConfig(seeds=a.seeds, n=a.n, samples=a.samples, epochs=a.epochs)
    report = run_robustness(cfg, threshold=a.threshold)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimated_b.json").write_text(json.dumps(report, indent=2, default=float))
    plot_robustness(report, out / "estimated_b")
    conditions = cast("dict[str, dict[str, object]]", report["conditions"])
    print(json.dumps({c: {"gap_est_vs_mlp": r["gap_est_vs_mlp"], "collapse_est_frac": r["collapse_est_frac"],
                          "degradation_est_vs_gt_frac": r["degradation_est_vs_gt_frac"], "recovery": r["recovery"]}
                      for c, r in conditions.items()}, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
