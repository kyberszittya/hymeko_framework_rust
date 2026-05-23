"""Aggregate all 2026-04-23 overnight-suite CSVs into a single results table.

For each CSV (= one experiment invocation), compute paired stats for
every (dataset × view × non-baseline-arm) present: mean Δ in percentage
points, paired t-stat, W/L/T counts, val-acc stdev shift, and the
entropy shift (mean H of regularised arm minus baseline).

Emits RESULTS_VIEWS_SUITE.md at the repo root.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "data" / "benchmarks"
OUT = REPO / "RESULTS_VIEWS_SUITE.md"

# EXT + PH2..PH6 runs span 2026-04-23..2026-04-25; glob all three dates.
FILES = sorted(
    list(BENCH.glob("thesis_iv_hard_20260423_*.csv"))
    + list(BENCH.glob("thesis_iv_hard_20260424_*.csv"))
    + list(BENCH.glob("thesis_iv_hard_20260425_*.csv"))
    + list(BENCH.glob("thesis_iv_hard_20260426_*.csv"))
)

# The 2-seed × 2-epoch smoke test CSV from my loader-validation run
# would be tiny (well under 10 rows). Exclude it.
MIN_ROWS = 10


def analyze_one(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    if len(df) < MIN_ROWS:
        return []
    n_epochs = len(df.iloc[0]["val_acc_per_epoch"].split(";"))

    out: list[dict] = []
    for (dataset, view), sub in df.groupby(["dataset", "view"]):
        piv_acc = sub.pivot(index="seed", columns="arm", values="final_val_acc")
        piv_H   = sub.pivot(index="seed", columns="arm", values="final_entropy")
        if "baseline" not in piv_acc.columns:
            continue
        base_acc = piv_acc["baseline"]
        base_H   = piv_H["baseline"]

        for arm in piv_acc.columns:
            if arm == "baseline":
                continue
            treat_acc = piv_acc[arm]
            treat_H   = piv_H[arm]

            # Drop seeds present in only one arm (shouldn't happen in practice,
            # but defensively so a partial run doesn't crash us).
            paired = pd.concat([base_acc, treat_acc, base_H, treat_H], axis=1).dropna()
            if len(paired) < 2:
                continue
            b_acc = paired.iloc[:, 0].values
            t_acc = paired.iloc[:, 1].values
            b_H   = paired.iloc[:, 2].values
            t_H   = paired.iloc[:, 3].values

            delta = t_acc - b_acc
            n     = len(delta)
            t_stat, p_val = stats.ttest_rel(t_acc, b_acc)

            wins   = int((delta > 0).sum())
            losses = int((delta < 0).sum())
            ties   = int((delta == 0).sum())

            out.append({
                "dataset":     dataset,
                "view":        view,
                "arm":         arm,
                "n_seeds":     n,
                "n_epochs":    n_epochs,
                "delta_pp":    float(delta.mean() * 100.0),
                "t_stat":      float(t_stat) if not math.isnan(t_stat) else float("nan"),
                "p_val":       float(p_val) if not math.isnan(p_val) else float("nan"),
                "WLT":         f"{wins}/{losses}/{ties}",
                "sd_base_pp":  float(b_acc.std(ddof=1) * 100.0),
                "sd_treat_pp": float(t_acc.std(ddof=1) * 100.0),
                "H_base":      float(b_H.mean()),
                "H_treat":     float(t_H.mean()),
                "H_delta":     float(t_H.mean() - b_H.mean()),
                "file":        path.name,
            })
    return out


def _sig(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""


def fmt_row(r: dict) -> str:
    dp = r["delta_pp"]
    arrow = ("▲" if dp > 0 else ("▼" if dp < 0 else "·"))
    sig = _sig(r["p_val"])
    sd_shift = r["sd_treat_pp"] - r["sd_base_pp"]
    return ("| {dataset:<14} | {view:<8} | {arm:<16} | {n:>3} | {ep:>3} | "
            "{arrow} {dp:+.3f} {sig:<3} | {t:+5.2f} | {wlt:<9} | "
            "{sdb:.3f} → {sdt:.3f} ({ssd:+.3f}) | {Hd:+.3f} |"
           ).format(
               dataset=r["dataset"], view=r["view"], arm=r["arm"],
               n=r["n_seeds"], ep=r["n_epochs"],
               arrow=arrow, dp=dp, sig=sig, t=r["t_stat"],
               wlt=r["WLT"], sdb=r["sd_base_pp"], sdt=r["sd_treat_pp"],
               ssd=sd_shift, Hd=r["H_delta"])


def main() -> None:
    rows: list[dict] = []
    for f in FILES:
        rows.extend(analyze_one(f))

    # Sort primarily by dataset, then arm, then view, then seeds desc.
    dataset_order = {
        "mnist_small": 0, "mnist_resnet_20": 1,
        "fashion_mnist": 10, "kmnist": 11, "cifar10": 20,
        "two_moons": 30, "spirals": 31, "circles": 32,
    }
    rows.sort(key=lambda r: (
        dataset_order.get(r["dataset"], 99),
        r["arm"], r["view"], -r["n_seeds"], -r["n_epochs"],
    ))

    lines: list[str] = []
    lines.append("# Thesis IV entropy-regularization suite — overnight 2026-04-23 results\n")
    lines.append("All paired comparisons of the form  **Δ = (treatment − baseline) val-acc**,\n"
                 "matched seeds, same optimiser config. Percentage-point units (pp); a Δ of\n"
                 "+0.100 means a one-tenth-of-a-percent accuracy gain.\n")
    lines.append(
        "| column       | meaning |\n"
        "|--------------|---------|\n"
        "| Δ pp         | mean accuracy shift, with ▲/▼ for sign and * for p-value |\n"
        "| t            | paired two-sided t-statistic |\n"
        "| W/L/T        | wins / losses / ties across seeds |\n"
        "| σ base → σ treat | stdev of val-acc per arm (drop = variance-reducing) |\n"
        "| ΔH           | final spectral entropy shift (treat − baseline) |\n"
        "| sig          | `***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `.` p<0.10 |\n")

    lines.append("\n## Anchor reference\n")
    lines.append("The original overnight-1 finding this suite is probing:\n"
                 "- **MNIST** plain-MLP (`mnist_small`), scalar_entropy / dataflow, 33 seeds × 15 epochs → **Δ = +0.149 pp**, t=+2.88, p≈0.007, W/L=22/11.\n"
                 "- **MNIST** plain-MLP, scalar_entropy / factor, 33 seeds × 15 epochs → **Δ = +0.135 pp**, t=+2.53, W/L=19/13.\n")

    lines.append("\n## All runs (sorted by dataset, arm, view)\n")
    lines.append(
        "| dataset        | view     | arm              |   n | ep  | Δ pp          |    t  | W/L/T     | σ base → σ treat              | ΔH     |\n"
        "|----------------|----------|------------------|-----|-----|---------------|-------|-----------|-------------------------------|--------|"
    )
    for r in rows:
        lines.append(fmt_row(r))

    # Highlights: what replicated, what didn't.
    positive_sig = [r for r in rows if r["p_val"] < 0.05 and r["delta_pp"] > 0]
    positive_marginal = [r for r in rows if 0.05 <= r["p_val"] < 0.10 and r["delta_pp"] > 0]
    negative_sig = [r for r in rows if r["p_val"] < 0.05 and r["delta_pp"] < 0]
    variance_reducers = [r for r in rows if r["sd_treat_pp"] < 0.9 * r["sd_base_pp"]]

    lines.append("\n## Summary\n")
    lines.append(f"- Total experiments analysed: **{len(rows)}**\n")
    lines.append(f"- Positive and significant (p < 0.05, Δ > 0): **{len(positive_sig)}** "
                 + (", ".join(f"`{r['dataset']}/{r['view']}/{r['arm']}`" for r in positive_sig) or "none")
                 + ".\n")
    lines.append(f"- Positive but only marginal (0.05 ≤ p < 0.10): **{len(positive_marginal)}** "
                 + (", ".join(f"`{r['dataset']}/{r['view']}/{r['arm']}`" for r in positive_marginal) or "none")
                 + ".\n")
    lines.append(f"- Negative and significant: **{len(negative_sig)}** "
                 + (", ".join(f"`{r['dataset']}/{r['view']}/{r['arm']}`" for r in negative_sig) or "none")
                 + ".\n")
    lines.append(f"- Variance-reducing (σ treat < 0.9 × σ base): **{len(variance_reducers)}** "
                 + (", ".join(f"`{r['dataset']}/{r['view']}/{r['arm']}`" for r in variance_reducers) or "none")
                 + ".\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
