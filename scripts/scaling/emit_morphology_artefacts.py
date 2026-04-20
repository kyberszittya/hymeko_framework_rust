#!/usr/bin/env python3
"""
emit_morphology_artefacts.py — produce the morphology-check figure and
LaTeX table for the humanoid/quadruped fixtures.

Outputs (into --out dir):
    morphology_check.pdf    HyMeKo vs competitor bundle cost on humanoid
                            and quadruped fixtures, overlaid on the
                            chain/tree power-law fit from the main sweep.
    morphology_table.tex    Per-fixture comparison (|V|, HyMeKo bundle,
                            competitor bundle, speedup).

Rationale: humanoid/quadruped are single-shot morphology samples, not a
size sweep, so they don't get a power-law fit — but their placement on
the chain/tree fit curve tells the reader the scaling story carries
over to realistic robot topologies.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EMITTERS_BUNDLE = ["compile", "urdf", "sdf", "mjcf"]


def load_hymeko_bundle(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["stage"].isin(EMITTERS_BUNDLE)]
    bundle = (df.groupby(["family", "name", "n_vertices", "rep"])
                .agg(wall_ms=("wall_ns", lambda s: s.sum() / 1e6))
                .reset_index())
    return bundle


def load_competitor_bundle(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["stage"] == "bundle_3fmt"]
    df["wall_ms"] = df["wall_ns"] / 1e6
    return df[["family", "name", "n_vertices", "rep", "wall_ms"]]


def median_per_fixture(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["family", "name", "n_vertices"])
              .agg(median_ms=("wall_ms", "median"),
                   q25_ms=("wall_ms", lambda s: s.quantile(0.25)),
                   q75_ms=("wall_ms", lambda s: s.quantile(0.75)))
              .reset_index())


def fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = (x > 0) & (y > 0)
    if m.sum() < 3:
        return (np.nan, np.nan)
    lx, ly = np.log(x[m]), np.log(y[m])
    slope, intercept, *_ = __import__("scipy.stats", fromlist=["linregress"]).linregress(lx, ly)
    return float(np.exp(intercept)), float(slope)


def plot_morphology(h_med: pd.DataFrame, c_med: pd.DataFrame,
                    out: Path) -> None:
    # chain/tree fit lines
    base_h = h_med[h_med["family"].isin(["chain", "tree"])]
    base_c = c_med[c_med["family"].isin(["chain", "tree"])]
    a_h, b_h = fit_power(base_h["n_vertices"].to_numpy(),
                         base_h["median_ms"].to_numpy())
    a_c, b_c = fit_power(base_c["n_vertices"].to_numpy(),
                         base_c["median_ms"].to_numpy())
    xs = np.logspace(np.log10(10), np.log10(70), 50)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # chain/tree backdrop (sparse markers for context)
    ax.loglog(base_h["n_vertices"], base_h["median_ms"],
              "o", color="#1f77b4", alpha=0.3, markersize=4,
              label="HyMeKo chain/tree sweep")
    ax.loglog(base_c["n_vertices"], base_c["median_ms"],
              "s", color="#d62728", alpha=0.3, markersize=4,
              label="competitor chain/tree sweep")
    # fitted extrapolations
    if np.isfinite(a_h):
        ax.loglog(xs, a_h * xs ** b_h, "-", color="#1f77b4",
                  alpha=0.6, linewidth=1,
                  label=f"HyMeKo fit (b={b_h:.2f})")
    if np.isfinite(a_c):
        ax.loglog(xs, a_c * xs ** b_c, "-", color="#d62728",
                  alpha=0.6, linewidth=1,
                  label=f"competitor fit (b={b_c:.2f})")

    # morphology points, distinct markers
    for family, marker, color_h, color_c in [
        ("humanoid",  "^", "#1f77b4", "#d62728"),
        ("quadruped", "D", "#1f77b4", "#d62728"),
    ]:
        hh = h_med[h_med["family"] == family].sort_values("n_vertices")
        cc = c_med[c_med["family"] == family].sort_values("n_vertices")
        ax.loglog(hh["n_vertices"], hh["median_ms"], marker,
                  color=color_h, markersize=9, markeredgecolor="black",
                  markeredgewidth=0.8,
                  label=f"HyMeKo {family}")
        ax.loglog(cc["n_vertices"], cc["median_ms"], marker,
                  color=color_c, markersize=9, markeredgecolor="black",
                  markeredgewidth=0.8,
                  label=f"competitor {family}")

    ax.set_xlabel(r"$|V|$ (links)")
    ax.set_ylabel("median URDF+SDF+MJCF bundle wall-clock [ms]")
    ax.set_title("Morphology check: humanoid + quadruped fall on the "
                 "chain/tree fit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def emit_table(h_med: pd.DataFrame, c_med: pd.DataFrame,
               out: Path) -> None:
    morph = h_med[h_med["family"].isin(["humanoid", "quadruped"])] \
        .sort_values(["family", "n_vertices"])
    comp_idx = c_med.set_index("name")

    pretty_labels = {
        "humanoid_f0":       "humanoid, baseline (Atlas-class)",
        "humanoid_f2":       "humanoid, 2 fingers / hand",
        "humanoid_f5":       "humanoid, 5 fingers / hand",
        "quadruped_d3_t0":   "quadruped, 3-DOF legs",
        "quadruped_d3_t3":   "quadruped, 3-DOF + tail",
        "quadruped_d5_t0":   "quadruped, 5-DOF legs (Spot/ANYmal-class)",
        "quadruped_d5_t3":   "quadruped, 5-DOF + tail",
        "quadruped_d7_t0":   "quadruped, 7-DOF legs",
        "quadruped_d7_t3":   "quadruped, 7-DOF + tail",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Morphology check: realistic humanoid and quadruped "
        r"topologies generated by the same fixture generator as the "
        r"chain/tree sweep. HyMeKo emits the full URDF+SDF+MJCF bundle "
        r"in a single in-process pipeline; the competitor column is "
        r"\texttt{xacro}$\to$\texttt{gz~sdf}$\to$\texttt{mujoco}.}",
        r"\label{tab:morphology}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Fixture & $|V|$ & HyMeKo [ms] & Competitor [ms] & Speed-up \\",
        r"\midrule",
    ]
    for _, r in morph.iterrows():
        name = r["name"]
        label = pretty_labels.get(name, name.replace("_", r"\_"))
        h_ms = r["median_ms"]
        if name in comp_idx.index:
            c_ms = comp_idx.loc[name, "median_ms"]
            if hasattr(c_ms, "iloc"):
                c_ms = c_ms.iloc[0]
            speedup = c_ms / h_ms
            lines.append(
                f"{label} & {int(r['n_vertices'])} & "
                f"{h_ms:.3f} & {c_ms:.1f} & {speedup:.0f}$\\times$ \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hymeko-csv", type=Path, required=True)
    ap.add_argument("--competitor-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    h = load_hymeko_bundle(args.hymeko_csv)
    c = load_competitor_bundle(args.competitor_csv)
    h_med = median_per_fixture(h)
    c_med = median_per_fixture(c)

    plot_morphology(h_med, c_med, args.out / "morphology_check.pdf")
    emit_table(h_med, c_med, args.out / "morphology_table.tex")

    # Console summary
    print("Morphology check (HyMeKo vs competitor bundle, tree family):")
    print(f"  {'fixture':35s} {'|V|':>4s} {'H [ms]':>10s} "
          f"{'C [ms]':>10s} {'speedup':>9s}")
    morph = h_med[h_med["family"].isin(["humanoid", "quadruped"])]
    for _, r in morph.sort_values(["family", "n_vertices"]).iterrows():
        name = r["name"]
        c_row = c_med[c_med["name"] == name]
        if c_row.empty:
            continue
        speedup = c_row["median_ms"].iloc[0] / r["median_ms"]
        print(f"  {name:35s} {int(r['n_vertices']):>4d} "
              f"{r['median_ms']:>10.3f} "
              f"{c_row['median_ms'].iloc[0]:>10.1f} "
              f"{speedup:>8.0f}×")


if __name__ == "__main__":
    main()
