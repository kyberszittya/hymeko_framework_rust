#!/usr/bin/env python3
"""
analyze_head_to_head.py — merge HyMeKo scaling results with the
competitor-stack (xacro + gz sdf + mujoco) results and produce the
head-to-head artefacts for the paper.

Inputs:
    --hymeko-csv     scaling_results.csv produced by hymeko_bench
    --competitor-csv competitor_results.csv produced by bench_competitors.py

Outputs (into --out dir):
    head_to_head.pdf        per-format median wall-clock, hymeko vs competitor,
                            with the n_vertices sweep on log-log axes.
    bundle_comparison.pdf   coherent 3-format bundle cost comparison.
    head_to_head.tex        LaTeX table of representative-size ratios.
    head_to_head_fits.json  log-log fit summaries for each (tool, stage).
    failures.json           which competitor invocations failed (e.g. mujoco
                            on large URDFs) — honesty marker.

Stage alignment:
    hymeko urdf             ↔   competitor xacro_urdf
    hymeko sdf              ↔   competitor gz_sdf
    hymeko mjcf             ↔   competitor mujoco_mjcf
    hymeko compile+urdf+sdf+mjcf (summed per rep) ↔ competitor bundle_3fmt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


STAGE_PAIRS = [
    ("urdf",   "xacro_urdf",   "URDF"),
    ("sdf",    "gz_sdf",       "SDF"),
    ("mjcf",   "mujoco_mjcf",  "MJCF"),
]


def load_hymeko(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tool"] = "hymeko"
    df["wall_ms"] = df["wall_ns"] / 1e6
    return df


def load_competitor(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_ms"] = df["wall_ns"] / 1e6
    return df


def detect_failures(comp: pd.DataFrame) -> pd.DataFrame:
    """
    A competitor row with output_bytes == 0 AND wall_ns present is a
    failure (subprocess returned non-zero). Bundle rows always have
    output_bytes=0 by construction, so filter them out first.
    """
    non_bundle = comp[comp["stage"] != "bundle_3fmt"]
    return non_bundle[non_bundle["output_bytes"] == 0].copy()


def median_per_fixture(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["tool", "family", "name", "n_vertices",
                        "n_hyperedges", "stage"])
              .agg(median_ms=("wall_ms", "median"),
                   q25_ms=("wall_ms", lambda s: s.quantile(0.25)),
                   q75_ms=("wall_ms", lambda s: s.quantile(0.75)),
                   n_reps=("rep", "count"))
              .reset_index())


def fit_loglog(x: np.ndarray, y: np.ndarray) -> dict:
    m = (x > 0) & (y > 0)
    if m.sum() < 3:
        return {"b": None, "a": None, "r2": None,
                "b_ci_low": None, "b_ci_high": None, "n": int(m.sum())}
    lx, ly = np.log(x[m]), np.log(y[m])
    slope, intercept, r, _, se = stats.linregress(lx, ly)
    t = stats.t.ppf(0.975, df=m.sum() - 2)
    return {"b": float(slope), "a": float(np.exp(intercept)),
            "r2": float(r ** 2),
            "b_ci_low": float(slope - t * se),
            "b_ci_high": float(slope + t * se),
            "n": int(m.sum())}


def hymeko_bundle_per_rep(h: pd.DataFrame) -> pd.DataFrame:
    """Sum compile+urdf+sdf+mjcf within each (fixture, rep)."""
    bundle_stages = ["compile", "urdf", "sdf", "mjcf"]
    sub = h[h["stage"].isin(bundle_stages)]
    bundle = (sub.groupby(["family", "name", "n_vertices", "n_hyperedges", "rep"])
                  .agg(wall_ms=("wall_ms", "sum"))
                  .reset_index())
    bundle["tool"] = "hymeko"
    bundle["stage"] = "bundle_3fmt"
    return bundle


def plot_head_to_head(agg: pd.DataFrame, comp_agg: pd.DataFrame,
                      failures: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (h_stage, c_stage, title) in zip(axes, STAGE_PAIRS):
        h_sub = agg[(agg["tool"] == "hymeko") & (agg["stage"] == h_stage)
                    & (agg["family"].isin(["chain", "tree"]))]
        c_sub = comp_agg[(comp_agg["stage"] == c_stage)
                         & (comp_agg["family"].isin(["chain", "tree"]))
                         & (comp_agg["median_ms"] > 0)]
        # Successful competitor rows (exclude failures — median==0 or tiny for failures)
        fail_names = set(failures[failures["stage"] == c_stage]["name"])
        c_sub = c_sub[~c_sub["name"].isin(fail_names)]

        ax.loglog(h_sub["n_vertices"], h_sub["median_ms"],
                  "o-", color="#1f77b4", label="HyMeKo", markersize=5)
        if not c_sub.empty:
            ax.loglog(c_sub["n_vertices"], c_sub["median_ms"],
                      "s-", color="#d62728",
                      label={"URDF": "xacro", "SDF": "gz sdf",
                             "MJCF": "mujoco"}[title],
                      markersize=5)
        # Mark failures with a cross
        if fail_names:
            fail_sub = comp_agg[(comp_agg["stage"] == c_stage)
                                & (comp_agg["name"].isin(fail_names))]
            if not fail_sub.empty:
                ax.plot(fail_sub["n_vertices"],
                        [c_sub["median_ms"].max() if not c_sub.empty else 1] *
                        len(fail_sub),
                        "x", color="black", markersize=10,
                        label="competitor failed")
        ax.set_title(title)
        ax.set_xlabel(r"$|V|$ (links)")
        ax.set_ylabel("median wall-clock [ms]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("HyMeKo vs.\\ standard single-target stack "
                 "(medians over reps, chain + tree combined)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_bundle(h_bundle: pd.DataFrame, comp_agg: pd.DataFrame,
                failures_any_fixture: set[str], out: Path) -> None:
    h_bundle_agg = (h_bundle.groupby(["family", "name", "n_vertices"])
                           .agg(median_ms=("wall_ms", "median"),
                                q25_ms=("wall_ms", lambda s: s.quantile(0.25)),
                                q75_ms=("wall_ms", lambda s: s.quantile(0.75)))
                           .reset_index())
    c_bundle = comp_agg[(comp_agg["stage"] == "bundle_3fmt")
                        & (comp_agg["family"].isin(["chain", "tree"]))]
    c_bundle = c_bundle[~c_bundle["name"].isin(failures_any_fixture)]
    h_bundle_agg = h_bundle_agg[h_bundle_agg["family"].isin(["chain", "tree"])]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(h_bundle_agg["n_vertices"], h_bundle_agg["median_ms"],
              "o-", color="#1f77b4", label="HyMeKo (in-process)",
              markersize=6)
    if not c_bundle.empty:
        ax.loglog(c_bundle["n_vertices"], c_bundle["median_ms"],
                  "s-", color="#d62728",
                  label="xacro + gz sdf + mujoco (subprocess)",
                  markersize=6)
    ax.set_xlabel(r"$|V|$ (links)")
    ax.set_ylabel("median wall-clock for URDF+SDF+MJCF bundle [ms]")
    ax.set_title("Coherent 3-format bundle cost")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    # Annotate speedup at representative points
    for target in [10, 100, 1000]:
        h_row = h_bundle_agg.iloc[
            (h_bundle_agg["n_vertices"] - target).abs().argsort()[:1]]
        c_row = c_bundle.iloc[
            (c_bundle["n_vertices"] - target).abs().argsort()[:1]] \
            if not c_bundle.empty else None
        if c_row is not None and not c_row.empty:
            speedup = c_row["median_ms"].iloc[0] / h_row["median_ms"].iloc[0]
            x = h_row["n_vertices"].iloc[0]
            y = (h_row["median_ms"].iloc[0] * c_row["median_ms"].iloc[0]) ** 0.5
            ax.annotate(f"{speedup:.0f}×",
                        (x, y), fontsize=9, color="black",
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="#ffe69c", ec="gray"))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def emit_table(h_bundle: pd.DataFrame, comp_agg: pd.DataFrame,
               agg: pd.DataFrame, out: Path,
               representative=(10, 100, 1000)) -> None:
    tree_h = agg[(agg["tool"] == "hymeko") & (agg["family"] == "tree")]
    tree_c = comp_agg[comp_agg["family"] == "tree"]
    h_b = h_bundle[h_bundle["family"] == "tree"]
    h_b_med = (h_b.groupby(["name", "n_vertices"])
                  .agg(bundle_ms=("wall_ms", "median")).reset_index())

    def pick_closest(df, target):
        if df.empty:
            return None
        sizes = df["n_vertices"].unique()
        best = min(sizes, key=lambda s: abs(s - target))
        return df[df["n_vertices"] == best]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Head-to-head wall-clock on the \textsf{tree} family at "
        r"representative sizes (median of release-profile repetitions). "
        r"HyMeKo's 3-format-bundle column is compile$+$URDF$+$SDF$+$MJCF "
        r"summed in a single in-process pipeline; the competitor column is "
        r"\texttt{xacro} $\to$ \texttt{gz sdf} $\to$ \texttt{mujoco} run as "
        r"three subprocesses. Speed-up is the ratio; FAIL denotes "
        r"invocations where a competitor tool returned an error (e.g.\ "
        r"\texttt{mujoco} raising on $\ge$\,2000-link URDFs).}",
        r"\label{tab:head_to_head}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$|V|$ & HyMeKo URDF+SDF+MJCF [ms] & Competitor stack [ms] & "
        r"Speed-up & Note \\",
        r"\midrule",
    ]
    for target in representative:
        h_row = pick_closest(h_b_med, target)
        if h_row is None or h_row.empty:
            continue
        size = int(h_row["n_vertices"].iloc[0])
        h_ms = h_row["bundle_ms"].iloc[0]
        c_bundle = tree_c[(tree_c["stage"] == "bundle_3fmt")
                          & (tree_c["n_vertices"] == size)]
        if c_bundle.empty:
            c_ms = None
            note = "N/A"
        else:
            c_ms = c_bundle["median_ms"].iloc[0]
            # Is this size a mujoco-fail? Check if mujoco_mjcf has
            # output_bytes=0 in the raw manifest (already filtered into
            # failures set by the caller — for the table we look at
            # comp_agg's bundle row and flag if the per-stage mujoco row
            # shows failure on this fixture name).
            name = c_bundle["name"].iloc[0]
            mujoco_row = tree_c[(tree_c["stage"] == "mujoco_mjcf")
                                & (tree_c["name"] == name)]
            note = ("mujoco FAIL" if (not mujoco_row.empty
                    and mujoco_row["median_ms"].iloc[0] < 500
                    and size >= 2000) else "")
        if c_ms is not None:
            speedup = c_ms / h_ms
            lines.append(f"{size} & {h_ms:.3f} & {c_ms:.1f} & "
                         f"{speedup:.0f}$\\times$ & {note} \\\\")
        else:
            lines.append(f"{size} & {h_ms:.3f} & N/A & -- & {note} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hymeko-csv", type=Path, required=True)
    ap.add_argument("--competitor-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    h = load_hymeko(args.hymeko_csv)
    c = load_competitor(args.competitor_csv)

    failures = detect_failures(c)
    failures_per_stage = failures[["family", "name", "stage", "n_vertices"]] \
        .drop_duplicates()
    (args.out / "failures.json").write_text(
        failures_per_stage.to_json(orient="records", indent=2),
        encoding="utf-8")

    fail_names = set(failures["name"])

    h_bundle = hymeko_bundle_per_rep(h)

    agg = median_per_fixture(h)
    comp_agg = median_per_fixture(c)

    # Fits
    fits = {}
    for (tool, df) in [("hymeko", agg), ("competitor", comp_agg)]:
        for stage in df["stage"].unique():
            sub = df[(df["stage"] == stage)
                     & (df["family"].isin(["chain", "tree"]))]
            sub = sub[~sub["name"].isin(fail_names)] if tool == "competitor" else sub
            fits[f"{tool}/{stage}"] = fit_loglog(
                sub["n_vertices"].to_numpy(),
                sub["median_ms"].to_numpy(),
            )
    (args.out / "head_to_head_fits.json").write_text(
        json.dumps(fits, indent=2, default=str), encoding="utf-8")

    plot_head_to_head(agg, comp_agg, failures,
                      args.out / "head_to_head.pdf")
    plot_bundle(h_bundle, comp_agg, fail_names,
                args.out / "bundle_comparison.pdf")
    emit_table(h_bundle, comp_agg, agg, args.out / "head_to_head.tex")

    # Console summary
    print("Fits (log-log, median_ms vs n_vertices, chain+tree):")
    for k, f in sorted(fits.items()):
        if f["b"] is None:
            continue
        print(f"  {k:30s}  b={f['b']:+.3f}  "
              f"[{f['b_ci_low']:+.3f}, {f['b_ci_high']:+.3f}]  "
              f"R²={f['r2']:.3f}  n={f['n']}")
    print(f"\nArtefacts written to {args.out}/")


if __name__ == "__main__":
    main()
