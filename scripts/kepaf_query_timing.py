"""Measure predicate-query response-time distribution for the canonical
fixture, write the LaTeX table snippet for Section 6.1, and emit a
small histogram figure.

Output:
  paper/kepaf_v1/figures/query_timing.{pdf,png}    histogram
  paper/kepaf_v1/results/query_timing.tex          \\input-able table

Each query runs 10 000 times; we report median, p99, max, and the
match count.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hymeko.hymeko as hkm

OUT_FIG = REPO / "paper" / "kepaf_v1" / "figures"
OUT_RES = REPO / "paper" / "kepaf_v1" / "results"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

QUERIES = [
    ("KIND(joint)",                                   "KIND(joint)"),
    ("INHERITS(link)",                                "INHERITS(link)"),
    ("SCOPEDIN(robot)",                               "SCOPEDIN(robot)"),
    ("KIND(joint) AND HASARCREF(+1, INHERITS(link))", "joint with $+$link"),
    ("KIND(joint) AND INHERITS(joint)",               "joint $\\cap$ joint"),
]


def main():
    e = hkm.PyHypergraphEngine()
    src = (REPO / "examples" / "paper" / "hymeko_robot.hymeko").read_text()
    ir = e.parse_dsl(src)
    n_v, n_e, n_a = ir.node_count, ir.edge_count, ir.arc_count
    print(f"canonical fixture: V={n_v}  E={n_e}  A={n_a}")

    rows = []  # (label, query, matches, [times])
    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=120)

    n_repeat = 10_000
    colors = ["#1b6ca8", "#b02a2a", "#2e7d32", "#7d4f1a", "#5b2a7a"]

    for (q, label), color in zip(QUERIES, colors):
        # Warm-up.
        for _ in range(100):
            ir.query(q)
        # Measure: per-call timing across n_repeat invocations.
        times = np.empty(n_repeat, dtype=np.float64)
        for i in range(n_repeat):
            t0 = time.perf_counter_ns()
            ir.query(q)
            times[i] = (time.perf_counter_ns() - t0) / 1e3  # μs

        matches = len(ir.query(q))
        rows.append((label, q, matches, times))
        med   = float(np.median(times))
        p99   = float(np.percentile(times, 99))
        mx    = float(times.max())
        print(f"  {q:55s}  matches={matches:3d}  "
              f"median={med:6.2f}us  p99={p99:6.2f}us  max={mx:7.2f}us")

        # Histogram contribution (clip outliers above 99.5p for readability).
        clipped = times[times <= np.percentile(times, 99.5)]
        ax.hist(clipped, bins=50, alpha=0.55, color=color, label=label,
                edgecolor=color)

    ax.set_xlabel(r"per-call query time ($\mu$s)")
    ax.set_ylabel("count over 10\\,000 calls")
    ax.set_title("Predicate-query response-time distribution")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(True, ls=":", color="#aaa", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "query_timing.pdf")
    fig.savefig(OUT_FIG / "query_timing.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_FIG / 'query_timing.pdf'}")

    # LaTeX table snippet.
    tex = []
    tex.append(r"\begin{tabular}{@{}lrrrrr@{}}")
    tex.append(r"\toprule")
    tex.append(r"query & matches & median & p$_{99}$ & max & "
               r"\multicolumn{1}{c}{(\,$\mu$s)} \\")
    tex.append(r"\midrule")
    for label, _, matches, times in rows:
        med = float(np.median(times))
        p99 = float(np.percentile(times, 99))
        mx  = float(times.max())
        tex.append(
            f"{label} & {matches} & {med:.2f} & {p99:.2f} & {mx:.2f} & \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    out_tex = OUT_RES / "query_timing.tex"
    out_tex.write_text("\n".join(tex) + "\n")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
