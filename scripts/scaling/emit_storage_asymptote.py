#!/usr/bin/env python3
r"""
emit_storage_asymptote.py — produce the storage-overhead asymptote
figure (witness for Proposition 4) from the highArityFixedPool fixture
manifest.

Plots the theoretical overhead bound

    bound(n, m, d̄) = (n + m) / (m · d̄)

across the swept arities, alongside the corresponding ρ = 1 + bound.
With the highArityFixedPool generator's choice n = n_pool fixed and
m fixed, the bound shrinks as 1/d̄, witnessing ρ → 1 as d̄ → ∞.

Output:
    storage_asymptote.pdf   log-log plot of bound and ρ vs d̄
    storage_asymptote.tex   LaTeX table at representative d̄ values
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Path to the fixtures index.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [x for x in manifest if x["family"] == "highArityFixedPool"]
    if not rows:
        raise SystemExit("no highArityFixedPool fixtures in manifest")

    rows.sort(key=lambda r: r["mean_arity"])
    d_bar = np.array([r["mean_arity"] for r in rows])
    n = np.array([r["n_vertices"] for r in rows], dtype=float)
    m = np.array([r["n_hyperedges"] for r in rows], dtype=float)
    bound = (n + m) / (m * d_bar)
    rho = 1.0 + bound
    n_pool = int(rows[0]["n_vertices"])
    m_fixed = int(rows[0]["n_hyperedges"])

    # ── Figure: log-log of bound vs d̄ ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(d_bar, bound, "o-", color="tab:blue", markersize=6,
              label=r"measured bound $(n+m)/(m\bar{d})$")
    ax.loglog(d_bar, rho - 1.0, "x", color="tab:blue", alpha=0.0,
              label=None)  # invisible — anchors second-axis units
    # Reference line: 1/d̄ slope (theoretical asymptote shape).
    # Anchor at d=2 to match measured values.
    anchor_idx = 0
    ref_const = bound[anchor_idx] * d_bar[anchor_idx]
    ax.loglog(d_bar, ref_const / d_bar, "--", color="gray", alpha=0.6,
              label=r"$\propto 1/\bar{d}$ reference slope")
    ax.set_xlabel(r"Mean hyperedge arity $\bar{d}$")
    ax.set_ylabel(r"$(n+m)/(m\bar{d})$  (bound; $\rho = 1 + \text{bound}$)")
    ax.set_title(r"Storage-overhead asymptote: $\rho \to 1$ as $\bar{d} \to \infty$"
                 f"\n(highArityFixedPool, $n={n_pool}$ fixed, $m={m_fixed}$ fixed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Annotate ρ values at the swept points
    for d_val, b_val in zip(d_bar, bound):
        ax.annotate(rf"$\rho={1+b_val:.3f}$",
                    xy=(d_val, b_val), xytext=(6, -10),
                    textcoords="offset points", fontsize=7,
                    color="tab:blue")

    fig.tight_layout()
    fig.savefig(args.out / "storage_asymptote.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── LaTeX table ───────────────────────────────────────────────────
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Storage-overhead asymptote witness: with $n$ and $m$ "
        r"held fixed and $\bar{d}$ swept, the bound "
        r"$(n+m)/(m\bar{d})$ shrinks as $1/\bar{d}$ and the overhead "
        r"ratio $\rho$ approaches unity as $\bar{d}$ grows. "
        rf"Family: \textsf{{highArityFixedPool}}, $n={n_pool}$, $m={m_fixed}$.}}",
        r"\label{tab:storage_asymptote}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$\bar{d}$ & $n$ & $m$ & bound $(n+m)/(m\bar{d})$ & $\rho$ \\",
        r"\midrule",
    ]
    for r in rows:
        d = r["mean_arity"]
        nv, me = r["n_vertices"], r["n_hyperedges"]
        b = (nv + me) / (me * d)
        rho_val = 1.0 + b
        lines.append(
            f"{int(d):>3d} & {nv} & {me} & {b:.4f} & {rho_val:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (args.out / "storage_asymptote.tex").write_text(
        "\n".join(lines), encoding="utf-8")

    # Console summary
    print(f"highArityFixedPool sweep (n={n_pool}, m={m_fixed} fixed):")
    print(f"  {'d̄':>4s}  {'bound':>8s}  {'ρ':>8s}")
    for r in rows:
        d = int(r["mean_arity"])
        b = (r["n_vertices"] + r["n_hyperedges"]) / (r["n_hyperedges"] * r["mean_arity"])
        print(f"  {d:>4d}  {b:>8.4f}  {1+b:>8.4f}")
    print(f"\nArtefacts: {args.out}/")


if __name__ == "__main__":
    main()
