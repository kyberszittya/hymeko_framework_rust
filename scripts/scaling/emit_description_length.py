#!/usr/bin/env python3
"""
emit_description_length.py — produce the description-length figure and
LaTeX table comparing source bytes a user maintains to get the same set
of emitted formats via HyMeKo vs the standard single-target stack.

Framing:
  HyMeKo      : 1 .hymeko source file → 6 emitted formats (by Prop. 2)
  URDF-alone  : 1 URDF file → 1 format (no coherence guarantee for others)
  N-file stack: N hand-maintained files → N formats (drift risk, O(N) upkeep)

Outputs:
  description_length.pdf   size-vs-n_vertices, four traces
  description_length.tex   LaTeX table at representative fixtures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REPRESENTATIVE = [
    ("chain_100",       "chain, 100 links"),
    ("tree_1000_k3",    "tree (k=3), 1000 links"),
    ("humanoid_f0",     "humanoid (Atlas-class)"),
    ("humanoid_f5",     "humanoid with 5 fingers/hand"),
    ("quadruped_d5_t0", "quadruped (Spot/ANYmal-class)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hymeko-manifest", type=Path, required=True)
    ap.add_argument("--urdf-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    h = {x["name"]: x for x in json.loads(args.hymeko_manifest.read_text())}
    u = {x["name"]: x for x in json.loads(args.urdf_manifest.read_text())}

    # ── Figure: bytes vs |V|, chain + tree families ──────────────────
    fam_sizes = []
    for name, entry in h.items():
        if entry["family"] not in ("chain", "tree"):
            continue
        if name not in u:
            continue
        fam_sizes.append((entry["family"], entry["n_vertices"],
                          entry["source_bytes"], u[name]["source_bytes"]))
    fam_sizes.sort(key=lambda r: (r[0], r[1]))
    arr = np.array([(v, hb, ub) for (_fam, v, hb, ub) in fam_sizes])
    V, H_bytes, U_bytes = arr[:, 0], arr[:, 1], arr[:, 2]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(V, H_bytes, "o-", color="#1f77b4", markersize=5,
              label="HyMeKo source (1 file → 6 formats)")
    ax.loglog(V, U_bytes, "s-", color="#ff7f0e", markersize=5,
              label="URDF source (1 file → 1 format only)")
    ax.loglog(V, 6 * U_bytes, "^--", color="#d62728", markersize=5,
              alpha=0.8,
              label=r"6 hand-maintained format files (drift risk)")
    ax.set_xlabel(r"$|V|$ (links)")
    ax.set_ylabel("source bytes a user maintains")
    ax.set_title("Description length to cover 6 coherent formats")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "description_length.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Table: per-format-maintained ratio ──────────────────────────
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Source bytes a user maintains to cover the same set of "
        r"emitted formats. HyMeKo's one source authoritatively describes all "
        r"six targets (Proposition~\ref{prop:alias}); the standard stack "
        r"needs one file per format with no coherence guarantee. "
        r"Ratio is URDF-alone bytes over HyMeKo bytes per format emitted.}",
        r"\label{tab:description_length}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Fixture & HyMeKo [B] & URDF [B] & HyMeKo / 6 fmts [B] & Ratio \\",
        r"\midrule",
    ]
    for name, label in REPRESENTATIVE:
        if name not in h or name not in u:
            continue
        hb = h[name]["source_bytes"]
        ub = u[name]["source_bytes"]
        per_fmt = hb / 6.0
        ratio = ub / per_fmt
        lines.append(
            f"{label} & {hb} & {ub} & {per_fmt:.0f} & {ratio:.1f}$\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (args.out / "description_length.tex").write_text(
        "\n".join(lines), encoding="utf-8")

    # Console summary
    print(f"{'fixture':25s} {'|V|':>4s} {'.hymeko':>9s} {'.urdf':>8s} "
          f"{'per-fmt ratio':>14s}")
    for name, _label in REPRESENTATIVE:
        if name not in h or name not in u:
            continue
        hb = h[name]["source_bytes"]
        ub = u[name]["source_bytes"]
        print(f"{name:25s} {h[name]['n_vertices']:>4d} {hb:>9d} {ub:>8d} "
              f"{ub * 6 / hb:>13.1f}×")
    print(f"\nArtefacts: {args.out}/")


if __name__ == "__main__":
    main()
