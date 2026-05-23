"""Phase 11 + 11b combined positivity-sweep summary.

Reads phase11_positivity_sweep.json (k34) and phase11b_k45_positivity.json
(k45/k345) and produces a Markdown curve table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SOURCES = [
    "signedkan_wip/experiments/results/phase11_positivity_sweep.json",
    "signedkan_wip/experiments/results/phase11b_k45_positivity.json",
]

POSITIVITIES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
ARCHS = ["gcn_blind", "hsikan_k34", "hsikan_k34_lean",
          "hsikan_k345", "hsikan_k45",
          "sgcn_balance", "sigat_attn"]


def main():
    cells = {}
    for path in SOURCES:
        if not Path(path).exists():
            continue
        d = json.load(open(path))
        for key, cell in d.get("summary", {}).items():
            arch, pos_key = key.split("|")
            pos = int(pos_key.replace("pos", ""))
            cells[(arch, pos)] = cell

    archs = sorted({a for (a, _) in cells})
    arch_order = [a for a in ARCHS if a in archs] + \
                  [a for a in archs if a not in ARCHS]

    lines = []
    lines.append("# Positivity-sweep results (phase 11 + 11b)")
    lines.append("")
    lines.append("Synthetic SBM with controllable per-edge sign-positivity.")
    lines.append("`pos_in` ∈ {50,...,95} (within-community P(+)). `pos_out=0.15`")
    lines.append("fixed. Realised %pos in the graph is lower because many edges")
    lines.append("are cross-community.")
    lines.append("")
    lines.append("Each cell: mean±std AUC over 3 seeds.")
    lines.append("")
    header = ["pos_in", "%pos"] + arch_order
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for pos in POSITIVITIES:
        row = [str(pos)]
        any_cell = next((cells[(a, pos)] for a in arch_order if (a, pos) in cells), None)
        if any_cell:
            row.append(f"{any_cell['frac_pos']:.2f}")
        else:
            row.append("—")
        for a in arch_order:
            c = cells.get((a, pos))
            if c:
                row.append(f"{c['auc_mean']:.3f}±{c['auc_std']:.3f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-positivity winner
    lines.append("## Best architecture per positivity (by mean AUC)")
    lines.append("")
    for pos in POSITIVITIES:
        candidates = []
        for a in arch_order:
            c = cells.get((a, pos))
            if not c: continue
            candidates.append((c["auc_mean"], a, c["auc_std"]))
        if not candidates: continue
        candidates.sort(reverse=True)
        top3 = candidates[:3]
        line = (f"- pos_in={pos} (real {next((cells[(a,pos)]['frac_pos'] for a in arch_order if (a,pos) in cells), '?'):.2f}): "
                + "; ".join(f"**{a}** {auc:.3f}±{std:.3f}" for auc, a, std in top3))
        lines.append(line)
    lines.append("")

    # αₖ for k45 / k345 cells
    if any(cells.get((a, p)) and cells[(a, p)].get("alpha_seeds")
           for a in ["hsikan_k45", "hsikan_k345"] for p in POSITIVITIES):
        lines.append("## Learned αₖ across positivity (mean over seeds)")
        lines.append("")
        for a in ["hsikan_k45", "hsikan_k345"]:
            row = [f"`{a}`"]
            for pos in POSITIVITIES:
                c = cells.get((a, pos))
                if not c or not c.get("alpha_seeds"):
                    row.append("—")
                else:
                    alphas = np.array(c["alpha_seeds"])
                    means = alphas.mean(axis=0)
                    row.append(", ".join(f"{m:.2f}" for m in means))
            lines.append(f"- {row[0]}")
            for pos, val in zip(POSITIVITIES, row[1:]):
                lines.append(f"  - pos_in={pos}: α={val}")
        lines.append("")

    out_path = Path("signedkan_wip/experiments/results/positivity_summary.md")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
