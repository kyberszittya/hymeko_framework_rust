"""Phase 10 — master architecture comparison table.

Consolidates phase 6 (small+synth panel), phase 7 (Slashdot k=3 sweep),
phase 8 (Bitcoin 5-seed + SiGAT), and phase 9 (k=3+4+5 mixed-arity)
into a single Markdown table with mean±std AUC and F1m for every
(arch, dataset) cell at 5 seeds.

Output: signedkan_wip/experiments/results/master_table.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SOURCES = [
    ("phase6", "signedkan_wip/experiments/results/phase6_small_synth.json"),
    ("phase7_15k", "signedkan_wip/experiments/results/phase7_slashdot.json"),
    ("phase7_100k", "signedkan_wip/experiments/results/phase7_slashdot_5seed_k3_100k.json"),
    ("phase7_200k", "signedkan_wip/experiments/results/phase7_slashdot_5seed_k3_200k.json"),
    ("phase8", "signedkan_wip/experiments/results/phase8_bitcoin_5seed.json"),
    ("phase9", "signedkan_wip/experiments/results/phase9_k345_mixed.json"),
]


def load_summaries():
    bag = {}  # (arch, dataset) -> {auc_seeds, f1m_seeds, source, alpha}
    for src, path in SOURCES:
        if not Path(path).exists():
            continue
        d = json.load(open(path))
        summary = d.get("summary", {})
        for key, cell in summary.items():
            if "|" in key:
                arch, dataset = key.split("|")
            else:
                arch, dataset = key, "slashdot"
            tag = (arch, dataset)
            existing = bag.get(tag)
            entry = {
                "source": src,
                "auc_seeds": cell.get("auc_seeds", []),
                "f1m_seeds": cell.get("f1m_seeds", []),
                "alpha_seeds": cell.get("alpha_seeds"),
                "n_seeds": cell.get("n_seeds", 0),
            }
            if existing is None or entry["n_seeds"] > existing["n_seeds"]:
                bag[tag] = entry
    return bag


def fmt_cell(entry):
    aucs = entry["auc_seeds"]; f1s = entry["f1m_seeds"]
    if not aucs:
        return "—"
    aucs = [a for a in aucs if not (a is None or a != a)]
    f1s  = [f for f in f1s  if not (f is None or f != f)]
    if not aucs:
        return f"NaN/F1={np.mean(f1s):.3f}"
    return (f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}"
            f"<br>F1={np.mean(f1s):.3f}")


def main():
    bag = load_summaries()
    archs = sorted({a for a, d in bag})
    datasets = sorted({d for a, d in bag})

    # Pretty arch ordering
    arch_order = [
        "mlp_blind", "gcn_blind",
        "signedkan_L1",
        "hsikan_k3_only_leanest", "hsikan_k34", "hsikan_k45", "hsikan_k345",
        "hsikan_mixed_leanest",  # alias
        "sgcn_balance", "sigat_attn",
    ]
    arch_order = [a for a in arch_order if a in archs] + \
                  [a for a in archs if a not in arch_order]
    archs = arch_order

    # Pretty dataset ordering
    ds_order = [
        "karate", "sbm_n200_k4_s0", "sbm_n400_k5_s0", "hier_n240_s0",
        "bitcoin_alpha", "bitcoin_otc", "slashdot",
    ]
    ds_order = [d for d in ds_order if d in datasets] + \
                [d for d in datasets if d not in ds_order]
    datasets = ds_order

    lines = []
    lines.append("# Master architecture comparison table")
    lines.append("")
    lines.append("Each cell: mean±std AUC over 5 seeds, with F1-macro on the next line.")
    lines.append("")
    lines.append("| arch \\ dataset | " + " | ".join(datasets) + " |")
    lines.append("|---" * (len(datasets) + 1) + "|")
    for a in archs:
        cells = [fmt_cell(bag[(a, d)]) if (a, d) in bag else "—" for d in datasets]
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-dataset best
    lines.append("## Best architecture per dataset (by mean AUC)")
    lines.append("")
    for d in datasets:
        candidates = []
        for a in archs:
            entry = bag.get((a, d))
            if not entry: continue
            aucs = [x for x in entry["auc_seeds"] if x is not None and x == x]
            if not aucs: continue
            candidates.append((np.mean(aucs), a, np.std(aucs)))
        if not candidates: continue
        candidates.sort(reverse=True)
        top = candidates[:3]
        line = f"- **{d}**: "
        line += "; ".join(f"{a} {auc:.3f}±{std:.3f}" for auc, a, std in top)
        lines.append(line)
    lines.append("")

    # αₖ summary for k=345 cells
    if any((a, d) in bag and bag[(a, d)].get("alpha_seeds")
           for a in ["hsikan_k345", "hsikan_k34", "hsikan_k45"]
           for d in datasets):
        lines.append("## Learned αₖ (mean over 5 seeds)")
        lines.append("")
        lines.append("| arch \\ dataset | " + " | ".join(datasets) + " |")
        lines.append("|---" * (len(datasets) + 1) + "|")
        for a in ["hsikan_k34", "hsikan_k45", "hsikan_k345"]:
            row = [a]
            for d in datasets:
                entry = bag.get((a, d))
                if not entry or not entry.get("alpha_seeds"):
                    row.append("—")
                else:
                    alphas = np.array(entry["alpha_seeds"])
                    means = alphas.mean(axis=0)
                    row.append(", ".join(f"{m:.2f}" for m in means))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    out_path = Path("signedkan_wip/experiments/results/master_table.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
