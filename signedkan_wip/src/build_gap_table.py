"""Process the gap-closing sweep results into a §IV.8 LaTeX table:
joint-config × (model, dataset) median ± std.

Reads the W (WiP baseline, from compare_h32_alpha.json), E (early-stop
only, from early_stop.json), EC and ECG (from gap_sweep.json) and
emits one consolidated table for the SignedKAN paper.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO    = Path(__file__).resolve().parents[2]
RESDIR  = REPO / "signedkan_wip" / "experiments" / "results"
SECTDIR = REPO / "signedkan_wip" / "paper" / "sections"
SECTDIR.mkdir(parents=True, exist_ok=True)


def load_cfg(label: str, path: str, model: str, dataset: str):
    """Load median+std AUC and macro-F1 from a JSON for a given
    (model, dataset, label) cell. The W (WiP baseline) row checks
    multiple result files because no single JSON has both models for
    both datasets at the right (h=32, 100ep, no-reg) configuration."""
    candidates = [path]
    if label == "W":
        # Multiple sources cover the baseline cell; first hit wins.
        candidates = ["entropy_sweep.json",
                      "compare_h32_alpha.json",
                      "saturation.json"]
    for cpath in candidates:
        p = RESDIR / cpath
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        cell = [r for r in rows
                if r["model"] == model
                and r["dataset"] == dataset
                and ((label == "W"   and not r.get("early_stopping", False)
                                    and not r.get("class_weighted", False)
                                    and r.get("grid", 5) == 5
                                    and r.get("entropy_lam0", 0.0) == 0.0
                                    and r.get("hidden", 32) == 32
                                    and r.get("n_epochs", 100) == 100)
                     or (label == "E"   and r.get("early_stopping", False)
                                        and not r.get("class_weighted", False)
                                        and r.get("grid", 5) == 5)
                     or (label == "EC"  and r.get("cfg") == "EC")
                     or (label == "ECG" and r.get("cfg") == "ECG")
                     or (label == "ECH" and r.get("cfg") == "EC+entropy"))]
        if cell:
            aucs = np.asarray([c["test_auc"] for c in cell])
            f1ms = np.asarray([c["test_f1_macro"] for c in cell])
            return (float(np.median(aucs)), float(np.std(aucs)),
                    float(np.median(f1ms)), float(np.std(f1ms)),
                    len(cell))
    return None


def main():
    DATASETS = [("bitcoin_alpha", "Bitcoin Alpha"),
                ("bitcoin_otc",   "Bitcoin OTC")]
    # entropy_sweep.json carries unregularised baseline rows for both
    # datasets (the "signedkan" rows where ereg is None), so use it as
    # the W lookup. Falls back to compare_h32_alpha if not present.
    LABELS = [
        ("W",   "WiP baseline (100 ep, $G\\!=\\!5$, full-batch BCE)",
         "entropy_sweep.json"),
        ("E",   "+ early stopping",                 "early_stop.json"),
        ("EC",  "+ class-weighted BCE",             "gap_sweep.json"),
        ("ECG", "+ spline grid $G\\!=\\!3$",         "gap_sweep.json"),
        ("ECH", "EC + spectral-entropy reg.",       "entropy_on_ec.json"),
    ]

    tex = []
    tex.append(r"\begin{tabular}{@{}llrrrr@{}}")
    tex.append(r"\toprule")
    tex.append(r"dataset & variant & "
               r"\multicolumn{2}{c}{SignedKAN} & "
               r"\multicolumn{2}{c}{Vanilla KAN} \\")
    tex.append(r"        &         & AUC & macro-$F_1$ "
               r"& AUC & macro-$F_1$ \\")
    tex.append(r"\midrule")
    for ds, ds_label in DATASETS:
        first = True
        for (label, label_text, path) in LABELS:
            sk = load_cfg(label, path, "signedkan",  ds)
            vk = load_cfg(label, path, "vanillakan", ds)
            if sk is None and vk is None:
                continue
            ds_cell = ds_label if first else ""
            first = False
            def fmt(cell):
                if cell is None:
                    return ("---", "---")
                a, _, f, _, _ = cell
                return (f"${a:.3f}$", f"${f:.3f}$")
            sa, sf = fmt(sk)
            va, vf = fmt(vk)
            tex.append(
                f"{ds_cell} & {label_text} & "
                f"{sa} & {sf} & {va} & {vf} \\\\"
            )
        tex.append(r"\midrule")
    tex.append(r"SGCN \cite{derr2018sgcn} (published) & "
               r"--- & "
               r"\textit{0.93} & \textit{---} & --- & --- \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    out = SECTDIR / "gap_table.tex"
    out.write_text("\n".join(tex) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
