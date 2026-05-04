"""Read signedkan_wip/experiments/results/compare.json, compute
median ± std across seeds for each (model, dataset) cell, and
write a LaTeX table snippet for the paper plus a small bar-plot
figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "signedkan_wip" / "experiments" / "results" / "compare.json"
PAPER_DIR = REPO / "signedkan_wip" / "paper"
FIG_DIR = PAPER_DIR / "figures"
SECT_DIR = PAPER_DIR / "sections"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SECT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = json.loads(RESULTS.read_text())
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["dataset"])].append(r)

    DISPLAY = {
        "signedkan":  "SignedKAN (sign-aware)",
        "vanillakan": "VanillaKAN (sign-blind)",
    }
    DSDISP = {"bitcoin_alpha": "Bitcoin Alpha",
              "bitcoin_otc":   "Bitcoin OTC"}

    # ─── LaTeX table ───────────────────────────────────────────────
    tex = []
    tex.append(r"\begin{tabular}{@{}llrrrr@{}}")
    tex.append(r"\toprule")
    tex.append(r"dataset & model & AUC & macro-F1 & binary-F1 & "
               r"params \\")
    tex.append(r"\midrule")
    for ds in ["bitcoin_alpha", "bitcoin_otc"]:
        for m in ["signedkan", "vanillakan"]:
            cell = grouped[(m, ds)]
            auc = np.median([c["test_auc"] for c in cell])
            aus = np.std([c["test_auc"] for c in cell])
            f1m = np.median([c["test_f1_macro"] for c in cell])
            f1ms = np.std([c["test_f1_macro"] for c in cell])
            f1b = np.median([c["test_f1_binary"] for c in cell])
            f1bs = np.std([c["test_f1_binary"] for c in cell])
            np_ = cell[0]["n_params"]
            ds_label = DSDISP[ds] if m == "signedkan" else ""
            tex.append(
                f"{ds_label} & {DISPLAY[m]} & "
                f"${auc:.3f}\\pm{aus:.3f}$ & "
                f"$\\mathbf{{{f1m:.3f}}}\\pm{f1ms:.3f}$ & "
                f"${f1b:.3f}\\pm{f1bs:.3f}$ & "
                f"{np_:,} \\\\"
                if m == "signedkan" else
                f"{ds_label} & {DISPLAY[m]} & "
                f"${auc:.3f}\\pm{aus:.3f}$ & "
                f"${f1m:.3f}\\pm{f1ms:.3f}$ & "
                f"${f1b:.3f}\\pm{f1bs:.3f}$ & "
                f"{np_:,} \\\\"
            )
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"  # last midrule → bottomrule
    tex.append(r"\end{tabular}")
    tex_out = SECT_DIR / "compare_table.tex"
    tex_out.write_text("\n".join(tex) + "\n")
    print(f"wrote {tex_out}")

    # ─── Bar plot: macro-F1 per (dataset, model) ──────────────────
    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=120)
    labels = ["Bitcoin Alpha", "Bitcoin OTC"]
    x = np.arange(len(labels))
    w = 0.35
    sk_med = []; sk_std = []
    vk_med = []; vk_std = []
    for ds in ["bitcoin_alpha", "bitcoin_otc"]:
        sk = [c["test_f1_macro"] for c in grouped[("signedkan", ds)]]
        vk = [c["test_f1_macro"] for c in grouped[("vanillakan", ds)]]
        sk_med.append(np.median(sk));   sk_std.append(np.std(sk))
        vk_med.append(np.median(vk));   vk_std.append(np.std(vk))
    ax.bar(x - w/2, sk_med, w, yerr=sk_std, label="SignedKAN",
           color="#1b6ca8", capsize=3)
    ax.bar(x + w/2, vk_med, w, yerr=vk_std, label="VanillaKAN",
           color="#b02a2a", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("macro-F1 (median over 3 seeds)")
    ax.set_title("Sign-aware vs.\\ sign-blind KAN on link sign prediction")
    ax.set_ylim(0.5, 0.85)
    ax.grid(True, axis="y", ls=":", color="#aaa", lw=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out_pdf = FIG_DIR / "compare_macroF1.pdf"
    out_png = FIG_DIR / "compare_macroF1.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
