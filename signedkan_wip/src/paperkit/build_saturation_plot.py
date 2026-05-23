"""Build saturation-curve figure: test AUC + macro-F1 vs epochs,
SignedKAN vs VanillaKAN, on both Bitcoin datasets. Drops figure at
signedkan_wip/paper/figures/saturation.{pdf,png}.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "signedkan_wip" / "experiments" / "results" / "saturation.json"
FIG_DIR = REPO / "signedkan_wip" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = json.loads(RESULTS.read_text())
    grouped: dict = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["dataset"], r["n_epochs"])].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=120)
    DSDISP = {"bitcoin_alpha": "Bitcoin Alpha",
              "bitcoin_otc":   "Bitcoin OTC"}
    COLOR  = {"signedkan":  "#1b6ca8",
              "vanillakan": "#b02a2a"}
    LABEL  = {"signedkan":  "SignedKAN",
              "vanillakan": "Vanilla KAN"}

    for ax, ds in zip(axes, ["bitcoin_alpha", "bitcoin_otc"]):
        for model in ("signedkan", "vanillakan"):
            xs, ys, ystds = [], [], []
            for (m, d, e), cell in sorted(grouped.items()):
                if m != model or d != ds:
                    continue
                aucs = [c["test_auc"] for c in cell]
                xs.append(e)
                ys.append(np.median(aucs))
                ystds.append(np.std(aucs))
            xs   = np.array(xs)
            ys   = np.array(ys)
            ystds = np.array(ystds)
            ax.errorbar(xs, ys, yerr=ystds, marker="o",
                        color=COLOR[model], lw=1.6, ms=6,
                        capsize=3, label=LABEL[model])
        ax.set_xlabel("training epochs")
        ax.set_ylabel("test AUC")
        ax.set_title(DSDISP[ds])
        ax.set_xscale("log")
        ax.grid(True, ls=":", color="#aaa", lw=0.5, alpha=0.5)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.suptitle("Saturation curve: test AUC vs.\\ training epochs "
                 "(median over 3 seeds, $\\pm$std)", fontsize=11)
    fig.tight_layout()

    out_pdf = FIG_DIR / "saturation.pdf"
    out_png = FIG_DIR / "saturation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
