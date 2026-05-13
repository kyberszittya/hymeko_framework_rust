"""Process signedkan_wip/experiments/results/entropy_sweep.json and
emit a LaTeX table for §IV.7 (entropy-regularised SignedKAN) plus a
small bar plot showing the best entropy-reg cell against the
unregularised baseline on each dataset.
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
RESULTS = REPO / "signedkan_wip" / "experiments" / "results" / "entropy_sweep.json"
FIG_DIR = REPO / "signedkan_wip" / "paper" / "figures"
SECT_DIR = REPO / "signedkan_wip" / "paper" / "sections"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SECT_DIR.mkdir(parents=True, exist_ok=True)


def median_std(rows, key):
    vs = np.asarray([r[key] for r in rows], dtype=float)
    return float(np.median(vs)), float(np.std(vs))


def main():
    rows = json.loads(RESULTS.read_text())
    grouped: dict = defaultdict(list)
    for r in rows:
        if r["model"] == "signedkan":
            key = (r["dataset"], "baseline", 0.0, 0.0)
        else:
            key = (r["dataset"], "entropy",
                   r.get("entropy_lam0", 0.0),
                   r.get("entropy_target", 0.0))
        grouped[key].append(r)

    DSDISP = {"bitcoin_alpha": "Bitcoin Alpha",
              "bitcoin_otc":   "Bitcoin OTC"}

    # ─── LaTeX table: full grid per dataset ────────────────────────
    tex = []
    tex.append(r"\begin{tabular}{@{}llrrrr@{}}")
    tex.append(r"\toprule")
    tex.append(r"dataset & arm & $\lambda_0$ / $H^*$ & "
               r"AUC & macro-$F_1$ & $H_{\mathrm{norm}}$ \\")
    tex.append(r"\midrule")
    for ds in ["bitcoin_alpha", "bitcoin_otc"]:
        # baseline first
        baseline = grouped.get((ds, "baseline", 0.0, 0.0), [])
        if baseline:
            auc_m, auc_s = median_std(baseline, "test_auc")
            f1_m, f1_s = median_std(baseline, "test_f1_macro")
            tex.append(
                f"{DSDISP[ds]} & SignedKAN (no reg) & --- & "
                f"${auc_m:.3f}\\pm{auc_s:.3f}$ & "
                f"${f1_m:.3f}\\pm{f1_s:.3f}$ & --- \\\\"
            )
        # entropy variants
        ent_keys = sorted(k for k in grouped if k[0] == ds and k[1] == "entropy")
        for k in ent_keys:
            cells = grouped[k]
            auc_m, auc_s = median_std(cells, "test_auc")
            f1_m, f1_s = median_std(cells, "test_f1_macro")
            h_m, h_s = median_std(cells, "last_h_norm")
            lam0 = k[2]; tgt = k[3]
            tex.append(
                f" & + entropy reg & "
                f"$\\lambda_0\\!=\\!{lam0:.3g}$, $H^*\\!=\\!{tgt:.2g}$ & "
                f"${auc_m:.3f}\\pm{auc_s:.3f}$ & "
                f"${f1_m:.3f}\\pm{f1_s:.3f}$ & "
                f"${h_m:.2f}\\pm{h_s:.2f}$ \\\\"
            )
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    out_tex = SECT_DIR / "entropy_table.tex"
    out_tex.write_text("\n".join(tex) + "\n")
    print(f"wrote {out_tex}")

    # ─── Bar plot: best entropy cell vs baseline per dataset ──────
    best_per_ds = {}
    for ds in ["bitcoin_alpha", "bitcoin_otc"]:
        ent_keys = [k for k in grouped if k[0] == ds and k[1] == "entropy"]
        if not ent_keys:
            continue
        best = max(ent_keys,
                   key=lambda k: median_std(grouped[k], "test_f1_macro")[0])
        best_per_ds[ds] = best

    if not best_per_ds:
        print("no entropy cells; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=120)
    x = np.arange(len(best_per_ds))
    w = 0.35
    base_med, base_std, ent_med, ent_std, labels = [], [], [], [], []
    for ds, k in best_per_ds.items():
        baseline = grouped.get((ds, "baseline", 0.0, 0.0), [])
        bm, bs = median_std(baseline, "test_f1_macro")
        em, es = median_std(grouped[k], "test_f1_macro")
        base_med.append(bm); base_std.append(bs)
        ent_med.append(em);  ent_std.append(es)
        labels.append(DSDISP[ds])
    ax.bar(x - w/2, base_med, w, yerr=base_std, label="SignedKAN (no reg)",
           color="#888", capsize=3)
    ax.bar(x + w/2, ent_med, w, yerr=ent_std, label="+ entropy reg (best $\\lambda_0$, $H^*$)",
           color="#2e7d32", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("macro-$F_1$ (median over 3 seeds)")
    ax.set_title("Effect of spectral-entropy regularisation on SignedKAN")
    ax.set_ylim(0.55, 0.85)
    ax.grid(True, axis="y", ls=":", color="#aaa", lw=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out_pdf = FIG_DIR / "entropy_effect.pdf"
    out_png = FIG_DIR / "entropy_effect.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
