"""Plot EQUAL_BUDGET_KMODE_ABLATION_V1: K6-rate per shape × K (equal total budget) + the selected-mode exploration
histogram (proof the K-mode arms did NOT collapse to K=1). Reuses the manifest; no re-measurement."""
import collections
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "reports/2026-07-24-kmode-budget-ablation"


def main():
    m = json.load(open(f"{OUT}/kmode_budget.json"))
    ks = m["K_values"]
    shapes = list(m["results"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # -- panel 1: grouped K6-rate bars per shape × K, plus the across-shape aggregate --
    x = np.arange(len(shapes) + 1)
    w = 0.8 / len(ks)
    agg = {k: [] for k in ks}
    for si, sh in enumerate(shapes):
        for k in ks:
            for rec in m["results"][sh]["records"]:
                agg[k] += rec[f"K{k}"]["k6"]
    for j, k in enumerate(ks):
        rates = [m["results"][sh]["k6_rate"][f"K{k}"] for sh in shapes] + [round(float(np.mean(agg[k])), 3)]
        bars = ax1.bar(x + j * w, rates, w, label=f"K={k}" + (" (single-head)" if k == 1 else ""))
        for b, r in zip(bars, rates):
            ax1.text(b.get_x() + b.get_width() / 2, r + 0.005, f"{r:.2f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x + 0.4 - w / 2)
    ax1.set_xticklabels([s.replace("_", " ") for s in shapes] + ["ALL (agg)"], fontsize=9)
    ax1.set_ylabel("physical K6 delivery rate")
    ax1.set_title(f"Equal total budget B={m['budget_total']}: single-head (K=1) vs K-mode\n"
                  "(K-mode gets NO extra candidates — B split across modes)", fontsize=9)
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_ylim(0, 0.42)
    ax1.grid(axis="y", alpha=0.3)

    # -- panel 2: selected-mode histogram at K=max (exploration happened; budget split starves non-argmax modes) --
    kmax = max(ks)
    picks = collections.Counter()
    split = None
    for sh in shapes:
        for rec in m["results"][sh]["records"]:
            for md in rec[f"K{kmax}"]["modes"]:
                picks[md] += 1
            split = rec[f"K{kmax}"]["budget"]
    modes = sorted(picks)
    ax2.bar([str(mm) for mm in modes], [picks[mm] for mm in modes], color="#c44", alpha=0.8)
    ax2.set_xlabel("winning template-mode id (0 = classifier argmax)")
    ax2.set_ylabel(f"times this mode won (K={kmax}, all shapes×seeds)")
    ax2.set_title(f"K={kmax} exploration: non-argmax modes DO win\nbudget split {split} → non-argmax modes get 1 probe each",
                  fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/kmode_budget.png", dpi=130)
    print(f"wrote {OUT}/kmode_budget.png")

    # aggregate Δ vs single-head (paired over states×seeds), bootstrap
    lo_k = min(ks)
    print("\nAggregate K6-rate (all shapes):", {f"K{k}": round(float(np.mean(agg[k])), 3) for k in ks})
    rng = np.random.default_rng(0)
    for k in ks:
        if k == lo_k:
            continue
        d = np.asarray(agg[k], float) - np.asarray(agg[lo_k], float)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        print(f"  Δ(K{k}-K{lo_k}) mean {d.mean():+.4f}  boot95 [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  (n={len(d)})")


if __name__ == "__main__":
    sys.exit(main())
