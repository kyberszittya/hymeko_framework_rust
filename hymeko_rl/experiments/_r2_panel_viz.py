"""§9 graphical output for the R2 relational update-0 panel: a per-state K6 comparison (flat R1 vs relational R2, oracle
reference) + the per-state nearest-dev-acceptable distance. Reads the frozen artifacts; no re-measurement. Run:

    python -m hymeko_rl.experiments._r2_panel_viz
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R2_DIR = "reports/2026-07-27-coin-r2-relational-update0"
R1_JSON = "reports/2026-07-27-coin-r1-update0/r1_update_zero.json"


def main() -> str:
    fp = json.load(open(f"{R2_DIR}/frozen_panel.json"))
    r1 = json.load(open(R1_JSON))["main_deploy"]["per_state"]
    tags = list(fp["panel"].keys())
    splits = [fp["panel"][t]["split"][:3] for t in tags]
    r2_ok = [int(fp["panel"][t]["delivery_success"]) for t in tags]
    r1_ok = [int(r1[t]["delivery_success"]) for t in tags]
    near = [fp["panel"][t]["nearest_dev_acceptable_dist"] for t in tags]
    labels = [f"{t}\n({s})" for t, s in zip(tags, splits)]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    x = np.arange(len(tags))
    w = 0.38
    ax[0].bar(x - w / 2, r1_ok, w, label="flat R1 (25 240 p)", color="tab:gray")
    ax[0].bar(x + w / 2, r2_ok, w, label="relational R2 (25 774 p)", color="tab:blue")
    ax[0].axhline(1.0, color="tab:green", ls=":", lw=1, label="oracle (teacher θ) = 1")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("frozen K6 delivered (budget 8)")
    ax[0].set_ylim(0, 1.25)
    ax[0].set_title(f"R2 update-0 per-cradle K6 — {fp['verdict']}")
    ax[0].legend(fontsize=8, loc="upper right")

    ax[1].bar(x, near, 0.6, color=["tab:blue" if s == "dev" else "tab:red" for s in splits])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("‖selected canonical θ − nearest dev-acceptable‖ (norm)")
    ax[1].set_title("R2 proposal distance to the dev acceptable set")
    fig.tight_layout()
    out = f"{R2_DIR}/r2_panel.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return out


def basin_main() -> str:
    """§9 plot for the frozen search-basin audit: per held-out cradle, K6 vs interpolation α (direct centre + budget-8),
    with the actor→teacher and actor→nearest-working distances (in SEARCH_STD units) annotated."""
    a = json.load(open(f"{R2_DIR}/basin_audit.json"))
    states = list(a["per_state"].keys())
    fig, ax = plt.subplots(1, len(states), figsize=(5.4 * len(states), 4.2), dpi=140, squeeze=False)
    for k, tag in enumerate(states):
        r = a["per_state"][tag]
        al = [s["alpha"] for s in r["sweep"]]
        direct = [int(s["direct_k6"]) for s in r["sweep"]]
        s8 = [int(s["search8_k6"]) for s in r["sweep"]]
        ax[0][k].plot(al, direct, "o-", color="tab:gray", label="direct centre (budget 1)")
        ax[0][k].plot(al, s8, "s-", color="tab:blue", label="budget-8 search")
        if r["alpha_search8_first_k6"] is not None:
            ax[0][k].axvline(r["alpha_search8_first_k6"], color="tab:green", ls=":", lw=1.2,
                             label=f"α*={r['alpha_search8_first_k6']} (budget-8 restores)")
        ax[0][k].set_xlabel("α  (0 = frozen actor θ → 1 = teacher θ)")
        ax[0][k].set_ylabel("frozen K6 delivered")
        ax[0][k].set_ylim(-0.1, 1.2)
        ax[0][k].set_title(f"{tag} ({r['split']}) — {r['mechanism']}\n"
                           f"d(actor→working)={r['d_actor_to_nearest_heldout_acc_norm']} "
                           f"= {r['reach_in_search_std_units']}×SEARCH_STD", fontsize=8)
        ax[0][k].legend(fontsize=7, loc="center right")
    fig.tight_layout()
    out = f"{R2_DIR}/basin_audit.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
