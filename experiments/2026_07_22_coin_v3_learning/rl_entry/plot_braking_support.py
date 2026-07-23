"""Figure for CONTACT_PRESERVING_BRAKING_PRIMITIVE_V2 Part A: (A) safe-beneficial support fraction by V1 outcome vs the
preregistered 0.5 bar; (B) supported states have HIGH pi_0 radial velocity — support exists only where there is excessive
target-directed velocity to decelerate. Reads braking_support_partA.json; no recompute."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
d = json.load(open(f"{D}/braking_support_partA.json"))
rows = d["per_state"]; bo = d["by_v1_outcome"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
order = ["contact_losing", "target_exit", "delivered_contact_preserving"]
order = [k for k in order if k in bo]
frac = [bo[k]["with_support"] / bo[k]["n"] for k in order]
a1.bar(range(len(order)), frac, color=["#c0392b", "#e67e22", "#27ae60"][:len(order)])
overall = d["gate"]["fraction_with_support"]
a1.axhline(0.5, color="k", ls="--", lw=1, label="preregistered bar 0.5")
a1.axhline(overall, color="#8e44ad", ls=":", lw=1.4, label=f"overall {overall:.0%}")
a1.set_xticks(range(len(order))); a1.set_xticklabels([k.replace("_", "\n") for k in order], fontsize=8)
a1.set_ylabel("fraction of braking states with support"); a1.set_ylim(0, 1); a1.legend(fontsize=8); a1.grid(axis="y", alpha=0.3)
a1.set_title(f"{d['verdict']}\nsupport spread across classes but below the bar", fontsize=9)

sup = [abs(r["pi0_radial_vel"]) for r in rows if r["support"]["n_safe_beneficial"] > 0]
nos = [abs(r["pi0_radial_vel"]) for r in rows if r["support"]["n_safe_beneficial"] == 0]
bp = a2.boxplot([sup, nos], patch_artist=True, widths=0.6)
a2.set_xticks([1, 2]); a2.set_xticklabels(["has\nsupport", "no\nsupport"])
for patch, col in zip(bp["boxes"], ["#27ae60", "#c0392b"]):
    patch.set_facecolor(col); patch.set_alpha(0.6)
a2.set_ylabel("|pi_0 radial coin velocity| (m/s)"); a2.grid(axis="y", alpha=0.3)
a2.set_title("Support exists only where the coin approaches FAST\n(nothing to decelerate when already slow)", fontsize=9)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{D}/braking_support.{ext}", dpi=140)
print(f"wrote {D}/braking_support.svg / .png")
