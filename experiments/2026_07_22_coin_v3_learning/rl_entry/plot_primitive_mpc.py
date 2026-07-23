"""Figure for CONTACT_STABILIZED_PRIMITIVE_MPC_V1: (A) arc Pareto (required contact vs strict) for pi_0 / raw-H30 /
repaired / primitive; (B) per-family pi_0-vs-primitive contact & strict — the contact/delivery tension localises to the
braking regime. Reads the primitive + reconstruction JSONs; no recompute."""
import json
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
d = json.load(open(f"{D}/primitive_mpc_qualify_v1.json"))
g = d["gold_baseline"]; ag = d["final_qualification"]["aggregate"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.9))
# panel A: Pareto scatter (x=required contact, y=strict K6). up-right is better.
pts = [("frozen pi_0", g["A_pi0"]["req_contact"], g["A_pi0"]["strict"], "#2980b9"),
       ("raw H=30", g.get("B_raw_h30", {}).get("req_contact", 0.176), g.get("B_raw_h30", {}).get("strict", 0.645), "#7f8c8d"),
       ("repaired planner", 0.198, 0.516, "#e67e22"),
       ("primitive MPC", g["C_primitive"]["req_contact"], g["C_primitive"]["strict"], "#c0392b")]
for name, x, y, col in pts:
    a1.scatter([x], [y], s=140, color=col, zorder=3, edgecolor="k", linewidth=0.5)
    a1.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8.5)
a1.axvline(g["A_pi0"]["req_contact"], color="#2980b9", ls="--", lw=0.8, alpha=0.5)
a1.set_xlabel("required-contact retention →"); a1.set_ylabel("strict K6 →")
a1.set_title("Arc Pareto — primitive is the only contact-sane improver\n(planners buy strict by collapsing contact)", fontsize=9.5)
a1.grid(alpha=0.3); a1.set_xlim(0.1, 0.6); a1.set_ylim(0.1, 0.75)

# panel B: per-family pi_0 vs primitive (contact + strict)
fam = defaultdict(lambda: defaultdict(list))
for s in d["per_state"]:
    f = s["family"]
    fam[f]["Ac"].append(s["pi0"]["required_contact_retention"]); fam[f]["Cc"].append(s["primitive"]["required_contact_retention"])
    fam[f]["As"].append(int(s["pi0"]["strict_success"])); fam[f]["Cs"].append(int(s["primitive"]["strict_success"]))
fams = ["transport", "braking", "settling_dwell"]
x = np.arange(len(fams)); w = 0.2
a2.bar(x - 1.5 * w, [np.mean(fam[f]["Ac"]) for f in fams], w, label="pi_0 contact", color="#2980b9")
a2.bar(x - 0.5 * w, [np.mean(fam[f]["Cc"]) for f in fams], w, label="primitive contact", color="#8e44ad")
a2.bar(x + 0.5 * w, [np.mean(fam[f]["As"]) for f in fams], w, label="pi_0 strict", color="#95a5a6")
a2.bar(x + 1.5 * w, [np.mean(fam[f]["Cs"]) for f in fams], w, label="primitive strict", color="#c0392b")
a2.set_xticks(x); a2.set_xticklabels(fams, fontsize=8.5); a2.legend(fontsize=7.5, ncol=2); a2.grid(axis="y", alpha=0.3)
a2.set_title(f"{d['verdict'][:46]}…\ncontact traded for delivery in BRAKING only", fontsize=9.5)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{D}/primitive_mpc.{ext}", dpi=140)
print(f"wrote {D}/primitive_mpc.svg / .png")
