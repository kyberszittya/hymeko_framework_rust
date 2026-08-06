"""Figure for REPAIR_H30_PLANNER_OBJECTIVE_V1: floor-robust UNQUALIFIED. (A) contact + strict at floor 0.75 / 0.50 vs
pi_0 (lowering the floor makes contact WORSE — not a threshold artifact); (B) delivering vs non-delivering per-state
contact (delivery costs contact at both floors). Reads the two floor JSONs; no recompute."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
d75 = json.load(open(f"{D}/planner_repair_qualify_v1_floor075.json"))
d05 = json.load(open(f"{D}/planner_repair_qualify_v1_floor05.json"))
pi0 = d75["pi0_reference"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
# panel A: required contact + strict, pi_0 vs repaired@0.75 vs repaired@0.50
groups = ["required contact", "strict K6", "exit<K6"]
pi0v = [pi0["req_contact"], pi0["strict_rate"], pi0["exit_before_k6"]]
r75 = [d75["final_qualification"]["aggregate"]["planner_req_contact"], d75["final_qualification"]["aggregate"]["planner_strict_rate"], d75["final_qualification"]["aggregate"]["planner_exit_before_k6"]]
r05 = [d05["final_qualification"]["aggregate"]["planner_req_contact"], d05["final_qualification"]["aggregate"]["planner_strict_rate"], d05["final_qualification"]["aggregate"]["planner_exit_before_k6"]]
x = np.arange(len(groups)); w = 0.26
a1.bar(x - w, pi0v, w, label="pi_0", color="#2980b9")
a1.bar(x, r75, w, label="repaired floor=0.75", color="#c0392b")
a1.bar(x + w, r05, w, label="repaired floor=0.50", color="#e67e22")
a1.axhline(pi0["req_contact"], color="#2980b9", ls="--", lw=0.8, alpha=0.6)
a1.set_xticks(x); a1.set_xticklabels(groups); a1.legend(fontsize=8); a1.grid(axis="y", alpha=0.3)
a1.set_title("Floor-robust: lower floor → contact WORSE (0.198→0.132), not better")

# panel B: per-state delivering vs non-delivering contact, both floors
def split(d):
    ps = d["per_state"]; b = "stable_entry"
    rc = np.array([s["repaired"][b]["metrics"]["required_contact_retention"] for s in ps])
    st = np.array([int(s["repaired"][b]["metrics"]["strict_success"]) for s in ps])
    return rc[st == 1], rc[st == 0]

d75_del, d75_non = split(d75); d05_del, d05_non = split(d05)
positions = [1, 2, 4, 5]
bp = a2.boxplot([d75_del, d75_non, d05_del, d05_non], positions=positions, widths=0.7, patch_artist=True)
for patch, col in zip(bp["boxes"], ["#c0392b", "#2980b9", "#e67e22", "#2980b9"]):
    patch.set_facecolor(col); patch.set_alpha(0.6)
a2.set_xticks([1, 2, 4, 5]); a2.set_xticklabels(["deliver\n@0.75", "no-deliver\n@0.75", "deliver\n@0.50", "no-deliver\n@0.50"], fontsize=8)
a2.set_ylabel("per-state required contact retention"); a2.grid(axis="y", alpha=0.3)
a2.set_title(f"{d75['verdict'][:40]}…\ndelivering states hold LESS contact (both floors)", fontsize=9)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{D}/planner_repair.{ext}", dpi=140)
print(f"wrote {D}/planner_repair.svg / .png")
