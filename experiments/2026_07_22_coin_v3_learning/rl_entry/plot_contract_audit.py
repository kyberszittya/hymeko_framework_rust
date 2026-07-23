"""Figure for COIN_TASK_CONTRACT_SENSITIVITY_AUDIT_V1: (A) success ladder — the K3→K6 dwell cliff that binary strict-K6
hides; (B) v3 reward per-step decomposition — the dense signal is dominated by NEGATIVE grasp-pose terms that penalise
valid push-delivery, zone_progress ≈ 0, leaving only the sparse K6 terminal aligned. Reads the audit JSON."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
d = json.load(open(f"{D}/task_contract_audit_v1.json"))
lad = d["success_ladder"]; rew = d["reward_vs_ladder"]
CS = ["pi0", "h30", "repaired"]; COL = {"pi0": "#2980b9", "h30": "#c0392b", "repaired": "#e67e22"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.9))
rungs = ["target_entry", "one_step_in_zone", "k3_dwell", "k6_dwell", "k10_dwell"]
xl = ["entry", "in-zone", "K3\n(0.03s)", "K6\n(0.06s)", "K10\n(0.10s)"]
for c in CS:
    a1.plot(range(len(rungs)), [lad[c][k] for k in rungs], "-o", color=COL[c], label=c)
a1.axvspan(2.5, 3.5, color="grey", alpha=0.12)
a1.annotate("canonical\nstrict-K6", (3, 0.05), fontsize=8, ha="center")
a1.set_xticks(range(len(rungs))); a1.set_xticklabels(xl, fontsize=8); a1.set_ylabel("fraction of dev states")
a1.set_title("Success ladder — steep K3→K6 cliff (K10=0 for all)\nbinary strict-K6 hides graded sub-K6 competence", fontsize=9.5)
a1.legend(fontsize=9); a1.grid(alpha=0.3)

terms = ["zone_progress", "grasp_approach", "both_approach", "terminal_deliver_graded", "body_progress_penalty", "out_of_bounds"]
x = np.arange(len(terms)); w = 0.38
a2.bar(x - w / 2, [rew["pi0"]["mean_component_per_step"][t] for t in terms], w, label="pi_0", color="#2980b9")
a2.bar(x + w / 2, [rew["h30"]["mean_component_per_step"][t] for t in terms], w, label="h30", color="#c0392b")
a2.axhline(0, color="k", lw=0.6)
a2.set_xticks(x); a2.set_xticklabels([t.replace("_", "\n") for t in terms], fontsize=7.5)
a2.set_ylabel("mean reward contribution / step"); a2.legend(fontsize=9); a2.grid(axis="y", alpha=0.3)
a2.set_title("v3 reward decomposition — negative grasp-pose terms dominate;\nzone_progress≈0; only the sparse K6 terminal is aligned", fontsize=9.5)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{D}/task_contract_audit.{ext}", dpi=140)
print(f"wrote {D}/task_contract_audit.svg / .png")
