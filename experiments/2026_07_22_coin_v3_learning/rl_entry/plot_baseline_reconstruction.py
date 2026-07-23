"""Figure for COIN_FEEDBACK_BASELINE_RECONSTRUCTION_V1: B (frozen pi_0) vs C (H=30 planner) on the qualification-
relevant metrics, over 31 disjoint dev states. Reads baseline_reconstruction_v1.json; no recompute."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
d = json.load(open(f"{D}/baseline_reconstruction_v1.json"))
b, c = d["gold_baseline"]["B_pi0"], d["gold_baseline"]["C_h30_planner"]
q = d["teacher_qualification"]["aggregate"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
# panel 1: normalized qualification metrics (bars), pi_0 vs planner
labels = ["required\ncontact", "strict\nK6", "target\nexit", "max dwell\n/6", "entry vel", "lost req\ncontact"]
bk = ["required_contact_retention", "strict_success", "target_exit", "max_dwell", "entry_velocity", "lost_required_contact"]
bv = [b["required_contact_retention"], b["strict_success"], b["target_exit"], b["max_dwell"] / 6, b["entry_velocity"], b["lost_required_contact"]]
cv = [c["required_contact_retention"], c["strict_success"], c["target_exit"], c["max_dwell"] / 6, c["entry_velocity"], c["lost_required_contact"]]
x = range(len(labels)); w = 0.38
a1.bar([i - w / 2 for i in x], bv, w, label="B: frozen pi_0", color="#2980b9")
a1.bar([i + w / 2 for i in x], cv, w, label="C: H=30 planner (teacher)", color="#c0392b")
a1.set_xticks(list(x)); a1.set_xticklabels(labels, fontsize=8); a1.legend(fontsize=8); a1.grid(axis="y", alpha=0.3)
a1.set_title("B (pi_0) vs C (H=30 teacher) — 31 disjoint dev states")
a1.axhline(0, color="k", lw=0.5)

# panel 2: the qualification verdict — the teacher improves the graded objective but violates the two constraints
clauses = [("contact\npreserved", q["mean_d_contact_retention"], -0.05, False),
           ("exit not\nincreased", -(q["planner_exit_rate"] - q["pi0_exit_rate"]), -0.05, False),
           ("improves\n(dwell Δ)", q["mean_d_dwell"], 0.0, True)]
names = [n for n, *_ in clauses]; vals = [v for _, v, *_ in clauses]
cols = ["#27ae60" if (v >= thr) == good_dir or (good_dir and v > thr) else "#c0392b"
        for (_, v, thr, good_dir) in clauses]
# clause pass: contact Δ>=-0.05 (fail), exit -(Δ)>= -0.05 i.e Δexit<=0.05 (fail), dwell Δ>0 (pass)
passfail = [q["mean_d_contact_retention"] >= -0.05, (q["planner_exit_rate"] - q["pi0_exit_rate"]) <= 0.05, q["mean_d_dwell"] > 0]
cols = ["#27ae60" if p else "#c0392b" for p in passfail]
a2.bar(range(len(names)), vals, color=cols)
a2.set_xticks(range(len(names))); a2.set_xticklabels(names, fontsize=8.5)
a2.axhline(0, color="k", lw=0.6); a2.grid(axis="y", alpha=0.3)
a2.set_title(f"{d['verdict']}\n(green=clause pass, red=fail; teacher gains strict but breaks contact+exit)", fontsize=9)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{D}/baseline_reconstruction.{ext}", dpi=140)
print(f"wrote {D}/baseline_reconstruction.svg / .png")
