"""Figure for CHUNK_SUPERVISED_M1_FEEDBACK_V1 + mixed-teacher audit: (A) contact/exit/strict at M=1 vs M=2 vs pi_0;
(B) the audit's 'conditions present, symptom absent' summary. Reads the two result JSONs; no recompute."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
m1 = json.load(open(f"{D}/chunk_m1_diagnostic.json")); au = json.load(open(f"{D}/mixed_teacher_audit.json"))


def _cr(ev, k):
    return ev["chunk"][k], ev["pi0"][k]


fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
metrics = [("contact_retention", "contact"), ("exited", "target exit"), ("strict_success", "strict K6")]
x = range(len(metrics)); w = 0.26
m1c = [m1["eval_M1_vs_pi0"]["chunk"][k] for k, _ in metrics]
m2c = [m1["eval_M2_vs_pi0"]["chunk"][k] for k, _ in metrics]
p0 = [m1["eval_M1_vs_pi0"]["pi0"][k] for k, _ in metrics]
a.bar([i - w for i in x], m1c, w, label="chunk M=1", color="#c0392b")
a.bar(list(x), m2c, w, label="chunk M=2", color="#e67e22")
a.bar([i + w for i in x], p0, w, label="frozen pi_0", color="#2980b9")
a.set_xticks(list(x)); a.set_xticklabels([lbl for _, lbl in metrics]); a.set_ylabel("rate")
a.set_title("M=1 vs M=2 vs pi_0 (M=1 NO_GAIN: worse than M=2)"); a.legend(fontsize=8); a.grid(axis="y", alpha=0.3)

# Panel B: averaging conditions present vs symptom absent
present = [("teacher gap /8", au["teacher_gap"]["median"] / 8.0), ("kNN mode\ndisagree", au["nn_mode_disagreement_mean"]),
          ("err~mix\ncorr", au["error_vs_mode_disagreement_corr"])]
absent = [("learned\nbetween", au["between_teachers_frac"]),
          ("planner seg\n(1=on target)", au["planner_states_mean_segment_pos"]),
          ("pulled to\npi0", au["planner_states_pulled_toward_pi0_frac"])]
labels = [l for l, _ in present] + [l for l, _ in absent]
vals = [v for _, v in present] + [v for _, v in absent]
colors = ["#8e44ad"] * 3 + ["#27ae60"] * 3
b.bar(range(len(vals)), vals, color=colors)
b.set_xticks(range(len(labels))); b.set_xticklabels(labels, fontsize=7.5)
b.set_title(f"{au['verdict']}\n(purple: conditions for averaging present · green: symptom absent)", fontsize=9)
b.grid(axis="y", alpha=0.3); b.set_ylim(0, 1.0)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{D}/chunk_m1_audit.{ext}", dpi=140)
print(f"wrote {D}/chunk_m1_audit.svg / .png")
