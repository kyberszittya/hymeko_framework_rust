"""Figure for STRICT_COUNTER_MARKOV_REPAIR_ABLATION_V1: (A) strict-conditioned Q rises toward the terminal in all runs —
the Markov critic now represents terminal proximity (the repair works at the critic level); (B) K6 unchanged vs pi_0 and
the transactional actor rejects 99% of updates — the binding constraint moved to the actor, not the critic."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
d = json.load(open(f"{D}/markov_ablation_v1.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.9))

for arm, col in (("A", "#c0392b"), ("B", "#e67e22")):
    for s in range(3):
        ck = d["runs"][f"{arm}_{s}"]["checkpoints"]; q = ck[max(ck, key=lambda k: int(k))]["strict_q"]
        a1.plot(range(7), q, "-o", color=col, alpha=0.7, label=f"Arm {arm}" if s == 0 else None)
a1.axvline(5, color="grey", ls="--", lw=0.8); a1.annotate("terminal K6\n(+30 fires)", (5, a1.get_ylim()[0]), fontsize=8, ha="center")
a1.set_xlabel("strict counter (distance-to-terminal)"); a1.set_ylabel("min-twin Q(state, actor)")
a1.set_title("Markov critic REPRESENTS terminal proximity\n(Q rises toward K5/K6 — hidden-counter critic could not)", fontsize=9.5)
a1.legend(fontsize=9); a1.grid(alpha=0.3)

pi0 = d["pi0_baseline"]["CONTINUATION"]["k6_rate"]
groups = ["pi_0", "Arm A", "Arm B"]
k6 = [pi0, np.mean(d["final_k6_continuation"]["A"]), np.mean(d["final_k6_continuation"]["B"])]
acc = [0] + [np.mean([d["runs"][f"{a}_{s}"]["accepted"] / (d["runs"][f"{a}_{s}"]["accepted"] + d["runs"][f"{a}_{s}"]["rejected"]) for s in range(3)]) for a in ("A", "B")]
x = np.arange(3); w = 0.35
a2.bar(x - w / 2, k6, w, label="CONTINUATION K6 rate", color="#2980b9")
a2.bar(x + w / 2, acc, w, label="transactional accept rate", color="#7f8c8d")
a2.axhline(pi0, color="#2980b9", ls="--", lw=0.8, alpha=0.6)
a2.set_xticks(x); a2.set_xticklabels(groups); a2.set_ylim(0, 0.5); a2.legend(fontsize=8.5); a2.grid(axis="y", alpha=0.3)
a2.set_title(f"{d['verdict']}\nK6 unchanged (0.417); actor rejects 99% of updates\n→ bottleneck moved to the actor, not the critic", fontsize=9.5)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{D}/markov_ablation.{ext}", dpi=140)
print(f"wrote {D}/markov_ablation.svg / .png")
