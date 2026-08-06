import json
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1])); out = sys.argv[2]
real = d["real"]
seeds = list(real.keys())
or_prem = [real[s]["or_premature"] for s in seeds]
and_prem = [real[s]["and_premature"] for s in seeds]
and_transport = [real[s].get("and_active_transport", 0) for s in seeds]
x = np.arange(len(seeds)); w = 0.4
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

a = ax[0]
a.bar(x - w / 2, or_prem, w, label="OR-gate (deployed)", color="#d62728")
a.bar(x + w / 2, and_prem, w, label="BILATERAL candidate", color="#2ca02c")
a.set_xticks(x); a.set_xticklabels([s.replace("pi0_", "π₀ ").replace("cert_", "cert ") for s in seeds],
                                   rotation=45, ha="right", fontsize=8)
a.set_ylabel("premature unilateral acquisition arm events")
a.set_title("§7 PREMATURE_UNILATERAL_ACTIVATION: OR arms on single-finger brushes, BILATERAL eliminates it")
a.legend(fontsize=9); a.grid(alpha=.3, axis="y")
a.annotate(f"OR: {d['summary']['traj_with_premature_unilateral_acq']}/14 traj premature\nBILATERAL: 0/14",
           (0.5, 0.85), xycoords="axes fraction", fontsize=9,
           bbox=dict(boxstyle="round", fc="#fff3cd", ec="gray"))

b = ax[1]
colors = ["#1f77b4" if t > 0 else "#bbbbbb" for t in and_transport]
b.bar(x, and_transport, color=colors)
b.set_xticks(x); b.set_xticklabels([s.replace("pi0_", "π₀ ").replace("cert_", "cert ") for s in seeds],
                                   rotation=45, ha="right", fontsize=8)
b.set_ylabel("BILATERAL residual-active steps in transport")
b.set_title("Coverage gap: push-valid deliveries (1447, 6005) form NO bilateral contact → residual never arms")
b.grid(alpha=.3, axis="y")
for i, (s, t) in enumerate(zip(seeds, and_transport)):
    if t == 0 and ("1447" in s or "6005" in s):
        b.annotate("push\ndelivery", (i, 2), fontsize=7, color="#d62728", ha="center")

fig.suptitle("PHASE_GATE_PREMATURE_UNILATERAL_ACTIVATION — bilateral-arm refinement removes premature arming "
             "but under-covers push-valid deliveries", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=140)
print("wrote", out)
