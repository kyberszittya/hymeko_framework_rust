"""Figure for RESIDUAL_CRITIC_ROUTE_BLOCKED: advantage-critic generalization gap (robust to 6x data), per-family
actor-relevant gradient wins, and standard-critic centered corr across checkpoints."""
import json
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

std = json.load(open(sys.argv[1])); adv = json.load(open(sys.argv[2])); out = sys.argv[3]
fams = ["transport", "entry", "settling", "contact_retention"]
cols = {"transport": "#1f77b4", "entry": "#2ca02c", "settling": "#ff7f0e", "contact_retention": "#d62728"}

fig, ax = plt.subplots(1, 3, figsize=(15.5, 5))

# A: advantage-critic generalization gap (in-sample vs held-out), robust to data scale
a = ax[0]
sizes = ["288 pairs\n(~32 states)", "1854 pairs\n(206 states)"]
in_s = [0.996, adv["in_sample_corr"]]; held = [0.111, adv["held_out_corr"]]
x = np.arange(2); w = 0.35
a.bar(x - w / 2, in_s, w, label="in-sample corr(pred, ΔG)", color="#2ca02c")
a.bar(x + w / 2, held, w, label="held-out corr", color="#d62728")
a.axhline(0, color="k", lw=.5); a.axhline(0.3, color="gray", ls=":", label="usable threshold 0.3")
a.set_xticks(x); a.set_xticklabels(sizes, fontsize=8); a.set_ylabel("corr(predicted ΔG, empirical ΔG)")
a.set_title("Advantage critic: fits in-sample, does NOT generalize\n(6× data does not close the gap ⇒ not starvation)")
a.legend(fontsize=8); a.grid(alpha=.3, axis="y"); a.set_ylim(-0.2, 1.05)

# B: per-family actor-relevant +gradA wins (advantage critic, scaled)
b = ax[1]
gw = [adv["dev_metrics"].get(f, {}).get("gradQ1_wins", np.nan) for f in fams]
ns = [adv["dev_metrics"].get(f, {}).get("n", 0) for f in fams]
bars = b.bar(range(4), gw, color=[cols[f] for f in fams])
b.axhline(0.5, color="red", ls="--", label="chance"); b.axhline(0.55, color="gray", ls=":", label="pass 0.55")
b.set_xticks(range(4)); b.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(fams, ns)], fontsize=8)
b.set_ylabel("+gradA beats −gradA (empirical, held-out)"); b.set_ylim(0, 1)
b.set_title("Actor-relevant gradient test: TRANSPORT generalizes (0.9),\ncontact_retention does not (0.4)")
b.legend(fontsize=8); b.grid(alpha=.3, axis="y")
for bar, v in zip(bars, gw):
    if not np.isnan(v):
        b.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v + 0.02), ha="center", fontsize=8)

# C: standard critic centered corr Q1 vs dG across checkpoints (transport + contact)
c = ax[2]
ck = sorted(int(k) for k in std["standard_dev"])
for f in ("transport", "contact_retention"):
    ys = [std["standard_dev"][str(k)].get(f, {}).get("centered_corr_Q1_vs_dG", np.nan) for k in ck]
    c.plot(ck, ys, "o-", color=cols[f], label=f)
c.axhline(0, color="k", lw=.5); c.axhline(0.2, color="gray", ls=":", label="useful 0.2")
c.set_xscale("symlog"); c.set_xlabel("critic update"); c.set_ylabel("centered corr(ΔQ1, ΔG)")
c.set_title("Standard critic: centered Q1 signal never useful\n(dev FAILURE across {0..40k})")
c.legend(fontsize=8); c.grid(alpha=.3)

fig.suptitle("RESIDUAL_CRITIC_ROUTE_BLOCKED — the residual-marginal is learnable in-sample but does NOT generalize "
             "from the 48-dim obs (both standard & advantage critics)", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=140)
print("wrote", out)
