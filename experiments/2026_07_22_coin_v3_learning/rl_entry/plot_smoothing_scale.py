"""Figure: A/B/C residual target-smoothing distribution — boundary-hit bars + residual-action norm histograms."""
import json
import sys

import matplotlib
import numpy as np
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1])); out = sys.argv[2]
BOUND = 0.25
regimes = {"A_disabled": (0.0, 0.0, "#999999"), "B_unscaled_abs": (0.2, 0.5, "#d62728"),
           "C_scale_correct": (0.05, 0.125, "#2ca02c")}

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# left: any-dim residual-bound-hit probability
a = ax[0]
names = list(regimes)
vals = [d["regimes"][n]["any_dim_bound_hit"] for n in names]
bars = a.bar(range(3), vals, color=[regimes[n][2] for n in names])
a.set_xticks(range(3))
a.set_xticklabels([f"{n}\nstd={regimes[n][0]} clip={regimes[n][1]}" for n in names], fontsize=8)
a.set_ylabel("P(any of 4 residual dims saturates ±0.25)")
a.set_title("Residual-bound saturation at zero target residual\n(B saturates 61%; C never)")
a.grid(alpha=.3, axis="y")
for bar, v in zip(bars, vals):
    a.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v + 0.01), ha="center", fontsize=9)

# right: residual-action norm distributions
b = ax[1]
for n, (std, clip, col) in regimes.items():
    g = torch.Generator().manual_seed(1)
    eps = torch.clamp(torch.randn(200000, 4, generator=g) * std, -clip, clip) if std > 0 else torch.zeros(200000, 4)
    tr = torch.clamp(eps, -BOUND, BOUND)
    b.hist(tr.norm(dim=-1).numpy(), bins=60, histtype="step", lw=1.8, color=col, label=n, density=True)
b.axvline(BOUND, color="k", ls="--", lw=1, label="residual bound 0.25")
b.set_xlabel("target residual-action L2 norm at zero base-residual"); b.set_ylabel("density")
b.set_title("Residual-action norm: B piles at the bound, C stays well inside")
b.legend(fontsize=8); b.grid(alpha=.3)

fig.suptitle(f"RESIDUAL_TARGET_SMOOTHING_SCALE_CONTRACT_PASS — scale-relative smoothing (std=0.05, clip=0.125) "
             f"[contract {d['contract_sha']}]", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=140)
print("wrote", out)
