"""Figure for the critic authorization: within-state critic-Q vs realized-return correlation (centered per state),
per-family rank accuracy vs chance, and the twin-disagreement-vs-signal diagnostic."""
import json
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1])); out = sys.argv[2]
fams = ["transport", "entry", "settling", "contact_retention"]
colors = {"transport": "#1f77b4", "entry": "#2ca02c", "settling": "#ff7f0e", "contact_retention": "#d62728"}

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# A: centered-per-state critic Q vs realized return (the action signal)
a = ax[0]
for f in fams:
    for st in d["detail"].get(f, []):
        tr = np.array(st["true"]); q = np.array(st["qmin"])
        if len(tr) < 2:
            continue
        a.scatter(tr - tr.mean(), q - q.mean(), s=10, alpha=.4, color=colors[f])
a.axhline(0, color="k", lw=.5); a.axvline(0, color="k", lw=.5)
a.set_xlabel("realized return − state mean (ground truth)")
a.set_ylabel("min-Q − state mean (CENTERED, not abs twin)")
a.set_title("Within-state action signal (DEVELOPMENT diagnostic)\ncritic min-Q vs realized return, per-state centered")
a.grid(alpha=.3)

# B: per-family rank accuracy
b = ax[1]
accs = [d["per_family"][f]["mean_rank_acc"] if d["per_family"][f]["n_states"] > 0 else np.nan for f in fams]
ns = [d["per_family"][f]["n_states"] for f in fams]
bars = b.bar(range(4), accs, color=[colors[f] for f in fams])
b.axhline(0.5, color="red", ls="--", label="chance 0.5")
b.axhline(0.6, color="gray", ls=":", label="reliable threshold 0.6")
b.set_xticks(range(4)); b.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(fams, ns)], fontsize=8)
b.set_ylabel("mean pairwise rank accuracy"); b.set_ylim(0, 1)
b.set_title("Local ranking ≈ chance in every family (best contact_retention 0.57)")
b.legend(fontsize=8); b.grid(alpha=.3, axis="y")
for bar, ac in zip(bars, accs):
    if not np.isnan(ac):
        b.annotate(f"{ac:.3f}", (bar.get_x() + bar.get_width() / 2, ac + 0.02), ha="center", fontsize=8)

# C: diagnostic text
c = ax[2]; c.axis("off")
sw = d["boundary_sweep"]
txt = ("STATUS: PHASE_GATED_RESIDUAL_CRITIC_\n"
       "        AUTHORIZATION_BLOCKED\n"
       "(development diagnostic — NOT final audit)\n\n"
       "Blocking finding:\n"
       "  TARGET_SMOOTHING_CONTRACT_MISMATCH\n"
       "  critic trained with target smoothing\n"
       "  DISABLED (hardcoded zero noise) — not\n"
       "  the declared TD3 contract. Corrected.\n\n"
       "Panel is DEVELOPMENT (adaptively tuned:\n"
       "  LR/batch/steps/noise/capture) — cannot\n"
       "  support a fundamental negative.\n\n"
       f"Validated infra: encoder {d['encoder_fingerprint']}\n"
       f"  critic {d['critic_contract_sha']}; panels\n"
       "  train/dev/final DISJOINT; final SEALED.\n\n"
       "NOT claimed: 'signal below noise floor'\n"
       "  (that used absolute twin disagreement).\n"
       "  Requires centered dQ1/dQ2 + margin-aware\n"
       "  metrics + sealed final audit on the\n"
       "  CORRECTED (smoothed) critic.\n\n"
       "Actor update NOT authorized.")
c.text(0.02, 0.98, txt, transform=c.transAxes, fontsize=8.5, va="top", family="monospace",
       bbox=dict(boxstyle="round", fc="#fde2e2", ec="gray"))

fig.suptitle("PHASE_GATED_RESIDUAL_CRITIC_AUTHORIZATION_BLOCKED — development diagnostic (smoothing-disabled critic; "
             "superseded pending corrected re-audit)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=140)
print("wrote", out)
