"""Figure for BENEFICIAL_SUPPORT_AUDIT_V1 → reports/figures/2026-07-23-beneficial-support-audit.png."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = sys.argv[1] if len(sys.argv) > 1 else "experiments/2026_07_22_coin_v3_learning/rl_entry/beneficial_support_audit_v1.json"
d = json.load(open(A))
KS = [1, 2, 4, 8, 16]; FAM = ("transport", "entry", "settling", "contact_retention")
COL = {"transport": "#4C78A8", "entry": "#F58518", "settling": "#54A24B", "contact_retention": "#E45756"}
BK = d["by_K_family"]; AK = d["across_K"]

fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
# 1: pooled beneficial (any) vs beneficial-contact-preserving vs good(safe+separable)
a = ax[0]
anyben = [np.mean([BK[str(K)][f]["item1_has_dG_gt1"]["frac"] for f in FAM]) for K in KS]
bcp = [AK["pooled_beneficial_contact_preserving"][str(K)] for K in KS]
good = [AK["pooled_good_candidate_support"][str(K)] for K in KS]
a.plot(KS, anyben, marker="o", label="≥1 beneficial (ΔG>1)", color="#4C78A8")
a.plot(KS, bcp, marker="s", label="beneficial + contact-preserving", color="#54A24B")
a.plot(KS, good, marker="^", label="safe + separable (usable)", color="#B279A2")
a.axhline(d["frozen"]["support_min"], color="k", ls=":", lw=.8, label=f"support_min {d['frozen']['support_min']}")
a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS); a.set_ylim(-0.02, 1.02)
a.set_title("Beneficial support collapses under safety", fontsize=10); a.set_xlabel("K"); a.set_ylabel("fraction of state groups"); a.legend(fontsize=7)
# 2: per-family beneficial-contact-preserving count
a = ax[1]
for f in FAM:
    a.plot(KS, [BK[str(K)][f]["item2_beneficial_contact_preserving"]["count"] for K in KS], marker="o", color=COL[f], label=f[:5])
a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS)
a.set_title("Beneficial contact-preserving groups (of 10)", fontsize=10); a.set_xlabel("K"); a.set_ylabel("count"); a.legend(fontsize=7); a.set_ylim(-0.2, 4)
# 3: median best NON-HARMFUL ΔG (the killer: 0 everywhere)
a = ax[2]
for f in FAM:
    a.plot(KS, [BK[str(K)][f]["item3_median_best_non_harmful_dG"] for f2 in [f] for K in KS], marker="o", color=COL[f], label=f[:5])
a.axhline(0, color="k", lw=.8)
a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS)
a.set_title("Median best NON-harmful ΔG (=0 ⇒ no safe gain)", fontsize=10); a.set_xlabel("K"); a.set_ylabel("median best safe ΔG"); a.legend(fontsize=7)
# 4: verdict text
a = ax[3]; a.axis("off")
lines = [f"STAGE A VERDICT:", f"  {d['verdict']}", "",
         "pooled 'only neutral/harmful' groups:",
         "  " + "  ".join(f"K{K}:{AK['pooled_only_neutral_or_harmful'][str(K)]:.2f}" for K in KS),
         "(drops with K ⇒ more beneficial candidates appear)", "",
         "but beneficial ⇒ break contact:",
         f"  usable (safe+separable) support = 0 at all K", "",
         f"eligible K: {d['K_selection']['eligible_K'] or 'NONE'}",
         f"selected K: {d['K_selection']['selected_K']}", "",
         "STAGE B (HARM_GATED_ADVANTAGE_CRITIC_V3):",
         "  NOT RUN — gate requires sufficient",
         "  beneficial support at an eligible K.", "",
         "The K-hold leverage increase is real but",
         "HARM-DOMINATED: holding a residual longer",
         "mostly destabilizes; safe improvement",
         "support is ~absent."]
a.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
fig.suptitle("BENEFICIAL_SUPPORT_AUDIT_V1 — is the K-hold leverage beneficial or merely harmful? (label-only)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("reports/figures/2026-07-23-beneficial-support-audit.png", dpi=130)
print("wrote reports/figures/2026-07-23-beneficial-support-audit.png verdict", d["verdict"])
