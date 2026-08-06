"""§ figures for RESIDUAL_HOLD_HORIZON_LEVERAGE_SWEEP_V1. Reads hold_sweep_v1_results.json →
reports/figures/2026-07-23-hold-sweep.png (leverage vs K, paired bootstrap, mechanism panels)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "experiments/2026_07_22_coin_v3_learning/rl_entry/hold_sweep_v1_results.json"
J = json.load(open(RES))
M = {int(k): v for k, v in J["metrics_by_K_family"].items()}
P = {int(k): v for k, v in J["paired_bootstrap_vs_K1"].items()}
KS = sorted(M)
FAM = ("transport", "entry", "settling", "contact_retention")
COL = {"transport": "#4C78A8", "entry": "#F58518", "settling": "#54A24B", "contact_retention": "#E45756"}


def series(fam, key):
    return [M[K][fam].get(key) if M[K][fam].get("n", 0) else np.nan for K in KS]


fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))
# 1: median |ΔG| vs K by family (the leverage curve)
a = ax[0, 0]
for f in FAM:
    a.plot(KS, series(f, "median_abs_dG"), marker="o", color=COL[f], label=f[:5])
a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS)
a.set_title("Leverage: median |ΔG| vs hold horizon K", fontsize=10); a.set_xlabel("K (hold steps)"); a.set_ylabel("median |ΔG|"); a.legend(fontsize=7)
# 2: paired bootstrap Δ(median leverage) vs K=1 with CIs
a = ax[0, 1]
mean = [P[K]["mean_paired_diff"] for K in KS]; lo = [P[K]["ci95"][0] for K in KS]; hi = [P[K]["ci95"][1] for K in KS]
a.errorbar(KS, mean, yerr=[np.array(mean) - np.array(lo), np.array(hi) - np.array(mean)], marker="s", capsize=4, color="#B279A2")
a.axhline(0, color="k", lw=.8); a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS)
a.set_title("Paired Δ(per-group median|ΔG|) vs K=1  (95% CI)", fontsize=10); a.set_xlabel("K"); a.set_ylabel("Δ leverage vs K=1")
# 3: fraction non-negligible |ΔG|>=5 vs K
a = ax[0, 2]
for f in FAM:
    a.plot(KS, series(f, "frac_nonneg_5"), marker="o", color=COL[f], label=f[:5])
a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS)
a.set_title("Fraction of candidates with |ΔG| ≥ 5", fontsize=10); a.set_xlabel("K"); a.set_ylabel("fraction"); a.legend(fontsize=7)
# 4: beneficial/neutral/harmful (pooled over families) vs K
a = ax[1, 0]
ben = [np.mean([M[K][f]["beneficial_frac"] for f in FAM if M[K][f].get("n")]) for K in KS]
neu = [np.mean([M[K][f]["neutral_frac"] for f in FAM if M[K][f].get("n")]) for K in KS]
har = [np.mean([M[K][f]["harmful_frac"] for f in FAM if M[K][f].get("n")]) for K in KS]
a.stackplot(KS, ben, neu, har, labels=["beneficial", "neutral", "harmful"], colors=["#54A24B", "#BAB0AC", "#E45756"], alpha=.85)
a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS)
a.set_title("Candidate outcome mix vs K (pooled)", fontsize=10); a.set_xlabel("K"); a.set_ylabel("fraction"); a.legend(fontsize=7, loc="upper left")
# 5: mechanism — contact break & target exit vs K (pooled)
a = ax[1, 1]
cb = [np.mean([M[K][f]["prob_contact_break"] for f in FAM if M[K][f].get("n")]) for K in KS]
te = [np.mean([M[K][f]["prob_target_exit"] for f in FAM if M[K][f].get("n")]) for K in KS]
a.plot(KS, cb, marker="o", label="P(contact break)", color="#E45756")
a.plot(KS, te, marker="s", label="P(target exit)", color="#F58518")
a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS)
a.set_title("Mechanism: destabilization vs K (pooled)", fontsize=10); a.set_xlabel("K"); a.set_ylabel("probability"); a.legend(fontsize=7)
# 6: verdict text
a = ax[1, 2]; a.axis("off")
lines = [f"VERDICT: {J['verdict']}", f"deterministic ×2: {J['deterministic_x2']}", "",
         "per-group median|ΔG| leverage vs K=1:"]
for K in KS:
    lines.append(f"  K={K:2d}: lev {P[K]['median_leverage']:.2f}  Δ {P[K]['mean_paired_diff']:+.2f}  ci95 {P[K]['ci95']}")
lines += ["", "state manifest sha " + J.get("state_manifest_sha", "?"), "candidate manifest sha " + J.get("candidate_manifest_sha", "?")]
a.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
fig.suptitle("RESIDUAL_HOLD_HORIZON_LEVERAGE_SWEEP_V1 — does the one-step residual gain more leverage when held K steps?", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("reports/figures/2026-07-23-hold-sweep.png", dpi=130)
print("wrote reports/figures/2026-07-23-hold-sweep.png  verdict", J["verdict"])
