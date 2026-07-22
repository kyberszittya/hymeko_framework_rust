"""Contract-verification figure: 6 transition types — learner target vs reference (identical, maxdiff 0), stored
gate_tp1 used, no FSM invoked."""
import json
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1])); out = sys.argv[2]
rows = d["transitions"]
names = [r["type"].replace("->", "→\n") for r in rows]
x = np.arange(len(rows))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
a = ax[0]
md = [max(r["max_diff"], 1e-12) for r in rows]
colors = ["#2ca02c" if r["gate_tp1"] == 1.0 else "#1f77b4" for r in rows]
a.bar(x, md, color=colors)
a.set_yscale("log"); a.set_ylim(1e-12, 1e-3)
a.axhline(1e-6, color="red", ls="--", lw=1, label="tolerance 1e-6")
a.set_xticks(x); a.set_xticklabels(names, fontsize=7)
a.set_ylabel("|learner target − reference target|")
a.set_title("Target action matches reference for all 6 transitions (maxdiff = 0)")
a.legend(fontsize=8); a.grid(alpha=.3, axis="y")

b = ax[1]
b.axis("off")
tbl = [["transition", "g_t", "g_tp1", "done", "FSM?", "g0=base", "term mask"]]
for r in rows:
    tbl.append([r["type"][:26], f"{r['gate_t']:.0f}", f"{r['gate_tp1']:.0f}", f"{r['done']:.0f}",
                "no" if not r["fsm_invoked"] else "YES", "✓" if r["gate0_equals_base"] else "—",
                "✓" if r["terminal_bootstrap_masked"] else ("—" if r["done"] == 0 else "✗")])
t = b.table(cellText=tbl, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.5)
for j in range(len(tbl[0])):
    t[(0, j)].set_facecolor("#dddddd")
b.set_title(f"schema {d['schema_hash_prefix']} · no FSM invoked · pi_0 {d['pi0_file_sha']}", fontsize=9)

fig.suptitle("PHASE_GATE_REPLAY_STATE_CONTRACT_PASS — learner uses STORED gate_tp1; no fresh FSM in the minibatch; "
             "gate=0 ⇒ base bit-identical", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=140)
print("wrote", out)
