"""Timeline figure for PHASE_GATE_RUNTIME_CONTRACT: contact + gate-active + dtz over time for representative
trajectories, showing gate OFF during approach, ARM at first stable transport contact, REACQUIRE on contact loss."""
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, ".")
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "pv", "/private/tmp/claude-501/-Users-kyberszittya-hakiko-ai-ws-03-implementation-hymeko-framework-rust/63ad1b54-314a-48f8-b561-ba4a163f847c/scratchpad/coin_phase_gate_validation.py")
# set argv so the module-level BASE/PI0_CKPT resolve
BASE = sys.argv[1]; PI0 = sys.argv[2]; OUT = sys.argv[3]
sys.argv = ["x", BASE, "/dev/null", PI0]
pv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pv)

_b, pi0, sha = pv.load_pi0()
reps = [("seed 1011 — clean K6 delivery", 1011, None),
        ("seed 1447 — bouncy transport (5 reacquisitions)", 1447, None),
        ("seed 6000 — certified handoff K6 delivery", 6000, "cert")]
fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=False)
for ax, (title, seed, kind) in zip(axes, reps):
    if kind == "cert":
        acts = np.load(BASE + f"/traj_{seed}.npz")["act"]
        tr, fc, g = pv.roll_gate(None, seed, replay_actions=acts)
    else:
        tr, fc, g = pv.roll_gate(pi0, seed)
    t = [s["t"] for s in tr]
    dtz = [s["dtz"] for s in tr]
    contact = [1 if s["contact"] else 0 for s in tr]
    armed = [1 if s["state"] == "LATE_CONTROL_ARMED" else 0 for s in tr]
    # shade armed intervals
    for i, s in enumerate(tr):
        if armed[i]:
            ax.axvspan(t[i] - 0.5, t[i] + 0.5, color="#2ca02c", alpha=0.16, lw=0)
    ax.plot(t, dtz, color="#1f77b4", lw=1.4, label="disk→zone (m)")
    ax.axhline(0.02, color="#1f77b4", ls=":", lw=1, alpha=.6)
    ax.fill_between(t, 0, [c * max(dtz) * 0.12 for c in contact], step="mid", color="#d62728",
                    alpha=.5, label="robot contact")
    first_act = next((s["t"] for s in tr if s["g_next"] == 1.0), None)
    if fc >= 0:
        ax.axvline(fc, color="#d62728", ls="--", lw=1.2)
        ax.annotate("first contact", (fc, max(dtz) * 0.92), fontsize=8, color="#d62728", rotation=90, va="top")
    if first_act is not None:
        ax.axvline(first_act, color="#2ca02c", ls="-", lw=1.4)
        ax.annotate("ARM (transport)", (first_act, max(dtz) * 0.5), fontsize=8, color="#2ca02c", rotation=90, va="top")
    ax.set_title(title, fontsize=10); ax.set_ylabel("dtz (m)"); ax.grid(alpha=.25)
    ax.set_xlim(0, len(tr)); ax.set_ylim(0, max(dtz) * 1.05)
axes[-1].set_xlabel("timestep")
handles = [Patch(color="#2ca02c", alpha=.3, label="residual gate ARMED (g=1)"),
           Patch(color="#d62728", alpha=.5, label="robot-attributed contact"),
           plt.Line2D([], [], color="#1f77b4", label="disk→zone distance")]
fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=.9)
fig.suptitle("PHASE_GATE_RUNTIME_CONTRACT_PASS — gate OFF in approach, ARMs at first stable transport contact, "
             "REACQUIREs on loss\n(deployable tactile signal only; pi_0 file-SHA 1902454c; arm_after=3, disarm_after=2)",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
