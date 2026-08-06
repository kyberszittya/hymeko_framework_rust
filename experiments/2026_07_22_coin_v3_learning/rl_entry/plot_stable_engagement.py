"""Figure for PHASE_GATE_STABLE_ENGAGEMENT_PASS: hybrid gate timelines (unilateral-push 1447, grasp-after-push 1011,
correctly-rejected 1358) + OR-vs-hybrid premature-activation summary."""
import json
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_stable_engagement import (  # noqa: E402
    StableEngagementConfig,
    StableEngagementGate,
    stable_engagement_signals,
)
from hymeko_rl.coin_delivery.rl_clip_actor import ActorEvalWrap, load_frozen_clip_actor  # noqa: E402
from hymeko_rl.experiments.coin_neutral_start import neutral_env  # noqa: E402

PI0 = sys.argv[1]; AUDIT = sys.argv[2]; OUT = sys.argv[3]


def roll(actor, seed, horizon=360):
    env, cf = neutral_env(prefix_steps=0); inner = cf._env
    env.set_stage(0); env.reset(seed=int(seed))
    gate = StableEngagementGate(StableEngagementConfig()); ew = ActorEvalWrap(actor)
    rows = []
    c0 = None
    for t in range(horizon):
        nf = np.asarray(inner.node_features(), np.float32).flatten()
        inner.step(np.clip(ew.act(nf), -4, 4).astype(np.float32))
        lc, rc, coin, ltip, rtip = stable_engagement_signals(inner)
        if c0 is None:
            c0 = coin.copy()
        g, mech = gate.update(lc, rc, coin, ltip, rtip)
        rows.append((lc, rc, float(np.linalg.norm(coin - c0)), g, mech, gate.s.mode))
    return rows


d = json.load(open(AUDIT))
pi0 = load_frozen_clip_actor(PI0, freeze=True)
reps = [(1447, "seed 1447 — unilateral PUSH delivery (never bilateral) → UNILATERAL_COMOTION, no target info"),
        (1011, "seed 1011 — early co-moving push then bilateral grasp → armed, held through transport"),
        (1358, "seed 1358 — sustained but UNCONTROLLED contact (coin ⊥ tip) → correctly NOT armed")]
fig, ax = plt.subplots(3, 1, figsize=(12, 8.4))
for a, (seed, title) in zip(ax, reps):
    rows = roll(pi0, seed)
    t = np.arange(len(rows))
    lc = np.array([r[0] for r in rows]); rc = np.array([r[1] for r in rows])
    disp = np.array([r[2] for r in rows]); g = np.array([r[3] for r in rows])
    for i in range(len(rows)):
        if g[i] > 0:
            a.axvspan(i - 0.5, i + 0.5, color="#2ca02c", alpha=0.16, lw=0)
    a.plot(t, disp, color="#1f77b4", lw=1.3, label="coin displacement (m)")
    mx = max(disp.max(), 1e-3)
    a.fill_between(t, 0, lc * mx * 0.10, step="mid", color="#d62728", alpha=.5, label="left contact")
    a.fill_between(t, -mx * 0.10, (rc * mx * 0.10) - mx * 0.10, step="mid", color="#ff7f0e", alpha=.5, label="right contact")
    fa = next((i for i, r in enumerate(rows) if r[3] > 0), None)
    if fa is not None:
        a.axvline(fa, color="#2ca02c", lw=1.5)
        a.annotate(f"ARM: {rows[fa][4]}", (fa, mx * 0.6), fontsize=8, color="#2ca02c", rotation=90, va="top")
    a.set_title(title, fontsize=9.5); a.set_ylabel("m"); a.grid(alpha=.25); a.set_xlim(0, len(rows))
ax[-1].set_xlabel("timestep")
handles = [Patch(color="#2ca02c", alpha=.3, label="residual gate ARMED"),
           Patch(color="#d62728", alpha=.5, label="left contact"),
           Patch(color="#ff7f0e", alpha=.5, label="right contact"),
           plt.Line2D([], [], color="#1f77b4", label="coin displacement")]
fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=.9)
c = d["checks"]
sub = (f"STABLE_OBJECT_ENGAGEMENT_V1 (SHA {d['gate_contract_v2_sha'][:12]}): approach/brush activation 0/14 "
       f"(rejected OR gate: 11/14 premature). All 3 deliveries armed; 1447 push via co-motion w/o target; "
       f"BILATERAL_FAST + UNILATERAL_COMOTION both fire.")
fig.suptitle("PHASE_GATE_STABLE_ENGAGEMENT_PASS — hybrid bilateral + unilateral-co-motion gate\n" + sub, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
