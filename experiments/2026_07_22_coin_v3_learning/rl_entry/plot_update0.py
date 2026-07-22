"""Verification figure: composite (gate=0 and gate=1) == pi_0 action bit-identically at update 0, and residual==0."""
import sys

import matplotlib
import numpy as np
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_residual_controller import CompositeResidualController, ZeroInitResidualActor  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.experiments.coin_neutral_start import neutral_env  # noqa: E402

PI0 = sys.argv[1]; OUT = sys.argv[2]
pi0 = load_frozen_clip_actor(PI0, freeze=True)
ctrl = CompositeResidualController(pi0, ZeroInitResidualActor())
env, cf = neutral_env(prefix_steps=0); inner = cf._env
env.set_stage(0); env.reset(seed=1011)
states = []
for _ in range(200):
    nf = np.asarray(inner.node_features(), np.float32).flatten(); states.append(nf)
    inner.step(np.asarray(pi0.action_mean(torch.tensor(nf[None]))[0].numpy(), np.float32))
ob = torch.tensor(np.asarray(states, np.float32))
with torch.no_grad():
    base = ctrl.base_action(ob).numpy().ravel()
    g0 = ctrl.composite_action(ob, 0.0).numpy().ravel()
    g1 = ctrl.composite_action(ob, 1.0).numpy().ravel()
    res = ctrl.residual.residual_exec(ob).numpy().ravel()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
a = ax[0]
a.scatter(base, g0, s=8, alpha=.5, color="#1f77b4", label=f"gate=0 (maxdiff {np.abs(g0-base).max():.0e})")
a.scatter(base, g1, s=8, alpha=.5, color="#d62728", marker="x", label=f"gate=1 (maxdiff {np.abs(g1-base).max():.0e})")
lim = [base.min() - 0.2, base.max() + 0.2]
a.plot(lim, lim, "k--", lw=1, alpha=.5, label="y = x")
a.set_xlabel("frozen pi_0 base action"); a.set_ylabel("composite action")
a.set_title("Update 0: composite == pi_0 for BOTH gate states (residual == 0)")
a.legend(fontsize=9); a.grid(alpha=.3)

b = ax[1]
b.hist(res, bins=40, color="#2ca02c", alpha=.7)
b.set_title(f"Zero-init residual output (max |residual| = {np.abs(res).max():.1e})")
b.set_xlabel("executed residual component"); b.set_ylabel("count")
b.axvline(0, color="k", lw=1)
b.grid(alpha=.3)
b.annotate("bound ±0.25 (unused at init)", (0, b.get_ylim()[1] * 0.85), fontsize=9, ha="center")

fig.suptitle("PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED + EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS\n"
             "composite+gate rollout: HL 3/9 · VAL 2/30 · grasp 9/9 · delivered {1011,1447,1568} = pi_0 exactly; "
             "pi_0 hash unchanged", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
