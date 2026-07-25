"""Controller-INDEPENDENT contact-capability oracle. A state is CONTACT-CAPABLE iff a pure acquire-only policy (close the
fingertip→coin gap along the geometric acquire gradient — no delivery, no zone, no push) can establish contact within a
budget. This defines the evaluation subset a-priori, so comparing controllers on it is NOT post-treatment selection (we do
not pick states because the tested controller happened to touch the coin). Runs on the shared governed stack via
``step_ablation`` so it sees the real ``_planar_metrics`` contact state.
"""
from __future__ import annotations

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig, _acquire_direction
from hymeko_rl.env.governed_arm import V3Stack, pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque


def acquire_oracle(rl, stack: V3Stack, *, budget: int = 120, probe_mag: float = 2.0) -> dict:
    """Drive the tip toward the coin along the acquire gradient through the SHARED governed stack; report whether contact
    is established, on which step, and the min tip→coin distance reached. Delivery-agnostic and controller-independent."""
    cfg = CarryControllerConfig(probe_mag=probe_mag)
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gov, prev_tau = stack.gov, None

    def _gcb(_model, data):
        data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)
    mujoco.set_mjcb_control(_gcb)
    established, at_step, min_dist = False, -1, float("inf")
    try:
        for t in range(budget):
            mpl = rl.inner._planar_metrics
            min_dist = min(min_dist, float(min(mpl.left_tip_dist, mpl.right_tip_dist)))
            if mpl.left_contact or mpl.right_contact:
                established, at_step = True, t
                break
            delta = 0.10 * _acquire_direction(rl, cfg)
            q, qd = d.qpos[:4].copy(), d.qvel[:4].copy()
            a = pd_governed_torque(q, qd, q + delta, stack, prev_tau, lo, hi)
            prev_tau = a
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    return {"contact_capable": bool(established), "acquire_step": at_step, "min_tip_dist": round(min_dist, 4)}


def contact_capable_subset(reconstruct, stack: V3Stack, seeds, *, budget: int = 120) -> dict:
    """Run the acquire oracle over an EXPLICIT set of (state index → seed) and return the pre-registered contact-capable
    subset. ``reconstruct(seed) -> rl`` builds a fresh env for a seed. Returns {seed: oracle_result} + the capable list."""
    per = {}
    for si, seed in seeds:
        rl = reconstruct(seed)
        per[si] = {"seed": seed, **acquire_oracle(rl, stack, budget=budget)}
    capable = [si for si, r in per.items() if r["contact_capable"]]
    return {"per_state": per, "contact_capable_states": capable, "n_capable": len(capable), "n_total": len(seeds)}
