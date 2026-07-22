"""GALAMBOS_PLANAR_DYNAMIC_PARITY (2026-07-22): after the golden-structure inertia repair, the canonical v3 robot
reproduces the golden ``make_planar_arms_mjcf`` contact-free EQUATIONS OF MOTION EXACTLY (bias + passive forces =
the mass/inertia/damping dynamics), is contact-STABLE (no NaN under sustained contact across embodiments), and its
short frozen-action rollouts track the legacy robot to tight tolerance. This gate is what the pre-repair 5×-heavy /
massless-contact-body designs failed (QACC NaN).

Note: v3 additionally EXCLUDES the golden's spurious adjacent-link self-contacts (the emitter's anti-joint-pinning
`<exclude>`), so raw ``qacc`` at self-colliding static poses differs by design; the contact-free forces (which encode
the inertia) are identical, and the operational rollout parity below confirms no behavioral divergence.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv, emit_galambos_v2_mjcf, make_planar_arms_mjcf

_V3 = "data/robotics/galambos_planar_v3.hymeko"
POSES = [[0.0, 0.0, 0.0, 0.0], [0.6, 0.6, -0.6, -0.6], [1.0, 0.8, -1.0, -0.4],
         [0.35, 0.13, 0.04, -0.56], [3.5, -3.5, 3.5, -3.5]]


def _arms():
    g = mujoco.MjModel.from_xml_string(make_planar_arms_mjcf())
    v = mujoco.MjModel.from_xml_string(emit_galambos_v2_mjcf(_V3))
    return (g, mujoco.MjData(g)), (v, mujoco.MjData(v))


def test_contact_free_eom_forces_match_golden_exactly():
    # qfrc_bias (Coriolis+gravity, from the mass/inertia) + qfrc_passive (damping) are the contact-free EoM. They must
    # match the golden body-for-body — this is the load-bearing inertia/dynamics parity.
    (g, dg), (v, dv) = _arms()
    worst_b = worst_p = 0.0
    for q in POSES:
        for m, d in ((g, dg), (v, dv)):
            d.qpos[:4] = q
            d.qvel[:4] = 0.3
            mujoco.mj_forward(m, d)
        worst_b = max(worst_b, float(np.max(np.abs(dg.qfrc_bias[:4] - dv.qfrc_bias[:4]))))
        worst_p = max(worst_p, float(np.max(np.abs(dg.qfrc_passive[:4] - dv.qfrc_passive[:4]))))
        assert np.all(np.isfinite(dv.qfrc_bias)) and np.all(np.isfinite(dv.qacc))
    assert worst_b < 1e-6, f"qfrc_bias parity {worst_b}"
    assert worst_p < 1e-6, f"qfrc_passive parity {worst_p}"


@pytest.mark.parametrize("embodiment", ["POINT", "CONCAVE_CLAMP", "FLAT_PAD"])
def test_contact_stability_stress(embodiment):
    # sustained randomised contact must never blow up (the massless-contact-body design NaN'd within ~10 steps).
    from hymeko_rl.coin_delivery.env_factory import make_coin_env
    env = make_coin_env(embodiment=embodiment)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        _o, _r, term, trunc, _i = env.step(rng.uniform(-2.0, 2.0, size=env.action_space.shape).astype(np.float32))
        assert np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)), "instability under contact"
        assert np.all(np.isfinite(env.data.qacc)), "QACC NaN under contact"
        if term or trunc:
            env.reset(seed=0)


def test_short_rollout_parity_vs_legacy():
    # identical frozen-action sequence on v3 vs the legacy robot → tight trajectory parity (no behavioral divergence).
    def roll(src):
        e = PlanarGraspEnv(robot_source=src, scene_source="hymeko_spec", max_steps=300, difficulty=0.3)
        e.reset(seed=0)
        rng = np.random.default_rng(1)
        qs = []
        for _ in range(60):
            e.step(rng.uniform(-1.0, 1.0, size=4).astype(np.float32))
            qs.append(e.data.qpos[:4].copy())
        return np.array(qs)
    worst = float(np.max(np.abs(roll("hymeko_spec") - roll("legacy_python"))))
    assert worst < 5e-2, f"short-rollout parity {worst} (tight-tolerance behavioral parity)"
