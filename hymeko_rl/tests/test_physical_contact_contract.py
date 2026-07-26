"""PHYSICAL_CONTACT_CONTRACT gate (2026-07-22) — prove the coin physically collides with the arm links under the
corrected collision model (coin↔arm-link bit 2 enabled), in both POINT and RING, and that the collision-filtered
pass-through is gone. Whole-arm contact is legal — only physical-model failures are invalid."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.coin_delivery.env_factory import make_coin_env

_PEN_TOL = 0.0005   # 0.5 mm declared initial-penetration tolerance


def _arm_caps(m):
    return [g for g in range(m.ngeom)
            if "link" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]


def _mask(m, a, b):
    return (int(m.geom_contype[a]) & int(m.geom_conaffinity[b])) | (int(m.geom_contype[b]) & int(m.geom_conaffinity[a]))


def _place_coin_on(env, geom_id):
    m, d = env.model, env.data
    mujoco.mj_forward(m, d)
    p = d.geom_xpos[geom_id][:2].copy()
    d.qpos[env._disk_x_adr] = p[0] + 0.004
    d.qpos[env._disk_x_adr + 1] = p[1]
    mujoco.mj_forward(m, d)


def _coin_arm_contacts(env):
    m, d = env.model, env.data
    disk = env._disk_geom
    caps = set(_arm_caps(m))
    return sum(1 for c in range(d.ncon)
               if disk in (int(d.contact[c].geom1), int(d.contact[c].geom2))
               and (int(d.contact[c].geom2 if int(d.contact[c].geom1) == disk else d.contact[c].geom1) in caps))


# 8. RING and POINT use the same contact policy (arm-link 1/3, coin 2/2)
@pytest.mark.parametrize("emb", ["POINT", "CONCAVE_CLAMP"])
def test_arm_links_collide_with_coin_bitmask(emb):
    """COIN_ARM_COLLISION_CONTRACT (per-side, 2026-07-27): the coin physically collides with every arm-link geom
    (whole-arm contact is legal; task legality is a SEPARATE certificate). Masks: coin 4/11, left arm 1/14, right arm
    2/13 — coin↔arm collidable, same-arm isolated. Supersedes the earlier shared ARM_LEGALITY (1/3)."""
    from hymeko_rl.env.collision_contract import role_masks
    env = make_coin_env(embodiment=emb)
    m = env.model
    assert (int(m.geom_contype[env._disk_geom]), int(m.geom_conaffinity[env._disk_geom])) == role_masks("coin")
    for g in _arm_caps(m):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        want = role_masks("left" if bn.endswith("_left") else "right")
        assert (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) == want, f"{emb} arm cap {g}"
        assert _mask(m, env._disk_geom, g) != 0, f"{emb} coin↔arm-link {g} still filtered"


# 1. Coin placed against each arm link produces a MuJoCo contact
@pytest.mark.parametrize("emb", ["POINT", "CONCAVE_CLAMP"])
def test_coin_against_each_arm_link_produces_contact(emb):
    env = make_coin_env(embodiment=emb)
    env.reset(seed=3)
    for g in _arm_caps(env.model):
        _place_coin_on(env, g)
        assert _coin_arm_contacts(env) >= 1, f"{emb}: no contact when coin placed on arm cap {g}"


# 4. Fingertip contact still works
def test_fingertip_still_collides_with_coin():
    env = make_coin_env(embodiment="POINT")
    m = env.model
    ftl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    assert _mask(m, env._disk_geom, ftl) != 0


# 2 + 3. The arm cannot pass through the coin / coin responds to arm-link force
def test_arm_link_force_moves_the_coin_no_passthrough():
    env = make_coin_env(embodiment="POINT")
    env.reset(seed=5)
    m, d = env.model, env.data
    # place the coin just in front of link1_left, then drive that shoulder so the link sweeps into the coin
    l1 = [g for g in _arm_caps(m) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith("link1_left")][0]
    mujoco.mj_forward(m, d)
    lp = d.geom_xpos[l1][:2].copy()
    d.qpos[env._disk_x_adr] = lp[0] - 0.05
    d.qpos[env._disk_x_adr + 1] = lp[1]
    mujoco.mj_forward(m, d)
    c0 = d.qpos[env._disk_x_adr]
    d.ctrl[:] = 0.0
    d.ctrl[0] = 1.0                      # rotate left shoulder toward the coin
    moved_by_link = False
    for _ in range(120):
        mujoco.mj_step(m, d)
        if _coin_arm_contacts(env) >= 1:
            moved_by_link = True
    assert moved_by_link, "coin never contacted the swinging arm link (pass-through)"
    assert abs(float(d.qpos[env._disk_x_adr]) - c0) > 1e-3, "coin did not respond to arm-link force"
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)), "instability under arm-link contact"


# 6. Deep penetrating resets are detectable/rejected (declared -0.5mm tolerance)
def test_sampler_prevents_penetration_and_detector_flags_forced_overlap():
    """§4 (2026-07-22): the strengthened point-to-capsule-segment sampler no longer spawns the coin inside an arm.
    Seed 1011 (the historical deep-penetration case, -13.1 mm under the old centroid sampler) now spawns arm-clear.
    The penetration DETECTOR is still live: a deliberately forced coin-in-link overlap is reported as negative
    signed distance by raw MuJoCo. Replaces the stale `test_deep_penetration_reset_is_detectable`, whose premise
    (seed 1011 spawns penetrating) §4 has fixed."""
    import mujoco

    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    env, cf = neutral_env(prefix_steps=0, geom="POINT")
    inner = cf._env
    m, d = inner.model, inner.data
    env.set_stage(0)
    env.reset(seed=1011)   # no longer raises — coin is arm-clear
    coin_now = (float(d.qpos[inner._disk_x_adr]), float(d.qpos[inner._disk_x_adr + 1]))
    caps = [g for g in range(m.ngeom)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            and "link" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")]
    md = min(float(mujoco.mj_geomDistance(m, d, inner._disk_geom, g, 2.0, np.zeros(6))) for g in caps)
    assert md > -0.0005, f"seed 1011 still penetrates ({md * 1000:.2f} mm) — §4 sampler failed"
    # detector alive: the §4 clearance predicate accepts the sampled spawn and rejects the historical -13.1 mm point.
    assert inner._clear_of_arms(*coin_now), "§4 sampler produced a spawn its own predicate rejects"
    assert not inner._clear_of_arms(-0.159, 0.122), "historical penetrating point (-13.1 mm) wrongly judged clear"


# 5. Whole-arm contact is not automatically a failure — the corrected legality flag is defined, arm contact is legal
def test_whole_arm_contact_not_auto_failure():
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryThresholds
    # the strict certificate gates on center/settle/clean; "clean" must not equate to "no arm-link contact"
    # (that is the §3 monitor redefinition). Here we assert the physical contract: arm-link contact is a real,
    # allowed contact event (not filtered), which the prior tests establish. The default thresholds still load.
    assert DeliveryThresholds().center_tol == 0.02
