"""Collision-contract regression tests for the coin-delivery model (locks in the VERIFIED coin<->arm collision).

Context (2026-07-27): a coin<->arm collision concern was investigated. An initial matrix that placed geoms at COINCIDENT
centres reported cylinder-vs-capsule = 0 contacts and suggested a narrowphase bug — that was a DEGENERATE convex-collision
configuration (GJK/MPR has no separating axis at coincident centres), not a real failure. Re-tested with NON-coincident
shallow-penetration + swept tests, the coin collides with EVERY arm geom (capsules included) and cannot tunnel through
them. These tests encode that discriminating methodology so the (correct) behaviour is locked in and a future regression —
or a newly added arm link left non-collidable with the coin — is caught.

Methodology (per the collision-contract spec): force the tested pair into a KNOWN shallow overlap along a known outward
normal (never coincident), call mj_forward, and inspect data.contact by GEOM ID (not just contype/conaffinity); for
tunneling, sweep across the geom in small increments and assert contact occurs before the far side is reached.

The model is the exact coin-teacher scene (ball-tip BALLTIP arm + cylinder coin); it builds in <1 s (no acquisition).
"""
from __future__ import annotations

import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")

PEN = 0.001                                          # 1 mm shallow penetration
EXPECTED_ARM_GEOMS = {"base_left", "link1_left", "link2_left", "fingertip_left",
                      "base_right", "link1_right", "link2_right", "fingertip_right"}


def _build_model():
    from coin_object_o2 import _ball_tf
    from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
    rl = CoinRL4Dof(geom="POINT", arm_mjcf_transform=_ball_tf, coin_shape="cylinder", disk_radius_override=0.020)
    rl.reset(seed=0)
    return rl


@pytest.fixture(scope="module")
def coin_model():
    rl = _build_model()
    return rl


def _arm_geoms(m):
    """Every physical arm COLLISION geom, keyed by the label (geom name, else body name). Side from the body suffix."""
    out = {}
    for g in range(m.ngeom):
        if int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0:
            continue                                 # visual-only
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if bn in ("world",) or gn == "disk":
            continue
        label = gn if (gn and gn.startswith("fingertip")) else bn
        side = "L" if bn.endswith("_left") else ("R" if bn.endswith("_right") else "W")
        out[g] = {"label": label, "body": bn, "side": side, "type": int(m.geom_type[g]), "radius": float(m.geom_size[g][0])}
    return out


def _disk(m):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")


def _in_plane_normal(d, g):
    """A robust in-plane (xy) outward normal for approaching geom ``g`` with the planar coin. For a horizontal-axis
    capsule this is perpendicular to the axis; for a vertical-axis base cylinder (axis ~ z) it falls back to a fixed
    horizontal radial direction (the coincident-perp degeneracy the original test tripped on)."""
    center = d.geom_xpos[g][:3].copy()
    axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
    perp = np.array([-axis[1], axis[0], 0.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([1.0, 0.0, 0.0])
    return center, perp / np.linalg.norm(perp)


def _place_coin_xy(rl, xy):
    adr = rl.inner._disk_x_adr
    rl.inner.data.qpos[adr:adr + 2] = xy
    mujoco.mj_forward(rl.inner.model, rl.inner.data)


def _contact_dist(m, d, ga, gb):
    for i in range(d.ncon):
        c = d.contact[i]
        if {int(c.geom1), int(c.geom2)} == {ga, gb}:
            f6 = np.zeros(6, np.float64)
            mujoco.mj_contactForce(m, d, i, f6)
            return float(c.dist), float(abs(f6[0]))
    return None, None


# ───────────────────────────── coverage: no arm geom silently uncovered ─────────────────────────────
def test_arm_geom_inventory_is_complete_and_expected(coin_model):
    m = coin_model.inner.model
    labels = {v["label"] for v in _arm_geoms(m).values()}
    assert labels == EXPECTED_ARM_GEOMS, (labels ^ EXPECTED_ARM_GEOMS)   # a new/removed link trips this
    coin = _disk(m)
    assert int(m.geom_contype[coin]) and int(m.geom_conaffinity[coin])   # coin is a collision geom


# ───────────────────────────── shallow-penetration: coin collides with EVERY arm geom ─────────────────────────────
def test_coin_collides_with_every_arm_geom_shallow_penetration(coin_model):
    """The discriminating test: place the coin at a KNOWN 1 mm overlap along each arm geom's outward normal (never
    coincident), mj_forward, and assert the exact (disk, geom) pair is in data.contact with dist ~ -1 mm AND a nonzero
    separating (normal) force. Parametric over ALL arm geoms — coverage guaranteed by the inventory test."""
    m = coin_model.inner.model
    coin = _disk(m)
    coin_r = float(m.geom_size[coin][0])
    arm = _arm_geoms(m)
    misses = []
    for g, info in arm.items():
        rl = _build_model()
        m2, d2 = rl.inner.model, rl.inner.data
        center, normal = _in_plane_normal(d2, g)
        surf = info["radius"] + coin_r
        _place_coin_xy(rl, (center + normal * (surf - PEN))[:2])
        dist, fn = _contact_dist(m2, d2, coin, g)
        if dist is None or dist > -PEN * 0.5 or (fn is not None and fn <= 0.0):
            misses.append((info["label"], dist, fn))
    assert not misses, f"coin did NOT properly collide (contact+force) with: {misses}"


# ───────────────────────────── swept: coin cannot tunnel through the capsule links ─────────────────────────────
def test_coin_cannot_tunnel_through_capsule_links_swept(coin_model):
    """Sweep the coin across each CAPSULE link in 0.25 mm increments (mj_forward each step); assert a (disk,link) contact
    appears before the coin reaches the far side (no tunneling). Capsules have a well-defined perpendicular sweep axis."""
    m = coin_model.inner.model
    coin = _disk(m)
    coin_r = float(m.geom_size[coin][0])
    caps = [g for g, i in _arm_geoms(m).items() if i["type"] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)]
    assert len(caps) == 4                                            # link1/link2 both arms
    for g in caps:
        rl = _build_model()
        m2, d2 = rl.inner.model, rl.inner.data
        center, normal = _in_plane_normal(d2, g)
        surf = float(m2.geom_size[g][0]) + coin_r
        start, end = center + normal * (surf + 0.006), center - normal * (surf + 0.006)
        n = int(np.linalg.norm(end - start) / 0.00025) + 1
        contacted, tunneled = False, False
        for k in range(n + 1):
            p = start + (end - start) * k / n
            _place_coin_xy(rl, p[:2])
            dist, _fn = _contact_dist(m2, d2, coin, g)
            if dist is not None:
                contacted = True
            if float(np.dot((p - center), normal)) < -1e-4 and not contacted:
                tunneled = True                                     # reached far side without ever contacting
        assert contacted and not tunneled, f"tunneling through capsule geom {g} (contacted={contacted})"


# ───────────────────────────── cross-arm vs same-arm structure ─────────────────────────────
def _mask_collidable(m, a, b):
    ca, aa = int(m.geom_contype[a]), int(m.geom_conaffinity[a])
    cb, ab = int(m.geom_contype[b]), int(m.geom_conaffinity[b])
    return bool((ca & ab) or (cb & aa))


def _excluded_bodies(m):
    pairs = set()
    for e in range(m.nexclude):
        sig = int(m.exclude_signature[e])
        pairs.add(tuple(sorted((sig >> 16, sig & 0xFFFF))))
    return pairs


def test_cross_arm_pairs_are_collidable(coin_model):
    """Contract #2: left-arm and right-arm geoms DO collide — mask-collidable and NOT excluded (they simply do not
    overlap in the delivery trajectories, but the contact is enabled if they meet)."""
    m = coin_model.inner.model
    arm = _arm_geoms(m)
    left = [g for g, i in arm.items() if i["side"] == "L"]
    right = [g for g, i in arm.items() if i["side"] == "R"]
    excl = _excluded_bodies(m)
    for a in left:
        for b in right:
            assert _mask_collidable(m, a, b)
            bp = tuple(sorted((int(m.geom_bodyid[a]), int(m.geom_bodyid[b]))))
            assert bp not in excl                                   # no cross-arm exclude


def test_same_arm_pairs_are_mask_isolated(coin_model):
    """Contract #1 (per-side mask, not just adjacent excludes): EVERY same-arm geom pair — adjacent AND non-adjacent — is
    mask-isolated (not collidable), so no same-arm self-collision can occur in any pose or on any future morphology.
    Also: no same-arm contact appears at the home pose."""
    m, d = coin_model.inner.model, coin_model.inner.data
    arm = _arm_geoms(m)
    for side in ("L", "R"):
        gs = [g for g, i in arm.items() if i["side"] == side]
        for i, a in enumerate(gs):
            for b in gs[i + 1:]:
                assert not _mask_collidable(m, a, b), f"same-arm pair {arm[a]['label']}~{arm[b]['label']} is collidable"
    mujoco.mj_forward(m, d)
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if a in arm and b in arm and arm[a]["side"] == arm[b]["side"]:
            raise AssertionError(f"unexpected same-arm contact at home pose: {arm[a]['label']}~{arm[b]['label']}")


def test_collision_contract_exact_mask_values(coin_model):
    """Lock the exact per-side category masks: LEFT=1/14, RIGHT=2/13, COIN=4/11, WORLD/floor=8/7."""
    from hymeko_rl.env.collision_contract import role_masks
    m = coin_model.inner.model
    arm = _arm_geoms(m)
    for g, info in arm.items():
        want = role_masks("left" if info["side"] == "L" else "right")
        assert (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) == want, info["label"]
    coin = _disk(m)
    assert (int(m.geom_contype[coin]), int(m.geom_conaffinity[coin])) == role_masks("coin")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert (int(m.geom_contype[floor]), int(m.geom_conaffinity[floor])) == role_masks("world")
