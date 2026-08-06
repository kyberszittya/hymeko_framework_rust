"""COIN-FINGERTIP-PAD-2 — geometry-valid parallel-jaw fingertip pad design + validity gate.

Answers: can a mirror-symmetric, inward-facing fingertip pad establish a persistent, actively-certified and
micro-transportable grasp on the CYLINDRICAL coin (kept cylindrical)? The load-bearing rule (§1): a valid pad's inner
face normal must satisfy n_L · ĉ > 0.95 (ĉ = the live left→right clamp axis) — an inward-facing surface, defined in the
fingertip LOCAL frame, NOT world coordinates. This module provides that GEOMETRY-VALIDITY gate + the clamp-axis / inward
measurement, so a naive world-aligned pad (invalid evidence) cannot enter the result table.

Key kinematic fact (measured): the planar 2-link Z-hinge arm has NO wrist DOF, so the rigid fingertip's local-frame
inward direction varies ~±36° across grasp poses — a FIXED flat-pad orientation cannot be inward-facing at more than a
sliver of poses. Orientation-ROBUST pads (tall-in-Z line contact / capsule) sidestep the orientation requirement.
Reuses coin_grasp_cert for the grasp evaluation. NO reward/CORE change; the fingertip_shape lever is additive
(default sphere = canonical).
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.train.coin_transport import restore_planar


def clamp_geometry(inner) -> dict:
    """Live clamp axis ĉ (left→right fingertip) + the inward directions (toward the coin) and their alignment with ĉ.
    n_L·ĉ and n_R·(−ĉ) near 1 mean a clean opposing pinch. All from the fingertip sites + coin body (world frame)."""
    import mujoco
    model, data = inner.model, inner.data
    tl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_left")
    tr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_right")
    cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "disk")
    pL, pR = data.site_xpos[tl][:2], data.site_xpos[tr][:2]
    coin = data.xpos[cb][:2]
    chat = (pR - pL) / (np.linalg.norm(pR - pL) + 1e-9)
    inL = (coin - pL) / (np.linalg.norm(coin - pL) + 1e-9)
    inR = (coin - pR) / (np.linalg.norm(coin - pR) + 1e-9)
    return {"clamp_axis": chat, "inward_L": inL, "inward_R": inR,
            "dotL": float(np.dot(inL, chat)), "dotR": float(np.dot(inR, -chat))}


def flat_pad_validity(env, handoffs) -> dict:
    """PAD-GEOMETRY-VALID for a fixed-orientation FLAT pad: measure the left fingertip's inward direction in its LOCAL
    (link2_left) frame across the grasp poses; a fixed inner-face normal can satisfy n·ĉ > 0.95 only if that local
    direction is consistent (low angular std). Returns the local-inward angular spread + the pass/fail."""
    import mujoco
    inner = env._env
    model, data = inner.model, inner.data
    l2l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2_left")
    tl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_left")
    cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "disk")
    angles, dots = [], []
    for h in handoffs:
        restore_planar(inner, h.snap)
        mujoco.mj_forward(model, data)
        pL = data.site_xpos[tl][:2]
        coin = data.xpos[cb][:2]
        rot = data.xmat[l2l].reshape(3, 3)[:2, :2]        # link2_left world rotation (xy block)
        inward_world = (coin - pL) / (np.linalg.norm(coin - pL) + 1e-9)
        inward_local = rot.T @ inward_world                # inward direction in the fingertip LOCAL frame
        angles.append(float(np.degrees(np.arctan2(inward_local[1], inward_local[0]))))
        dots.append(clamp_geometry(inner)["dotL"])
    std = float(np.std(angles)) if angles else 0.0
    return {"n": len(handoffs), "local_inward_angle_std_deg": round(std, 1),
            "local_inward_angle_range_deg": [round(min(angles), 1), round(max(angles), 1)] if angles else [],
            "mean_inward_dot_clamp_axis": round(float(np.mean(dots)), 3) if dots else 0.0,
            # a fixed flat pad can face inward (n·ĉ>0.95, ~18°) only if the local inward direction is consistent (<15° std)
            "fixed_flat_pad_geometry_valid": bool(std < 15.0)}
