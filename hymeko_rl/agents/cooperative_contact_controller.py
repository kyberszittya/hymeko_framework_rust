"""Cooperative contact-formation controller (v12) — the JOINT two-fingertip action representation BELOW the option
abstraction that v11 showed the EntryPrepParams wrapper could not reach.

The policy emits a 6-d bounded COOPERATIVE action — [midpoint velocity (2), aperture velocity (1), squeeze (1),
left/right balance (1), approach-angle (1)] — which moves the two fingertip TARGETS together each step; the targets
are mapped to the 4-d position-servo command via the existing per-arm IK (`_ik_action` / `planar_2link_ik`). This is
low-level enough to shape contact formation, yet still a single cooperative controller over BOTH fingertips — NOT two
independent agents, NOT unshielded raw motor commands. All per-step deltas are bounded.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from hymeko_rl.env.planar_arm_kinematics import extract_arms as _extract_arms
from hymeko_rl.env.planar_arm_kinematics import ik_action as _ik_action

# per-step bounds (metres / radians) for each cooperative action component, applied to |action| in [0,1]
_STEP_MID = 0.012      # midpoint translation
_STEP_AP = 0.35        # aperture scale fraction (half-vector length × (1 ± this·a))
_STEP_SQ = 0.010       # squeeze of both tips toward the coin
_STEP_BAL = 0.008      # asymmetric midpoint shift along the fingertip axis
_STEP_ANG = 0.20       # rotation of the fingertip axis
ACTION_DIM = 6


def _rot2(v: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)


class CooperativeContactController:
    """Maps the 6-d cooperative action to the 4-d position-servo command. One controller over BOTH fingertips.

    # Preconditions ``env`` is a stepped PlanarGraspEnv (two planar arms). ``action`` is a length-6 vector; each
      component is clipped to [-1, 1] before scaling. # Postconditions returns a 4-d action clipped to the env's
      action bounds; invalid / non-finite actions fall back to holding the current fingertip positions (safety)."""

    def __init__(self, env: Any) -> None:
        self._arms = _extract_arms(env.model)

    def _tips(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        lft = np.asarray(env.data.site_xpos[self._arms["left"].tip_site][:2], np.float64)
        rft = np.asarray(env.data.site_xpos[self._arms["right"].tip_site][:2], np.float64)
        return lft, rft

    def targets(self, env: Any, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cooperative fingertip-target update (exposed for tests / imitation logging)."""
        a = np.clip(np.asarray(action, np.float64).reshape(ACTION_DIM), -1.0, 1.0)
        lft, rft = self._tips(env)
        mid = 0.5 * (lft + rft)
        axis = rft - lft
        u_axis = axis / (np.linalg.norm(axis) + 1e-9)
        coin = np.asarray(env._planar_metrics.disk_pos[:2], np.float64)
        new_mid = mid + _STEP_MID * a[:2] + _STEP_BAL * a[4] * u_axis           # translate + balance shift
        half = _rot2(0.5 * axis, _STEP_ANG * a[5]) * (1.0 + _STEP_AP * a[2])    # rotate + aperture
        new_lft, new_rft = new_mid - half, new_mid + half
        for tip in (new_lft, new_rft):                                          # squeeze both toward the coin
            d = coin - tip
            tip += _STEP_SQ * a[3] * d / (np.linalg.norm(d) + 1e-9)
        return new_lft, new_rft

    def action(self, env: Any, coop_action: np.ndarray) -> np.ndarray:
        a = np.asarray(coop_action, np.float64).reshape(-1)
        if a.size < ACTION_DIM or not np.all(np.isfinite(a)):
            lft, rft = self._tips(env)                                          # safe fallback: hold position
            return _ik_action(((self._arms["left"], lft), (self._arms["right"], rft)), env)
        new_lft, new_rft = self.targets(env, a)
        return _ik_action(((self._arms["left"], new_lft), (self._arms["right"], new_rft)), env)
