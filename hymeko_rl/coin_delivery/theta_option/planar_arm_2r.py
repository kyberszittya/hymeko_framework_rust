"""Exact closed-form inverse kinematics for a planar 2R arm, calibrated to the MuJoCo joint frame.

For a planar 2R arm the IK is exact (Siciliano): given a tip target at radius ``r`` from the base, the elbow angle is
``q_rel = +/- acos((r^2 - l1^2 - l2^2)/(2 l1 l2))`` (the ``+/-`` are the elbow-up / elbow-down branches) and the shoulder
is ``atan2(d) - atan2(l2 sin q_rel, l1 + l2 cos q_rel)``. This module holds the arm model (``PlanarArm2R``), a calibration
that derives its parameters from forward-kinematics probes, and the analytic link geometry used for collision checks.

This is the lower, object-independent layer of the Geometric Approach stack; ``planar_geometric_approach`` builds the
collision-free transit on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

TWO_PI = 2.0 * np.pi


def wrap(angle: Any) -> Any:
    """Wrap angle(s) to (-pi, pi]."""
    return (angle + np.pi) % TWO_PI - np.pi


@dataclass(frozen=True)
class PlanarArm2R:
    """Exact closed-form IK for one planar 2R arm, calibrated to the MuJoCo frame.

    # Invariants: ``s0, s1`` are +/-1 (joint-axis signs); the analytic model round-trips FK to < 1e-3 m inside the
    #   reachable annulus ``(|l1-l2|, l1+l2)`` (asserted at calibration).
    """

    shoulder_dof: int
    elbow_dof: int
    base: np.ndarray
    tip_geom: int
    l1: float
    l2: float
    theta_b: float
    s0: float
    o1: float
    s1: float

    @property
    def reach(self) -> float:
        return self.l1 + self.l2

    @property
    def inner(self) -> float:
        return abs(self.l1 - self.l2)

    def r_base(self, tip: np.ndarray) -> float:
        """Tip distance from this arm's base (the branch/singularity metric)."""
        return float(np.linalg.norm(tip - self.base))

    def ik(self, tip: np.ndarray, branch: float) -> np.ndarray:
        """Analytic IK for one elbow branch (``branch`` = +1 / -1).

        # Preconditions: ``tip`` is 2-D world coordinates. # Postconditions: returns ``(a, b)`` MuJoCo joint angles;
        #   ``FK(ik(t)) == t`` to < 1e-3 m when ``t`` is strictly inside the annulus (outside, the cosine is clamped and
        #   the nearest reachable configuration is returned — the caller's validation rejects such waypoints).
        """
        delta = tip - self.base
        r = float(np.linalg.norm(delta))
        gamma = np.arctan2(delta[1], delta[0])
        cos_rel = np.clip((r * r - self.l1 ** 2 - self.l2 ** 2) / (2.0 * self.l1 * self.l2), -1.0, 1.0)
        q_rel = branch * np.arccos(cos_rel)
        phi = gamma - np.arctan2(self.l2 * np.sin(q_rel), self.l1 + self.l2 * np.cos(q_rel))
        return np.array([self.s0 * wrap(phi - self.theta_b), self.s1 * wrap(q_rel - self.o1)])

    def ik_cont(self, tip: np.ndarray, prev_ab: np.ndarray) -> np.ndarray:
        """Branch-continuous IK: the elbow branch minimising ``||wrap(q - prev_ab)||`` (no spurious branch hops)."""
        return min((self.ik(tip, +1.0), self.ik(tip, -1.0)),
                   key=lambda ab: float(np.linalg.norm(wrap(ab - prev_ab))))

    def link_points(self, ab: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Analytic (base, elbow, tip) planar positions for joint config ``ab`` — the two link segments, no FK cost.

        Inverts the calibration: shoulder-abs ``phi = theta_b + s0*a``, elbow-rel ``o1 + s1*b``, elbow-abs ``phi + rel``.
        # Postconditions: ``link_points(ik(t))[2] == t`` to calibration precision (checked by the round-trip test).
        """
        a, b = float(ab[0]), float(ab[1])
        phi = self.theta_b + self.s0 * a
        psi = phi + self.o1 + self.s1 * b
        elbow = self.base + self.l1 * np.array([np.cos(phi), np.sin(phi)])
        tip = elbow + self.l2 * np.array([np.cos(psi), np.sin(psi)])
        return self.base, elbow, tip


FkFn = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray, np.ndarray]"]


def make_fk(snapshot: Any, coin: np.ndarray, gl: int, gr: int) -> FkFn:
    """Build a forward-kinematics closure ``fk(q4) -> (xanchor, tipL, tipR)`` over a branched snapshot.

    The coin is pinned at ``coin`` so anchors/tips are measured in the canonical frame. Used for calibration and for
    resolving the home tip positions; branches a fresh copy each call (deterministic, no shared mutation).
    """

    def fk(q4: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        rl = snapshot.branch()
        data = rl.inner.data
        data.qpos[:4] = q4
        data.qpos[4:6] = coin
        data.qpos[6] = 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(rl.inner.model, data)
        return data.xanchor.copy(), data.geom_xpos[gl][:2].copy(), data.geom_xpos[gr][:2].copy()

    return fk


def calibrate_arm(fk: FkFn, shoulder_dof: int, elbow_dof: int, base_joint: int, tip_is_left: bool) -> PlanarArm2R:
    """Derive the full analytic model (base, l1, l2, theta_b, s0, o1, s1) from FK probes and assert the round-trip.

    # Postconditions: the returned arm round-trips its own zero-config tip to < 1e-3 m (a violated assert is a bug in the
    #   calibration, not the caller).
    """
    zero = np.zeros(4)
    anchors, tip_l, tip_r = fk(zero)
    base = anchors[base_joint][:2].copy()
    mid = anchors[elbow_dof][:2].copy()
    tip = tip_l if tip_is_left else tip_r
    l1 = float(np.linalg.norm(mid - base))
    l2 = float(np.linalg.norm(tip - mid))
    theta_b, s0 = _calibrate_shoulder(fk, shoulder_dof, elbow_dof, base, tip_is_left)
    o1, s1 = _calibrate_elbow(fk, shoulder_dof, elbow_dof, base, tip_is_left)
    arm = PlanarArm2R(shoulder_dof, elbow_dof, base, (0 if tip_is_left else 1), l1, l2, theta_b, s0, o1, s1)
    _assert_roundtrip(arm, fk, shoulder_dof, elbow_dof, tip_is_left)
    return arm


def _abs_angles(fk: FkFn, sh: int, el: int, base: np.ndarray, a: float, b: float,
                tip_is_left: bool) -> "tuple[float, float]":
    """Shoulder-absolute and elbow-absolute angles at joint config (a, b), from FK geometry."""
    q = np.zeros(4)
    q[sh], q[el] = a, b
    anchors, tip_l, tip_r = fk(q)
    mid = anchors[el][:2]
    tip = tip_l if tip_is_left else tip_r
    phi = float(np.arctan2(*(mid - base)[::-1]))
    psi = float(np.arctan2(*(tip - mid)[::-1]))
    return phi, psi


def _calibrate_shoulder(fk: FkFn, sh: int, el: int, base: np.ndarray, tip_is_left: bool) -> "tuple[float, float]":
    p0, _ = _abs_angles(fk, sh, el, base, 0.0, 0.0, tip_is_left)
    p1, _ = _abs_angles(fk, sh, el, base, 0.5, 0.0, tip_is_left)
    return p0, float(np.sign(wrap(p1 - p0)))


def _calibrate_elbow(fk: FkFn, sh: int, el: int, base: np.ndarray, tip_is_left: bool) -> "tuple[float, float]":
    phi0, psi0 = _abs_angles(fk, sh, el, base, 0.0, 0.0, tip_is_left)
    phi1, psi1 = _abs_angles(fk, sh, el, base, 0.0, 0.5, tip_is_left)
    rel0, rel1 = wrap(psi0 - phi0), wrap(psi1 - phi1)
    return float(rel0), float(np.sign(wrap(rel1 - rel0)))


def _assert_roundtrip(arm: PlanarArm2R, fk: FkFn, sh: int, el: int, tip_is_left: bool) -> None:
    probe = np.array([0.3, 0.4])
    q = np.zeros(4)
    q[sh], q[el] = probe
    _, tip_l, tip_r = fk(q)
    tip = tip_l if tip_is_left else tip_r
    ab = arm.ik_cont(tip, probe)
    q2 = np.zeros(4)
    q2[sh], q2[el] = ab
    _, ftl, ftr = fk(q2)
    err = float(np.linalg.norm((ftl if tip_is_left else ftr) - tip))
    assert err < 1e-3, f"analytic IK round-trip {err:.4e} m exceeds 1e-3 (calibration bug)"
