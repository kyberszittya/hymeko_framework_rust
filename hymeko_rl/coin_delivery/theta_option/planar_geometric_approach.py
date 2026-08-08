"""Object-generic Geometric Approach layer — explicit closed-form planar 2R IK transit HOME -> READY.

For a planar 2R arm the inverse kinematics is exact (Siciliano): given a tip target ``t`` relative to the arm base at
radius ``r``, the elbow angle is ``q_rel = +/- acos((r^2 - l1^2 - l2^2)/(2 l1 l2))`` (the ``+/-`` are the elbow-up /
elbow-down branches) and the shoulder is ``atan2(d) - atan2(l2 sin q_rel, l1 + l2 cos q_rel)``. This module

  1. inverts each arm analytically, calibrated to the MuJoCo joint frame (``theta_b, s0, o1, s1`` measured from FK);
  2. routes each fingertip around the object along a *validated collision-free* tip-space path (radial-out -> keep-out
     arc -> radial-in to the assigned side), always selecting the IK branch continuous with the previous configuration
     (``argmin ||wrap(q - q_prev)||``) so a single elbow branch is held throughout;
  3. time-parameterises the reference so every step respects the frozen slew cap, then tracks it through the *same*
     governed-torque servo the rest of the system uses (no new torque authority, no state edit, no hidden force).

Learning is reserved for what kinematics cannot cover — the contact-rich dynamic capture and the R2 transport, both
downstream and untouched. The object-geometry seam (``ApproachTargets``) makes the pre-contact tip targets swappable per
object; ``CoinStraddleTargets`` is the coin instance.

The transit is fully deterministic (no RNG): the candidate arc radii/directions and waypoint count are fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, _fingertip_geoms, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option.planar_arm_2r import (
    FkFn, PlanarArm2R, TWO_PI, calibrate_arm, make_fk, wrap)
from hymeko_rl.env.motion_contract import govern_torque


class TransitInfeasible(RuntimeError):
    """Raised when no collision-free branch-continuous path routes an arm to its pre-contact target.

    Carries the failing reason so callers never mistake a silent truncation for success.
    """


# --------------------------------------------------------------------------------------------------------------------
# Object-geometry seam
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PrecontactTargets:
    """Pre-contact tip targets and the assigned side angle (around the object) for each arm."""

    tip_left: np.ndarray
    tip_right: np.ndarray
    side_left_rad: float
    side_right_rad: float


class ApproachTargets(Protocol):
    """The object-geometry seam: turn an object + approach choice into per-arm pre-contact tip targets.

    A pyramid/prism instance replaces *only* this — the analytic IK, path routing, validation, and servo are generic.
    """

    def precontact(self) -> PrecontactTargets: ...


# Straddle standoff geometry (R11.7A): shell_dist = object footprint + fingertip radius + surface margin. For the
# reference coin (footprint 0.02) this reproduces the historical 0.055.
_FINGERTIP_R = 0.02
_SURFACE_MARGIN = 0.015


@dataclass(frozen=True)
class CoinStraddleTargets:
    """Coin instance of ``ApproachTargets``: two tips at the shell radius on opposite assigned sides of the coin.

    ``side_left_deg`` / ``side_right_deg`` are the tip angles around the coin centre at READY (the pre-contact
    straddle); ``shell_dist`` is the tip-to-coin-centre distance there.
    """

    coin: np.ndarray
    shell_dist: float = 0.055           # 55 mm tip-centre distance: 15 mm surface margin (fingertip r=coin r=0.02)
    side_left_deg: float = 107.0
    side_right_deg: float = -118.0

    @classmethod
    def for_object(cls, coin: np.ndarray, footprint_radius: float, *, fingertip_radius: float = _FINGERTIP_R,
                   margin: float = _SURFACE_MARGIN, side_left_deg: float = 107.0,
                   side_right_deg: float = -118.0) -> "CoinStraddleTargets":
        """Straddle targets sized to the object footprint (R11.7A): the pre-contact tips sit at
        ``footprint_radius + fingertip_radius + margin`` from the object centre, so a larger disk or a box
        gets a proportionally wider straddle instead of the coin's fixed 0.055.

        # Preconditions ``footprint_radius > 0`` (use :meth:`ObjectSpec.footprint_radius`).
        # Postconditions for the reference coin (``footprint_radius=0.02``) the shell_dist is exactly 0.055,
          so O0's reach geometry is unchanged.
        """
        assert footprint_radius > 0.0, f"footprint_radius must be > 0, got {footprint_radius}"
        return cls(coin=coin, shell_dist=footprint_radius + fingertip_radius + margin,
                   side_left_deg=side_left_deg, side_right_deg=side_right_deg)

    def rotated(self, yaw_deg: float) -> "CoinStraddleTargets":
        """A copy with the straddle axis rotated by ``yaw_deg`` — both assigned-side angles shifted equally, so the tips
        approach a yaw-rotated object along its rotated faces (R12.2-A': the axis-aligned straddle caps a rotated box at
        ~30° certified; rotating the straddle with the object lifts that). # Postconditions: ``yaw_deg=0`` returns an
        equal straddle, so the frozen yaw=0 acquisition is unchanged."""
        return CoinStraddleTargets(coin=self.coin, shell_dist=self.shell_dist,
                                   side_left_deg=self.side_left_deg + yaw_deg,
                                   side_right_deg=self.side_right_deg + yaw_deg)

    def precontact(self) -> PrecontactTargets:
        al, ar = np.radians(self.side_left_deg), np.radians(self.side_right_deg)
        tip_l = self.coin + self.shell_dist * np.array([np.cos(al), np.sin(al)])
        tip_r = self.coin + self.shell_dist * np.array([np.cos(ar), np.sin(ar)])
        return PrecontactTargets(tip_l, tip_r, al, ar)


# --------------------------------------------------------------------------------------------------------------------
# Collision-free path planning
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TransitConfig:
    """Geometric + servo parameters for the transit. Deterministic; no RNG."""

    keepout: float = 0.045          # min tip-centre -> coin-centre distance (fingertip r + coin r = 0.04 contact; +5 mm)
    link_keepout: float = 0.033     # min link-segment -> coin-centre distance (link capsule r ~0.012 + coin r + margin)
    r_base_max: float = 0.298       # non-singular ceiling (reach is 0.30)
    r_base_min: float = 0.03        # inner-annulus floor
    arc_radii: "tuple[float, ...]" = (0.055, 0.06, 0.058, 0.052)  # candidate arc radii (coin-centre distance; = shell)
    n_waypoints: int = 150
    substeps: int = 3
    servo_kp: float = 12.0
    servo_kd: float = 4.0
    hold_steps: int = 25
    joint_lo: float = -4.0
    joint_hi: float = 4.0


def _polar(point: np.ndarray, coin: np.ndarray) -> "tuple[float, float]":
    delta = point - coin
    return float(np.linalg.norm(delta)), float(np.arctan2(delta[1], delta[0]))


def _arc_waypoints(start: np.ndarray, goal: np.ndarray, coin: np.ndarray, rho: float, direction: float,
                   n: int) -> np.ndarray:
    """Tip-space path in coin polar coordinates: radial-out to ``rho`` -> arc (``direction``) -> radial-in to goal."""
    r0, a0 = _polar(start, coin)
    r1, a1 = _polar(goal, coin)
    d_ang = wrap(a1 - a0)
    if direction > 0 and d_ang < 0:
        d_ang += TWO_PI
    if direction < 0 and d_ang > 0:
        d_ang -= TWO_PI
    third = n // 3
    seg = [(r0 + f * (rho - r0), a0) for f in np.linspace(0, 1, third)]
    seg += [(rho, a0 + f * d_ang) for f in np.linspace(0, 1, third)]
    seg += [(rho + f * (r1 - rho), a1) for f in np.linspace(0, 1, n - 2 * third)]
    return np.array([coin + r * np.array([np.cos(a), np.sin(a)]) for r, a in seg])


def _seg_point_dist(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Min distance from point ``p`` to segment ``ab`` (for link-capsule -> coin clearance)."""
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (float(np.dot(ab, ab)) + 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _tip_feasible(path: np.ndarray, coin: np.ndarray, arm: PlanarArm2R, cfg: TransitConfig) -> bool:
    """Tip-space feasibility: coin keep-out and non-singular in-annulus at every waypoint."""
    clr = np.linalg.norm(path - coin, axis=1)
    rbase = np.linalg.norm(path - arm.base, axis=1)
    return bool(clr.min() >= cfg.keepout and rbase.max() < cfg.r_base_max and rbase.min() > cfg.r_base_min)


def _links_clear(traj: np.ndarray, arm: PlanarArm2R, coin: np.ndarray, cfg: TransitConfig) -> bool:
    """Both link segments (base->elbow, elbow->tip) stay clear of the coin at every joint config (analytic, no FK)."""
    for ab in traj:
        base, elbow, tip = arm.link_points(ab)
        if min(_seg_point_dist(base, elbow, coin), _seg_point_dist(elbow, tip, coin)) < cfg.link_keepout:
            return False
    return True


def _solve_arm_traj(path: np.ndarray, arm: PlanarArm2R, q0: np.ndarray) -> np.ndarray:
    """Branch-continuous IK along a tip path, warm-started from ``q0`` (the home shoulder/elbow angles)."""
    prev = q0
    traj = np.empty((len(path), 2))
    for i, tip in enumerate(path):
        prev = arm.ik_cont(tip, prev)
        traj[i] = prev
    return traj


def _route_arm(start: np.ndarray, goal: np.ndarray, coin: np.ndarray, arm: PlanarArm2R, q0: np.ndarray,
               cfg: TransitConfig) -> "tuple[np.ndarray, np.ndarray]":
    """First fully-validated (tip keep-out, non-singular, single-branch, link-clear) route over the candidate set.

    Prefers the shorter sweep. # Raises: ``TransitInfeasible`` if no (direction, arc-radius) candidate validates
    (e.g. the object's far edge is beyond the arm's reach, forcing the singular circle).
    """
    r0, a0 = _polar(start, coin)
    r1, a1 = _polar(goal, coin)
    short = +1.0 if wrap(a1 - a0) >= 0 else -1.0
    for direction in (short, -short):
        for rho in cfg.arc_radii:
            path = _arc_waypoints(start, goal, coin, rho, direction, cfg.n_waypoints)
            if not _tip_feasible(path, coin, arm, cfg):
                continue
            traj = _solve_arm_traj(path, arm, q0)
            if _branch_consistent(traj) and _links_clear(traj, arm, coin, cfg):
                return path, traj
    raise TransitInfeasible(
        f"no validated arc for arm base {np.round(arm.base, 3)} "
        f"(start r_base {arm.r_base(start):.3f}, goal r_base {arm.r_base(goal):.3f}, reach {arm.reach:.3f})")


def _branch_consistent(traj: np.ndarray) -> bool:
    """A single elbow branch is held iff the elbow angle keeps one sign across the trajectory."""
    signs = np.sign(traj[:, 1])
    nonzero = signs[signs != 0]
    return bool(nonzero.size == 0 or np.all(nonzero == nonzero[0]))


def _pad(traj: np.ndarray, n: int) -> np.ndarray:
    """Right-pad a (m, 2) trajectory to length ``n`` by holding its last row (arm arrived, waits)."""
    if len(traj) >= n:
        return traj[:n]
    return np.vstack([traj, np.repeat(traj[-1:], n - len(traj), axis=0)])


def _compose_simultaneous(traj_l: np.ndarray, traj_r: np.ndarray, home_q: np.ndarray) -> np.ndarray:
    n = max(len(traj_l), len(traj_r))
    tl, tr = _pad(traj_l, n), _pad(traj_r, n)
    qref = np.empty((n + 1, 4))
    qref[0] = home_q
    qref[1:, 0:2] = tl
    qref[1:, 2:4] = tr
    return qref


def _compose_sequential(traj_l: np.ndarray, traj_r: np.ndarray, home_q: np.ndarray) -> np.ndarray:
    """Right arm reaches its shell first (left held at home), then the left arm reaches its shell (right held)."""
    phase1 = np.empty((len(traj_r), 4))
    phase1[:, 0:2] = home_q[0:2]
    phase1[:, 2:4] = traj_r
    phase2 = np.empty((len(traj_l), 4))
    phase2[:, 0:2] = traj_l
    phase2[:, 2:4] = traj_r[-1]
    return np.vstack([home_q, phase1, phase2])


def _densify_slew(qref: np.ndarray, slew: float) -> np.ndarray:
    """Insert linearly-interpolated rows wherever a step exceeds the slew cap (keeps every step trackable).

    Interpolation stays on the arm's continuous branch (small in-annulus steps), so no branch hop is introduced.
    """
    out = [qref[0]]
    for prev, nxt in zip(qref[:-1], qref[1:]):
        step = float(np.abs(wrap(nxt - prev)).max())
        k = max(1, int(np.ceil(step / slew)))
        for j in range(1, k + 1):
            out.append(prev + wrap(nxt - prev) * (j / k))
    return np.array(out)


@dataclass(frozen=True)
class TransitPlan:
    """A geometrically-validated, slew-feasible joint reference for the HOME -> READY transit."""

    qref: np.ndarray
    mode: str                          # "simultaneous" | "sequential"
    max_joint_step: float
    single_branch: bool
    diagnostics: dict = field(default_factory=dict)


def plan_collision_free_transit(home: Any, home_q: np.ndarray, targets: CoinStraddleTargets, arm_l: PlanarArm2R,
                                arm_r: PlanarArm2R, cfg: TransitConfig, fk: FkFn, *,
                                mode: str = "simultaneous") -> TransitPlan:
    """Plan a branch-continuous collision-free joint reference from ``home_q`` to the pre-contact targets.

    # Preconditions: ``fk`` measures tips in the same (coin-pinned) frame the arms were calibrated in; ``home_q`` is the
    #   length-4 home arm posture; ``mode`` in {"simultaneous", "sequential"}.
    # Postconditions: the returned ``qref`` has every step <= slew (asserted) and (per arm) holds one elbow branch;
    #   the coin keep-out and non-singular annulus were satisfied at every planned waypoint.
    # Raises: ``TransitInfeasible`` if either arm cannot be routed collision-free / non-singularly.
    """
    home_q = np.asarray(home_q, dtype=float)[:4]
    coin = targets.coin
    pre = targets.precontact()
    _, home_tip_l, home_tip_r = fk(home_q)
    path_l, traj_l = _route_arm(home_tip_l, pre.tip_left, coin, arm_l, home_q[0:2], cfg)
    path_r, traj_r = _route_arm(home_tip_r, pre.tip_right, coin, arm_r, home_q[2:4], cfg)
    single_branch = _branch_consistent(traj_l) and _branch_consistent(traj_r)
    compose = _compose_sequential if mode == "sequential" else _compose_simultaneous
    slew = float(home.stack.tau_rate * home.stack.control_dt)
    qref = _densify_slew(compose(traj_l, traj_r, home_q), slew)
    max_step = float(np.abs(wrap(np.diff(qref, axis=0))).max())
    assert max_step <= slew + 1e-9, f"densified qref step {max_step:.4f} exceeds slew {slew:.4f}"
    diag = {"n_waypoints_L": int(len(path_l)), "n_waypoints_R": int(len(path_r)),
            "qref_rows": int(len(qref)), "coin": [float(x) for x in coin]}
    return TransitPlan(qref, mode, max_step, single_branch, diag)


# --------------------------------------------------------------------------------------------------------------------
# Governed-servo execution
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TransitResult:
    """Outcome of tracking a ``TransitPlan`` through the governed servo (all measurements are physical, no state edit)."""

    reached_qerr: float
    coin_pert_mm: float
    min_clearance_mm: float
    contacts: int
    peak_qvel: float
    single_branch: bool
    mode: str
    ready_snapshot: Any


class GeometricTransitController:
    """Track a ``TransitPlan``'s joint reference through the frozen ``step_ablation`` -> ``govern_torque`` stack.

    No new torque authority: the reference is followed by a governed PD whose output passes the same slew/governor clip
    as every other controller in the system.
    """

    def __init__(self, home: Any, cfg: TransitConfig, coin: np.ndarray) -> None:
        self.home = home
        self.cfg = cfg
        self.coin = coin
        self.slew = float(home.stack.tau_rate * home.stack.control_dt)
        self.lo = np.asarray(home.lo, dtype=float)
        self.hi = np.asarray(home.hi, dtype=float)
        self.gl, self.gr = _fingertip_geoms(home.branch().inner.model)

    def roll(self, plan: TransitPlan) -> TransitResult:
        """Execute the plan; report reach error, coin perturbation, min clearance, contacts, peak qvel, READY snapshot.

        # Postconditions: deterministic given ``(home, plan)``; the returned snapshot is the terminal READY state that
        #   the READY -> capture boundary consumes (its ``prev_tau`` is the last governed torque).
        """
        rl = self.home.branch()
        prev = np.zeros(4)
        mujoco.set_mjcb_control(self._governor())
        try:
            prev, minclr, pert, peak = self._track(rl, plan.qref, prev)
        finally:
            mujoco.set_mjcb_control(None)
        return self._result(rl, plan, prev, minclr, pert, peak)

    def _governor(self) -> Callable:
        gov = self.home.stack.gov

        def cb(_model: Any, data: Any) -> None:
            data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)

        return cb

    def _track(self, rl: Any, qref: np.ndarray, prev: np.ndarray) -> "tuple[np.ndarray, float, float, float]":
        cfg = self.cfg
        minclr, pert, peak = 9.0, 0.0, 0.0
        total = len(qref) + cfg.hold_steps
        for i in range(total):
            target = qref[min(i, len(qref) - 1)]
            for _ in range(cfg.substeps):
                data = rl.inner.data
                dtau = np.clip(cfg.servo_kp * (target - data.qpos[:4]) - cfg.servo_kd * data.qvel[:4],
                               -self.slew, self.slew)
                prev = np.clip(prev + dtau, self.lo, self.hi)
                step_ablation(rl, np.asarray(prev, np.float32), "A")
                minclr, pert, peak = self._measure(rl, minclr, pert, peak)
        return prev, minclr, pert, peak

    def _measure(self, rl: Any, minclr: float, pert: float, peak: float) -> "tuple[float, float, float]":
        data = rl.inner.data
        coin_now = _coin_xy(rl)
        tl = data.geom_xpos[self.gl][:2]
        tr = data.geom_xpos[self.gr][:2]
        minclr = min(minclr, float(np.linalg.norm(tl - coin_now)), float(np.linalg.norm(tr - coin_now)))
        pert = max(pert, float(np.linalg.norm(coin_now - self.coin)))
        peak = max(peak, float(np.abs(data.qvel[:4]).max()))
        return minclr, pert, peak

    def _result(self, rl: Any, plan: TransitPlan, prev: np.ndarray, minclr: float, pert: float,
                peak: float) -> TransitResult:
        from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
        con = primary_fingertip_contacts(rl)
        contacts = int(con["left"] is not None) + int(con["right"] is not None)
        ready = kc.TransportSnapshot.from_live(rl_deepcopy(rl), self.home.stack, prev.copy())
        return TransitResult(
            reached_qerr=float(np.linalg.norm(rl.inner.data.qpos[:4] - plan.qref[-1])),
            coin_pert_mm=round(pert * 1000, 2), min_clearance_mm=round(minclr * 1000, 1), contacts=contacts,
            peak_qvel=round(peak, 3), single_branch=plan.single_branch, mode=plan.mode, ready_snapshot=ready)


def rl_deepcopy(rl: Any) -> Any:
    """Deep-copy the live rollout so the terminal snapshot owns its own MuJoCo state (deterministic branch source)."""
    import copy
    return copy.deepcopy(rl)


def build_arms(home: Any, coin: np.ndarray) -> "tuple[PlanarArm2R, PlanarArm2R]":
    """Calibrate both arms against the home snapshot's model (left dofs 0/1, right dofs 2/3)."""
    gl, gr = _fingertip_geoms(home.branch().inner.model)
    fk = make_fk(home, coin, gl, gr)
    arm_l = calibrate_arm(fk, shoulder_dof=0, elbow_dof=1, base_joint=0, tip_is_left=True)
    arm_r = calibrate_arm(fk, shoulder_dof=2, elbow_dof=3, base_joint=2, tip_is_left=False)
    return arm_l, arm_r


def execute_transit(home: Any, home_q: np.ndarray, targets: CoinStraddleTargets,
                    cfg: TransitConfig) -> TransitResult:
    """Plan + roll the HOME -> READY transit, falling back to sequential execution if the simultaneous roll contacts.

    # Postconditions: returns the successful ``TransitResult`` (its ``.mode`` records which route was used). If both
    #   simultaneous and sequential contact the coin, the (last) result is returned with its non-zero perturbation so
    #   the caller's gate fails loudly — never a silent success.
    """
    coin = targets.coin
    arm_l, arm_r = build_arms(home, coin)
    gl, gr = _fingertip_geoms(home.branch().inner.model)
    fk = make_fk(home, coin, gl, gr)
    ctrl = GeometricTransitController(home, cfg, coin)
    result = None
    for mode in ("simultaneous", "sequential"):
        plan = plan_collision_free_transit(home, home_q, targets, arm_l, arm_r, cfg, fk, mode=mode)
        result = ctrl.roll(plan)
        if result.contacts == 0 and result.coin_pert_mm <= 1.0:
            return result
    assert result is not None
    return result
