"""R10 Stage 1B — phase-shaped moving-precursor dynamic capture: READY -> moving handoff -> frozen downstream -> K6.

The cradle handoff ``(q*, qvel*, tau*)`` is a *phase-point of a moving orbit*, not a global capture target: ``q`` is the
position and ``qvel`` the *tangent* of the same trajectory, ``tau*`` the control-history memory. Hitting all three
jointly under a terminal-state objective is ill-conditioned (position xor velocity, measured). Instead we arrive *moving
along the orbit's tangent*:

  1. a quintic terminal segment drives the tips from READY to a backward-tangent precursor ``q_pre(dt) = q* - dt*qvel*``
     with terminal velocity ``~ s*qvel*`` (direction is what matters — the deliverable tube is wide on direction cosine,
     tight on preload);
  2. the ``prev_tau`` preload ramps *during* that moving segment (applying ``tau*`` afterward accelerates the arm out of
     the deliverable velocity band — ``tau*`` is the accelerating torque, ``qdot`` jumps 0.66->1.07 in one step);
  3. a short residual under a low-dimensional structured CEM closes the last margin.

The frozen downstream (APPROACH -> HANDOFF_RESET -> R2 -> coast -> K6) then generates the coupling itself. No state edit,
no snapshot injection: every command passes the governed ``step_ablation`` stack. Deterministic given the planner seed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, delivery_success, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.env.motion_contract import govern_torque

R2_ALPHA = 0.15
CAPTURE_HORIZON = 80
_NO_DELAY = 999                                    # sentinel: the two tips never both made contact


# --------------------------------------------------------------------------------------------------------------------
# Handoff reference (the moving orbit's phase-point) and the quintic terminal segment
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HandoffReference:
    """The cradle handoff as a phase-point of the winning orbit: position ``q_star``, tangent ``qvel_star``, preload
    ``tau_star``. The capture targets the backward-tangent precursor ``q_star - n*dt*qvel_star`` (a deliverable tube)."""

    q_star: np.ndarray
    qvel_star: np.ndarray
    tau_star: np.ndarray
    control_dt: float

    @classmethod
    def from_cradle(cls, cradle: Any) -> "HandoffReference":
        d = cradle.branch().inner.data
        return cls(d.qpos[:4].copy(), d.qvel[:4].copy(), np.asarray(cradle.prev_tau, float).copy(),
                   float(cradle.stack.control_dt))

    def precursor(self, n: float) -> np.ndarray:
        """Backward-tangent precursor ``q_star - n*control_dt*qvel_star`` (n=0 is the handoff itself)."""
        return self.q_star - n * self.control_dt * self.qvel_star

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.qvel_star))


def quintic_coeffs(q0: np.ndarray, v0: np.ndarray, qf: np.ndarray, vf: np.ndarray,
                   T: float) -> "tuple[np.ndarray, ...]":
    """Per-joint quintic with boundary ``(q0,v0,0) -> (qf,vf,0)`` over ``[0,T]`` (zero terminal acceleration)."""
    a3 = (20 * (qf - q0) - (8 * vf + 12 * v0) * T) / (2 * T ** 3)
    a4 = (30 * (q0 - qf) + (14 * vf + 16 * v0) * T) / (2 * T ** 4)
    a5 = (12 * (qf - q0) - (6 * vf + 6 * v0) * T) / (2 * T ** 5)
    return q0, v0, np.zeros(4), a3, a4, a5


def quintic_eval(c: "tuple[np.ndarray, ...]", t: float) -> "tuple[np.ndarray, np.ndarray]":
    """Position and velocity of the quintic at time ``t``."""
    a0, a1, a2, a3, a4, a5 = c
    q = a0 + a1 * t + a2 * t * t + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    v = a1 + 2 * a2 * t + 3 * a3 * t * t + 4 * a4 * t ** 3 + 5 * a5 * t ** 4
    return q, v


# --------------------------------------------------------------------------------------------------------------------
# Capture parameters + outcome
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureParams:
    """The structured, bounded capture parameters (what the CEM searches; everything else is fixed contract)."""

    n: float = 3.0                 # backward-tangent precursor index (Delta t in control steps)
    s: float = 0.3                 # terminal velocity scale (of qvel_star)
    preload_start: float = 0.5     # fraction of the segment at which the prev_tau ramp begins
    bmax: float = 0.6              # max preload-blend weight
    residual: np.ndarray = field(default_factory=lambda: np.zeros((2, 4)))
    steps: int = 20                # segment length
    kp: float = 16.0
    kd: float = 8.0


@dataclass(frozen=True)
class CaptureOutcome:
    """Result of a phase-shaped capture: the terminal snapshot + the tube-diagnostic metrics + downstream verdict.

    The ``bilateral_dwell`` / ``*_contact_relvel`` / ``coin_disp_capture_mm`` fields are read-only contact-formation
    diagnostics recorded *during* the roll (non-invasive: they never touch the sim). They default to the no-contact values
    so the grasp-agnostic default cost and every existing consumer are unaffected.
    """

    snapshot: Any
    cos_dir: float                 # cosine(terminal qvel, qvel_star)
    vel_scale: float               # |terminal qvel| / |qvel_star|
    dtau: float                    # ||terminal prev_tau - tau_star||
    contacts: int
    k6: bool = False
    min_dtz_mm: float = float("nan")
    safe: bool = True
    bilateral_dwell: int = 0                       # max consecutive both-tips-in-contact steps during the roll
    first_contact_relvel: float = float("nan")     # coin speed at the first tip contact (impact softness proxy)
    second_contact_relvel: float = float("nan")    # coin speed at the first BILATERAL contact (soft-seat proxy)
    coin_disp_capture_mm: float = float("nan")     # peak coin displacement during the roll (ejection guard)
    left_right_contact_delay: int = _NO_DELAY      # steps between first-left and first-right contact (999 if never both)
    terminal_coin_speed: float = 0.0               # coin speed at the last step (grasp stability proxy)


@dataclass(frozen=True)
class GraspObjective:
    """Opt-in grasp-aware CANDIDATE RANKING for the capture CEM (a class-based lexicographic rank, not a weighted sum).

    The differential audit (bank_c0_3 seed-0 vs seed-1) showed the released capture fails by a *left-tip miss* under an
    early/strong preload, while the grasped one seats both tips near-simultaneously and holds ~8 steps. The default cost
    rewards only K6/min_dtz, so a held bilateral grasp is incidental — and an ungrasped nudge can win on min_dtz or even K6.

    The fix ranks candidates by a grasp CLASS first (a NO_CONTACT nudge can never outrank a real grasp on min_dtz), with
    numeric terms only breaking ties *within* a class — so there are no cross-class weights to tune, and no hand-coded
    parameter bounds (``preload_start``/``bmax`` are left to the search; the score, not a rule, prefers the held grasp).

    ``dwell_target`` is the minimum consecutive bilateral-contact steps for the top GRASP_CERTIFIED class. ``dtz_elite``
    reserves that many CEM elite slots for the overall min_dtz-best candidates (a *hybrid elite*): the grasp-class elite
    alone floods the abundant low-preload held grasps and never samples the narrow high-preload deliverable basin the
    min_dtz elite finds (measured on bank_c0_3 seed-3), so a few delivery-gradient elites keep that basin in view.
    """

    dwell_target: int = 4
    dtz_elite: int = 3


@dataclass(frozen=True)
class CaptureSearchSpec:
    """Bounds for the structured CEM (bounded structural parameters only — never raw per-step torques).

    ``grasp_objective`` is opt-in: ``None`` (default) keeps the grasp-agnostic K6/min_dtz cost and early-exit bit-exact;
    setting it turns on the richer grasp-aware shaping + a grasped-only early-exit.
    """

    s_max: float = 0.6
    steps: int = 20
    kp: float = 16.0
    kd: float = 8.0
    knots: int = 2
    population: int = 44
    iters: int = 14
    elite: int = 9
    init_std: float = 0.35
    grasp_objective: "GraspObjective | None" = None


# --------------------------------------------------------------------------------------------------------------------
# The phase-shaped capture roller
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class _ContactTrace:
    """Read-only per-step observation of contact formation during a capture roll.

    # Invariants: never writes sim state (only reads contacts + coin kinematics), so the roll dynamics — and therefore
    #   every existing CaptureOutcome field — are bit-identical whether or not this trace is recorded.
    """

    coin0: np.ndarray
    dwell: int = 0
    max_dwell: int = 0
    first_l_step: int = -1
    first_r_step: int = -1
    first_relvel: float = float("nan")
    second_relvel: float = float("nan")
    coin_disp_mm: float = 0.0
    terminal_coin_speed: float = 0.0

    def observe(self, rl: Any, i: int) -> None:
        con = primary_fingertip_contacts(rl)
        cl, cr = con["left"] is not None, con["right"] is not None
        speed = float(np.linalg.norm(np.asarray(rl.inner._planar_metrics.disk_vel, float)[:2]))
        self._note_first_contacts(cl, cr, i, speed)
        self.dwell = self.dwell + 1 if (cl and cr) else 0
        self.max_dwell = max(self.max_dwell, self.dwell)
        self.coin_disp_mm = max(self.coin_disp_mm, float(np.linalg.norm(_coin_xy(rl) - self.coin0)) * 1000)
        self.terminal_coin_speed = speed

    def _note_first_contacts(self, cl: bool, cr: bool, i: int, speed: float) -> None:
        """Record the first left/right contact step."""
        if cl and self.first_l_step < 0:
            self.first_l_step = i
        if cr and self.first_r_step < 0:
            self.first_r_step = i
        self._note_relvels(cl, cr, speed)

    def _note_relvels(self, cl: bool, cr: bool, speed: float) -> None:
        """Record the coin speed at the first any-contact and the first bilateral-contact step (impact-softness proxies)."""
        if (cl or cr) and np.isnan(self.first_relvel):
            self.first_relvel = speed
        if cl and cr and np.isnan(self.second_relvel):
            self.second_relvel = speed

    @property
    def contact_delay(self) -> int:
        """Steps between the first left and first right contact (``_NO_DELAY`` if the two never both contacted)."""
        if self.first_l_step >= 0 and self.first_r_step >= 0:
            return abs(self.first_l_step - self.first_r_step)
        return _NO_DELAY


class PhaseShapeCapture:
    """Roll a phase-shaped moving precursor from READY through the governed servo (no state edit, no hidden force).

    Tracks a quintic ``READY -> q_pre`` (terminal velocity ``s*qvel_star``) with a bounded residual, and ramps the
    commanded torque toward ``tau_star`` *during* the moving segment. Every command passes ``step_ablation`` + governor.
    """

    def __init__(self, ready: Any, ref: HandoffReference, stack: Any) -> None:
        self.ready = ready
        self.ref = ref
        self.stack = stack
        self.slew = float(stack.tau_rate * stack.control_dt)
        self.lo = np.asarray(ready.lo, dtype=float)
        self.hi = np.asarray(ready.hi, dtype=float)
        d = ready.branch().inner.data
        self.q0, self.v0 = d.qpos[:4].copy(), d.qvel[:4].copy()
        self.prev0 = np.asarray(ready.prev_tau, float).copy()

    def roll(self, params: CaptureParams) -> CaptureOutcome:
        """Execute the capture; return the terminal snapshot + tube metrics (downstream verdict filled in by the caller).

        # Postconditions: deterministic given ``(ready, params)``; the coin is untouched by direct edit; the returned
        #   snapshot's ``prev_tau`` is the last governed command.
        """
        dt = self.ref.control_dt
        q_pre = self.ref.precursor(params.n)
        coeffs = quintic_coeffs(self.q0, self.v0, q_pre, params.s * self.ref.qvel_star, params.steps * dt)
        mujoco.set_mjcb_control(self._governor())
        try:
            rl, prev, trace = self._track(coeffs, params)
        finally:
            mujoco.set_mjcb_control(None)
        return self._outcome(rl, prev, trace)

    def _governor(self) -> Callable:
        gov = self.stack.gov

        def cb(_model: Any, data: Any) -> None:
            data.ctrl[:4] = govern_torque(data.ctrl[:4], data.qvel[:4], gov)

        return cb

    def _track(self, coeffs: "tuple[np.ndarray, ...]",
               params: CaptureParams) -> "tuple[Any, np.ndarray, _ContactTrace]":
        rl = self.ready.branch()
        prev = self.prev0.copy()
        knot_t = np.linspace(0, len(params.residual) - 1, params.steps)
        trace = _ContactTrace(coin0=_coin_xy(rl))
        for i in range(params.steps):
            prev = self.apply_step(rl, prev, self.scaffold_action(rl, prev, i, coeffs, params, knot_t))
            trace.observe(rl, i)                                     # read-only: does not perturb the roll
        return rl, prev, trace

    def scaffold_action(self, rl: Any, prev: np.ndarray, i: int, coeffs: "tuple[np.ndarray, ...]",
                        params: CaptureParams, knot_t: np.ndarray) -> np.ndarray:
        """The scaffold's per-step slew-normalised action in [-1,1]^4 (servo + residual-knots + preload blend).

        This is ``u_pi0(phase)`` in the action-preserving residual ``u = clip(u_pi0 + alpha*tanh(delta), -1, 1)``.
        """
        q_ref, v_ref = quintic_eval(coeffs, i * self.ref.control_dt)
        data = rl.inner.data
        servo = np.clip(params.kp * (q_ref - data.qpos[:4]) + params.kd * (v_ref - data.qvel[:4]),
                        -self.slew, self.slew)
        res = self._residual(params.residual, knot_t, i) * self.slew
        target = self._preload_blend(prev + servo + res, i, params)
        return np.clip(target - prev, -self.slew, self.slew) / self.slew

    def apply_step(self, rl: Any, prev: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply a slew-normalised action through the governed step; return the new ``prev`` (last commanded torque).

        # Preconditions: ``action`` in [-1,1]^4. Passing ``scaffold_action(...)`` reproduces the scaffold exactly.
        """
        prev = np.clip(prev + np.clip(action, -1.0, 1.0) * self.slew, self.lo, self.hi)
        step_ablation(rl, np.asarray(prev, np.float32), "A")
        return prev

    @staticmethod
    def _residual(knots: np.ndarray, knot_t: np.ndarray, i: int) -> np.ndarray:
        idx = range(len(knots))
        return np.array([np.interp(knot_t[i], idx, knots[:, j]) for j in range(4)])

    def _preload_blend(self, target: np.ndarray, i: int, params: CaptureParams) -> np.ndarray:
        """Blend the tracking target toward ``tau_star`` once the segment fraction passes ``preload_start``."""
        frac = i / (params.steps - 1)
        if frac < params.preload_start:
            return target
        weight = params.bmax * (frac - params.preload_start) / (1.0 - params.preload_start + 1e-9)
        return (1.0 - weight) * target + weight * self.ref.tau_star

    def _outcome(self, rl: Any, prev: np.ndarray, trace: _ContactTrace) -> CaptureOutcome:
        d = rl.inner.data
        qv = d.qvel[:4]
        speed = float(np.linalg.norm(qv))
        cos = float(qv @ self.ref.qvel_star / (speed * self.ref.speed + 1e-9))
        con = primary_fingertip_contacts(rl)
        snap = kc.TransportSnapshot.from_live(copy.deepcopy(rl), self.stack, prev.copy())
        return CaptureOutcome(snapshot=snap, cos_dir=round(cos, 3), vel_scale=round(speed / (self.ref.speed + 1e-9), 3),
                              dtau=round(float(np.linalg.norm(prev - self.ref.tau_star)), 3),
                              contacts=int(con["left"] is not None) + int(con["right"] is not None),
                              bilateral_dwell=trace.max_dwell, first_contact_relvel=round(trace.first_relvel, 4),
                              second_contact_relvel=round(trace.second_relvel, 4),
                              coin_disp_capture_mm=round(trace.coin_disp_mm, 2),
                              left_right_contact_delay=trace.contact_delay,
                              terminal_coin_speed=round(trace.terminal_coin_speed, 4))


# --------------------------------------------------------------------------------------------------------------------
# Frozen downstream adapter + structured CEM planner
# --------------------------------------------------------------------------------------------------------------------
class FrozenDownstream:
    """Thin adapter over the frozen HANDOFF_RESET -> R2 -> coast -> K6 chain (imported as-is; never modified here)."""

    def __init__(self, model: Any, norm: Any, r2_fn: Any, stack: Any, horizon: int = CAPTURE_HORIZON) -> None:
        self.model, self.norm, self.r2_fn, self.stack = model, norm, r2_fn, stack
        self.cfg = replace(DELIVERY_CFG, horizon=horizon)

    def deliver(self, snapshot: Any) -> "tuple[bool, float, bool]":
        """Roll the frozen downstream from ``snapshot``; return (strict K6, min_dtz_mm, safe)."""
        k6, md, safe, _ = self.deliver_with_trace(snapshot)
        return k6, md, safe

    def deliver_with_trace(self, snapshot: Any) -> "tuple[bool, float, bool, list]":
        """As ``deliver`` but also return the controller's per-step kind trace (for the boundary-transition gates)."""
        from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
        ctrl = HandoffResetTemporalController(snapshot, CloneActor(self.model, self.norm), self.r2_fn,
                                              ResidualBounds(alpha=R2_ALPHA))
        m = velocity_rollout(snapshot, ctrl, self.cfg)
        safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
        kinds = [r["kind"] for r in ctrl.clone_trace]
        return bool(delivery_success(m, self.cfg)), round(_min_dtz_mm(snapshot, m), 2), safe, kinds


def _theta_to_params(theta: np.ndarray, spec: CaptureSearchSpec) -> CaptureParams:
    """Map a flat CEM vector to bounded structural parameters (never raw per-step torques)."""
    return CaptureParams(n=3.0, s=float(spec.s_max * np.clip(theta[0], 0.0, 1.0)),
                         preload_start=float(np.clip(theta[1], 0.0, 1.0)), bmax=float(np.clip(theta[2], 0.0, 1.0)),
                         residual=theta[3:].reshape(spec.knots, 4), steps=spec.steps, kp=spec.kp, kd=spec.kd)


GRASP_CERTIFIED, BILATERAL_TRANSIENT, SINGLE_CONTACT_ONLY, NO_CONTACT, SAFETY_FAILURE = 0, 1, 2, 3, 4


def _grasp_class(outcome: CaptureOutcome, dwell_target: int) -> int:
    """Rank a candidate's grasp quality into a class (lower = better). This is the dominant ranking signal: no numeric
    min_dtz/K6 advantage lets a lower class outrank a higher one — a NO_CONTACT nudge can never beat a real grasp."""
    if not outcome.safe:
        return SAFETY_FAILURE
    if outcome.contacts == 2 and outcome.bilateral_dwell >= dwell_target:
        return GRASP_CERTIFIED                                      # both tips held for the required dwell -> delivery-ready
    if outcome.bilateral_dwell > 0 or not np.isnan(outcome.second_contact_relvel):
        return BILATERAL_TRANSIENT                                  # both tips touched at some point but not held
    if not np.isnan(outcome.first_contact_relvel):
        return SINGLE_CONTACT_ONLY                                  # only one tip ever contacted
    return NO_CONTACT


def rank_capture(outcome: CaptureOutcome, obj: "GraspObjective | None") -> Any:
    """Public grasp-aware ranking key for a capture outcome (lower = better); ``obj=None`` = the grasp-agnostic scalar.

    Lets the demo-bank pipeline pick the best capture across the teacher budget with the SAME semantics the in-CEM search
    uses, instead of re-deriving a `(k6, min_dtz)` ladder at the layer boundary.
    """
    return _rank_key(outcome, obj)


def is_certified_grasp(outcome: CaptureOutcome, obj: "GraspObjective | None") -> bool:
    """True iff ``outcome`` is a delivery-ready held bilateral grasp (top GRASP_CERTIFIED class) under ``obj``.

    Used to gate the capture-teacher budget's early-exit and downstream success on an actual grasp (never a nudge-K6).
    """
    return obj is not None and _grasp_class(outcome, obj.dwell_target) == GRASP_CERTIFIED


def _rank_key(outcome: CaptureOutcome, obj: "GraspObjective | None" = None) -> Any:
    """Candidate ranking key (lower = better).

    Default (``obj is None``) reproduces the original grasp-agnostic scalar cost exactly. With ``obj`` set, returns a
    class-based lexicographic tuple whose ordering was fixed by the A/B on bank_c0_3:

    1. grasp CLASS (a NO_CONTACT nudge can never outrank a real grasp — this is the fix for the ranking-contract bug);
    2. within a class, DOWNSTREAM DELIVERY (min_dtz): dwell is a *classification* criterion, not a ranking one — ranking
       held grasps by dwell picked stable-but-undeliverable grasps (near-zero preload) and regressed valid K6 to 0, so
       delivery decides among grasps of the same class;
    3. then dwell / left-right delay / terminal coin speed as stability tie-breaks.

    No cross-class weights, no hand-coded param bounds. Postcondition: ``obj is None`` -> the exact prior float cost.
    """
    if not outcome.safe:
        return 1e3 if obj is None else (SAFETY_FAILURE, 1e3, 0, _NO_DELAY, 0.0)
    if obj is None:
        return 0.0 if outcome.k6 else outcome.min_dtz_mm
    md = outcome.min_dtz_mm if not np.isnan(outcome.min_dtz_mm) else 1e3
    return (_grasp_class(outcome, obj.dwell_target), round(md, 2), -outcome.bilateral_dwell,
            outcome.left_right_contact_delay, round(outcome.terminal_coin_speed, 3))


def _evaluate(theta: np.ndarray, cap: PhaseShapeCapture, spec: CaptureSearchSpec,
              downstream: FrozenDownstream) -> CaptureOutcome:
    params = _theta_to_params(theta, spec)
    out = cap.roll(params)
    k6, md, safe = downstream.deliver(out.snapshot)
    return replace(out, k6=k6, min_dtz_mm=md, safe=safe)


def _solution_found(best: "tuple[Any, np.ndarray, CaptureOutcome] | None", obj: "GraspObjective | None") -> bool:
    """CEM stopping condition: any K6 (default) — or, when grasp-aware, only a *held* grasped K6 (never a nudge-K6)."""
    if best is None or not best[2].k6:
        return False
    return obj is None or _grasp_class(best[2], obj.dwell_target) == GRASP_CERTIFIED


def _select_elite(cand: "list[tuple[Any, CaptureOutcome, np.ndarray]]", spec: CaptureSearchSpec,
                  obj: "GraspObjective | None") -> np.ndarray:
    """The CEM elite that drives the mean/std update, as a stack of theta vectors.

    Default (``obj is None``): top ``spec.elite`` by rank — bit-exact to the prior ``scored.sort``. Grasp-aware: a HYBRID
    elite — ``spec.elite - obj.dtz_elite`` slots by the grasp-aware rank plus ``obj.dtz_elite`` slots by overall min_dtz,
    so the search keeps covering BOTH the grasp basin and the min_dtz (deliverable) basin. Ranking alone cannot pick a
    deliverable grasp the search never samples (bank_c0_3 seed-3).
    """
    by_rank = sorted(range(len(cand)), key=lambda i: cand[i][0])
    if obj is None:
        return np.stack([cand[i][2] for i in by_rank[:spec.elite]])
    n_dtz = min(obj.dtz_elite, spec.elite)
    chosen = list(by_rank[:spec.elite - n_dtz])
    seen = set(chosen)
    by_dtz = sorted(range(len(cand)),
                    key=lambda i: cand[i][1].min_dtz_mm if not np.isnan(cand[i][1].min_dtz_mm) else 1e9)
    for i in by_dtz:
        if len(chosen) >= spec.elite:
            break
        if i not in seen:
            chosen.append(i)
            seen.add(i)
    return np.stack([cand[i][2] for i in chosen])


@dataclass(frozen=True)
class CaptureResult:
    """A solved capture: the winning parameters, its outcome, and the seed (for deterministic replay + provenance)."""

    seed: int
    params: CaptureParams
    outcome: CaptureOutcome


def plan_capture(ready: Any, ref: HandoffReference, stack: Any, downstream: FrozenDownstream, *, seed: int,
                 spec: CaptureSearchSpec = CaptureSearchSpec(),
                 candidate_hook: "Callable[[CaptureOutcome], None] | None" = None) -> CaptureResult:
    """Structured CEM over the bounded capture parameters, scored by the frozen downstream (stops on the first K6).

    # Preconditions: ``ready`` is the frozen analytic-transit READY snapshot; ``ref`` the cradle phase-point.
    # Postconditions: deterministic given ``seed``; the returned params reproduce the outcome via ``PhaseShapeCapture``.
    # ``candidate_hook`` (optional) observes every evaluated ``CaptureOutcome`` — read-only audit instrumentation, does not
    #   touch the search (``None`` is bit-exact to the prior behaviour).
    """
    cap = PhaseShapeCapture(ready, ref, stack)
    dim = 3 + spec.knots * 4
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim)
    mean[:3] = [0.6, 0.5, 0.5]                       # priors: s_raw, preload_start, bmax
    std = np.full(dim, spec.init_std)
    obj = spec.grasp_objective
    best: "tuple[Any, np.ndarray, CaptureOutcome] | None" = None
    for _ in range(spec.iters):
        pop = mean[None] + std[None] * rng.standard_normal((spec.population, dim))
        cand: "list[tuple[Any, CaptureOutcome, np.ndarray]]" = []
        for theta in pop:
            out = _evaluate(theta, cap, spec, downstream)
            if candidate_hook is not None:
                candidate_hook(out)                  # read-only observer; never affects the search
            key = _rank_key(out, obj)
            cand.append((key, out, theta))
            if best is None or key < best[0]:
                best = (key, theta, out)
        elite = _select_elite(cand, spec, obj)
        mean, std = elite.mean(0), elite.std(0) * 0.88 + 1e-3
        if _solution_found(best, obj):
            break
    assert best is not None
    return CaptureResult(seed=seed, params=_theta_to_params(best[1], spec), outcome=best[2])
